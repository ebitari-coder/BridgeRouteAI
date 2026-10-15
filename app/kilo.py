"""Kilo AI Gateway provider.

Kilo exposes an OpenAI-compatible API at ``https://api.kilo.ai/api/gateway``.
Its public catalog (``GET /models``, no auth required) tags free models via
``isFree`` / a ``:free`` suffix and reports context length, supported
parameters, and benchmark scores.

This module provides:

* live discovery of the free model catalog (cached),
* capability/health ranked candidate ordering for the auto-router, and
* an OpenAI-compatible streaming client.

Free models are available anonymously (rate-limited per IP); supplying a
``KILO_API_KEY`` raises those limits.
"""
import uuid
import time
import json
import asyncio
import logging
from typing import AsyncIterator

import httpx

from app.config import settings
from app.provider_utils import QuotaError, usage_chunk, sse_chunk, apply_sampling, SamplingParams, to_openai_messages

logger = logging.getLogger(__name__)

PROVIDER_NAME = "kilo"

# Virtual model id for the advanced auto-router. Requests for this model are
# routed across every discovered free Kilo model, ranked by capability.
AUTO_MODEL = "bridgeroute/auto"

# Used only when the live catalog cannot be fetched. Mirrors the candidate set
# of Kilo's own ``kilo-auto/free`` router (plus that router as a safety net).
KILO_FALLBACK_FREE_MODELS = [
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nex-agi/nex-n2.5-pro:free",
    "inclusionai/ling-3.0-flash-vl:free",
    "dots-studio/dots-3-note-preview:free",
    "kilo-auto/free",
]

_CACHE: list[dict] = []
_CACHE_IDS: set[str] = set()
_CACHE_TIME: float = 0.0
_CACHE_TTL: float = 300.0  # 5 minutes


def _headers(accept: str = "application/json") -> dict:
    headers = {"Accept": accept, "Content-Type": "application/json"}
    if settings.kilo_api_key:
        headers["Authorization"] = f"Bearer {settings.kilo_api_key}"
    return headers


def _models_url() -> str:
    return f"{settings.kilo_base_url.rstrip('/')}/models"


def _chat_url() -> str:
    return f"{settings.kilo_base_url.rstrip('/')}/chat/completions"


def is_kilo_model(model: str) -> bool:
    """Whether a model id belongs to the Kilo gateway.

    Recognises the auto-router, ``kilo/`` + ``kilo-auto/`` namespaces, anything
    present in the discovered catalog, and the static free fallback list.
    """
    if model == AUTO_MODEL or model.startswith("kilo/") or model.startswith("kilo-auto/"):
        return True
    if model in _CACHE_IDS:
        return True
    return model in KILO_FALLBACK_FREE_MODELS


_NON_TEXT_OUTPUT_MODALITIES = {"audio", "image", "video"}


def _is_chat_model(m: dict) -> bool:
    """Whether a catalog entry is a text chat model.

    Excludes generation models (music/image/video) that cannot serve chat
    completions even though they may be zero-priced.
    """
    outputs = (m.get("architecture") or {}).get("output_modalities")
    if not outputs:
        return True  # unknown — assume chat-capable
    return "text" in outputs and not (_NON_TEXT_OUTPUT_MODALITIES & set(outputs))


def _is_free(m: dict) -> bool:
    free = m.get("isFree") is True or str(m.get("id", "")).endswith(":free")
    if not free:
        # Some entries carry zero pricing without the explicit free markers.
        pricing = m.get("pricing") or {}
        try:
            free = (
                str(pricing.get("prompt", "")) in ("0", "0.0")
                and str(pricing.get("completion", "")) in ("0", "0.0")
            )
        except Exception:
            free = False
    return free and _is_chat_model(m)


async def fetch_kilo_models(force_refresh: bool = False) -> list[dict]:
    """Fetch (and cache) the Kilo model catalog. Returns the raw model list."""
    global _CACHE, _CACHE_IDS, _CACHE_TIME

    if not force_refresh and _CACHE and (time.time() - _CACHE_TIME) < _CACHE_TTL:
        return _CACHE

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_models_url(), headers=_headers())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    _CACHE = data
                    _CACHE_IDS = {m.get("id") for m in data if m.get("id")}
                    _CACHE_TIME = time.time()
                    free = sum(1 for m in data if _is_free(m))
                    logger.info("Discovered %d Kilo models (%d free)", len(data), free)
                    return _CACHE
            else:
                logger.warning("Kilo models endpoint returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("Failed to fetch Kilo models: %s", e)

    return _CACHE


async def ensure_models() -> list[dict]:
    """Refresh the catalog if it is missing or stale."""
    if not _CACHE or (time.time() - _CACHE_TIME) >= _CACHE_TTL:
        await fetch_kilo_models()
    return _CACHE


async def catalog_refresh_loop(interval: float | None = None) -> None:
    """Refresh the Kilo catalog on a fixed interval until the task is cancelled.

    Keeps the free-model pool current without a process restart. Failures are
    logged and retried on the next tick rather than killing the loop.
    """
    seconds = interval or settings.kilo_catalog_refresh_seconds
    while True:
        await asyncio.sleep(seconds)
        try:
            await fetch_kilo_models(force_refresh=True)
            logger.info("Kilo catalog refreshed (%d free models)", len(get_free_models()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Kilo catalog refresh failed: %s", exc)


def get_free_models() -> list[dict]:
    """Return the free Kilo models (live catalog, or a static fallback)."""
    if _CACHE:
        return [m for m in _CACHE if _is_free(m)]
    return [
        {"id": mid, "isFree": True, "supported_parameters": ["tools"], "context_length": None}
        for mid in KILO_FALLBACK_FREE_MODELS
    ]


def get_free_model_ids() -> list[str]:
    return [m["id"] for m in get_free_models()]


def _score(m: dict) -> float:
    """Heuristic capability score for routing preference."""
    score = 0.0
    # Prefer longer context windows.
    score += min(m.get("context_length") or 0, 2_000_000) / 1000.0
    # Prefer models with benchmark evidence.
    tb = (m.get("terminalBench") or {}).get("overallScore")
    if isinstance(tb, (int, float)):
        score += tb * 1000
    # Mild preference for newer releases.
    score += min(m.get("created") or 0, 2_000_000_000) / 1e7
    return score


def rank_free_models(need_tools: bool = False) -> list[str]:
    """Return free model ids ordered best-first for the auto-router.

    When ``need_tools`` is set, models that do not advertise tool support are
    dropped from the candidate pool.
    """
    candidates = []
    for m in get_free_models():
        params = m.get("supported_parameters") or []
        if need_tools and "tools" not in params:
            continue
        candidates.append((_score(m), m["id"]))

    # Deterministic: score desc, then id for stable tie-breaking.
    candidates.sort(key=lambda c: (-c[0], c[1]))

    ordered = [mid for _, mid in candidates]

    # Always keep Kilo's own free router as a final safety net.
    if "kilo-auto/free" not in ordered:
        ordered.append("kilo-auto/free")

    return ordered


async def kilo_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a single chat completion from the Kilo gateway."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    chat_messages = to_openai_messages(messages)

    payload = {"model": model, "messages": chat_messages, "stream": True}
    if tools:
        payload["tools"] = tools
    apply_sampling(payload, sampling)

    prompt_tokens_est = max(1, len(json.dumps(chat_messages, ensure_ascii=False)) // 4)
    accumulated_text = ""
    yielded_role = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            _chat_url(),
            json=payload,
            headers=_headers(accept="text/event-stream"),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"Kilo {response.status_code}: {error_msg}",
                    provider=PROVIDER_NAME,
                    model=model,
                    status_code=response.status_code,
                )

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                line_data = line[6:].strip()
                if line_data == "[DONE]":
                    if include_usage:
                        yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
                    yield "data: [DONE]\n\n"
                    return

                try:
                    parsed = json.loads(line_data)
                except json.JSONDecodeError:
                    continue

                choices = parsed.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                tool_calls = delta.get("tool_calls")
                finish_reason = choices[0].get("finish_reason")

                if content:
                    accumulated_text += content

                if content or tool_calls:
                    if not yielded_role:
                        yield sse_chunk(chat_id, created, model, role="assistant")
                        yielded_role = True
                    yield sse_chunk(chat_id, created, model, content=content, tool_calls=tool_calls)

                if finish_reason and finish_reason != "null":
                    if include_usage:
                        yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
                    if yielded_role:
                        yield sse_chunk(chat_id, created, model, finish_reason=finish_reason)
                    yield "data: [DONE]\n\n"
                    return

            # Stream ended without an explicit finish signal
            if include_usage:
                yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
            if yielded_role:
                yield sse_chunk(chat_id, created, model, finish_reason="stop")
            yield "data: [DONE]\n\n"
