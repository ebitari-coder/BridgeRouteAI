"""Routeway provider — free models only.

Routeway is an OpenAI-compatible gateway (``https://api.routeway.ai/v1``)
providing 200+ models behind one key. Only the free tier — ids ending in
``:free`` (e.g. ``deepseek-v4-flash:free``) — is used here; PAYG ids are
never advertised or routed.

``GET /v1/models`` is public and returns rich metadata (context length,
capabilities, supported_parameters), so the catalog needs no key; chat
calls need ``ROUTEWAY_API_KEY``.

Free-tier limits (upstream): 5 requests/minute and 200 requests/day on
``:free`` models — exceeding them returns 429 ``rate_limit_exceeded``.

This module provides:
- live discovery of the ``:free`` catalog (public API, cached, TTL refresh),
- capability-ranked candidate ordering for the auto-router,
- an OpenAI-compatible streaming client.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator

import httpx

from app.config import settings
from app.provider_utils import (
    QuotaError,
    SamplingParams,
    apply_sampling,
    msg_get,
    sse_chunk,
    to_openai_messages,
    usage_chunk,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "routeway"

FREE_SUFFIX = ":free"

# Seed used until the first successful catalog fetch (and as an offline fallback).
STATIC_FREE_MODELS: list[dict] = [
    {"id": "muse-glimmer-30b:free", "name": "Glimmer 30B (Free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "gemma-4-26b-a4b-it-chimerax:free", "name": "Gemma 4 26b A4b It Chimerax (Free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 62000},
    {"id": "gemma-4-26b-a4b-it-darksoul:free", "name": "Gemma 4 26b A4b It Darksoul (Free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 62000},
    {"id": "gemma-4-26b-a4b-it-luminous:free", "name": "Gemma 4 26b A4b It Luminous (Free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 62000},
    {"id": "gemma-4-26b-a4b-it-meromero:free", "name": "Gemma 4 26B MeroMero (Free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 62000},
    {"id": "gemma-4-26b-a4b-it-moonlight:free", "name": "Gemma 4 26b A4b It Moonlight (Free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 62000},
    {"id": "gemma-4-26b-a4b-it-musica:free", "name": "Gemma 4 26b A4b It Musica (Free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 62000},
    {"id": "deepseek-v4-flash:free", "name": "DeepSeek V4 Flash (Free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 42000},
    {"id": "minimax-m2.7:free", "name": "MiniMax M2.7 (Free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 42000},
]

_CACHE: list[dict] = []
_CACHE_TIME: float = 0.0

# Models whose upstream reported them as missing are parked here (id -> expiry)
# so the router stops burning attempts on them; they rejoin after the TTL.
_DEAD: dict[str, float] = {}
_DEAD_TTL_SECONDS = 6 * 3600.0


def _base_url() -> str:
    return (settings.routeway_base_url or "https://api.routeway.ai/v1").rstrip("/")


def _ttl() -> float:
    return float(getattr(settings, "routeway_catalog_refresh_seconds", 300) or 300)


def _headers() -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if settings.routeway_api_key:
        headers["Authorization"] = f"Bearer {settings.routeway_api_key}"
    return headers


def is_configured() -> bool:
    """Chat calls need a key; the catalog does not."""
    return bool(settings.routeway_api_key)


async def fetch_models(force_refresh: bool = False) -> list[dict]:
    """Refresh the ``:free`` catalog from the public ``GET /v1/models``."""
    global _CACHE, _CACHE_TIME

    if _CACHE and not force_refresh and (time.time() - _CACHE_TIME) < _ttl():
        return _CACHE

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(f"{_base_url()}/models")
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.json().get("data") or []

        free: list[dict] = []
        for m in data:
            mid = m.get("id") or ""
            if not mid.endswith(FREE_SUFFIX):
                continue
            if m.get("available") is False:
                continue
            caps = m.get("capabilities") or {}
            params = m.get("supported_parameters") or []
            has_tools = "tools" in params or bool(caps.get("function_call"))
            free.append({
                "id": mid,
                "name": m.get("short_name") or m.get("name") or mid,
                "supported_parameters": (
                    ["tools", "temperature", "top_p", "max_tokens"]
                    if has_tools else ["temperature", "top_p", "max_tokens"]
                ),
                "context_length": m.get("context_length"),
            })
        if not free:
            raise RuntimeError("no `:free` models in catalog")

        _CACHE = free
        _CACHE_TIME = time.time()
        logger.info(
            "Discovered %d Routeway free models: %s",
            len(free),
            ", ".join(m["id"] for m in free),
        )
    except Exception as exc:
        logger.warning(
            "Routeway catalog refresh failed (%s); keeping %d cached model(s)",
            exc,
            len(_CACHE),
        )

    return _CACHE


async def ensure_models() -> list[dict]:
    """Refresh the catalog when it is missing or stale (no key required)."""
    return await fetch_models()


async def catalog_refresh_loop(interval: float | None = None) -> None:
    """Refresh the Routeway catalog on a fixed interval until cancelled."""
    seconds = interval or _ttl()
    while True:
        await asyncio.sleep(seconds)
        try:
            await fetch_models(force_refresh=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Routeway catalog refresh failed: %s", exc)


def get_models() -> list[dict]:
    """Live catalog first, then the static seed for anything it does not list."""
    seen: dict[str, dict] = {}
    for m in (*_CACHE, *STATIC_FREE_MODELS):
        seen.setdefault(m["id"], m)
    return list(seen.values())


def is_routeway_model(model: str) -> bool:
    """Catalog membership decides ownership (bare ``:free`` ids)."""
    if not model:
        return False
    return model in {m["id"] for m in get_models()}


def quarantine(model: str, reason: str = "") -> None:
    """Park a model whose upstream reported it as missing (self-heals via TTL)."""
    _DEAD[model] = time.time() + _DEAD_TTL_SECONDS
    logger.warning(
        "Routeway quarantining %s for %.0fh%s",
        model,
        _DEAD_TTL_SECONDS / 3600,
        f": {reason}" if reason else "",
    )


def _drop_expired_quarantine() -> None:
    now = time.time()
    for mid in [mid for mid, expiry in _DEAD.items() if expiry < now]:
        del _DEAD[mid]


def get_free_models() -> list[dict]:
    """Free models available to the auto-router.

    Chat calls need an API key — without one we return nothing so the router
    does not burn attempts on 401s. Quarantined (upstream-missing) models are
    excluded until their TTL lapses.
    """
    if not is_configured():
        return []
    _drop_expired_quarantine()
    return [m for m in get_models() if m["id"] not in _DEAD]


def rank_free_models(need_tools: bool = False) -> list[str]:
    """Free model ids ordered best-first (tool support / larger context first)."""
    models = get_free_models()
    if need_tools:
        models = [m for m in models if "tools" in (m.get("supported_parameters") or [])]

    def _score(m: dict) -> tuple:
        ctx = m.get("context_length") or 0
        return (0 if "tools" in (m.get("supported_parameters") or []) else 1, -ctx, m["id"])

    return [m["id"] for m in sorted(models, key=_score)]


async def routeway_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from Routeway (OpenAI-compatible)."""
    if not is_configured():
        raise QuotaError(
            "Routeway API key not configured (set ROUTEWAY_API_KEY)",
            provider=PROVIDER_NAME,
            model=model,
            status_code=401,
        )

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    openai_messages = to_openai_messages(messages)

    payload = {
        "model": model,
        "messages": openai_messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    apply_sampling(payload, sampling)

    prompt_tokens_est = max(
        1, sum(len(str(msg_get(msg, "content") or "")) for msg in messages) // 4
    )
    accumulated_text = ""
    yielded_role = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{_base_url()}/chat/completions",
            json=payload,
            headers=_headers(),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                if (
                    "model_not_found" in error_msg
                    or ("model" in error_msg.lower() and "not found" in error_msg.lower())
                ):
                    quarantine(model, f"HTTP {response.status_code}")
                raise QuotaError(
                    f"Routeway {response.status_code}: {error_msg}",
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
                    if yielded_role:
                        yield sse_chunk(chat_id, created, model, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return
                if not line_data:
                    continue

                try:
                    parsed = json.loads(line_data)
                except json.JSONDecodeError:
                    continue

                if "error" in parsed:
                    err = parsed["error"]
                    if not isinstance(err, dict):
                        err = {"message": str(err), "type": "api_error"}
                    yield "data: " + json.dumps({"error": err}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return

                choices = parsed.get("choices") or []
                if not choices:
                    # keep-alive / usage-only frames
                    if parsed.get("usage") and include_usage and yielded_role:
                        yield usage_chunk(
                            chat_id, created, model,
                            parsed["usage"].get("prompt_tokens", prompt_tokens_est),
                            accumulated_text,
                        )
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
