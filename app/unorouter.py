"""UnoRouter provider.

UnoRouter is an OpenAI-compatible API gateway that aggregates
200+ models from 48+ providers. The free tier (132 models) offers
rate-limited access at no cost.

This module provides:
- live discovery of the free model catalog (cached),
- stability monitoring worker that tracks model uptime/success,
- capability-ranked candidate ordering for the auto-router,
- an OpenAI-compatible streaming client.
"""
import asyncio
import uuid
import time
import json
import logging
from typing import AsyncIterator

import httpx

from app.config import settings
from app.provider_utils import QuotaError, SamplingParams, apply_sampling, msg_get, sse_chunk, to_openai_messages, usage_chunk

logger = logging.getLogger(__name__)

PROVIDER_NAME = "unorouter"

UNOROUTER_BASE_URL = "https://api.unorouter.com/v1"
UNOROUTER_CATALOG_URL = "https://api.unorouter.com/api/pricing/catalog"

UNOROUTER_AUTO_MODEL = "unorouter/auto"

_CACHE: list[dict] = []
_CACHE_IDS: set[str] = set()
_CACHE_TIME: float = 0.0
_CACHE_TTL: float = 300.0  # 5 minutes

_NON_TEXT_OUTPUT_MODALITIES = {"audio", "image", "video"}

# Quality thresholds for free model selection
_MIN_UPTIME = 95.0   # minimum uptime percentage (24h)
_MIN_SUCCESS = 50.0  # minimum success rate percentage

# Stable models set — populated by the background worker
# Only models in this set are eligible for routing
STABLE_MODELS: set[str] = set()
_stability_check_time: float = 0.0


def _headers() -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if settings.unorouter_api_key:
        headers["Authorization"] = f"Bearer {settings.unorouter_api_key}"
    return headers


def _is_free(m: dict) -> bool:
    if m.get("is_free") is not True or m.get("online") is not True:
        return False
    # Filter by quality: uptime and success rate must meet thresholds
    uptime = m.get("uptime_24h")
    success = m.get("success_rate")
    if uptime is not None and uptime < _MIN_UPTIME:
        return False
    if success is not None and success < _MIN_SUCCESS:
        return False
    return True


def _is_chat_model(m: dict) -> bool:
    outputs = (m.get("architecture") or {}).get("output_modalities")
    if not outputs:
        return True
    return "text" in outputs and not (_NON_TEXT_OUTPUT_MODALITIES & set(outputs))


async def fetch_unorouter_models(force_refresh: bool = False) -> list[dict]:
    """Fetch (and cache) the UnoRouter model catalog."""
    global _CACHE, _CACHE_IDS, _CACHE_TIME

    if not force_refresh and _CACHE and (time.time() - _CACHE_TIME) < _CACHE_TTL:
        return _CACHE

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(UNOROUTER_CATALOG_URL)
            if resp.status_code == 200:
                data = resp.json().get("models", [])
                free_models = [m for m in data if _is_free(m) and _is_chat_model(m)]
                _CACHE = free_models
                _CACHE_IDS = {_model_id(m) for m in free_models if m.get("model_name")}
                _CACHE_TIME = time.time()
                logger.info("Discovered %d UnoRouter free chat models", len(free_models))
                return _CACHE
            else:
                logger.warning("UnoRouter catalog returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("Failed to fetch UnoRouter models: %s", e)

    return _CACHE


def is_unorouter_model(model: str) -> bool:
    if model == UNOROUTER_AUTO_MODEL:
        return True
    if model in _CACHE_IDS:
        return True
    return False


async def ensure_models() -> list[dict]:
    """Refresh the catalog if it is missing or stale."""
    if not _CACHE or (time.time() - _CACHE_TIME) >= _CACHE_TTL:
        await fetch_unorouter_models()
    return _CACHE


def _model_id(m: dict) -> str:
    """Derive the API model ID from a catalog entry."""
    name = m.get("model_name", "")
    return name if name.endswith(":free") else f"{name}:free"


def get_free_models() -> list[dict]:
    if _CACHE:
        result = []
        for m in _CACHE:
            raw_tags = m.get("tags", [])
            if isinstance(raw_tags, str):
                raw_tags = [t.strip() for t in raw_tags.split(",")]
            tags_lower = [t.lower() for t in raw_tags] if raw_tags else []
            result.append({
                "id": _model_id(m),
                "isFree": True,
                "supported_parameters": ["tools"] if "tools" in tags_lower else [],
                "context_length": None,
                "uptime": m.get("uptime_24h"),
                "success_rate": m.get("success_rate"),
            })
        return result
    return []


def get_free_model_ids() -> list[str]:
    return [m["id"] for m in get_free_models()]


def check_model_stability(m: dict) -> bool:
    """Check if a single model meets stability thresholds."""
    if m.get("is_free") is not True or m.get("online") is not True:
        return False
    uptime = m.get("uptime_24h")
    success = m.get("success_rate")
    if uptime is not None and uptime < _MIN_UPTIME:
        return False
    if success is not None and success < _MIN_SUCCESS:
        return False
    return True


async def refresh_stability() -> int:
    """Refresh the catalog and update STABLE_MODELS. Returns count of stable models."""
    global _stability_check_time

    await fetch_unorouter_models(force_refresh=True)

    stable = set()
    for m in _CACHE:
        if check_model_stability(m):
            stable.add(_model_id(m))

    added = stable - STABLE_MODELS
    removed = STABLE_MODELS - stable

    if added:
        logger.info("UnoRouter stability: +[%s] now stable", ", ".join(sorted(added)))
    if removed:
        logger.info("UnoRouter stability: -[%s] now unstable", ", ".join(sorted(removed)))

    STABLE_MODELS.clear()
    STABLE_MODELS.update(stable)
    _stability_check_time = time.time()

    logger.info(
        "UnoRouter stability check complete: %d/%d models stable",
        len(STABLE_MODELS),
        len(_CACHE),
    )
    return len(STABLE_MODELS)


async def stability_worker() -> None:
    """Background worker that periodically checks UnoRouter model stability.

    Runs every ``settings.unorouter_stability_check_seconds`` (default 300s).
    When models cross the uptime/success thresholds they are added/removed
    from STABLE_MODELS, which the routing layer checks before attempting
    a request.
    """
    interval = settings.unorouter_stability_check_seconds
    # Initial check immediately on startup
    try:
        count = await refresh_stability()
        logger.info("UnoRouter stability worker initial check: %d stable models", count)
    except Exception as exc:
        logger.warning("UnoRouter stability initial check failed: %s", exc)

    while True:
        await asyncio.sleep(interval)
        try:
            await refresh_stability()
        except Exception as exc:
            logger.warning("UnoRouter stability check failed: %s", exc)


def _score(m: dict) -> float:
    score = 0.0
    # Uptime and success rate are primary quality signals
    uptime = m.get("uptime_24h") or 0
    success = m.get("success_rate") or 0
    score += uptime * 10.0  # max +1000 for 100% uptime
    score += success * 5.0   # max +500 for 100% success
    # Context length bonus
    cl = m.get("context_length") or 0
    score += min(cl, 2_000_000) / 1000.0
    # Capability bonuses
    raw_tags = m.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in raw_tags.split(",")]
    tags = [t.lower() for t in raw_tags] if raw_tags else []
    if "reasoning" in tags:
        score += 500
    if "tools" in tags:
        score += 200
    return score


def rank_free_models(need_tools: bool = False) -> list[str]:
    candidates = []
    for m in get_free_models():
        params = m.get("supported_parameters") or []
        if need_tools and "tools" not in params:
            continue
        model_id = m["id"]
        cached = next((cm for cm in _CACHE if _model_id(cm) == model_id), None)
        candidates.append((_score(cached or m), model_id))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return [mid for _, mid in candidates]


async def unorouter_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from UnoRouter.

    Uses the OpenAI-compatible endpoint at ``/v1/chat/completions``.
    """
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

    prompt_tokens_est = max(1, sum(len(str(msg_get(msg, "content") or "")) for msg in messages) // 4)
    accumulated_text = ""
    yielded_role = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{UNOROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=_headers(),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"UnoRouter {response.status_code}: {error_msg}",
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
                    yield "data: " + json.dumps({
                        "error": parsed["error"]
                        if isinstance(parsed["error"], dict)
                        else {"message": str(parsed["error"]), "type": "api_error"},
                    }) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return

                choices = parsed.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                finish_reason = choices[0].get("finish_reason")
                tool_calls = delta.get("tool_calls")

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

            if include_usage:
                yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
            if yielded_role:
                yield sse_chunk(chat_id, created, model, finish_reason="stop")
            yield "data: [DONE]\n\n"
