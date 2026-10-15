"""Tiarina Cloud provider — OpenAI-compatible model gateway.

Tiarina exposes chat completions at ``https://ai.tiarina.cloud/v1`` and uses
Bearer API keys. The authenticated ``GET /v1/models`` catalog is used when
available; ``deepseek-v4.1-flash`` remains available as an offline seed for
the configured provider.
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

PROVIDER_NAME = "tiarina"
DEFAULT_BASE_URL = "https://ai.tiarina.cloud/v1"

STATIC_MODELS: list[dict] = [
    {
        "id": "deepseek-v4.1-flash",
        "name": "DeepSeek V4.1 Flash",
        "supported_parameters": ["temperature", "top_p", "max_tokens", "stop"],
    },
]

_CACHE: list[dict] = []
_CACHE_TIME = 0.0
_DEAD: dict[str, float] = {}
_DEAD_TTL_SECONDS = 6 * 3600.0


def _base_url() -> str:
    return (settings.tiarina_base_url or DEFAULT_BASE_URL).rstrip("/")


def _ttl() -> float:
    return float(getattr(settings, "tiarina_catalog_refresh_seconds", 300) or 300)


def _headers(accept: str = "application/json") -> dict[str, str]:
    headers = {"Accept": accept, "Content-Type": "application/json"}
    if settings.tiarina_api_key:
        headers["Authorization"] = f"Bearer {settings.tiarina_api_key}"
    return headers


def _ti_enabled() -> bool:
    """Provider kill switch — TIARINA_ENABLED=false fully removes Tiarina
    from routing, catalogs, and failover plans (default: disabled). The
    account is currently billing-locked (403 free_tier_locked / 402
    insufficient_quota), so requests through it degrade every chain that
    includes it.
    """
    try:
        return bool(settings.tiarina_enabled)
    except Exception:
        return False


def is_configured() -> bool:
    return bool(settings.tiarina_api_key)


def _normalize_model(model: dict) -> dict | None:
    model_id = model.get("id") or model.get("name")
    if not model_id:
        return None
    capabilities = model.get("capabilities") or {}
    if not isinstance(capabilities, dict):
        capabilities = {}
    supported = list(model.get("supported_parameters") or [])
    if capabilities.get("tools") or capabilities.get("function_call"):
        if "tools" not in supported:
            supported.append("tools")
    return {
        "id": str(model_id),
        "name": model.get("name") or model.get("display_name") or model_id,
        "context_length": model.get("context_length") or model.get("context_window") or model.get("max_context_tokens"),
        "supported_parameters": supported,
        "vision": bool(model.get("vision") or capabilities.get("vision")),
        "reasoning": bool(model.get("reasoning") or capabilities.get("reasoning")),
    }


async def fetch_models(force_refresh: bool = False) -> list[dict]:
    """Fetch Tiarina's authenticated OpenAI-compatible model catalog."""
    global _CACHE, _CACHE_TIME
    if _CACHE and not force_refresh and time.time() - _CACHE_TIME < _ttl():
        return _CACHE
    if not is_configured():
        return _CACHE

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(f"{_base_url()}/models", headers=_headers())
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            payload = response.json()
            raw_models = payload.get("data") if isinstance(payload, dict) else payload
            raw_models = raw_models or []

        models = [normalized for raw in raw_models if isinstance(raw, dict)
                  if (normalized := _normalize_model(raw)) is not None]
        if not models:
            raise RuntimeError("no models in Tiarina catalog")
        _CACHE = models
        _CACHE_TIME = time.time()
        logger.info("Discovered %d Tiarina models: %s", len(models), ", ".join(m["id"] for m in models))
    except Exception as exc:
        logger.warning("Tiarina catalog refresh failed (%s); keeping %d cached model(s)", exc, len(_CACHE))
    return _CACHE


async def ensure_models() -> list[dict]:
    if not _ti_enabled():
        return []
    return await fetch_models()


async def catalog_refresh_loop(interval: float | None = None) -> None:
    seconds = interval or _ttl()
    while True:
        await asyncio.sleep(seconds)
        try:
            await fetch_models(force_refresh=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Tiarina catalog refresh failed: %s", exc)


def get_models() -> list[dict]:
    if _CACHE:
        return list(_CACHE)
    return list(STATIC_MODELS)


def is_tiarina_model(model: str) -> bool:
    if not _ti_enabled():
        return False
    return is_configured() and bool(model) and model in {m["id"] for m in get_models()}


def quarantine(model: str, reason: str = "") -> None:
    _DEAD[model] = time.time() + _DEAD_TTL_SECONDS
    logger.warning("Tiarina quarantining %s for %.0fh%s", model, _DEAD_TTL_SECONDS / 3600, f": {reason}" if reason else "")


def _drop_expired_quarantine() -> None:
    now = time.time()
    for model, expiry in list(_DEAD.items()):
        if expiry <= now:
            del _DEAD[model]


def get_available_models() -> list[dict]:
    if not _ti_enabled():
        return []
    if not is_configured():
        return []
    _drop_expired_quarantine()
    return [m for m in get_models() if m["id"] not in _DEAD]


def rank_models(need_tools: bool = False) -> list[str]:
    if not _ti_enabled():
        return []
    models = get_available_models()
    if need_tools:
        models = [m for m in models if "tools" in (m.get("supported_parameters") or [])]

    def score(model: dict) -> tuple:
        params = model.get("supported_parameters") or []
        return (0 if "tools" in params else 1, 0 if model.get("reasoning") else 1, -(model.get("context_length") or 0), model["id"])

    return [m["id"] for m in sorted(models, key=score)]


async def tiarina_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream an OpenAI-compatible Tiarina chat completion."""
    if not is_configured():
        raise QuotaError(
            "Tiarina API key not configured (set TIARINA_API_KEY)",
            provider=PROVIDER_NAME,
            model=model,
            status_code=401,
        )

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    payload = {"model": model, "messages": to_openai_messages(messages), "stream": True}
    if tools:
        payload["tools"] = tools
    apply_sampling(payload, sampling)
    prompt_tokens_est = max(1, sum(len(str(msg_get(message, "content") or "")) for message in messages) // 4)
    accumulated_text = ""
    yielded_role = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{_base_url()}/chat/completions", json=payload, headers=_headers("text/event-stream")) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")[:500]
                if response.status_code == 400 and "model" in body.lower() and "not" in body.lower():
                    quarantine(model, f"HTTP {response.status_code}")
                raise QuotaError(
                    f"Tiarina {response.status_code}: {body}",
                    provider=PROVIDER_NAME,
                    model=model,
                    status_code=response.status_code,
                )

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    if include_usage:
                        yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
                    if yielded_role:
                        yield sse_chunk(chat_id, created, model, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return
                if not data:
                    continue
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if parsed.get("error"):
                    error = parsed["error"]
                    if not isinstance(error, dict):
                        error = {"message": str(error), "type": "api_error"}
                    yield "data: " + json.dumps({"error": error}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
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

            if include_usage:
                yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
            if yielded_role:
                yield sse_chunk(chat_id, created, model, finish_reason="stop")
            yield "data: [DONE]\n\n"
