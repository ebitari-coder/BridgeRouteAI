"""Clean APIs provider — OpenAI-compatible gateway.

Clean APIs provides an OpenAI-compatible API at ``https://cleanapis.com/v1``
with 33 models from various providers (Anthropic, OpenAI, Google, Meta, etc.).

This module provides:
- live discovery of the model catalog (cached, TTL refresh, requires auth),
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

PROVIDER_NAME = "cleanapis"

# Static seed used until the first successful catalog fetch (and as an offline fallback).
STATIC_MODELS: list[dict] = [
    {
        "id": "claude-opus-4.8",
        "name": "Claude Opus 4.8",
        "context_length": 200000,
        "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop", "reasoning", "vision"],
        "vision": True,
        "reasoning": True,
    },
    {
        "id": "claude-sonnet-4.5",
        "name": "Claude Sonnet 4.5",
        "context_length": 200000,
        "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop", "reasoning", "vision"],
        "vision": True,
        "reasoning": True,
    },
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "context_length": 128000,
        "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop", "vision"],
        "vision": True,
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "context_length": 128000,
        "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop", "vision"],
        "vision": True,
    },
    {
        "id": "gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "context_length": 2000000,
        "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop", "reasoning", "vision"],
        "vision": True,
        "reasoning": True,
    },
    {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "context_length": 2000000,
        "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop", "reasoning", "vision"],
        "vision": True,
        "reasoning": True,
    },
    {
        "id": "llama-3.1-405b",
        "name": "Llama 3.1 405B",
        "context_length": 128000,
        "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"],
    },
    {
        "id": "llama-3.1-70b",
        "name": "Llama 3.1 70B",
        "context_length": 128000,
        "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"],
    },
    {
        "id": "llama-3.1-8b",
        "name": "Llama 3.1 8B",
        "context_length": 128000,
        "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"],
    },
]

_CACHE: list[dict] = []
_CACHE_TIME: float = 0.0


def _base_url() -> str:
    return (settings.cleanapis_base_url or "https://cleanapis.com/v1").rstrip("/")


def _ttl() -> float:
    return float(getattr(settings, "cleanapis_catalog_refresh_seconds", 300) or 300)


def _headers(accept: str = "application/json") -> dict:
    headers = {"Accept": accept, "Content-Type": "application/json"}
    if settings.cleanapis_api_key:
        headers["Authorization"] = f"Bearer {settings.cleanapis_api_key}"
    return headers


def is_configured() -> bool:
    """Chat calls need a key; the catalog does too."""
    return bool(settings.cleanapis_api_key)


async def fetch_models(force_refresh: bool = False) -> list[dict]:
    """Fetch the Clean APIs catalog."""
    global _CACHE, _CACHE_TIME

    if _CACHE and not force_refresh and (time.time() - _CACHE_TIME) < _ttl():
        return _CACHE

    if not is_configured():
        logger.warning("Clean APIs API key not configured; using static models")
        _CACHE = STATIC_MODELS
        _CACHE_TIME = time.time()
        return _CACHE

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{_base_url()}/models", headers=_headers())
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.json().get("data") or []
            models = []
            for m in data:
                mid = m.get("id")
                if not mid:
                    continue
                caps = m.get("capabilities") or []
                supported_params = ["tools", "temperature", "top_p", "max_tokens", "stop"]
                if "vision" in caps:
                    supported_params.append("vision")
                if "reasoning" in caps:
                    supported_params.append("reasoning")
                if "json_mode" in caps:
                    supported_params.append("json_mode")
                if "web_search" in caps:
                    supported_params.append("web_search")
                models.append({
                    "id": mid,
                    "name": m.get("name") or mid,
                    "context_length": m.get("context_window"),
                    "supported_parameters": supported_params,
                    "vision": "vision" in caps,
                    "reasoning": "reasoning" in caps,
                })
            if not models:
                raise RuntimeError("no models in catalog")
            _CACHE = models
            _CACHE_TIME = time.time()
            logger.info(
                "Discovered %d Clean APIs models: %s",
                len(models),
                ", ".join(m["id"] for m in models),
            )
    except Exception as exc:
        logger.warning(
            "Clean APIs catalog refresh failed (%s); keeping %d cached model(s)",
            exc,
            len(_CACHE),
        )
        if not _CACHE:
            _CACHE = STATIC_MODELS
            _CACHE_TIME = time.time()

    return _CACHE


async def ensure_models() -> list[dict]:
    """Refresh the catalog when it is missing or stale."""
    return await fetch_models()


async def catalog_refresh_loop(interval: float | None = None) -> None:
    """Refresh the Clean APIs catalog on a fixed interval until cancelled."""
    seconds = interval or _ttl()
    while True:
        await asyncio.sleep(seconds)
        try:
            await fetch_models(force_refresh=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Clean APIs catalog refresh failed: %s", exc)


def get_models() -> list[dict]:
    """Live catalog first, then the static seed for anything it does not list."""
    seen: dict[str, dict] = {}
    for m in (*_CACHE, *STATIC_MODELS):
        seen.setdefault(m["id"], m)
    return list(seen.values())


def is_cleanapis_model(model: str) -> bool:
    """Catalog membership decides ownership."""
    if not model:
        return False
    return model in {m["id"] for m in get_models()}


def get_free_models() -> list[dict]:
    """Models available to the auto-router.

    The catalog requires an API key, and every chat call needs one — without one
    we return nothing so the router does not burn attempts on 401s.
    """
    if not is_configured():
        return []
    return get_models()


def get_free_model_ids() -> list[str]:
    return [m["id"] for m in get_free_models()]


def rank_free_models(need_tools: bool = False) -> list[str]:
    """Model ids ordered best-first (tool support first, then by capability)."""
    models = get_free_models()
    if need_tools:
        models = [m for m in models if "tools" in (m.get("supported_parameters") or [])]

    def _score(m: dict) -> tuple:
        has_tools = "tools" in (m.get("supported_parameters") or [])
        has_reasoning = m.get("reasoning") is True
        has_vision = m.get("vision") is True
        ctx = m.get("context_length") or 0
        # Prefer: tool support > reasoning > vision > larger context
        return (
            0 if has_tools else 1,
            0 if has_reasoning else 1,
            0 if has_vision else 1,
            -ctx,
            m["id"],
        )

    return [m["id"] for m in sorted(models, key=_score)]


async def cleanapis_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from Clean APIs (OpenAI-compatible)."""
    if not is_configured():
        raise QuotaError(
            "Clean APIs API key not configured (set CLEANAPIS_API_KEY)",
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
            headers=_headers(accept="text/event-stream"),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"Clean APIs {response.status_code}: {error_msg}",
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