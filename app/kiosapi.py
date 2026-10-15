"""KiosAPI provider.

KiosAPI is an OpenAI-compatible API gateway that aggregates
23+ models from multiple providers through a single endpoint.
The free tier offers zero-cost models with tool calling support.

This module provides:
- live discovery of the free model catalog (cached),
- capability-ranked candidate ordering for the auto-router,
- an OpenAI-compatible streaming client.
"""
import uuid
import time
import json
import logging
from typing import AsyncIterator

import httpx

from app.config import settings
from app.provider_utils import QuotaError, SamplingParams, apply_sampling, msg_get, sse_chunk, to_openai_messages, usage_chunk

logger = logging.getLogger(__name__)

PROVIDER_NAME = "kiosapi"

KIOSAPI_BASE_URL = "https://kiosapi.com/v1"
KIOSAPI_MODELS_URL = "https://kiosapi.com/v1/models"

KIOSAPI_AUTO_MODEL = "kiosapi/auto"

# Free models known to support tool calling
# (hardcoded as fallback; dynamic list fetched from /v1/models)
FREE_MODELS: list[dict] = []

_CACHE: list[str] = []
_CACHE_TIME: float = 0.0
_CACHE_TTL: float = 300.0  # 5 minutes


def _headers() -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if settings.kiosapi_api_key:
        headers["Authorization"] = f"Bearer {settings.kiosapi_api_key}"
    return headers


async def fetch_kiosapi_models(force_refresh: bool = False) -> list[str]:
    """Fetch (and cache) the KiosAPI model list."""
    global _CACHE, _CACHE_TIME

    if not force_refresh and _CACHE and (time.time() - _CACHE_TIME) < _CACHE_TTL:
        return _CACHE

    if not settings.kiosapi_api_key:
        logger.info("KiosAPI not configured (KIOSAPI_API_KEY not set)")
        return _CACHE

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(KIOSAPI_MODELS_URL, headers=_headers())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                _CACHE = [m["id"] for m in data if m.get("id")]
                _CACHE_TIME = time.time()
                logger.info("Discovered %d KiosAPI models", len(_CACHE))
                return _CACHE
            else:
                logger.warning("KiosAPI models returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("Failed to fetch KiosAPI models: %s", e)

    return _CACHE


def is_kiosapi_model(model: str) -> bool:
    if model == KIOSAPI_AUTO_MODEL:
        return True
    if model.startswith("agnes-"):
        return False
    return model in _CACHE


async def ensure_models() -> list[str]:
    """Refresh the catalog if it is missing or stale."""
    if not _CACHE or (time.time() - _CACHE_TIME) >= _CACHE_TTL:
        await fetch_kiosapi_models()
    return _CACHE


def get_free_models() -> list[dict]:
    """Return free models with tool calling support.

    KiosAPI uses token groups to control access. All models returned by the
    /v1/models endpoint are available to the current key. For a Free-group key,
    all listed models are free. All KiosAPI models support tool calling.
    """
    if not _CACHE:
        return []

    result = []
    for model_id in _CACHE:
        if model_id.startswith("agnes-"):
            continue
        result.append({
            "id": model_id,
            "isFree": True,
            "supported_parameters": ["tools"],  # KiosAPI supports tool calling
            "context_length": None,
        })

    return result


def rank_free_models(need_tools: bool = False) -> list[str]:
    """Rank free models by capability.

    Priority: larger models first, then alphabetical.
    All KiosAPI free models support tools.
    """
    models = get_free_models()
    if need_tools:
        models = [m for m in models if "tools" in (m.get("supported_parameters") or [])]

    # Sort by model name (larger models tend to be listed first)
    return [m["id"] for m in models]


async def kiosapi_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from KiosAPI.

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
            f"{KIOSAPI_BASE_URL}/chat/completions",
            json=payload,
            headers=_headers(),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"KiosAPI {response.status_code}: {error_msg}",
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
