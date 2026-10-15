"""Agnes AI provider.

Agnes AI provides multimodal AI model services with OpenAI-compatible API.
Free models (currently $0/M tokens): agnes-2.5-flash, agnes-3.0-flash.
Both support 512K context, tool calling, and image understanding.

Base URL: https://apihub.agnes-ai.com/v1
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

PROVIDER_NAME = "agnes"

AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_MODELS_URL = "https://apihub.agnes-ai.com/v1/models"

# Free models for routing (both $0/M tokens currently)
FREE_MODELS = {"agnes-2.5-flash", "agnes-3.0-flash"}

# Model capabilities
MODEL_CAPABILITIES = {
    "agnes-2.5-flash": {
        "context_length": 512000,
        "max_output_tokens": 65536,
        "tools_supported": True,
        "image_understanding": True,
        "thinking": True,
        "display_name": "Agnes 2.5 Flash",
    },
    # 3.0-flash advertises tool support in its catalog entry but in practice
    # ignores the `tools` payload — it answers directly and even misidentifies
    # itself as another vendor's model. Verified live against the agentic loop:
    # a `tool_choice: "required"` request returned plain text with no tool_calls
    # array, while agnes-2.5-flash emitted a real function call. Until Agnes
    # fixes the hub, treat 3.0-flash as tool-free so routing never silently
    # drops tool execution.
    "agnes-3.0-flash": {
        "context_length": 512000,
        "max_output_tokens": 65536,
        "tools_supported": False,
        "image_understanding": True,
        "thinking": True,
        "display_name": "Agnes 3.0 Flash",
    },
    "agnes-2.5-pro": {
        "context_length": 512000,
        "max_output_tokens": 65536,
        "tools_supported": True,
        "image_understanding": True,
        "thinking": True,
        "display_name": "Agnes 2.5 Pro",
    },
    "agnes-2.5-pro-beta": {
        "context_length": 512000,
        "max_output_tokens": 65536,
        "tools_supported": True,
        "image_understanding": True,
        "thinking": True,
        "display_name": "Agnes 2.5 Pro Beta",
    },
}

# Cache
_CACHE: list[dict] = []
_CACHE_TIME: float = 0.0
_CACHE_TTL: float = 300.0


def _headers() -> dict:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.agnes_api_key}",
    }


async def fetch_agnes_models(force_refresh: bool = False) -> list[dict]:
    """Fetch (and cache) the Agnes AI model catalog."""
    global _CACHE, _CACHE_TIME

    if not force_refresh and _CACHE and (time.time() - _CACHE_TIME) < _CACHE_TTL:
        return _CACHE

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(AGNES_MODELS_URL, headers=_headers())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                _CACHE = data
                _CACHE_TIME = time.time()
                logger.info("Discovered %d Agnes AI models", len(_CACHE))
                return _CACHE
            else:
                logger.warning("Agnes AI models returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("Failed to fetch Agnes AI models: %s", e)

    return _CACHE


def is_agnes_model(model: str) -> bool:
    return model.startswith("agnes-") or any(m.get("id") == model for m in _CACHE)


async def ensure_models() -> list[dict]:
    """Refresh the catalog if it is missing or stale."""
    if not _CACHE or (time.time() - _CACHE_TIME) >= _CACHE_TTL:
        await fetch_agnes_models()
    return _CACHE


def rank_combo_models(models: list[tuple[str, str]], need_tools: bool = False) -> list[tuple[str, str]]:
    """Reorder a combo's ``(provider, model_id)`` list so tool-bearing requests
    lead with a model that actually emits ``tool_calls``.

    Only Agnes entries are reordered — every other provider keeps its declared
    position. The correction is capability-driven (see ``MODEL_CAPABILITIES``),
    so a model whose tool support is downgraded here is automatically deprioritized
    everywhere, including the auto-router fallback plan in providers.py.
    """
    if not models:
        return models
    agnes_entries = [(p, m) for p, m in models if p == PROVIDER_NAME]
    other = [(p, m) for p, m in models if p != PROVIDER_NAME]

    def _key(pm: tuple[str, str]) -> tuple[int, int]:
        caps = MODEL_CAPABILITIES.get(pm[1], {})
        tool_ok = 1 if caps.get("tools_supported") else 0
        # Prefer 3.0-flash for plain text, 2.5-flash second.
        preference = 1 if pm[1] == "agnes-3.0-flash" else 0
        if need_tools:
            # Tool-capable models first; among them, prefer 2.5-flash (the one
            # that actually works) over 3.0-flash.
            return (tool_ok, 1 - preference)
        return (preference, tool_ok)

    agnes_entries.sort(key=_key, reverse=True)
    return agnes_entries + other


def get_free_models() -> list[dict]:
    """Return free models from Agnes AI."""
    result = []
    for mid in FREE_MODELS:
        caps = MODEL_CAPABILITIES.get(mid, {})
        result.append({
            "id": mid,
            "isFree": True,
            "supported_parameters": ["tools"] if caps.get("tools_supported") else [],
            "context_length": caps.get("context_length"),
            "max_output_tokens": caps.get("max_output_tokens"),
            "display_name": caps.get("display_name", mid),
            "owned_by": "agnes",
        })
    return result


def rank_free_models(need_tools: bool = False) -> list[str]:
    """Rank free models by capability.

    Order is driven by ``MODEL_CAPABILITIES`` rather than a hardcoded list, so a
    capability correction (e.g. 3.0-flash's broken tool support) is honored
    everywhere this function is called — there is only one source of truth.

    Tool-bearing requests lead with a model that actually emits ``tool_calls``
    (verified live: 3.0-flash ignores the ``tools`` payload and answers as plain
    text). Plain-text requests prefer the newer 3.0-flash first.
    """
    def _rank(m: str) -> tuple[int, int]:
        caps = MODEL_CAPABILITIES.get(m, {})
        tool_ok = 1 if caps.get("tools_supported") else 0
        # Prefer 3.0-flash for plain text, 2.5-flash second.
        preference = 1 if m == "agnes-3.0-flash" else 0
        if need_tools:
            # Tool-capable models first; among them, prefer 2.5-flash (the one
            # that actually works) over 3.0-flash.
            return (tool_ok, 1 - preference)
        return (preference, tool_ok)

    return sorted(FREE_MODELS, key=_rank, reverse=True)


async def agnes_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from Agnes AI.

    Handles the reasoning_content field used by Agnes models for thinking.
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
    accumulated_reasoning = ""
    yielded_role = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{AGNES_BASE_URL}/chat/completions",
            json=payload,
            headers=_headers(),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"Agnes AI {response.status_code}: {error_msg}",
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
                reasoning = delta.get("reasoning_content")
                finish_reason = choices[0].get("finish_reason")
                tool_calls = delta.get("tool_calls")

                if reasoning:
                    accumulated_reasoning += reasoning

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
