"""DreamPrompting provider — unified gateway for multiple upstream providers.

DreamPrompting exposes an OpenAI-compatible API at ``https://dreamprompting.com/api/v1``
with a virtual ``auto`` router and models from Groq, Mistral, NVIDIA, Cohere, Google,
OpenRouter, Cloudflare, HuggingFace, Codestral, and Chat.

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

PROVIDER_NAME = "dreamprompting"

# Seed used until the first successful catalog fetch (and as an offline fallback).
STATIC_MODELS: list[dict] = [
    {"id": "auto", "name": "DreamPrompting Auto Router", "owned_by": "dreamprompting", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "groq/openai/gpt-oss-120b", "name": "GPT-OSS 120B (Groq)", "owned_by": "groq", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "groq/openai/gpt-oss-20b", "name": "GPT-OSS 20B (Groq)", "owned_by": "groq", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "groq/qwen/qwen3.6-27b", "name": "Qwen 3.6 27B (Groq)", "owned_by": "groq", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "groq/groq/compound-mini", "name": "Compound Mini (Groq)", "owned_by": "groq", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "mistral/mistral-small-latest", "name": "Mistral Small (Mistral)", "owned_by": "mistral", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "mistral/open-mistral-nemo", "name": "Mistral Nemo (Mistral)", "owned_by": "mistral", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "mistral/mistral-large-latest", "name": "Mistral Large (Mistral)", "owned_by": "mistral", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "nvidia/nvidia/nemotron-3-super-120b-a12b", "name": "Nemotron 3 Super (NVIDIA)", "owned_by": "nvidia", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "name": "Nemotron 3 Nano (NVIDIA)", "owned_by": "nvidia", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "nvidia-vision/meta/llama-3.2-11b-vision-instruct", "name": "Llama 3.2 11B Vision (NVIDIA)", "owned_by": "nvidia-vision", "supported_parameters": ["temperature", "top_p", "max_tokens", "stop"], "vision": True},
    {"id": "cohere/command-r-08-2024", "name": "Command R (Cohere)", "owned_by": "cohere", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "cohere/command-r-plus-08-2024", "name": "Command R+ (Cohere)", "owned_by": "cohere", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "cohere/command-a-03-2025", "name": "Command A (Cohere)", "owned_by": "cohere", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "google/gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash Lite (Google)", "owned_by": "google", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash (Google)", "owned_by": "google", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "openrouter/nvidia/nemotron-3-super-120b-a12b:free", "name": "Nemotron 3 Super (OpenRouter Free)", "owned_by": "openrouter", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "openrouter/google/gemma-4-31b-it:free", "name": "Gemma 4 31B IT (OpenRouter Free)", "owned_by": "openrouter", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "openrouter/openai/gpt-oss-20b:free", "name": "GPT-OSS 20B (OpenRouter Free)", "owned_by": "openrouter", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast", "name": "Llama 3.3 70B (Cloudflare)", "owned_by": "cloudflare", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "cloudflare/@cf/ibm-granite/granite-4.0-h-micro", "name": "Granite 4.0 H Micro (Cloudflare)", "owned_by": "cloudflare", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "cloudflare/@cf/nvidia/nemotron-3-120b-a12b", "name": "Nemotron 3 120B (Cloudflare)", "owned_by": "cloudflare", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "cloudflare/@cf/openai/gpt-oss-120b", "name": "GPT-OSS 120B (Cloudflare)", "owned_by": "cloudflare", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "cloudflare-vision/@cf/meta/llama-4-scout-17b-16e-instruct", "name": "Llama 4 Scout (Cloudflare Vision)", "owned_by": "cloudflare-vision", "supported_parameters": ["temperature", "top_p", "max_tokens", "stop"], "vision": True},
    {"id": "huggingface/meta-llama/Llama-3.3-70B-Instruct", "name": "Llama 3.3 70B (HuggingFace)", "owned_by": "huggingface", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "huggingface/Qwen/Qwen2.5-72B-Instruct", "name": "Qwen 2.5 72B (HuggingFace)", "owned_by": "huggingface", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "codestral/codestral-latest", "name": "Codestral Latest", "owned_by": "codestral", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
    {"id": "chat/ch.at", "name": "CH.AT", "owned_by": "chat", "supported_parameters": ["tools", "temperature", "top_p", "max_tokens", "stop"]},
]

_CACHE: list[dict] = []
_CACHE_TIME: float = 0.0


def _base_url() -> str:
    return (settings.dreamprompting_base_url or "https://dreamprompting.com/api/v1").rstrip("/")


def _ttl() -> float:
    return float(getattr(settings, "dreamprompting_catalog_refresh_seconds", 300) or 300)


def _headers(accept: str = "application/json") -> dict:
    headers = {"Accept": accept, "Content-Type": "application/json"}
    if settings.dreamprompting_api_key:
        headers["Authorization"] = f"Bearer {settings.dreamprompting_api_key}"
    return headers


def is_configured() -> bool:
    """Chat calls need a key; the catalog does too."""
    return bool(settings.dreamprompting_api_key)


async def fetch_models(force_refresh: bool = False) -> list[dict]:
    """Fetch the DreamPrompting catalog."""
    global _CACHE, _CACHE_TIME

    if _CACHE and not force_refresh and (time.time() - _CACHE_TIME) < _ttl():
        return _CACHE

    if not is_configured():
        logger.warning("DreamPrompting API key not configured; using static models")
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
                models.append({
                    "id": mid,
                    "name": m.get("name") or mid,
                    "owned_by": m.get("owned_by") or "dreamprompting",
                    "context_length": m.get("context_length"),
                    "supported_parameters": m.get("supported_parameters") or [],
                    "vision": m.get("vision") or False,
                })
            if not models:
                raise RuntimeError("no models in catalog")
            _CACHE = models
            _CACHE_TIME = time.time()
            logger.info(
                "Discovered %d DreamPrompting models: %s",
                len(models),
                ", ".join(m["id"] for m in models),
            )
    except Exception as exc:
        logger.warning(
            "DreamPrompting catalog refresh failed (%s); keeping %d cached model(s)",
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
    """Refresh the DreamPrompting catalog on a fixed interval until cancelled."""
    seconds = interval or _ttl()
    while True:
        await asyncio.sleep(seconds)
        try:
            await fetch_models(force_refresh=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("DreamPrompting catalog refresh failed: %s", exc)


def get_models() -> list[dict]:
    """Live catalog first, then the static seed for anything it does not list."""
    seen: dict[str, dict] = {}
    for m in (*_CACHE, *STATIC_MODELS):
        seen.setdefault(m["id"], m)
    return list(seen.values())


def is_dreamprompting_model(model: str) -> bool:
    """Catalog membership decides ownership."""
    if not model:
        return False
    return model in {m["id"] for m in get_models()}


def get_free_models() -> list[dict]:
    """Models available to the auto-router. Requires API key for chat."""
    if not is_configured():
        return []
    return get_models()


def get_free_model_ids() -> list[str]:
    return [m["id"] for m in get_free_models()]


def rank_free_models(need_tools: bool = False) -> list[str]:
    """Model ids ordered best-first (tool support first, then by provider priority)."""
    models = get_free_models()
    if need_tools:
        models = [m for m in models if "tools" in (m.get("supported_parameters") or [])]

    # Provider priority: prefer direct providers over free tiers
    provider_priority = {
        "dreamprompting": 0,  # auto router first
        "groq": 1,
        "mistral": 2,
        "nvidia": 3,
        "cohere": 4,
        "google": 5,
        "nvidia-vision": 6,
        "codestral": 7,
        "huggingface": 8,
        "chat": 9,
        "cloudflare": 10,
        "cloudflare-vision": 11,
        "openrouter": 12,
    }

    def _score(m: dict) -> tuple:
        owned = m.get("owned_by", "")
        # Extract provider prefix (e.g., "groq" from "groq/openai/gpt-oss-120b")
        provider = owned.split("/")[0] if "/" in owned else owned
        prio = provider_priority.get(provider, 99)
        has_tools = "tools" in (m.get("supported_parameters") or [])
        return (prio, 0 if has_tools else 1, m["id"])

    return [m["id"] for m in sorted(models, key=_score)]


async def dreamprompting_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from DreamPrompting (OpenAI-compatible)."""
    if not is_configured():
        raise QuotaError(
            "DreamPrompting API key not configured (set DREAMPROMPTING_API_KEY)",
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
                    f"DreamPrompting {response.status_code}: {error_msg}",
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