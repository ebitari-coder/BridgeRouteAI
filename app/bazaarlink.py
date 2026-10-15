"""BazaarLink provider.

BazaarLink is an OpenAI-compatible API gateway that aggregates
models from multiple providers. The free tier offers rate-limited
access to select models at no cost.

This module provides:
- chat completion streaming in OpenAI-compatible format,
- model catalog (static list of free chat models from BazaarLink),
- capability-based model ranking.
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

PROVIDER_NAME = "bazaarlink"

BAZAARLINK_BASE_URL = "https://api.bazaarlink.ai/v1"

BAZAARLINK_AUTO_MODEL = "auto:free"

BAZAARLINK_CHAT_MODELS = [
    "deepseek/deepseek-v4-flash-0731free:free",
    "qwen/qwen3.7-flash:free",
]

BAZAARLINK_MODEL_CONTEXT: dict[str, int] = {
    "deepseek/deepseek-v4-flash-0731free:free": 1_048_576,
    "qwen/qwen3.7-flash:free": 1_000_000,
}

BAZAARLINK_MODEL_TOOLS: dict[str, bool] = {m: False for m in BAZAARLINK_CHAT_MODELS}

BAZAARLINK_MODEL_BENCHMARK: dict[str, float] = {
    "deepseek/deepseek-v4-flash-0731free:free": 0.85,
    "qwen/qwen3.7-flash:free": 0.65,
}


def _headers() -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if settings.bazaarlink_api_key:
        headers["Authorization"] = f"Bearer {settings.bazaarlink_api_key}"
    return headers


def _bl_enabled() -> bool:
    """Kill switch — BAZAARLINK_ENABLED=false removes BazaarLink from
    routing/catalogs/plans (default: disabled).
    """
    try:
        return bool(settings.bazaarlink_enabled)
    except Exception:
        return False


def is_bazaarlink_model(model: str) -> bool:
    if not _bl_enabled():
        return False
    if model == BAZAARLINK_AUTO_MODEL:
        return True
    return model in BAZAARLINK_CHAT_MODELS


def get_free_models() -> list[dict]:
    if not _bl_enabled():
        return []
    return [
        {
            "id": mid,
            "isFree": True,
            "supported_parameters": ["tools"] if BAZAARLINK_MODEL_TOOLS.get(mid) else [],
            "context_length": BAZAARLINK_MODEL_CONTEXT.get(mid),
        }
        for mid in BAZAARLINK_CHAT_MODELS
    ]


def get_free_model_ids() -> list[str]:
    return [m["id"] for m in get_free_models()]


def _score(m: dict) -> float:
    model_id = m.get("id", "")
    score = 0.0
    score += min(m.get("context_length") or 0, 2_000_000) / 1000.0
    bench = BAZAARLINK_MODEL_BENCHMARK.get(model_id, 0)
    score += bench * 1000
    return score


def rank_free_models(need_tools: bool = False) -> list[str]:
    if not _bl_enabled():
        return []
    candidates = []
    for m in get_free_models():
        params = m.get("supported_parameters") or []
        if need_tools and "tools" not in params:
            continue
        candidates.append((_score(m), m["id"]))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return [mid for _, mid in candidates]


async def bazaarlink_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from BazaarLink.

    Uses the OpenAI-compatible endpoint at ``/v1/chat/completions``
    with standard OpenAI request body and SSE streaming response.
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

    prompt_tokens_est = max(1, sum(len(msg_get(msg, "content", "")) for msg in messages) // 4)
    accumulated_text = ""
    yielded_role = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{BAZAARLINK_BASE_URL}/chat/completions",
            json=payload,
            headers=_headers(),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"BazaarLink {response.status_code}: {error_msg}",
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
