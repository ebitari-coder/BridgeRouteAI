"""Cloudflare Workers AI provider.

Cloudflare Workers AI offers a free tier (10,000 requests/day) on all chat
models via their REST API at
``https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{model}``.

This module provides:

* chat completion streaming in OpenAI-compatible format,
* capability-based model ranking,
* model catalog (static list of free chat models from the Workers AI catalog).
"""
import uuid
import time
import json
import logging
from typing import AsyncIterator

import httpx

from app.config import settings
from app.provider_utils import (
    QuotaError,
    STRICT_MAX_TOKENS_CAP,
    SamplingParams,
    apply_sampling,
    sse_chunk,
    usage_chunk,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "cloudflare"

BASE_URL = "https://api.cloudflare.com/client/v4"

CLOUDFLARE_AUTO_MODEL = "cloudflare/auto"

CLOUDFLARE_CHAT_MODELS = [
    "@cf/meta/llama-3.2-1b-instruct",
    "@cf/meta/llama-3.2-3b-instruct",
    "@cf/meta/llama-3.1-8b-instruct-fp8",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/meta/llama-2-7b-chat-hf-lora",
    "@cf/meta/gemma-sea-lion-v4-27b-it",
    "@cf/meta/kimi-k2.6",
    "@cf/meta/kimi-k2.7-code",
    "@cf/mistral/mistral-7b-instruct-v0.2-lora",
    "@cf/mistral/mistral-small-3.1-24b-instruct",
    "@cf/google/gemma-2b-it-lora",
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/google/gemma-7b-it-lora",
    "@cf/deepseek/deepseek-r1-distill-qwen-32b",
    "@cf/deepseek/deepseek-v4-flash-0731",
    "@cf/deepseek/deepseek-v4-pro-0813",
    "@cf/zai-org/glm-4.7-flash",
    "@cf/zai-org/glm-5.2",
    "@cf/zai-org/glm-5.3",
    "@cf/zai-org/glm-5.3-flash",
    "@cf/openai/gpt-oss-120b",
    "@cf/openai/gpt-oss-20b",
    "@cf/ibm/granite-4.0-h-micro",
    "@cf/nvidia/nemotron-3-120b-a12b",
    "@cf/qwen/qwen2.5-coder-32b-instruct",
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/qwen/qwen3.8-27b",
    "@cf/qwen/qwq-32b",
]

CLOUDFLARE_MODEL_CONTEXT: dict[str, int] = {
    "@cf/meta/llama-3.2-1b-instruct": 60_000,
    "@cf/meta/llama-3.2-3b-instruct": 80_000,
    "@cf/meta/llama-3.1-8b-instruct-fp8": 32_000,
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": 24_000,
    "@cf/meta/llama-4-scout-17b-16e-instruct": 131_000,
    "@cf/meta/llama-2-7b-chat-hf-lora": 8_200,
    "@cf/meta/gemma-sea-lion-v4-27b-it": 128_000,
    "@cf/meta/kimi-k2.6": 262_100,
    "@cf/meta/kimi-k2.7-code": 262_100,
    "@cf/mistral/mistral-7b-instruct-v0.2-lora": 15_000,
    "@cf/mistral/mistral-small-3.1-24b-instruct": 128_000,
    "@cf/google/gemma-2b-it-lora": 3_500,
    "@cf/google/gemma-4-26b-a4b-it": 256_000,
    "@cf/google/gemma-7b-it-lora": 3_500,
    "@cf/deepseek/deepseek-r1-distill-qwen-32b": 80_000,
    "@cf/deepseek/deepseek-v4-flash-0731": 1_000_000,
    "@cf/deepseek/deepseek-v4-pro-0813": 1_000_000,
    "@cf/zai-org/glm-4.7-flash": 131_100,
    "@cf/zai-org/glm-5.2": 262_100,
    "@cf/zai-org/glm-5.3": 1_300_000,
    "@cf/zai-org/glm-5.3-flash": 1_300_000,
    "@cf/openai/gpt-oss-120b": 128_000,
    "@cf/openai/gpt-oss-20b": 128_000,
    "@cf/ibm/granite-4.0-h-micro": 131_000,
    "@cf/nvidia/nemotron-3-120b-a12b": 256_000,
    "@cf/qwen/qwen2.5-coder-32b-instruct": 32_800,
    "@cf/qwen/qwen3-30b-a3b-fp8": 32_800,
    "@cf/qwen/qwen3.8-27b": 262_100,
    "@cf/qwen/qwq-32b": 24_000,
}

# The legacy `/ai/run` payload this module speaks carries messages/stream only,
# so tool calling is never actually forwarded: advertise it as unsupported
# (see get_attempt_plan, whose tool-capable branch already excludes Cloudflare).
CLOUDFLARE_MODEL_TOOLS: dict[str, bool] = {m: False for m in CLOUDFLARE_CHAT_MODELS}

CLOUDFLARE_MODEL_BENCHMARK: dict[str, float] = {
    "@cf/meta/llama-3.2-1b-instruct": 0.3,
    "@cf/meta/llama-3.2-3b-instruct": 0.5,
    "@cf/meta/llama-3.1-8b-instruct-fp8": 0.55,
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": 0.85,
    "@cf/meta/llama-4-scout-17b-16e-instruct": 0.8,
    "@cf/meta/llama-2-7b-chat-hf-lora": 0.25,
    "@cf/meta/gemma-sea-lion-v4-27b-it": 0.55,
    "@cf/meta/kimi-k2.6": 0.9,
    "@cf/meta/kimi-k2.7-code": 0.92,
    "@cf/mistral/mistral-7b-instruct-v0.2-lora": 0.35,
    "@cf/mistral/mistral-small-3.1-24b-instruct": 0.65,
    "@cf/google/gemma-2b-it-lora": 0.2,
    "@cf/google/gemma-4-26b-a4b-it": 0.78,
    "@cf/google/gemma-7b-it-lora": 0.3,
    "@cf/deepseek/deepseek-r1-distill-qwen-32b": 0.82,
    "@cf/deepseek/deepseek-v4-flash-0731": 0.95,
    "@cf/deepseek/deepseek-v4-pro-0813": 0.98,
    "@cf/zai-org/glm-4.7-flash": 0.85,
    "@cf/zai-org/glm-5.2": 0.95,
    "@cf/zai-org/glm-5.3": 0.99,
    "@cf/zai-org/glm-5.3-flash": 0.93,
    "@cf/openai/gpt-oss-120b": 0.88,
    "@cf/openai/gpt-oss-20b": 0.72,
    "@cf/ibm/granite-4.0-h-micro": 0.6,
    "@cf/nvidia/nemotron-3-120b-a12b": 0.8,
    "@cf/qwen/qwen2.5-coder-32b-instruct": 0.7,
    "@cf/qwen/qwen3-30b-a3b-fp8": 0.8,
    "@cf/qwen/qwen3.8-27b": 0.85,
    "@cf/qwen/qwq-32b": 0.83,
}


_CACHE: list[dict] = []
_CACHE_TIME: float = 0.0
_CACHE_TTL: float = 300.0


async def fetch_cloudflare_models(force_refresh: bool = False) -> list[dict]:
    """Fetch the live Workers AI catalog (models/search) for this account.

    The static seed below drifts out of date fast — models get retired ("No
    route for that URI") or moved to paid plans. Live ids win; on failure the
    seed remains the fallback.
    """
    global _CACHE, _CACHE_TIME
    if not force_refresh and _CACHE and (time.time() - _CACHE_TIME) < _CACHE_TTL:
        return _CACHE
    url = f"{BASE_URL}/accounts/{settings.cloudflare_account_id}/ai/models/search?per_page=100"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=_headers())
            payload = resp.json()
            if resp.status_code == 200 and payload.get("success"):
                live = []
                for m in payload.get("result", []):
                    if not isinstance(m, dict):
                        continue
                    task = (m.get("task") or {})
                    task_name = task.get("name", "") if isinstance(task, dict) else ""
                    if "Text Generation" not in task_name:
                        continue  # image/audio/embedding models can't chat
                    # properties is a LIST of {property_id, value} pairs
                    props: dict = {}
                    for pr in m.get("properties") or []:
                        if isinstance(pr, dict) and pr.get("property_id"):
                            props[pr["property_id"]] = pr.get("value")
                    try:
                        ctx = int(props.get("context_window") or 0)
                    except (TypeError, ValueError):
                        ctx = 0
                    live.append({"id": m.get("name"), "context_length": ctx or None})
                if live:
                    _CACHE = live
                    _CACHE_TIME = time.time()
                    logger.info("Discovered %d live Cloudflare chat models", len(_CACHE))
                    return _CACHE
    except Exception as e:
        logger.warning("Cloudflare catalog refresh failed: %s", e)
    return _CACHE


async def ensure_models() -> list[dict]:
    await fetch_cloudflare_models()
    return _CACHE


def _headers(accept: str = "application/json") -> dict:
    headers = {"Accept": accept, "Content-Type": "application/json"}
    if settings.cloudflare_api_key:
        headers["Authorization"] = f"Bearer {settings.cloudflare_api_key}"
    return headers


def _run_url(model: str) -> str:
    return f"{BASE_URL}/accounts/{settings.cloudflare_account_id}/ai/run/{model}"


def is_cloudflare_model(model: str) -> bool:
    if model == CLOUDFLARE_AUTO_MODEL or model.startswith("@cf/"):
        return True
    return model in CLOUDFLARE_CHAT_MODELS


def get_free_models() -> list[dict]:
    """Live catalog first (tool calling enabled — the OpenAI-compatible
    gateway forwards tools), static seed as fallback when refresh fails."""
    live = {m["id"]: m for m in _CACHE} if _CACHE else {}
    ids = list(live.keys()) or CLOUDFLARE_CHAT_MODELS
    return [
        {
            "id": mid,
            "isFree": True,
            "supported_parameters": ["tools"],
            "context_length": (live.get(mid) or {}).get("context_length")
            or CLOUDFLARE_MODEL_CONTEXT.get(mid),
        }
        for mid in ids
    ]


def get_free_model_ids() -> list[str]:
    return [m["id"] for m in get_free_models()]


def _score(m: dict) -> float:
    model_id = m.get("id", "")
    score = 0.0
    score += min(m.get("context_length") or 0, 2_000_000) / 1000.0
    bench = CLOUDFLARE_MODEL_BENCHMARK.get(model_id, 0)
    score += bench * 1000
    return score


def rank_free_models(need_tools: bool = False) -> list[str]:
    candidates = []
    for m in get_free_models():
        params = m.get("supported_parameters") or []
        if need_tools and "tools" not in params:
            continue
        candidates.append((_score(m), m["id"]))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return [mid for _, mid in candidates]


async def cloudflare_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from Cloudflare Workers AI.

    Uses the legacy ``ai/run/{model}`` endpoint which accepts a simplified
    ``{"messages": [...]}`` body and returns SSE with ``data: {"response": "..."}``
    chunks where ``response`` is a text delta.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    simple_messages = []
    for msg in messages:
        role = msg_get(msg, "role", "user")
        content = msg_get(msg, "content") or ""
        simple_messages.append({"role": role, "content": str(content)})

    # OpenAI-compatible gateway (forwards tools + tool_choice, unlike the
    # legacy /ai/run used before).
    payload: dict = {"model": model, "messages": messages, "stream": True}
    if tools:
        payload["tools"] = tools
    apply_sampling(payload, sampling, max_tokens_cap=STRICT_MAX_TOKENS_CAP)

    prompt_tokens_est = max(1, sum(len(str(msg_get(m, "content") or "")) for m in messages) // 4)
    accumulated_text = ""
    yielded_role = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/accounts/{settings.cloudflare_account_id}/ai/v1/chat/completions",
            json=payload,
            headers=_headers(accept="text/event-stream"),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"Cloudflare {response.status_code}: {error_msg}",
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

                # Advanced OpenAI-compatible format: choices[0].delta.content
                choices = parsed.get("choices") or []
                text_delta = None
                finish_reason = None
                if choices:
                    delta = choices[0].get("delta") or {}
                    text_delta = delta.get("content")
                    finish_reason = choices[0].get("finish_reason")
                    if text_delta is None and delta.get("reasoning"):
                        text_delta = delta.get("reasoning")

                # Simple format: response field
                if text_delta is None:
                    text_delta = parsed.get("response")

                if text_delta:
                    accumulated_text += text_delta

                if text_delta:
                    if not yielded_role:
                        yield sse_chunk(chat_id, created, model, role="assistant")
                        yielded_role = True
                    yield sse_chunk(chat_id, created, model, content=text_delta)

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


def msg_get(msg, key: str, default=None):
    if hasattr(msg, key):
        return getattr(msg, key)
    if hasattr(msg, "get"):
        return msg.get(key, default)
    return default


async def cloudflare_collect(
    model: str,
    messages: list,
    tools: list = None,
) -> tuple[str, str, dict | None]:
    """Collect a full non-streaming response with failover."""
    full_text = ""
    actual_model = model
    error = None
    tool_call_acc: dict[int, dict] = {}

    async for chunk in cloudflare_stream_single(model, messages, include_usage=False, tools=tools):
        if not chunk.startswith("data:"):
            continue
        data_str = chunk[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            parsed = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if "error" in parsed:
            error = parsed["error"]
            break

        if parsed.get("model"):
            actual_model = parsed["model"]

        choices = parsed.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                full_text += content
            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index", 0)
                slot = tool_call_acc.setdefault(idx, {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                args = fn.get("arguments")
                if args:
                    slot["function"]["arguments"] += args

    if error:
        return "", actual_model, error

    if tool_call_acc:
        tool_calls = []
        for idx in sorted(tool_call_acc):
            slot = tool_call_acc[idx]
            if not slot["id"]:
                slot["id"] = f"call_{uuid.uuid4().hex[:24]}"
            tool_calls.append(slot)
        return json.dumps({"content": full_text, "tool_calls": tool_calls}), actual_model, None

    return full_text, actual_model, None