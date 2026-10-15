"""ZyloAI provider.

ZyloAI is an OpenAI-compatible model gateway. The public catalog lists 126
models under a ``text`` key (not the usual ``data``); each entry carries a
``capabilities`` array where ``tool-use`` marks tool-capable models and a
zero-priced free tier.

The module provides:
- chat completion streaming in OpenAI-compatible format,
- live model catalog fetched from ``/v1/models``,
- capability-based model ranking driven by the catalog's ``tool-use`` flag.
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

PROVIDER_NAME = "zyloai"

# Base URL is configurable via ZYLOAI_BASE_URL (defaults to the public gateway).
ZYLOAI_BASE_URL = settings.zyloai_base_url

# Models confirmed working against the live gateway (tool-use capable, free).
ZYLOAI_FREE_MODELS = [
    "glm-5.3",
    "kimi-k3",
    "muse-glimmer-30b",
    "nemotron-3-ultra",
    "nemotron-3.5-lightning",
]

# Cache
_CACHE: list[dict] = []
_CACHE_TIME: float = 0.0
_CACHE_TTL: float = 300.0


def _headers() -> dict:
    h = {"Accept": "application/json", "Content-Type": "application/json"}
    key = settings.zyloai_api_key
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def is_configured() -> bool:
    return bool(getattr(settings, "zyloai_api_key", ""))


def is_zyloai_model(model: str) -> bool:
    """True when ``model`` is in the ZyloAI catalog."""
    if not model:
        return False
    if model in ZYLOAI_FREE_MODELS:
        return True
    return any(m.get("id") == model for m in _CACHE)


async def fetch_zyloai_models(force_refresh: bool = False) -> list[dict]:
    """Fetch (and cache) the ZyloAI model catalog from ``/v1/models``.

    The catalog is keyed ``text`` rather than ``data``. Entries carry a
    ``capabilities`` list; ``tool-use`` marks tool-capable models.
    """
    global _CACHE, _CACHE_TIME
    if not force_refresh and _CACHE and (time.time() - _CACHE_TIME) < _CACHE_TTL:
        return _CACHE
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{ZYLOAI_BASE_URL}/models", headers=_headers())
            if resp.status_code == 200:
                payload = resp.json()
                raw = payload.get("text") or payload.get("data") or []
                _CACHE = raw
                _CACHE_TIME = time.time()
                logger.info("Discovered %d ZyloAI models", len(_CACHE))
                return _CACHE
            logger.warning("ZyloAI /models returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("Failed to fetch ZyloAI models: %s", e)
    return _CACHE


async def ensure_models() -> list[dict]:
    if not _CACHE or (time.time() - _CACHE_TIME) >= _CACHE_TTL:
        await fetch_zyloai_models()
    return _CACHE


def get_models() -> list[dict]:
    return _CACHE


def get_free_models() -> list[dict]:
    """Catalog entries for the confirmed-working models only.

    The live gateway publishes 126 models, but only the five in
    ``ZYLOAI_FREE_MODELS`` are verified reachable, so routing is restricted to
    those. Metadata (context length, capabilities) is enriched from the live
    catalog when the id is present there.
    """
    by_id = {m.get("id"): m for m in _CACHE}
    out = []
    for mid in ZYLOAI_FREE_MODELS:
        live = by_id.get(mid) or {}
        caps = live.get("capabilities") or ["tool-use"]
        out.append({
            "id": mid,
            "isFree": True,
            "supported_parameters": ["tools"] if "tool-use" in caps else [],
            "context_length": live.get("context_length") or live.get("context_window"),
            "display_name": live.get("name") or mid,
            "owned_by": PROVIDER_NAME,
        })
    return out


def get_model_ids() -> list[str]:
    return [m["id"] for m in get_free_models()]


def _score(m: dict) -> tuple:
    """Rank by tool support, then larger context."""
    params = m.get("supported_parameters") or []
    ctx = m.get("context_length") or 0
    return (0 if "tools" in params else 1, -ctx, m["id"])


def rank_free_models(need_tools: bool = False) -> list[str]:
    """Free model ids ordered best-first (tool support, then context)."""
    models = get_free_models()
    if need_tools:
        models = [m for m in models if "tools" in (m.get("supported_parameters") or [])]
    ranked = sorted(models, key=_score)
    return [m["id"] for m in ranked]


async def zyloai_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from ZyloAI.

    Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint.
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
            f"{ZYLOAI_BASE_URL}/chat/completions",
            json=payload,
            headers=_headers(),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"ZyloAI {response.status_code}: {error_msg}",
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