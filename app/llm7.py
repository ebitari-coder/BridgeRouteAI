"""LLM7 provider.

LLM7.io is an AI gateway with 49 models. The free (turbo) tier offers
6 models at zero cost with 500K-1M tokens/day.

This module provides:
- live discovery of the free model catalog (cached),
- stability monitoring worker that health-checks each free model,
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

PROVIDER_NAME = "llm7"

LLM7_BASE_URL = "https://api.llm7.io/v1"
LLM7_MODELS_URL = "https://api.llm7.io/v1/models"

# Free models cache
_CACHE: list[dict] = []
_CACHE_TIME: float = 0.0
_CACHE_TTL: float = 300.0  # 5 minutes

# ── Stability tracking ──────────────────────────────────────────────────────
_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]
_PROBE_TIMEOUT = 15.0

STABLE_MODELS: set[str] = set()
_model_health: dict[str, dict] = {}

_MIN_SUCCESS_RATE = 50.0
_MIN_CHECKS = 1


def _headers() -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if settings.llm7_api_key:
        headers["Authorization"] = f"Bearer {settings.llm7_api_key}"
    return headers


async def fetch_llm7_models(force_refresh: bool = False) -> list[dict]:
    """Fetch (and cache) the LLM7 model catalog."""
    global _CACHE, _CACHE_TIME

    if not force_refresh and _CACHE and (time.time() - _CACHE_TIME) < _CACHE_TTL:
        return _CACHE

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(LLM7_MODELS_URL, headers=_headers())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                _CACHE = data
                _CACHE_TIME = time.time()
                logger.info("Discovered %d LLM7 models", len(_CACHE))
                return _CACHE
            else:
                logger.warning("LLM7 models returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("Failed to fetch LLM7 models: %s", e)

    return _CACHE


def is_llm7_model(model: str) -> bool:
    return any(m.get("id") == model for m in _CACHE)


async def ensure_models() -> list[dict]:
    """Refresh the catalog if it is missing or stale."""
    if not _CACHE or (time.time() - _CACHE_TIME) >= _CACHE_TTL:
        await fetch_llm7_models()
    return _CACHE


def get_free_models() -> list[dict]:
    """Return free (turbo) models from LLM7."""
    if not _CACHE:
        return []

    result = []
    for m in _CACHE:
        if m.get("tier") == "turbo" and m.get("model_type") == "chat":
            mid = m.get("id", "")
            result.append({
                "id": mid,
                "isFree": True,
                "supported_parameters": ["tools"] if m.get("tools_calling") else [],
                "context_length": (m.get("context_window") or {}).get("tokens"),
                "max_output_tokens": None,
                "display_name": mid.replace("-", " ").replace("_", " ").title(),
                "owned_by": "llm7",
            })

    return result


def _get_health(model_id: str) -> dict:
    if model_id not in _model_health:
        _model_health[model_id] = {"ok": 0, "fail": 0, "last_ok": 0.0}
    return _model_health[model_id]


def _success_rate(h: dict) -> float:
    total = h["ok"] + h["fail"]
    if total == 0:
        return 0.0
    return (h["ok"] / total) * 100.0


def is_model_stable(model_id: str) -> bool:
    if model_id not in STABLE_MODELS:
        return False
    h = _model_health.get(model_id)
    if not h:
        return False
    total = h["ok"] + h["fail"]
    if total < _MIN_CHECKS:
        return False
    return _success_rate(h) >= _MIN_SUCCESS_RATE


async def _probe_model(model_id: str) -> bool:
    """Send a minimal health check request to a model."""
    if not settings.llm7_api_key:
        return False
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            resp = await client.post(
                f"{LLM7_BASE_URL}/chat/completions",
                json={
                    "model": model_id,
                    "messages": _PROBE_MESSAGES,
                    "max_tokens": 1,
                    "stream": False,
                },
                headers=_headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if choices:
                    return True
            if resp.status_code == 429:
                return True
            return False
    except Exception:
        return False


async def refresh_stability() -> int:
    """Refresh the catalog and health-check all free models."""
    await fetch_llm7_models(force_refresh=True)

    free_models = get_free_models()
    free_ids = {m["id"] for m in free_models}

    stale = [mid for mid in _model_health if mid not in free_ids]
    for mid in stale:
        del _model_health[mid]

    for m in free_models:
        mid = m["id"]
        h = _get_health(mid)
        ok = await _probe_model(mid)
        if ok:
            h["ok"] += 1
            h["last_ok"] = time.time()
        else:
            h["fail"] += 1

    stable = set()
    for mid in free_ids:
        h = _model_health.get(mid)
        if h and (h["ok"] + h["fail"]) >= _MIN_CHECKS and _success_rate(h) >= _MIN_SUCCESS_RATE:
            stable.add(mid)

    added = stable - STABLE_MODELS
    removed = STABLE_MODELS - stable

    if added:
        logger.info("LLM7 stability: +[%s] now stable", ", ".join(sorted(added)))
    if removed:
        logger.info("LLM7 stability: -[%s] now unstable", ", ".join(sorted(removed)))

    STABLE_MODELS.clear()
    STABLE_MODELS.update(stable)

    logger.info(
        "LLM7 stability check complete: %d/%d models stable",
        len(STABLE_MODELS),
        len(free_ids),
    )
    return len(STABLE_MODELS)


async def stability_worker() -> None:
    """Background worker that periodically checks LLM7 model stability."""
    interval = settings.llm7_stability_check_seconds
    try:
        count = await refresh_stability()
        logger.info("LLM7 stability worker initial check: %d stable models", count)
    except Exception as exc:
        logger.warning("LLM7 stability initial check failed: %s", exc)

    while True:
        await asyncio.sleep(interval)
        try:
            await refresh_stability()
        except Exception as exc:
            logger.warning("LLM7 stability check failed: %s", exc)


def _score(m: dict) -> float:
    """Score a model for ranking. Higher is better."""
    score = 0.0
    mid = m.get("id", "")
    h = _model_health.get(mid)
    if h:
        score += _success_rate(h) * 10.0
        age = time.time() - h.get("last_ok", 0)
        if age < 600:
            score += 500
        elif age < 1800:
            score += 200
    return score


def rank_free_models(need_tools: bool = False) -> list[str]:
    """Rank free models by capability and stability."""
    models = get_free_models()

    stable = [(m, _score(m)) for m in models if is_model_stable(m["id"])]
    unstable = [(m, _score(m)) for m in models if not is_model_stable(m["id"])]

    stable.sort(key=lambda x: (-x[1], x[0]["id"]))
    unstable.sort(key=lambda x: (-x[1], x[0]["id"]))

    return [m["id"] for m, _ in stable] + [m["id"] for m, _ in unstable]


async def llm7_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from LLM7."""
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
            f"{LLM7_BASE_URL}/chat/completions",
            json=payload,
            headers=_headers(),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"LLM7 {response.status_code}: {error_msg}",
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
