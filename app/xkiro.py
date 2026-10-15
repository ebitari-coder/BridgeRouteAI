"""Xkiro provider.

Xkiro is an OpenAI-compatible API gateway with 111+ models.
The free tier offers 37+ models with tool calling support at zero cost.

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

PROVIDER_NAME = "xkiro"

XKIRO_BASE_URL = "https://api.xkiro.com/v1"
XKIRO_MODELS_URL = "https://api.xkiro.com/v1/models"

XKIRO_AUTO_MODEL = "xkiro/auto"

# Free models cache
FREE_MODELS: list[dict] = []

_CACHE: list[dict] = []
_CACHE_TIME: float = 0.0
_CACHE_TTL: float = 300.0  # 5 minutes

# ── Stability tracking ──────────────────────────────────────────────────────
# Health check probe: minimal request to verify a model is alive.
_PROBE_MODEL = "qwen/qwen3.7-flash:free"  # small, fast, free
_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]
_PROBE_TIMEOUT = 15.0  # seconds

# Stable models set — populated by the background worker.
# Only models in this set are eligible for routing.
STABLE_MODELS: set[str] = set()
_stability_check_time: float = 0.0

# Per-model health tracking: {model_id: {"ok": int, "fail": int, "last_ok": float}}
_model_health: dict[str, dict] = {}

# Thresholds: a model must have succeeded at least once and have a
# success rate >= _MIN_SUCCESS_RATE to be considered stable.
_MIN_SUCCESS_RATE = 50.0  # percent
_MIN_CHECKS = 1           # minimum health checks before gating


def _headers() -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if settings.xkiro_api_key:
        headers["Authorization"] = f"Bearer {settings.xkiro_api_key}"
    return headers


async def fetch_xkiro_models(force_refresh: bool = False) -> list[dict]:
    """Fetch (and cache) the Xkiro model catalog with metadata."""
    global _CACHE, _CACHE_TIME

    if not force_refresh and _CACHE and (time.time() - _CACHE_TIME) < _CACHE_TTL:
        return _CACHE

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(XKIRO_MODELS_URL, headers=_headers())
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                _CACHE = data
                _CACHE_TIME = time.time()
                logger.info("Discovered %d Xkiro models", len(_CACHE))
                return _CACHE
            else:
                logger.warning("Xkiro models returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("Failed to fetch Xkiro models: %s", e)

    return _CACHE


def is_xkiro_model(model: str) -> bool:
    if model == XKIRO_AUTO_MODEL:
        return True
    return any(m.get("id") == model for m in _CACHE)


async def ensure_models() -> list[dict]:
    """Refresh the catalog if it is missing or stale."""
    if not _CACHE or (time.time() - _CACHE_TIME) >= _CACHE_TTL:
        await fetch_xkiro_models()
    return _CACHE


def get_free_models() -> list[dict]:
    """Return free models with tool calling support.

    Xkiro models include an `access_tier` field. Free models have
    `access_tier == "free"`. All models support tool calling per docs.
    """
    if not _CACHE:
        return []

    result = []
    for m in _CACHE:
        if m.get("access_tier") == "free":
            caps = m.get("capabilities", {})
            result.append({
                "id": m["id"],
                "isFree": True,
                "supported_parameters": ["tools"] if caps.get("tools") else [],
                "context_length": m.get("context_length"),
                "max_output_tokens": m.get("max_output_tokens"),
                "display_name": m.get("display_name", m["id"]),
                "owned_by": m.get("owned_by", "unknown"),
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
    """Check if a model meets stability thresholds for routing."""
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
    """Send a minimal health check request to a model. Returns True on success."""
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            resp = await client.post(
                f"{XKIRO_BASE_URL}/chat/completions",
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
            return False
    except Exception:
        return False


async def refresh_stability() -> int:
    """Refresh the catalog and health-check all free models.

    Returns the count of models that passed stability checks.
    """
    global _stability_check_time

    await fetch_xkiro_models(force_refresh=True)

    free_models = get_free_models()
    free_ids = {m["id"] for m in free_models}

    # Prune health entries for models no longer in the catalog
    stale = [mid for mid in _model_health if mid not in free_ids]
    for mid in stale:
        del _model_health[mid]

    # Health-check each free model
    for m in free_models:
        mid = m["id"]
        h = _get_health(mid)
        ok = await _probe_model(mid)
        if ok:
            h["ok"] += 1
            h["last_ok"] = time.time()
        else:
            h["fail"] += 1

    # Update STABLE_MODELS set
    stable = set()
    for mid in free_ids:
        h = _model_health.get(mid)
        if h and (h["ok"] + h["fail"]) >= _MIN_CHECKS and _success_rate(h) >= _MIN_SUCCESS_RATE:
            stable.add(mid)

    added = stable - STABLE_MODELS
    removed = STABLE_MODELS - stable

    if added:
        logger.info("Xkiro stability: +[%s] now stable", ", ".join(sorted(added)))
    if removed:
        logger.info("Xkiro stability: -[%s] now unstable", ", ".join(sorted(removed)))

    STABLE_MODELS.clear()
    STABLE_MODELS.update(stable)
    _stability_check_time = time.time()

    logger.info(
        "Xkiro stability check complete: %d/%d models stable",
        len(STABLE_MODELS),
        len(free_ids),
    )
    return len(STABLE_MODELS)


async def stability_worker() -> None:
    """Background worker that periodically checks Xkiro model stability.

    Runs every ``settings.xkiro_stability_check_seconds`` (default 300s).
    When models pass health checks they are added to STABLE_MODELS, which
    the routing layer checks before attempting a request.
    """
    interval = settings.xkiro_stability_check_seconds
    # Initial check immediately on startup
    try:
        count = await refresh_stability()
        logger.info("Xkiro stability worker initial check: %d stable models", count)
    except Exception as exc:
        logger.warning("Xkiro stability initial check failed: %s", exc)

    while True:
        await asyncio.sleep(interval)
        try:
            await refresh_stability()
        except Exception as exc:
            logger.warning("Xkiro stability check failed: %s", exc)


def _score(m: dict) -> float:
    """Score a model for ranking. Higher is better."""
    score = 0.0
    mid = m.get("id", "")
    h = _model_health.get(mid)
    if h:
        # Bonus for high success rate
        score += _success_rate(h) * 10.0  # max +1000
        # Bonus for recency of last successful check
        age = time.time() - h.get("last_ok", 0)
        if age < 600:
            score += 500  # recently verified
        elif age < 1800:
            score += 200
    # Context length bonus
    cl = m.get("context_length") or 0
    score += min(cl, 1_000_000) / 1000.0  # max +1000
    return score


def rank_free_models(need_tools: bool = False) -> list[str]:
    """Rank free models by capability and stability.

    Priority: stable models first (sorted by score), then unstable ones.
    All Xkiro free models support tools.
    """
    models = get_free_models()
    if need_tools:
        models = [m for m in models if "tools" in (m.get("supported_parameters") or [])]

    stable = [(m, _score(m)) for m in models if is_model_stable(m["id"])]
    unstable = [(m, _score(m)) for m in models if not is_model_stable(m["id"])]

    stable.sort(key=lambda x: (-x[1], x[0]["id"]))
    unstable.sort(key=lambda x: (-x[1], x[0]["id"]))

    return [m["id"] for m, _ in stable] + [m["id"] for m, _ in unstable]


async def xkiro_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from Xkiro.

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
            f"{XKIRO_BASE_URL}/chat/completions",
            json=payload,
            headers=_headers(),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"Xkiro {response.status_code}: {error_msg}",
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
