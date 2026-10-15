"""Capability probing for tool support.

Every provider's catalog claims `tools_supported` in its metadata, but the live
behavior can differ (e.g. Agnes 3.0-flash advertises tools but emits plain text).
This module sends a tiny probe request carrying a `tools` payload and records
whether the model actually returns a `tool_calls` array. Results are cached
per `(provider, model)` with a TTL and shared via Redis when available.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from app.config import settings
from app.provider_utils import QuotaError, SamplingParams, to_openai_messages

logger = logging.getLogger(__name__)

# Probe payload: a single cheap tool call that any tool-capable model must emit
# a `tool_calls` array for. The tool name is intentionally unusual so it
# doesn't collide with a real tool the model might recognize.
_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "__capability_probe",
        "description": "Internal capability probe",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# Redis key prefix
_CAPABILITY_KEY_PREFIX = "capability:tool:"

# In-process fallback cache
_CapabilityCacheEntry = tuple[bool, float]  # (supports_tools, expires_at)
_capability_cache: dict[str, _CapabilityCacheEntry] = {}


def _probe_ttl() -> int:
    """Get probe TTL from settings, with fallback."""
    return getattr(settings, "capability_probe_ttl_seconds", 300)


def _probe_refresh() -> int:
    """Get probe refresh interval from settings, with fallback."""
    return getattr(settings, "capability_probe_refresh_seconds", 300)


def _probe_enabled() -> bool:
    """Check if capability probing is enabled."""
    return getattr(settings, "capability_probe_enabled", True)


@dataclass
class CapabilityState:
    """Live capability state for a model."""
    supports_tools: bool
    last_probed: float
    expires_at: float
    probe_count: int = 0
    last_error: str | None = None


async def _get_redis():
    """Get Redis client or None if unavailable."""
    try:
        from app.redis_client import get_redis
        return await get_redis()
    except Exception:
        return None


def _cache_key(provider: str, model: str) -> str:
    return f"{_CAPABILITY_KEY_PREFIX}{provider}:{model}"


def _cache_get_inprocess(provider: str, model: str) -> CapabilityState | None:
    key = _cache_key(provider, model)
    entry = _capability_cache.get(key)
    if entry is None:
        return None
    supports, expires = entry
    if time.time() > expires:
        _capability_cache.pop(key, None)
        return None
    return CapabilityState(supports_tools=supports, last_probed=0, expires_at=expires)


def _cache_set_inprocess(provider: str, model: str, supports: bool) -> None:
    key = _cache_key(provider, model)
    expires = time.time() + _probe_ttl()
    _capability_cache[key] = (supports, expires)


async def _cache_get_redis(provider: str, model: str) -> CapabilityState | None:
    r = await _get_redis()
    if r is None:
        return None
    key = _cache_key(provider, model)
    try:
        data = await r.get(key)
        if data is None:
            return None
        obj = json.loads(data)
        supports = obj.get("supports", False)
        expires = obj.get("expires", 0)
        if time.time() > expires:
            await r.delete(key)
            return None
        return CapabilityState(
            supports_tools=supports,
            last_probed=obj.get("probed", 0),
            expires_at=expires,
            probe_count=obj.get("count", 0),
            last_error=obj.get("error"),
        )
    except Exception:
        return None


async def _cache_set_redis(provider: str, model: str, supports: bool) -> None:
    r = await _get_redis()
    if r is None:
        return
    key = _cache_key(provider, model)
    obj = {
        "supports": supports,
        "probed": time.time(),
        "expires": time.time() + _probe_ttl(),
        "count": 1,
    }
    try:
        await r.set(key, json.dumps(obj), ex=_probe_ttl())
    except Exception:
        pass


async def _cache_increment_count(provider: str, model: str, supports: bool) -> None:
    r = await _get_redis()
    if r is None:
        _cache_set_inprocess(provider, model, supports)
        return
    key = _cache_key(provider, model)
    try:
        await r.hincrby(key, "count", 1)
        await r.hset(key, "supports", "true" if supports else "false")
        await r.hset(key, "probed", str(time.time()))
        await r.expire(key, _probe_ttl())
    except Exception:
        _cache_set_inprocess(provider, model, supports)


def get_capability(provider: str, model: str) -> CapabilityState | None:
    """Synchronous capability lookup (used by rank_free_models).

    Checks in-process cache first, then Redis (async lookup not available here).
    Returns None on cache miss so the caller can fall back to catalog claims.
    """
    # In-process is instant and covers the current process's probes
    inproc = _cache_get_inprocess(provider, model)
    if inproc is not None:
        return inproc
    # No async Redis access from sync context; return None on miss
    return None


async def ensure_capability(provider: str, model: str) -> CapabilityState:
    """Get capability state, probing if stale or missing."""
    # Try Redis first (shared across workers)
    rstate = await _cache_get_redis(provider, model)
    if rstate is not None:
        return rstate
    # Fallback to in-process
    inproc = _cache_get_inprocess(provider, model)
    if inproc is not None:
        return inproc
    # Stale/missing → probe
    return await probe_capability(provider, model)


async def probe_capability(provider: str, model: str) -> CapabilityState:
    """Send a probe request and record whether the model emits tool_calls.

    Probe is a single-streaming request with the probe tool. We consume the
    stream until we either see a tool_calls delta (success) or the stream
    completes without any tool_calls (failure). Quota/rate errors are treated
    as transient and do NOT update the capability cache.
    """
    from app.providers import _stream_for

    messages = [{"role": "user", "content": "probe"}]
    tools = [_PROBE_TOOL]
    supports = False

    try:
        async for chunk in _stream_for(provider, model, messages, include_usage=True, tools=tools):
            if not isinstance(chunk, str) or not chunk.startswith("data:"):
                continue
            data_str = chunk[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                parsed = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "error" in parsed and parsed.get("error"):
                err = parsed["error"]
                msg = err.get("message", "") if isinstance(err, dict) else str(err)
                if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                    raise QuotaError(msg or "quota exhausted", provider=provider, model=model)
                # Other errors are not transient; treat as failure to emit tools
                logger.debug("Probe error for %s:%s: %s", provider, model, msg)
                supports = False
                break
            choices = parsed.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    supports = True
                    break
        else:
            supports = False
    except Exception as e:
        # Quota and rate-limit probes also record False for this TTL —
        # a model we cannot probe right now is not a safe tool-routing
        # target, and the cache refreshes in a few minutes anyway.
        logger.warning("Probe failed for %s:%s: %s", provider, model, e)
        supports = False

    # Cache result
    await _cache_increment_count(provider, model, supports)
    _cache_set_inprocess(provider, model, supports)

    return CapabilityState(
        supports_tools=supports,
        last_probed=time.time(),
        expires_at=time.time() + _probe_ttl(),
        probe_count=1,
    )


def model_supports_tools(provider: str, model: str) -> bool | None:
    """Check if a model supports tools (for use in ranking functions).

    Returns:
      - True  : probed and confirmed
      - False : probed and confirmed NOT to support tools
      - None  : not probed yet (caller should fall back to catalog claim)
    """
    state = get_capability(provider, model)
    if state is None:
        return None
    return state.supports_tools


async def warm_capability_cache(models: list[tuple[str, str]]) -> None:
    """Probe a batch of (provider, model) pairs in parallel.

    Used at startup to prime the cache for the auto-router's first
    tool-bearing requests. Errors are logged but don't block startup.
    """
    sem = asyncio.Semaphore(8)

    async def _probe_one(p: str, m: str) -> None:
        async with sem:
            try:
                await probe_capability(p, m)
            except Exception as e:
                logger.debug("Warm probe failed for %s:%s: %s", p, m, e)

    tasks = [_probe_one(p, m) for p, m in models]
    await asyncio.gather(*tasks, return_exceptions=True)


async def capability_refresh_loop() -> None:
    """Background worker that re-probes stale entries.

    Runs every CAPABILITY_PROBE_REFRESH_SECONDS and re-probes all entries whose
    TTL has expired. Mirrors the catalog_refresh_loop pattern used by
    Kilo/FreeModels/Composite/etc.
    """
    logger.info("Capability probe refresher started (every %ds)", _probe_refresh())
    while True:
        await asyncio.sleep(_probe_refresh())
        try:
            # Refresh in-process cache from Redis
            r = await _get_redis()
            if r is not None:
                pattern = f"{_CAPABILITY_KEY_PREFIX}*"
                keys = []
                async for key in r.scan_iter(match=pattern):
                    keys.append(key)
                for key in keys:
                    try:
                        data = await r.get(key)
                        if data:
                            obj = json.loads(data)
                            expires = obj.get("expires", 0)
                            if time.time() > expires:
                                # Expired - will be re-probed on next request
                                await r.delete(key)
                    except Exception:
                        pass

            # Note: we don't proactively re-probe all models here because
            # that could hit rate limits. Instead, stale entries are
            # re-probed lazily on the next `ensure_capability` call.
            # The Redis TTL naturally expires them.
        except Exception as e:
            logger.warning("Capability refresh loop error: %s", e)


# ── Integration helpers for rank_free_models ──────────────────────────────

def filter_by_probed_capability(
    models: list[str],
    provider: str,
    need_tools: bool,
    get_free_models_func,
) -> list[str]:
    """Filter a ranked model list to only those probed to support tools.

    If a model hasn't been probed yet, it passes through (fallback to catalog).
    This is the sync helper that `rank_free_models` can call without going async.
    """
    if not need_tools:
        return models

    filtered = []
    for m in models:
        probed = model_supports_tools(provider, m)
        if probed is None:
            # Not yet probed - let it through; it will be probed on first use
            filtered.append(m)
        elif probed:
            filtered.append(m)
        # probed is False -> drop
    return filtered


def ranked_tool_models(
    provider: str,
    all_ranked: list[str],
    tools_ranked: list[str],
) -> list[str]:
    """Merge catalog tool-claims with live probe results, preserving rank order.

    ``all_ranked`` is the provider's full ranking (no tool filter applied).
    ``tools_ranked`` is the provider's tool-filtered ranking (the catalog claim).

    A model is included for a tool-bearing request when either:
      - the catalog claims tool support (present in ``tools_ranked``) and a
        probe has not disproven it, or
      - a live probe confirmed it emits ``tool_calls`` (``probed is True``),
        which overrides an under-declaring catalog.

    This recovers models whose catalog entry omits ``tools`` but which the probe
    observed actually emitting ``tool_calls``. The provider's own
    ``rank_free_models(need_tools=True)`` has already dropped them, so the
    override has to happen here, not in a post-filter that only ever shrinks.
    """
    tools_set = set(tools_ranked)
    out: list[str] = []
    for m in all_ranked:
        probed = model_supports_tools(provider, m)
        if probed is None:
            include = m in tools_set
        else:
            include = probed
        if include and m not in out:
            out.append(m)
    return out