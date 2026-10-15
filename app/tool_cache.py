"""Redis-backed tool result cache with in-process fallback.

Caches successful results for a small allow-list of idempotent/read-only tools.
Key = hash(tool name + canonical JSON of sanitized args). TTL from TOOL_CACHE_TTL.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Successful, side-effect-free (or idempotent) tools only.
CACHEABLE_TOOLS = frozenset({
    "calculate",
    "read_file",
    "web_search",
    "http_request",
    "echo_test",
})

# Explicit non-cache list (side effects / non-determinism).
NON_CACHEABLE_TOOLS = frozenset({
    "bash_command",
    "write_file",
    "sleep",
})

_fallback_store: dict[str, tuple[str, float]] = {}
_fallback_lock = threading.Lock()

DEFAULT_TTL = 300


def _ttl() -> int:
    try:
        return max(1, int(os.environ.get("TOOL_CACHE_TTL", DEFAULT_TTL)))
    except ValueError:
        return DEFAULT_TTL


def cache_enabled() -> bool:
    return os.environ.get("TOOL_CACHE_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def should_cache(tool_name: str) -> bool:
    if not cache_enabled():
        return False
    if tool_name in NON_CACHEABLE_TOOLS:
        return False
    return tool_name in CACHEABLE_TOOLS


def make_cache_key(tool_name: str, args: dict | list | str | None) -> str:
    """Canonical key: tool name + stable JSON of args."""
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            parsed = args
        raw = parsed
    else:
        raw = args if args is not None else {}
    try:
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        canonical = str(raw)
    digest = hashlib.sha256(f"{tool_name}\n{canonical}".encode("utf-8")).hexdigest()
    return f"toolcache:{tool_name}:{digest}"


async def get_cached(key: str) -> str | None:
    """Return cached payload string or None. Redis first, then in-process fallback."""
    try:
        from app.redis_client import get_redis
        r = await get_redis()
        val = await r.get(key)
        if val is not None:
            return str(val)
    except Exception:
        pass

    now = time.time()
    with _fallback_lock:
        item = _fallback_store.get(key)
        if item is None:
            return None
        value, expires = item
        if expires <= now:
            _fallback_store.pop(key, None)
            return None
        return value


async def set_cached(key: str, value: str, ttl: int | None = None) -> None:
    """Store payload in Redis (SET EX) with in-process fallback."""
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    effective_ttl = int(ttl) if ttl else _ttl()

    try:
        from app.redis_client import get_redis
        r = await get_redis()
        await r.set(key, value, ex=effective_ttl)
        return
    except Exception:
        logger.debug("tool cache Redis unavailable, using in-process store", exc_info=True)

    with _fallback_lock:
        # Cap fallback size to avoid unbounded growth
        if len(_fallback_store) >= 2048:
            now = time.time()
            expired = [k for k, (_, exp) in _fallback_store.items() if exp <= now]
            for k in expired:
                _fallback_store.pop(k, None)
            while len(_fallback_store) >= 2048:
                _fallback_store.pop(next(iter(_fallback_store)), None)
        _fallback_store[key] = (value, time.time() + effective_ttl)


async def get_or_none(tool_name: str, args: dict | list | str | None) -> str | None:
    if not should_cache(tool_name):
        return None
    return await get_cached(make_cache_key(tool_name, args))


async def put(tool_name: str, args: dict | list | str | None, value: str) -> None:
    if not should_cache(tool_name):
        return
    await set_cached(make_cache_key(tool_name, args), value)
