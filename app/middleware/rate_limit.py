"""Rate limiting middleware using slowapi + Redis-backed per-key/combo limits."""
import logging
import time
import threading
from collections import defaultdict
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Global limiter — 60 req/min for general, 5/min for login
limiter = Limiter(key_func=get_remote_address)

# Fallback limit for credentials with no `api_keys` row (the admin key and
# dashboard JWTs) when they are used against /v1/*. These used to bypass rate
# limiting entirely, so a single JWT could hammer the gateway unbounded.
DEFAULT_V1_LIMIT_PER_MINUTE = 120

# In-process fallback counters: key -> list of (timestamp, count-window start)
_fallback_counts: dict[str, list[float]] = defaultdict(list)
_fallback_lock = threading.Lock()


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    logger.warning("Rate limit exceeded for %s on %s", request.client.host if request.client else "unknown", request.url.path)
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": "Too many requests. Please try again later.",
                "type": "rate_limit_error",
                "param": None,
                "code": "rate_limit_exceeded",
            }
        },
        headers={"Retry-After": "60"},
    )


def _rate_limit_error(retry_after: int = 60) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": f"Rate limit exceeded. Try again in {retry_after}s.",
                "type": "rate_limit_error",
                "param": None,
                "code": "rate_limit_exceeded",
            }
        },
        headers={"Retry-After": str(retry_after)},
    )


async def _window_counter(scope_id: str, limit_per_minute: int) -> bool:
    """Fixed-window per-minute counter for one scope. True = allowed.

    ``scope_id`` is either an API key id or ``combo:{combo_id}`` so that every
    key scoped to the same combo shares one budget. Uses Redis when available
    and falls back to an in-process dict (TR-15.2).
    """
    if not scope_id or not limit_per_minute or limit_per_minute <= 0:
        return True

    window = int(time.time() // 60)
    redis_key = f"ratelimit:{scope_id}:{window}"

    # Try Redis first
    try:
        from app.redis_client import get_redis
        r = await get_redis()
        count = await r.incr(redis_key)
        if count == 1:
            await r.expire(redis_key, 120)
        return int(count) <= int(limit_per_minute)
    except Exception:
        # In-process fallback
        now = time.time()
        with _fallback_lock:
            bucket = _fallback_counts[scope_id]
            # Drop entries outside current minute window
            _fallback_counts[scope_id] = [t for t in bucket if t >= window * 60]
            _fallback_counts[scope_id].append(now)
            count = len(_fallback_counts[scope_id])
        return count <= int(limit_per_minute)


async def check_key_rate_limit(api_key_id: str, limit_per_minute: int | None) -> bool:
    """Per-key fixed-window counter. True = allowed, False = over limit."""
    if not limit_per_minute:
        return True
    return await _window_counter(str(api_key_id), int(limit_per_minute))


async def enforce_api_key_rate_limit(request: Request) -> JSONResponse | None:
    """Middleware helper: enforce per-key (and per-combo) rate limits.

    Returns a 429 response if the request is blocked.

    * Credentials with an ``api_keys`` row are limited by that row's
      ``rate_limit_per_minute``; if the key is scoped to a combo, a second,
      *shared* counter (``ratelimit:combo:{id}:{window}``) is applied so all
      keys on that combo consume one budget.
    * The admin key and dashboard JWTs no longer bypass this path: they get
      ``DEFAULT_V1_LIMIT_PER_MINUTE`` on ``/v1/*`` (their bypass still holds
      for admin routes, which never reach this middleware).
    """
    api_key_id = getattr(request.state, "api_key_id", None)
    if not api_key_id:
        return None

    identity = str(api_key_id)
    is_builtin = identity == "admin-key" or identity.startswith("jwt-")

    limit: int | None = None
    combo_id: str | None = None

    if not is_builtin:
        try:
            from app.database import async_session
            from sqlalchemy import text
            async with async_session() as db:
                row = (
                    await db.execute(
                        text(
                            "SELECT rate_limit_per_minute, combo_id "
                            "FROM api_keys WHERE id = :id AND is_active = true"
                        ),
                        {"id": identity},
                    )
                ).first()
            if row:
                limit = int(row[0]) if row[0] else None
                combo_id = row[1]
                # Monthly token budget per key (0 = unlimited): aggregate of
                # request_logs for this month; rejected before any upstream.
                try:
                    from app.database import async_session as _sess
                    async with _sess() as bdb:
                        bmonth = (await bdb.execute(text(
                            "SELECT COALESCE(SUM(tokens_used),0) FROM request_logs "
                            "WHERE api_key_id = :id AND tokens_used IS NOT NULL "
                            "AND timestamp >= date_trunc('month', now())"
                        ), {"id": identity})).scalar() or 0
                        budget = (await bdb.execute(text(
                            "SELECT monthly_token_budget FROM api_keys WHERE id = :id"
                        ), {"id": identity})).scalar() or 0
                    if budget and int(budget) > 0 and int(bmonth) >= int(budget):
                        return JSONResponse(
                            status_code=429,
                            content={"error": {"message": (
                                f"Monthly token budget exhausted for this key "
                                f"({int(bmonth)}/{int(budget)} tokens). "
                                f"Resets on the 1st."), "type": "rate_limit_error",
                                "code": "budget_exhausted"}},
                        )
                except Exception:
                    pass  # budget check must never block traffic
        except Exception:
            limit = None

    if limit is None:
        limit = DEFAULT_V1_LIMIT_PER_MINUTE if is_builtin else 60

    request.state.combo_id = combo_id

    if not await _window_counter(identity, limit):
        return _rate_limit_error(retry_after=max(1, 60 - int(time.time() % 60)))

    # Shared per-combo budget: several keys on one combo share `limit` RPM.
    if combo_id and not await _window_counter(f"combo:{combo_id}", limit):
        return _rate_limit_error(retry_after=max(1, 60 - int(time.time() % 60)))

    return None
