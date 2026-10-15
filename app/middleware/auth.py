"""API key authentication middleware.

Enforces API key verification for all /v1/* endpoints.
Supports both Authorization: Bearer <key> and x-api-key header.

Admin auth checks both the env var AND the database superadmin key.
"""
import logging
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def _is_admin_key(key: str) -> bool:
    """Check if the given key is a valid admin key (env var or admin JWT only).

    Regular database API keys are user keys and must NOT grant admin access.
    """
    # 1. Check env var (fast path)
    if key == settings.admin_api_key:
        return True

    # 2. Check if it's a JWT token with admin role
    try:
        from app.routes.auth import decode_token
        payload = decode_token(key)
        if payload and payload.get("role") == "admin":
            return True
    except Exception:
        pass

    return False


async def verify_admin_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """Verify the admin API key from the Authorization header."""
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Use: Bearer <api-key>",
        )

    if not await _is_admin_key(credentials.credentials):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return credentials.credentials


async def _verify_user_api_key(request: Request) -> str | JSONResponse:
    """Extract and verify an API key from the request against the database.

    Returns the API key ID (str) if valid, None if no key needed,
    or a JSONResponse with OpenAI-format error if key is invalid.
    """
    # Extract key from Authorization header or x-api-key header
    api_key = None

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:].strip()
    elif auth_header:
        api_key = auth_header.strip()

    if not api_key:
        api_key = request.headers.get("x-api-key", "").strip()

    # Skip auth for endpoints that don't need it
    path = request.url.path
    if path in ("/", "/health", "/docs", "/openapi.json", "/redoc"):
        return None
    if path.startswith("/admin"):
        return None  # Admin routes use their own auth
    if path.startswith("/auth"):
        return None  # Auth routes use their own auth

    # If no key provided, reject
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Missing API key. Provide via Authorization: Bearer <key> or x-api-key header.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )

    # Check against admin key (shortcut — always valid)
    if api_key == settings.admin_api_key:
        return "admin-key"

    # Check if it's a valid JWT token (admin dashboard users)
    try:
        from app.routes.auth import decode_token
        payload = decode_token(api_key)
        if payload and payload.get("role") in ("admin", "user"):
            return f"jwt-{payload.get('sub', 'unknown')}"
    except Exception:
        pass

    # Check against database API keys
    try:
        from app.database import async_session
        from app.crud import verify_api_key

        async with async_session() as db:
            row = await verify_api_key(db, api_key)

            if not row:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "Invalid API key.",
                            "type": "authentication_error",
                            "param": None,
                            "code": "invalid_api_key",
                        }
                    },
                )

            # Update last_used_at
            from datetime import datetime, timezone
            from sqlalchemy import text
            await db.execute(
                text(
                    "UPDATE api_keys SET last_used_at = :now, request_count = request_count + 1 WHERE id = :id"
                ),
                {"now": datetime.now(timezone.utc), "id": row.id},
            )
            await db.commit()

            return str(row.id)

    except Exception as exc:
        logger.warning("API key DB check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "API key verification unavailable. Try again later.",
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
        )


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces API key verification on /v1/* endpoints
    and per-key rate limits."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Only enforce on /v1/* endpoints
        if not path.startswith("/v1/"):
            return await call_next(request)

        # Skip auth for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        result = await _verify_user_api_key(request)

        if isinstance(result, JSONResponse):
            return result

        # result is the api_key_id str (or None)
        if result is not None:
            request.state.api_key_id = result

            # Per-key rate limiting (uses rate_limit_per_minute from api_keys table)
            from app.middleware.rate_limit import enforce_api_key_rate_limit
            blocked = await enforce_api_key_rate_limit(request)
            if blocked is not None:
                return blocked

        return await call_next(request)
