import time
import json
import logging
import starlette.responses
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from app.database import async_session
from app.crud import log_request

logger = logging.getLogger(__name__)

EXCLUDE_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

# JSON endpoints whose body carries the requested model. The body must be read
# *before* call_next: once the response has started streaming, request.body()
# raises "Stream consumed" and the field would silently stay empty.
BODY_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/embeddings"}


def _redact_and_truncate(data, max_total_bytes: int = 16_000, max_str: int = 600):
    """Redact secrets and bound size before persisting a payload snapshot."""
    if not isinstance(data, dict):
        return None
    cleaned = json.loads(json.dumps(data, default=str))  # deep copy
    cleaned.pop("api_key", None)
    msgs = cleaned.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and isinstance(m.get("content"), str) and len(m["content"]) > max_str:
                m["content"] = m["content"][:max_str] + f"...[truncated {len(m['content'])} chars]"
    out = json.dumps(cleaned, ensure_ascii=False, default=str)
    if len(out.encode()) > max_total_bytes:
        out = out[:max_total_bytes]
    try:
        return json.loads(out)
    except Exception:
        return None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every API request to Postgres for dashboard analytics."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in EXCLUDE_PATHS or path.startswith("/docs"):
            return await call_next(request)

        model = ""
        if request.method in ("POST", "PUT", "PATCH") and path in BODY_PATHS:
            try:
                raw = await request.body()
                data = json.loads(raw) if raw else None
                if isinstance(data, dict):
                    model = data.get("model") or ""
                    # Stash for downstream use (Starlette shares scope["state"]).
                    request.state.request_body = data
            except Exception:
                pass
            request.state.request_model = model

        start = time.time()
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)

        try:
            async with async_session() as db:
                # Replay snapshots: request payload always (bounded), response
                # payload only for failures — that's what replay is for.
                request_payload = _redact_and_truncate(
                    getattr(request.state, "request_body", None)
                ) if path in BODY_PATHS else None
                response_payload = None
                if response.status_code >= 400:
                    try:
                        raw_resp = b""
                        async for chunk in response.body_iterator:
                            raw_resp += chunk
                        response = starlette.responses.Response(
                            content=raw_resp, status_code=response.status_code,
                            media_type=response.media_type or "application/json",
                        )
                        response_payload = raw_resp.decode(errors="replace")[:16_000]
                    except Exception:
                        pass
                await log_request(
                    db,
                    method=request.method,
                    path=path,
                    status_code=response.status_code,
                    model=model,
                    duration_ms=duration_ms,
                    user_agent=request.headers.get("user-agent", ""),
                    ip_address=request.client.host if request.client else "",
                    api_key_id=getattr(request.state, "api_key_id", None),
                    tokens_used=getattr(request.state, "tokens_used", None),
                    request_payload=request_payload,
                    response_payload=response_payload,
                    combo_history=getattr(request.state, "combo_history", None),
                    fallback_chain=getattr(request.state, "fallback_chain", None),
                )
                await db.commit()
        except Exception as e:
            logger.warning("Failed to log request: %s", e)

        return response
