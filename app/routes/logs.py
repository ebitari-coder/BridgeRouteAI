import httpx

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db, async_session
from app.middleware.auth import verify_admin_key
from app.models import RequestLog, RequestLogListResponse
from app.crud import get_request_logs
from app.db_models import DBRequestLog
from app.config import settings

router = APIRouter(prefix="/admin/logs", tags=["Request Logs"])


@router.get("", response_model=RequestLogListResponse)
async def list_request_logs(
    limit: int = 100,
    offset: int = 0,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get request logs with pagination."""
    logs, total = await get_request_logs(db, limit, offset)
    return RequestLogListResponse(
        data=[
            RequestLog(
                id=log.id,
                timestamp=log.timestamp.isoformat(),
                method=log.method,
                path=log.path,
                status_code=log.status_code,
                model=log.model,
                duration_ms=log.duration_ms,
                user_agent=log.user_agent,
                ip_address=log.ip_address,
                api_key_id=log.api_key_id,
                tokens_used=log.tokens_used,
            )
            for log in logs
        ],
        total=total,
    )


@router.post("/{log_id}/replay")
async def replay_request(
    log_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Re-fire a captured request through the gateway.

    Works for /v1/chat/completions (and other BODY_PATHS) log entries whose
    request_payload snapshot survived redaction. The replay is issued with
    the admin credential and every upstream/provider guards apply as normal.
    """
    result = await db.execute(select(DBRequestLog).where(DBRequestLog.id == log_id))
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Log entry not found")
    if not row.request_payload:
        raise HTTPException(422, "No request payload captured for this log entry")

    internal_base = "http://127.0.0.1:8000"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{internal_base}{row.path}",
                json=row.request_payload,
                headers={
                    "Authorization": f"Bearer {settings.admin_api_key}",
                    "Content-Type": "application/json",
                },
            )
        return {
            "replay_of": log_id,
            "original_status": row.status_code,
            "status": resp.status_code,
            "response": resp.json() if "json" in (resp.headers.get("content-type") or "") else resp.text[:2000],
        }
    except Exception as exc:
        raise HTTPException(502, f"Replay failed: {exc}")
