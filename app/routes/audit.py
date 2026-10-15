from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.middleware.auth import verify_admin_key
from app.models import AuditLog, AuditLogListResponse
from app.crud import get_audit_logs

router = APIRouter(prefix="/admin/audit", tags=["Audit Log"])


@router.get("", response_model=AuditLogListResponse)
async def list_audit_logs(
    limit: int = 100,
    offset: int = 0,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get audit logs with pagination."""
    logs, total = await get_audit_logs(db, limit, offset)
    return AuditLogListResponse(
        data=[
            AuditLog(
                id=log.id,
                timestamp=log.timestamp.isoformat(),
                action=log.action,
                actor=log.actor,
                target=log.target,
                details=log.details,
                ip_address=log.ip_address,
            )
            for log in logs
        ],
        total=total,
    )
