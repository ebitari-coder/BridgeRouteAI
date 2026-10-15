from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.middleware.auth import verify_admin_key
from app.models import SystemSettings, StatusResponse
from app.crud import get_settings, update_settings

router = APIRouter(prefix="/admin/settings", tags=["Settings"])


@router.get("", response_model=SystemSettings)
async def get_system_settings(
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get current system settings."""
    data = await get_settings(db)
    return SystemSettings(**data)


@router.put("", response_model=SystemSettings)
async def update_system_settings(
    body: SystemSettings,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Update system settings."""
    updates = body.model_dump()
    data = await update_settings(db, updates)
    return SystemSettings(**data)
