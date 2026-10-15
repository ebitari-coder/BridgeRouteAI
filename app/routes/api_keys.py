from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.middleware.auth import verify_admin_key
from app.models import ApiKey, ApiKeyCreate, ApiKeyCreated, ApiKeyListResponse, StatusResponse
from app.crud import create_api_key, get_api_keys, revoke_api_key

router = APIRouter(prefix="/admin/api-keys", tags=["API Keys"])


def _to_api_key_schema(ak) -> ApiKey:
    mask = getattr(ak, "key_mask", None) or ""
    return ApiKey(
        id=ak.id,
        name=ak.name,
        key=mask,
        key_mask=mask,
        created_at=ak.created_at.isoformat(),
        last_used_at=ak.last_used_at.isoformat() if ak.last_used_at else None,
        is_active=ak.is_active,
        request_count=ak.request_count,
        combo_id=getattr(ak, "combo_id", None),
        rate_limit_per_minute=getattr(ak, "rate_limit_per_minute", None),
        monthly_token_budget=getattr(ak, "monthly_token_budget", 0),
    )


@router.get("", response_model=ApiKeyListResponse)
async def list_api_keys(
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys (masked — raw keys are never returned after creation)."""
    keys = await get_api_keys(db)
    return ApiKeyListResponse(data=[_to_api_key_schema(k) for k in keys])


@router.post("", response_model=ApiKeyCreated, status_code=201)
async def create_new_api_key(
    body: ApiKeyCreate,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key. `full_key` is disclosed exactly once in this response."""
    key, raw_key = await create_api_key(db, body.name)
    if body.combo_id is not None or body.rate_limit_per_minute is not None or body.monthly_token_budget is not None:
        if body.combo_id is not None:
            key.combo_id = body.combo_id
        if body.rate_limit_per_minute is not None:
            key.rate_limit_per_minute = body.rate_limit_per_minute
        if body.monthly_token_budget is not None:
            key.monthly_token_budget = body.monthly_token_budget
        await db.flush()
        await db.commit()
    return ApiKeyCreated(**_to_api_key_schema(key).model_dump(), full_key=raw_key)


@router.delete("/{key_id}", response_model=StatusResponse)
async def revoke_existing_api_key(
    key_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Revoke an API key."""
    success = await revoke_api_key(db, key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    return StatusResponse(status="ok", message=f"API key {key_id} revoked")
