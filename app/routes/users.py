from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.middleware.auth import verify_admin_key
from app.models import (
    User, UserCreate, UserUpdate, UserListResponse, StatusResponse,
)
from app.crud import create_user, get_users, get_user, update_user, delete_user

router = APIRouter(prefix="/admin/users", tags=["Users"])


def _to_user_schema(u) -> User:
    return User(
        id=u.id,
        email=u.email,
        name=u.name,
        role=u.role,
        created_at=u.created_at.isoformat(),
        last_login_at=u.last_login_at.isoformat() if u.last_login_at else None,
        is_active=u.is_active,
        api_key_count=u.api_key_count,
        request_count=u.request_count,
    )


@router.get("", response_model=UserListResponse)
async def list_users(
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """List all users."""
    users = await get_users(db)
    return UserListResponse(data=[_to_user_schema(u) for u in users])


@router.post("", response_model=User, status_code=201)
async def create_new_user(
    body: UserCreate,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Create a new user."""
    user = await create_user(db, body.email, body.name, body.role)
    return _to_user_schema(user)


@router.get("/{user_id}", response_model=User)
async def get_single_user(
    user_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get a single user by ID."""
    user = await get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_user_schema(user)


@router.patch("/{user_id}", response_model=User)
async def update_existing_user(
    user_id: str,
    body: UserUpdate,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Update a user."""
    updates = body.model_dump(exclude_unset=True)
    user = await update_user(db, user_id, updates)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_user_schema(user)


@router.delete("/{user_id}", response_model=StatusResponse)
async def delete_existing_user(
    user_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Delete a user."""
    success = await delete_user(db, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return StatusResponse(status="ok", message=f"User {user_id} deleted")
