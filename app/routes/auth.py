"""Authentication routes — email + password login, JWT tokens."""
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import jwt
import bcrypt
from app.config import settings
from app.database import async_session
from app.middleware.rate_limit import limiter
from sqlalchemy import text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])

# In-memory failed login tracker: {email: [(timestamp, ...), locked_until]}
_failed_logins: dict[str, list[float]] = {}
_LOCKOUT_THRESHOLD = 5
_LOCKOUT_DURATION = 900  # 15 minutes


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict
    must_change_password: bool = False


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=12)

    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password complexity."""
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain an uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain a lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain a digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain a special character")
        return v


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_token(user_id: str, email: str, role: str, must_change: bool = False) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "must_change_password": must_change,
        "exp": datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expire_hours),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def _is_locked_out(email: str) -> bool:
    """Check if account is locked out from too many failed attempts."""
    attempts = _failed_logins.get(email, [])
    # Remove expired attempts
    cutoff = time.time() - _LOCKOUT_DURATION
    _failed_logins[email] = [t for t in attempts if t > cutoff]
    return len(_failed_logins[email]) >= _LOCKOUT_THRESHOLD


def _record_failed_attempt(email: str) -> None:
    """Record a failed login attempt."""
    if email not in _failed_logins:
        _failed_logins[email] = []
    _failed_logins[email].append(time.time())
    logger.warning("Failed login attempt for %s (attempts: %d)", email, len(_failed_logins[email]))


def _clear_failed_attempts(email: str) -> None:
    """Clear failed attempts on successful login."""
    _failed_logins.pop(email, None)


def _extract_user_from_request(request: Request) -> dict:
    """Extract and verify user from JWT token in Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = decode_token(auth_header[7:])
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return payload


@router.post("/login", response_model=LoginResponse)
@limiter.limit("5/minute")
async def login(request: Request, body: LoginRequest):
    """Authenticate with email and password. Returns a JWT token."""
    # Check lockout
    if _is_locked_out(body.email):
        logger.warning("Locked out login attempt for %s from %s", body.email, request.client.host if request.client else "unknown")
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.")

    async with async_session() as db:
        result = await db.execute(
            text(
                "SELECT id, email, name, password_hash, role, is_active, password_changed FROM users WHERE email = :email LIMIT 1"
            ),
            {"email": body.email},
        )
        user = result.mappings().first()

        # Uniform error message — same for wrong email and wrong password (#28)
        if not user or not user["password_hash"] or not verify_password(body.password, user["password_hash"]):
            _record_failed_attempt(body.email)
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if not user["is_active"]:
            raise HTTPException(status_code=403, detail="Account is deactivated")

        # Clear failed attempts on success
        _clear_failed_attempts(body.email)

        # Check if password was ever changed (#20)
        must_change = not user.get("password_changed", False)

        # Update last_login_at
        await db.execute(
            text("UPDATE users SET last_login_at = :now WHERE id = :id"),
            {"now": datetime.now(timezone.utc), "id": user["id"]},
        )
        await db.commit()

        token = create_token(user["id"], user["email"], user["role"], must_change)

        return LoginResponse(
            access_token=token,
            must_change_password=must_change,
            user={
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "role": user["role"],
            },
        )


@router.get("/me")
async def get_current_user(request: Request):
    """Get the currently authenticated user from JWT token."""
    payload = _extract_user_from_request(request)

    async with async_session() as db:
        result = await db.execute(
            text("SELECT id, email, name, role, is_active, password_changed FROM users WHERE id = :id LIMIT 1"),
            {"id": payload["sub"]},
        )
        user = result.mappings().first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "is_active": user["is_active"],
            "must_change_password": not user.get("password_changed", False),
        }


@router.post("/change-password")
async def change_password(body: PasswordChangeRequest, request: Request):
    """Change password for the authenticated user."""
    # Validate password strength (#6)
    PasswordChangeRequest.validate_password_strength(body.new_password)

    payload = _extract_user_from_request(request)
    user_id = payload["sub"]

    async with async_session() as db:
        result = await db.execute(
            text("SELECT id, password_hash FROM users WHERE id = :id LIMIT 1"),
            {"id": user_id},
        )
        user = result.mappings().first()

        if not user or not user["password_hash"]:
            raise HTTPException(status_code=404, detail="User not found")

        if not verify_password(body.current_password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect")

        new_hash = hash_password(body.new_password)
        await db.execute(
            text("UPDATE users SET password_hash = :hash, password_changed = true WHERE id = :id"),
            {"hash": new_hash, "id": user_id},
        )
        await db.commit()

    return {"status": "ok", "message": "Password updated"}
