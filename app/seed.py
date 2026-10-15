"""Database seeding — runs on startup after migrations.

Seeds the superadmin user with email/password and API key.
"""
import logging
from datetime import datetime, timezone
from sqlalchemy import text
import bcrypt
from app.config import settings
from app.database import async_session
from app.crud import hash_api_key, mask_api_key

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


async def seed_superadmin():
    """Seed superadmin user and API key if they don't exist."""
    async with async_session() as db:
        # Check if superadmin user exists
        result = await db.execute(
            text("SELECT id FROM users WHERE email = :email LIMIT 1"),
            {"email": settings.superadmin_email},
        )
        user = result.mappings().first()

        if not user:
            user_id = "user-superadmin"
            password_hash = hash_password(settings.superadmin_password)
            await db.execute(
                text(
                    """INSERT INTO users (id, email, name, password_hash, role, is_active, created_at)
                       VALUES (:id, :email, :name, :pw_hash, 'admin', true, :now)
                       ON CONFLICT (id) DO NOTHING"""
                ),
                {
                    "id": user_id,
                    "email": settings.superadmin_email,
                    "name": "Super Admin",
                    "pw_hash": password_hash,
                    "now": datetime.now(timezone.utc),
                },
            )
            logger.info("Seeded superadmin user: %s", settings.superadmin_email)
        else:
            user_id = user["id"]
            logger.info("Superadmin user already exists: %s", user_id)

        # Check if superadmin API key exists (hash lookup — plaintext is not stored)
        admin_key_hash = hash_api_key(settings.admin_api_key)
        result = await db.execute(
            text("SELECT id FROM api_keys WHERE key_hash = :key_hash LIMIT 1"),
            {"key_hash": admin_key_hash},
        )
        key_row = result.mappings().first()

        if not key_row:
            key_id = "key-superadmin"
            await db.execute(
                text(
                    """INSERT INTO api_keys (id, name, key, key_hash, key_mask, is_active,
                                             rate_limit_per_minute, created_at)
                       VALUES (:id, :name, NULL, :key_hash, :key_mask, true, 60, :now)
                       ON CONFLICT (id) DO NOTHING"""
                ),
                {
                    "id": key_id,
                    "name": "Super Admin Key",
                    "key_hash": admin_key_hash,
                    "key_mask": mask_api_key(settings.admin_api_key),
                    "now": datetime.now(timezone.utc),
                },
            )
            logger.info("Seeded superadmin API key")
        else:
            logger.info("Superadmin API key already exists: %s", key_row["id"])

        # Store superadmin info in settings table
        await db.execute(
            text(
                """INSERT INTO settings (key, value) VALUES ('superadmin.email', :val)
                   ON CONFLICT (key) DO UPDATE SET value = :val"""
            ),
            {"val": settings.superadmin_email},
        )
        await db.execute(
            text(
                """INSERT INTO settings (key, value) VALUES ('superadmin.user_id', :val)
                   ON CONFLICT (key) DO UPDATE SET value = :val"""
            ),
            {"val": user_id},
        )

        await db.commit()
        logger.info("Superadmin seeding complete")
