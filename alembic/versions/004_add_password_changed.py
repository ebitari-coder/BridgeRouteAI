"""Add password_changed column to users table.

Revision ID: 004_add_password_changed
Revises: 003_add_api_key_hash
Create Date: 2026-09-12
"""
from alembic import op
import sqlalchemy as sa

revision = "004_add_password_changed"
down_revision = "003_add_api_key_hash"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("password_changed", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("users", "password_changed")
