"""Add router_combos table for admin-curated model chains.

Revision ID: 005_add_router_combos
Revises: 004_add_password_changed
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "005_add_router_combos"
down_revision = "004_add_password_changed"


def upgrade() -> None:
    op.create_table(
        "router_combos",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("router_type", sa.String(50), nullable=False, index=True),
        sa.Column("models", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_router_combos_type_active", "router_combos", ["router_type", "is_active"])


def downgrade() -> None:
    op.drop_index("ix_router_combos_type_active")
    op.drop_table("router_combos")
