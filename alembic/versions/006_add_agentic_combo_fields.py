"""Add agentic fields to router_combos table.

Revision ID: 006_add_agentic_combo_fields
Revises: 005_add_router_combos
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "006_add_agentic_combo_fields"
down_revision = "005_add_router_combos"


def upgrade() -> None:
    op.add_column("router_combos", sa.Column("agentic_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("router_combos", sa.Column("max_iterations", sa.Integer(), nullable=False, server_default=sa.text("5")))
    op.add_column("router_combos", sa.Column("tool_calling", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    op.add_column("router_combos", sa.Column("deterministic", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    op.add_column("router_combos", sa.Column("loop_strategy", sa.String(20), nullable=False, server_default=sa.text("'sequential'")))
    op.add_column("router_combos", sa.Column("compression_target_tokens", sa.Integer(), nullable=False, server_default=sa.text("4000")))
    op.alter_column("router_combos", "agentic_enabled", server_default=None)
    op.alter_column("router_combos", "max_iterations", server_default=None)
    op.alter_column("router_combos", "tool_calling", server_default=None)
    op.alter_column("router_combos", "deterministic", server_default=None)
    op.alter_column("router_combos", "loop_strategy", server_default=None)
    op.alter_column("router_combos", "compression_target_tokens", server_default=None)


def downgrade() -> None:
    op.drop_column("router_combos", "agentic_enabled")
    op.drop_column("router_combos", "max_iterations")
    op.drop_column("router_combos", "tool_calling")
    op.drop_column("router_combos", "deterministic")
    op.drop_column("router_combos", "loop_strategy")
    op.drop_column("router_combos", "compression_target_tokens")
