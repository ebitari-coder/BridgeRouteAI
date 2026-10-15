"""Add budget_alerts table.

Revision ID: 009_add_budget_alerts
Revises: 008_add_agentic_advanced_schema
"""
from alembic import op
import sqlalchemy as sa

revision = "009_add_budget_alerts"
down_revision = "008_add_agentic_advanced_schema"


def upgrade() -> None:
    op.create_table(
        "budget_alerts",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("combo_id", sa.String(32), nullable=False, index=True),
        sa.Column("month_year", sa.String(7), nullable=False, index=True),
        sa.Column("level", sa.String(20), nullable=False),
        sa.Column("percentage", sa.Integer(), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=False),
        sa.Column("tokens_budget", sa.Integer(), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_budget_alerts_combo_month_level",
        "budget_alerts",
        ["combo_id", "month_year", "level"],
    )


def downgrade() -> None:
    op.drop_index("ix_budget_alerts_combo_month_level", "budget_alerts")
    op.drop_table("budget_alerts")
