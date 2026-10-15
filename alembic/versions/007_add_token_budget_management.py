"""Add monthly_token_budget to router_combos and create token_budgets table.

Revision ID: 007_add_token_budget_management
Revises: 006_add_agentic_combo_fields
"""
from alembic import op
import sqlalchemy as sa

revision = "007_add_token_budget_management"
down_revision = "006_add_agentic_combo_fields"


def upgrade() -> None:
    op.add_column("router_combos", sa.Column("monthly_token_budget", sa.Integer(), nullable=False, server_default=sa.text("0")))
    op.create_table(
        "token_budgets",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("combo_id", sa.String(32), nullable=False, index=True),
        sa.Column("month_year", sa.String(7), nullable=False, index=True),
        sa.Column("tokens_used", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("tokens_budget", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_token_budgets_combo_month", "token_budgets", ["combo_id", "month_year"])
    op.create_index("ix_token_budgets_month", "token_budgets", ["month_year"])


def downgrade() -> None:
    op.drop_index("ix_token_budgets_combo_month", "token_budgets")
    op.drop_index("ix_token_budgets_month", "token_budgets")
    op.drop_table("token_budgets")
    op.drop_column("router_combos", "monthly_token_budget")
