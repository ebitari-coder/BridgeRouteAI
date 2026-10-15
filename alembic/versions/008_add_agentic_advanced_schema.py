"""Add agentic advanced schema: tools_allowed, strategy_preset, combo_id/rate_limit to api_keys, agentic_sessions and agentic_analytics tables.

Revision ID: 008_add_agentic_advanced_schema
Revises: 007_add_token_budget_management
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "008_add_agentic_advanced_schema"
down_revision = "007_add_token_budget_management"


def upgrade() -> None:
    op.add_column("router_combos", sa.Column("tools_allowed", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")))
    op.add_column("router_combos", sa.Column("strategy_preset", sa.String(length=50), nullable=True))
    op.alter_column("router_combos", "tools_allowed", server_default=None)

    op.add_column("api_keys", sa.Column("combo_id", sa.String(length=32), nullable=True))
    op.add_column("api_keys", sa.Column("rate_limit_per_minute", sa.Integer(), nullable=False, server_default=sa.text("60")))
    op.create_index("ix_api_keys_combo_id", "api_keys", ["combo_id"])
    op.alter_column("api_keys", "rate_limit_per_minute", server_default=None)

    op.create_table(
        "agentic_sessions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("session_id", sa.String(64), nullable=False, unique=True),
        sa.Column("combo_id", sa.String(32), nullable=False),
        sa.Column("messages", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("iteration_index", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("step_history", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_agentic_sessions_session_id", "agentic_sessions", ["session_id"], unique=True)
    op.create_index("ix_agentic_sessions_combo_id", "agentic_sessions", ["combo_id"])

    op.create_table(
        "agentic_analytics",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("combo_id", sa.String(32), nullable=False),
        sa.Column("request_id", sa.String(32), nullable=True),
        sa.Column("month_year", sa.String(7), nullable=False),
        sa.Column("requests_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_iterations", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("stop_reason_counts", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("tool_call_counts", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("model_stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_agentic_analytics_combo_id", "agentic_analytics", ["combo_id"])
    op.create_index("ix_agentic_analytics_month_year", "agentic_analytics", ["month_year"])
    op.create_index("ix_agentic_analytics_combo_month", "agentic_analytics", ["combo_id", "month_year"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_agentic_analytics_combo_month", "agentic_analytics")
    op.drop_index("ix_agentic_analytics_month_year", "agentic_analytics")
    op.drop_index("ix_agentic_analytics_combo_id", "agentic_analytics")
    op.drop_table("agentic_analytics")

    op.drop_index("ix_agentic_sessions_combo_id", "agentic_sessions")
    op.drop_index("ix_agentic_sessions_session_id", "agentic_sessions")
    op.drop_table("agentic_sessions")

    op.drop_index("ix_api_keys_combo_id", "api_keys")
    op.drop_column("api_keys", "rate_limit_per_minute")
    op.drop_column("api_keys", "combo_id")

    op.drop_column("router_combos", "strategy_preset")
    op.drop_column("router_combos", "tools_allowed")
