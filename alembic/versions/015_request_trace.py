"""Agentic trace + fallback chain on request_logs.

Revision ID: 015_request_trace
Revises: 014_key_budgets_replay

Adds two JSONB columns so the backoffice logs page can show:
  combo_history   — full agentic loop step-by-step (which model did what at
                    which iteration, tool calls, tokens, stop_reason)
  fallback_chain  — which providers were tried on a request, in order, with
                    skip/error/fallback markers, for routing diagnostics
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "015_request_trace"
down_revision = "014_key_budgets_replay"


def upgrade() -> None:
    op.add_column("request_logs", sa.Column("combo_history", JSONB, nullable=True))
    op.add_column("request_logs", sa.Column("fallback_chain", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("request_logs", "fallback_chain")
    op.drop_column("request_logs", "combo_history")
