"""Per-key token budgets + request replay capture.

Revision ID: 014_key_budgets_replay
Revises: 013_combo_load_spread

1. api_keys.monthly_token_budget — monthly token cap per API key
   (0 = unlimited), enforced like the combo budget: aggregate of
   request_logs.tokens_used for the key this month.
2. request_logs.request_payload / response_payload — redacted, truncated
   JSONB snapshots so failed requests can be replayed from the backoffice.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "014_key_budgets_replay"
down_revision = "013_combo_load_spread"


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column("monthly_token_budget", sa.Integer, nullable=False, server_default=sa.text("0")),
    )
    op.add_column("request_logs", sa.Column("request_payload", JSONB, nullable=True))
    op.add_column("request_logs", sa.Column("response_payload", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("request_logs", "response_payload")
    op.drop_column("request_logs", "request_payload")
    op.drop_column("api_keys", "monthly_token_budget")
