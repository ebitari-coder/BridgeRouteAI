"""Per-combo routing mode and group strategies.

Revision ID: 011_add_routing_mode
Revises: 010_api_key_hash_only

Routing is no longer selected by asking for a virtual ``model="agentic"`` id.
Every combo now declares how a request bound to it is executed:

  routing_mode      'agentic' — multi-iteration tool loop (the old behavior)
                    'group'   — models grouped under the combo name, one shot
  routing_strategy  group only: 'fallback' (ordered failover),
                    'round_robin' (rotate across requests),
                    'fusion' (parallel panel + judge, N+1 calls)
  judge_model       fusion only: 'provider/model_id' judge, NULL = models[0]

Backfill maps the legacy boolean so existing agentic combos keep their engine
and everything else becomes an ordered fallback group. agentic_enabled is kept
in sync on write so older clients/backoffice builds keep working.
"""
from alembic import op
import sqlalchemy as sa

revision = "011_add_routing_mode"
down_revision = "010_api_key_hash_only"


def upgrade() -> None:
    op.add_column(
        "router_combos",
        sa.Column("routing_mode", sa.String(20), nullable=False, server_default=sa.text("'group'")),
    )
    op.add_column(
        "router_combos",
        sa.Column("routing_strategy", sa.String(20), nullable=False, server_default=sa.text("'fallback'")),
    )
    op.add_column(
        "router_combos",
        sa.Column("judge_model", sa.String(255), nullable=True),
    )

    # Legacy agentic combos keep the agentic engine; everything else becomes a
    # plain ordered fallback group (the behavior they had before this change).
    op.execute(
        """
        UPDATE router_combos
        SET routing_mode = CASE WHEN agentic_enabled THEN 'agentic' ELSE 'group' END
        """
    )


def downgrade() -> None:
    op.drop_column("router_combos", "judge_model")
    op.drop_column("router_combos", "routing_strategy")
    op.drop_column("router_combos", "routing_mode")
