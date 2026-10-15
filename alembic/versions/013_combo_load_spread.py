"""Per-combo load_spread flag for the agentic sequential loop.

Revision ID: 013_combo_load_spread
Revises: 012_combo_integrity

The momentum lock holds one model for a whole loop run (reasoning
continuity). Admins who prefer the legacy round-robin — spreading load
across the combo's providers per iteration at the cost of mid-loop model
switches — set load_spread=true on the combo.
"""
from alembic import op
import sqlalchemy as sa

revision = "013_combo_load_spread"
down_revision = "012_combo_integrity"


def upgrade() -> None:
    op.add_column(
        "router_combos",
        sa.Column("load_spread", sa.Boolean, nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    op.drop_column("router_combos", "load_spread")
