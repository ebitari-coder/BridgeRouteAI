"""Combo referential integrity, uniqueness, and activation invariants.

Revision ID: 012_combo_integrity
Revises: 011_add_routing_mode

Closes four structural gaps in the combo implementation:

1. Foreign keys — api_keys.combo_id, agentic_sessions.combo_id,
   token_budgets.combo_id, budget_alerts.combo_id and
   agentic_analytics.combo_id were bare strings. A deleted combo silently
   orphaned all of them. Keys are audited configuration (ON DELETE SET NULL);
   the session/budget/alert/analytics tables are dependents (ON DELETE
   CASCADE). Existing orphan rows are removed up front so the constraints
   can be created.

2. Unique combo name — the name is a routable model id
   (providers.resolve_combo_ref exact-matches it first); duplicates made that
   resolution order-dependent. Existing duplicates get a "-2"/"-3" suffix
   before the constraint is added.

3. Single active combo per router_type — activate_router_combo() deactivates
   peers in one UPDATE, but two concurrent activations could both commit.
   A partial unique index on (router_type) WHERE is_active enforces the
   invariant at the storage layer.

4. Optimistic concurrency — a monotonically incremented version column so
   concurrent edits are detectable (If-Match / expected_version) instead of
   last-write-wins.
"""
from alembic import op
import sqlalchemy as sa

revision = "012_combo_integrity"
down_revision = "011_add_routing_mode"

_FK_TABLES = [
    # (table, combo_id nullable -> SET NULL, else CASCADE)
    ("api_keys", True),
    ("agentic_sessions", False),
    ("token_budgets", False),
    ("budget_alerts", False),
    ("agentic_analytics", False),
]


def upgrade() -> None:
    conn = op.get_bind()

    # ── 2a. De-duplicate combo names (keep earliest, suffix the rest) ──────
    dupes = conn.execute(sa.text(
        "SELECT name FROM router_combos GROUP BY name HAVING count(*) > 1"
    )).scalars().all()
    for name in dupes:
        rows = conn.execute(sa.text(
            "SELECT id FROM router_combos WHERE name = :n ORDER BY created_at ASC"
        ), {"n": name}).scalars().all()
        for i, combo_id in enumerate(rows[1:], start=2):
            conn.execute(sa.text(
                "UPDATE router_combos SET name = :nn WHERE id = :id"
            ), {"nn": f"{name}-{i}", "id": combo_id})

    op.create_unique_constraint("uq_router_combos_name", "router_combos", ["name"])

    # ── 3. One active combo per router_type (storage-level invariant) ──────
    op.create_index(
        "uq_router_combos_one_active_per_type",
        "router_combos",
        ["router_type"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )

    # ── 4. Version column for optimistic concurrency ───────────────────────
    op.add_column(
        "router_combos",
        sa.Column("version", sa.Integer, nullable=False, server_default=sa.text("1")),
    )

    # ── 1. Foreign keys (clean orphans first, then constrain) ─────────────
    for table, nullable in _FK_TABLES:
        if nullable:
            # Scoped keys become unscoped when their combo disappears.
            conn.execute(sa.text(
                f"UPDATE {table} SET combo_id = NULL "
                "WHERE combo_id IS NOT NULL AND combo_id NOT IN "
                "(SELECT id FROM router_combos)"
            ))
            on_delete = "SET NULL"
        else:
            # Dependents of a deleted combo have no meaning on their own.
            conn.execute(sa.text(
                f"DELETE FROM {table} "
                "WHERE combo_id NOT IN (SELECT id FROM router_combos)"
            ))
            on_delete = "CASCADE"
        op.create_foreign_key(
            f"fk_{table}_combo_id_router_combos",
            table,
            "router_combos",
            ["combo_id"],
            ["id"],
            ondelete=on_delete,
        )


def downgrade() -> None:
    for table, _nullable in _FK_TABLES:
        op.drop_constraint(f"fk_{table}_combo_id_router_combos", table, type_="foreignkey")
    op.drop_column("router_combos", "version")
    op.drop_index("uq_router_combos_one_active_per_type", table_name="router_combos")
    op.drop_constraint("uq_router_combos_name", "router_combos", type_="unique")
