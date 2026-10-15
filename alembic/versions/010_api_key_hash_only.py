"""API keys hash-only storage, rate-limit server default, token budget uniqueness.

Revision ID: 010_api_key_hash_only
Revises: 009_add_budget_alerts

S1  - api_keys.key becomes nullable and is cleared after backfilling key_hash and a
      display mask, so plaintext credentials are no longer recoverable from the DB.
C6  - restores a server default on rate_limit_per_minute (removed by 008) so raw
      INSERTs such as app/seed.py cannot hit a NOT NULL violation on a fresh database.
C17 - collapses duplicate (combo_id, month_year) rows in token_budgets (summing usage)
      and enforces uniqueness so concurrent budgets cannot race into two rows.
"""
from alembic import op
import sqlalchemy as sa

revision = "010_api_key_hash_only"
down_revision = "009_add_budget_alerts"


def upgrade() -> None:
    # ── C6: give rate_limit_per_minute a persistent default ───────────────────
    op.alter_column(
        "api_keys",
        "rate_limit_per_minute",
        existing_type=sa.Integer(),
        existing_nullable=False,
        server_default=sa.text("60"),
    )

    # ── S1: add display mask, backfill, then drop plaintext ───────────────────
    op.add_column(
        "api_keys",
        sa.Column("key_mask", sa.String(64), nullable=False, server_default=sa.text("''")),
    )

    # Backfill hash for any row that predates 003 or lost its hash.
    op.execute(
        """
        UPDATE api_keys
        SET key_hash = encode(sha256(convert_to(key, 'UTF8')), 'hex')
        WHERE key IS NOT NULL AND key_hash IS NULL
        """
    )
    # Relax NOT NULL first — the clearing step below depends on it.
    op.alter_column(
        "api_keys",
        "key",
        existing_type=sa.String(255),
        existing_unique=True,
        existing_index=True,
        nullable=True,
    )
    # Record a non-reversible display mask (prefix + fixed asterisks), then clear
    # the plaintext. The raw key itself cannot be reconstructed afterwards; copies
    # held by clients keep working because authentication is hash-based.
    op.execute(
        """
        UPDATE api_keys
        SET key_mask = left(key, 11) || '******',
            key = NULL
        WHERE key IS NOT NULL
        """
    )

    # ── C17: one budget row per (combo_id, month_year) ────────────────────────
    # 1. Fold duplicates into the row we intend to keep (newest updated_at).
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY combo_id, month_year
                       ORDER BY updated_at DESC, created_at DESC, id DESC
                   ) AS rn,
                   SUM(tokens_used)  OVER (PARTITION BY combo_id, month_year) AS sum_used,
                   MAX(tokens_budget) OVER (PARTITION BY combo_id, month_year) AS max_budget
            FROM token_budgets
        )
        UPDATE token_budgets tb
        SET tokens_used = ranked.sum_used,
            tokens_budget = ranked.max_budget,
            updated_at = NOW()
        FROM ranked
        WHERE tb.id = ranked.id AND ranked.rn = 1
        """
    )
    # 2. Drop the losers. There are no foreign keys referencing token_budgets.
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY combo_id, month_year
                       ORDER BY updated_at DESC, created_at DESC, id DESC
                   ) AS rn
            FROM token_budgets
        )
        DELETE FROM token_budgets tb
        USING ranked
        WHERE tb.id = ranked.id AND ranked.rn > 1
        """
    )
    # 3. Replace the non-unique lookup index with a unique one (same name so the
    #    ORM Index("ix_token_budgets_combo_month", ..., unique=True) matches).
    op.drop_index("ix_token_budgets_combo_month", table_name="token_budgets")
    op.create_index(
        "ix_token_budgets_combo_month",
        "token_budgets",
        ["combo_id", "month_year"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_token_budgets_combo_month", table_name="token_budgets")
    op.create_index("ix_token_budgets_combo_month", "token_budgets", ["combo_id", "month_year"])

    # Plaintext keys cannot be recovered: the mask is substituted back so the
    # column can be made NOT NULL again. Clients must rotate their keys.
    op.execute(
        """
        UPDATE api_keys
        SET key = COALESCE(key, key_mask, '')
        WHERE key IS NULL
        """
    )
    op.alter_column(
        "api_keys",
        "key",
        existing_type=sa.String(255),
        existing_unique=True,
        existing_index=True,
        nullable=False,
    )
    op.drop_column("api_keys", "key_mask")
    op.alter_column(
        "api_keys",
        "rate_limit_per_minute",
        existing_type=sa.Integer(),
        existing_nullable=False,
        server_default=None,
    )
