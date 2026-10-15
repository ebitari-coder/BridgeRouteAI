"""Add key_hash column to api_keys table.

Revision ID: 003_add_api_key_hash
Revises: 002
Create Date: 2026-09-12
"""
from alembic import op
import sqlalchemy as sa

revision = "003_add_api_key_hash"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add key_hash column
    op.add_column(
        "api_keys",
        sa.Column("key_hash", sa.String(64), nullable=True, index=True, unique=True),
    )

    # Backfill existing keys with SHA-256 hashes
    op.execute(
        """
        UPDATE api_keys
        SET key_hash = encode(sha256(key::bytea), 'hex')
        WHERE key_hash IS NULL
        """
    )

    # Make key_hash NOT NULL after backfill
    op.alter_column("api_keys", "key_hash", nullable=False)

    # Add unique constraint
    op.create_unique_constraint("uq_api_keys_key_hash", "api_keys", ["key_hash"])


def downgrade() -> None:
    op.drop_constraint("uq_api_keys_key_hash", "api_keys", type_="unique")
    op.drop_column("api_keys", "key_hash")
