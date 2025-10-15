"""Initial migration anchor (no-op).

Revision ID: 0001_initial
Revises:
Create Date: 2025-09-17 00:00:00.000000
"""

from __future__ import annotations

from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # No-op: anchor migration to start revision chain
    pass


def downgrade():
    # No-op
    pass
