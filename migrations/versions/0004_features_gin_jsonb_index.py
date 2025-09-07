"""add gin jsonb index on features_json (postgres optimization)

Revision ID: 0004_features_gin_jsonb_index
Revises: 0003_features_indexes
Create Date: 2025-09-07 00:40:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = "0004_features_gin_jsonb_index"
down_revision = "0003_features_indexes"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    dialect = conn.dialect.name.lower()
    # Create GIN index on features_json cast to jsonb for Postgres
    if dialect == "postgresql":
        # use jsonb_path_ops for compact containment indexing if available
        try:
            op.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_features_jsonb_gin ON features USING GIN ((features_json::jsonb) jsonb_path_ops);"
                )
            )
        except Exception:
            # fallback to default gin index on jsonb
            op.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_features_jsonb_gin ON features USING GIN ((features_json::jsonb));"
                )
            )
    else:
        # no-op for other dialects
        pass


def downgrade():
    conn = op.get_bind()
    dialect = conn.dialect.name.lower()
    if dialect == "postgresql":
        op.execute(text("DROP INDEX IF EXISTS ix_features_jsonb_gin;"))
