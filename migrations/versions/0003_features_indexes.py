"""add indexes for features table

Revision ID: 0003_features_indexes
Revises: 0002_add_features_table
Create Date: 2025-09-07 00:20:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0003_features_indexes"
down_revision = "0002_add_features_table"
branch_labels = None
depends_on = None


def upgrade():
    # index on pipeline_version for faster model/pipeline queries
    op.create_index("ix_features_pipeline_version", "features", ["pipeline_version"], unique=False)
    # index on ts_created_ms for retention/compaction queries
    op.create_index("ix_features_ts_created", "features", ["ts_created_ms"], unique=False)
    # index on ts_utc for time-range queries
    op.create_index("ix_features_ts_utc", "features", ["ts_utc"], unique=False)


def downgrade():
    op.drop_index("ix_features_ts_utc", table_name="features")
    op.drop_index("ix_features_ts_created", table_name="features")
    op.drop_index("ix_features_pipeline_version", table_name="features")
