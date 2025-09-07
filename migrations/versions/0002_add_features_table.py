"""add features table

Revision ID: 0002_add_features_table
Revises: 0001_initial
Create Date: 2025-09-07 00:10:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0002_add_features_table"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "features",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(64), nullable=False, index=True),
        sa.Column("timeframe", sa.String(16), nullable=False, index=True),
        sa.Column("ts_utc", sa.Integer, nullable=False, index=True),
        sa.Column("pipeline_version", sa.String(64), nullable=True),
        sa.Column("features_json", sa.Text, nullable=False),
        sa.Column("ts_created_ms", sa.Integer, nullable=False),
    )
    op.create_index("ux_features_symbol_tf_ts", "features", ["symbol", "timeframe", "ts_utc"], unique=False)


def downgrade():
    op.drop_index("ux_features_symbol_tf_ts", table_name="features")
    op.drop_table("features")
