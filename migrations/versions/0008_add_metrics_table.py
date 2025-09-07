"""add metrics table

Revision ID: 0008_add_metrics_table
Revises: 0007_add_regime_label
Create Date: 2025-09-07 02:40:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0008_add_metrics_table"
down_revision = "0007_add_regime_label"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "metrics",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(128), nullable=False, index=True),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("labels", sa.Text, nullable=True),
        sa.Column("ts_created_ms", sa.Integer, nullable=False, index=True),
    )
    op.create_index("ix_metrics_name_ts", "metrics", ["name", "ts_created_ms"], unique=False)


def downgrade():
    op.drop_index("ix_metrics_name_ts", table_name="metrics")
    op.drop_table("metrics")
