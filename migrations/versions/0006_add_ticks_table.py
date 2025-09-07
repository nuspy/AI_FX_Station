"""add ticks_aggregate table

Revision ID: 0006_add_ticks_table
Revises: 0005_add_latents_table
Create Date: 2025-09-07 01:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0006_add_ticks_table"
down_revision = "0005_add_latents_table"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ticks_aggregate",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(64), nullable=False, index=True),
        sa.Column("timeframe", sa.String(16), nullable=False, index=True),
        sa.Column("ts_utc", sa.Integer, nullable=False, index=True),  # minute period end
        sa.Column("tick_count", sa.Integer, nullable=False),
        sa.Column("ts_created_ms", sa.Integer, nullable=False),
    )
    op.create_index("ix_ticks_symbol_tf_ts", "ticks_aggregate", ["symbol", "timeframe", "ts_utc"], unique=False)


def downgrade():
    op.drop_index("ix_ticks_symbol_tf_ts", table_name="ticks_aggregate")
    op.drop_table("ticks_aggregate")
