"""initial

Revision ID: 0001_initial
Revises: 
Create Date: 2025-09-07 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # market_data_candles
    op.create_table(
        "market_data_candles",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(64), nullable=False, index=True),
        sa.Column("timeframe", sa.String(16), nullable=False, index=True),
        sa.Column("ts_utc", sa.Integer, nullable=False, index=True),
        sa.Column("open", sa.Float, nullable=False),
        sa.Column("high", sa.Float, nullable=False),
        sa.Column("low", sa.Float, nullable=False),
        sa.Column("close", sa.Float, nullable=False),
        sa.Column("volume", sa.Float, nullable=True),
        sa.Column("resampled", sa.Boolean, nullable=True),
    )
    op.create_index("ux_market_symbol_tf_ts", "market_data_candles", ["symbol", "timeframe", "ts_utc"], unique=True)

    # predictions
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("timeframe", sa.String(16), nullable=False),
        sa.Column("ts_created_ms", sa.Integer, nullable=False),
        sa.Column("horizon", sa.String(32), nullable=False),
        sa.Column("q05", sa.Float, nullable=False),
        sa.Column("q50", sa.Float, nullable=False),
        sa.Column("q95", sa.Float, nullable=False),
        sa.Column("meta", sa.Text, nullable=True),
    )

    # calibration_records
    op.create_table(
        "calibration_records",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("timeframe", sa.String(16), nullable=False),
        sa.Column("ts_created_ms", sa.Integer, nullable=False),
        sa.Column("alpha", sa.Float, nullable=False),
        sa.Column("half_life_days", sa.Float, nullable=False),
        sa.Column("delta_global", sa.Float, nullable=False),
        sa.Column("cov_hat", sa.Float, nullable=False),
        sa.Column("details", sa.Text, nullable=True),
    )

    # signals
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("timeframe", sa.String(16), nullable=False),
        sa.Column("ts_created_ms", sa.Integer, nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("target_price", sa.Float, nullable=False),
        sa.Column("stop_price", sa.Float, nullable=False),
        sa.Column("metrics", sa.Text, nullable=True),
    )


def downgrade():
    op.drop_table("signals")
    op.drop_table("calibration_records")
    op.drop_table("predictions")
    op.drop_index("ux_market_symbol_tf_ts", table_name="market_data_candles")
    op.drop_table("market_data_candles")
