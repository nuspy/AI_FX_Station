
"""fix ticks table and add unique constraints for candles/ticks

Revision ID: 0003_fix_ticks_and_constraints
Revises: 0001_initial
Create Date: 2025-09-12 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0003_fix_ticks_and_constraints"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

def _has_table(bind, name: str) -> bool:
    insp = sa.inspect(bind)
    return insp.has_table(name)

def _has_index(bind, table: str, index_name: str) -> bool:
    insp = sa.inspect(bind)
    try:
        idxs = insp.get_indexes(table)
    except Exception:
        return False
    return any(ix.get("name") == index_name for ix in idxs)

def upgrade() -> None:
    bind = op.get_bind()

    # Ensure market_data_candles has UNIQUE(symbol,timeframe,ts_utc)
    if _has_table(bind, "market_data_candles"):
        if not _has_index(bind, "market_data_candles", "ux_market_data_candles_sym_tf_ts"):
            op.create_index(
                "ux_market_data_candles_sym_tf_ts",
                "market_data_candles",
                ["symbol", "timeframe", "ts_utc"],
                unique=True,
            )
        # Helpful non-unique index (if you want it)
        if not _has_index(bind, "market_data_candles", "ix_market_data_candles_tf_ts"):
            op.create_index(
                "ix_market_data_candles_tf_ts",
                "market_data_candles",
                ["timeframe", "ts_utc"],
                unique=False,
            )

    # Create/ensure market_data_ticks
    if not _has_table(bind, "market_data_ticks"):
        op.create_table(
            "market_data_ticks",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("symbol", sa.String(length=64), nullable=False),
            sa.Column("timeframe", sa.String(length=16), nullable=False),
            sa.Column("ts_utc", sa.BigInteger, nullable=False),
            sa.Column("price", sa.Float, nullable=True),
            sa.Column("bid", sa.Float, nullable=True),
            sa.Column("ask", sa.Float, nullable=True),
            sa.Column("volume", sa.Float, nullable=True),
            sa.Column("ts_created_ms", sa.BigInteger, nullable=True),
        )
    # Unique & helper indexes for ticks
    if not _has_index(bind, "market_data_ticks", "ux_market_data_ticks_sym_tf_ts"):
        op.create_index(
            "ux_market_data_ticks_sym_tf_ts",
            "market_data_ticks",
            ["symbol", "timeframe", "ts_utc"],
            unique=True,
        )
    if not _has_index(bind, "market_data_ticks", "ix_market_data_ticks_tf_ts"):
        op.create_index(
            "ix_market_data_ticks_tf_ts",
            "market_data_ticks",
            ["timeframe", "ts_utc"],
            unique=False,
        )

def downgrade() -> None:
    bind = op.get_bind()
    # Drop helper indexes (leave core tables if they pre-existed from 0001)
    if _has_index(bind, "market_data_ticks", "ix_market_data_ticks_tf_ts"):
        op.drop_index("ix_market_data_ticks_tf_ts", table_name="market_data_ticks")
    if _has_index(bind, "market_data_ticks", "ux_market_data_ticks_sym_tf_ts"):
        op.drop_index("ux_market_data_ticks_sym_tf_ts", table_name="market_data_ticks")
    # We drop the table only if it exists – safe on both branches
    if _has_table(bind, "market_data_ticks"):
        op.drop_table("market_data_ticks")

    # Candles indexes (only drop those we created here)
    if _has_index(bind, "market_data_candles", "ix_market_data_candles_tf_ts"):
        op.drop_index("ix_market_data_candles_tf_ts", table_name="market_data_candles")
    if _has_index(bind, "market_data_candles", "ux_market_data_candles_sym_tf_ts"):
        op.drop_index("ux_market_data_candles_sym_tf_ts", table_name="market_data_candles")
