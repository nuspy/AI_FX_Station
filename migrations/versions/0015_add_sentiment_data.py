"""add sentiment_data table

Revision ID: 0015_add_sentiment_data
Revises: 0014
Create Date: 2025-10-13 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0015'
down_revision = '544e5525b0f5'
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    insp = sa.inspect(bind)
    return insp.has_table(name)


def upgrade():
    bind = op.get_bind()
    if not _has_table(bind, "sentiment_data"):
        # Create sentiment_data table for storing market sentiment metrics
        op.create_table(
            'sentiment_data',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('symbol', sa.String(length=64), nullable=False),
            sa.Column('ts_utc', sa.BigInteger, nullable=False),
            sa.Column('long_pct', sa.Float, nullable=True),
            sa.Column('short_pct', sa.Float, nullable=True),
            sa.Column('total_traders', sa.Integer, nullable=True),
            sa.Column('confidence', sa.Float, nullable=True),
            sa.Column('sentiment', sa.String(length=32), nullable=True),
            sa.Column('ratio', sa.Float, nullable=True),
            sa.Column('buy_volume', sa.Float, nullable=True),
            sa.Column('sell_volume', sa.Float, nullable=True),
            sa.Column('provider', sa.String(length=64), nullable=True),
            sa.Column('ts_created_ms', sa.BigInteger, nullable=True)
        )

        # Create index for efficient queries
        op.create_index(
            'idx_sentiment_symbol_ts',
            'sentiment_data',
            ['symbol', 'ts_utc']
        )

        # Create index for time-based queries
        op.create_index(
            'idx_sentiment_ts',
            'sentiment_data',
            ['ts_utc']
        )


def downgrade():
    bind = op.get_bind()
    if _has_table(bind, "sentiment_data"):
        op.drop_index('idx_sentiment_ts', table_name='sentiment_data')
        op.drop_index('idx_sentiment_symbol_ts', table_name='sentiment_data')
        op.drop_table('sentiment_data')
