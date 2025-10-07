"""create market_data_ticks table

Revision ID: 0002_add_market_data_ticks
Revises: 0001_initial
Create Date: 2025-09-11 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    insp = sa.inspect(bind)
    return insp.has_table(name)


def upgrade():
    bind = op.get_bind()
    if not _has_table(bind, "market_data_ticks"):
        # create market_data_ticks table
        op.create_table(
            'market_data_ticks',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('symbol', sa.String(length=64), nullable=False),
            sa.Column('timeframe', sa.String(length=32), nullable=False),
            sa.Column('ts_utc', sa.BigInteger, nullable=False),
            sa.Column('price', sa.Float, nullable=True),
            sa.Column('bid', sa.Float, nullable=True),
            sa.Column('ask', sa.Float, nullable=True),
            sa.Column('volume', sa.Float, nullable=True),
            sa.Column('ts_created_ms', sa.BigInteger, nullable=True),
            sa.UniqueConstraint('symbol', 'timeframe', 'ts_utc', name='uq_ticks_symbol_tf_ts')
        )
    # else: table already exists; skip creation (idempotent)


def downgrade():
    bind = op.get_bind()
    if _has_table(bind, "market_data_ticks"):
        op.drop_table('market_data_ticks')
