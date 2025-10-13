"""add sentiment, ratio, volumes columns to sentiment_data

Revision ID: 0016_add_sentiment_data_columns
Revises: 0015
Create Date: 2025-10-13 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0016'
down_revision = '0015'
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    insp = sa.inspect(bind)
    return insp.has_table(name)


def _has_column(bind, table_name: str, column_name: str) -> bool:
    insp = sa.inspect(bind)
    if not insp.has_table(table_name):
        return False
    columns = [col['name'] for col in insp.get_columns(table_name)]
    return column_name in columns


def upgrade():
    bind = op.get_bind()

    # Add sentiment column if it doesn't exist
    if _has_table(bind, 'sentiment_data'):
        if not _has_column(bind, 'sentiment_data', 'sentiment'):
            op.add_column('sentiment_data', sa.Column('sentiment', sa.String(length=32), nullable=True))

        if not _has_column(bind, 'sentiment_data', 'ratio'):
            op.add_column('sentiment_data', sa.Column('ratio', sa.Float, nullable=True))

        if not _has_column(bind, 'sentiment_data', 'buy_volume'):
            op.add_column('sentiment_data', sa.Column('buy_volume', sa.Float, nullable=True))

        if not _has_column(bind, 'sentiment_data', 'sell_volume'):
            op.add_column('sentiment_data', sa.Column('sell_volume', sa.Float, nullable=True))

        if not _has_column(bind, 'sentiment_data', 'ts_created_ms'):
            op.add_column('sentiment_data', sa.Column('ts_created_ms', sa.BigInteger, nullable=True))


def downgrade():
    bind = op.get_bind()

    if _has_table(bind, 'sentiment_data'):
        if _has_column(bind, 'sentiment_data', 'ts_created_ms'):
            op.drop_column('sentiment_data', 'ts_created_ms')

        if _has_column(bind, 'sentiment_data', 'sell_volume'):
            op.drop_column('sentiment_data', 'sell_volume')

        if _has_column(bind, 'sentiment_data', 'buy_volume'):
            op.drop_column('sentiment_data', 'buy_volume')

        if _has_column(bind, 'sentiment_data', 'ratio'):
            op.drop_column('sentiment_data', 'ratio')

        if _has_column(bind, 'sentiment_data', 'sentiment'):
            op.drop_column('sentiment_data', 'sentiment')
