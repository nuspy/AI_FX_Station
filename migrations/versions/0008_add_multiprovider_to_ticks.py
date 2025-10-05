"""add multi-provider columns to market_data_ticks

Extends market_data_ticks table with multi-provider support columns:
- tick_volume: Number of ticks/trades in the aggregation period
- real_volume: Real traded volume (if available from provider)
- provider_source: Data provider identifier (tiingo, ctrader, etc.)

This enables the aggregator service to properly track volume metrics
and data provenance for multi-provider market data.

Revision ID: 0008_add_multiprovider_to_ticks
Revises: 0007_add_ctrader_features
Create Date: 2025-10-05 03:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0008_add_multiprovider_to_ticks'
down_revision = '0007_add_ctrader_features'
branch_labels = None
depends_on = None


def upgrade():
    """
    Add multi-provider support columns to market_data_ticks table.

    These columns are nullable for backward compatibility with existing data.
    The aggregator service uses these columns to track volume metrics and
    data provenance across multiple market data providers.

    This migration is idempotent - it checks if columns exist before adding them.
    """
    from sqlalchemy import inspect
    from alembic import context

    # Get the current connection
    conn = op.get_bind()
    inspector = inspect(conn)

    # Get existing columns
    existing_columns = [col['name'] for col in inspector.get_columns('market_data_ticks')]

    # Columns to add
    columns_to_add = {
        'tick_volume': sa.Column('tick_volume', sa.Integer, nullable=True,
                                comment='Number of ticks/trades in aggregation period'),
        'real_volume': sa.Column('real_volume', sa.Float, nullable=True,
                                comment='Real traded volume from provider'),
        'provider_source': sa.Column('provider_source', sa.String(32), nullable=True,
                                    server_default='tiingo',
                                    comment='Data provider identifier')
    }

    # Add only missing columns
    with op.batch_alter_table('market_data_ticks', schema=None) as batch_op:
        for col_name, col_def in columns_to_add.items():
            if col_name not in existing_columns:
                batch_op.add_column(col_def)

        # Create index on provider_source if it doesn't exist
        existing_indexes = [idx['name'] for idx in inspector.get_indexes('market_data_ticks')]
        if 'ix_ticks_provider' not in existing_indexes:
            batch_op.create_index(
                'ix_ticks_provider',
                ['provider_source']
            )

    # Create composite index if it doesn't exist
    existing_indexes = [idx['name'] for idx in inspector.get_indexes('market_data_ticks')]
    if 'ix_ticks_symbol_tf_ts_provider' not in existing_indexes:
        op.create_index(
            'ix_ticks_symbol_tf_ts_provider',
            'market_data_ticks',
            ['symbol', 'timeframe', 'ts_utc', 'provider_source']
        )

    # Backfill existing data with default provider
    # All existing tick data is from Tiingo, so set provider_source accordingly
    if 'provider_source' in existing_columns or 'provider_source' in columns_to_add:
        op.execute(
            "UPDATE market_data_ticks SET provider_source = 'tiingo' WHERE provider_source IS NULL"
        )


def downgrade():
    """
    Remove multi-provider columns from market_data_ticks (DESTRUCTIVE).

    WARNING: This will permanently delete tick_volume, real_volume, and
    provider_source data from the market_data_ticks table.
    """

    # Drop composite index
    op.drop_index('ix_ticks_symbol_tf_ts_provider', table_name='market_data_ticks')

    # Remove columns from market_data_ticks
    with op.batch_alter_table('market_data_ticks', schema=None) as batch_op:
        # Drop provider index first
        batch_op.drop_index('ix_ticks_provider')

        # Drop columns
        batch_op.drop_column('provider_source')
        batch_op.drop_column('real_volume')
        batch_op.drop_column('tick_volume')
