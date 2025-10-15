"""add ctrader multi-provider features

Adds comprehensive support for cTrader and multi-provider architecture:
- Extends market_data_candles with real_volume, tick_volume, provider_source
- Adds market_depth table for DOM (Depth of Market) data
- Adds sentiment_data table for trader sentiment
- Adds news_events table for news feed
- Adds economic_calendar table for economic events
- Adds composite indexes for multi-provider queries
- Maintains backward compatibility with existing data

Revision ID: 0007_add_ctrader_features
Revises: 0006_add_optimization_system
Create Date: 2025-01-05 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade():
    """
    Add cTrader and multi-provider support to database schema.
    """

    # 1. Extend market_data_candles table with new columns
    # NOTE: SQLite doesn't support ALTER COLUMN, so we add columns only
    with op.batch_alter_table('market_data_candles', schema=None) as batch_op:
        # Add volume type columns (nullable for backward compatibility)
        batch_op.add_column(sa.Column('tick_volume', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('real_volume', sa.Float, nullable=True))

        # Add provider source tracking
        batch_op.add_column(sa.Column('provider_source', sa.String(32), nullable=True, server_default='tiingo'))

        # Create index on provider_source for filtering
        batch_op.create_index('ix_candles_provider', ['provider_source'])

    # Create composite index for multi-provider queries
    op.create_index(
        'ix_candles_symbol_tf_ts_provider',
        'market_data_candles',
        ['symbol', 'timeframe', 'ts_utc', 'provider_source']
    )

    # 2. Create market_depth table for DOM data
    op.create_table(
        'market_depth',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('symbol', sa.String(64), nullable=False, index=True),
        sa.Column('ts_utc', sa.Integer, nullable=False, index=True),
        sa.Column('provider', sa.String(32), nullable=False, index=True),

        # DOM data (JSON for flexibility)
        sa.Column('bids', sa.JSON, nullable=False),  # [(price, volume), ...]
        sa.Column('asks', sa.JSON, nullable=False),  # [(price, volume), ...]

        # Derived metrics
        sa.Column('mid_price', sa.Float, nullable=True),
        sa.Column('spread', sa.Float, nullable=True),
        sa.Column('imbalance', sa.Float, nullable=True),  # bid_volume / ask_volume

        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),

        # Composite unique constraint
        sa.UniqueConstraint('symbol', 'ts_utc', 'provider', name='uq_depth_symbol_ts_provider')
    )

    # Create indexes for market_depth
    op.create_index('ix_depth_symbol_ts', 'market_depth', ['symbol', 'ts_utc'])
    op.create_index('ix_depth_provider', 'market_depth', ['provider'])

    # 3. Create sentiment_data table
    op.create_table(
        'sentiment_data',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('symbol', sa.String(64), nullable=False, index=True),
        sa.Column('ts_utc', sa.Integer, nullable=False, index=True),
        sa.Column('provider', sa.String(32), nullable=False, index=True),

        # Sentiment metrics
        sa.Column('long_pct', sa.Float, nullable=False),
        sa.Column('short_pct', sa.Float, nullable=False),
        sa.Column('total_traders', sa.Integer, nullable=True),

        # Metadata
        sa.Column('confidence', sa.Float, nullable=True),  # Confidence level

        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),

        # Composite unique constraint
        sa.UniqueConstraint('symbol', 'ts_utc', 'provider', name='uq_sentiment_symbol_ts_provider')
    )

    # Create indexes for sentiment_data
    op.create_index('ix_sentiment_symbol_ts', 'sentiment_data', ['symbol', 'ts_utc'])
    op.create_index('ix_sentiment_provider', 'sentiment_data', ['provider'])

    # 4. Create news_events table
    op.create_table(
        'news_events',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('ts_utc', sa.Integer, nullable=False, index=True),
        sa.Column('provider', sa.String(32), nullable=False, index=True),

        # News content
        sa.Column('title', sa.String(512), nullable=False),
        sa.Column('content', sa.Text, nullable=True),
        sa.Column('url', sa.String(1024), nullable=True),

        # Categorization
        sa.Column('currency', sa.String(16), nullable=True, index=True),
        sa.Column('impact', sa.String(16), nullable=True, index=True),  # high, medium, low
        sa.Column('category', sa.String(64), nullable=True),

        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),

        # Unique constraint on title + timestamp to avoid duplicates
        sa.UniqueConstraint('title', 'ts_utc', 'provider', name='uq_news_title_ts_provider')
    )

    # Create indexes for news_events
    op.create_index('ix_news_ts', 'news_events', ['ts_utc'])
    op.create_index('ix_news_currency', 'news_events', ['currency'])
    op.create_index('ix_news_impact', 'news_events', ['impact'])
    op.create_index('ix_news_provider', 'news_events', ['provider'])

    # 5. Create economic_calendar table
    op.create_table(
        'economic_calendar',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('event_id', sa.String(128), nullable=False, unique=True, index=True),
        sa.Column('ts_utc', sa.Integer, nullable=False, index=True),
        sa.Column('provider', sa.String(32), nullable=False, index=True),

        # Event details
        sa.Column('event_name', sa.String(256), nullable=False),
        sa.Column('currency', sa.String(16), nullable=False, index=True),
        sa.Column('country', sa.String(64), nullable=True),

        # Economic data
        sa.Column('forecast', sa.String(64), nullable=True),
        sa.Column('previous', sa.String(64), nullable=True),
        sa.Column('actual', sa.String(64), nullable=True),

        # Impact level
        sa.Column('impact', sa.String(16), nullable=True, index=True),  # high, medium, low

        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),
    )

    # Create indexes for economic_calendar
    op.create_index('ix_calendar_ts', 'economic_calendar', ['ts_utc'])
    op.create_index('ix_calendar_currency', 'economic_calendar', ['currency'])
    op.create_index('ix_calendar_impact', 'economic_calendar', ['impact'])
    op.create_index('ix_calendar_provider', 'economic_calendar', ['provider'])

    # 6. Backfill existing data with default provider
    # Set provider_source to 'tiingo' for existing candles (safe default)
    op.execute(
        "UPDATE market_data_candles SET provider_source = 'tiingo' WHERE provider_source IS NULL"
    )


def downgrade():
    """
    Remove cTrader and multi-provider features (DESTRUCTIVE).
    """

    # Drop tables in reverse order
    op.drop_table('economic_calendar')
    op.drop_table('news_events')
    op.drop_table('sentiment_data')
    op.drop_table('market_depth')

    # Remove indexes from market_data_candles
    op.drop_index('ix_candles_symbol_tf_ts_provider', table_name='market_data_candles')

    with op.batch_alter_table('market_data_candles', schema=None) as batch_op:
        batch_op.drop_index('ix_candles_provider')
        batch_op.drop_column('provider_source')
        batch_op.drop_column('real_volume')
        batch_op.drop_column('tick_volume')
