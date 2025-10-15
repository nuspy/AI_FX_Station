"""Execution quality tracking

Revision ID: 0012
Revises: 0011
Create Date: 2025-10-07

Adds fields for tracking execution quality metrics including slippage,
latency, and broker-specific information.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic
revision = '0012'
down_revision = '0011'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade database schema."""
    # Enhance trades table with execution quality fields
    # (Assuming trades table exists from backtesting)
    op.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20),
            direction VARCHAR(10),
            entry_time TIMESTAMP,
            entry_price FLOAT,
            exit_time TIMESTAMP,
            exit_price FLOAT,
            size FLOAT,
            pnl FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # Add execution quality fields
    op.add_column('trades',
                  sa.Column('intended_entry_price',
                           sa.Float,
                           nullable=True))

    op.add_column('trades',
                  sa.Column('intended_exit_price',
                           sa.Float,
                           nullable=True))

    op.add_column('trades',
                  sa.Column('entry_slippage',
                           sa.Float,
                           nullable=True))

    op.add_column('trades',
                  sa.Column('exit_slippage',
                           sa.Float,
                           nullable=True))

    op.add_column('trades',
                  sa.Column('entry_spread',
                           sa.Float,
                           nullable=True))

    op.add_column('trades',
                  sa.Column('exit_spread',
                           sa.Float,
                           nullable=True))

    op.add_column('trades',
                  sa.Column('entry_latency_ms',
                           sa.Integer,
                           nullable=True))

    op.add_column('trades',
                  sa.Column('exit_latency_ms',
                           sa.Integer,
                           nullable=True))

    op.add_column('trades',
                  sa.Column('broker_name',
                           sa.String(50),
                           nullable=True))

    op.add_column('trades',
                  sa.Column('order_type',
                           sa.String(20),
                           nullable=True))

    op.add_column('trades',
                  sa.Column('execution_venue',
                           sa.String(50),
                           nullable=True))

    # Create execution quality summary table
    op.create_table(
        'execution_quality_daily',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('date', sa.Date, nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('broker_name', sa.String(50), nullable=True),
        sa.Column('total_trades', sa.Integer, nullable=False),
        sa.Column('avg_entry_slippage', sa.Float, nullable=True),
        sa.Column('avg_exit_slippage', sa.Float, nullable=True),
        sa.Column('avg_entry_latency_ms', sa.Float, nullable=True),
        sa.Column('avg_exit_latency_ms', sa.Float, nullable=True),
        sa.Column('avg_spread', sa.Float, nullable=True),
        sa.Column('total_slippage_cost', sa.Float, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
    )

    op.create_index('idx_eqd_date_symbol',
                   'execution_quality_daily',
                   ['date', 'symbol'])


def downgrade():
    """Downgrade database schema."""
    op.drop_index('idx_eqd_date_symbol', table_name='execution_quality_daily')
    op.drop_table('execution_quality_daily')

    op.drop_column('trades', 'execution_venue')
    op.drop_column('trades', 'order_type')
    op.drop_column('trades', 'broker_name')
    op.drop_column('trades', 'exit_latency_ms')
    op.drop_column('trades', 'entry_latency_ms')
    op.drop_column('trades', 'exit_spread')
    op.drop_column('trades', 'entry_spread')
    op.drop_column('trades', 'exit_slippage')
    op.drop_column('trades', 'entry_slippage')
    op.drop_column('trades', 'intended_exit_price')
    op.drop_column('trades', 'intended_entry_price')
