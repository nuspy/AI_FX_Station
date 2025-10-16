"""add vix_data table

Revision ID: 0016
Revises: f19e2695024b
Create Date: 2025-01-08 23:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0016'
down_revision = 'f19e2695024b'  # Depends on E2E optimization tables
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    """Check if table exists."""
    insp = sa.inspect(bind)
    return insp.has_table(name)


def upgrade():
    """Create vix_data table for storing CBOE Volatility Index data."""
    bind = op.get_bind()
    
    if not _has_table(bind, "vix_data"):
        # Create vix_data table
        op.create_table(
            'vix_data',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('ts_utc', sa.BigInteger, nullable=False, comment='Unix timestamp in seconds'),
            sa.Column('value', sa.Float, nullable=False, comment='VIX value'),
            sa.Column('classification', sa.String(length=32), nullable=False, comment='Complacency/Normal/Concern/Fear'),
            sa.Column('ts_created_ms', sa.BigInteger, nullable=False, comment='Creation timestamp in milliseconds')
        )

        # Create index for efficient time-based queries
        op.create_index(
            'idx_vix_ts',
            'vix_data',
            ['ts_utc']
        )
        
        print("Created vix_data table with index")
    else:
        print("Table vix_data already exists, skipping")


def downgrade():
    """Drop vix_data table."""
    bind = op.get_bind()
    
    if _has_table(bind, "vix_data"):
        op.drop_index('idx_vix_ts', table_name='vix_data')
        op.drop_table('vix_data')
        print("Dropped vix_data table")
    else:
        print("Table vix_data does not exist, skipping")
