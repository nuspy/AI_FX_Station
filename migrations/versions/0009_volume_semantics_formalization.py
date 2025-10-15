"""Volume semantics formalization

Revision ID: 0009
Revises: 0008
Create Date: 2025-10-07

Formalizes volume semantics (tick volume vs actual volume) and adds
provider metadata tracking for data quality monitoring.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic
revision = '0009'
down_revision = '0008'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade database schema."""
    # Rename volume column to tick_volume for clarity
    op.alter_column('ticks', 'volume',
                    new_column_name='tick_volume',
                    existing_type=sa.BigInteger())

    # Add volume_type metadata
    op.add_column('ticks',
                  sa.Column('volume_type',
                           sa.String(20),
                           nullable=False,
                           server_default='TICK_COUNT'))

    # Add data provider tracking
    op.add_column('ticks',
                  sa.Column('data_provider',
                           sa.String(50),
                           nullable=True))

    # Add data quality flag
    op.add_column('ticks',
                  sa.Column('quality_flag',
                           sa.String(20),
                           nullable=True))

    # Create index for provider queries
    op.create_index('idx_ticks_provider',
                   'ticks',
                   ['data_provider'])


def downgrade():
    """Downgrade database schema."""
    op.drop_index('idx_ticks_provider', table_name='ticks')
    op.drop_column('ticks', 'quality_flag')
    op.drop_column('ticks', 'data_provider')
    op.drop_column('ticks', 'volume_type')
    op.alter_column('ticks', 'tick_volume',
                    new_column_name='volume',
                    existing_type=sa.BigInteger())
