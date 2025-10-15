"""Forecast validation tracking

Revision ID: 0011
Revises: 0010
Create Date: 2025-10-07

Creates tables for tracking forecast predictions vs actuals for
retrospective validation and drift detection.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic
revision = '0011'
down_revision = '0010'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade database schema."""
    # Create forecast_validations table
    op.create_table(
        'forecast_validations',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('model_id', sa.Integer, nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('forecast_timestamp', sa.TIMESTAMP, nullable=False),
        sa.Column('horizon_minutes', sa.Integer, nullable=False),
        sa.Column('predicted_value', sa.Float, nullable=False),
        sa.Column('predicted_lower', sa.Float, nullable=True),
        sa.Column('predicted_upper', sa.Float, nullable=True),
        sa.Column('confidence_level', sa.Float, nullable=True),
        sa.Column('actual_value', sa.Float, nullable=True),
        sa.Column('error', sa.Float, nullable=True),
        sa.Column('absolute_error', sa.Float, nullable=True),
        sa.Column('squared_error', sa.Float, nullable=True),
        sa.Column('directional_correct', sa.Boolean, nullable=True),
        sa.Column('within_interval', sa.Boolean, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('validated_at', sa.TIMESTAMP, nullable=True),
    )

    # Create indexes for efficient queries
    op.create_index('idx_fv_model_timestamp',
                   'forecast_validations',
                   ['model_id', 'forecast_timestamp'])

    op.create_index('idx_fv_symbol_horizon',
                   'forecast_validations',
                   ['symbol', 'horizon_minutes'])

    op.create_index('idx_fv_validated_at',
                   'forecast_validations',
                   ['validated_at'])

    # Create model_performance_snapshots table
    op.create_table(
        'model_performance_snapshots',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('model_id', sa.Integer, nullable=False),
        sa.Column('snapshot_date', sa.Date, nullable=False),
        sa.Column('window_days', sa.Integer, nullable=False),
        sa.Column('total_forecasts', sa.Integer, nullable=False),
        sa.Column('mae', sa.Float, nullable=True),
        sa.Column('rmse', sa.Float, nullable=True),
        sa.Column('mape', sa.Float, nullable=True),
        sa.Column('directional_accuracy', sa.Float, nullable=True),
        sa.Column('interval_coverage', sa.Float, nullable=True),
        sa.Column('bias', sa.Float, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
    )

    op.create_index('idx_mps_model_date',
                   'model_performance_snapshots',
                   ['model_id', 'snapshot_date'])


def downgrade():
    """Downgrade database schema."""
    op.drop_index('idx_mps_model_date', table_name='model_performance_snapshots')
    op.drop_table('model_performance_snapshots')

    op.drop_index('idx_fv_validated_at', table_name='forecast_validations')
    op.drop_index('idx_fv_symbol_horizon', table_name='forecast_validations')
    op.drop_index('idx_fv_model_timestamp', table_name='forecast_validations')
    op.drop_table('forecast_validations')
