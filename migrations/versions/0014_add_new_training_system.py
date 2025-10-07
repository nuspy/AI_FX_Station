"""add new training system tables

Revision ID: 0014
Revises: 0013
Create Date: 2025-10-07 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '0014'
down_revision = '0013'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create new training system tables."""
    
    # Table: training_runs
    op.create_table(
        'training_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_uuid', sa.String(36), nullable=False, unique=True),
        sa.Column('status', sa.String(20), nullable=False),
        
        # Model Configuration
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('encoder', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('base_timeframe', sa.String(10), nullable=False),
        sa.Column('days_history', sa.Integer(), nullable=False),
        sa.Column('horizon', sa.Integer(), nullable=False),
        
        # Feature Configuration (JSON)
        sa.Column('indicator_tfs', sa.JSON(), nullable=True),
        sa.Column('additional_features', sa.JSON(), nullable=True),
        sa.Column('preprocessing_params', sa.JSON(), nullable=True),
        sa.Column('model_hyperparams', sa.JSON(), nullable=True),
        
        # Training Results
        sa.Column('training_metrics', sa.JSON(), nullable=True),
        sa.Column('feature_count', sa.Integer(), nullable=True),
        sa.Column('training_duration_seconds', sa.Float(), nullable=True),
        
        # File Management
        sa.Column('model_file_path', sa.String(500), nullable=True),
        sa.Column('model_file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('is_model_kept', sa.Boolean(), default=False),
        
        # Regime Performance
        sa.Column('best_regimes', sa.JSON(), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        
        # Provenance
        sa.Column('created_by', sa.String(50), default='system'),
        sa.Column('config_hash', sa.String(64), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Indexes for training_runs
    op.create_index('idx_tr_status', 'training_runs', ['status'])
    op.create_index('idx_tr_config_hash', 'training_runs', ['config_hash'])
    op.create_index('idx_tr_symbol_timeframe', 'training_runs', ['symbol', 'base_timeframe'])
    op.create_index('idx_tr_model_type', 'training_runs', ['model_type'])
    op.create_index('idx_tr_created_at', 'training_runs', ['created_at'])
    
    # Table: inference_backtests
    op.create_table(
        'inference_backtests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('backtest_uuid', sa.String(36), nullable=False, unique=True),
        sa.Column('training_run_id', sa.Integer(), nullable=False),
        
        # Inference Configuration
        sa.Column('prediction_method', sa.String(50), nullable=False),
        sa.Column('ensemble_method', sa.String(50), nullable=True),
        sa.Column('confidence_threshold', sa.Float(), nullable=True),
        sa.Column('lookback_window', sa.Integer(), nullable=True),
        sa.Column('inference_params', sa.JSON(), nullable=True),
        
        # Backtest Results
        sa.Column('backtest_metrics', sa.JSON(), nullable=True),
        sa.Column('backtest_duration_seconds', sa.Float(), nullable=True),
        sa.Column('regime_metrics', sa.JSON(), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['training_run_id'], ['training_runs.id'], ondelete='CASCADE')
    )
    
    # Indexes for inference_backtests
    op.create_index('idx_ib_training_run', 'inference_backtests', ['training_run_id'])
    op.create_index('idx_ib_prediction_method', 'inference_backtests', ['prediction_method'])
    
    # Table: regime_definitions
    op.create_table(
        'regime_definitions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('regime_name', sa.String(50), nullable=False, unique=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('detection_rules', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column('is_active', sa.Boolean(), default=True),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Table: regime_best_models
    op.create_table(
        'regime_best_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('regime_name', sa.String(50), nullable=False),
        sa.Column('training_run_id', sa.Integer(), nullable=False),
        sa.Column('inference_backtest_id', sa.Integer(), nullable=False),
        
        # Performance Metrics
        sa.Column('performance_score', sa.Float(), nullable=False),
        sa.Column('secondary_metrics', sa.JSON(), nullable=True),
        
        # Timestamp
        sa.Column('achieved_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['training_run_id'], ['training_runs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['inference_backtest_id'], ['inference_backtests.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['regime_name'], ['regime_definitions.regime_name'], ondelete='CASCADE'),
        sa.UniqueConstraint('regime_name', name='uq_regime_best_model')
    )
    
    # Index for regime_best_models
    op.create_index('idx_rbm_regime', 'regime_best_models', ['regime_name'])
    
    # Table: training_queue
    op.create_table(
        'training_queue',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('queue_uuid', sa.String(36), nullable=False, unique=True),
        
        # Configuration
        sa.Column('config_grid', sa.JSON(), nullable=False),
        sa.Column('current_index', sa.Integer(), default=0),
        sa.Column('total_configs', sa.Integer(), nullable=False),
        
        # Status
        sa.Column('status', sa.String(20), nullable=False),
        
        # Progress Tracking
        sa.Column('completed_count', sa.Integer(), default=0),
        sa.Column('failed_count', sa.Integer(), default=0),
        sa.Column('skipped_count', sa.Integer(), default=0),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('paused_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        
        # Settings
        sa.Column('priority', sa.Integer(), default=0),
        sa.Column('max_parallel', sa.Integer(), default=1),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Indexes for training_queue
    op.create_index('idx_tq_status', 'training_queue', ['status'])
    op.create_index('idx_tq_priority', 'training_queue', ['priority'], postgresql_ops={'priority': 'DESC'})
    
    # Insert default regime definitions
    op.execute("""
        INSERT INTO regime_definitions (regime_name, description, detection_rules, is_active) VALUES
        ('bull_trending', 'Strong upward trend', '{"trend_strength": "> 0.7", "returns": "> 0"}', 1),
        ('bear_trending', 'Strong downward trend', '{"trend_strength": "> 0.7", "returns": "< 0"}', 1),
        ('volatile_ranging', 'High volatility, no clear trend', '{"trend_strength": "< 0.3", "volatility": "> 75th percentile"}', 1),
        ('calm_ranging', 'Low volatility, no clear trend', '{"trend_strength": "< 0.3", "volatility": "< 50th percentile"}', 1)
    """)


def downgrade() -> None:
    """Drop new training system tables."""
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('regime_best_models')
    op.drop_table('inference_backtests')
    op.drop_table('training_queue')
    op.drop_table('regime_definitions')
    op.drop_table('training_runs')
