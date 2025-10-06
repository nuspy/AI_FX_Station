"""Add SSSD model support with multi-asset architecture

Revision ID: 0013
Revises: 0012
Create Date: 2025-10-06 19:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0013'
down_revision = '0012'
branch_labels = None
depends_on = None


def upgrade():
    """
    Create tables for SSSD (Structured State Space Diffusion) model support.

    Includes:
    - sssd_models: Model metadata and configuration
    - sssd_checkpoints: Training checkpoints
    - sssd_training_runs: Training history
    - sssd_inference_logs: Inference logging
    - sssd_performance_metrics: Performance tracking over time
    """

    # Table 1: sssd_models - Store SSSD model metadata
    op.create_table(
        'sssd_models',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('asset', sa.String(20), nullable=False, index=True,
                  comment='Asset symbol (EURUSD, GBPUSD, etc.)'),
        sa.Column('model_name', sa.String(100), nullable=False, unique=True,
                  comment='Unique model identifier (e.g., sssd_v1_eurusd_5m)'),
        sa.Column('model_type', sa.String(50), nullable=False, server_default='sssd_diffusion',
                  comment='Model type identifier'),
        sa.Column('architecture_config', sa.JSON(), nullable=False,
                  comment='S4 and diffusion architecture parameters'),
        sa.Column('training_config', sa.JSON(), nullable=False,
                  comment='Training hyperparameters (lr, batch_size, optimizer, etc.)'),
        sa.Column('horizon_config', sa.JSON(), nullable=False,
                  comment='Forecast horizons and weights'),
        sa.Column('feature_config', sa.JSON(), nullable=False,
                  comment='Feature engineering configuration'),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now(),
                  onupdate=sa.func.now()),
        sa.Column('created_by', sa.String(100), nullable=True,
                  comment='User or system that created the model'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('asset', 'model_name', name='uq_asset_model_name'),
        comment='SSSD model metadata and configuration per asset'
    )

    # Indexes for sssd_models
    op.create_index('idx_sssd_models_asset', 'sssd_models', ['asset'])
    op.create_index('idx_sssd_models_asset_model', 'sssd_models', ['asset', 'model_name'])
    op.create_index('idx_sssd_models_created_at', 'sssd_models', ['created_at'])

    # Table 2: sssd_checkpoints - Store model checkpoints
    op.create_table(
        'sssd_checkpoints',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.Integer(), nullable=False,
                  comment='Foreign key to sssd_models.id'),
        sa.Column('checkpoint_path', sa.String(500), nullable=False, unique=True,
                  comment='Filesystem path to checkpoint file'),
        sa.Column('epoch', sa.Integer(), nullable=False,
                  comment='Training epoch number'),
        sa.Column('training_loss', sa.Float(), nullable=False,
                  comment='Training loss at this checkpoint'),
        sa.Column('validation_loss', sa.Float(), nullable=True,
                  comment='Validation loss (if available)'),
        sa.Column('validation_metrics', sa.JSON(), nullable=True,
                  comment='Validation metrics (directional_accuracy, rmse, mae)'),
        sa.Column('checkpoint_size_mb', sa.Float(), nullable=False,
                  comment='Checkpoint file size in MB'),
        sa.Column('is_best', sa.Boolean(), nullable=False, server_default='false',
                  comment='True if this is the best checkpoint by validation loss'),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_id'], ['sssd_models.id'], ondelete='CASCADE'),
        comment='SSSD model checkpoints for resumable training'
    )

    # Indexes for sssd_checkpoints
    op.create_index('idx_sssd_checkpoints_model_id', 'sssd_checkpoints', ['model_id'])
    op.create_index('idx_sssd_checkpoints_is_best', 'sssd_checkpoints', ['is_best'])
    op.create_index('idx_sssd_checkpoints_epoch', 'sssd_checkpoints', ['epoch'])
    op.create_index('idx_sssd_checkpoints_created_at', 'sssd_checkpoints', ['created_at'])

    # Table 3: sssd_training_runs - Track training history
    op.create_table(
        'sssd_training_runs',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.Integer(), nullable=False,
                  comment='Foreign key to sssd_models.id'),
        sa.Column('run_name', sa.String(200), nullable=False,
                  comment='Unique run identifier'),
        sa.Column('training_data_range', sa.JSON(), nullable=False,
                  comment='Date ranges for train/val/test splits'),
        sa.Column('final_training_loss', sa.Float(), nullable=True),
        sa.Column('final_validation_loss', sa.Float(), nullable=True),
        sa.Column('best_epoch', sa.Integer(), nullable=True,
                  comment='Epoch with best validation loss'),
        sa.Column('total_epochs', sa.Integer(), nullable=False,
                  comment='Total epochs run'),
        sa.Column('training_duration_seconds', sa.Integer(), nullable=True,
                  comment='Total training time in seconds'),
        sa.Column('gpu_type', sa.String(100), nullable=True,
                  comment='GPU model used for training'),
        sa.Column('hyperparameters', sa.JSON(), nullable=False,
                  comment='Full hyperparameter snapshot for reproducibility'),
        sa.Column('training_logs_path', sa.String(500), nullable=True,
                  comment='Path to training log file'),
        sa.Column('status', sa.String(20), nullable=False, server_default='running',
                  comment='Status: running, completed, failed, interrupted'),
        sa.Column('error_message', sa.Text(), nullable=True,
                  comment='Error details if status=failed'),
        sa.Column('started_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column('completed_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_id'], ['sssd_models.id'], ondelete='CASCADE'),
        sa.CheckConstraint("status IN ('running', 'completed', 'failed', 'interrupted')",
                          name='ck_sssd_training_runs_status'),
        comment='SSSD training run history and metadata'
    )

    # Indexes for sssd_training_runs
    op.create_index('idx_sssd_training_runs_model_id', 'sssd_training_runs', ['model_id'])
    op.create_index('idx_sssd_training_runs_status', 'sssd_training_runs', ['status'])
    op.create_index('idx_sssd_training_runs_started_at', 'sssd_training_runs', ['started_at'])

    # Table 4: sssd_inference_logs - Log inference requests
    op.create_table(
        'sssd_inference_logs',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.Integer(), nullable=False,
                  comment='Foreign key to sssd_models.id'),
        sa.Column('symbol', sa.String(20), nullable=False,
                  comment='Trading symbol (EURUSD, etc.)'),
        sa.Column('timeframe', sa.String(10), nullable=False,
                  comment='Timeframe (5m, 15m, 1h, 4h)'),
        sa.Column('inference_timestamp', sa.TIMESTAMP(), nullable=False,
                  comment='When inference was requested'),
        sa.Column('data_timestamp', sa.TIMESTAMP(), nullable=False,
                  comment='Timestamp of last data bar used'),
        sa.Column('horizons', sa.JSON(), nullable=False,
                  comment='List of forecast horizons (minutes)'),
        sa.Column('predictions', sa.JSON(), nullable=False,
                  comment='Predictions with uncertainty (mean, std, quantiles)'),
        sa.Column('inference_time_ms', sa.Float(), nullable=False,
                  comment='Inference latency in milliseconds'),
        sa.Column('gpu_used', sa.Boolean(), nullable=False, server_default='false',
                  comment='Whether GPU was used for inference'),
        sa.Column('batch_size', sa.Integer(), nullable=False, server_default='1',
                  comment='Batch size for inference'),
        sa.Column('context_features', sa.JSON(), nullable=True,
                  comment='Summary of input features (for debugging)'),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_id'], ['sssd_models.id'], ondelete='CASCADE'),
        comment='SSSD inference request logs (retained for 30 days)'
    )

    # Indexes for sssd_inference_logs
    op.create_index('idx_sssd_inference_logs_model_id', 'sssd_inference_logs', ['model_id'])
    op.create_index('idx_sssd_inference_logs_symbol_timeframe', 'sssd_inference_logs',
                   ['symbol', 'timeframe'])
    op.create_index('idx_sssd_inference_logs_inference_ts', 'sssd_inference_logs',
                   ['inference_timestamp'])

    # Table 5: sssd_performance_metrics - Track performance over time
    op.create_table(
        'sssd_performance_metrics',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.Integer(), nullable=False,
                  comment='Foreign key to sssd_models.id'),
        sa.Column('evaluation_date', sa.Date(), nullable=False,
                  comment='Date of evaluation'),
        sa.Column('evaluation_period_start', sa.TIMESTAMP(), nullable=False,
                  comment='Start of evaluation window'),
        sa.Column('evaluation_period_end', sa.TIMESTAMP(), nullable=False,
                  comment='End of evaluation window'),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('directional_accuracy', sa.Float(), nullable=False,
                  comment='Percentage of correct directional predictions'),
        sa.Column('rmse', sa.Float(), nullable=False,
                  comment='Root Mean Squared Error'),
        sa.Column('mae', sa.Float(), nullable=False,
                  comment='Mean Absolute Error'),
        sa.Column('mape', sa.Float(), nullable=True,
                  comment='Mean Absolute Percentage Error'),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True,
                  comment='Sharpe ratio if integrated with trading'),
        sa.Column('win_rate', sa.Float(), nullable=True,
                  comment='Win rate if used for trading'),
        sa.Column('profit_factor', sa.Float(), nullable=True,
                  comment='Profit factor if used for trading'),
        sa.Column('max_drawdown', sa.Float(), nullable=True,
                  comment='Maximum drawdown if used for trading'),
        sa.Column('num_predictions', sa.Integer(), nullable=False,
                  comment='Number of predictions evaluated'),
        sa.Column('num_trades', sa.Integer(), nullable=True,
                  comment='Number of trades executed (if applicable)'),
        sa.Column('confidence_calibration', sa.JSON(), nullable=True,
                  comment='Confidence interval calibration statistics'),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_id'], ['sssd_models.id'], ondelete='CASCADE'),
        comment='SSSD model performance metrics tracked over time'
    )

    # Indexes for sssd_performance_metrics
    op.create_index('idx_sssd_perf_metrics_model_date', 'sssd_performance_metrics',
                   ['model_id', 'evaluation_date'])
    op.create_index('idx_sssd_perf_metrics_symbol_tf', 'sssd_performance_metrics',
                   ['symbol', 'timeframe'])
    op.create_index('idx_sssd_perf_metrics_eval_date', 'sssd_performance_metrics',
                   ['evaluation_date'])

    # Modify existing tables to add SSSD support

    # Add SSSD-specific columns to model_metadata (if table exists)
    try:
        op.add_column('models', sa.Column('sssd_model_id', sa.Integer(), nullable=True,
                                          comment='Link to sssd_models table'))
        op.add_column('models', sa.Column('is_sssd_model', sa.Boolean(), nullable=False,
                                          server_default='false',
                                          comment='Flag to identify SSSD models'))
        op.create_foreign_key('fk_models_sssd_model_id', 'models', 'sssd_models',
                            ['sssd_model_id'], ['id'], ondelete='SET NULL')
        op.create_index('idx_models_is_sssd', 'models', ['is_sssd_model'])
    except Exception:
        # Table might not exist or columns already exist
        pass

    # Add SSSD-specific columns to ensemble_weights (if table exists)
    try:
        op.add_column('ensemble_weights', sa.Column('sssd_confidence_weight', sa.Float(),
                                                    nullable=False, server_default='1.0',
                                                    comment='Multiplicative weight based on SSSD uncertainty'))
        op.add_column('ensemble_weights', sa.Column('last_reweighting_date', sa.TIMESTAMP(),
                                                    nullable=True,
                                                    comment='When weights were last updated'))
    except Exception:
        # Table might not exist or columns already exist
        pass


def downgrade():
    """
    Remove SSSD-related tables and columns.
    """

    # Drop modified columns from existing tables
    try:
        op.drop_constraint('fk_models_sssd_model_id', 'models', type_='foreignkey')
        op.drop_index('idx_models_is_sssd', 'models')
        op.drop_column('models', 'sssd_model_id')
        op.drop_column('models', 'is_sssd_model')
    except Exception:
        pass

    try:
        op.drop_column('ensemble_weights', 'sssd_confidence_weight')
        op.drop_column('ensemble_weights', 'last_reweighting_date')
    except Exception:
        pass

    # Drop SSSD tables (order matters due to foreign keys)
    op.drop_table('sssd_performance_metrics')
    op.drop_table('sssd_inference_logs')
    op.drop_table('sssd_training_runs')
    op.drop_table('sssd_checkpoints')
    op.drop_table('sssd_models')
