"""add_ldm4ts_support

Revision ID: 0020_20251017_182545
Revises: 0019_add_vix_data
Create Date: 2025-01-17 18:25:45

Adds tables for LDM4TS (Latent Diffusion Models for Time Series):
- ldm4ts_predictions: Store forecast predictions with uncertainty
- ldm4ts_model_metadata: Track model versions and performance
- ldm4ts_inference_metrics: Monitor inference time and quality
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '0020_20251017_182545'
down_revision = '0019_add_vix_data'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Upgrade schema to support LDM4TS predictions.
    """
    
    # Table 1: LDM4TS Predictions
    # Stores forecasts with full uncertainty quantification
    op.create_table(
        'ldm4ts_predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('prediction_id', sa.String(length=100), nullable=False, comment='Unique prediction identifier'),
        sa.Column('symbol', sa.String(length=20), nullable=False, comment='Trading symbol (e.g., EUR/USD)'),
        sa.Column('timeframe', sa.String(length=10), nullable=False, comment='Timeframe (1m, 5m, 15m, etc.)'),
        sa.Column('forecast_time', sa.DateTime(), nullable=False, comment='When forecast was generated'),
        sa.Column('horizon_minutes', sa.Integer(), nullable=False, comment='Forecast horizon in minutes'),
        sa.Column('target_time', sa.DateTime(), nullable=False, comment='Target time for prediction'),
        
        # Point estimates
        sa.Column('current_price', sa.Float(), nullable=False, comment='Price at forecast time'),
        sa.Column('mean_pred', sa.Float(), nullable=False, comment='Mean predicted price'),
        sa.Column('median_pred', sa.Float(), nullable=False, comment='Median predicted price (q50)'),
        
        # Uncertainty quantification
        sa.Column('std_pred', sa.Float(), nullable=False, comment='Standard deviation of prediction'),
        sa.Column('q05_pred', sa.Float(), nullable=False, comment='5th percentile (pessimistic)'),
        sa.Column('q95_pred', sa.Float(), nullable=False, comment='95th percentile (optimistic)'),
        
        # Direction & strength
        sa.Column('direction', sa.String(length=10), nullable=False, comment='bull, bear, neutral'),
        sa.Column('signal_strength', sa.Float(), nullable=False, comment='Signal strength [0-1]'),
        sa.Column('uncertainty_pct', sa.Float(), nullable=False, comment='Uncertainty as % of price'),
        
        # Quality scores
        sa.Column('quality_score', sa.Float(), nullable=True, comment='Overall quality score [0-1]'),
        sa.Column('regime', sa.String(length=20), nullable=True, comment='Market regime at forecast time'),
        sa.Column('regime_confidence', sa.Float(), nullable=True, comment='Regime detection confidence'),
        
        # Actual outcomes (filled later)
        sa.Column('actual_price', sa.Float(), nullable=True, comment='Actual price at target time'),
        sa.Column('mae', sa.Float(), nullable=True, comment='Mean Absolute Error'),
        sa.Column('mse', sa.Float(), nullable=True, comment='Mean Squared Error'),
        sa.Column('mape', sa.Float(), nullable=True, comment='Mean Absolute Percentage Error'),
        sa.Column('interval_hit', sa.Boolean(), nullable=True, comment='Did actual fall in [q05, q95]?'),
        sa.Column('direction_correct', sa.Boolean(), nullable=True, comment='Was direction prediction correct?'),
        
        # Metadata
        sa.Column('model_version', sa.String(length=50), nullable=False, comment='Model checkpoint version'),
        sa.Column('inference_time_ms', sa.Float(), nullable=False, comment='Inference latency in milliseconds'),
        sa.Column('num_samples', sa.Integer(), nullable=False, comment='Monte Carlo samples used'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('prediction_id', name='uq_ldm4ts_predictions_prediction_id'),
        comment='LDM4TS forecast predictions with uncertainty quantification'
    )
    
    # Indexes for fast queries
    op.create_index('idx_ldm4ts_predictions_symbol_time', 'ldm4ts_predictions', 
                   ['symbol', 'forecast_time'], unique=False)
    op.create_index('idx_ldm4ts_predictions_target_time', 'ldm4ts_predictions', 
                   ['target_time'], unique=False)
    op.create_index('idx_ldm4ts_predictions_horizon', 'ldm4ts_predictions', 
                   ['horizon_minutes'], unique=False)
    
    
    # Table 2: LDM4TS Model Metadata
    # Track different model versions and their performance
    op.create_table(
        'ldm4ts_model_metadata',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False, comment='Model version/checkpoint name'),
        sa.Column('model_path', sa.String(length=500), nullable=False, comment='Path to checkpoint file'),
        sa.Column('symbol', sa.String(length=20), nullable=False, comment='Symbol trained on'),
        sa.Column('timeframe', sa.String(length=10), nullable=False, comment='Timeframe trained on'),
        
        # Architecture config
        sa.Column('image_size', sa.Integer(), nullable=False, comment='Vision transform image size (224)'),
        sa.Column('vae_model', sa.String(length=100), nullable=False, comment='VAE model name (stabilityai/sd-vae-ft-mse)'),
        sa.Column('diffusion_steps', sa.Integer(), nullable=False, comment='Number of diffusion steps (50)'),
        sa.Column('num_mc_samples', sa.Integer(), nullable=False, comment='Monte Carlo samples for uncertainty'),
        
        # Training info
        sa.Column('training_start', sa.DateTime(), nullable=True, comment='Training start time'),
        sa.Column('training_end', sa.DateTime(), nullable=True, comment='Training end time'),
        sa.Column('num_epochs', sa.Integer(), nullable=True, comment='Number of training epochs'),
        sa.Column('training_samples', sa.Integer(), nullable=True, comment='Number of training samples'),
        sa.Column('validation_mse', sa.Float(), nullable=True, comment='Best validation MSE'),
        sa.Column('validation_mae', sa.Float(), nullable=True, comment='Best validation MAE'),
        
        # Production metrics (updated over time)
        sa.Column('total_predictions', sa.Integer(), nullable=False, default=0, comment='Total predictions made'),
        sa.Column('avg_inference_ms', sa.Float(), nullable=True, comment='Average inference time'),
        sa.Column('avg_mae', sa.Float(), nullable=True, comment='Average MAE on production data'),
        sa.Column('avg_mse', sa.Float(), nullable=True, comment='Average MSE on production data'),
        sa.Column('directional_accuracy', sa.Float(), nullable=True, comment='% correct direction predictions'),
        sa.Column('interval_coverage', sa.Float(), nullable=True, comment='% actuals in [q05, q95]'),
        
        # Status
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True, comment='Is model currently in use?'),
        sa.Column('deployed_at', sa.DateTime(), nullable=True, comment='When deployed to production'),
        sa.Column('retired_at', sa.DateTime(), nullable=True, comment='When retired from production'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_version', name='uq_ldm4ts_model_metadata_version'),
        comment='LDM4TS model versions and performance tracking'
    )
    
    op.create_index('idx_ldm4ts_model_metadata_symbol', 'ldm4ts_model_metadata', 
                   ['symbol', 'timeframe'], unique=False)
    op.create_index('idx_ldm4ts_model_metadata_active', 'ldm4ts_model_metadata', 
                   ['is_active'], unique=False)
    
    
    # Table 3: LDM4TS Inference Metrics
    # Monitor inference quality and performance
    op.create_table(
        'ldm4ts_inference_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False, comment='Model version'),
        sa.Column('symbol', sa.String(length=20), nullable=False, comment='Trading symbol'),
        sa.Column('timeframe', sa.String(length=10), nullable=False, comment='Timeframe'),
        sa.Column('metric_time', sa.DateTime(), nullable=False, comment='When metric was recorded'),
        
        # Inference performance
        sa.Column('inference_count', sa.Integer(), nullable=False, comment='Number of inferences in window'),
        sa.Column('avg_inference_ms', sa.Float(), nullable=False, comment='Average inference time'),
        sa.Column('p95_inference_ms', sa.Float(), nullable=False, comment='95th percentile inference time'),
        sa.Column('p99_inference_ms', sa.Float(), nullable=False, comment='99th percentile inference time'),
        sa.Column('max_inference_ms', sa.Float(), nullable=False, comment='Max inference time'),
        
        # Quality metrics (rolling window)
        sa.Column('avg_uncertainty_pct', sa.Float(), nullable=True, comment='Average uncertainty %'),
        sa.Column('avg_signal_strength', sa.Float(), nullable=True, comment='Average signal strength'),
        sa.Column('avg_quality_score', sa.Float(), nullable=True, comment='Average quality score'),
        
        # Accuracy metrics (for completed predictions)
        sa.Column('completed_predictions', sa.Integer(), nullable=True, comment='Predictions with actual outcomes'),
        sa.Column('rolling_mae', sa.Float(), nullable=True, comment='Rolling MAE'),
        sa.Column('rolling_mse', sa.Float(), nullable=True, comment='Rolling MSE'),
        sa.Column('rolling_directional_acc', sa.Float(), nullable=True, comment='Rolling directional accuracy'),
        sa.Column('rolling_interval_coverage', sa.Float(), nullable=True, comment='Rolling interval coverage'),
        
        # Errors
        sa.Column('error_count', sa.Integer(), nullable=False, default=0, comment='Number of inference errors'),
        sa.Column('last_error', sa.Text(), nullable=True, comment='Last error message'),
        
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        
        sa.PrimaryKeyConstraint('id'),
        comment='LDM4TS inference performance and quality monitoring'
    )
    
    op.create_index('idx_ldm4ts_inference_metrics_time', 'ldm4ts_inference_metrics', 
                   ['metric_time'], unique=False)
    op.create_index('idx_ldm4ts_inference_metrics_model', 'ldm4ts_inference_metrics', 
                   ['model_version', 'symbol'], unique=False)
    
    
    # Table 4: Add LDM4TS config to existing trading_engine_configs
    # (Extend existing table if it exists, otherwise document for manual addition)
    try:
        # Check if table exists
        conn = op.get_bind()
        if conn.dialect.has_table(conn, 'trading_engine_configs'):
            # Add LDM4TS columns to existing configs
            op.add_column('trading_engine_configs', 
                         sa.Column('use_ldm4ts', sa.Boolean(), nullable=False, 
                                  server_default='0', comment='Enable LDM4TS forecasts'))
            op.add_column('trading_engine_configs', 
                         sa.Column('ldm4ts_model_version', sa.String(length=50), nullable=True, 
                                  comment='LDM4TS model version to use'))
            op.add_column('trading_engine_configs', 
                         sa.Column('ldm4ts_horizons', sa.String(length=100), nullable=True, 
                                  comment='Forecast horizons (comma-separated minutes)'))
            op.add_column('trading_engine_configs', 
                         sa.Column('ldm4ts_uncertainty_threshold', sa.Float(), nullable=False, 
                                  server_default='0.5', comment='Max uncertainty % to accept'))
            op.add_column('trading_engine_configs', 
                         sa.Column('ldm4ts_min_strength', sa.Float(), nullable=False, 
                                  server_default='0.3', comment='Min signal strength'))
            op.add_column('trading_engine_configs', 
                         sa.Column('ldm4ts_position_scaling', sa.Boolean(), nullable=False, 
                                  server_default='1', comment='Scale position by uncertainty'))
    except Exception:
        # Table doesn't exist, skip
        pass


def downgrade() -> None:
    """
    Downgrade schema (remove LDM4TS support).
    """
    
    # Drop indexes first
    op.drop_index('idx_ldm4ts_inference_metrics_model', table_name='ldm4ts_inference_metrics')
    op.drop_index('idx_ldm4ts_inference_metrics_time', table_name='ldm4ts_inference_metrics')
    
    op.drop_index('idx_ldm4ts_model_metadata_active', table_name='ldm4ts_model_metadata')
    op.drop_index('idx_ldm4ts_model_metadata_symbol', table_name='ldm4ts_model_metadata')
    
    op.drop_index('idx_ldm4ts_predictions_horizon', table_name='ldm4ts_predictions')
    op.drop_index('idx_ldm4ts_predictions_target_time', table_name='ldm4ts_predictions')
    op.drop_index('idx_ldm4ts_predictions_symbol_time', table_name='ldm4ts_predictions')
    
    # Drop tables
    op.drop_table('ldm4ts_inference_metrics')
    op.drop_table('ldm4ts_model_metadata')
    op.drop_table('ldm4ts_predictions')
    
    # Remove columns from trading_engine_configs (if they exist)
    try:
        conn = op.get_bind()
        if conn.dialect.has_table(conn, 'trading_engine_configs'):
            op.drop_column('trading_engine_configs', 'ldm4ts_position_scaling')
            op.drop_column('trading_engine_configs', 'ldm4ts_min_strength')
            op.drop_column('trading_engine_configs', 'ldm4ts_uncertainty_threshold')
            op.drop_column('trading_engine_configs', 'ldm4ts_horizons')
            op.drop_column('trading_engine_configs', 'ldm4ts_model_version')
            op.drop_column('trading_engine_configs', 'use_ldm4ts')
    except Exception:
        pass
