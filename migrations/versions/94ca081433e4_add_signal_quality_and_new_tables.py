"""add_signal_quality_and_new_tables

Revision ID: 94ca081433e4
Revises: 0014
Create Date: 2025-10-07 16:18:57.887206

Adds signal quality scoring, order flow metrics, correlation matrices,
event signals, parameter adaptations, and extends existing tables for
the enhanced trading system.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '94ca081433e4'
down_revision = '0014'
branch_labels = None
depends_on = None


def upgrade():
    # ========================================
    # 1. Extend signals table with quality dimensions
    # ========================================
    with op.batch_alter_table('signals', schema=None) as batch_op:
        batch_op.add_column(sa.Column('signal_type', sa.String(50), nullable=True))
        batch_op.add_column(sa.Column('source', sa.String(50), nullable=True))  # pattern, harmonic, orderflow, news, correlation

        # Quality dimensions (0-1 scores)
        batch_op.add_column(sa.Column('quality_pattern_strength', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('quality_mtf_agreement', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('quality_regime_confidence', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('quality_volume_confirmation', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('quality_sentiment_alignment', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('quality_correlation_safety', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('quality_composite_score', sa.Float, nullable=True))

        # Execution tracking
        batch_op.add_column(sa.Column('executed', sa.Boolean, default=False))
        batch_op.add_column(sa.Column('execution_reason', sa.String(200), nullable=True))
        batch_op.add_column(sa.Column('outcome', sa.String(20), nullable=True))  # win, loss, breakeven, null

    # ========================================
    # 2. Extend pattern_events with harmonic-specific fields
    # ========================================
    with op.batch_alter_table('pattern_events', schema=None) as batch_op:
        batch_op.add_column(sa.Column('fibonacci_ratios', sa.Text, nullable=True))  # JSON: [[0.618, 0.632], ...]
        batch_op.add_column(sa.Column('formation_quality', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('volume_profile', sa.Text, nullable=True))  # JSON: volume data
        batch_op.add_column(sa.Column('multi_tf_confirmation', sa.Boolean, nullable=True))
        batch_op.add_column(sa.Column('pattern_family', sa.String(50), nullable=True))  # gartley, bat, butterfly, etc.

    # ========================================
    # 3. Create order_flow_metrics table
    # ========================================
    op.create_table(
        'order_flow_metrics',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(32), nullable=False, index=True),
        sa.Column('timeframe', sa.String(16), nullable=False),
        sa.Column('timestamp', sa.BigInteger, nullable=False, index=True),

        # Order book metrics
        sa.Column('bid_ask_spread', sa.Float, nullable=True),
        sa.Column('bid_depth', sa.Float, nullable=True),
        sa.Column('ask_depth', sa.Float, nullable=True),
        sa.Column('depth_imbalance', sa.Float, nullable=True),  # (bid - ask) / (bid + ask)

        # Order flow
        sa.Column('buy_volume', sa.Float, nullable=True),
        sa.Column('sell_volume', sa.Float, nullable=True),
        sa.Column('volume_imbalance', sa.Float, nullable=True),  # (buy - sell) / (buy + sell)
        sa.Column('large_order_count', sa.Integer, nullable=True),

        # Statistical metrics
        sa.Column('spread_zscore', sa.Float, nullable=True),
        sa.Column('imbalance_zscore', sa.Float, nullable=True),
        sa.Column('absorption_detected', sa.Boolean, nullable=True),
        sa.Column('exhaustion_detected', sa.Boolean, nullable=True),

        # Metadata
        sa.Column('regime', sa.String(32), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
    )

    # Index for fast queries
    op.create_index('idx_orderflow_symbol_ts', 'order_flow_metrics', ['symbol', 'timestamp'])

    # ========================================
    # 4. Create correlation_matrices table
    # ========================================
    op.create_table(
        'correlation_matrices',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('timestamp', sa.BigInteger, nullable=False, index=True),
        sa.Column('window_size', sa.Integer, nullable=False),  # rolling window in bars
        sa.Column('timeframe', sa.String(16), nullable=False),

        # Correlation data (stored as JSON)
        sa.Column('matrix_data', sa.Text, nullable=False),  # JSON: {('EURUSD', 'GBPUSD'): 0.85, ...}
        sa.Column('asset_list', sa.Text, nullable=False),  # JSON: ['EURUSD', 'GBPUSD', ...]

        # Regime info
        sa.Column('regime', sa.String(32), nullable=True),
        sa.Column('correlation_regime', sa.String(32), nullable=True),  # high, low, mixed

        # Statistics
        sa.Column('avg_correlation', sa.Float, nullable=True),
        sa.Column('max_correlation', sa.Float, nullable=True),
        sa.Column('min_correlation', sa.Float, nullable=True),

        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
    )

    op.create_index('idx_corr_timestamp', 'correlation_matrices', ['timestamp'])

    # ========================================
    # 5. Create event_signals table
    # ========================================
    op.create_table(
        'event_signals',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('event_id', sa.Integer, nullable=True),  # Link to economic_calendar if exists
        sa.Column('event_type', sa.String(50), nullable=False),  # scheduled, news, announcement
        sa.Column('event_name', sa.String(200), nullable=False),
        sa.Column('event_timestamp', sa.BigInteger, nullable=False, index=True),

        # Affected symbols
        sa.Column('affected_symbols', sa.Text, nullable=False),  # JSON: ['EURUSD', 'GBPUSD']
        sa.Column('impact_level', sa.String(20), nullable=False),  # high, medium, low

        # Signal information
        sa.Column('signal_direction', sa.String(10), nullable=True),  # bull, bear, neutral
        sa.Column('signal_strength', sa.Float, nullable=True),  # 0-1
        sa.Column('signal_timing', sa.String(20), nullable=False),  # pre_event, post_event, reaction

        # Sentiment
        sa.Column('sentiment_score', sa.Float, nullable=True),  # -1 to +1
        sa.Column('sentiment_velocity', sa.Float, nullable=True),  # rate of change
        sa.Column('surprise_factor', sa.Float, nullable=True),  # consensus vs actual

        # Execution window
        sa.Column('valid_from', sa.BigInteger, nullable=True),
        sa.Column('valid_until', sa.BigInteger, nullable=True),

        # Metadata
        sa.Column('executed', sa.Boolean, default=False),
        sa.Column('outcome', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
    )

    op.create_index('idx_event_signals_ts', 'event_signals', ['event_timestamp'])

    # ========================================
    # 6. Create signal_quality_history table
    # ========================================
    op.create_table(
        'signal_quality_history',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('signal_id', sa.Integer, nullable=True),  # Link to signals table
        sa.Column('timestamp', sa.BigInteger, nullable=False, index=True),
        sa.Column('signal_source', sa.String(50), nullable=False),

        # Quality dimensions (same as signals table)
        sa.Column('quality_pattern_strength', sa.Float, nullable=True),
        sa.Column('quality_mtf_agreement', sa.Float, nullable=True),
        sa.Column('quality_regime_confidence', sa.Float, nullable=True),
        sa.Column('quality_volume_confirmation', sa.Float, nullable=True),
        sa.Column('quality_sentiment_alignment', sa.Float, nullable=True),
        sa.Column('quality_correlation_safety', sa.Float, nullable=True),
        sa.Column('quality_composite_score', sa.Float, nullable=False),

        # Quality threshold used
        sa.Column('threshold_used', sa.Float, nullable=False),
        sa.Column('passed_threshold', sa.Boolean, nullable=False),

        # Outcome tracking
        sa.Column('executed', sa.Boolean, default=False),
        sa.Column('outcome', sa.String(20), nullable=True),
        sa.Column('pnl', sa.Float, nullable=True),

        # Regime context
        sa.Column('regime', sa.String(32), nullable=True),

        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
    )

    op.create_index('idx_quality_hist_ts', 'signal_quality_history', ['timestamp'])
    op.create_index('idx_quality_hist_source', 'signal_quality_history', ['signal_source'])

    # ========================================
    # 7. Create parameter_adaptations table
    # ========================================
    op.create_table(
        'parameter_adaptations',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('adaptation_timestamp', sa.BigInteger, nullable=False, index=True),
        sa.Column('trigger_reason', sa.String(100), nullable=False),  # performance_drop, regime_change, etc.

        # Performance metrics that triggered adaptation
        sa.Column('trigger_metrics', sa.Text, nullable=True),  # JSON: {win_rate: 0.45, ...}
        sa.Column('lookback_trades', sa.Integer, nullable=False),

        # Parameter changes
        sa.Column('parameter_name', sa.String(100), nullable=False),
        sa.Column('old_value', sa.Float, nullable=True),
        sa.Column('new_value', sa.Float, nullable=True),
        sa.Column('parameter_type', sa.String(50), nullable=True),  # confidence_threshold, position_size, etc.

        # Scope
        sa.Column('regime', sa.String(32), nullable=True),  # If regime-specific
        sa.Column('symbol', sa.String(32), nullable=True),  # If symbol-specific
        sa.Column('timeframe', sa.String(16), nullable=True),  # If timeframe-specific

        # Validation
        sa.Column('validation_method', sa.String(50), nullable=True),
        sa.Column('validation_passed', sa.Boolean, nullable=True),
        sa.Column('improvement_expected', sa.Float, nullable=True),
        sa.Column('improvement_actual', sa.Float, nullable=True),

        # Status
        sa.Column('deployed', sa.Boolean, default=False),
        sa.Column('deployed_at', sa.DateTime, nullable=True),
        sa.Column('rollback_at', sa.DateTime, nullable=True),

        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
    )

    op.create_index('idx_param_adapt_ts', 'parameter_adaptations', ['adaptation_timestamp'])
    op.create_index('idx_param_adapt_name', 'parameter_adaptations', ['parameter_name'])

    # ========================================
    # 8. Extend regime_definitions for transition states
    # ========================================
    with op.batch_alter_table('regime_definitions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_transition', sa.Boolean, default=False))
        batch_op.add_column(sa.Column('probability_entropy', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('min_duration_bars', sa.Integer, nullable=True))
        batch_op.add_column(sa.Column('pause_trading', sa.Boolean, default=False))

    # ========================================
    # 9. Extend calibration_records for regime-specific calibration
    # ========================================
    with op.batch_alter_table('calibration_records', schema=None) as batch_op:
        batch_op.add_column(sa.Column('regime', sa.String(32), nullable=True))
        batch_op.add_column(sa.Column('calibration_window', sa.Integer, nullable=True))
        batch_op.add_column(sa.Column('asymmetric_up_delta', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('asymmetric_down_delta', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('coverage_accuracy', sa.Float, nullable=True))
        batch_op.add_column(sa.Column('interval_sharpness', sa.Float, nullable=True))

    # ========================================
    # 10. Create ensemble_model_predictions table
    # ========================================
    op.create_table(
        'ensemble_model_predictions',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('timestamp', sa.BigInteger, nullable=False, index=True),
        sa.Column('symbol', sa.String(32), nullable=False),
        sa.Column('timeframe', sa.String(16), nullable=False),

        # Model identification
        sa.Column('model_name', sa.String(50), nullable=False),  # xgboost, lstm, transformer, etc.
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=True),

        # Prediction
        sa.Column('prediction', sa.Float, nullable=False),
        sa.Column('confidence', sa.Float, nullable=True),
        sa.Column('prediction_horizon', sa.Integer, nullable=False),

        # Performance tracking
        sa.Column('regime', sa.String(32), nullable=True),
        sa.Column('recent_accuracy', sa.Float, nullable=True),
        sa.Column('weight_in_ensemble', sa.Float, nullable=True),

        # Ensemble context
        sa.Column('ensemble_id', sa.String(50), nullable=True),
        sa.Column('ensemble_prediction', sa.Float, nullable=True),
        sa.Column('disagreement_score', sa.Float, nullable=True),

        sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
    )

    op.create_index('idx_ensemble_pred_ts', 'ensemble_model_predictions', ['timestamp'])
    op.create_index('idx_ensemble_pred_model', 'ensemble_model_predictions', ['model_name'])


def downgrade():
    # Drop new tables
    op.drop_table('ensemble_model_predictions')
    op.drop_table('parameter_adaptations')
    op.drop_table('signal_quality_history')
    op.drop_table('event_signals')
    op.drop_table('correlation_matrices')
    op.drop_table('order_flow_metrics')

    # Remove columns from signals
    with op.batch_alter_table('signals', schema=None) as batch_op:
        batch_op.drop_column('outcome')
        batch_op.drop_column('execution_reason')
        batch_op.drop_column('executed')
        batch_op.drop_column('quality_composite_score')
        batch_op.drop_column('quality_correlation_safety')
        batch_op.drop_column('quality_sentiment_alignment')
        batch_op.drop_column('quality_volume_confirmation')
        batch_op.drop_column('quality_regime_confidence')
        batch_op.drop_column('quality_mtf_agreement')
        batch_op.drop_column('quality_pattern_strength')
        batch_op.drop_column('source')
        batch_op.drop_column('signal_type')

    # Remove columns from pattern_events
    with op.batch_alter_table('pattern_events', schema=None) as batch_op:
        batch_op.drop_column('pattern_family')
        batch_op.drop_column('multi_tf_confirmation')
        batch_op.drop_column('volume_profile')
        batch_op.drop_column('formation_quality')
        batch_op.drop_column('fibonacci_ratios')

    # Remove columns from regime_definitions
    with op.batch_alter_table('regime_definitions', schema=None) as batch_op:
        batch_op.drop_column('pause_trading')
        batch_op.drop_column('min_duration_bars')
        batch_op.drop_column('probability_entropy')
        batch_op.drop_column('is_transition')

    # Remove columns from calibration_records
    with op.batch_alter_table('calibration_records', schema=None) as batch_op:
        batch_op.drop_column('interval_sharpness')
        batch_op.drop_column('coverage_accuracy')
        batch_op.drop_column('asymmetric_down_delta')
        batch_op.drop_column('asymmetric_up_delta')
        batch_op.drop_column('calibration_window')
        batch_op.drop_column('regime')
