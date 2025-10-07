"""add_signal_quality_and_new_tables

Revision ID: 94ca081433e4
Revises: 0014
Create Date: 2025-10-07 16:18:57.887206

Adds signal quality scoring, order flow metrics, correlation matrices,
event signals, parameter adaptations, and extends existing tables for
the enhanced trading system.

This migration uses pure Alembic operations for maximum compatibility.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = '94ca081433e4'
down_revision = '0014'
branch_labels = None
depends_on = None


def table_exists(table_name: str) -> bool:
    """Check if table exists in database."""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if column exists in table."""
    if not table_exists(table_name):
        return False

    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade():
    """
    Apply migration to add signal quality tracking, order flow metrics,
    correlation matrices, event signals, and parameter adaptation.
    """

    # ========================================
    # 1. Extend signals table with quality dimensions
    # ========================================
    if table_exists('signals'):
        with op.batch_alter_table('signals', schema=None) as batch_op:
            # Signal classification
            if not column_exists('signals', 'signal_type'):
                batch_op.add_column(sa.Column('signal_type', sa.String(50), nullable=True))

            if not column_exists('signals', 'source'):
                batch_op.add_column(sa.Column(
                    'source', sa.String(50), nullable=True,
                    comment='Signal source: pattern, harmonic, orderflow, news, correlation'
                ))

            # Quality dimensions (0-1 scores)
            quality_columns = [
                'quality_pattern_strength',
                'quality_mtf_agreement',
                'quality_regime_confidence',
                'quality_volume_confirmation',
                'quality_sentiment_alignment',
                'quality_correlation_safety',
                'quality_composite_score'
            ]

            for col_name in quality_columns:
                if not column_exists('signals', col_name):
                    batch_op.add_column(sa.Column(col_name, sa.Float, nullable=True))

            # Execution tracking
            if not column_exists('signals', 'executed'):
                batch_op.add_column(sa.Column('executed', sa.Boolean, default=False, nullable=True))

            if not column_exists('signals', 'execution_reason'):
                batch_op.add_column(sa.Column('execution_reason', sa.String(200), nullable=True))

            if not column_exists('signals', 'outcome'):
                batch_op.add_column(sa.Column(
                    'outcome', sa.String(20), nullable=True,
                    comment='Outcome: win, loss, breakeven, null'
                ))

    # ========================================
    # 2. Extend pattern_events with harmonic-specific fields
    # ========================================
    if table_exists('pattern_events'):
        with op.batch_alter_table('pattern_events', schema=None) as batch_op:
            if not column_exists('pattern_events', 'fibonacci_ratios'):
                batch_op.add_column(sa.Column(
                    'fibonacci_ratios', sa.Text, nullable=True,
                    comment='JSON: Fibonacci ratio measurements [[0.618, 0.632], ...]'
                ))

            if not column_exists('pattern_events', 'formation_quality'):
                batch_op.add_column(sa.Column('formation_quality', sa.Float, nullable=True))

            if not column_exists('pattern_events', 'volume_profile'):
                batch_op.add_column(sa.Column(
                    'volume_profile', sa.Text, nullable=True,
                    comment='JSON: Volume distribution data'
                ))

            if not column_exists('pattern_events', 'multi_tf_confirmation'):
                batch_op.add_column(sa.Column('multi_tf_confirmation', sa.Boolean, nullable=True))

            if not column_exists('pattern_events', 'pattern_family'):
                batch_op.add_column(sa.Column(
                    'pattern_family', sa.String(50), nullable=True,
                    comment='Harmonic family: gartley, bat, butterfly, crab, etc.'
                ))

    # ========================================
    # 3. Create order_flow_metrics table
    # ========================================
    if not table_exists('order_flow_metrics'):
        op.create_table(
            'order_flow_metrics',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('symbol', sa.String(32), nullable=False, index=True),
            sa.Column('timeframe', sa.String(16), nullable=False),
            sa.Column('timestamp', sa.BigInteger, nullable=False, index=True),

            # Order book metrics
            sa.Column('bid_ask_spread', sa.Float, nullable=True, comment='Spread in price units'),
            sa.Column('bid_depth', sa.Float, nullable=True, comment='Total volume on bid side'),
            sa.Column('ask_depth', sa.Float, nullable=True, comment='Total volume on ask side'),
            sa.Column('depth_imbalance', sa.Float, nullable=True, comment='(bid - ask) / (bid + ask)'),

            # Order flow
            sa.Column('buy_volume', sa.Float, nullable=True),
            sa.Column('sell_volume', sa.Float, nullable=True),
            sa.Column('volume_imbalance', sa.Float, nullable=True, comment='(buy - sell) / (buy + sell)'),
            sa.Column('large_order_count', sa.Integer, nullable=True),

            # Statistical metrics
            sa.Column('spread_zscore', sa.Float, nullable=True),
            sa.Column('imbalance_zscore', sa.Float, nullable=True),
            sa.Column('absorption_detected', sa.Boolean, nullable=True),
            sa.Column('exhaustion_detected', sa.Boolean, nullable=True),

            # Metadata
            sa.Column('regime', sa.String(32), nullable=True),
            sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),

            # Composite indexes
            sa.Index('idx_orderflow_symbol_ts', 'symbol', 'timestamp'),
            sa.Index('idx_orderflow_symbol_tf', 'symbol', 'timeframe'),
        )

    # ========================================
    # 4. Create correlation_matrices table
    # ========================================
    if not table_exists('correlation_matrices'):
        op.create_table(
            'correlation_matrices',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('timestamp', sa.BigInteger, nullable=False, index=True),
            sa.Column('window_size', sa.Integer, nullable=False, comment='Rolling window in bars'),
            sa.Column('timeframe', sa.String(16), nullable=False),

            # Correlation data (stored as JSON)
            sa.Column('matrix_data', sa.Text, nullable=False, comment="JSON: {('EURUSD', 'GBPUSD'): 0.85, ...}"),
            sa.Column('asset_list', sa.Text, nullable=False, comment="JSON: ['EURUSD', 'GBPUSD', ...]"),

            # Regime info
            sa.Column('regime', sa.String(32), nullable=True),
            sa.Column('correlation_regime', sa.String(32), nullable=True, comment='high, low, mixed'),

            # Statistics
            sa.Column('avg_correlation', sa.Float, nullable=True),
            sa.Column('max_correlation', sa.Float, nullable=True),
            sa.Column('min_correlation', sa.Float, nullable=True),

            sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),

            sa.Index('idx_corr_timestamp', 'timestamp'),
            sa.Index('idx_corr_tf_window', 'timeframe', 'window_size'),
        )

    # ========================================
    # 5. Create event_signals table
    # ========================================
    if not table_exists('event_signals'):
        op.create_table(
            'event_signals',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('event_id', sa.Integer, nullable=True, comment='Link to economic_calendar if exists'),
            sa.Column('event_type', sa.String(50), nullable=False, comment='scheduled, news, announcement'),
            sa.Column('event_name', sa.String(200), nullable=False),
            sa.Column('event_timestamp', sa.BigInteger, nullable=False, index=True),

            # Affected symbols
            sa.Column('affected_symbols', sa.Text, nullable=False, comment="JSON: ['EURUSD', 'GBPUSD']"),
            sa.Column('impact_level', sa.String(20), nullable=False, comment='high, medium, low'),

            # Signal information
            sa.Column('signal_direction', sa.String(10), nullable=True, comment='bull, bear, neutral'),
            sa.Column('signal_strength', sa.Float, nullable=True, comment='0-1 scale'),
            sa.Column('signal_timing', sa.String(20), nullable=False, comment='pre_event, post_event, reaction'),

            # Sentiment
            sa.Column('sentiment_score', sa.Float, nullable=True, comment='-1 to +1 scale'),
            sa.Column('sentiment_velocity', sa.Float, nullable=True, comment='Rate of sentiment change'),
            sa.Column('surprise_factor', sa.Float, nullable=True, comment='Consensus vs actual deviation'),

            # Execution window
            sa.Column('valid_from', sa.BigInteger, nullable=True),
            sa.Column('valid_until', sa.BigInteger, nullable=True),

            # Metadata
            sa.Column('executed', sa.Boolean, default=False, nullable=True),
            sa.Column('outcome', sa.String(20), nullable=True),
            sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),

            sa.Index('idx_event_signals_ts', 'event_timestamp'),
            sa.Index('idx_event_signals_type', 'event_type'),
            sa.Index('idx_event_signals_impact', 'impact_level'),
        )

    # ========================================
    # 6. Create signal_quality_history table
    # ========================================
    if not table_exists('signal_quality_history'):
        op.create_table(
            'signal_quality_history',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('signal_id', sa.Integer, nullable=True, comment='Link to signals table'),
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
            sa.Column('executed', sa.Boolean, default=False, nullable=True),
            sa.Column('outcome', sa.String(20), nullable=True),
            sa.Column('pnl', sa.Float, nullable=True),

            # Regime context
            sa.Column('regime', sa.String(32), nullable=True),

            sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),

            sa.Index('idx_quality_hist_ts', 'timestamp'),
            sa.Index('idx_quality_hist_source', 'signal_source'),
            sa.Index('idx_quality_hist_composite', 'quality_composite_score'),
        )

    # ========================================
    # 7. Create parameter_adaptations table
    # ========================================
    if not table_exists('parameter_adaptations'):
        op.create_table(
            'parameter_adaptations',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('adaptation_timestamp', sa.BigInteger, nullable=False, index=True),
            sa.Column('trigger_reason', sa.String(100), nullable=False, comment='performance_drop, regime_change, etc.'),

            # Performance metrics that triggered adaptation
            sa.Column('trigger_metrics', sa.Text, nullable=True, comment='JSON: {win_rate: 0.45, ...}'),
            sa.Column('lookback_trades', sa.Integer, nullable=False),

            # Parameter changes
            sa.Column('parameter_name', sa.String(100), nullable=False),
            sa.Column('old_value', sa.Float, nullable=True),
            sa.Column('new_value', sa.Float, nullable=True),
            sa.Column('parameter_type', sa.String(50), nullable=True, comment='confidence_threshold, position_size, etc.'),

            # Scope
            sa.Column('regime', sa.String(32), nullable=True, comment='Regime-specific adaptation'),
            sa.Column('symbol', sa.String(32), nullable=True, comment='Symbol-specific adaptation'),
            sa.Column('timeframe', sa.String(16), nullable=True, comment='Timeframe-specific adaptation'),

            # Validation
            sa.Column('validation_method', sa.String(50), nullable=True),
            sa.Column('validation_passed', sa.Boolean, nullable=True),
            sa.Column('improvement_expected', sa.Float, nullable=True),
            sa.Column('improvement_actual', sa.Float, nullable=True),

            # Status
            sa.Column('deployed', sa.Boolean, default=False, nullable=True),
            sa.Column('deployed_at', sa.DateTime, nullable=True),
            sa.Column('rollback_at', sa.DateTime, nullable=True),

            sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),

            sa.Index('idx_param_adapt_ts', 'adaptation_timestamp'),
            sa.Index('idx_param_adapt_name', 'parameter_name'),
            sa.Index('idx_param_adapt_deployed', 'deployed'),
        )

    # ========================================
    # 8. Extend regime_definitions for transition states
    # ========================================
    if table_exists('regime_definitions'):
        with op.batch_alter_table('regime_definitions', schema=None) as batch_op:
            if not column_exists('regime_definitions', 'is_transition'):
                batch_op.add_column(sa.Column('is_transition', sa.Boolean, default=False, nullable=True))

            if not column_exists('regime_definitions', 'probability_entropy'):
                batch_op.add_column(sa.Column('probability_entropy', sa.Float, nullable=True))

            if not column_exists('regime_definitions', 'min_duration_bars'):
                batch_op.add_column(sa.Column('min_duration_bars', sa.Integer, nullable=True))

            if not column_exists('regime_definitions', 'pause_trading'):
                batch_op.add_column(sa.Column('pause_trading', sa.Boolean, default=False, nullable=True))

    # ========================================
    # 9. Extend calibration_records for regime-specific calibration
    # ========================================
    if table_exists('calibration_records'):
        with op.batch_alter_table('calibration_records', schema=None) as batch_op:
            if not column_exists('calibration_records', 'regime'):
                batch_op.add_column(sa.Column('regime', sa.String(32), nullable=True))

            if not column_exists('calibration_records', 'calibration_window'):
                batch_op.add_column(sa.Column('calibration_window', sa.Integer, nullable=True))

            if not column_exists('calibration_records', 'asymmetric_up_delta'):
                batch_op.add_column(sa.Column('asymmetric_up_delta', sa.Float, nullable=True))

            if not column_exists('calibration_records', 'asymmetric_down_delta'):
                batch_op.add_column(sa.Column('asymmetric_down_delta', sa.Float, nullable=True))

            if not column_exists('calibration_records', 'coverage_accuracy'):
                batch_op.add_column(sa.Column('coverage_accuracy', sa.Float, nullable=True))

            if not column_exists('calibration_records', 'interval_sharpness'):
                batch_op.add_column(sa.Column('interval_sharpness', sa.Float, nullable=True))

    # ========================================
    # 10. Create ensemble_model_predictions table
    # ========================================
    if not table_exists('ensemble_model_predictions'):
        op.create_table(
            'ensemble_model_predictions',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('timestamp', sa.BigInteger, nullable=False, index=True),
            sa.Column('symbol', sa.String(32), nullable=False),
            sa.Column('timeframe', sa.String(16), nullable=False),

            # Model identification
            sa.Column('model_name', sa.String(50), nullable=False, comment='xgboost, lstm, transformer, etc.'),
            sa.Column('model_type', sa.String(50), nullable=False),
            sa.Column('model_version', sa.String(50), nullable=True),

            # Prediction
            sa.Column('prediction', sa.Float, nullable=False),
            sa.Column('confidence', sa.Float, nullable=True),
            sa.Column('prediction_horizon', sa.Integer, nullable=False, comment='Bars ahead'),

            # Performance tracking
            sa.Column('regime', sa.String(32), nullable=True),
            sa.Column('recent_accuracy', sa.Float, nullable=True),
            sa.Column('weight_in_ensemble', sa.Float, nullable=True),

            # Ensemble context
            sa.Column('ensemble_id', sa.String(50), nullable=True),
            sa.Column('ensemble_prediction', sa.Float, nullable=True),
            sa.Column('disagreement_score', sa.Float, nullable=True, comment='Model disagreement indicator'),

            sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),

            sa.Index('idx_ensemble_pred_ts', 'timestamp'),
            sa.Index('idx_ensemble_pred_model', 'model_name'),
            sa.Index('idx_ensemble_pred_symbol', 'symbol', 'timeframe'),
        )


def downgrade():
    """
    Revert migration - remove all added tables and columns.
    """

    # Drop new tables (in reverse order of creation)
    tables_to_drop = [
        'ensemble_model_predictions',
        'parameter_adaptations',
        'signal_quality_history',
        'event_signals',
        'correlation_matrices',
        'order_flow_metrics'
    ]

    for table_name in tables_to_drop:
        if table_exists(table_name):
            op.drop_table(table_name)

    # Remove columns from signals
    if table_exists('signals'):
        with op.batch_alter_table('signals', schema=None) as batch_op:
            columns_to_drop = [
                'outcome',
                'execution_reason',
                'executed',
                'quality_composite_score',
                'quality_correlation_safety',
                'quality_sentiment_alignment',
                'quality_volume_confirmation',
                'quality_regime_confidence',
                'quality_mtf_agreement',
                'quality_pattern_strength',
                'source',
                'signal_type'
            ]

            for col_name in columns_to_drop:
                if column_exists('signals', col_name):
                    batch_op.drop_column(col_name)

    # Remove columns from pattern_events
    if table_exists('pattern_events'):
        with op.batch_alter_table('pattern_events', schema=None) as batch_op:
            columns_to_drop = [
                'pattern_family',
                'multi_tf_confirmation',
                'volume_profile',
                'formation_quality',
                'fibonacci_ratios'
            ]

            for col_name in columns_to_drop:
                if column_exists('pattern_events', col_name):
                    batch_op.drop_column(col_name)

    # Remove columns from regime_definitions
    if table_exists('regime_definitions'):
        with op.batch_alter_table('regime_definitions', schema=None) as batch_op:
            columns_to_drop = [
                'pause_trading',
                'min_duration_bars',
                'probability_entropy',
                'is_transition'
            ]

            for col_name in columns_to_drop:
                if column_exists('regime_definitions', col_name):
                    batch_op.drop_column(col_name)

    # Remove columns from calibration_records
    if table_exists('calibration_records'):
        with op.batch_alter_table('calibration_records', schema=None) as batch_op:
            columns_to_drop = [
                'interval_sharpness',
                'coverage_accuracy',
                'asymmetric_down_delta',
                'asymmetric_up_delta',
                'calibration_window',
                'regime'
            ]

            for col_name in columns_to_drop:
                if column_exists('calibration_records', col_name):
                    batch_op.drop_column(col_name)
