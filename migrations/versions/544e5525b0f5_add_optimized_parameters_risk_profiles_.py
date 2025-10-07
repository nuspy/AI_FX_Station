"""add_optimized_parameters_risk_profiles_advanced_metrics

Revision ID: 544e5525b0f5
Revises: 94ca081433e4
Create Date: 2025-10-07 23:21:36.742990

Adds three critical tables for automated trading system:
- optimized_parameters: Store backtesting-optimized parameters per pattern/regime
- risk_profiles: Predefined and custom risk management profiles
- advanced_metrics: Extended performance metrics (Sortino, Calmar, MAR, etc.)

This migration uses pure Alembic operations with idempotency checks.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = '544e5525b0f5'
down_revision = '94ca081433e4'
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
    Add optimized_parameters, risk_profiles, and advanced_metrics tables
    for automated trading system with parameter optimization and risk management.
    """

    # ========================================================================
    # Table: optimized_parameters
    # Stores backtesting-optimized parameters per pattern/symbol/regime
    # ========================================================================
    if not table_exists('optimized_parameters'):
        op.create_table(
            'optimized_parameters',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('pattern_type', sa.String(50), nullable=False,
                     comment='Pattern type: harmonic, orderflow, correlation, etc.'),
            sa.Column('symbol', sa.String(20), nullable=False,
                     comment='Trading pair symbol'),
            sa.Column('timeframe', sa.String(10), nullable=False,
                     comment='Timeframe: 1m, 5m, 15m, 1h, 4h, 1d'),
            sa.Column('market_regime', sa.String(50), nullable=True,
                     comment='Market regime: trending_up, trending_down, ranging, high_volatility, etc.'),

            # Form parameters (pattern recognition)
            sa.Column('form_params', sa.Text, nullable=False,
                     comment='JSON: Pattern-specific form parameters from optimization'),

            # Action parameters (entry/exit logic)
            sa.Column('action_params', sa.Text, nullable=False,
                     comment='JSON: Entry/exit parameters (SL, TP multipliers, filters, etc.)'),

            # Performance metrics from optimization
            sa.Column('performance_metrics', sa.Text, nullable=False,
                     comment='JSON: {sharpe, sortino, max_dd, win_rate, profit_factor, etc.}'),

            # Optimization metadata
            sa.Column('optimization_timestamp', sa.DateTime, nullable=False,
                     comment='When these parameters were optimized'),
            sa.Column('data_range_start', sa.DateTime, nullable=False,
                     comment='Start of data range used for optimization'),
            sa.Column('data_range_end', sa.DateTime, nullable=False,
                     comment='End of data range used for optimization'),
            sa.Column('sample_count', sa.Integer, nullable=False,
                     comment='Number of samples in optimization dataset'),

            # Validation
            sa.Column('validation_status', sa.String(20), nullable=False, server_default='pending',
                     comment='Status: pending, validated, rejected, deployed'),
            sa.Column('validation_metrics', sa.Text, nullable=True,
                     comment='JSON: Out-of-sample validation results'),
            sa.Column('deployment_date', sa.DateTime, nullable=True,
                     comment='When parameters were deployed to live/paper trading'),

            # Tracking
            sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),

            # Composite indexes for fast retrieval
            sa.Index('idx_opt_params_pattern_symbol_tf', 'pattern_type', 'symbol', 'timeframe'),
            sa.Index('idx_opt_params_regime', 'market_regime'),
            sa.Index('idx_opt_params_validation', 'validation_status'),
            sa.Index('idx_opt_params_timestamp', 'optimization_timestamp'),
        )

    # ========================================================================
    # Table: risk_profiles
    # Predefined and custom risk management profiles
    # ========================================================================
    if not table_exists('risk_profiles'):
        op.create_table(
            'risk_profiles',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('profile_name', sa.String(50), nullable=False, unique=True,
                     comment='Profile name: Conservative, Moderate, Aggressive, or custom name'),
            sa.Column('profile_type', sa.String(20), nullable=False,
                     comment='Type: predefined or custom'),
            sa.Column('is_active', sa.Boolean, nullable=False, server_default='0',
                     comment='Whether this profile is currently active'),

            # Position sizing parameters
            sa.Column('max_risk_per_trade_pct', sa.Float, nullable=False,
                     comment='Maximum risk per trade as % of capital (e.g., 1.0 for 1%)'),
            sa.Column('max_portfolio_risk_pct', sa.Float, nullable=False,
                     comment='Maximum total portfolio risk % (e.g., 5.0 for 5%)'),
            sa.Column('position_sizing_method', sa.String(20), nullable=False,
                     comment='Method: fixed_fractional, kelly, optimal_f, volatility_adjusted'),
            sa.Column('kelly_fraction', sa.Float, nullable=True,
                     comment='Kelly Criterion fraction (0.25 = quarter Kelly, safer than full)'),

            # Stop loss parameters
            sa.Column('base_sl_atr_multiplier', sa.Float, nullable=False,
                     comment='Base stop loss as ATR multiplier (e.g., 2.0)'),
            sa.Column('base_tp_atr_multiplier', sa.Float, nullable=False,
                     comment='Base take profit as ATR multiplier (e.g., 3.0)'),
            sa.Column('use_trailing_stop', sa.Boolean, nullable=False, server_default='1',
                     comment='Enable trailing stop loss'),
            sa.Column('trailing_activation_pct', sa.Float, nullable=True,
                     comment='Profit % to activate trailing stop (e.g., 50.0 for 50% of target)'),

            # Adaptive adjustments
            sa.Column('regime_adjustment_enabled', sa.Boolean, nullable=False, server_default='1',
                     comment='Adjust parameters based on market regime'),
            sa.Column('volatility_adjustment_enabled', sa.Boolean, nullable=False, server_default='1',
                     comment='Adjust position size based on volatility'),
            sa.Column('news_awareness_enabled', sa.Boolean, nullable=False, server_default='1',
                     comment='Reduce positions before high-impact news'),

            # Diversification
            sa.Column('max_correlated_positions', sa.Integer, nullable=False,
                     comment='Maximum number of highly correlated positions (e.g., 2)'),
            sa.Column('correlation_threshold', sa.Float, nullable=False,
                     comment='Correlation threshold to consider positions correlated (e.g., 0.7)'),
            sa.Column('max_positions_per_symbol', sa.Integer, nullable=False,
                     comment='Maximum concurrent positions per symbol'),
            sa.Column('max_total_positions', sa.Integer, nullable=False,
                     comment='Maximum total concurrent positions'),

            # Drawdown protection
            sa.Column('max_daily_loss_pct', sa.Float, nullable=False,
                     comment='Max daily loss % before stopping trading (e.g., 3.0)'),
            sa.Column('max_drawdown_pct', sa.Float, nullable=False,
                     comment='Max drawdown % before stopping trading (e.g., 10.0)'),
            sa.Column('recovery_mode_threshold_pct', sa.Float, nullable=False,
                     comment='Drawdown % to enter recovery mode (e.g., 5.0)'),
            sa.Column('recovery_risk_multiplier', sa.Float, nullable=False,
                     comment='Risk multiplier in recovery mode (e.g., 0.5 for half risk)'),

            # Metadata
            sa.Column('description', sa.Text, nullable=True,
                     comment='Profile description and notes'),
            sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),

            # Indexes
            sa.Index('idx_risk_profile_name', 'profile_name'),
            sa.Index('idx_risk_profile_active', 'is_active'),
        )

    # ========================================================================
    # Table: advanced_metrics
    # Extended performance metrics beyond basic backtest results
    # ========================================================================
    if not table_exists('advanced_metrics'):
        op.create_table(
            'advanced_metrics',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('metric_type', sa.String(20), nullable=False,
                     comment='Type: backtest, live, paper, validation'),
            sa.Column('reference_id', sa.Integer, nullable=True,
                     comment='ID of related backtest/optimization run'),
            sa.Column('symbol', sa.String(20), nullable=True,
                     comment='Symbol if metric is symbol-specific'),
            sa.Column('timeframe', sa.String(10), nullable=True,
                     comment='Timeframe if metric is timeframe-specific'),
            sa.Column('market_regime', sa.String(50), nullable=True,
                     comment='Market regime if metric is regime-specific'),

            # Time period
            sa.Column('period_start', sa.DateTime, nullable=False,
                     comment='Start of measurement period'),
            sa.Column('period_end', sa.DateTime, nullable=False,
                     comment='End of measurement period'),

            # Basic metrics (for reference)
            sa.Column('total_return_pct', sa.Float, nullable=True),
            sa.Column('total_trades', sa.Integer, nullable=True),
            sa.Column('win_rate_pct', sa.Float, nullable=True),
            sa.Column('sharpe_ratio', sa.Float, nullable=True),

            # Advanced risk-adjusted metrics
            sa.Column('sortino_ratio', sa.Float, nullable=True,
                     comment='Sortino Ratio: Return / Downside Deviation (better than Sharpe)'),
            sa.Column('calmar_ratio', sa.Float, nullable=True,
                     comment='Calmar Ratio: Annual Return / Max Drawdown'),
            sa.Column('mar_ratio', sa.Float, nullable=True,
                     comment='MAR Ratio: CAGR / Max Drawdown (similar to Calmar)'),
            sa.Column('omega_ratio', sa.Float, nullable=True,
                     comment='Omega Ratio: Probability-weighted gains/losses'),
            sa.Column('gain_to_pain_ratio', sa.Float, nullable=True,
                     comment='Sum of returns / Sum of absolute negative returns'),

            # Drawdown metrics
            sa.Column('max_drawdown_pct', sa.Float, nullable=True),
            sa.Column('max_drawdown_duration_days', sa.Integer, nullable=True,
                     comment='Longest drawdown period in days'),
            sa.Column('avg_drawdown_pct', sa.Float, nullable=True,
                     comment='Average drawdown percentage'),
            sa.Column('recovery_time_days', sa.Integer, nullable=True,
                     comment='Average time to recover from drawdowns'),
            sa.Column('ulcer_index', sa.Float, nullable=True,
                     comment='Ulcer Index: Depth and duration of drawdowns'),

            # Return distribution
            sa.Column('return_skewness', sa.Float, nullable=True,
                     comment='Skewness: Asymmetry of return distribution'),
            sa.Column('return_kurtosis', sa.Float, nullable=True,
                     comment='Kurtosis: Tail risk (>3 = fat tails)'),
            sa.Column('var_95_pct', sa.Float, nullable=True,
                     comment='Value at Risk at 95% confidence'),
            sa.Column('cvar_95_pct', sa.Float, nullable=True,
                     comment='Conditional VaR (Expected Shortfall) at 95%'),

            # Win/Loss analysis
            sa.Column('avg_win_pct', sa.Float, nullable=True),
            sa.Column('avg_loss_pct', sa.Float, nullable=True),
            sa.Column('largest_win_pct', sa.Float, nullable=True),
            sa.Column('largest_loss_pct', sa.Float, nullable=True),
            sa.Column('win_loss_ratio', sa.Float, nullable=True,
                     comment='Average Win / Average Loss'),
            sa.Column('profit_factor', sa.Float, nullable=True,
                     comment='Gross Profit / Gross Loss'),

            # Consistency metrics
            sa.Column('win_streak_max', sa.Integer, nullable=True),
            sa.Column('loss_streak_max', sa.Integer, nullable=True),
            sa.Column('monthly_win_rate_pct', sa.Float, nullable=True,
                     comment='Percentage of profitable months'),
            sa.Column('expectancy_per_trade', sa.Float, nullable=True,
                     comment='Expected profit per trade'),

            # System quality
            sa.Column('system_quality_number', sa.Float, nullable=True,
                     comment='SQN: Expectancy * sqrt(N) / StdDev'),
            sa.Column('k_ratio', sa.Float, nullable=True,
                     comment='K-Ratio: Slope / Error of log equity curve'),

            # Additional data
            sa.Column('extra_metrics', sa.Text, nullable=True,
                     comment='JSON: Additional custom metrics'),

            # Metadata
            sa.Column('calculated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),

            # Indexes
            sa.Index('idx_adv_metrics_type', 'metric_type'),
            sa.Index('idx_adv_metrics_reference', 'reference_id'),
            sa.Index('idx_adv_metrics_symbol_tf', 'symbol', 'timeframe'),
            sa.Index('idx_adv_metrics_period', 'period_start', 'period_end'),
        )


def downgrade():
    """
    Remove optimized_parameters, risk_profiles, and advanced_metrics tables.
    """

    # Drop tables in reverse order
    tables_to_drop = [
        'advanced_metrics',
        'risk_profiles',
        'optimized_parameters',
    ]

    for table_name in tables_to_drop:
        if table_exists(table_name):
            op.drop_table(table_name)
