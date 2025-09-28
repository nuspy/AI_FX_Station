"""add optimization system tables

Adds comprehensive tables for agentic pattern optimization and backtesting:
- optimization_studies: N-dimensional matrix cells for pattern/direction/asset/timeframe/regime
- optimization_trials: individual parameter trials with TaskID idempotency
- trial_metrics: multi-dataset/multi-objective performance metrics
- best_parameters: promoted parameter sets with per-regime overrides
- parameter_changelog: audit trail for parameter promotions/rollbacks
- regime_classifications: economic/market regime mappings for time periods
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0006_add_optimization_system'
down_revision = '0005_add_pattern_tables'
branch_labels = None
depends_on = None

def upgrade():
    # optimization_studies table - N-dimensional matrix cells
    op.create_table('optimization_studies',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),

        # Matrix dimensions
        sa.Column('pattern_key', sa.String(128), nullable=False, index=True),
        sa.Column('direction', sa.String(16), nullable=False, index=True),
        sa.Column('asset', sa.String(64), nullable=False, index=True),
        sa.Column('timeframe', sa.String(16), nullable=False, index=True),
        sa.Column('regime_tag', sa.String(64), nullable=True, index=True),

        # Study metadata
        sa.Column('study_name', sa.String(256), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('status', sa.String(32), nullable=False, server_default='pending'),

        # Dataset configuration
        sa.Column('dataset_1_config', sa.JSON, nullable=False),
        sa.Column('dataset_2_config', sa.JSON, nullable=True),

        # Multi-objective configuration
        sa.Column('is_multi_objective', sa.Boolean, server_default='1'),
        sa.Column('objective_weights', sa.JSON, nullable=True),

        # Parameter space configuration
        sa.Column('parameter_ranges', sa.JSON, nullable=False),

        # Optimization configuration
        sa.Column('max_trials', sa.Integer, server_default='1000'),
        sa.Column('max_duration_hours', sa.Float, server_default='24.0'),
        sa.Column('early_stopping_alpha', sa.Float, server_default='0.8'),
        sa.Column('min_trades_for_pruning', sa.Integer, server_default='10'),
        sa.Column('min_duration_months', sa.Integer, server_default='3'),

        # Invalidation rules (75th percentile logic)
        sa.Column('k_time_multiplier', sa.Float, server_default='4.0'),
        sa.Column('k_loss_multiplier', sa.Float, server_default='4.0'),
        sa.Column('quantile_threshold', sa.Float, server_default='0.75'),

        # Walk-forward validation settings
        sa.Column('walk_forward_months', sa.Integer, server_default='6'),
        sa.Column('purge_days', sa.Integer, server_default='1'),
        sa.Column('embargo_days', sa.Integer, server_default='2'),

        # Constraints
        sa.Column('min_signals_required', sa.Integer, server_default='20'),
        sa.Column('min_temporal_coverage_months', sa.Integer, server_default='6'),
        sa.Column('max_drawdown_threshold', sa.Float, server_default='0.20'),

        # Recency weighting
        sa.Column('recency_decay_months', sa.Float, server_default='12.0'),
        sa.Column('max_history_years', sa.Integer, server_default='10'),

        # Timestamps
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),

        # Unique constraint for matrix cell
        sa.UniqueConstraint('pattern_key', 'direction', 'asset', 'timeframe', 'regime_tag',
                           name='uq_study_matrix_cell')
    )

    # Create indexes for studies (unique constraint is in table definition)
    op.create_index('idx_study_status', 'optimization_studies', ['status'])
    op.create_index('idx_study_pattern', 'optimization_studies', ['pattern_key'])
    op.create_index('idx_study_asset_tf', 'optimization_studies', ['asset', 'timeframe'])

    # optimization_trials table - individual parameter trials
    op.create_table('optimization_trials',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('study_id', sa.Integer, sa.ForeignKey('optimization_studies.id'), nullable=False),

        # Deterministic TaskID for idempotency
        sa.Column('task_id', sa.String(64), nullable=False, unique=True, index=True),

        # Trial metadata
        sa.Column('trial_number', sa.Integer, nullable=False),
        sa.Column('status', sa.String(32), nullable=False, server_default='queued'),

        # Parameters being tested
        sa.Column('form_parameters', sa.JSON, nullable=False),
        sa.Column('action_parameters', sa.JSON, nullable=False),

        # Execution tracking
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('pruned_at', sa.DateTime, nullable=True),

        # Intermediate results for early stopping
        sa.Column('partial_trades_count', sa.Integer, server_default='0'),
        sa.Column('partial_success_rate', sa.Float, nullable=True),
        sa.Column('partial_expectancy', sa.Float, nullable=True),
        sa.Column('partial_duration_months', sa.Float, server_default='0.0'),

        # Error tracking
        sa.Column('error_message', sa.Text, nullable=True),

        # Resource tracking
        sa.Column('execution_time_seconds', sa.Float, nullable=True),
        sa.Column('worker_id', sa.String(64), nullable=True),

        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),
    )

    # Create indexes for trials
    op.create_index('idx_trial_study_status', 'optimization_trials', ['study_id', 'status'])
    op.create_index('idx_trial_task_id', 'optimization_trials', ['task_id'])
    op.create_index('idx_trial_number', 'optimization_trials', ['study_id', 'trial_number'])

    # trial_metrics table - performance metrics with multi-dataset support
    op.create_table('trial_metrics',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('trial_id', sa.Integer, sa.ForeignKey('optimization_trials.id'), nullable=False),

        # Dataset and regime context
        sa.Column('dataset_id', sa.String(32), nullable=False),
        sa.Column('regime_tag', sa.String(64), nullable=True),

        # Core performance metrics
        sa.Column('total_signals', sa.Integer, server_default='0'),
        sa.Column('successful_signals', sa.Integer, server_default='0'),
        sa.Column('success_rate', sa.Float, nullable=True),

        # Financial metrics
        sa.Column('total_return', sa.Float, server_default='0.0'),
        sa.Column('profit_factor', sa.Float, nullable=True),
        sa.Column('expectancy', sa.Float, nullable=True),
        sa.Column('max_drawdown', sa.Float, nullable=True),

        # Risk metrics
        sa.Column('sharpe_ratio', sa.Float, nullable=True),
        sa.Column('sortino_ratio', sa.Float, nullable=True),
        sa.Column('calmar_ratio', sa.Float, nullable=True),

        # Timing metrics
        sa.Column('avg_holding_period_hours', sa.Float, nullable=True),
        sa.Column('hit_rate_by_time', sa.JSON, nullable=True),

        # Stability metrics
        sa.Column('variance_across_blocks', sa.Float, nullable=True),
        sa.Column('consistency_score', sa.Float, nullable=True),

        # Temporal coverage
        sa.Column('first_signal_date', sa.DateTime, nullable=True),
        sa.Column('last_signal_date', sa.DateTime, nullable=True),
        sa.Column('temporal_coverage_months', sa.Float, nullable=True),

        # Recency-weighted metrics
        sa.Column('recency_weighted_success_rate', sa.Float, nullable=True),
        sa.Column('recency_weighted_expectancy', sa.Float, nullable=True),

        # Invalidation rule statistics
        sa.Column('avg_k_time_actual', sa.Float, nullable=True),
        sa.Column('avg_k_loss_actual', sa.Float, nullable=True),
        sa.Column('quantile_time_threshold', sa.Float, nullable=True),
        sa.Column('quantile_loss_threshold', sa.Float, nullable=True),

        # Multi-objective scores
        sa.Column('pareto_rank', sa.Integer, nullable=True),
        sa.Column('crowding_distance', sa.Float, nullable=True),
        sa.Column('combined_score', sa.Float, nullable=True),

        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),

        # Unique constraint for trial metrics context
        sa.UniqueConstraint('trial_id', 'dataset_id', 'regime_tag',
                           name='uq_trial_metrics_context')
    )

    # Create indexes for metrics
    op.create_index('idx_metrics_dataset', 'trial_metrics', ['dataset_id'])
    op.create_index('idx_metrics_regime', 'trial_metrics', ['regime_tag'])
    op.create_index('idx_metrics_success_rate', 'trial_metrics', ['success_rate'])
    op.create_index('idx_metrics_pareto', 'trial_metrics', ['pareto_rank', 'crowding_distance'])

    # best_parameters table - promoted parameter sets
    op.create_table('best_parameters',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('study_id', sa.Integer, sa.ForeignKey('optimization_studies.id'), nullable=False),

        # Override context (NULL for global best, specific regime for overrides)
        sa.Column('regime_tag', sa.String(64), nullable=True, index=True),

        # Strategy selection ("high_return", "low_risk", "balanced")
        sa.Column('strategy_tag', sa.String(32), nullable=True, index=True),

        # Best parameters
        sa.Column('form_parameters', sa.JSON, nullable=False),
        sa.Column('action_parameters', sa.JSON, nullable=False),

        # Performance summary
        sa.Column('best_trial_id', sa.Integer, sa.ForeignKey('optimization_trials.id'), nullable=False),
        sa.Column('combined_score', sa.Float, nullable=False),

        # Multi-objective breakdown
        sa.Column('d1_success_rate', sa.Float, nullable=True),
        sa.Column('d2_success_rate', sa.Float, nullable=True),
        sa.Column('d1_expectancy', sa.Float, nullable=True),
        sa.Column('d2_expectancy', sa.Float, nullable=True),

        # Validation metrics
        sa.Column('total_signals', sa.Integer, server_default='0'),
        sa.Column('temporal_coverage_months', sa.Float, nullable=True),
        sa.Column('robustness_score', sa.Float, nullable=True),

        # Promotion tracking
        sa.Column('is_promoted', sa.Boolean, server_default='0'),
        sa.Column('promoted_at', sa.DateTime, nullable=True),
        sa.Column('promoted_by', sa.String(128), nullable=True),

        # Versioning for rollback
        sa.Column('params_hash', sa.String(64), nullable=False, index=True),
        sa.Column('version', sa.Integer, server_default='1'),

        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),

        # Unique constraint for best parameters context
        sa.UniqueConstraint('study_id', 'regime_tag',
                           name='uq_best_params_context')
    )

    # Create indexes for best parameters
    op.create_index('idx_best_params_promoted', 'best_parameters', ['is_promoted'])
    op.create_index('idx_best_params_hash', 'best_parameters', ['params_hash'])

    # parameter_changelog table - audit trail
    op.create_table('parameter_changelog',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),

        # Change context
        sa.Column('pattern_key', sa.String(128), nullable=False, index=True),
        sa.Column('direction', sa.String(16), nullable=False),
        sa.Column('asset', sa.String(64), nullable=False),
        sa.Column('timeframe', sa.String(16), nullable=False),
        sa.Column('regime_tag', sa.String(64), nullable=True),

        # Change details
        sa.Column('action', sa.String(32), nullable=False),
        sa.Column('old_parameters', sa.JSON, nullable=True),
        sa.Column('new_parameters', sa.JSON, nullable=False),

        # Performance justification
        sa.Column('performance_improvement', sa.JSON, nullable=True),
        sa.Column('validation_metrics', sa.JSON, nullable=True),

        # Metadata
        sa.Column('changed_by', sa.String(128), nullable=False),
        sa.Column('reason', sa.Text, nullable=True),
        sa.Column('source_trial_id', sa.Integer, sa.ForeignKey('optimization_trials.id'), nullable=True),

        # Rollback tracking
        sa.Column('is_rollback', sa.Boolean, server_default='0'),
        sa.Column('rollback_of_change_id', sa.Integer, sa.ForeignKey('parameter_changelog.id'), nullable=True),

        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),
    )

    # Create indexes for changelog
    op.create_index('idx_changelog_pattern', 'parameter_changelog', ['pattern_key', 'direction'])
    op.create_index('idx_changelog_asset', 'parameter_changelog', ['asset', 'timeframe'])
    op.create_index('idx_changelog_action', 'parameter_changelog', ['action'])
    op.create_index('idx_changelog_date', 'parameter_changelog', ['created_at'])

    # regime_classifications table - economic/market regime mappings
    op.create_table('regime_classifications',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),

        # Time period
        sa.Column('start_date', sa.DateTime, nullable=False, index=True),
        sa.Column('end_date', sa.DateTime, nullable=False, index=True),

        # Regime tags
        sa.Column('regime_tag', sa.String(64), nullable=False, index=True),
        sa.Column('confidence', sa.Float, server_default='1.0'),

        # Source data and methodology
        sa.Column('source', sa.String(128), nullable=False),
        sa.Column('raw_value', sa.Float, nullable=True),
        sa.Column('threshold_used', sa.Float, nullable=True),

        # Market context
        sa.Column('asset_class', sa.String(32), nullable=True),

        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.current_timestamp()),
    )

    # Create indexes for regime classifications
    op.create_index('idx_regime_period', 'regime_classifications', ['start_date', 'end_date'])
    op.create_index('idx_regime_tag', 'regime_classifications', ['regime_tag'])
    op.create_index('idx_regime_source', 'regime_classifications', ['source'])

def downgrade():
    # Drop tables in reverse order due to foreign key constraints
    op.drop_table('regime_classifications')
    op.drop_table('parameter_changelog')
    op.drop_table('best_parameters')
    op.drop_table('trial_metrics')
    op.drop_table('optimization_trials')
    op.drop_table('optimization_studies')