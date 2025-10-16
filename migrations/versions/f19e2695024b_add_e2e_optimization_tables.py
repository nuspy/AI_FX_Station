"""add_e2e_optimization_tables

Revision ID: f19e2695024b
Revises: 94ca081433e4
Create Date: 2025-01-08 19:47:28.503691

E2E Optimization System Database Schema

Creates 5 new tables for comprehensive end-to-end parameter optimization:
1. e2e_optimization_runs - Master optimization run records
2. e2e_optimization_parameters - Parameter values per trial (90+ parameters)
3. e2e_optimization_results - Backtest results per trial
4. e2e_regime_parameters - Best parameters per regime (deployment-ready)
5. e2e_deployment_configs - Active deployments tracking
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f19e2695024b'
down_revision = '94ca081433e4'
branch_labels = None
depends_on = None


def upgrade():
    """Create E2E optimization tables"""
    
    # Table 1: e2e_optimization_runs
    op.create_table(
        'e2e_optimization_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_uuid', sa.String(length=36), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('optimization_method', sa.String(length=20), nullable=False),
        sa.Column('regime_mode', sa.String(length=20), nullable=False),
        sa.Column('start_date', sa.DateTime(), nullable=False),
        sa.Column('end_date', sa.DateTime(), nullable=False),
        sa.Column('n_trials', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('best_sharpe', sa.Float(), nullable=True),
        sa.Column('best_max_drawdown', sa.Float(), nullable=True),
        sa.Column('best_win_rate', sa.Float(), nullable=True),
        sa.Column('best_profit_factor', sa.Float(), nullable=True),
        sa.Column('best_calmar_ratio', sa.Float(), nullable=True),
        sa.Column('best_trial_number', sa.Integer(), nullable=True),
        sa.Column('total_duration_sec', sa.Integer(), nullable=True),
        sa.Column('avg_trial_duration_sec', sa.Float(), nullable=True),
        sa.Column('convergence_trial', sa.Integer(), nullable=True),
        sa.Column('optimize_sssd', sa.Boolean(), server_default='0'),
        sa.Column('optimize_riskfolio', sa.Boolean(), server_default='0'),
        sa.Column('optimize_patterns', sa.Boolean(), server_default='0'),
        sa.Column('optimize_rl', sa.Boolean(), server_default='0'),
        sa.Column('optimize_vix_filter', sa.Boolean(), server_default='0'),
        sa.Column('optimize_sentiment_filter', sa.Boolean(), server_default='0'),
        sa.Column('optimize_volume_filter', sa.Boolean(), server_default='0'),
        sa.Column('objectives_config', sa.Text(), nullable=True),
        sa.Column('constraints_config', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_uuid')
    )
    
    # Indexes for e2e_optimization_runs
    op.create_index('idx_e2e_run_symbol_timeframe', 'e2e_optimization_runs', ['symbol', 'timeframe'])
    op.create_index('idx_e2e_run_status', 'e2e_optimization_runs', ['status'])
    op.create_index('idx_e2e_run_created_at', 'e2e_optimization_runs', ['created_at'])
    op.create_index('idx_e2e_run_uuid', 'e2e_optimization_runs', ['run_uuid'])
    
    # Table 2: e2e_optimization_parameters
    op.create_table(
        'e2e_optimization_parameters',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('trial_number', sa.Integer(), nullable=False),
        sa.Column('regime', sa.String(length=20), nullable=False, server_default='global'),
        sa.Column('parameter_group', sa.String(length=50), nullable=False),
        sa.Column('parameter_name', sa.String(length=100), nullable=False),
        sa.Column('parameter_value', sa.Text(), nullable=False),
        sa.Column('parameter_type', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['run_id'], ['e2e_optimization_runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id', 'trial_number', 'regime', 'parameter_group', 'parameter_name', 
                           name='uix_e2e_params_unique')
    )
    
    # Indexes for e2e_optimization_parameters
    op.create_index('idx_e2e_params_run_trial', 'e2e_optimization_parameters', ['run_id', 'trial_number'])
    op.create_index('idx_e2e_params_regime', 'e2e_optimization_parameters', ['regime'])
    op.create_index('idx_e2e_params_group', 'e2e_optimization_parameters', ['parameter_group'])
    op.create_index('idx_e2e_params_run_id', 'e2e_optimization_parameters', ['run_id'])
    
    # Table 3: e2e_optimization_results
    op.create_table(
        'e2e_optimization_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('trial_number', sa.Integer(), nullable=False),
        sa.Column('regime', sa.String(length=20), server_default='global'),
        sa.Column('total_return', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('total_return_pct', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('sharpe_ratio', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('sortino_ratio', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('calmar_ratio', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('max_drawdown', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('max_drawdown_pct', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('win_rate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('profit_factor', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('total_trades', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('winning_trades', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('losing_trades', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_win', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('avg_loss', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('expectancy', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('total_costs', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('avg_cost_per_trade', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('avg_holding_time_hrs', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('objective_value', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('is_pareto_optimal', sa.Boolean(), server_default='0'),
        sa.Column('additional_metrics', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['run_id'], ['e2e_optimization_runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Indexes for e2e_optimization_results
    op.create_index('idx_e2e_results_run_trial', 'e2e_optimization_results', ['run_id', 'trial_number'])
    op.create_index('idx_e2e_results_sharpe', 'e2e_optimization_results', ['sharpe_ratio'])
    op.create_index('idx_e2e_results_drawdown', 'e2e_optimization_results', ['max_drawdown_pct'])
    op.create_index('idx_e2e_results_pareto', 'e2e_optimization_results', ['is_pareto_optimal'])
    op.create_index('idx_e2e_results_run_id', 'e2e_optimization_results', ['run_id'])
    
    # Table 4: e2e_regime_parameters
    op.create_table(
        'e2e_regime_parameters',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('regime', sa.String(length=20), nullable=False),
        sa.Column('optimization_run_id', sa.Integer(), nullable=False),
        sa.Column('trial_number', sa.Integer(), nullable=False),
        sa.Column('parameters_json', sa.Text(), nullable=False),
        sa.Column('sharpe_ratio', sa.Float(), nullable=False),
        sa.Column('max_drawdown_pct', sa.Float(), nullable=False),
        sa.Column('win_rate', sa.Float(), nullable=False),
        sa.Column('profit_factor', sa.Float(), nullable=False),
        sa.Column('calmar_ratio', sa.Float(), nullable=False),
        sa.Column('total_trades', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='0'),
        sa.Column('activated_at', sa.DateTime(), nullable=True),
        sa.Column('deactivated_at', sa.DateTime(), nullable=True),
        sa.Column('oos_sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('oos_max_drawdown_pct', sa.Float(), nullable=True),
        sa.Column('oos_win_rate', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['optimization_run_id'], ['e2e_optimization_runs.id']),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Indexes for e2e_regime_parameters
    op.create_index('idx_e2e_regime_params_active', 'e2e_regime_parameters', ['symbol', 'timeframe', 'regime', 'is_active'])
    op.create_index('idx_e2e_regime_params_sharpe', 'e2e_regime_parameters', ['sharpe_ratio'])
    op.create_index('idx_e2e_regime_params_symbol', 'e2e_regime_parameters', ['symbol'])
    op.create_index('idx_e2e_regime_params_timeframe', 'e2e_regime_parameters', ['timeframe'])
    op.create_index('idx_e2e_regime_params_regime', 'e2e_regime_parameters', ['regime'])
    
    # Table 5: e2e_deployment_configs
    op.create_table(
        'e2e_deployment_configs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('deployment_mode', sa.String(length=20), nullable=False),
        sa.Column('global_params_id', sa.Integer(), nullable=True),
        sa.Column('regime_params_mapping', sa.Text(), nullable=True),
        sa.Column('deployed_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deployed_by', sa.String(length=100), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='1'),
        sa.Column('deactivated_at', sa.DateTime(), nullable=True),
        sa.Column('deactivation_reason', sa.Text(), nullable=True),
        sa.Column('performance_metrics', sa.Text(), nullable=True),
        sa.Column('performance_alert', sa.Boolean(), server_default='0'),
        sa.Column('alert_message', sa.Text(), nullable=True),
        sa.Column('alert_triggered_at', sa.DateTime(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['global_params_id'], ['e2e_regime_parameters.id']),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Indexes for e2e_deployment_configs
    op.create_index('idx_e2e_deployment_active', 'e2e_deployment_configs', ['symbol', 'timeframe', 'is_active'])
    op.create_index('idx_e2e_deployment_date', 'e2e_deployment_configs', ['deployed_at'])
    op.create_index('idx_e2e_deployment_symbol', 'e2e_deployment_configs', ['symbol'])
    op.create_index('idx_e2e_deployment_timeframe', 'e2e_deployment_configs', ['timeframe'])


def downgrade():
    """Drop E2E optimization tables"""
    
    # Drop tables in reverse order (respect foreign keys)
    op.drop_table('e2e_deployment_configs')
    op.drop_table('e2e_regime_parameters')
    op.drop_table('e2e_optimization_results')
    op.drop_table('e2e_optimization_parameters')
    op.drop_table('e2e_optimization_runs')
