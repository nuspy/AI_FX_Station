"""
Database Models for E2E Optimization System

Complete models for End-to-End parameter optimization across all trading components.
Stores optimization runs, parameters, results, regime-specific configurations, and deployments.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean, 
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class E2EOptimizationRun(Base):
    """
    Master record for each E2E optimization run.
    
    Tracks metadata and results for complete end-to-end optimization including
    SSSD, Riskfolio, Patterns, RL, and market filters.
    """
    __tablename__ = 'e2e_optimization_runs'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Unique identifier
    run_uuid = Column(String(36), unique=True, nullable=False, index=True)
    
    # Configuration
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    optimization_method = Column(String(20), nullable=False)  # 'bayesian' or 'genetic'
    regime_mode = Column(String(20), nullable=False)  # 'global' or 'per_regime'
    
    # Date range
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Optimization parameters
    n_trials = Column(Integer, nullable=False)  # Number of trials/generations
    
    # Status
    status = Column(String(20), nullable=False, default='pending')  # 'pending', 'running', 'completed', 'failed', 'stopped'
    
    # Best results
    best_sharpe = Column(Float)
    best_max_drawdown = Column(Float)
    best_win_rate = Column(Float)
    best_profit_factor = Column(Float)
    best_calmar_ratio = Column(Float)
    best_trial_number = Column(Integer)
    
    # Performance metrics
    total_duration_sec = Column(Integer)
    avg_trial_duration_sec = Column(Float)
    convergence_trial = Column(Integer)  # Trial where convergence detected
    
    # Component flags (what was optimized)
    optimize_sssd = Column(Boolean, default=False)
    optimize_riskfolio = Column(Boolean, default=False)
    optimize_patterns = Column(Boolean, default=False)
    optimize_rl = Column(Boolean, default=False)
    optimize_vix_filter = Column(Boolean, default=False)
    optimize_sentiment_filter = Column(Boolean, default=False)
    optimize_volume_filter = Column(Boolean, default=False)
    
    # Objectives configuration (JSON)
    objectives_config = Column(Text)  # JSON: {'sharpe': 0.5, 'max_dd': 0.3, 'pf': 0.15, 'cost': 0.05}
    
    # Constraints configuration (JSON)
    constraints_config = Column(Text)  # JSON: {'max_drawdown_pct': 15.0, 'min_sharpe': 1.0}
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Notes
    notes = Column(Text)
    error_message = Column(Text)
    
    # Relationships
    parameters = relationship('E2EOptimizationParameter', back_populates='run', cascade='all, delete-orphan')
    results = relationship('E2EOptimizationResult', back_populates='run', cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        Index('idx_e2e_run_symbol_timeframe', 'symbol', 'timeframe'),
        Index('idx_e2e_run_status', 'status'),
        Index('idx_e2e_run_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<E2EOptimizationRun(id={self.id}, uuid={self.run_uuid}, symbol={self.symbol}, status={self.status})>"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'run_uuid': self.run_uuid,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'optimization_method': self.optimization_method,
            'regime_mode': self.regime_mode,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'n_trials': self.n_trials,
            'status': self.status,
            'best_sharpe': self.best_sharpe,
            'best_max_drawdown': self.best_max_drawdown,
            'best_win_rate': self.best_win_rate,
            'best_profit_factor': self.best_profit_factor,
            'best_calmar_ratio': self.best_calmar_ratio,
            'best_trial_number': self.best_trial_number,
            'total_duration_sec': self.total_duration_sec,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'objectives_config': json.loads(self.objectives_config) if self.objectives_config else {},
            'constraints_config': json.loads(self.constraints_config) if self.constraints_config else {},
            'notes': self.notes,
        }


class E2EOptimizationParameter(Base):
    """
    Parameter values for each trial.
    
    Stores individual parameter values (90+ parameters) for each trial.
    Supports regime-specific parameters.
    """
    __tablename__ = 'e2e_optimization_parameters'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key
    run_id = Column(Integer, ForeignKey('e2e_optimization_runs.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Trial identification
    trial_number = Column(Integer, nullable=False, index=True)
    
    # Regime (for per-regime optimization)
    regime = Column(String(20), nullable=False, default='global', index=True)  # 'global', 'trending_up', 'trending_down', 'ranging', 'volatile'
    
    # Parameter details
    parameter_group = Column(String(50), nullable=False, index=True)  # 'sssd', 'riskfolio', 'patterns', 'rl', 'risk', 'sizing', 'filters'
    parameter_name = Column(String(100), nullable=False, index=True)
    parameter_value = Column(Text, nullable=False)  # JSON-serialized for complex types
    parameter_type = Column(String(20), nullable=False)  # 'int', 'float', 'str', 'bool', 'list', 'dict'
    
    # Timestamp
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationship
    run = relationship('E2EOptimizationRun', back_populates='parameters')
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_e2e_params_run_trial', 'run_id', 'trial_number'),
        Index('idx_e2e_params_regime', 'regime'),
        Index('idx_e2e_params_group', 'parameter_group'),
        UniqueConstraint('run_id', 'trial_number', 'regime', 'parameter_group', 'parameter_name', 
                        name='uix_e2e_params_unique'),
    )
    
    def __repr__(self):
        return f"<E2EOptimizationParameter(run_id={self.run_id}, trial={self.trial_number}, param={self.parameter_name})>"
    
    def get_value(self) -> Any:
        """Parse and return the actual parameter value"""
        if self.parameter_type == 'int':
            return int(self.parameter_value)
        elif self.parameter_type == 'float':
            return float(self.parameter_value)
        elif self.parameter_type == 'bool':
            return self.parameter_value.lower() == 'true'
        elif self.parameter_type == 'str':
            return self.parameter_value
        elif self.parameter_type in ('list', 'dict'):
            return json.loads(self.parameter_value)
        else:
            return self.parameter_value


class E2EOptimizationResult(Base):
    """
    Backtest results for each trial.
    
    Stores comprehensive performance metrics from integrated backtest.
    """
    __tablename__ = 'e2e_optimization_results'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key
    run_id = Column(Integer, ForeignKey('e2e_optimization_runs.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Trial identification
    trial_number = Column(Integer, nullable=False, index=True)
    
    # Regime (for per-regime results)
    regime = Column(String(20), default='global', index=True)
    
    # Performance metrics
    total_return = Column(Float, nullable=False, default=0.0)
    total_return_pct = Column(Float, nullable=False, default=0.0)
    sharpe_ratio = Column(Float, nullable=False, default=0.0, index=True)
    sortino_ratio = Column(Float, nullable=False, default=0.0)
    calmar_ratio = Column(Float, nullable=False, default=0.0)
    max_drawdown = Column(Float, nullable=False, default=0.0)
    max_drawdown_pct = Column(Float, nullable=False, default=0.0, index=True)
    
    # Trade statistics
    win_rate = Column(Float, nullable=False, default=0.0, index=True)
    profit_factor = Column(Float, nullable=False, default=0.0, index=True)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    avg_win = Column(Float, nullable=False, default=0.0)
    avg_loss = Column(Float, nullable=False, default=0.0)
    expectancy = Column(Float, nullable=False, default=0.0)
    
    # Cost analysis
    total_costs = Column(Float, nullable=False, default=0.0)
    avg_cost_per_trade = Column(Float, nullable=False, default=0.0)
    
    # Time metrics
    avg_holding_time_hrs = Column(Float, nullable=False, default=0.0)
    
    # Multi-objective score
    objective_value = Column(Float, nullable=False, default=0.0)  # Combined multi-objective score
    is_pareto_optimal = Column(Boolean, default=False, index=True)
    
    # Additional metrics (JSON)
    additional_metrics = Column(Text)  # JSON: {'var_95': -0.02, 'cvar_95': -0.03, etc.}
    
    # Timestamp
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationship
    run = relationship('E2EOptimizationRun', back_populates='results')
    
    # Indexes
    __table_args__ = (
        Index('idx_e2e_results_run_trial', 'run_id', 'trial_number'),
        Index('idx_e2e_results_sharpe', 'sharpe_ratio'),
        Index('idx_e2e_results_drawdown', 'max_drawdown_pct'),
        Index('idx_e2e_results_pareto', 'is_pareto_optimal'),
    )
    
    def __repr__(self):
        return f"<E2EOptimizationResult(run_id={self.run_id}, trial={self.trial_number}, sharpe={self.sharpe_ratio:.2f})>"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'run_id': self.run_id,
            'trial_number': self.trial_number,
            'regime': self.regime,
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'expectancy': self.expectancy,
            'total_costs': self.total_costs,
            'avg_cost_per_trade': self.avg_cost_per_trade,
            'avg_holding_time_hrs': self.avg_holding_time_hrs,
            'objective_value': self.objective_value,
            'is_pareto_optimal': self.is_pareto_optimal,
            'additional_metrics': json.loads(self.additional_metrics) if self.additional_metrics else {},
        }


class E2ERegimeParameter(Base):
    """
    Best parameter sets per regime.
    
    Stores optimized parameters for specific (symbol, timeframe, regime) combinations.
    Used for deployment to live trading.
    """
    __tablename__ = 'e2e_regime_parameters'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    regime = Column(String(20), nullable=False, index=True)  # 'trending_up', 'trending_down', 'ranging', 'volatile', 'global'
    
    # Link to optimization run
    optimization_run_id = Column(Integer, ForeignKey('e2e_optimization_runs.id'), nullable=False)
    trial_number = Column(Integer, nullable=False)
    
    # Parameters (complete set as JSON)
    parameters_json = Column(Text, nullable=False)  # JSON: full parameter dict
    
    # Performance metrics (from backtest)
    sharpe_ratio = Column(Float, nullable=False, index=True)
    max_drawdown_pct = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=False)
    calmar_ratio = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    
    # Deployment status
    is_active = Column(Boolean, default=False, index=True)
    activated_at = Column(DateTime)
    deactivated_at = Column(DateTime)
    
    # Validation metrics (out-of-sample)
    oos_sharpe_ratio = Column(Float)
    oos_max_drawdown_pct = Column(Float)
    oos_win_rate = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_e2e_regime_params_active', 'symbol', 'timeframe', 'regime', 'is_active'),
        Index('idx_e2e_regime_params_sharpe', 'sharpe_ratio'),
        UniqueConstraint('symbol', 'timeframe', 'regime', 'is_active',
                        name='uix_e2e_regime_params_active',
                        sqlite_where='is_active = 1'),  # Only one active per (symbol, timeframe, regime)
    )
    
    def __repr__(self):
        return f"<E2ERegimeParameter(id={self.id}, symbol={self.symbol}, timeframe={self.timeframe}, regime={self.regime}, active={self.is_active})>"
    
    def get_parameters(self) -> Dict:
        """Parse and return parameters dictionary"""
        return json.loads(self.parameters_json) if self.parameters_json else {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'regime': self.regime,
            'optimization_run_id': self.optimization_run_id,
            'trial_number': self.trial_number,
            'parameters': self.get_parameters(),
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'is_active': self.is_active,
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
            'deactivated_at': self.deactivated_at.isoformat() if self.deactivated_at else None,
            'oos_sharpe_ratio': self.oos_sharpe_ratio,
            'oos_max_drawdown_pct': self.oos_max_drawdown_pct,
            'oos_win_rate': self.oos_win_rate,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class E2EDeploymentConfig(Base):
    """
    Deployment configuration tracking.
    
    Tracks which parameter sets are deployed to live trading and their performance.
    """
    __tablename__ = 'e2e_deployment_configs'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    
    # Deployment mode
    deployment_mode = Column(String(20), nullable=False)  # 'global', 'per_regime', 'manual'
    
    # Parameter references
    global_params_id = Column(Integer, ForeignKey('e2e_regime_parameters.id'))  # For global mode
    regime_params_mapping = Column(Text)  # JSON: {'trending_up': 123, 'ranging': 124, ...}
    
    # Deployment metadata
    deployed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    deployed_by = Column(String(100))  # user or 'system'
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    deactivated_at = Column(DateTime)
    deactivation_reason = Column(Text)
    
    # Performance tracking (live vs backtest)
    performance_metrics = Column(Text)  # JSON: {'expected_sharpe': 1.5, 'actual_sharpe': 1.4, 'deviation_pct': -6.7}
    
    # Alerts
    performance_alert = Column(Boolean, default=False)
    alert_message = Column(Text)
    alert_triggered_at = Column(DateTime)
    
    # Notes
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_e2e_deployment_active', 'symbol', 'timeframe', 'is_active'),
        Index('idx_e2e_deployment_date', 'deployed_at'),
        UniqueConstraint('symbol', 'timeframe', 'is_active',
                        name='uix_e2e_deployment_active',
                        sqlite_where='is_active = 1'),  # Only one active deployment per (symbol, timeframe)
    )
    
    def __repr__(self):
        return f"<E2EDeploymentConfig(id={self.id}, symbol={self.symbol}, timeframe={self.timeframe}, mode={self.deployment_mode}, active={self.is_active})>"
    
    def get_regime_params_mapping(self) -> Dict[str, int]:
        """Parse and return regime parameters mapping"""
        return json.loads(self.regime_params_mapping) if self.regime_params_mapping else {}
    
    def get_performance_metrics(self) -> Dict:
        """Parse and return performance metrics"""
        return json.loads(self.performance_metrics) if self.performance_metrics else {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'deployment_mode': self.deployment_mode,
            'global_params_id': self.global_params_id,
            'regime_params_mapping': self.get_regime_params_mapping(),
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'deployed_by': self.deployed_by,
            'is_active': self.is_active,
            'deactivated_at': self.deactivated_at.isoformat() if self.deactivated_at else None,
            'deactivation_reason': self.deactivation_reason,
            'performance_metrics': self.get_performance_metrics(),
            'performance_alert': self.performance_alert,
            'alert_message': self.alert_message,
            'alert_triggered_at': self.alert_triggered_at.isoformat() if self.alert_triggered_at else None,
            'notes': self.notes,
        }
