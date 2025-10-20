# src/forex_diffusion/training_pipeline/database_models.py
"""
SQLAlchemy ORM models for new training system tables.
"""

from __future__ import annotations
from typing import Dict, Any
import json

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey, Text,
    UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class TrainingRun(Base):
    """Represents a single model training run."""
    
    __tablename__ = 'training_runs'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    run_uuid = Column(String(36), unique=True, nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # pending, running, completed, failed, cancelled
    
    # Model Configuration (Mutually Exclusive Parameters)
    model_type = Column(String(50), nullable=False, index=True)
    encoder = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    base_timeframe = Column(String(10), nullable=False)
    days_history = Column(Integer, nullable=False)
    horizon = Column(Integer, nullable=False)
    
    # Feature Configuration (Combinable Parameters - JSON)
    indicator_tfs = Column(JSON, nullable=True)  # {"rsi": ["5m", "15m"], ...}
    additional_features = Column(JSON, nullable=True)  # {"returns": true, ...}
    preprocessing_params = Column(JSON, nullable=True)  # All preprocessing hyperparams
    model_hyperparams = Column(JSON, nullable=True)  # Model-specific params
    
    # Training Results
    training_metrics = Column(JSON, nullable=True)  # MAE, RMSE, R2, etc.
    feature_count = Column(Integer, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    
    # File Management
    model_file_path = Column(String(500), nullable=True)
    model_file_size_bytes = Column(Integer, nullable=True)
    is_model_kept = Column(Boolean, default=False)
    
    # Regime Performance
    best_regimes = Column(JSON, nullable=True)  # ["bull", "volatile"]
    
    # Timestamps
    created_at = Column(DateTime, default=func.current_timestamp())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Provenance
    created_by = Column(String(50), default='system')  # 'ui', 'api', 'scheduled'
    config_hash = Column(String(64), nullable=False, index=True)
    
    # Relationships
    inference_backtests = relationship(
        "InferenceBacktest",
        back_populates="training_run",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_tr_symbol_timeframe', 'symbol', 'base_timeframe'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'run_uuid': self.run_uuid,
            'status': self.status,
            'model_type': self.model_type,
            'encoder': self.encoder,
            'symbol': self.symbol,
            'base_timeframe': self.base_timeframe,
            'days_history': self.days_history,
            'horizon': self.horizon,
            'indicator_tfs': self.indicator_tfs,
            'additional_features': self.additional_features,
            'preprocessing_params': self.preprocessing_params,
            'model_hyperparams': self.model_hyperparams,
            'training_metrics': self.training_metrics,
            'feature_count': self.feature_count,
            'training_duration_seconds': self.training_duration_seconds,
            'model_file_path': self.model_file_path,
            'model_file_size_bytes': self.model_file_size_bytes,
            'is_model_kept': self.is_model_kept,
            'best_regimes': self.best_regimes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_by': self.created_by,
            'config_hash': self.config_hash
        }


class InferenceBacktest(Base):
    """Represents an inference backtest on a trained model."""
    
    __tablename__ = 'inference_backtests'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    backtest_uuid = Column(String(36), unique=True, nullable=False)
    training_run_id = Column(Integer, ForeignKey('training_runs.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Inference Configuration (Internal Loop Parameters)
    prediction_method = Column(String(50), nullable=False, index=True)  # direct, recursive, direct_multi
    ensemble_method = Column(String(50), nullable=True)  # mean, weighted, stacking
    confidence_threshold = Column(Float, nullable=True)
    lookback_window = Column(Integer, nullable=True)
    inference_params = Column(JSON, nullable=True)  # Other inference-specific params
    
    # Backtest Results
    backtest_metrics = Column(JSON, nullable=True)  # Sharpe, max_drawdown, win_rate, etc.
    backtest_duration_seconds = Column(Float, nullable=True)
    regime_metrics = Column(JSON, nullable=True)  # Performance by regime
    
    # Timestamps
    created_at = Column(DateTime, default=func.current_timestamp())
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    training_run = relationship("TrainingRun", back_populates="inference_backtests")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'backtest_uuid': self.backtest_uuid,
            'training_run_id': self.training_run_id,
            'prediction_method': self.prediction_method,
            'ensemble_method': self.ensemble_method,
            'confidence_threshold': self.confidence_threshold,
            'lookback_window': self.lookback_window,
            'inference_params': self.inference_params,
            'backtest_metrics': self.backtest_metrics,
            'backtest_duration_seconds': self.backtest_duration_seconds,
            'regime_metrics': self.regime_metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class RegimeDefinition(Base):
    """Defines a market regime for performance tracking."""
    
    __tablename__ = 'regime_definitions'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    regime_name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    detection_rules = Column(JSON, nullable=True)  # Rules for regime detection
    created_at = Column(DateTime, default=func.current_timestamp())
    is_active = Column(Boolean, default=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'regime_name': self.regime_name,
            'description': self.description,
            'detection_rules': self.detection_rules,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_active': self.is_active
        }


class RegimeBestModel(Base):
    """Tracks the best performing model for each regime."""
    
    __tablename__ = 'regime_best_models'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    regime_name = Column(String(50), ForeignKey('regime_definitions.regime_name', ondelete='CASCADE'), nullable=False)
    training_run_id = Column(Integer, ForeignKey('training_runs.id', ondelete='CASCADE'), nullable=False)
    inference_backtest_id = Column(Integer, ForeignKey('inference_backtests.id', ondelete='CASCADE'), nullable=False)
    
    # Performance Metrics
    performance_score = Column(Float, nullable=False)  # Primary metric (e.g., Sharpe ratio)
    secondary_metrics = Column(JSON, nullable=True)  # Other metrics
    
    # Timestamp
    achieved_at = Column(DateTime, default=func.current_timestamp())
    
    # Unique constraint: only one best model per regime
    __table_args__ = (
        UniqueConstraint('regime_name', name='uq_regime_best_model'),
        Index('idx_rbm_regime', 'regime_name'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'regime_name': self.regime_name,
            'training_run_id': self.training_run_id,
            'inference_backtest_id': self.inference_backtest_id,
            'performance_score': self.performance_score,
            'secondary_metrics': self.secondary_metrics,
            'achieved_at': self.achieved_at.isoformat() if self.achieved_at else None
        }


class TrainingQueue(Base):
    """Manages a queue of training configurations."""
    
    __tablename__ = 'training_queue'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    queue_uuid = Column(String(36), unique=True, nullable=False)
    
    # Configuration
    config_grid = Column(JSON, nullable=False)  # Full grid of configurations
    current_index = Column(Integer, default=0)  # Current position in grid
    total_configs = Column(Integer, nullable=False)
    
    # Status
    status = Column(String(20), nullable=False, index=True)  # pending, running, paused, completed, cancelled
    
    # Progress Tracking
    completed_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    skipped_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.current_timestamp())
    started_at = Column(DateTime, nullable=True)
    paused_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Settings
    priority = Column(Integer, default=0)  # Higher = more important
    max_parallel = Column(Integer, default=1)  # Max parallel training jobs
    
    __table_args__ = (
        Index('idx_tq_status', 'status'),
        Index('idx_tq_priority', 'priority'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'queue_uuid': self.queue_uuid,
            'config_grid': self.config_grid,
            'current_index': self.current_index,
            'total_configs': self.total_configs,
            'status': self.status,
            'completed_count': self.completed_count,
            'failed_count': self.failed_count,
            'skipped_count': self.skipped_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'paused_at': self.paused_at.isoformat() if self.paused_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'priority': self.priority,
            'max_parallel': self.max_parallel
        }


class OptimizedParameters(Base):
    """Stores backtesting-optimized parameters per pattern/symbol/regime."""

    __tablename__ = 'optimized_parameters'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identification
    pattern_type = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    market_regime = Column(String(50), nullable=True, index=True)

    # Parameters (JSON format)
    form_params = Column(Text, nullable=False)  # Pattern-specific form parameters
    action_params = Column(Text, nullable=False)  # Entry/exit parameters
    performance_metrics = Column(Text, nullable=False)  # Performance from optimization

    # Optimization metadata
    optimization_timestamp = Column(DateTime, nullable=False, index=True)
    data_range_start = Column(DateTime, nullable=False)
    data_range_end = Column(DateTime, nullable=False)
    sample_count = Column(Integer, nullable=False)

    # Validation
    validation_status = Column(String(20), nullable=False, server_default='pending', index=True)
    validation_metrics = Column(Text, nullable=True)
    deployment_date = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = Column(DateTime, nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_opt_params_pattern_symbol_tf', 'pattern_type', 'symbol', 'timeframe'),
        Index('idx_opt_params_regime', 'market_regime'),
        Index('idx_opt_params_validation', 'validation_status'),
        Index('idx_opt_params_timestamp', 'optimization_timestamp'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'pattern_type': self.pattern_type,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'market_regime': self.market_regime,
            'form_params': json.loads(self.form_params) if self.form_params else None,
            'action_params': json.loads(self.action_params) if self.action_params else None,
            'performance_metrics': json.loads(self.performance_metrics) if self.performance_metrics else None,
            'optimization_timestamp': self.optimization_timestamp.isoformat() if self.optimization_timestamp else None,
            'data_range_start': self.data_range_start.isoformat() if self.data_range_start else None,
            'data_range_end': self.data_range_end.isoformat() if self.data_range_end else None,
            'sample_count': self.sample_count,
            'validation_status': self.validation_status,
            'validation_metrics': json.loads(self.validation_metrics) if self.validation_metrics else None,
            'deployment_date': self.deployment_date.isoformat() if self.deployment_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class RiskProfile(Base):
    """Defines risk management profiles (predefined and custom)."""

    __tablename__ = 'risk_profiles'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identification
    profile_name = Column(String(50), nullable=False, unique=True, index=True)
    profile_type = Column(String(20), nullable=False)  # predefined or custom
    is_active = Column(Boolean, nullable=False, server_default='0', index=True)

    # Position sizing
    max_risk_per_trade_pct = Column(Float, nullable=False)
    max_portfolio_risk_pct = Column(Float, nullable=False)
    position_sizing_method = Column(String(20), nullable=False)  # fixed_fractional, kelly, optimal_f, volatility_adjusted
    kelly_fraction = Column(Float, nullable=True)

    # Stop loss/Take profit
    base_sl_atr_multiplier = Column(Float, nullable=False)
    base_tp_atr_multiplier = Column(Float, nullable=False)
    use_trailing_stop = Column(Boolean, nullable=False, server_default='1')
    trailing_activation_pct = Column(Float, nullable=True)

    # Adaptive adjustments
    regime_adjustment_enabled = Column(Boolean, nullable=False, server_default='1')
    volatility_adjustment_enabled = Column(Boolean, nullable=False, server_default='1')
    news_awareness_enabled = Column(Boolean, nullable=False, server_default='1')

    # Diversification
    max_correlated_positions = Column(Integer, nullable=False)
    correlation_threshold = Column(Float, nullable=False)
    max_positions_per_symbol = Column(Integer, nullable=False)
    max_total_positions = Column(Integer, nullable=False)

    # Drawdown protection
    max_daily_loss_pct = Column(Float, nullable=False)
    max_drawdown_pct = Column(Float, nullable=False)
    recovery_mode_threshold_pct = Column(Float, nullable=False)
    recovery_risk_multiplier = Column(Float, nullable=False)

    # Metadata
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = Column(DateTime, nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_risk_profile_name', 'profile_name'),
        Index('idx_risk_profile_active', 'is_active'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'profile_name': self.profile_name,
            'profile_type': self.profile_type,
            'is_active': self.is_active,
            'max_risk_per_trade_pct': self.max_risk_per_trade_pct,
            'max_portfolio_risk_pct': self.max_portfolio_risk_pct,
            'position_sizing_method': self.position_sizing_method,
            'kelly_fraction': self.kelly_fraction,
            'base_sl_atr_multiplier': self.base_sl_atr_multiplier,
            'base_tp_atr_multiplier': self.base_tp_atr_multiplier,
            'use_trailing_stop': self.use_trailing_stop,
            'trailing_activation_pct': self.trailing_activation_pct,
            'regime_adjustment_enabled': self.regime_adjustment_enabled,
            'volatility_adjustment_enabled': self.volatility_adjustment_enabled,
            'news_awareness_enabled': self.news_awareness_enabled,
            'max_correlated_positions': self.max_correlated_positions,
            'correlation_threshold': self.correlation_threshold,
            'max_positions_per_symbol': self.max_positions_per_symbol,
            'max_total_positions': self.max_total_positions,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'recovery_mode_threshold_pct': self.recovery_mode_threshold_pct,
            'recovery_risk_multiplier': self.recovery_risk_multiplier,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class AdvancedMetrics(Base):
    """Stores advanced performance metrics beyond basic backtest results."""

    __tablename__ = 'advanced_metrics'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Type and reference
    metric_type = Column(String(20), nullable=False, index=True)  # backtest, live, paper, validation
    reference_id = Column(Integer, nullable=True, index=True)  # ID of related backtest/optimization
    symbol = Column(String(20), nullable=True, index=True)
    timeframe = Column(String(10), nullable=True, index=True)
    market_regime = Column(String(50), nullable=True)

    # Time period
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)

    # Basic metrics (for reference)
    total_return_pct = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    win_rate_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)

    # Advanced risk-adjusted metrics
    sortino_ratio = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    mar_ratio = Column(Float, nullable=True)
    omega_ratio = Column(Float, nullable=True)
    gain_to_pain_ratio = Column(Float, nullable=True)

    # Drawdown metrics
    max_drawdown_pct = Column(Float, nullable=True)
    max_drawdown_duration_days = Column(Integer, nullable=True)
    avg_drawdown_pct = Column(Float, nullable=True)
    recovery_time_days = Column(Integer, nullable=True)
    ulcer_index = Column(Float, nullable=True)

    # Return distribution
    return_skewness = Column(Float, nullable=True)
    return_kurtosis = Column(Float, nullable=True)
    var_95_pct = Column(Float, nullable=True)
    cvar_95_pct = Column(Float, nullable=True)

    # Win/Loss analysis
    avg_win_pct = Column(Float, nullable=True)
    avg_loss_pct = Column(Float, nullable=True)
    largest_win_pct = Column(Float, nullable=True)
    largest_loss_pct = Column(Float, nullable=True)
    win_loss_ratio = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)

    # Consistency metrics
    win_streak_max = Column(Integer, nullable=True)
    loss_streak_max = Column(Integer, nullable=True)
    monthly_win_rate_pct = Column(Float, nullable=True)
    expectancy_per_trade = Column(Float, nullable=True)

    # System quality
    system_quality_number = Column(Float, nullable=True)
    k_ratio = Column(Float, nullable=True)

    # Additional data
    extra_metrics = Column(Text, nullable=True)  # JSON for custom metrics

    # Metadata
    calculated_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_adv_metrics_type', 'metric_type'),
        Index('idx_adv_metrics_reference', 'reference_id'),
        Index('idx_adv_metrics_symbol_tf', 'symbol', 'timeframe'),
        Index('idx_adv_metrics_period', 'period_start', 'period_end'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'metric_type': self.metric_type,
            'reference_id': self.reference_id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'market_regime': self.market_regime,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
            'total_return_pct': self.total_return_pct,
            'total_trades': self.total_trades,
            'win_rate_pct': self.win_rate_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'mar_ratio': self.mar_ratio,
            'omega_ratio': self.omega_ratio,
            'gain_to_pain_ratio': self.gain_to_pain_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_drawdown_duration_days': self.max_drawdown_duration_days,
            'avg_drawdown_pct': self.avg_drawdown_pct,
            'recovery_time_days': self.recovery_time_days,
            'ulcer_index': self.ulcer_index,
            'return_skewness': self.return_skewness,
            'return_kurtosis': self.return_kurtosis,
            'var_95_pct': self.var_95_pct,
            'cvar_95_pct': self.cvar_95_pct,
            'avg_win_pct': self.avg_win_pct,
            'avg_loss_pct': self.avg_loss_pct,
            'largest_win_pct': self.largest_win_pct,
            'largest_loss_pct': self.largest_loss_pct,
            'win_loss_ratio': self.win_loss_ratio,
            'profit_factor': self.profit_factor,
            'win_streak_max': self.win_streak_max,
            'loss_streak_max': self.loss_streak_max,
            'monthly_win_rate_pct': self.monthly_win_rate_pct,
            'expectancy_per_trade': self.expectancy_per_trade,
            'system_quality_number': self.system_quality_number,
            'k_ratio': self.k_ratio,
            'extra_metrics': json.loads(self.extra_metrics) if self.extra_metrics else None,
            'calculated_at': self.calculated_at.isoformat() if self.calculated_at else None,
        }
