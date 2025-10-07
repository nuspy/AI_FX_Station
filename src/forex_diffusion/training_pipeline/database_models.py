# src/forex_diffusion/training_pipeline/database_models.py
"""
SQLAlchemy ORM models for new training system tables.
"""

from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional, Any
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
