"""
Database models for Optimization Studies

Stores optimization results for parameter refresh system.
Part of PROC-002 implementation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class OptimizationStudy(Base):
    """
    Optimization study results.
    
    Tracks optimization runs for different pattern/asset/timeframe combinations.
    """
    __tablename__ = 'optimization_studies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Pattern identification
    pattern_key = Column(String(100), nullable=False, index=True)
    asset = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    # Best parameters found
    best_parameters = Column(JSON, nullable=False)  # Dict of parameter values
    
    # Performance metrics
    performance = Column(JSON, nullable=False)  # Dict with sharpe, win_rate, etc.
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    
    # Optimization metadata
    algorithm = Column(String(50))  # 'nsga2', 'grid_search', etc.
    n_trials = Column(Integer)
    n_generations = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Status
    status = Column(String(20), default='active')  # 'active', 'outdated', 'archived'
    needs_refresh = Column(Boolean, default=False)
    
    # Relationships
    outcomes = relationship('PatternOutcome', back_populates='study', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<OptimizationStudy(id={self.id}, pattern={self.pattern_key}, asset={self.asset}, timeframe={self.timeframe})>"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'pattern_key': self.pattern_key,
            'asset': self.asset,
            'timeframe': self.timeframe,
            'best_parameters': self.best_parameters,
            'performance': self.performance,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'algorithm': self.algorithm,
            'n_trials': self.n_trials,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'status': self.status,
            'needs_refresh': self.needs_refresh
        }


class PatternOutcome(Base):
    """
    Individual pattern detection outcomes.
    
    Tracks real-world performance of detected patterns.
    """
    __tablename__ = 'pattern_outcomes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Link to study
    study_id = Column(Integer, ForeignKey('optimization_studies.id'), nullable=False, index=True)
    
    # Detection details
    detection_time = Column(DateTime, nullable=False, index=True)
    asset = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    # Prediction details
    predicted_direction = Column(String(10))  # 'long', 'short'
    predicted_target = Column(Float)
    predicted_stop = Column(Float)
    confidence_score = Column(Float)
    
    # Actual outcome
    success = Column(Boolean)
    actual_high = Column(Float)
    actual_low = Column(Float)
    pnl = Column(Float)
    pnl_percentage = Column(Float)
    
    # Exit details
    exit_time = Column(DateTime)
    exit_reason = Column(String(50))  # 'target', 'stop', 'timeout'
    bars_held = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationship
    study = relationship('OptimizationStudy', back_populates='outcomes')
    
    def __repr__(self):
        return f"<PatternOutcome(id={self.id}, study_id={self.study_id}, success={self.success}, pnl={self.pnl})>"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'study_id': self.study_id,
            'detection_time': self.detection_time.isoformat() if self.detection_time else None,
            'asset': self.asset,
            'timeframe': self.timeframe,
            'predicted_direction': self.predicted_direction,
            'success': self.success,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'exit_reason': self.exit_reason,
            'bars_held': self.bars_held
        }


class RefreshSchedule(Base):
    """
    Parameter refresh schedule.
    
    Tracks when parameters should be refreshed.
    """
    __tablename__ = 'refresh_schedules'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Study reference
    study_id = Column(Integer, ForeignKey('optimization_studies.id'), nullable=False, index=True)
    
    # Schedule
    refresh_frequency_days = Column(Integer, default=30)  # Refresh every N days
    last_refresh = Column(DateTime)
    next_refresh = Column(DateTime, index=True)
    
    # Thresholds for automatic refresh
    performance_threshold_win_rate = Column(Float, default=0.50)
    performance_threshold_sharpe = Column(Float, default=1.0)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<RefreshSchedule(study_id={self.study_id}, next_refresh={self.next_refresh})>"


# Helper functions
def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(engine)


def drop_tables(engine):
    """Drop all tables"""
    Base.metadata.drop_all(engine)
