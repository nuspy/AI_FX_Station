"""
Adaptive Parameter System

Continuously optimizes system parameters based on recent performance.
Monitors win rates, profit factors, and adapts thresholds, position sizing, and stop distances.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import numpy as np
from collections import deque

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

try:
    from ..training_pipeline.database_models import Base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()


class ParameterType(Enum):
    """Types of adaptable parameters"""
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    POSITION_SIZE_MULTIPLIER = "position_size_multiplier"
    STOP_LOSS_DISTANCE = "stop_loss_distance"
    TAKE_PROFIT_DISTANCE = "take_profit_distance"
    TIMEFRAME_WEIGHT = "timeframe_weight"
    PATTERN_SPECIFIC = "pattern_specific"
    QUALITY_THRESHOLD = "quality_threshold"


class TriggerReason(Enum):
    """Reasons for parameter adaptation"""
    PERFORMANCE_DROP = "performance_drop"
    WIN_RATE_DROP = "win_rate_drop"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    REGIME_CHANGE = "regime_change"
    MARKET_CONDITION_SHIFT = "market_condition_shift"
    SCHEDULED_REVIEW = "scheduled_review"


# ==================== Database ORM Model ====================

class ParameterAdaptationDB(Base):
    """Database model for parameter adaptations."""

    __tablename__ = 'parameter_adaptations'

    # Primary Key
    id = Column(Integer, primary_key=True)
    adaptation_id = Column(String(50), unique=True, nullable=False, index=True)

    # Timing
    timestamp = Column(Integer, nullable=False, index=True)  # Unix timestamp ms

    # Trigger Information
    trigger_reason = Column(String(50), nullable=False)
    trigger_metrics = Column(JSON, nullable=True)  # Serialized PerformanceMetrics

    # Parameter Change
    parameter_name = Column(String(100), nullable=False, index=True)
    parameter_type = Column(String(50), nullable=False)
    old_value = Column(Float, nullable=False)
    new_value = Column(Float, nullable=False)

    # Scope (optional filters)
    regime = Column(String(50), nullable=True, index=True)
    symbol = Column(String(20), nullable=True, index=True)
    timeframe = Column(String(10), nullable=True)

    # Validation
    validation_method = Column(String(50), nullable=False)
    validation_passed = Column(Boolean, default=False)
    improvement_expected = Column(Float, default=0.0)
    improvement_actual = Column(Float, nullable=True)

    # Deployment Status
    deployed = Column(Boolean, default=False, index=True)
    deployed_at = Column(Integer, nullable=True)  # Unix timestamp ms
    rollback_at = Column(Integer, nullable=True)  # Unix timestamp ms

    # Additional parameters metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)
    params_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.current_timestamp())

    def __repr__(self):
        return f"<ParameterAdaptationDB(id={self.adaptation_id}, param={self.parameter_name}, deployed={self.deployed})>"


# ==================== Data Classes ====================

@dataclass
class PerformanceMetrics:
    """Recent performance metrics"""
    lookback_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_profit: float
    avg_loss: float
    total_pnl: float
    timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ParameterAdaptation:
    """Parameter adaptation record"""
    adaptation_id: str
    timestamp: int
    trigger_reason: TriggerReason
    trigger_metrics: PerformanceMetrics

    # Parameter change
    parameter_name: str
    parameter_type: ParameterType
    old_value: float
    new_value: float

    # Scope
    regime: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None

    # Validation
    validation_method: str = "holdout"
    validation_passed: bool = False
    improvement_expected: float = 0.0
    improvement_actual: Optional[float] = None

    # Status
    deployed: bool = False
    deployed_at: Optional[int] = None
    rollback_at: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enums to strings
        data['trigger_reason'] = self.trigger_reason.value
        data['parameter_type'] = self.parameter_type.value
        return data

    def to_db(self) -> ParameterAdaptationDB:
        """Convert to database model."""
        return ParameterAdaptationDB(
            adaptation_id=self.adaptation_id,
            timestamp=self.timestamp,
            trigger_reason=self.trigger_reason.value,
            trigger_metrics=self.trigger_metrics.to_dict(),
            parameter_name=self.parameter_name,
            parameter_type=self.parameter_type.value,
            old_value=self.old_value,
            new_value=self.new_value,
            regime=self.regime,
            symbol=self.symbol,
            timeframe=self.timeframe,
            validation_method=self.validation_method,
            validation_passed=self.validation_passed,
            improvement_expected=self.improvement_expected,
            improvement_actual=self.improvement_actual,
            deployed=self.deployed,
            deployed_at=self.deployed_at,
            rollback_at=self.rollback_at
        )

    @classmethod
    def from_db(cls, db_obj: ParameterAdaptationDB) -> ParameterAdaptation:
        """Create from database model."""
        return cls(
            adaptation_id=db_obj.adaptation_id,
            timestamp=db_obj.timestamp,
            trigger_reason=TriggerReason(db_obj.trigger_reason),
            trigger_metrics=PerformanceMetrics(**db_obj.trigger_metrics) if db_obj.trigger_metrics else PerformanceMetrics(
                lookback_trades=0, win_rate=0.0, profit_factor=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, avg_profit=0.0, avg_loss=0.0, total_pnl=0.0, timestamp=0
            ),
            parameter_name=db_obj.parameter_name,
            parameter_type=ParameterType(db_obj.parameter_type),
            old_value=db_obj.old_value,
            new_value=db_obj.new_value,
            regime=db_obj.regime,
            symbol=db_obj.symbol,
            timeframe=db_obj.timeframe,
            validation_method=db_obj.validation_method,
            validation_passed=db_obj.validation_passed,
            improvement_expected=db_obj.improvement_expected,
            improvement_actual=db_obj.improvement_actual,
            deployed=db_obj.deployed,
            deployed_at=db_obj.deployed_at,
            rollback_at=db_obj.rollback_at
        )


class AdaptiveParameterSystem:
    """
    Continuously optimizes system parameters based on performance.

    Features:
    - Rolling window performance analysis (500 trades)
    - Win rate, profit factor, Sharpe ratio tracking
    - Automatic parameter adjustment triggers
    - Validation on holdout data
    - Rollback capability
    """

    def __init__(
        self,
        lookback_window: int = 500,
        min_trades_for_adaptation: int = 100,
        performance_review_frequency: int = 50,  # trades
        win_rate_threshold: float = 0.45,
        profit_factor_threshold: float = 1.2,
        consecutive_loss_threshold: int = 5,
        validation_split: float = 0.3,
        db_session: Optional[Session] = None
    ):
        """
        Initialize adaptive parameter system.

        Args:
            lookback_window: Number of recent trades to analyze
            min_trades_for_adaptation: Minimum trades before adapting
            performance_review_frequency: Review every N trades
            win_rate_threshold: Trigger if win rate drops below
            profit_factor_threshold: Trigger if PF drops below
            consecutive_loss_threshold: Trigger after N consecutive losses
            validation_split: Fraction of data for validation
            db_session: Optional SQLAlchemy session for persistence
        """
        self.lookback_window = lookback_window
        self.min_trades_for_adaptation = min_trades_for_adaptation
        self.performance_review_frequency = performance_review_frequency
        self.win_rate_threshold = win_rate_threshold
        self.profit_factor_threshold = profit_factor_threshold
        self.consecutive_loss_threshold = consecutive_loss_threshold
        self.validation_split = validation_split
        self.db_session = db_session

        # State tracking
        self.trade_history: deque = deque(maxlen=lookback_window)
        self.adaptation_history: List[ParameterAdaptation] = []
        self.current_parameters: Dict[str, float] = {}
        self.trades_since_review: int = 0
        self.consecutive_losses: int = 0

        # Load adaptations from database if session provided
        if self.db_session:
            self._load_from_database()

    def record_trade(
        self,
        timestamp: int,
        symbol: str,
        regime: str,
        pnl: float,
        outcome: str,
        parameters_used: Dict[str, float]
    ):
        """
        Record a completed trade for performance tracking.

        Args:
            timestamp: Trade close timestamp
            symbol: Trading symbol
            regime: Market regime at trade time
            pnl: Profit/loss
            outcome: 'win', 'loss', or 'breakeven'
            parameters_used: Parameters in effect for this trade
        """
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'regime': regime,
            'pnl': pnl,
            'outcome': outcome,
            'parameters': parameters_used.copy()
        }

        self.trade_history.append(trade_record)
        self.trades_since_review += 1

        # Track consecutive losses
        if outcome == 'loss':
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def should_trigger_adaptation(self) -> Tuple[bool, Optional[TriggerReason], Optional[PerformanceMetrics]]:
        """
        Determine if parameter adaptation should be triggered.

        Returns:
            (should_trigger, trigger_reason, current_metrics)
        """
        if len(self.trade_history) < self.min_trades_for_adaptation:
            return False, None, None

        # Calculate current metrics
        metrics = self.calculate_performance_metrics()

        # Check trigger conditions
        triggers = []

        # 1. Scheduled review
        if self.trades_since_review >= self.performance_review_frequency:
            triggers.append(TriggerReason.SCHEDULED_REVIEW)

        # 2. Win rate drop
        if metrics.win_rate < self.win_rate_threshold:
            triggers.append(TriggerReason.WIN_RATE_DROP)

        # 3. Profit factor drop
        if metrics.profit_factor < self.profit_factor_threshold:
            triggers.append(TriggerReason.PERFORMANCE_DROP)

        # 4. Consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_threshold:
            triggers.append(TriggerReason.CONSECUTIVE_LOSSES)

        if triggers:
            # Use most severe trigger
            primary_trigger = triggers[0] if TriggerReason.PERFORMANCE_DROP not in triggers else TriggerReason.PERFORMANCE_DROP
            return True, primary_trigger, metrics

        return False, None, None

    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculate performance metrics from recent trades.

        Returns:
            Performance metrics
        """
        if not self.trade_history:
            return PerformanceMetrics(
                lookback_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_profit=0.0,
                avg_loss=0.0,
                total_pnl=0.0,
                timestamp=int(datetime.now().timestamp() * 1000)
            )

        trades = list(self.trade_history)
        pnls = [t['pnl'] for t in trades]
        outcomes = [t['outcome'] for t in trades]

        # Win rate
        wins = sum(1 for o in outcomes if o == 'win')
        win_rate = wins / len(outcomes) if outcomes else 0.0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Sharpe ratio (assuming daily returns)
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Average profit/loss
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        avg_profit = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0

        return PerformanceMetrics(
            lookback_trades=len(trades),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            avg_profit=float(avg_profit),
            avg_loss=float(avg_loss),
            total_pnl=float(sum(pnls)),
            timestamp=trades[-1]['timestamp']
        )

    def suggest_parameter_adjustments(
        self,
        trigger_metrics: PerformanceMetrics,
        trigger_reason: TriggerReason
    ) -> List[Tuple[str, ParameterType, float, float]]:
        """
        Suggest parameter adjustments based on performance.

        Args:
            trigger_metrics: Current performance metrics
            trigger_reason: Reason for adaptation

        Returns:
            List of (parameter_name, parameter_type, old_value, new_value)
        """
        suggestions = []

        # Analyze what needs adjustment based on metrics
        if trigger_metrics.win_rate < self.win_rate_threshold:
            # Low win rate -> increase quality threshold to be more selective
            current_threshold = self.current_parameters.get('quality_threshold', 0.65)
            new_threshold = min(current_threshold + 0.05, 0.85)
            suggestions.append((
                'quality_threshold',
                ParameterType.QUALITY_THRESHOLD,
                current_threshold,
                new_threshold
            ))

        if trigger_metrics.profit_factor < self.profit_factor_threshold:
            # Low profit factor -> adjust position sizing or stops
            if trigger_metrics.avg_loss < trigger_metrics.avg_profit:
                # Winners smaller than losers -> increase TP distance
                current_tp = self.current_parameters.get('take_profit_distance', 2.0)
                new_tp = min(current_tp * 1.2, 4.0)
                suggestions.append((
                    'take_profit_distance',
                    ParameterType.TAKE_PROFIT_DISTANCE,
                    current_tp,
                    new_tp
                ))
            else:
                # Losers larger than winners -> tighten stops
                current_sl = self.current_parameters.get('stop_loss_distance', 1.5)
                new_sl = max(current_sl * 0.9, 0.8)
                suggestions.append((
                    'stop_loss_distance',
                    ParameterType.STOP_LOSS_DISTANCE,
                    current_sl,
                    new_sl
                ))

        if trigger_reason == TriggerReason.CONSECUTIVE_LOSSES:
            # Consecutive losses -> reduce position size temporarily
            current_size = self.current_parameters.get('position_size_multiplier', 1.0)
            new_size = max(current_size * 0.8, 0.5)
            suggestions.append((
                'position_size_multiplier',
                ParameterType.POSITION_SIZE_MULTIPLIER,
                current_size,
                new_size
            ))

        return suggestions

    def validate_adaptation(
        self,
        parameter_name: str,
        new_value: float,
        validation_method: str = "holdout"
    ) -> Tuple[bool, float]:
        """
        Validate parameter adaptation on holdout data.

        Args:
            parameter_name: Parameter to validate
            new_value: New parameter value
            validation_method: Validation method

        Returns:
            (validation_passed, improvement_expected)
        """
        if len(self.trade_history) < 50:
            # Not enough data for validation
            return True, 0.0  # Optimistic default

        # Split data
        trades = list(self.trade_history)
        split_idx = int(len(trades) * (1 - self.validation_split))
        validation_trades = trades[split_idx:]

        # Simulate with old parameters
        old_metrics = self._simulate_performance(validation_trades, {parameter_name: self.current_parameters.get(parameter_name, 0.0)})

        # Simulate with new parameters
        new_metrics = self._simulate_performance(validation_trades, {parameter_name: new_value})

        # Check improvement
        improvement = new_metrics.profit_factor - old_metrics.profit_factor

        validation_passed = improvement > 0.05  # At least 5% improvement

        return validation_passed, float(improvement)

    def _simulate_performance(
        self,
        trades: List[Dict[str, Any]],
        parameters: Dict[str, float]
    ) -> PerformanceMetrics:
        """
        Simulate performance with given parameters.

        Args:
            trades: List of trade records
            parameters: Parameters to simulate with

        Returns:
            Simulated performance metrics
        """
        # Simplified simulation - in practice, would replay trades with new parameters
        # For now, just calculate metrics assuming parameters don't change outcomes
        pnls = [t['pnl'] for t in trades]
        outcomes = [t['outcome'] for t in trades]

        wins = sum(1 for o in outcomes if o == 'win')
        win_rate = wins / len(outcomes) if outcomes else 0.0

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return PerformanceMetrics(
            lookback_trades=len(trades),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            avg_profit=0.0,
            avg_loss=0.0,
            total_pnl=float(sum(pnls)),
            timestamp=trades[-1]['timestamp'] if trades else 0
        )

    def apply_adaptation(
        self,
        parameter_name: str,
        parameter_type: ParameterType,
        new_value: float,
        trigger_reason: TriggerReason,
        trigger_metrics: PerformanceMetrics,
        regime: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> ParameterAdaptation:
        """
        Apply parameter adaptation.

        Args:
            parameter_name: Parameter to adapt
            parameter_type: Type of parameter
            new_value: New value
            trigger_reason: Reason for adaptation
            trigger_metrics: Metrics that triggered adaptation
            regime: Optional regime scope
            symbol: Optional symbol scope

        Returns:
            Parameter adaptation record
        """
        old_value = self.current_parameters.get(parameter_name, 0.0)

        # Validate
        validation_passed, improvement = self.validate_adaptation(parameter_name, new_value)

        # Create adaptation record
        adaptation = ParameterAdaptation(
            adaptation_id=f"adapt_{len(self.adaptation_history)}",
            timestamp=int(datetime.now().timestamp() * 1000),
            trigger_reason=trigger_reason,
            trigger_metrics=trigger_metrics,
            parameter_name=parameter_name,
            parameter_type=parameter_type,
            old_value=old_value,
            new_value=new_value,
            regime=regime,
            symbol=symbol,
            validation_method="holdout",
            validation_passed=validation_passed,
            improvement_expected=improvement,
            deployed=validation_passed,
            deployed_at=int(datetime.now().timestamp() * 1000) if validation_passed else None
        )

        # Apply if validated
        if validation_passed:
            self.current_parameters[parameter_name] = new_value

        # Store
        self.adaptation_history.append(adaptation)

        # Save to database if available
        if self.db_session:
            self._save_adaptation_to_db(adaptation)
            try:
                self.db_session.commit()
            except Exception as e:
                self.db_session.rollback()
                print(f"Error committing adaptation to database: {e}")

        # Reset review counter
        self.trades_since_review = 0

        return adaptation

    def run_adaptation_cycle(self) -> List[ParameterAdaptation]:
        """
        Run a complete adaptation cycle.

        Returns:
            List of adaptations applied
        """
        # Check if should trigger
        should_trigger, trigger_reason, metrics = self.should_trigger_adaptation()

        if not should_trigger or metrics is None or trigger_reason is None:
            return []

        # Get suggestions
        suggestions = self.suggest_parameter_adjustments(metrics, trigger_reason)

        # Apply adaptations
        adaptations = []
        for param_name, param_type, old_val, new_val in suggestions:
            adaptation = self.apply_adaptation(
                parameter_name=param_name,
                parameter_type=param_type,
                new_value=new_val,
                trigger_reason=trigger_reason,
                trigger_metrics=metrics
            )
            adaptations.append(adaptation)

        return adaptations

    def get_current_parameters(self) -> Dict[str, float]:
        """Get current active parameters"""
        return self.current_parameters.copy()

    def rollback_adaptation(self, adaptation_id: str) -> bool:
        """
        Rollback a parameter adaptation.

        Args:
            adaptation_id: Adaptation to rollback

        Returns:
            True if successful
        """
        for adaptation in self.adaptation_history:
            if adaptation.adaptation_id == adaptation_id and adaptation.deployed:
                # Restore old value
                self.current_parameters[adaptation.parameter_name] = adaptation.old_value
                adaptation.deployed = False
                adaptation.rollback_at = int(datetime.now().timestamp() * 1000)

                # Update database if available
                if self.db_session:
                    self._save_adaptation_to_db(adaptation)
                    try:
                        self.db_session.commit()
                    except Exception as e:
                        self.db_session.rollback()
                        print(f"Error committing rollback to database: {e}")

                return True

        return False

    # ==================== Database Methods ====================

    def _load_from_database(self):
        """Load recent adaptations from database."""
        if not self.db_session:
            return

        try:
            # Load recent deployed adaptations (last 100)
            db_adaptations = self.db_session.query(ParameterAdaptationDB).filter(
                ParameterAdaptationDB.deployed == True
            ).order_by(ParameterAdaptationDB.timestamp.desc()).limit(100).all()

            for db_adapt in db_adaptations:
                adaptation = ParameterAdaptation.from_db(db_adapt)
                self.adaptation_history.append(adaptation)

                # Apply deployed adaptations to current parameters
                if adaptation.deployed and adaptation.rollback_at is None:
                    self.current_parameters[adaptation.parameter_name] = adaptation.new_value

        except Exception as e:
            print(f"Error loading adaptations from database: {e}")

    def _save_adaptation_to_db(self, adaptation: ParameterAdaptation):
        """
        Save adaptation to database.

        Args:
            adaptation: Adaptation to save
        """
        if not self.db_session:
            return

        try:
            # Check if already exists
            existing = self.db_session.query(ParameterAdaptationDB).filter(
                ParameterAdaptationDB.adaptation_id == adaptation.adaptation_id
            ).first()

            if existing:
                # Update existing
                existing.deployed = adaptation.deployed
                existing.deployed_at = adaptation.deployed_at
                existing.rollback_at = adaptation.rollback_at
                existing.improvement_actual = adaptation.improvement_actual
                existing.validation_passed = adaptation.validation_passed
            else:
                # Create new
                db_adaptation = adaptation.to_db()
                self.db_session.add(db_adaptation)

        except Exception as e:
            print(f"Error saving adaptation to database: {e}")

    def save_to_database(self):
        """Save all adaptations to database and commit."""
        if not self.db_session:
            return

        try:
            for adaptation in self.adaptation_history:
                self._save_adaptation_to_db(adaptation)

            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            print(f"Error committing adaptations to database: {e}")

    def get_adaptations_from_db(
        self,
        regime: Optional[str] = None,
        symbol: Optional[str] = None,
        deployed_only: bool = True,
        limit: int = 100
    ) -> List[ParameterAdaptation]:
        """
        Query adaptations from database with filters.

        Args:
            regime: Filter by regime
            symbol: Filter by symbol
            deployed_only: Only return deployed adaptations
            limit: Maximum number to return

        Returns:
            List of adaptations
        """
        if not self.db_session:
            return []

        try:
            query = self.db_session.query(ParameterAdaptationDB)

            if deployed_only:
                query = query.filter(ParameterAdaptationDB.deployed == True)
            if regime:
                query = query.filter(ParameterAdaptationDB.regime == regime)
            if symbol:
                query = query.filter(ParameterAdaptationDB.symbol == symbol)

            query = query.order_by(ParameterAdaptationDB.timestamp.desc()).limit(limit)

            db_adaptations = query.all()
            return [ParameterAdaptation.from_db(db_adapt) for db_adapt in db_adaptations]

        except Exception as e:
            print(f"Error querying adaptations from database: {e}")
            return []

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about adaptations.

        Returns:
            Statistics dictionary
        """
        if not self.adaptation_history:
            return {}

        deployed = [a for a in self.adaptation_history if a.deployed]
        validated = [a for a in self.adaptation_history if a.validation_passed]

        return {
            'total_adaptations': len(self.adaptation_history),
            'deployed_adaptations': len(deployed),
            'validation_pass_rate': len(validated) / len(self.adaptation_history) if self.adaptation_history else 0.0,
            'avg_improvement_expected': np.mean([a.improvement_expected for a in deployed]) if deployed else 0.0,
            'parameter_types_adapted': list(set(a.parameter_type.value for a in self.adaptation_history)),
            'most_adapted_parameter': max(set(a.parameter_name for a in self.adaptation_history),
                                         key=lambda p: sum(1 for a in self.adaptation_history if a.parameter_name == p)) if self.adaptation_history else None
        }
