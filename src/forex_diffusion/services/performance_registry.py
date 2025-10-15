"""
Performance Registry System for tracking real-time model and prediction accuracy.

Tracks:
- Model performance across different horizons
- Regime-specific accuracy
- Degradation detection and alerts
- Historical performance trends
- Multi-model comparison metrics
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path
import threading
from collections import defaultdict, deque
import statistics

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.horizon_converter import MarketRegime, timeframe_to_minutes


class PerformanceMetric(Enum):
    """Types of performance metrics tracked."""
    ACCURACY = "accuracy"
    MAE = "mae"  # Mean Absolute Error
    RMSE = "rmse"  # Root Mean Square Error
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    SHARPE_RATIO = "sharpe_ratio"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PredictionRecord:
    """Single prediction record for tracking."""
    model_name: str
    symbol: str
    timeframe: str
    horizon: str
    prediction: float
    actual: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    regime: str = "unknown"
    volatility: float = 0.0
    confidence: float = 0.0
    scaling_mode: str = "linear"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance degradation alert."""
    alert_id: str
    level: AlertLevel
    metric: PerformanceMetric
    model_name: str
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


@dataclass
class ModelPerformanceStats:
    """Comprehensive performance statistics for a model."""
    model_name: str
    total_predictions: int
    accuracy: float
    mae: float
    rmse: float
    directional_accuracy: float
    win_rate: float
    avg_confidence: float
    performance_by_horizon: Dict[str, float]
    performance_by_regime: Dict[str, float]
    recent_trend: str  # "improving", "degrading", "stable"
    last_updated: datetime = field(default_factory=datetime.now)


class PerformanceRegistry:
    """Main performance tracking and monitoring system."""

    def __init__(self, db_path: Optional[str] = None, max_memory_records: int = 10000):
        """
        Initialize the performance registry.

        Args:
            db_path: Path to SQLite database for persistence
            max_memory_records: Maximum records to keep in memory per model
        """
        self.db_path = db_path or "data/performance_registry.db"
        self.max_memory_records = max_memory_records

        # In-memory storage for fast access
        self.predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_memory_records))
        self.performance_cache: Dict[str, ModelPerformanceStats] = {}
        self.active_alerts: List[PerformanceAlert] = []

        # Thread safety
        self.lock = threading.RLock()

        # Performance thresholds for alerts
        self.performance_thresholds = {
            PerformanceMetric.ACCURACY: 0.45,  # Alert if accuracy drops below 45%
            PerformanceMetric.DIRECTIONAL_ACCURACY: 0.48,
            PerformanceMetric.WIN_RATE: 0.40,
            PerformanceMetric.MAE: 0.05,  # Alert if MAE exceeds 5%
        }

        # Initialize database
        self._init_database()

        logger.info(f"Performance Registry initialized with database at {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database for persistence."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    actual REAL,
                    timestamp TEXT NOT NULL,
                    regime TEXT,
                    volatility REAL,
                    confidence REAL,
                    scaling_mode TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    level TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_model_time
                ON predictions(model_name, timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_model_resolved
                ON alerts(model_name, resolved)
            """)

    def record_prediction(
        self,
        model_name: str,
        symbol: str,
        timeframe: str,
        horizon: str,
        prediction: float,
        regime: str = "unknown",
        volatility: float = 0.0,
        confidence: float = 0.0,
        scaling_mode: str = "linear",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a new prediction."""

        record = PredictionRecord(
            model_name=model_name,
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            prediction=prediction,
            regime=regime,
            volatility=volatility,
            confidence=confidence,
            scaling_mode=scaling_mode,
            metadata=metadata or {}
        )

        with self.lock:
            # Add to memory
            self.predictions[model_name].append(record)

            # Persist to database
            self._persist_prediction(record)

            # Invalidate cache for this model
            if model_name in self.performance_cache:
                del self.performance_cache[model_name]

        record_id = f"{model_name}_{record.timestamp.isoformat()}"
        logger.debug(f"Recorded prediction for {model_name}: {prediction}")

        return record_id

    def update_actual_value(
        self,
        model_name: str,
        prediction_timestamp: datetime,
        actual_value: float,
        tolerance_minutes: int = 5
    ) -> bool:
        """Update the actual value for a prediction."""

        with self.lock:
            # Find matching prediction in memory
            for record in reversed(self.predictions[model_name]):
                time_diff = abs((record.timestamp - prediction_timestamp).total_seconds() / 60)
                if time_diff <= tolerance_minutes and record.actual is None:
                    record.actual = actual_value

                    # Update in database
                    self._update_actual_in_db(record, actual_value)

                    # Invalidate cache
                    if model_name in self.performance_cache:
                        del self.performance_cache[model_name]

                    # Check for performance degradation
                    self._check_performance_alerts(model_name)

                    logger.debug(f"Updated actual value for {model_name}: {actual_value}")
                    return True

        return False

    def get_model_performance(
        self,
        model_name: str,
        horizon: Optional[str] = None,
        regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        days_back: int = 30
    ) -> ModelPerformanceStats:
        """Get comprehensive performance statistics for a model."""

        cache_key = f"{model_name}_{horizon}_{regime}_{timeframe}_{days_back}"

        with self.lock:
            if cache_key in self.performance_cache:
                return self.performance_cache[cache_key]

            # Filter predictions
            predictions = self._filter_predictions(
                model_name, horizon, regime, timeframe, days_back
            )

            if not predictions:
                return ModelPerformanceStats(
                    model_name=model_name,
                    total_predictions=0,
                    accuracy=0.0,
                    mae=float('inf'),
                    rmse=float('inf'),
                    directional_accuracy=0.0,
                    win_rate=0.0,
                    avg_confidence=0.0,
                    performance_by_horizon={},
                    performance_by_regime={},
                    recent_trend="unknown"
                )

            # Calculate metrics
            stats = self._calculate_performance_stats(model_name, predictions)

            # Cache results
            self.performance_cache[cache_key] = stats

            return stats

    def get_active_alerts(self, model_name: Optional[str] = None) -> List[PerformanceAlert]:
        """Get active performance alerts."""
        with self.lock:
            if model_name:
                return [alert for alert in self.active_alerts
                       if alert.model_name == model_name and not alert.resolved]
            return [alert for alert in self.active_alerts if not alert.resolved]

    def get_model_comparison(
        self,
        model_names: List[str],
        metric: PerformanceMetric = PerformanceMetric.ACCURACY,
        days_back: int = 30
    ) -> Dict[str, float]:
        """Compare performance of multiple models."""
        comparison = {}

        for model_name in model_names:
            stats = self.get_model_performance(model_name, days_back=days_back)

            if metric == PerformanceMetric.ACCURACY:
                comparison[model_name] = stats.accuracy
            elif metric == PerformanceMetric.MAE:
                comparison[model_name] = stats.mae
            elif metric == PerformanceMetric.RMSE:
                comparison[model_name] = stats.rmse
            elif metric == PerformanceMetric.DIRECTIONAL_ACCURACY:
                comparison[model_name] = stats.directional_accuracy
            elif metric == PerformanceMetric.WIN_RATE:
                comparison[model_name] = stats.win_rate
            else:
                comparison[model_name] = 0.0

        return comparison

    def get_performance_trend(
        self,
        model_name: str,
        metric: PerformanceMetric = PerformanceMetric.ACCURACY,
        periods: int = 7
    ) -> List[Tuple[datetime, float]]:
        """Get performance trend over time periods."""

        with self.lock:
            predictions = self.predictions[model_name]
            if not predictions:
                return []

            # Group predictions by time periods
            now = datetime.now()
            trend_data = []

            for i in range(periods):
                period_start = now - timedelta(days=i+1)
                period_end = now - timedelta(days=i)

                period_predictions = [
                    p for p in predictions
                    if period_start <= p.timestamp < period_end and p.actual is not None
                ]

                if period_predictions:
                    if metric == PerformanceMetric.ACCURACY:
                        errors = [abs(p.prediction - p.actual) / abs(p.actual)
                                for p in period_predictions if p.actual != 0]
                        accuracy = 1.0 - (sum(errors) / len(errors)) if errors else 0.0
                        trend_data.append((period_start, max(0.0, accuracy)))
                    # Add other metrics as needed

            return list(reversed(trend_data))

    def export_performance_report(
        self,
        model_name: Optional[str] = None,
        format: str = "json"
    ) -> str:
        """Export performance report."""

        if model_name:
            models = [model_name]
        else:
            models = list(self.predictions.keys())

        report = {
            "generated_at": datetime.now().isoformat(),
            "models": {}
        }

        for model in models:
            stats = self.get_model_performance(model)
            alerts = self.get_active_alerts(model)

            report["models"][model] = {
                "performance": {
                    "total_predictions": stats.total_predictions,
                    "accuracy": stats.accuracy,
                    "mae": stats.mae,
                    "rmse": stats.rmse,
                    "directional_accuracy": stats.directional_accuracy,
                    "win_rate": stats.win_rate,
                    "avg_confidence": stats.avg_confidence,
                    "recent_trend": stats.recent_trend,
                    "last_updated": stats.last_updated.isoformat()
                },
                "performance_by_horizon": stats.performance_by_horizon,
                "performance_by_regime": stats.performance_by_regime,
                "active_alerts": [
                    {
                        "level": alert.level.value,
                        "metric": alert.metric.value,
                        "message": alert.message,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in alerts
                ]
            }

        if format == "json":
            return json.dumps(report, indent=2)
        else:
            # Add other formats as needed
            return str(report)

    def _filter_predictions(
        self,
        model_name: str,
        horizon: Optional[str],
        regime: Optional[str],
        timeframe: Optional[str],
        days_back: int
    ) -> List[PredictionRecord]:
        """Filter predictions based on criteria."""

        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered = []

        for record in self.predictions[model_name]:
            if record.timestamp < cutoff_date:
                continue
            if record.actual is None:  # Only predictions with actual values
                continue
            if horizon and record.horizon != horizon:
                continue
            if regime and record.regime != regime:
                continue
            if timeframe and record.timeframe != timeframe:
                continue

            filtered.append(record)

        return filtered

    def _calculate_performance_stats(
        self,
        model_name: str,
        predictions: List[PredictionRecord]
    ) -> ModelPerformanceStats:
        """Calculate comprehensive performance statistics."""

        if not predictions:
            return ModelPerformanceStats(
                model_name=model_name,
                total_predictions=0,
                accuracy=0.0,
                mae=float('inf'),
                rmse=float('inf'),
                directional_accuracy=0.0,
                win_rate=0.0,
                avg_confidence=0.0,
                performance_by_horizon={},
                performance_by_regime={},
                recent_trend="unknown"
            )

        # Basic metrics
        errors = []
        directional_correct = 0
        confidences = []

        # Group by horizon and regime
        by_horizon = defaultdict(list)
        by_regime = defaultdict(list)

        for pred in predictions:
            if pred.actual == 0:
                continue

            error = abs(pred.prediction - pred.actual) / abs(pred.actual)
            errors.append(error)

            # Directional accuracy (same direction of change)
            if (pred.prediction > 0) == (pred.actual > 0):
                directional_correct += 1

            confidences.append(pred.confidence)

            # Group for detailed analysis
            by_horizon[pred.horizon].append(error)
            by_regime[pred.regime].append(error)

        # Calculate metrics
        accuracy = 1.0 - (sum(errors) / len(errors)) if errors else 0.0
        mae = sum(errors) / len(errors) if errors else float('inf')
        rmse = np.sqrt(sum([e**2 for e in errors]) / len(errors)) if errors else float('inf')
        directional_accuracy = directional_correct / len(predictions) if predictions else 0.0
        win_rate = len([e for e in errors if e < 0.02]) / len(errors) if errors else 0.0  # Within 2%
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Performance by horizon
        performance_by_horizon = {}
        for horizon, horizon_errors in by_horizon.items():
            performance_by_horizon[horizon] = 1.0 - (sum(horizon_errors) / len(horizon_errors))

        # Performance by regime
        performance_by_regime = {}
        for regime, regime_errors in by_regime.items():
            performance_by_regime[regime] = 1.0 - (sum(regime_errors) / len(regime_errors))

        # Recent trend analysis
        recent_trend = self._analyze_trend(predictions)

        return ModelPerformanceStats(
            model_name=model_name,
            total_predictions=len(predictions),
            accuracy=max(0.0, accuracy),
            mae=mae,
            rmse=rmse,
            directional_accuracy=max(0.0, directional_accuracy),
            win_rate=max(0.0, win_rate),
            avg_confidence=max(0.0, avg_confidence),
            performance_by_horizon=performance_by_horizon,
            performance_by_regime=performance_by_regime,
            recent_trend=recent_trend
        )

    def _analyze_trend(self, predictions: List[PredictionRecord]) -> str:
        """Analyze recent performance trend."""
        if len(predictions) < 10:
            return "insufficient_data"

        # Split into recent and older predictions
        mid_point = len(predictions) // 2
        recent = predictions[mid_point:]
        older = predictions[:mid_point]

        # Calculate accuracies
        recent_errors = [abs(p.prediction - p.actual) / abs(p.actual)
                        for p in recent if p.actual != 0]
        older_errors = [abs(p.prediction - p.actual) / abs(p.actual)
                       for p in older if p.actual != 0]

        if not recent_errors or not older_errors:
            return "insufficient_data"

        recent_accuracy = 1.0 - (sum(recent_errors) / len(recent_errors))
        older_accuracy = 1.0 - (sum(older_errors) / len(older_errors))

        improvement = recent_accuracy - older_accuracy

        if improvement > 0.05:  # 5% improvement
            return "improving"
        elif improvement < -0.05:  # 5% degradation
            return "degrading"
        else:
            return "stable"

    def _check_performance_alerts(self, model_name: str):
        """Check for performance degradation and create alerts."""
        stats = self.get_model_performance(model_name, days_back=7)  # Recent week

        # Check accuracy threshold
        if stats.accuracy < self.performance_thresholds[PerformanceMetric.ACCURACY]:
            self._create_alert(
                level=AlertLevel.WARNING,
                metric=PerformanceMetric.ACCURACY,
                model_name=model_name,
                current_value=stats.accuracy,
                threshold=self.performance_thresholds[PerformanceMetric.ACCURACY],
                message=f"Accuracy dropped to {stats.accuracy:.1%}"
            )

        # Check directional accuracy
        if stats.directional_accuracy < self.performance_thresholds[PerformanceMetric.DIRECTIONAL_ACCURACY]:
            self._create_alert(
                level=AlertLevel.WARNING,
                metric=PerformanceMetric.DIRECTIONAL_ACCURACY,
                model_name=model_name,
                current_value=stats.directional_accuracy,
                threshold=self.performance_thresholds[PerformanceMetric.DIRECTIONAL_ACCURACY],
                message=f"Directional accuracy dropped to {stats.directional_accuracy:.1%}"
            )

        # Check for critical degradation trend
        if stats.recent_trend == "degrading":
            self._create_alert(
                level=AlertLevel.CRITICAL,
                metric=PerformanceMetric.ACCURACY,
                model_name=model_name,
                current_value=stats.accuracy,
                threshold=0.0,
                message=f"Performance is degrading - investigate model drift"
            )

    def _create_alert(
        self,
        level: AlertLevel,
        metric: PerformanceMetric,
        model_name: str,
        current_value: float,
        threshold: float,
        message: str
    ):
        """Create a new performance alert."""

        alert_id = f"{model_name}_{metric.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Check if similar alert already exists
        existing = [a for a in self.active_alerts
                   if a.model_name == model_name and a.metric == metric and not a.resolved]

        if existing:
            return  # Don't create duplicate alerts

        alert = PerformanceAlert(
            alert_id=alert_id,
            level=level,
            metric=metric,
            model_name=model_name,
            message=message,
            current_value=current_value,
            threshold=threshold
        )

        self.active_alerts.append(alert)
        self._persist_alert(alert)

        logger.warning(f"Performance alert for {model_name}: {message}")

    def _persist_prediction(self, record: PredictionRecord):
        """Persist prediction to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO predictions
                    (model_name, symbol, timeframe, horizon, prediction, actual,
                     timestamp, regime, volatility, confidence, scaling_mode, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.model_name, record.symbol, record.timeframe, record.horizon,
                    record.prediction, record.actual, record.timestamp.isoformat(),
                    record.regime, record.volatility, record.confidence,
                    record.scaling_mode, json.dumps(record.metadata)
                ))
        except Exception as e:
            logger.error(f"Failed to persist prediction: {e}")

    def _update_actual_in_db(self, record: PredictionRecord, actual_value: float):
        """Update actual value in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE predictions
                    SET actual = ?
                    WHERE model_name = ? AND timestamp = ?
                """, (actual_value, record.model_name, record.timestamp.isoformat()))
        except Exception as e:
            logger.error(f"Failed to update actual value: {e}")

    def _persist_alert(self, alert: PerformanceAlert):
        """Persist alert to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts
                    (alert_id, level, metric, model_name, message, current_value,
                     threshold, timestamp, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id, alert.level.value, alert.metric.value,
                    alert.model_name, alert.message, alert.current_value,
                    alert.threshold, alert.timestamp.isoformat(), int(alert.resolved)
                ))
        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")


# Global registry instance
_performance_registry = None

def get_performance_registry() -> PerformanceRegistry:
    """Get the global performance registry instance."""
    global _performance_registry
    if _performance_registry is None:
        _performance_registry = PerformanceRegistry()
    return _performance_registry