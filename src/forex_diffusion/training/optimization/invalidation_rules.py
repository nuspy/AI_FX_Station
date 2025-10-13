"""
Invalidation rules engine with 75th percentile quantile logic.

This module implements the sophisticated invalidation rules that determine when
a pattern signal should be considered failed based on time and loss thresholds
that adapt to historical performance statistics.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

@dataclass
class InvalidationThresholds:
    """Adaptive invalidation thresholds for a pattern/context"""

    pattern_key: str
    direction: str
    asset: str
    timeframe: str
    regime_tag: Optional[str] = None

    # Base thresholds (configuration)
    k_time_multiplier: float = 4.0
    k_loss_multiplier: float = 4.0
    quantile_threshold: float = 0.75

    # Adaptive thresholds (learned from data)
    adaptive_time_threshold: Optional[float] = None
    adaptive_loss_threshold: Optional[float] = None

    # Statistics used for calculation
    sample_size: int = 0
    last_updated: Optional[datetime] = None

@dataclass
class InvalidationEvent:
    """Record of a pattern invalidation"""

    signal_id: str
    invalidation_time: datetime
    reason: str  # "time_exceeded", "loss_exceeded", "both"

    time_to_invalidation_bars: int
    loss_at_invalidation: float
    potential_gain: float

    k_time_actual: float  # Actual time as multiple of threshold
    k_loss_actual: float  # Actual loss as multiple of potential gain

class InvalidationRuleEngine:
    """
    Implements sophisticated invalidation rules with adaptive thresholds.

    The engine learns from historical data to determine the minimum thresholds
    beyond which patterns rarely succeed, using the 75th percentile rule.
    """

    def __init__(self):
        self.thresholds_cache: Dict[str, InvalidationThresholds] = {}
        self.invalidation_history: List[InvalidationEvent] = []

    async def initialize_thresholds(self, pattern_key: str, direction: str,
                                  asset: str, timeframe: str,
                                  quantile: float = 0.75) -> InvalidationThresholds:
        """
        Initialize or update adaptive thresholds for a pattern context.

        Args:
            pattern_key: Pattern identifier
            direction: Bull/bear direction
            asset: Asset symbol
            timeframe: Timeframe string
            quantile: Percentile threshold for adaptation

        Returns:
            InvalidationThresholds object
        """
        cache_key = self._get_cache_key(pattern_key, direction, asset, timeframe)

        # Check if we have cached thresholds
        if cache_key in self.thresholds_cache:
            thresholds = self.thresholds_cache[cache_key]

            # Check if update is needed (daily update)
            if (thresholds.last_updated and
                datetime.now() - thresholds.last_updated < timedelta(days=1)):
                return thresholds

        # Create or update thresholds
        thresholds = InvalidationThresholds(
            pattern_key=pattern_key,
            direction=direction,
            asset=asset,
            timeframe=timeframe,
            quantile_threshold=quantile
        )

        # Load historical data and calculate adaptive thresholds
        historical_data = await self._load_historical_patterns(
            pattern_key, direction, asset, timeframe
        )

        if historical_data:
            self._calculate_adaptive_thresholds(thresholds, historical_data)

        thresholds.last_updated = datetime.now()
        self.thresholds_cache[cache_key] = thresholds

        logger.info(
            f"Initialized invalidation thresholds for {cache_key}: "
            f"time={thresholds.adaptive_time_threshold:.2f}, "
            f"loss={thresholds.adaptive_loss_threshold:.2f}"
        )

        return thresholds

    def apply_rules(self, metrics: Dict[str, Dict[str, Any]],
                   k_time_multiplier: float, k_loss_multiplier: float) -> Dict[str, Dict[str, Any]]:
        """
        Apply invalidation rules to backtest metrics, filtering out invalid signals.

        Args:
            metrics: Performance metrics by dataset
            k_time_multiplier: Base time multiplier
            k_loss_multiplier: Base loss multiplier

        Returns:
            Filtered metrics with invalidation rules applied
        """
        filtered_metrics = {}

        for dataset_id, dataset_metrics in metrics.items():
            # Apply invalidation filtering to signal-level data
            if "signal_details" in dataset_metrics:
                filtered_signals = self._filter_signals_by_invalidation(
                    dataset_metrics["signal_details"],
                    k_time_multiplier,
                    k_loss_multiplier
                )

                # Recalculate metrics from filtered signals
                filtered_dataset_metrics = self._recalculate_metrics_from_signals(
                    filtered_signals, dataset_metrics
                )

                # Add invalidation statistics
                invalidation_stats = self._calculate_invalidation_statistics(
                    dataset_metrics["signal_details"], filtered_signals
                )
                filtered_dataset_metrics.update(invalidation_stats)

                filtered_metrics[dataset_id] = filtered_dataset_metrics
            else:
                # No signal-level data available, return original metrics
                filtered_metrics[dataset_id] = dataset_metrics

        return filtered_metrics

    def check_signal_invalidation(self, signal_data: Dict[str, Any],
                                thresholds: InvalidationThresholds,
                                current_time: datetime,
                                current_price: float) -> Optional[InvalidationEvent]:
        """
        Check if a specific signal should be invalidated.

        Args:
            signal_data: Signal information
            thresholds: Invalidation thresholds
            current_time: Current timestamp
            current_price: Current price

        Returns:
            InvalidationEvent if signal should be invalidated, None otherwise
        """
        entry_time = signal_data.get("entry_time")
        entry_price = signal_data.get("entry_price")
        target_price = signal_data.get("target_price")
        direction = signal_data.get("direction", "bull")

        if not all([entry_time, entry_price, target_price]):
            return None

        # Calculate time elapsed
        time_elapsed = current_time - entry_time
        timeframe_duration = self._get_timeframe_duration(thresholds.timeframe)
        time_multiplier = time_elapsed.total_seconds() / timeframe_duration.total_seconds()

        # Calculate current loss
        if direction.lower() in ["bull", "long"]:
            potential_gain = abs(target_price - entry_price)
            current_loss = max(0, entry_price - current_price)
        else:
            potential_gain = abs(entry_price - target_price)
            current_loss = max(0, current_price - entry_price)

        loss_multiplier = current_loss / potential_gain if potential_gain > 0 else 0

        # Check time invalidation
        time_threshold = (thresholds.adaptive_time_threshold or
                         thresholds.k_time_multiplier)
        time_exceeded = time_multiplier > time_threshold

        # Check loss invalidation
        loss_threshold = (thresholds.adaptive_loss_threshold or
                         thresholds.k_loss_multiplier)
        loss_exceeded = loss_multiplier > loss_threshold

        # Determine invalidation
        if time_exceeded or loss_exceeded:
            if time_exceeded and loss_exceeded:
                reason = "both"
            elif time_exceeded:
                reason = "time_exceeded"
            else:
                reason = "loss_exceeded"

            return InvalidationEvent(
                signal_id=signal_data.get("signal_id", "unknown"),
                invalidation_time=current_time,
                reason=reason,
                time_to_invalidation_bars=int(time_multiplier *
                                            self._timeframe_to_bars_per_day(thresholds.timeframe)),
                loss_at_invalidation=current_loss,
                potential_gain=potential_gain,
                k_time_actual=time_multiplier,
                k_loss_actual=loss_multiplier
            )

        return None

    def _filter_signals_by_invalidation(self, signals: List[Dict[str, Any]],
                                      k_time_multiplier: float,
                                      k_loss_multiplier: float) -> List[Dict[str, Any]]:
        """Filter signals that would be invalidated by the rules"""

        filtered_signals = []

        for signal in signals:
            # Check if signal reached target before invalidation
            target_reached = signal.get("target_reached", False)
            time_to_target = signal.get("time_to_target_bars", float('inf'))
            max_loss_ratio = signal.get("max_loss_ratio", 0.0)

            # Calculate invalidation thresholds
            timeframe_bars = self._timeframe_to_bars_per_day(signal.get("timeframe", "1h"))
            time_threshold_bars = k_time_multiplier * timeframe_bars
            loss_threshold = k_loss_multiplier

            # Signal is valid if it reached target before invalidation limits
            if target_reached:
                # Check if target was reached within time limit
                if time_to_target <= time_threshold_bars:
                    # Check if max loss during trade was acceptable
                    if max_loss_ratio <= loss_threshold:
                        filtered_signals.append(signal)
                        continue

            # Signal is invalid - add invalidation info
            signal_copy = signal.copy()
            signal_copy["invalidated"] = True
            signal_copy["invalidation_reason"] = self._determine_invalidation_reason(
                time_to_target, time_threshold_bars, max_loss_ratio, loss_threshold
            )

            # For metrics calculation, treat as failed signal
            signal_copy["target_reached"] = False
            signal_copy["final_return"] = -max_loss_ratio if max_loss_ratio > 0 else -0.01

            filtered_signals.append(signal_copy)

        return filtered_signals

    def _recalculate_metrics_from_signals(self, signals: List[Dict[str, Any]],
                                        original_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Recalculate performance metrics from filtered signals"""

        if not signals:
            return {
                "total_signals": 0,
                "successful_signals": 0,
                "success_rate": 0.0,
                "total_return": 0.0,
                "expectancy": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0
            }

        # Basic counts
        total_signals = len(signals)
        successful_signals = sum(1 for s in signals if s.get("target_reached", False))
        success_rate = successful_signals / total_signals if total_signals > 0 else 0.0

        # Returns calculation
        returns = [s.get("final_return", 0.0) for s in signals]
        total_return = sum(returns)
        expectancy = np.mean(returns) if returns else 0.0

        # Profit factor
        profits = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        profit_factor = (sum(profits) / sum(losses)) if losses else float('inf')

        # Maximum drawdown (simplified)
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        metrics = {
            "total_signals": total_signals,
            "successful_signals": successful_signals,
            "success_rate": success_rate,
            "total_return": total_return,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown
        }

        # Copy over other metrics that don't depend on signal filtering
        for key, value in original_metrics.items():
            if key not in metrics and key != "signal_details":
                metrics[key] = value

        return metrics

    def _calculate_invalidation_statistics(self, original_signals: List[Dict[str, Any]],
                                         filtered_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about invalidation rule effects"""

        original_count = len(original_signals)
        filtered_count = len([s for s in filtered_signals if not s.get("invalidated", False)])
        invalidated_count = original_count - filtered_count

        # Count invalidation reasons
        time_invalidations = len([s for s in filtered_signals
                                if s.get("invalidated") and "time" in s.get("invalidation_reason", "")])
        loss_invalidations = len([s for s in filtered_signals
                                if s.get("invalidated") and "loss" in s.get("invalidation_reason", "")])

        # Calculate average k_time and k_loss for remaining signals
        valid_signals = [s for s in filtered_signals if not s.get("invalidated", False)]

        if valid_signals:
            avg_time_ratios = [s.get("k_time_actual", 0) for s in valid_signals
                             if "k_time_actual" in s]
            avg_loss_ratios = [s.get("k_loss_actual", 0) for s in valid_signals
                             if "k_loss_actual" in s]

            avg_k_time = np.mean(avg_time_ratios) if avg_time_ratios else 0.0
            avg_k_loss = np.mean(avg_loss_ratios) if avg_loss_ratios else 0.0
        else:
            avg_k_time = 0.0
            avg_k_loss = 0.0

        return {
            "invalidation_rate": invalidated_count / original_count if original_count > 0 else 0.0,
            "signals_invalidated": invalidated_count,
            "time_invalidations": time_invalidations,
            "loss_invalidations": loss_invalidations,
            "avg_k_time_actual": avg_k_time,
            "avg_k_loss_actual": avg_k_loss
        }

    async def _load_historical_patterns(self, pattern_key: str, direction: str,
                                       asset: str, timeframe: str) -> Optional[List[Dict[str, Any]]]:
        """Load historical pattern data for threshold calculation"""

        # This would query the database for historical pattern events
        # For now, return None to indicate no historical data available
        # In real implementation, this would load from pattern_events table

        logger.debug(f"Loading historical patterns for {pattern_key}_{direction}_{asset}_{timeframe}")
        return None

    def _calculate_adaptive_thresholds(self, thresholds: InvalidationThresholds,
                                     historical_data: List[Dict[str, Any]]) -> None:
        """Calculate adaptive thresholds from historical pattern data"""

        if not historical_data:
            return

        # Extract time and loss ratios from successful patterns
        successful_patterns = [p for p in historical_data if p.get("success", False)]

        if len(successful_patterns) < 10:  # Need minimum sample size
            logger.warning(f"Insufficient historical data for {thresholds.pattern_key}: "
                         f"{len(successful_patterns)} successful patterns")
            return

        # Calculate time ratios (time to target / timeframe)
        time_ratios = []
        loss_ratios = []

        for pattern in successful_patterns:
            time_to_target = pattern.get("time_to_target_bars", 0)
            timeframe_bars = self._timeframe_to_bars_per_day(thresholds.timeframe)

            if timeframe_bars > 0:
                time_ratio = time_to_target / timeframe_bars
                time_ratios.append(time_ratio)

            max_adverse_excursion = pattern.get("max_adverse_excursion", 0)
            potential_gain = pattern.get("potential_gain", 1)

            if potential_gain > 0:
                loss_ratio = max_adverse_excursion / potential_gain
                loss_ratios.append(loss_ratio)

        # Calculate quantile thresholds
        if time_ratios:
            thresholds.adaptive_time_threshold = np.percentile(
                time_ratios, thresholds.quantile_threshold * 100
            )

        if loss_ratios:
            thresholds.adaptive_loss_threshold = np.percentile(
                loss_ratios, thresholds.quantile_threshold * 100
            )

        thresholds.sample_size = len(successful_patterns)

        logger.info(
            f"Calculated adaptive thresholds from {thresholds.sample_size} patterns: "
            f"time={thresholds.adaptive_time_threshold:.2f}, "
            f"loss={thresholds.adaptive_loss_threshold:.2f}"
        )

    def _get_cache_key(self, pattern_key: str, direction: str,
                      asset: str, timeframe: str, regime_tag: str = None) -> str:
        """Generate cache key for thresholds"""
        components = [pattern_key, direction, asset, timeframe]
        if regime_tag:
            components.append(regime_tag)
        return "_".join(components)

    def _get_timeframe_duration(self, timeframe: str) -> timedelta:
        """Convert timeframe string to timedelta"""
        timeframe_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1)
        }
        return timeframe_map.get(timeframe, timedelta(hours=1))

    def _timeframe_to_bars_per_day(self, timeframe: str) -> int:
        """Convert timeframe to number of bars per day"""
        timeframe_map = {
            "1m": 1440,  # 24 * 60
            "5m": 288,   # 24 * 12
            "15m": 96,   # 24 * 4
            "30m": 48,   # 24 * 2
            "1h": 24,
            "4h": 6,
            "1d": 1
        }
        return timeframe_map.get(timeframe, 24)

    def _determine_invalidation_reason(self, time_to_target: float, time_threshold: float,
                                     max_loss_ratio: float, loss_threshold: float) -> str:
        """Determine the reason for signal invalidation"""
        time_exceeded = time_to_target > time_threshold
        loss_exceeded = max_loss_ratio > loss_threshold

        if time_exceeded and loss_exceeded:
            return "time_and_loss_exceeded"
        elif time_exceeded:
            return "time_exceeded"
        elif loss_exceeded:
            return "loss_exceeded"
        else:
            return "other"

    def get_invalidation_summary(self, pattern_key: str, direction: str,
                               asset: str, timeframe: str) -> Dict[str, Any]:
        """Get summary of invalidation rules and statistics"""

        cache_key = self._get_cache_key(pattern_key, direction, asset, timeframe)
        thresholds = self.thresholds_cache.get(cache_key)

        if not thresholds:
            return {"error": "No thresholds found for this pattern context"}

        # Calculate recent invalidation statistics
        recent_invalidations = [
            event for event in self.invalidation_history
            if (datetime.now() - event.invalidation_time).days <= 30
        ]

        summary = {
            "pattern_key": pattern_key,
            "direction": direction,
            "asset": asset,
            "timeframe": timeframe,
            "base_time_multiplier": thresholds.k_time_multiplier,
            "base_loss_multiplier": thresholds.k_loss_multiplier,
            "adaptive_time_threshold": thresholds.adaptive_time_threshold,
            "adaptive_loss_threshold": thresholds.adaptive_loss_threshold,
            "quantile_threshold": thresholds.quantile_threshold,
            "sample_size": thresholds.sample_size,
            "last_updated": thresholds.last_updated,
            "recent_invalidations": len(recent_invalidations),
            "invalidation_reasons": {
                "time_exceeded": len([e for e in recent_invalidations
                                    if "time" in e.reason]),
                "loss_exceeded": len([e for e in recent_invalidations
                                    if "loss" in e.reason]),
                "both": len([e for e in recent_invalidations if e.reason == "both"])
            }
        }

        return summary