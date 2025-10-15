"""
Early stopping manager with dynamic thresholds for optimization trials.

This module implements sophisticated early stopping logic that prunes
underperforming trials based on partial results and dynamic performance thresholds.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping behavior"""

    # Minimum requirements before pruning is enabled
    min_trials_for_statistics: int = 10
    min_duration_months: int = 3
    min_trades_for_pruning: int = 10

    # Dynamic threshold parameters
    alpha: float = 0.8  # Multiplier for best observed performance
    min_absolute_threshold: float = 0.1  # Minimum success rate to avoid pruning

    # Performance metric weights for combined scoring
    success_rate_weight: float = 0.6
    expectancy_weight: float = 0.4

    # Temporal considerations
    recent_performance_window: int = 5  # Number of recent trials to consider
    performance_trend_weight: float = 0.2  # Weight for performance trend

@dataclass
class TrialPerformance:
    """Performance tracking for a trial"""

    trial_id: int
    current_duration_months: float
    trades_count: int
    partial_success_rate: float
    partial_expectancy: float
    combined_score: float
    last_updated: datetime

class EarlyStoppingManager:
    """
    Manages early stopping decisions for optimization trials.

    Implements dynamic threshold logic where trials are pruned if they fall
    below a threshold that adapts based on the best performance observed so far.
    """

    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        self.config = config or EarlyStoppingConfig()
        self.trial_performances: Dict[int, TrialPerformance] = {}
        self.best_rolling_performance: Dict[str, float] = {}
        self.performance_history: List[Tuple[datetime, float]] = []

    def should_stop(self, trial_results: List[Dict[str, Any]], alpha: float,
                   min_trades: int, min_duration_months: int) -> bool:
        """
        Determine if optimization should stop early based on convergence.

        Args:
            trial_results: List of trial results so far
            alpha: Dynamic threshold coefficient
            min_trades: Minimum trades required for pruning
            min_duration_months: Minimum duration before pruning

        Returns:
            True if optimization should stop early
        """
        if len(trial_results) < self.config.min_trials_for_statistics:
            return False

        # Update performance tracking
        self._update_performance_tracking(trial_results)

        # Check convergence criteria
        convergence_indicators = self._check_convergence_indicators(trial_results)

        # Stop if we have clear convergence
        if convergence_indicators["strong_convergence"]:
            logger.info("Early stopping: Strong convergence detected")
            return True

        # Stop if we haven't improved in a while and have sufficient trials
        if (convergence_indicators["stagnation"] and
            len(trial_results) >= self.config.min_trials_for_statistics * 2):
            logger.info("Early stopping: Performance stagnation detected")
            return True

        return False

    def should_prune_trial(self, trial_id: int, partial_metrics: Dict[str, Any],
                          all_trial_results: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Determine if a specific trial should be pruned based on partial results.

        Args:
            trial_id: Trial identifier
            partial_metrics: Current partial performance metrics
            all_trial_results: All trial results for context

        Returns:
            (should_prune, reason)
        """
        # Check minimum requirements
        duration_months = partial_metrics.get("partial_duration_months", 0)
        trades_count = partial_metrics.get("partial_trades_count", 0)

        if duration_months < self.config.min_duration_months:
            return False, "Insufficient duration for pruning evaluation"

        if trades_count < self.config.min_trades_for_pruning:
            return False, "Insufficient trades for pruning evaluation"

        # Calculate current performance score
        current_score = self._calculate_performance_score(partial_metrics)

        # Update trial performance tracking
        self._update_trial_performance(trial_id, partial_metrics, current_score)

        # Get dynamic threshold
        threshold = self._get_dynamic_threshold(all_trial_results)

        # Make pruning decision
        if current_score < threshold:
            reason = (f"Performance {current_score:.3f} below dynamic threshold {threshold:.3f}")
            logger.info(f"Pruning trial {trial_id}: {reason}")
            return True, reason

        return False, f"Performance {current_score:.3f} above threshold {threshold:.3f}"

    def _update_performance_tracking(self, trial_results: List[Dict[str, Any]]) -> None:
        """Update internal performance tracking from trial results"""

        completed_trials = [t for t in trial_results if t.get("status") == "completed"]

        if not completed_trials:
            return

        # Calculate recent performance trend
        recent_scores = []
        for trial in completed_trials[-self.config.recent_performance_window:]:
            if "scores" in trial:
                score = trial["scores"].get("combined_score", 0)
                recent_scores.append(score)

        if recent_scores:
            recent_avg = np.mean(recent_scores)
            self.performance_history.append((datetime.now(), recent_avg))

            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.performance_history = [
                (ts, score) for ts, score in self.performance_history if ts > cutoff_time
            ]

    def _update_trial_performance(self, trial_id: int, partial_metrics: Dict[str, Any],
                                 combined_score: float) -> None:
        """Update performance tracking for a specific trial"""

        performance = TrialPerformance(
            trial_id=trial_id,
            current_duration_months=partial_metrics.get("partial_duration_months", 0),
            trades_count=partial_metrics.get("partial_trades_count", 0),
            partial_success_rate=partial_metrics.get("partial_success_rate", 0),
            partial_expectancy=partial_metrics.get("partial_expectancy", 0),
            combined_score=combined_score,
            last_updated=datetime.now()
        )

        self.trial_performances[trial_id] = performance

    def _calculate_performance_score(self, partial_metrics: Dict[str, Any]) -> float:
        """Calculate combined performance score from partial metrics"""

        success_rate = partial_metrics.get("partial_success_rate", 0.0)
        expectancy = partial_metrics.get("partial_expectancy", 0.0)

        # Normalize expectancy to 0-1 range (assuming reasonable bounds)
        normalized_expectancy = max(0, min(1, (expectancy + 0.5) / 1.0))

        # Weighted combination
        score = (self.config.success_rate_weight * success_rate +
                self.config.expectancy_weight * normalized_expectancy)

        return score

    def _get_dynamic_threshold(self, all_trial_results: List[Dict[str, Any]]) -> float:
        """Calculate dynamic pruning threshold based on best observed performance"""

        completed_trials = [t for t in all_trial_results if t.get("status") == "completed"]

        if len(completed_trials) < self.config.min_trials_for_statistics:
            return self.config.min_absolute_threshold

        # Get best rolling performance
        best_scores = []
        for trial in completed_trials:
            if "scores" in trial:
                score = trial["scores"].get("combined_score", 0)
                best_scores.append(score)

        if not best_scores:
            return self.config.min_absolute_threshold

        # Calculate rolling best performance
        rolling_best = np.maximum.accumulate(best_scores)
        current_best = rolling_best[-1]

        # Dynamic threshold is alpha * best_performance
        dynamic_threshold = self.config.alpha * current_best

        # Apply minimum threshold
        final_threshold = max(dynamic_threshold, self.config.min_absolute_threshold)

        return final_threshold

    def _check_convergence_indicators(self, trial_results: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Check various indicators of optimization convergence"""

        completed_trials = [t for t in trial_results if t.get("status") == "completed"]

        if len(completed_trials) < self.config.min_trials_for_statistics:
            return {"strong_convergence": False, "stagnation": False, "sufficient_exploration": False}

        # Extract performance scores
        scores = []
        for trial in completed_trials:
            if "scores" in trial:
                score = trial["scores"].get("combined_score", 0)
                scores.append(score)

        if len(scores) < 5:
            return {"strong_convergence": False, "stagnation": False, "sufficient_exploration": False}

        # Check for strong convergence (low variance in recent performance)
        recent_scores = scores[-10:]  # Last 10 trials
        recent_variance = np.var(recent_scores)
        strong_convergence = recent_variance < 0.001  # Very low variance

        # Check for stagnation (no improvement in best score)
        best_scores = np.maximum.accumulate(scores)
        recent_best = best_scores[-5:]  # Last 5 best scores
        stagnation = len(set(recent_best)) == 1  # No improvement

        # Check if we have sufficient exploration
        score_range = np.max(scores) - np.min(scores)
        sufficient_exploration = score_range > 0.1  # Explored reasonable range

        return {
            "strong_convergence": strong_convergence,
            "stagnation": stagnation,
            "sufficient_exploration": sufficient_exploration,
            "score_variance": recent_variance,
            "score_range": score_range
        }

    def get_pruning_statistics(self) -> Dict[str, Any]:
        """Get statistics about pruning behavior"""

        active_trials = len(self.trial_performances)

        if active_trials == 0:
            return {
                "active_trials": 0,
                "avg_performance": 0.0,
                "performance_variance": 0.0,
                "best_performance": 0.0
            }

        # Calculate statistics from active trials
        performances = [tp.combined_score for tp in self.trial_performances.values()]

        stats = {
            "active_trials": active_trials,
            "avg_performance": np.mean(performances),
            "performance_variance": np.var(performances),
            "best_performance": np.max(performances),
            "worst_performance": np.min(performances),
            "performance_range": np.max(performances) - np.min(performances)
        }

        # Add performance distribution
        if len(performances) >= 5:
            stats["performance_percentiles"] = {
                "25th": np.percentile(performances, 25),
                "50th": np.percentile(performances, 50),
                "75th": np.percentile(performances, 75),
                "90th": np.percentile(performances, 90)
            }

        # Add trend information
        if len(self.performance_history) >= 3:
            recent_scores = [score for _, score in self.performance_history[-10:]]
            if len(recent_scores) >= 3:
                # Simple linear trend
                x = np.arange(len(recent_scores))
                trend_slope = np.polyfit(x, recent_scores, 1)[0]
                stats["performance_trend"] = {
                    "slope": trend_slope,
                    "direction": "improving" if trend_slope > 0.001 else
                                "declining" if trend_slope < -0.001 else "stable"
                }

        return stats

    def reset_for_new_study(self) -> None:
        """Reset state for a new optimization study"""
        self.trial_performances.clear()
        self.best_rolling_performance.clear()
        self.performance_history.clear()
        logger.info("Early stopping manager reset for new study")

    def update_config(self, **kwargs) -> None:
        """Update early stopping configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated early stopping config: {key} = {value}")

    def get_trial_status_summary(self, trial_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific trial"""

        if trial_id not in self.trial_performances:
            return None

        performance = self.trial_performances[trial_id]

        return {
            "trial_id": trial_id,
            "duration_months": performance.current_duration_months,
            "trades_count": performance.trades_count,
            "success_rate": performance.partial_success_rate,
            "expectancy": performance.partial_expectancy,
            "combined_score": performance.combined_score,
            "last_updated": performance.last_updated,
            "meets_min_duration": performance.current_duration_months >= self.config.min_duration_months,
            "meets_min_trades": performance.trades_count >= self.config.min_trades_for_pruning,
            "eligible_for_pruning": (
                performance.current_duration_months >= self.config.min_duration_months and
                performance.trades_count >= self.config.min_trades_for_pruning
            )
        }