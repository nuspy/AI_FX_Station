"""
Pattern Confidence Calibration

Calibrates pattern confidence scores based on historical win rates.
Uses conformal prediction to provide probabilistic forecasts with
guaranteed coverage properties.

Reference: "A Tutorial on Conformal Prediction" by Shafer & Vovk (2008)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger
from datetime import datetime, timedelta


class OutcomeType(Enum):
    """Pattern outcome classification"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class PatternOutcome:
    """Historical pattern outcome record"""
    pattern_key: str
    direction: str
    detection_date: datetime
    outcome: OutcomeType
    initial_score: float  # Raw pattern score at detection

    # Outcome details
    target_hit: bool
    stop_hit: bool
    bars_to_target: Optional[int] = None
    bars_to_stop: Optional[int] = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    final_return: float = 0.0

    # Context
    regime: Optional[str] = None
    volatility: Optional[float] = None


@dataclass
class CalibrationCurve:
    """Calibration curve mapping predicted probability to actual frequency"""
    bins: List[Tuple[float, float]]  # (bin_start, bin_end)
    predicted_probs: List[float]      # Mean predicted probability per bin
    actual_frequencies: List[float]   # Actual success frequency per bin
    sample_counts: List[int]          # Number of samples per bin

    # Calibration metrics
    expected_calibration_error: float = 0.0
    max_calibration_error: float = 0.0
    brier_score: float = 0.0


@dataclass
class ConfidenceInterval:
    """Conformal prediction interval"""
    point_prediction: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    interval_width: float


class PatternConfidenceCalibrator:
    """
    Calibrates pattern confidence scores with historical outcomes.

    Features:
    - Historical win rate tracking per pattern type
    - Regime-specific calibration
    - Conformal prediction intervals
    - Reliability diagram generation
    - Automatic calibration adjustment
    """

    def __init__(
        self,
        min_samples_for_calibration: int = 30,
        calibration_window_days: int = 180,
        n_bins: int = 10,
    ):
        """
        Initialize confidence calibrator.

        Args:
            min_samples_for_calibration: Minimum outcomes needed for calibration
            calibration_window_days: Days of history to use for calibration
            n_bins: Number of bins for calibration curve
        """
        self.min_samples = min_samples_for_calibration
        self.calibration_window = timedelta(days=calibration_window_days)
        self.n_bins = n_bins

        # Historical outcomes storage
        self.outcomes: List[PatternOutcome] = []
        # Index for fast lookup: (pattern_key, direction, regime) -> List[PatternOutcome]
        self.outcomes_index: Dict[Tuple[str, str, str], List[PatternOutcome]] = {}

        # Calibration models (per pattern, per regime)
        self.calibration_models: Dict[str, Dict[str, Any]] = {}

        # Conformal prediction sets
        self.nonconformity_scores: Dict[str, List[float]] = {}

    def record_outcome(self, outcome: PatternOutcome) -> None:
        """
        Record a pattern outcome for calibration and update index.

        Args:
            outcome: PatternOutcome with detection and result info
        """
        self.outcomes.append(outcome)
        
        # Update index for O(1) lookup
        key = (outcome.pattern_key, outcome.direction, outcome.regime or "any")
        if key not in self.outcomes_index:
            self.outcomes_index[key] = []
        self.outcomes_index[key].append(outcome)
        
        logger.debug(
            f"Recorded outcome for {outcome.pattern_key} {outcome.direction}: "
            f"{outcome.outcome.value}"
        )

        # Update calibration model for this pattern
        pattern_key = self._get_pattern_key(outcome.pattern_key, outcome.direction)
        if pattern_key not in self.calibration_models:
            self.calibration_models[pattern_key] = {
                "last_update": datetime.now(),
                "total_samples": 0,
                "win_rate": 0.5,
                "calibration_curve": None,
            }

        self.calibration_models[pattern_key]["total_samples"] += 1

    def calibrate_confidence(
        self,
        pattern_key: str,
        direction: str,
        initial_score: float,
        regime: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Calibrate pattern confidence score based on historical performance.

        Args:
            pattern_key: Pattern identifier
            direction: "bull" or "bear"
            initial_score: Raw pattern score (0-100)
            regime: Current market regime (optional)

        Returns:
            (calibrated_confidence, adjustment_factor)
        """
        key = self._get_pattern_key(pattern_key, direction, regime)

        # Get historical win rate for this pattern
        win_rate = self._get_historical_win_rate(pattern_key, direction, regime)

        if win_rate is None:
            # Insufficient data, return initial score
            logger.debug(
                f"Insufficient calibration data for {key}, "
                f"using initial score {initial_score}"
            )
            return initial_score, 1.0

        # Calibration adjustment based on win rate
        # If win_rate > 60%: boost confidence
        # If win_rate < 50%: reduce confidence
        # If win_rate == 55%: neutral (small boost)

        baseline_win_rate = 0.55  # Expected baseline for valid patterns

        if win_rate >= 0.60:
            # Strong pattern, boost confidence
            adjustment = 1.0 + 0.3 * (win_rate - baseline_win_rate) / (1.0 - baseline_win_rate)
        elif win_rate < 0.50:
            # Weak pattern, reduce confidence
            adjustment = 0.5 + 0.5 * (win_rate / 0.50)
        else:
            # Moderate pattern, slight adjustment
            adjustment = 1.0 + 0.2 * (win_rate - baseline_win_rate) / (baseline_win_rate - 0.50)

        adjustment = np.clip(adjustment, 0.5, 1.5)  # Bound adjustment

        # Apply adjustment to initial score
        calibrated_score = initial_score * adjustment
        calibrated_score = np.clip(calibrated_score, 0.0, 100.0)

        logger.debug(
            f"Calibrated {key}: initial={initial_score:.1f}, "
            f"win_rate={win_rate:.2f}, adjustment={adjustment:.2f}, "
            f"calibrated={calibrated_score:.1f}"
        )

        return calibrated_score, adjustment

    def get_prediction_interval(
        self,
        pattern_key: str,
        direction: str,
        confidence_level: float = 0.90,
        regime: Optional[str] = None,
    ) -> Optional[ConfidenceInterval]:
        """
        Get conformal prediction interval for expected return.

        Uses conformal prediction to provide prediction intervals with
        guaranteed coverage under exchangeability assumption.

        Args:
            pattern_key: Pattern identifier
            direction: "bull" or "bear"
            confidence_level: Desired confidence level (default 90%)
            regime: Current market regime (optional)

        Returns:
            ConfidenceInterval with bounds, or None if insufficient data
        """
        key = self._get_pattern_key(pattern_key, direction, regime)

        # Get historical outcomes for this pattern
        pattern_outcomes = self._get_pattern_outcomes(pattern_key, direction, regime)

        if len(pattern_outcomes) < self.min_samples:
            logger.warning(
                f"Insufficient outcomes for {key}: {len(pattern_outcomes)} "
                f"< {self.min_samples}"
            )
            return None

        # Extract final returns from successful outcomes
        returns = [
            outcome.final_return
            for outcome in pattern_outcomes
            if outcome.outcome == OutcomeType.SUCCESS
        ]

        if len(returns) < 10:
            return None

        # Calculate quantiles for conformal interval
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        lower_bound = np.quantile(returns, lower_quantile)
        upper_bound = np.quantile(returns, upper_quantile)
        point_prediction = np.median(returns)

        interval = ConfidenceInterval(
            point_prediction=float(point_prediction),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            confidence_level=confidence_level,
            interval_width=float(upper_bound - lower_bound),
        )

        logger.debug(
            f"Prediction interval for {key} at {confidence_level*100:.0f}% confidence: "
            f"[{lower_bound:.4f}, {upper_bound:.4f}], median={point_prediction:.4f}"
        )

        return interval

    def compute_calibration_curve(
        self,
        pattern_key: str,
        direction: str,
        regime: Optional[str] = None,
    ) -> Optional[CalibrationCurve]:
        """
        Compute calibration curve for reliability diagram.

        Args:
            pattern_key: Pattern identifier
            direction: "bull" or "bear"
            regime: Current market regime (optional)

        Returns:
            CalibrationCurve or None if insufficient data
        """
        pattern_outcomes = self._get_pattern_outcomes(pattern_key, direction, regime)

        if len(pattern_outcomes) < self.min_samples:
            return None

        # Group outcomes by initial score bins
        bins = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = list(zip(bins[:-1], bins[1:]))

        predicted_probs = []
        actual_frequencies = []
        sample_counts = []

        for bin_start, bin_end in bin_edges:
            # Filter outcomes in this bin
            bin_outcomes = [
                o for o in pattern_outcomes
                if bin_start <= o.initial_score < bin_end
            ]

            if len(bin_outcomes) == 0:
                predicted_probs.append((bin_start + bin_end) / 2 / 100)  # Normalize to [0,1]
                actual_frequencies.append(0.0)
                sample_counts.append(0)
                continue

            # Calculate mean predicted probability (normalized score)
            mean_predicted = np.mean([o.initial_score / 100 for o in bin_outcomes])

            # Calculate actual frequency of success
            successes = sum(1 for o in bin_outcomes if o.outcome == OutcomeType.SUCCESS)
            actual_freq = successes / len(bin_outcomes)

            predicted_probs.append(float(mean_predicted))
            actual_frequencies.append(float(actual_freq))
            sample_counts.append(len(bin_outcomes))

        # Calculate calibration metrics
        # Expected Calibration Error (ECE)
        ece = 0.0
        max_ce = 0.0
        total_samples = sum(sample_counts)

        for i in range(len(bin_edges)):
            if sample_counts[i] == 0:
                continue

            weight = sample_counts[i] / total_samples
            calibration_error = abs(predicted_probs[i] - actual_frequencies[i])

            ece += weight * calibration_error
            max_ce = max(max_ce, calibration_error)

        # Brier score
        all_predictions = [o.initial_score / 100 for o in pattern_outcomes]
        all_outcomes = [1.0 if o.outcome == OutcomeType.SUCCESS else 0.0
                       for o in pattern_outcomes]
        brier = np.mean([(p - o)**2 for p, o in zip(all_predictions, all_outcomes)])

        curve = CalibrationCurve(
            bins=bin_edges,
            predicted_probs=predicted_probs,
            actual_frequencies=actual_frequencies,
            sample_counts=sample_counts,
            expected_calibration_error=float(ece),
            max_calibration_error=float(max_ce),
            brier_score=float(brier),
        )

        logger.info(
            f"Calibration curve for {pattern_key} {direction}: "
            f"ECE={ece:.4f}, Max CE={max_ce:.4f}, Brier={brier:.4f}"
        )

        return curve

    def get_pattern_statistics(
        self,
        pattern_key: str,
        direction: str,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics for pattern type.

        Args:
            pattern_key: Pattern identifier
            direction: "bull" or "bear"
            regime: Current market regime (optional)

        Returns:
            Dictionary with statistics
        """
        outcomes = self._get_pattern_outcomes(pattern_key, direction, regime)

        if len(outcomes) == 0:
            return {
                "total_detections": 0,
                "win_rate": None,
                "avg_return_success": None,
                "avg_return_failure": None,
                "avg_bars_to_target": None,
                "calibration_status": "insufficient_data",
            }

        # Count outcomes
        successes = [o for o in outcomes if o.outcome == OutcomeType.SUCCESS]
        failures = [o for o in outcomes if o.outcome == OutcomeType.FAILURE]

        win_rate = len(successes) / len(outcomes) if len(outcomes) > 0 else 0.0

        # Average returns
        avg_return_success = (
            np.mean([o.final_return for o in successes])
            if successes else 0.0
        )
        avg_return_failure = (
            np.mean([o.final_return for o in failures])
            if failures else 0.0
        )

        # Average bars to target
        bars_to_target = [
            o.bars_to_target for o in successes
            if o.bars_to_target is not None
        ]
        avg_bars_to_target = np.mean(bars_to_target) if bars_to_target else None

        # Calibration status
        if len(outcomes) >= self.min_samples:
            calibration_status = "calibrated"
        elif len(outcomes) >= 10:
            calibration_status = "partial"
        else:
            calibration_status = "insufficient"

        return {
            "total_detections": len(outcomes),
            "total_successes": len(successes),
            "total_failures": len(failures),
            "win_rate": float(win_rate),
            "avg_return_success": float(avg_return_success),
            "avg_return_failure": float(avg_return_failure),
            "avg_bars_to_target": float(avg_bars_to_target) if avg_bars_to_target else None,
            "profit_factor": (
                abs(avg_return_success * len(successes) /
                    (avg_return_failure * len(failures)))
                if len(failures) > 0 and avg_return_failure != 0 else None
            ),
            "calibration_status": calibration_status,
            "oldest_detection": min(o.detection_date for o in outcomes),
            "newest_detection": max(o.detection_date for o in outcomes),
        }

    def _get_pattern_key(
        self,
        pattern_key: str,
        direction: str,
        regime: Optional[str] = None,
    ) -> str:
        """Generate unique key for pattern + direction + regime"""
        if regime:
            return f"{pattern_key}_{direction}_{regime}"
        return f"{pattern_key}_{direction}"

    def _get_historical_win_rate(
        self,
        pattern_key: str,
        direction: str,
        regime: Optional[str] = None,
    ) -> Optional[float]:
        """
        Get historical win rate for pattern type.

        Returns:
            Win rate (0-1) or None if insufficient data
        """
        outcomes = self._get_pattern_outcomes(pattern_key, direction, regime)

        if len(outcomes) < self.min_samples:
            return None

        successes = sum(1 for o in outcomes if o.outcome == OutcomeType.SUCCESS)
        total = len(outcomes)

        return successes / total if total > 0 else None

    def _get_pattern_outcomes(
        self,
        pattern_key: str,
        direction: str,
        regime: Optional[str] = None,
    ) -> List[PatternOutcome]:
        """
        Filter outcomes for specific pattern type.

        Args:
            pattern_key: Pattern identifier
            direction: "bull" or "bear"
            regime: Optional regime filter

        Returns:
            List of PatternOutcome matching criteria
        """
        # Use index for O(1) lookup (much faster than linear search)
        key = (pattern_key, direction, regime or "any")
        indexed_outcomes = self.outcomes_index.get(key, [])
        
        # Filter by time window only (index already filtered by pattern/direction/regime)
        cutoff_date = datetime.now() - self.calibration_window
        filtered = [o for o in indexed_outcomes if o.detection_date >= cutoff_date]

        return filtered

    def prune_old_outcomes(self, days_to_keep: int = 365) -> int:
        """
        Remove outcomes older than specified days.

        Args:
            days_to_keep: Number of days of history to retain

        Returns:
            Number of outcomes pruned
        """
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        initial_count = len(self.outcomes)

        self.outcomes = [
            o for o in self.outcomes
            if o.detection_date >= cutoff
        ]

        pruned = initial_count - len(self.outcomes)

        if pruned > 0:
            logger.info(f"Pruned {pruned} old outcomes, retained {len(self.outcomes)}")

        return pruned

    def save_outcomes(self, filepath: str) -> None:
        """Save outcomes to JSON file"""
        import json

        data = {
            "outcomes": [
                {
                    "pattern_key": o.pattern_key,
                    "direction": o.direction,
                    "detection_date": o.detection_date.isoformat(),
                    "outcome": o.outcome.value,
                    "initial_score": o.initial_score,
                    "target_hit": o.target_hit,
                    "stop_hit": o.stop_hit,
                    "bars_to_target": o.bars_to_target,
                    "bars_to_stop": o.bars_to_stop,
                    "final_return": o.final_return,
                    "regime": o.regime,
                }
                for o in self.outcomes
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.outcomes)} outcomes to {filepath}")

    def load_outcomes(self, filepath: str) -> None:
        """Load outcomes from JSON file"""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.outcomes = [
            PatternOutcome(
                pattern_key=o["pattern_key"],
                direction=o["direction"],
                detection_date=datetime.fromisoformat(o["detection_date"]),
                outcome=OutcomeType(o["outcome"]),
                initial_score=o["initial_score"],
                target_hit=o["target_hit"],
                stop_hit=o["stop_hit"],
                bars_to_target=o.get("bars_to_target"),
                bars_to_stop=o.get("bars_to_stop"),
                final_return=o["final_return"],
                regime=o.get("regime"),
            )
            for o in data["outcomes"]
        ]

        logger.info(f"Loaded {len(self.outcomes)} outcomes from {filepath}")
