"""
Enhanced Conformal Prediction Calibration

Extends the existing calibration system with:
- Increased calibration window (200 → 500 trades)
- Asymmetric calibration for upside vs downside
- Regime-specific calibration deltas
- Adaptive recalibration triggers
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import json


@dataclass
class AsymmetricCalibrationResult:
    """Asymmetric calibration for upside and downside predictions"""
    upside_delta: float  # Calibration adjustment for bullish predictions
    downside_delta: float  # Calibration adjustment for bearish predictions
    coverage_upside: float  # Actual coverage for upside predictions
    coverage_downside: float  # Actual coverage for downside predictions
    n_samples_upside: int
    n_samples_downside: int
    regime: Optional[str] = None


@dataclass
class CalibrationMetrics:
    """Quality metrics for calibration"""
    coverage_accuracy: float  # How close actual coverage is to theoretical
    interval_sharpness: float  # Average interval width (narrower is better)
    miscalibration_score: float  # Overall miscalibration (lower is better)
    adaptation_trigger: bool  # Whether recalibration is needed


class EnhancedConformalCalibrator:
    """
    Enhanced conformal prediction with asymmetric and regime-specific calibration.

    Improvements over basic calibration:
    - Larger calibration window (500 trades vs 200)
    - Separate calibration for upside/downside
    - Regime-specific calibration deltas
    - Adaptive recalibration based on coverage drift
    """

    def __init__(
        self,
        calibration_window: int = 500,
        min_samples_per_regime: int = 100,
        target_coverage: float = 0.90,
        recalibration_threshold: float = 0.10,
        sharpness_penalty: float = 0.5
    ):
        """
        Initialize enhanced calibrator.

        Args:
            calibration_window: Number of recent trades for calibration
            min_samples_per_regime: Minimum samples needed per regime
            target_coverage: Target coverage level (e.g., 0.90 for 90%)
            recalibration_threshold: Coverage drift threshold for recalibration
            sharpness_penalty: Weight for interval width in quality score
        """
        self.calibration_window = calibration_window
        self.min_samples_per_regime = min_samples_per_regime
        self.target_coverage = target_coverage
        self.recalibration_threshold = recalibration_threshold
        self.sharpness_penalty = sharpness_penalty

        # Calibration state
        self.regime_calibrations: Dict[str, AsymmetricCalibrationResult] = {}
        self.global_calibration: Optional[AsymmetricCalibrationResult] = None
        self.last_calibration_time: Optional[datetime] = None

    def calibrate_asymmetric(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        directions: np.ndarray,  # 1 for upside, -1 for downside
        confidence_level: float = 0.90,
        regime: Optional[str] = None
    ) -> AsymmetricCalibrationResult:
        """
        Compute asymmetric calibration deltas for upside and downside predictions.

        Args:
            predictions: Point predictions
            actuals: Actual outcomes
            directions: Signal directions (+1/-1)
            confidence_level: Target coverage level
            regime: Optional regime label

        Returns:
            Asymmetric calibration result
        """
        # Split by direction
        upside_mask = directions > 0
        downside_mask = directions < 0

        predictions_up = predictions[upside_mask]
        actuals_up = actuals[upside_mask]
        predictions_down = predictions[downside_mask]
        actuals_down = actuals[downside_mask]

        # Compute residuals
        residuals_up = actuals_up - predictions_up
        residuals_down = actuals_down - predictions_down

        # Calculate quantiles for conformal intervals
        alpha = 1 - confidence_level

        if len(residuals_up) > 0:
            # For upside, we care about the upper quantile (how much we underestimate)
            upside_delta = np.quantile(np.abs(residuals_up), 1 - alpha)
            # Calculate coverage
            interval_width_up = 2 * upside_delta
            coverage_up = np.mean(np.abs(residuals_up) <= upside_delta)
        else:
            upside_delta = 0.0
            coverage_up = 0.0

        if len(residuals_down) > 0:
            # For downside, we care about the lower quantile (how much we overestimate)
            downside_delta = np.quantile(np.abs(residuals_down), 1 - alpha)
            # Calculate coverage
            interval_width_down = 2 * downside_delta
            coverage_down = np.mean(np.abs(residuals_down) <= downside_delta)
        else:
            downside_delta = 0.0
            coverage_down = 0.0

        return AsymmetricCalibrationResult(
            upside_delta=float(upside_delta),
            downside_delta=float(downside_delta),
            coverage_upside=float(coverage_up),
            coverage_downside=float(coverage_down),
            n_samples_upside=int(np.sum(upside_mask)),
            n_samples_downside=int(np.sum(downside_mask)),
            regime=regime
        )

    def calibrate_by_regime(
        self,
        df: pd.DataFrame,
        prediction_col: str = 'prediction',
        actual_col: str = 'actual',
        direction_col: str = 'direction',
        regime_col: str = 'regime',
        confidence_level: float = 0.90
    ) -> Dict[str, AsymmetricCalibrationResult]:
        """
        Compute regime-specific asymmetric calibrations.

        Args:
            df: DataFrame with predictions, actuals, directions, and regimes
            prediction_col: Column name for predictions
            actual_col: Column name for actual outcomes
            direction_col: Column name for signal directions
            regime_col: Column name for regime labels
            confidence_level: Target coverage

        Returns:
            Dictionary mapping regime to calibration results
        """
        regime_results = {}

        # Get unique regimes
        regimes = df[regime_col].unique()

        for regime in regimes:
            regime_df = df[df[regime_col] == regime]

            # Skip if insufficient samples
            if len(regime_df) < self.min_samples_per_regime:
                continue

            predictions = regime_df[prediction_col].values
            actuals = regime_df[actual_col].values
            directions = regime_df[direction_col].values

            result = self.calibrate_asymmetric(
                predictions=predictions,
                actuals=actuals,
                directions=directions,
                confidence_level=confidence_level,
                regime=regime
            )

            regime_results[regime] = result

        # Also compute global calibration
        if len(df) >= self.min_samples_per_regime:
            global_result = self.calibrate_asymmetric(
                predictions=df[prediction_col].values,
                actuals=df[actual_col].values,
                directions=df[direction_col].values,
                confidence_level=confidence_level,
                regime='global'
            )
            regime_results['global'] = global_result

        # Update state
        self.regime_calibrations = regime_results
        if 'global' in regime_results:
            self.global_calibration = regime_results['global']
        self.last_calibration_time = datetime.now()

        return regime_results

    def get_prediction_interval(
        self,
        point_prediction: float,
        direction: float,  # +1 or -1
        regime: Optional[str] = None,
        confidence_level: float = 0.90
    ) -> Tuple[float, float]:
        """
        Get calibrated prediction interval.

        Args:
            point_prediction: Point forecast
            direction: Signal direction (+1 upside, -1 downside)
            regime: Current regime
            confidence_level: Desired coverage level

        Returns:
            (lower_bound, upper_bound)
        """
        # Get appropriate calibration
        if regime and regime in self.regime_calibrations:
            calib = self.regime_calibrations[regime]
        elif self.global_calibration:
            calib = self.global_calibration
        else:
            # No calibration available, use default wide interval
            default_delta = 0.02  # 2% default
            return (point_prediction - default_delta, point_prediction + default_delta)

        # Apply asymmetric delta based on direction
        if direction > 0:
            delta = calib.upside_delta
        else:
            delta = calib.downside_delta

        # Adjust for confidence level if different from calibration
        # Simple scaling (more sophisticated methods possible)
        alpha = 1 - confidence_level
        calib_alpha = 1 - self.target_coverage
        if calib_alpha > 0:
            delta = delta * (alpha / calib_alpha)

        lower = point_prediction - delta
        upper = point_prediction + delta

        return (lower, upper)

    def evaluate_calibration_quality(
        self,
        df: pd.DataFrame,
        prediction_col: str = 'prediction',
        actual_col: str = 'actual',
        direction_col: str = 'direction',
        regime_col: str = 'regime'
    ) -> CalibrationMetrics:
        """
        Evaluate quality of current calibration.

        Args:
            df: Recent data to evaluate
            prediction_col: Predictions column
            actual_col: Actuals column
            direction_col: Directions column
            regime_col: Regime column

        Returns:
            Calibration quality metrics
        """
        if len(df) == 0:
            return CalibrationMetrics(
                coverage_accuracy=0.0,
                interval_sharpness=1.0,
                miscalibration_score=1.0,
                adaptation_trigger=True
            )

        # Calculate actual coverage with current calibration
        coverages = []
        interval_widths = []

        for idx, row in df.iterrows():
            pred = row[prediction_col]
            actual = row[actual_col]
            direction = row[direction_col]
            regime = row[regime_col] if regime_col in row.index else None

            lower, upper = self.get_prediction_interval(pred, direction, regime)
            interval_width = upper - lower
            covered = (lower <= actual <= upper)

            coverages.append(covered)
            interval_widths.append(interval_width)

        actual_coverage = np.mean(coverages)
        avg_interval_width = np.mean(interval_widths)

        # Coverage accuracy: how close to target
        coverage_accuracy = 1.0 - abs(actual_coverage - self.target_coverage)

        # Interval sharpness: normalized average width (smaller is better)
        # Normalize by typical price range
        price_std = df[actual_col].std()
        if price_std > 0:
            interval_sharpness = avg_interval_width / (2 * price_std)
        else:
            interval_sharpness = 1.0

        # Miscalibration score: combines coverage error and sharpness
        coverage_error = abs(actual_coverage - self.target_coverage)
        miscalibration_score = coverage_error + self.sharpness_penalty * interval_sharpness

        # Trigger recalibration if coverage drifts too much
        adaptation_trigger = coverage_error > self.recalibration_threshold

        return CalibrationMetrics(
            coverage_accuracy=coverage_accuracy,
            interval_sharpness=interval_sharpness,
            miscalibration_score=miscalibration_score,
            adaptation_trigger=adaptation_trigger
        )

    def should_recalibrate(
        self,
        recent_data: pd.DataFrame,
        **eval_kwargs
    ) -> Tuple[bool, CalibrationMetrics]:
        """
        Determine if recalibration is needed.

        Args:
            recent_data: Recent prediction/actual data
            **eval_kwargs: Arguments for evaluate_calibration_quality

        Returns:
            (should_recalibrate, metrics)
        """
        metrics = self.evaluate_calibration_quality(recent_data, **eval_kwargs)
        return metrics.adaptation_trigger, metrics

    def to_dict(self) -> Dict[str, Any]:
        """Export calibration state"""
        return {
            'calibration_window': self.calibration_window,
            'min_samples_per_regime': self.min_samples_per_regime,
            'target_coverage': self.target_coverage,
            'recalibration_threshold': self.recalibration_threshold,
            'sharpness_penalty': self.sharpness_penalty,
            'regime_calibrations': {
                k: {
                    'upside_delta': v.upside_delta,
                    'downside_delta': v.downside_delta,
                    'coverage_upside': v.coverage_upside,
                    'coverage_downside': v.coverage_downside,
                    'n_samples_upside': v.n_samples_upside,
                    'n_samples_downside': v.n_samples_downside,
                    'regime': v.regime
                }
                for k, v in self.regime_calibrations.items()
            },
            'last_calibration_time': self.last_calibration_time.isoformat() if self.last_calibration_time else None
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'EnhancedConformalCalibrator':
        """Load calibrator from dictionary"""
        calibrator = cls(
            calibration_window=config.get('calibration_window', 500),
            min_samples_per_regime=config.get('min_samples_per_regime', 100),
            target_coverage=config.get('target_coverage', 0.90),
            recalibration_threshold=config.get('recalibration_threshold', 0.10),
            sharpness_penalty=config.get('sharpness_penalty', 0.5)
        )

        # Restore regime calibrations
        if 'regime_calibrations' in config:
            for regime, calib_dict in config['regime_calibrations'].items():
                calibrator.regime_calibrations[regime] = AsymmetricCalibrationResult(**calib_dict)

        if 'global' in calibrator.regime_calibrations:
            calibrator.global_calibration = calibrator.regime_calibrations['global']

        if 'last_calibration_time' in config and config['last_calibration_time']:
            calibrator.last_calibration_time = datetime.fromisoformat(config['last_calibration_time'])

        return calibrator


def integrate_with_existing_calibrator(
    enhanced_calibrator: EnhancedConformalCalibrator,
    historical_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Helper function to integrate enhanced calibrator with existing system.

    Args:
        enhanced_calibrator: Enhanced calibrator instance
        historical_data: Historical prediction/actual data

    Returns:
        Integration results and statistics
    """
    # Perform regime-specific calibration
    regime_results = enhanced_calibrator.calibrate_by_regime(historical_data)

    # Evaluate quality
    metrics = enhanced_calibrator.evaluate_calibration_quality(historical_data)

    # Prepare results
    integration_results = {
        'calibration_successful': len(regime_results) > 0,
        'regimes_calibrated': list(regime_results.keys()),
        'quality_metrics': {
            'coverage_accuracy': metrics.coverage_accuracy,
            'interval_sharpness': metrics.interval_sharpness,
            'miscalibration_score': metrics.miscalibration_score,
            'adaptation_needed': metrics.adaptation_trigger
        },
        'regime_details': {
            regime: {
                'upside_delta': result.upside_delta,
                'downside_delta': result.downside_delta,
                'coverage_upside': result.coverage_upside,
                'coverage_downside': result.coverage_downside,
                'n_samples': result.n_samples_upside + result.n_samples_downside
            }
            for regime, result in regime_results.items()
        }
    }

    return integration_results
