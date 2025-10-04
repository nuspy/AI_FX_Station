"""
Conformal Prediction calibration for uncertainty quantification.

Provides distribution-free, finite-sample valid prediction intervals
using split conformal prediction method.
"""
from __future__ import annotations

from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ConformalCalibrationResult:
    """Results from conformal calibration."""
    alpha: float  # Miscoverage level (e.g., 0.05 for 95% coverage)
    calibrated_quantiles: Tuple[float, float]  # (lower, upper) quantile values
    conformity_scores: np.ndarray  # Calibration set conformity scores
    coverage_guarantee: float  # Theoretical coverage (1 - alpha)
    empirical_coverage: float  # Actual coverage on calibration set
    interval_width: float  # Average interval width
    n_calibration: int  # Number of calibration samples


class SplitConformalPredictor:
    """
    Split conformal prediction for regression.

    Provides calibrated prediction intervals with finite-sample guarantees.
    Coverage guarantee: P(Y ∈ [ŷ - q_lo, ŷ + q_hi]) ≥ 1 - α

    Reference:
    Lei et al. (2018) "Distribution-Free Predictive Inference for Regression"
    """

    def __init__(self, alpha: float = 0.05, method: str = 'split'):
        """
        Initialize conformal predictor.

        Args:
            alpha: Miscoverage level (e.g., 0.05 for 95% coverage)
            method: Conformal method ('split' or 'jackknife')
        """
        self.alpha = alpha
        self.method = method
        self.calibrated_quantiles: Optional[Tuple[float, float]] = None
        self.conformity_scores: Optional[np.ndarray] = None

    def calibrate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        symmetric: bool = True
    ) -> ConformalCalibrationResult:
        """
        Calibrate prediction intervals on calibration set.

        Args:
            predictions: Model predictions on calibration set (N,)
            actuals: Actual values on calibration set (N,)
            symmetric: Whether to use symmetric intervals

        Returns:
            ConformalCalibrationResult with calibration info
        """
        if len(predictions) != len(actuals):
            raise ValueError("predictions and actuals must have same length")

        n = len(predictions)

        # Compute conformity scores (absolute residuals)
        residuals = np.abs(actuals - predictions)
        self.conformity_scores = residuals

        # Compute quantile for conformal intervals
        # For finite-sample guarantee, use (1 - alpha)(1 + 1/n) quantile
        adjusted_level = min(1.0, (1 - self.alpha) * (1 + 1 / n))
        q = np.quantile(residuals, adjusted_level)

        if symmetric:
            # Symmetric intervals: [ŷ - q, ŷ + q]
            self.calibrated_quantiles = (q, q)
        else:
            # Asymmetric intervals: use quantile regression approach
            # Lower and upper quantiles
            q_lo = np.quantile(residuals, self.alpha / 2)
            q_hi = np.quantile(residuals, 1 - self.alpha / 2)
            self.calibrated_quantiles = (q_lo, q_hi)

        # Compute empirical coverage on calibration set
        if symmetric:
            covered = np.abs(actuals - predictions) <= q
        else:
            lower = predictions - q_lo
            upper = predictions + q_hi
            covered = (actuals >= lower) & (actuals <= upper)

        empirical_coverage = np.mean(covered)

        # Average interval width
        if symmetric:
            avg_width = 2 * q
        else:
            avg_width = q_lo + q_hi

        result = ConformalCalibrationResult(
            alpha=self.alpha,
            calibrated_quantiles=self.calibrated_quantiles,
            conformity_scores=self.conformity_scores,
            coverage_guarantee=1 - self.alpha,
            empirical_coverage=empirical_coverage,
            interval_width=avg_width,
            n_calibration=n
        )

        logger.info(
            f"Conformal calibration: coverage={empirical_coverage:.3f} "
            f"(target={1-self.alpha:.3f}), width={avg_width:.6f}"
        )

        return result

    def predict(
        self,
        predictions: np.ndarray,
        return_intervals: bool = True
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate conformal prediction intervals.

        Args:
            predictions: Model predictions (N,)
            return_intervals: Whether to return intervals

        Returns:
            (predictions, (lower_bounds, upper_bounds)) if return_intervals
            else predictions
        """
        if self.calibrated_quantiles is None:
            raise RuntimeError("Must call calibrate() before predict()")

        q_lo, q_hi = self.calibrated_quantiles

        if not return_intervals:
            return predictions, None

        # Compute intervals
        lower = predictions - q_lo
        upper = predictions + q_hi

        return predictions, (lower, upper)


class AdaptiveConformalPredictor(SplitConformalPredictor):
    """
    Adaptive conformal prediction with time-varying coverage.

    Adjusts quantiles dynamically based on recent conformity scores.
    Useful for non-stationary time series.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        window_size: int = 100,
        adaptation_rate: float = 0.01
    ):
        """
        Initialize adaptive conformal predictor.

        Args:
            alpha: Target miscoverage level
            window_size: Sliding window size for quantile estimation
            adaptation_rate: Learning rate for quantile adjustment
        """
        super().__init__(alpha=alpha)
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.residual_buffer: List[float] = []
        self.current_quantile: Optional[float] = None

    def update(self, prediction: float, actual: float):
        """
        Update adaptive quantile with new observation.

        Args:
            prediction: Model prediction
            actual: Actual value
        """
        # Compute residual
        residual = abs(actual - prediction)

        # Add to buffer
        self.residual_buffer.append(residual)

        # Maintain window size
        if len(self.residual_buffer) > self.window_size:
            self.residual_buffer.pop(0)

        # Recompute quantile from buffer
        if len(self.residual_buffer) >= 10:  # Minimum samples
            adjusted_level = min(1.0, (1 - self.alpha) * (1 + 1 / len(self.residual_buffer)))
            self.current_quantile = np.quantile(self.residual_buffer, adjusted_level)

    def predict_adaptive(self, prediction: float) -> Tuple[float, float, float]:
        """
        Generate adaptive conformal interval.

        Args:
            prediction: Point prediction

        Returns:
            (prediction, lower_bound, upper_bound)
        """
        if self.current_quantile is None:
            # Not enough data yet, use conservative interval
            q = 0.1  # Placeholder
        else:
            q = self.current_quantile

        return prediction, prediction - q, prediction + q


def evaluate_coverage(
    predictions: np.ndarray,
    intervals: Tuple[np.ndarray, np.ndarray],
    actuals: np.ndarray
) -> dict:
    """
    Evaluate empirical coverage of prediction intervals.

    Args:
        predictions: Point predictions
        intervals: (lower_bounds, upper_bounds)
        actuals: Actual values

    Returns:
        Dictionary with coverage metrics
    """
    lower, upper = intervals

    # Check coverage
    covered = (actuals >= lower) & (actuals <= upper)
    coverage = np.mean(covered)

    # Interval widths
    widths = upper - lower
    avg_width = np.mean(widths)
    std_width = np.std(widths)

    # Tightness: average relative width
    # Avoid division by zero
    relative_widths = widths / (np.abs(actuals) + 1e-8)
    avg_relative_width = np.mean(relative_widths)

    # Undercoverage and overcoverage rates
    undercoverage = np.mean(actuals < lower)
    overcoverage = np.mean(actuals > upper)

    return {
        'coverage': coverage,
        'avg_width': avg_width,
        'std_width': std_width,
        'avg_relative_width': avg_relative_width,
        'undercoverage_rate': undercoverage,
        'overcoverage_rate': overcoverage,
        'n_samples': len(predictions)
    }


def compute_conformity_scores(
    predictions: np.ndarray,
    actuals: np.ndarray,
    method: str = 'absolute'
) -> np.ndarray:
    """
    Compute conformity scores for conformal prediction.

    Args:
        predictions: Model predictions
        actuals: Actual values
        method: Conformity score method ('absolute', 'normalized', 'cqr')

    Returns:
        Conformity scores
    """
    if method == 'absolute':
        # Absolute residuals
        return np.abs(actuals - predictions)

    elif method == 'normalized':
        # Normalized by prediction scale
        scale = np.abs(predictions) + 1e-8
        return np.abs(actuals - predictions) / scale

    elif method == 'cqr':
        # Conformalized Quantile Regression (requires quantile predictions)
        raise NotImplementedError("CQR requires quantile predictions")

    else:
        raise ValueError(f"Unknown method: {method}")


def split_calibration_test(
    data: pd.DataFrame,
    split_ratio: float = 0.5,
    stratify_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into calibration and test sets.

    Args:
        data: DataFrame with predictions and actuals
        split_ratio: Fraction for calibration set
        stratify_column: Optional column for stratified split

    Returns:
        (calibration_df, test_df)
    """
    n = len(data)
    n_cal = int(n * split_ratio)

    if stratify_column is not None:
        # Stratified split
        from sklearn.model_selection import train_test_split
        cal_df, test_df = train_test_split(
            data,
            train_size=split_ratio,
            stratify=data[stratify_column],
            random_state=42
        )
    else:
        # Random split
        indices = np.random.permutation(n)
        cal_indices = indices[:n_cal]
        test_indices = indices[n_cal:]

        cal_df = data.iloc[cal_indices]
        test_df = data.iloc[test_indices]

    logger.info(f"Split: {len(cal_df)} calibration, {len(test_df)} test samples")

    return cal_df, test_df
