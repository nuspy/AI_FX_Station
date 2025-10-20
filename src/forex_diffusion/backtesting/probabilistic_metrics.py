"""
Probabilistic Metrics for Forecast Evaluation
Advanced statistical metrics for evaluating probabilistic forecasts from generative models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
from enum import Enum

import logging
logger = logging.getLogger(__name__)

class CalibrationMethod(Enum):
    """Methods for calibration assessment"""
    PIT = "pit"  # Probability Integral Transform
    RELIABILITY = "reliability"  # Reliability diagram
    SHARPNESS = "sharpness"  # Sharpness assessment

@dataclass
class ProbabilisticResult:
    """Container for probabilistic forecast evaluation results"""
    crps: float
    log_score: float
    brier_score: float
    pit_uniformity_pvalue: float
    calibration_error: float
    sharpness: float
    resolution: float

    # Interval-specific metrics
    interval_coverage: Dict[str, float]
    interval_width: Dict[str, float]

    # Meta information
    sample_size: int
    forecast_horizon: str
    evaluation_period: Tuple[str, str]

class ProbabilisticMetrics:
    """
    Advanced probabilistic metrics for forecast evaluation.

    Implements state-of-the-art forecast evaluation methods:
    - CRPS (Continuous Ranked Probability Score)
    - Logarithmic Score
    - Brier Score for multiple thresholds
    - PIT (Probability Integral Transform) uniformity
    - Calibration assessment (reliability, sharpness, resolution)
    - Interval coverage and width metrics
    """

    def __init__(self, quantile_levels: Optional[List[float]] = None):
        self.quantile_levels = quantile_levels or [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        self.prediction_intervals = [
            (0.05, 0.95),  # 90% interval
            (0.1, 0.9),    # 80% interval
            (0.25, 0.75),  # 50% interval
        ]

    def continuous_ranked_probability_score(self,
                                          forecast_quantiles: Dict[str, float],
                                          observation: float) -> float:
        """
        Calculate Continuous Ranked Probability Score (CRPS).

        CRPS measures the accuracy of probabilistic forecasts.
        Lower values indicate better forecasts.

        Args:
            forecast_quantiles: Dict with quantile forecasts (e.g., {'q05': 1.1, 'q50': 1.15, 'q95': 1.2})
            observation: Actual observed value

        Returns:
            CRPS value (lower is better)
        """
        try:
            # Extract quantiles and values
            quantiles = []
            values = []

            for q_name, q_value in forecast_quantiles.items():
                if q_name.startswith('q') and len(q_name) > 1:
                    q_level = float(q_name[1:]) / 100.0
                    if 0 <= q_level <= 1:
                        quantiles.append(q_level)
                        values.append(q_value)

            if len(quantiles) < 3:
                logger.warning("Insufficient quantiles for CRPS calculation")
                return float('inf')

            # Sort by quantile level
            sorted_pairs = sorted(zip(quantiles, values))
            quantiles = [p[0] for p in sorted_pairs]
            values = [p[1] for p in sorted_pairs]

            # Calculate CRPS using quantile approximation
            crps = 0.0

            for i in range(len(quantiles)):
                q = quantiles[i]
                forecast_val = values[i]

                # Heaviside function: H(forecast_val - observation)
                heaviside = 1.0 if forecast_val >= observation else 0.0

                # CRPS integral approximation
                crps += (heaviside - q) * (forecast_val - observation)

            return abs(crps / len(quantiles))

        except Exception as e:
            logger.error(f"CRPS calculation failed: {e}")
            return float('inf')

    def logarithmic_score(self,
                         forecast_distribution: Dict[str, float],
                         observation: float,
                         epsilon: float = 1e-10) -> float:
        """
        Calculate logarithmic score (log score).

        Measures the quality of probabilistic predictions.
        Higher values indicate better forecasts.

        Args:
            forecast_distribution: Forecast parameters or quantiles
            observation: Actual observed value
            epsilon: Small value to avoid log(0)

        Returns:
            Log score (higher is better)
        """
        try:
            # Estimate density at observation from quantiles
            density = self._estimate_density_from_quantiles(forecast_distribution, observation)

            # Logarithmic score = log(p(observation))
            log_score = np.log(max(density, epsilon))

            return log_score

        except Exception as e:
            logger.error(f"Log score calculation failed: {e}")
            return float('-inf')

    def _estimate_density_from_quantiles(self,
                                       quantiles: Dict[str, float],
                                       point: float) -> float:
        """Estimate probability density at a point using quantile forecasts"""

        # Extract quantile levels and values
        q_levels = []
        q_values = []

        for q_name, q_val in quantiles.items():
            if q_name.startswith('q'):
                q_level = float(q_name[1:]) / 100.0
                q_levels.append(q_level)
                q_values.append(q_val)

        if len(q_levels) < 3:
            return 0.001  # Default low density

        # Sort by quantile level
        sorted_pairs = sorted(zip(q_levels, q_values))
        q_levels = [p[0] for p in sorted_pairs]
        q_values = [p[1] for p in sorted_pairs]

        # Find interval containing the point
        for i in range(len(q_values) - 1):
            if q_values[i] <= point <= q_values[i + 1]:
                # Estimate density in this interval
                interval_width = q_values[i + 1] - q_values[i]
                prob_mass = q_levels[i + 1] - q_levels[i]

                if interval_width > 0:
                    return prob_mass / interval_width

        # Point outside quantile range - assign low density
        return 0.001

    def brier_score(self,
                   forecast_probabilities: List[float],
                   binary_outcomes: List[int],
                   thresholds: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate Brier Score for multiple probability thresholds.

        Args:
            forecast_probabilities: Predicted probabilities for each threshold
            binary_outcomes: Binary outcomes (0 or 1) for each threshold
            thresholds: Thresholds used to create binary outcomes

        Returns:
            Dict with Brier scores for each threshold
        """
        if thresholds is None:
            thresholds = [0.5]  # Default threshold

        if len(forecast_probabilities) != len(binary_outcomes):
            raise ValueError("Forecast probabilities and outcomes must have same length")

        brier_scores = {}

        for i, threshold in enumerate(thresholds):
            # Brier Score = mean((forecast_prob - outcome)^2)
            squared_diffs = [(p - o) ** 2 for p, o in zip(forecast_probabilities, binary_outcomes)]
            brier_score = np.mean(squared_diffs)

            brier_scores[f'threshold_{threshold}'] = brier_score

        return brier_scores

    def probability_integral_transform(self,
                                     forecasts: List[Dict[str, float]],
                                     observations: List[float]) -> Tuple[List[float], float]:
        """
        Calculate Probability Integral Transform (PIT) values and uniformity test.

        PIT values should be uniformly distributed for well-calibrated forecasts.

        Args:
            forecasts: List of quantile forecasts
            observations: List of actual observations

        Returns:
            Tuple of (PIT values, uniformity test p-value)
        """
        pit_values = []

        for forecast, obs in zip(forecasts, observations):
            # Calculate PIT value for this observation
            pit_val = self._calculate_pit_value(forecast, obs)
            pit_values.append(pit_val)

        # Test uniformity using Kolmogorov-Smirnov test
        if len(pit_values) >= 10:
            ks_stat, p_value = stats.kstest(pit_values, 'uniform')
            return pit_values, p_value
        else:
            return pit_values, 0.0

    def _calculate_pit_value(self, quantiles: Dict[str, float], observation: float) -> float:
        """Calculate PIT value for a single observation"""

        # Extract and sort quantiles
        q_levels = []
        q_values = []

        for q_name, q_val in quantiles.items():
            if q_name.startswith('q'):
                q_level = float(q_name[1:]) / 100.0
                q_levels.append(q_level)
                q_values.append(q_val)

        if len(q_levels) < 2:
            return 0.5  # Default to median if insufficient data

        sorted_pairs = sorted(zip(q_levels, q_values))
        q_levels = [p[0] for p in sorted_pairs]
        q_values = [p[1] for p in sorted_pairs]

        # Find PIT value by interpolation
        if observation <= q_values[0]:
            return 0.0
        elif observation >= q_values[-1]:
            return 1.0
        else:
            # Linear interpolation between quantiles
            for i in range(len(q_values) - 1):
                if q_values[i] <= observation <= q_values[i + 1]:
                    # Interpolate
                    weight = (observation - q_values[i]) / (q_values[i + 1] - q_values[i])
                    pit_value = q_levels[i] + weight * (q_levels[i + 1] - q_levels[i])
                    return max(0.0, min(1.0, pit_value))

        return 0.5  # Fallback

    def interval_coverage_and_width(self,
                                  forecasts: List[Dict[str, float]],
                                  observations: List[float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate prediction interval coverage rates and average widths.

        Args:
            forecasts: List of quantile forecasts
            observations: List of actual observations

        Returns:
            Dict with coverage and width metrics for each interval
        """
        results = {}

        for lower_q, upper_q in self.prediction_intervals:
            lower_name = f'q{int(lower_q * 100):02d}'
            upper_name = f'q{int(upper_q * 100):02d}'

            coverage_count = 0
            widths = []
            valid_count = 0

            for forecast, obs in zip(forecasts, observations):
                if lower_name in forecast and upper_name in forecast:
                    lower_bound = forecast[lower_name]
                    upper_bound = forecast[upper_name]

                    # Check coverage
                    if lower_bound <= obs <= upper_bound:
                        coverage_count += 1

                    # Calculate width
                    widths.append(upper_bound - lower_bound)
                    valid_count += 1

            if valid_count > 0:
                nominal_coverage = upper_q - lower_q
                empirical_coverage = coverage_count / valid_count
                avg_width = np.mean(widths)

                results[f'{int(lower_q*100)}-{int(upper_q*100)}%'] = {
                    'empirical_coverage': empirical_coverage,
                    'nominal_coverage': nominal_coverage,
                    'coverage_error': abs(empirical_coverage - nominal_coverage),
                    'average_width': avg_width,
                    'width_std': np.std(widths),
                    'sample_size': valid_count
                }

        return results

    def calibration_assessment(self,
                             forecasts: List[Dict[str, float]],
                             observations: List[float]) -> Dict[str, float]:
        """
        Comprehensive calibration assessment using reliability-sharpness decomposition.

        Args:
            forecasts: List of quantile forecasts
            observations: List of actual observations

        Returns:
            Dict with calibration metrics (reliability, sharpness, resolution)
        """

        # Calculate PIT values for calibration assessment
        pit_values, uniformity_p = self.probability_integral_transform(forecasts, observations)

        # Reliability (calibration error)
        reliability = self._calculate_reliability(pit_values)

        # Sharpness (average prediction interval width)
        sharpness = self._calculate_sharpness(forecasts)

        # Resolution (ability to discriminate between different outcomes)
        resolution = self._calculate_resolution(forecasts, observations)

        return {
            'reliability': reliability,
            'sharpness': sharpness,
            'resolution': resolution,
            'pit_uniformity_pvalue': uniformity_p,
            'calibration_error': reliability  # Alias for clarity
        }

    def _calculate_reliability(self, pit_values: List[float], n_bins: int = 10) -> float:
        """Calculate reliability (calibration error) from PIT values"""

        if len(pit_values) < n_bins:
            return float('inf')

        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        reliability = 0.0
        total_count = len(pit_values)

        for i in range(n_bins):
            # Count observations in this bin
            in_bin = [(pit_val >= bin_edges[i] and pit_val < bin_edges[i + 1]) for pit_val in pit_values]
            bin_count = sum(in_bin)

            if bin_count > 0:
                # Expected frequency vs observed frequency
                expected_freq = 1.0 / n_bins
                observed_freq = bin_count / total_count

                # Weighted squared error
                reliability += bin_count * (observed_freq - expected_freq) ** 2

        return reliability / total_count

    def _calculate_sharpness(self, forecasts: List[Dict[str, float]]) -> float:
        """Calculate sharpness (average prediction uncertainty)"""

        interval_widths = []

        # Use 90% prediction interval for sharpness
        lower_name = 'q05'
        upper_name = 'q95'

        for forecast in forecasts:
            if lower_name in forecast and upper_name in forecast:
                width = forecast[upper_name] - forecast[lower_name]
                interval_widths.append(width)

        return np.mean(interval_widths) if interval_widths else float('inf')

    def _calculate_resolution(self,
                            forecasts: List[Dict[str, float]],
                            observations: List[float]) -> float:
        """Calculate resolution (ability to discriminate)"""

        # Use variance of median forecasts as proxy for resolution
        median_forecasts = []

        for forecast in forecasts:
            median_val = forecast.get('q50', forecast.get('q25', 0))
            median_forecasts.append(median_val)

        if len(median_forecasts) < 2:
            return 0.0

        return np.var(median_forecasts)

    def comprehensive_evaluation(self,
                               forecasts: List[Dict[str, float]],
                               observations: List[float],
                               forecast_horizon: str = "1h",
                               evaluation_period: Optional[Tuple[str, str]] = None) -> ProbabilisticResult:
        """
        Run comprehensive probabilistic forecast evaluation.

        Args:
            forecasts: List of quantile forecasts
            observations: List of actual observations
            forecast_horizon: Forecast horizon string
            evaluation_period: Start and end dates for evaluation

        Returns:
            ProbabilisticResult with all metrics
        """

        if len(forecasts) != len(observations):
            raise ValueError("Forecasts and observations must have same length")

        # Calculate all metrics
        logger.info(f"Running comprehensive evaluation for {len(forecasts)} forecasts")

        # CRPS
        crps_scores = []
        for forecast, obs in zip(forecasts, observations):
            crps = self.continuous_ranked_probability_score(forecast, obs)
            if not np.isinf(crps):
                crps_scores.append(crps)
        avg_crps = np.mean(crps_scores) if crps_scores else float('inf')

        # Log Score
        log_scores = []
        for forecast, obs in zip(forecasts, observations):
            log_score = self.logarithmic_score(forecast, obs)
            if not np.isinf(log_score):
                log_scores.append(log_score)
        avg_log_score = np.mean(log_scores) if log_scores else float('-inf')

        # Brier Score (simplified - using median forecast)
        median_probs = [f.get('q50', 0.5) for f in forecasts]
        median_obs = np.median(observations)
        binary_outcomes = [1 if obs > median_obs else 0 for obs in observations]
        brier_scores = self.brier_score(median_probs, binary_outcomes)
        avg_brier = np.mean(list(brier_scores.values()))

        # PIT uniformity
        pit_values, pit_p_value = self.probability_integral_transform(forecasts, observations)

        # Interval coverage and width
        interval_metrics = self.interval_coverage_and_width(forecasts, observations)

        # Coverage rates and widths
        coverage_rates = {k: v['empirical_coverage'] for k, v in interval_metrics.items()}
        interval_widths = {k: v['average_width'] for k, v in interval_metrics.items()}

        # Calibration assessment
        calibration_metrics = self.calibration_assessment(forecasts, observations)

        # Evaluation period
        if evaluation_period is None:
            evaluation_period = ("unknown", "unknown")

        return ProbabilisticResult(
            crps=avg_crps,
            log_score=avg_log_score,
            brier_score=avg_brier,
            pit_uniformity_pvalue=pit_p_value,
            calibration_error=calibration_metrics['reliability'],
            sharpness=calibration_metrics['sharpness'],
            resolution=calibration_metrics['resolution'],
            interval_coverage=coverage_rates,
            interval_width=interval_widths,
            sample_size=len(forecasts),
            forecast_horizon=forecast_horizon,
            evaluation_period=evaluation_period
        )

# Utility functions

def compare_probabilistic_forecasts(results_a: ProbabilisticResult,
                                  results_b: ProbabilisticResult) -> Dict[str, float]:
    """Compare two probabilistic forecast results"""

    return {
        'crps_improvement': (results_a.crps - results_b.crps) / results_a.crps * 100,
        'log_score_improvement': (results_b.log_score - results_a.log_score) / abs(results_a.log_score) * 100,
        'calibration_improvement': (results_a.calibration_error - results_b.calibration_error) / results_a.calibration_error * 100,
        'sharpness_improvement': (results_a.sharpness - results_b.sharpness) / results_a.sharpness * 100,
    }

def generate_probabilistic_report(result: ProbabilisticResult) -> str:
    """Generate human-readable report for probabilistic evaluation"""

    report = []
    report.append("=" * 60)
    report.append("PROBABILISTIC FORECAST EVALUATION REPORT")
    report.append("=" * 60)
    report.append(f"Forecast Horizon: {result.forecast_horizon}")
    report.append(f"Evaluation Period: {result.evaluation_period[0]} to {result.evaluation_period[1]}")
    report.append(f"Sample Size: {result.sample_size}")
    report.append("")

    report.append("ACCURACY METRICS:")
    report.append(f"  CRPS: {result.crps:.6f} (lower is better)")
    report.append(f"  Log Score: {result.log_score:.6f} (higher is better)")
    report.append(f"  Brier Score: {result.brier_score:.6f} (lower is better)")
    report.append("")

    report.append("CALIBRATION ASSESSMENT:")
    report.append(f"  PIT Uniformity p-value: {result.pit_uniformity_pvalue:.4f}")
    report.append(f"  Calibration Error: {result.calibration_error:.6f}")
    report.append(f"  Sharpness: {result.sharpness:.6f}")
    report.append(f"  Resolution: {result.resolution:.6f}")
    report.append("")

    report.append("PREDICTION INTERVALS:")
    for interval, coverage in result.interval_coverage.items():
        width = result.interval_width.get(interval, 0)
        report.append(f"  {interval}: Coverage {coverage:.3f}, Width {width:.6f}")

    # Interpretation
    report.append("")
    report.append("INTERPRETATION:")

    if result.pit_uniformity_pvalue > 0.05:
        report.append("  ✓ Forecasts appear well-calibrated (PIT uniformity)")
    else:
        report.append("  ⚠ Forecasts may be poorly calibrated (PIT non-uniform)")

    avg_coverage_error = np.mean([abs(c - 0.9) for c in result.interval_coverage.values()])
    if avg_coverage_error < 0.05:
        report.append("  ✓ Prediction intervals well-calibrated")
    else:
        report.append("  ⚠ Prediction intervals may be miscalibrated")

    return "\n".join(report)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Probabilistic Metrics...")

    metrics = ProbabilisticMetrics()

    # Create sample forecast data
    np.random.seed(42)
    n_forecasts = 100

    forecasts = []
    observations = []

    for i in range(n_forecasts):
        # Simulate forecast quantiles
        center = 1.1 + np.random.normal(0, 0.01)
        forecast = {
            'q05': center - 0.005,
            'q25': center - 0.002,
            'q50': center,
            'q75': center + 0.002,
            'q95': center + 0.005
        }
        forecasts.append(forecast)

        # Simulate observation (with some bias for testing)
        obs = center + np.random.normal(0, 0.003)
        observations.append(obs)

    # Run comprehensive evaluation
    result = metrics.comprehensive_evaluation(
        forecasts,
        observations,
        forecast_horizon="1h",
        evaluation_period=("2024-01-01", "2024-01-31")
    )

    # Generate report
    report = generate_probabilistic_report(result)
    print(report)