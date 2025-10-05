"""
Model Drift Detection

Monitors model performance and data distribution drift:
- KL Divergence for distribution shifts
- Performance degradation tracking
- Automated retraining triggers

Reference: "Streaming Data" by Andrew Psaltis
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from loguru import logger


@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_type: str  # "performance", "data_drift", "concept_drift"
    severity: str  # "low", "medium", "high", "critical"
    metric_name: str
    baseline_value: float
    current_value: float
    threshold: float
    timestamp: str
    recommendation: str


class DriftDetector:
    """
    Detects model and data drift for production monitoring.

    Triggers alerts when model performance degrades or input distribution shifts.
    """

    def __init__(
        self,
        # Performance drift thresholds
        mae_degradation_threshold: float = 0.2,  # 20% worse = alert
        sharpe_degradation_threshold: float = 0.3,  # 30% worse Sharpe = alert

        # Data drift thresholds
        kl_div_threshold: float = 0.1,  # KL divergence threshold
        js_div_threshold: float = 0.15,  # Jensen-Shannon distance threshold
        psi_threshold: float = 0.2,  # Population Stability Index threshold

        # Monitoring windows
        baseline_window: int = 100,  # Baseline window size
        monitoring_window: int = 50,  # Current window for comparison
        min_samples: int = 30,  # Minimum samples before checking
    ):
        """
        Initialize drift detector.

        Args:
            mae_degradation_threshold: Max acceptable MAE increase (ratio)
            sharpe_degradation_threshold: Max acceptable Sharpe decrease (ratio)
            kl_div_threshold: KL divergence alert threshold
            js_div_threshold: Jensen-Shannon distance threshold
            psi_threshold: Population Stability Index threshold
            baseline_window: Size of baseline distribution window
            monitoring_window: Size of current monitoring window
            min_samples: Minimum samples before drift detection
        """
        self.mae_threshold = mae_degradation_threshold
        self.sharpe_threshold = sharpe_degradation_threshold
        self.kl_threshold = kl_div_threshold
        self.js_threshold = js_div_threshold
        self.psi_threshold = psi_threshold

        self.baseline_window = baseline_window
        self.monitoring_window = monitoring_window
        self.min_samples = min_samples

        # Baseline distributions
        self.baseline_features: Optional[pd.DataFrame] = None
        self.baseline_performance: Optional[Dict[str, float]] = None

        # Alert history
        self.alerts: List[DriftAlert] = []

    def set_baseline(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actual: np.ndarray,
    ):
        """
        Set baseline distribution and performance.

        Args:
            features: Baseline feature matrix
            predictions: Model predictions on baseline
            actual: Actual values for baseline
        """
        # Store baseline features
        self.baseline_features = features.iloc[-self.baseline_window:].copy()

        # Calculate baseline performance
        mae = np.mean(np.abs(actual - predictions))
        mse = np.mean((actual - predictions) ** 2)
        rmse = np.sqrt(mse)

        # Sharpe-like ratio for predictions
        returns_pred = np.diff(predictions)
        sharpe = (
            np.mean(returns_pred) / (np.std(returns_pred) + 1e-10)
            if len(returns_pred) > 0
            else 0.0
        )

        self.baseline_performance = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "sharpe": sharpe,
            "n_samples": len(actual),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Baseline set: MAE={mae:.6f}, Sharpe={sharpe:.4f}, N={len(actual)}")

    def calculate_kl_divergence(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 20,
    ) -> float:
        """
        Calculate KL divergence between two distributions.

        Args:
            baseline: Baseline samples
            current: Current samples
            bins: Number of bins for histograms

        Returns:
            KL divergence (0 = identical, higher = more drift)
        """
        # Create histograms
        range_min = min(baseline.min(), current.min())
        range_max = max(baseline.max(), current.max())

        hist_baseline, _ = np.histogram(
            baseline, bins=bins, range=(range_min, range_max), density=True
        )
        hist_current, _ = np.histogram(
            current, bins=bins, range=(range_min, range_max), density=True
        )

        # Add small epsilon to avoid log(0)
        hist_baseline = hist_baseline + 1e-10
        hist_current = hist_current + 1e-10

        # Normalize
        hist_baseline = hist_baseline / hist_baseline.sum()
        hist_current = hist_current / hist_current.sum()

        # KL divergence
        kl_div = stats.entropy(hist_current, hist_baseline)

        return float(kl_div)

    def calculate_js_distance(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 20,
    ) -> float:
        """
        Calculate Jensen-Shannon distance (symmetric version of KL).

        Args:
            baseline: Baseline samples
            current: Current samples
            bins: Number of bins

        Returns:
            JS distance (0-1, 0 = identical)
        """
        # Create histograms
        range_min = min(baseline.min(), current.min())
        range_max = max(baseline.max(), current.max())

        hist_baseline, _ = np.histogram(
            baseline, bins=bins, range=(range_min, range_max), density=True
        )
        hist_current, _ = np.histogram(
            current, bins=bins, range=(range_min, range_max), density=True
        )

        # Normalize
        hist_baseline = hist_baseline + 1e-10
        hist_current = hist_current + 1e-10
        hist_baseline = hist_baseline / hist_baseline.sum()
        hist_current = hist_current / hist_current.sum()

        # Jensen-Shannon distance
        js_dist = jensenshannon(hist_baseline, hist_current)

        return float(js_dist)

    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures distribution shift:
        - PSI < 0.1: No significant shift
        - 0.1 ≤ PSI < 0.2: Moderate shift
        - PSI ≥ 0.2: Significant shift

        Args:
            baseline: Baseline samples
            current: Current samples
            bins: Number of bins

        Returns:
            PSI value
        """
        # Create quantile-based bins from baseline
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.quantile(baseline, quantiles)

        # Get distributions
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
        current_dist, _ = np.histogram(current, bins=bin_edges)

        # Convert to percentages
        baseline_pct = baseline_dist / len(baseline) + 1e-10
        current_pct = current_dist / len(current) + 1e-10

        # PSI calculation
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return float(psi)

    def check_data_drift(
        self,
        current_features: pd.DataFrame,
    ) -> List[DriftAlert]:
        """
        Check for data distribution drift.

        Args:
            current_features: Current feature matrix

        Returns:
            List of drift alerts
        """
        if self.baseline_features is None:
            logger.warning("No baseline set - cannot check drift")
            return []

        if len(current_features) < self.min_samples:
            return []

        alerts = []
        current_window = current_features.iloc[-self.monitoring_window:]

        # Check each feature
        for col in self.baseline_features.columns:
            if col not in current_window.columns:
                continue

            baseline_vals = self.baseline_features[col].dropna().values
            current_vals = current_window[col].dropna().values

            if len(baseline_vals) < 10 or len(current_vals) < 10:
                continue

            # KL Divergence
            kl_div = self.calculate_kl_divergence(baseline_vals, current_vals)

            if kl_div > self.kl_threshold:
                severity = "high" if kl_div > self.kl_threshold * 2 else "medium"

                alerts.append(DriftAlert(
                    alert_type="data_drift",
                    severity=severity,
                    metric_name=f"kl_div_{col}",
                    baseline_value=0.0,
                    current_value=kl_div,
                    threshold=self.kl_threshold,
                    timestamp=datetime.now().isoformat(),
                    recommendation=(
                        f"Feature '{col}' distribution shifted (KL={kl_div:.4f}). "
                        "Consider retraining model with recent data."
                    ),
                ))

            # Jensen-Shannon distance
            js_dist = self.calculate_js_distance(baseline_vals, current_vals)

            if js_dist > self.js_threshold:
                alerts.append(DriftAlert(
                    alert_type="data_drift",
                    severity="medium",
                    metric_name=f"js_dist_{col}",
                    baseline_value=0.0,
                    current_value=js_dist,
                    threshold=self.js_threshold,
                    timestamp=datetime.now().isoformat(),
                    recommendation=f"Feature '{col}' JS distance={js_dist:.4f}. Monitor closely.",
                ))

            # PSI
            psi = self.calculate_psi(baseline_vals, current_vals)

            if psi > self.psi_threshold:
                severity = "critical" if psi > 0.25 else "high"

                alerts.append(DriftAlert(
                    alert_type="data_drift",
                    severity=severity,
                    metric_name=f"psi_{col}",
                    baseline_value=0.0,
                    current_value=psi,
                    threshold=self.psi_threshold,
                    timestamp=datetime.now().isoformat(),
                    recommendation=(
                        f"Feature '{col}' PSI={psi:.4f} indicates significant shift. "
                        "Retrain model immediately."
                    ),
                ))

        return alerts

    def check_performance_drift(
        self,
        predictions: np.ndarray,
        actual: np.ndarray,
    ) -> List[DriftAlert]:
        """
        Check for model performance degradation.

        Args:
            predictions: Recent predictions
            actual: Actual values

        Returns:
            List of performance alerts
        """
        if self.baseline_performance is None:
            logger.warning("No baseline performance set")
            return []

        if len(predictions) < self.min_samples:
            return []

        alerts = []

        # Calculate current performance
        current_mae = np.mean(np.abs(actual - predictions))
        baseline_mae = self.baseline_performance["mae"]

        # MAE degradation check
        mae_ratio = current_mae / (baseline_mae + 1e-10)
        if mae_ratio > (1 + self.mae_threshold):
            severity = "critical" if mae_ratio > 1.5 else "high"

            alerts.append(DriftAlert(
                alert_type="performance",
                severity=severity,
                metric_name="mae",
                baseline_value=baseline_mae,
                current_value=current_mae,
                threshold=baseline_mae * (1 + self.mae_threshold),
                timestamp=datetime.now().isoformat(),
                recommendation=(
                    f"MAE degraded {(mae_ratio - 1) * 100:.1f}% "
                    f"(baseline={baseline_mae:.6f}, current={current_mae:.6f}). "
                    "Retrain model."
                ),
            ))

        # Sharpe degradation check
        returns_pred = np.diff(predictions)
        current_sharpe = (
            np.mean(returns_pred) / (np.std(returns_pred) + 1e-10)
            if len(returns_pred) > 0
            else 0.0
        )
        baseline_sharpe = self.baseline_performance.get("sharpe", 0.0)

        if baseline_sharpe > 0:
            sharpe_ratio = current_sharpe / (baseline_sharpe + 1e-10)
            if sharpe_ratio < (1 - self.sharpe_threshold):
                alerts.append(DriftAlert(
                    alert_type="performance",
                    severity="high",
                    metric_name="sharpe",
                    baseline_value=baseline_sharpe,
                    current_value=current_sharpe,
                    threshold=baseline_sharpe * (1 - self.sharpe_threshold),
                    timestamp=datetime.now().isoformat(),
                    recommendation=(
                        f"Sharpe degraded {(1 - sharpe_ratio) * 100:.1f}% "
                        f"(baseline={baseline_sharpe:.4f}, current={current_sharpe:.4f})."
                    ),
                ))

        return alerts

    def monitor(
        self,
        current_features: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        actual: Optional[np.ndarray] = None,
    ) -> Dict[str, any]:
        """
        Run comprehensive drift monitoring.

        Args:
            current_features: Current feature matrix
            predictions: Recent predictions (optional)
            actual: Actual values (optional)

        Returns:
            Monitoring report with alerts
        """
        data_alerts = self.check_data_drift(current_features)

        perf_alerts = []
        if predictions is not None and actual is not None:
            perf_alerts = self.check_performance_drift(predictions, actual)

        all_alerts = data_alerts + perf_alerts

        # Store alerts
        self.alerts.extend(all_alerts)

        # Determine overall status
        critical_count = sum(1 for a in all_alerts if a.severity == "critical")
        high_count = sum(1 for a in all_alerts if a.severity == "high")

        if critical_count > 0:
            status = "CRITICAL - Retrain immediately"
        elif high_count > 2:
            status = "WARNING - Retrain recommended"
        elif len(all_alerts) > 0:
            status = "ATTENTION - Monitor closely"
        else:
            status = "OK"

        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "alerts": [
                {
                    "type": a.alert_type,
                    "severity": a.severity,
                    "metric": a.metric_name,
                    "baseline": a.baseline_value,
                    "current": a.current_value,
                    "threshold": a.threshold,
                    "recommendation": a.recommendation,
                }
                for a in all_alerts
            ],
            "alert_counts": {
                "critical": critical_count,
                "high": high_count,
                "medium": sum(1 for a in all_alerts if a.severity == "medium"),
                "low": sum(1 for a in all_alerts if a.severity == "low"),
            },
        }
