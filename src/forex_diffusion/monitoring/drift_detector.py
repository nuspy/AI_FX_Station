"""
Model Drift Detection System

Detects distribution shifts and performance degradation using
Kolmogorov-Smirnov tests and composite drift scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import ks_2samp
from loguru import logger


@dataclass
class DriftReport:
    """Drift detection report."""
    overall_drift_score: float
    feature_drift_scores: Dict[str, float]
    performance_drift_score: float
    alert_level: str
    drifted_features: List[str]
    recommendations: List[str]


class DriftDetector:
    """Detects model drift through distribution shift analysis."""

    def __init__(
        self,
        warning_threshold: float = 0.3,
        critical_threshold: float = 0.5,
        ks_significance: float = 0.05
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.ks_significance = ks_significance

    def detect_drift(
        self,
        train_features: pd.DataFrame,
        prod_features: pd.DataFrame,
        train_performance: Optional[Dict[str, float]] = None,
        prod_performance: Optional[Dict[str, float]] = None
    ) -> DriftReport:
        logger.info("Detecting drift...")

        feature_drift = self._compute_feature_drift(train_features, prod_features)
        overall_drift = np.mean(list(feature_drift.values()))

        drifted_features = [
            feat for feat, score in feature_drift.items()
            if score > self.warning_threshold
        ]

        perf_drift = 0.0
        if train_performance and prod_performance:
            perf_drift = self._compute_performance_drift(
                train_performance,
                prod_performance
            )

        alert_level = self._determine_alert_level(overall_drift, perf_drift)
        recommendations = self._generate_recommendations(
            overall_drift,
            drifted_features,
            alert_level
        )

        logger.info(f"Drift detected: score={overall_drift:.3f}, alert={alert_level}")

        return DriftReport(
            overall_drift_score=overall_drift,
            feature_drift_scores=feature_drift,
            performance_drift_score=perf_drift,
            alert_level=alert_level,
            drifted_features=drifted_features,
            recommendations=recommendations
        )

    def _compute_feature_drift(
        self,
        train_features: pd.DataFrame,
        prod_features: pd.DataFrame
    ) -> Dict[str, float]:
        drift_scores = {}
        common_features = set(train_features.columns) & set(prod_features.columns)

        for feature in common_features:
            train_values = train_features[feature].dropna()
            prod_values = prod_features[feature].dropna()

            if len(train_values) < 30 or len(prod_values) < 30:
                continue

            statistic, pvalue = ks_2samp(train_values, prod_values)
            drift_score = statistic if pvalue < self.ks_significance else statistic * 0.5
            drift_scores[feature] = drift_score

        return drift_scores

    def _compute_performance_drift(
        self,
        train_metrics: Dict[str, float],
        prod_metrics: Dict[str, float]
    ) -> float:
        common_metrics = set(train_metrics.keys()) & set(prod_metrics.keys())

        if not common_metrics:
            return 0.0

        degradations = []

        for metric in common_metrics:
            train_val = train_metrics[metric]
            prod_val = prod_metrics[metric]

            if train_val == 0:
                continue

            if 'r2' in metric.lower() or 'accuracy' in metric.lower():
                degradation = max(0, (train_val - prod_val) / abs(train_val))
            else:
                degradation = max(0, (prod_val - train_val) / abs(train_val))

            degradations.append(degradation)

        if degradations:
            return min(1.0, np.mean(degradations))
        else:
            return 0.0

    def _determine_alert_level(self, drift_score: float, perf_drift: float) -> str:
        max_drift = max(drift_score, perf_drift)

        if max_drift >= self.critical_threshold:
            return 'critical'
        elif max_drift >= self.warning_threshold:
            return 'warning'
        else:
            return 'none'

    def _generate_recommendations(
        self,
        drift_score: float,
        drifted_features: List[str],
        alert_level: str
    ) -> List[str]:
        recommendations = []

        if alert_level == 'critical':
            recommendations.append("CRITICAL: Trigger model retraining immediately")
            recommendations.append("Consider disabling autotrading until model is retrained")

        elif alert_level == 'warning':
            recommendations.append("WARNING: Monitor performance closely")
            recommendations.append("Schedule model retraining within 7 days")

        if drifted_features:
            top_drifted = drifted_features[:5]
            recommendations.append(
                f"Top drifted features: {', '.join(top_drifted)}"
            )

        return recommendations
