"""
Feature Selection Framework

Implements RFE, Mutual Information, and Variance Threshold selection
to prevent feature bloat and improve interpretability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Literal
from dataclasses import dataclass
from loguru import logger

from sklearn.feature_selection import (
    RFE,
    mutual_info_regression,
    VarianceThreshold,
    SelectKBest,
    f_regression
)
from sklearn.ensemble import RandomForestRegressor


SelectionMethod = Literal['rfe', 'mutual_info', 'variance', 'f_test', 'all']


@dataclass
class FeatureSelectionResult:
    """Result of feature selection."""
    selected_features: List[str]
    rejected_features: List[str]
    feature_scores: pd.Series
    selection_method: str
    n_features_selected: int


class FeatureSelector:
    """
    Feature selection with multiple strategies.

    Supports:
    - Recursive Feature Elimination (RFE)
    - Mutual Information
    - Variance Threshold
    - F-statistic
    """

    def __init__(
        self,
        method: SelectionMethod = 'rfe',
        n_features: Optional[int] = None,
        percentile: Optional[float] = None,
        variance_threshold: float = 0.01
    ):
        """
        Initialize feature selector.

        Args:
            method: Selection method
            n_features: Target number of features (if specified)
            percentile: Keep top N% of features (if specified)
            variance_threshold: Minimum variance for variance method
        """
        self.method = method
        self.n_features = n_features
        self.percentile = percentile
        self.variance_threshold = variance_threshold

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: Optional[Any] = None
    ) -> FeatureSelectionResult:
        """
        Select features using configured method.

        Args:
            X: Feature matrix
            y: Target variable
            estimator: Estimator for RFE (if None, uses RandomForest)

        Returns:
            FeatureSelectionResult
        """
        logger.info(f"Selecting features using method: {self.method}")

        if self.method == 'rfe':
            return self._select_rfe(X, y, estimator)
        elif self.method == 'mutual_info':
            return self._select_mutual_info(X, y)
        elif self.method == 'variance':
            return self._select_variance(X)
        elif self.method == 'f_test':
            return self._select_f_test(X, y)
        elif self.method == 'all':
            return self._select_ensemble(X, y, estimator)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

    def _select_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: Optional[Any] = None
    ) -> FeatureSelectionResult:
        """Recursive Feature Elimination."""
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        n_features_to_select = self._calculate_target_features(X)

        logger.info(f"Running RFE to select {n_features_to_select} features...")

        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=0.1)
        selector.fit(X, y)

        selected_mask = selector.support_
        selected_features = X.columns[selected_mask].tolist()
        rejected_features = X.columns[~selected_mask].tolist()

        # Get feature rankings
        rankings = pd.Series(selector.ranking_, index=X.columns)
        scores = 1.0 / rankings  # Inverse ranking as score

        return FeatureSelectionResult(
            selected_features=selected_features,
            rejected_features=rejected_features,
            feature_scores=scores,
            selection_method='rfe',
            n_features_selected=len(selected_features)
        )

    def _select_mutual_info(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> FeatureSelectionResult:
        """Mutual Information selection."""
        logger.info("Computing mutual information scores...")

        mi_scores = mutual_info_regression(X, y, random_state=42)
        scores = pd.Series(mi_scores, index=X.columns)

        n_features_to_select = self._calculate_target_features(X)

        # Select top-K by MI score
        selected_features = scores.nlargest(n_features_to_select).index.tolist()
        rejected_features = [f for f in X.columns if f not in selected_features]

        return FeatureSelectionResult(
            selected_features=selected_features,
            rejected_features=rejected_features,
            feature_scores=scores,
            selection_method='mutual_info',
            n_features_selected=len(selected_features)
        )

    def _select_variance(
        self,
        X: pd.DataFrame
    ) -> FeatureSelectionResult:
        """Variance Threshold selection."""
        logger.info(f"Filtering features with variance < {self.variance_threshold}...")

        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(X)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        rejected_features = X.columns[~selected_mask].tolist()

        # Compute variances
        variances = pd.Series(X.var(), index=X.columns)

        logger.info(f"Kept {len(selected_features)} features with sufficient variance")

        return FeatureSelectionResult(
            selected_features=selected_features,
            rejected_features=rejected_features,
            feature_scores=variances,
            selection_method='variance',
            n_features_selected=len(selected_features)
        )

    def _select_f_test(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> FeatureSelectionResult:
        """F-statistic selection."""
        logger.info("Computing F-test scores...")

        n_features_to_select = self._calculate_target_features(X)

        selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
        selector.fit(X, y)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        rejected_features = X.columns[~selected_mask].tolist()

        scores = pd.Series(selector.scores_, index=X.columns)

        return FeatureSelectionResult(
            selected_features=selected_features,
            rejected_features=rejected_features,
            feature_scores=scores,
            selection_method='f_test',
            n_features_selected=len(selected_features)
        )

    def _select_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: Optional[Any] = None
    ) -> FeatureSelectionResult:
        """
        Ensemble selection: Use multiple methods and take intersection.
        """
        logger.info("Running ensemble feature selection (RFE + MI + Variance)...")

        # Run all methods
        rfe_result = self._select_rfe(X, y, estimator)
        mi_result = self._select_mutual_info(X, y)
        var_result = self._select_variance(X)

        # Take intersection of features selected by all methods
        selected_set = (
            set(rfe_result.selected_features) &
            set(mi_result.selected_features) &
            set(var_result.selected_features)
        )

        selected_features = list(selected_set)
        rejected_features = [f for f in X.columns if f not in selected_features]

        # Combine scores (normalized)
        rfe_norm = (rfe_result.feature_scores - rfe_result.feature_scores.min()) / \
                   (rfe_result.feature_scores.max() - rfe_result.feature_scores.min() + 1e-10)
        mi_norm = (mi_result.feature_scores - mi_result.feature_scores.min()) / \
                  (mi_result.feature_scores.max() - mi_result.feature_scores.min() + 1e-10)

        combined_scores = (rfe_norm + mi_norm) / 2

        logger.info(f"Ensemble selected {len(selected_features)} features (intersection)")

        return FeatureSelectionResult(
            selected_features=selected_features,
            rejected_features=rejected_features,
            feature_scores=combined_scores,
            selection_method='ensemble',
            n_features_selected=len(selected_features)
        )

    def _calculate_target_features(self, X: pd.DataFrame) -> int:
        """Calculate target number of features to select."""
        total_features = X.shape[1]

        if self.n_features is not None:
            return min(self.n_features, total_features)

        if self.percentile is not None:
            return max(1, int(total_features * self.percentile))

        # Default: use heuristic (sqrt of n_samples or 50% of features, whichever is smaller)
        n_samples = X.shape[0]
        heuristic_max = int(np.sqrt(n_samples))
        default_pct = int(total_features * 0.5)

        return min(heuristic_max, default_pct, total_features)

    def generate_report(
        self,
        result: FeatureSelectionResult,
        X: pd.DataFrame
    ) -> Dict:
        """
        Generate feature selection report.

        Args:
            result: FeatureSelectionResult
            X: Original feature matrix

        Returns:
            Dict with report data
        """
        # Compute correlation matrix of selected features
        selected_X = X[result.selected_features]
        corr_matrix = selected_X.corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(result.selected_features)):
            for j in range(i + 1, len(result.selected_features)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((
                        result.selected_features[i],
                        result.selected_features[j],
                        corr_val
                    ))

        report = {
            'method': result.selection_method,
            'total_features': len(X.columns),
            'selected_count': result.n_features_selected,
            'rejected_count': len(result.rejected_features),
            'reduction_pct': (1 - result.n_features_selected / len(X.columns)) * 100,
            'selected_features': result.selected_features,
            'rejected_features': result.rejected_features,
            'top_10_features': result.feature_scores.nlargest(10).to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'warnings': []
        }

        # Add warnings
        if result.n_features_selected < 10:
            report['warnings'].append("Very few features selected - may underfit")

        if len(high_corr_pairs) > 5:
            report['warnings'].append(
                f"{len(high_corr_pairs)} highly correlated feature pairs - consider further reduction"
            )

        return report
