"""
Time-Series Cross-Validation Framework

Implements walk-forward validation with proper temporal ordering
to prevent data leakage and evaluate model performance realistically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


CVStrategy = Literal['expanding', 'sliding', 'gap']


@dataclass
class CVSplit:
    """Single cross-validation split."""
    fold_index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    train_dates: Tuple[datetime, datetime]
    val_dates: Tuple[datetime, datetime]


@dataclass
class CVResults:
    """Cross-validation results."""
    splits: List[CVSplit]
    fold_metrics: List[dict]
    aggregate_metrics: dict
    anomalous_folds: List[int]
    forecast_stability: float


class TimeSeriesCV:
    """
    Time-Series Cross-Validation Splitter.

    Maintains chronological order: training data always precedes validation data.
    Supports multiple splitting strategies to evaluate model robustness.

    Features:
    - Expanding window: Training set grows over time
    - Sliding window: Training set size stays fixed
    - Gap-based: Skip data between train/val to prevent leakage

    Example:
        >>> cv = TimeSeriesCV(
        ...     strategy='expanding',
        ...     n_splits=5,
        ...     train_months=6,
        ...     val_months=1
        ... )
        >>> for fold_idx, (train_idx, val_idx) in enumerate(cv.split(data)):
        ...     X_train, y_train = X[train_idx], y[train_idx]
        ...     X_val, y_val = X[val_idx], y[val_idx]
        ...     # Train and evaluate
    """

    def __init__(
        self,
        strategy: CVStrategy = 'expanding',
        n_splits: int = 5,
        train_months: int = 6,
        val_months: int = 1,
        gap_days: int = 0,
        min_train_samples: int = 1000
    ):
        """
        Initialize time-series cross-validator.

        Args:
            strategy: Splitting strategy ('expanding', 'sliding', 'gap')
            n_splits: Number of cross-validation folds
            train_months: Months of data for training window
            val_months: Months of data for validation window
            gap_days: Days to skip between train/val (prevents leakage)
            min_train_samples: Minimum samples required for training
        """
        self.strategy = strategy
        self.n_splits = n_splits
        self.train_months = train_months
        self.val_months = val_months
        self.gap_days = gap_days
        self.min_train_samples = min_train_samples

        logger.info(f"TimeSeriesCV initialized: strategy={strategy}, "
                   f"n_splits={n_splits}, train={train_months}m, val={val_months}m")

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation index splits.

        Args:
            X: Feature dataframe with DatetimeIndex
            y: Target series (optional)
            groups: Not used (for sklearn compatibility)

        Yields:
            (train_indices, val_indices) for each fold
        """
        n_samples = len(X)

        # Convert months to approximate sample counts (assuming 5min bars)
        # 1 month ≈ 30 days × 24 hours × 12 (5min bars/hour) = 8640 samples
        samples_per_month = 8640
        train_size = self.train_months * samples_per_month
        val_size = self.val_months * samples_per_month
        gap_size = self.gap_days * 288  # 288 5-min bars per day

        # Validate we have enough data
        min_required = train_size + val_size + gap_size
        if n_samples < min_required:
            raise ValueError(
                f"Insufficient data: {n_samples} samples, "
                f"need at least {min_required} for 1 split"
            )

        # Generate splits based on strategy
        if self.strategy == 'expanding':
            yield from self._expanding_window_split(
                n_samples, train_size, val_size, gap_size
            )
        elif self.strategy == 'sliding':
            yield from self._sliding_window_split(
                n_samples, train_size, val_size, gap_size
            )
        elif self.strategy == 'gap':
            yield from self._gap_based_split(
                n_samples, train_size, val_size, gap_size
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _expanding_window_split(
        self,
        n_samples: int,
        train_size: int,
        val_size: int,
        gap_size: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Expanding window: Training set grows with each fold.

        Fold 1: [----train----][gap][val]
        Fold 2: [-------train-------][gap][val]
        Fold 3: [----------train----------][gap][val]
        """
        # Calculate step size to fit n_splits
        available = n_samples - train_size - gap_size - val_size
        step = available // (self.n_splits - 1) if self.n_splits > 1 else 0

        for i in range(self.n_splits):
            train_end = min(train_size + i * step, n_samples - gap_size - val_size)
            train_start = 0
            val_start = train_end + gap_size
            val_end = val_start + val_size

            # Ensure we don't exceed bounds
            if val_end > n_samples:
                break

            # Verify minimum training size
            if (train_end - train_start) < self.min_train_samples:
                continue

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)

            yield train_idx, val_idx

    def _sliding_window_split(
        self,
        n_samples: int,
        train_size: int,
        val_size: int,
        gap_size: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Sliding window: Training set size stays fixed.

        Fold 1: [----train----][gap][val]
        Fold 2:      [----train----][gap][val]
        Fold 3:           [----train----][gap][val]
        """
        window_size = train_size + gap_size + val_size
        available = n_samples - window_size
        step = available // (self.n_splits - 1) if self.n_splits > 1 else 0

        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + train_size
            val_start = train_end + gap_size
            val_end = val_start + val_size

            # Ensure we don't exceed bounds
            if val_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)

            yield train_idx, val_idx

    def _gap_based_split(
        self,
        n_samples: int,
        train_size: int,
        val_size: int,
        gap_size: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Gap-based: Like expanding but with mandatory gap.
        Useful for strategies with holding periods.

        Fold 1: [----train----][  gap  ][val]
        Fold 2: [-------train-------][  gap  ][val]
        """
        # Similar to expanding but enforces gap
        if gap_size == 0:
            logger.warning("Gap-based split with gap_size=0, using 1 day default")
            gap_size = 288  # 1 day

        yield from self._expanding_window_split(
            n_samples, train_size, val_size, gap_size
        )

    def get_split_info(
        self,
        X: pd.DataFrame
    ) -> List[CVSplit]:
        """
        Get detailed information about each split.

        Args:
            X: Feature dataframe with DatetimeIndex

        Returns:
            List of CVSplit objects with metadata
        """
        splits = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.split(X)):
            # Extract date ranges
            train_dates = (X.index[train_idx[0]], X.index[train_idx[-1]])
            val_dates = (X.index[val_idx[0]], X.index[val_idx[-1]])

            split = CVSplit(
                fold_index=fold_idx,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                val_start=val_idx[0],
                val_end=val_idx[-1],
                train_dates=train_dates,
                val_dates=val_dates
            )
            splits.append(split)

        return splits

    def validate_splits(
        self,
        X: pd.DataFrame
    ) -> bool:
        """
        Validate that splits maintain temporal ordering.

        Args:
            X: Feature dataframe with DatetimeIndex

        Returns:
            True if all validations pass

        Raises:
            ValueError if validation fails
        """
        for fold_idx, (train_idx, val_idx) in enumerate(self.split(X)):
            # Check no overlap
            if set(train_idx) & set(val_idx):
                raise ValueError(f"Fold {fold_idx}: Train and val indices overlap!")

            # Check temporal ordering
            train_max_date = X.index[train_idx].max()
            val_min_date = X.index[val_idx].min()

            if train_max_date >= val_min_date:
                raise ValueError(
                    f"Fold {fold_idx}: Temporal ordering violated! "
                    f"Train max: {train_max_date}, Val min: {val_min_date}"
                )

            logger.debug(f"Fold {fold_idx} validated: "
                        f"train={len(train_idx)}, val={len(val_idx)}")

        logger.info("✓ All splits validated successfully")
        return True


class CVEvaluator:
    """
    Cross-validation evaluator with performance aggregation.

    Computes per-fold metrics and aggregates with anomaly detection.
    """

    def __init__(
        self,
        cv: TimeSeriesCV,
        anomaly_threshold: float = 2.0
    ):
        """
        Initialize CV evaluator.

        Args:
            cv: TimeSeriesCV instance
            anomaly_threshold: Std deviations for anomaly detection
        """
        self.cv = cv
        self.anomaly_threshold = anomaly_threshold

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory,
        metrics: List[callable]
    ) -> CVResults:
        """
        Run cross-validation evaluation.

        Args:
            X: Features
            y: Target
            model_factory: Callable that returns untrained model
            metrics: List of metric functions (y_true, y_pred) -> float

        Returns:
            CVResults with comprehensive evaluation
        """
        fold_metrics = []
        predictions_by_fold = {}
        splits = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X)):
            logger.info(f"Evaluating fold {fold_idx + 1}/{self.cv.n_splits}")

            # Split data
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Train model
            model = model_factory()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Compute metrics
            fold_metric = {'fold': fold_idx}
            for metric_fn in metrics:
                metric_name = metric_fn.__name__
                value = metric_fn(y_val, y_pred)
                fold_metric[metric_name] = value

            fold_metrics.append(fold_metric)
            predictions_by_fold[fold_idx] = y_pred

            # Store split info
            train_dates = (X.index[train_idx[0]], X.index[train_idx[-1]])
            val_dates = (X.index[val_idx[0]], X.index[val_idx[-1]])
            splits.append(CVSplit(
                fold_index=fold_idx,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                val_start=val_idx[0],
                val_end=val_idx[-1],
                train_dates=train_dates,
                val_dates=val_dates
            ))

        # Aggregate metrics
        aggregate = self._aggregate_metrics(fold_metrics)

        # Detect anomalous folds
        anomalous = self._detect_anomalous_folds(fold_metrics)

        # Compute forecast stability
        stability = self._compute_forecast_stability(predictions_by_fold, X, y)

        return CVResults(
            splits=splits,
            fold_metrics=fold_metrics,
            aggregate_metrics=aggregate,
            anomalous_folds=anomalous,
            forecast_stability=stability
        )

    def _aggregate_metrics(
        self,
        fold_metrics: List[dict]
    ) -> dict:
        """Aggregate metrics across folds."""
        if not fold_metrics:
            return {}

        # Get metric names (exclude 'fold')
        metric_names = [k for k in fold_metrics[0].keys() if k != 'fold']

        aggregate = {}
        for metric_name in metric_names:
            values = [fm[metric_name] for fm in fold_metrics]
            aggregate[f'{metric_name}_mean'] = np.mean(values)
            aggregate[f'{metric_name}_std'] = np.std(values)
            aggregate[f'{metric_name}_min'] = np.min(values)
            aggregate[f'{metric_name}_max'] = np.max(values)

        return aggregate

    def _detect_anomalous_folds(
        self,
        fold_metrics: List[dict]
    ) -> List[int]:
        """Detect folds with anomalous performance."""
        if not fold_metrics or len(fold_metrics) < 3:
            return []

        # Use first metric for anomaly detection
        metric_names = [k for k in fold_metrics[0].keys() if k != 'fold']
        if not metric_names:
            return []

        primary_metric = metric_names[0]
        values = np.array([fm[primary_metric] for fm in fold_metrics])

        mean = np.mean(values)
        std = np.std(values)

        anomalous = []
        for fold in fold_metrics:
            fold_idx = fold['fold']
            value = fold[primary_metric]

            if abs(value - mean) > self.anomaly_threshold * std:
                anomalous.append(fold_idx)
                logger.warning(
                    f"Fold {fold_idx} anomalous: "
                    f"{primary_metric}={value:.4f} "
                    f"(mean={mean:.4f}, std={std:.4f})"
                )

        return anomalous

    def _compute_forecast_stability(
        self,
        predictions_by_fold: dict,
        X: pd.DataFrame,
        y: pd.Series
    ) -> float:
        """
        Compute forecast stability across folds.

        Measures correlation of predictions for overlapping validation periods.
        """
        # Simple implementation: return 1.0 if single fold
        if len(predictions_by_fold) < 2:
            return 1.0

        # For multiple folds, compute average pairwise correlation
        # (Simplified - full implementation would find overlapping periods)
        correlations = []
        folds = list(predictions_by_fold.keys())

        for i in range(len(folds) - 1):
            # Compare consecutive folds
            # (In full implementation, would align by timestamp)
            fold_a = predictions_by_fold[folds[i]]
            fold_b = predictions_by_fold[folds[i + 1]]

            # Use minimum length for comparison
            min_len = min(len(fold_a), len(fold_b))
            if min_len > 10:  # Require minimum overlap
                corr = np.corrcoef(fold_a[:min_len], fold_b[:min_len])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if correlations:
            return float(np.mean(correlations))
        else:
            return 1.0
