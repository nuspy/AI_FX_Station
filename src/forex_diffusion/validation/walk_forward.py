"""
Walk-Forward Validation for Time Series Models

Implements proper time-series cross-validation with:
- Temporal ordering preservation (NO shuffle)
- Purge periods to prevent look-ahead bias
- Embargo periods to prevent overlap
- Multiple train/test splits

Reference: "Advances in Financial Machine Learning" by Marcos López de Prado
"""
from __future__ import annotations

from typing import Iterator, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class WalkForwardSplit:
    """Single walk-forward train/test split with metadata"""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purge_start: Optional[int] = None  # Start of purge period
    purge_end: Optional[int] = None    # End of purge period
    embargo_start: Optional[int] = None  # Start of embargo period
    embargo_end: Optional[int] = None    # End of embargo period

    @property
    def train_indices(self) -> np.ndarray:
        """Get training indices"""
        return np.arange(self.train_start, self.train_end)

    @property
    def test_indices(self) -> np.ndarray:
        """Get test indices (excluding purge/embargo)"""
        indices = np.arange(self.test_start, self.test_end)

        # Remove purge period if exists
        if self.purge_start is not None and self.purge_end is not None:
            purge_mask = (indices < self.purge_start) | (indices >= self.purge_end)
            indices = indices[purge_mask]

        # Remove embargo period if exists
        if self.embargo_start is not None and self.embargo_end is not None:
            embargo_mask = (indices < self.embargo_start) | (indices >= self.embargo_end)
            indices = embargo_mask[embargo_mask]

        return indices

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "train_start": int(self.train_start),
            "train_end": int(self.train_end),
            "test_start": int(self.test_start),
            "test_end": int(self.test_end),
            "purge_start": int(self.purge_start) if self.purge_start is not None else None,
            "purge_end": int(self.purge_end) if self.purge_end is not None else None,
            "embargo_start": int(self.embargo_start) if self.embargo_start is not None else None,
            "embargo_end": int(self.embargo_end) if self.embargo_end is not None else None,
            "train_size": int(self.train_end - self.train_start),
            "test_size": int(len(self.test_indices)),
        }


class WalkForwardValidator:
    """
    Walk-Forward Validation for time series.

    Key Features:
    1. **Anchored Mode**: Training window starts from beginning (expanding window)
    2. **Rolling Mode**: Training window has fixed size (sliding window)
    3. **Purge Period**: Remove samples between train and test to prevent label leakage
    4. **Embargo Period**: Remove recent test samples to account for model delay

    Example:
        ```python
        validator = WalkForwardValidator(
            n_splits=5,
            test_size=0.2,
            anchored=True,
            purge_pct=0.02,  # 2% gap between train and test
            embargo_pct=0.01  # 1% embargo at end of test
        )

        for split in validator.split(X, y):
            X_train = X[split.train_indices]
            X_test = X[split.test_indices]
            # ... train and evaluate
        ```
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        anchored: bool = True,
        purge_pct: float = 0.0,
        embargo_pct: float = 0.0,
        min_train_size: Optional[int] = None,
    ):
        """
        Initialize Walk-Forward Validator.

        Args:
            n_splits: Number of train/test splits
            test_size: Fraction of data for test in each split (0.0-1.0)
            anchored: If True, use expanding window (train from start).
                     If False, use rolling window (fixed train size)
            purge_pct: Percentage of data to purge between train and test (0.0-1.0)
            embargo_pct: Percentage of test data to embargo at end (0.0-1.0)
            min_train_size: Minimum number of samples in training set
        """
        if not 1 <= n_splits <= 20:
            raise ValueError(f"n_splits must be between 1 and 20, got {n_splits}")
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        if not 0.0 <= purge_pct < 1.0:
            raise ValueError(f"purge_pct must be between 0 and 1, got {purge_pct}")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError(f"embargo_pct must be between 0 and 1, got {embargo_pct}")

        self.n_splits = n_splits
        self.test_size = test_size
        self.anchored = anchored
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.min_train_size = min_train_size

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Optional[np.ndarray | pd.Series] = None
    ) -> Iterator[WalkForwardSplit]:
        """
        Generate train/test splits with temporal ordering.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (optional, for validation)

        Yields:
            WalkForwardSplit objects containing train/test indices
        """
        n_samples = len(X)

        if self.min_train_size is not None and self.min_train_size >= n_samples:
            raise ValueError(
                f"min_train_size ({self.min_train_size}) >= n_samples ({n_samples})"
            )

        # Calculate test size for each split
        test_samples = int(n_samples * self.test_size)
        if test_samples < 1:
            raise ValueError("test_size too small, results in 0 test samples")

        # Calculate purge/embargo sizes
        purge_samples = int(n_samples * self.purge_pct)
        embargo_samples = int(test_samples * self.embargo_pct)

        logger.info(
            f"[WalkForward] n_samples={n_samples}, n_splits={self.n_splits}, "
            f"test_size={test_samples}, purge={purge_samples}, embargo={embargo_samples}, "
            f"mode={'anchored' if self.anchored else 'rolling'}"
        )

        # Generate splits
        for i in range(self.n_splits):
            # Calculate test window for this split
            # Each split tests on a different segment moving forward in time
            test_start = int(n_samples * (1 - self.test_size) * (i + 1) / self.n_splits)
            test_end = min(test_start + test_samples, n_samples)

            if test_end <= test_start:
                logger.warning(f"[WalkForward] Split {i+1}: test window empty, skipping")
                continue

            # Calculate train window
            if self.anchored:
                # Anchored: train from beginning to test_start (expanding window)
                train_start = 0
                train_end = test_start - purge_samples
            else:
                # Rolling: train window has same size as test window
                train_start = max(0, test_start - test_samples - purge_samples)
                train_end = test_start - purge_samples

            # Apply minimum training size constraint
            if self.min_train_size is not None:
                if train_end - train_start < self.min_train_size:
                    # Not enough training data yet, skip this split
                    logger.warning(
                        f"[WalkForward] Split {i+1}: train_size={train_end - train_start} "
                        f"< min_train_size={self.min_train_size}, skipping"
                    )
                    continue

            # Ensure valid ranges
            if train_end <= train_start:
                logger.warning(f"[WalkForward] Split {i+1}: train window empty, skipping")
                continue

            # Define purge period (gap between train and test)
            purge_start = None
            purge_end = None
            if purge_samples > 0:
                purge_start = train_end
                purge_end = test_start

            # Define embargo period (end of test set)
            embargo_start = None
            embargo_end = None
            if embargo_samples > 0:
                embargo_start = test_end - embargo_samples
                embargo_end = test_end

            split = WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_start=purge_start,
                purge_end=purge_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
            )

            logger.debug(
                f"[WalkForward] Split {i+1}/{self.n_splits}: "
                f"train=[{train_start}:{train_end}] ({train_end - train_start}), "
                f"test=[{test_start}:{test_end}] ({len(split.test_indices)})"
            )

            yield split

    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> int:
        """Get number of splits (sklearn compatibility)"""
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV)

    Advanced CV method from López de Prado that:
    1. Generates multiple train/test splits
    2. Purges overlapping samples to prevent leakage
    3. Computes all possible combinations of test paths
    4. Provides unbiased performance estimates

    More sophisticated than standard walk-forward, but computationally expensive.
    Use this for final model validation, not hyperparameter tuning.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        purge_pct: float = 0.02,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize Combinatorial Purged CV.

        Args:
            n_splits: Number of splits for generating test groups
            n_test_groups: Number of groups to combine (typically 2-3)
            purge_pct: Percentage to purge between groups
            embargo_pct: Percentage to embargo at end
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Optional[np.ndarray | pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorial purged train/test splits.

        Yields:
            Tuples of (train_indices, test_indices)
        """
        from itertools import combinations

        n_samples = len(X)

        # Create groups
        group_size = n_samples // self.n_splits
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = min((i + 1) * group_size, n_samples)
            groups.append(np.arange(start, end))

        # Generate all combinations of test groups
        test_combinations = list(combinations(range(self.n_splits), self.n_test_groups))

        logger.info(
            f"[CPCV] Generating {len(test_combinations)} combinations "
            f"({self.n_splits} groups, {self.n_test_groups} test groups)"
        )

        for combo_idx, test_group_indices in enumerate(test_combinations):
            # Combine test groups
            test_indices = np.concatenate([groups[i] for i in test_group_indices])

            # Train on all other groups (with purging)
            train_groups = [i for i in range(self.n_splits) if i not in test_group_indices]
            train_indices = np.concatenate([groups[i] for i in train_groups])

            # Apply purging: remove samples near test groups
            purge_samples = int(n_samples * self.purge_pct)
            if purge_samples > 0:
                for test_group_idx in test_group_indices:
                    # Purge before test group
                    purge_start = max(0, groups[test_group_idx][0] - purge_samples)
                    purge_end = groups[test_group_idx][0]
                    train_indices = train_indices[
                        (train_indices < purge_start) | (train_indices >= purge_end)
                    ]

                    # Purge after test group
                    purge_start = groups[test_group_idx][-1] + 1
                    purge_end = min(n_samples, purge_start + purge_samples)
                    train_indices = train_indices[
                        (train_indices < purge_start) | (train_indices >= purge_end)
                    ]

            # Apply embargo to test set
            embargo_samples = int(len(test_indices) * self.embargo_pct)
            if embargo_samples > 0:
                test_indices = test_indices[:-embargo_samples]

            logger.debug(
                f"[CPCV] Combo {combo_idx+1}/{len(test_combinations)}: "
                f"train={len(train_indices)}, test={len(test_indices)}"
            )

            yield train_indices, test_indices

    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> int:
        """Get number of splits"""
        from math import comb
        return comb(self.n_splits, self.n_test_groups)


# Utility functions
def purge_embargo_split(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    purge_samples: int = 0,
    embargo_samples: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply purge and embargo to train/test split.

    Args:
        train_indices: Training sample indices
        test_indices: Test sample indices
        purge_samples: Number of samples to remove between train and test
        embargo_samples: Number of samples to remove from end of test

    Returns:
        Tuple of (purged_train_indices, embargoed_test_indices)
    """
    # Purge: remove training samples near test set
    if purge_samples > 0 and len(test_indices) > 0:
        test_start = test_indices[0]
        purge_start = max(0, test_start - purge_samples)
        train_indices = train_indices[train_indices < purge_start]

    # Embargo: remove recent test samples
    if embargo_samples > 0 and len(test_indices) > embargo_samples:
        test_indices = test_indices[:-embargo_samples]

    return train_indices, test_indices
