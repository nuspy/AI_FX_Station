"""
Example: Walk-Forward Validation

Demonstrates proper time-series cross-validation with:
- Multiple train/test splits maintaining temporal order
- Purge periods to prevent look-ahead bias
- Embargo periods to account for model delay
- Comparison between anchored vs rolling windows

Usage:
    python examples/walk_forward_validation_example.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger

from forex_diffusion.validation import WalkForwardValidator, CombinatorialPurgedCV


def demonstrate_walk_forward_anchored():
    """Demonstrate anchored (expanding window) walk-forward validation"""

    logger.info("=" * 80)
    logger.info("ANCHORED WALK-FORWARD VALIDATION (Expanding Window)")
    logger.info("=" * 80)

    # Generate sample data (1000 samples)
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)  # 10 features
    y = np.random.randn(n_samples)

    # Create validator
    validator = WalkForwardValidator(
        n_splits=5,
        test_size=0.2,
        anchored=True,  # Expanding window
        purge_pct=0.02,  # 2% gap between train and test
        embargo_pct=0.01,  # 1% embargo at end of test
    )

    logger.info(f"Dataset: {n_samples} samples, {X.shape[1]} features")
    logger.info(f"Configuration: {validator.n_splits} splits, test_size={validator.test_size}")
    logger.info(f"Purge: {validator.purge_pct*100:.1f}%, Embargo: {validator.embargo_pct*100:.1f}%")
    logger.info("")

    # Perform walk-forward validation
    scores = []
    for i, split in enumerate(validator.split(X, y), 1):
        X_train = X[split.train_indices]
        y_train = y[split.train_indices]
        X_test = X[split.test_indices]
        y_test = y[split.test_indices]

        # Simple model: mean prediction
        y_pred = np.full_like(y_test, y_train.mean())
        mae = np.abs(y_test - y_pred).mean()
        scores.append(mae)

        logger.info(f"Split {i}/{validator.n_splits}:")
        logger.info(f"  Train: [{split.train_start}:{split.train_end}] ({len(X_train)} samples)")
        logger.info(f"  Test:  [{split.test_start}:{split.test_end}] ({len(X_test)} samples)")
        if split.purge_start is not None:
            logger.info(f"  Purge: [{split.purge_start}:{split.purge_end}] "
                       f"({split.purge_end - split.purge_start} samples removed)")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info("")

    logger.info(f"Average MAE across {len(scores)} splits: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    logger.info("")


def demonstrate_walk_forward_rolling():
    """Demonstrate rolling (sliding window) walk-forward validation"""

    logger.info("=" * 80)
    logger.info("ROLLING WALK-FORWARD VALIDATION (Sliding Window)")
    logger.info("=" * 80)

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples)

    # Create validator
    validator = WalkForwardValidator(
        n_splits=5,
        test_size=0.2,
        anchored=False,  # Rolling window (fixed size)
        purge_pct=0.02,
        embargo_pct=0.01,
    )

    logger.info(f"Dataset: {n_samples} samples, {X.shape[1]} features")
    logger.info(f"Configuration: {validator.n_splits} splits, test_size={validator.test_size}")
    logger.info(f"Purge: {validator.purge_pct*100:.1f}%, Embargo: {validator.embargo_pct*100:.1f}%")
    logger.info("")

    # Perform walk-forward validation
    scores = []
    for i, split in enumerate(validator.split(X, y), 1):
        X_train = X[split.train_indices]
        y_train = y[split.train_indices]
        X_test = X[split.test_indices]
        y_test = y[split.test_indices]

        # Simple model
        y_pred = np.full_like(y_test, y_train.mean())
        mae = np.abs(y_test - y_pred).mean()
        scores.append(mae)

        logger.info(f"Split {i}/{validator.n_splits}:")
        logger.info(f"  Train: [{split.train_start}:{split.train_end}] ({len(X_train)} samples)")
        logger.info(f"  Test:  [{split.test_start}:{split.test_end}] ({len(X_test)} samples)")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info("")

    logger.info(f"Average MAE across {len(scores)} splits: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    logger.info("")


def demonstrate_cpcv():
    """Demonstrate Combinatorial Purged Cross-Validation"""

    logger.info("=" * 80)
    logger.info("COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)")
    logger.info("=" * 80)

    # Generate sample data (smaller for CPCV due to computational cost)
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples)

    # Create CPCV validator
    validator = CombinatorialPurgedCV(
        n_splits=5,
        n_test_groups=2,  # Combine 2 groups at a time
        purge_pct=0.02,
        embargo_pct=0.01,
    )

    n_combos = validator.get_n_splits()
    logger.info(f"Dataset: {n_samples} samples, {X.shape[1]} features")
    logger.info(f"Configuration: {validator.n_splits} splits, {validator.n_test_groups} test groups")
    logger.info(f"Total combinations: {n_combos}")
    logger.info("")

    # Perform CPCV
    scores = []
    for i, (train_idx, test_idx) in enumerate(validator.split(X, y), 1):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Simple model
        y_pred = np.full_like(y_test, y_train.mean())
        mae = np.abs(y_test - y_pred).mean()
        scores.append(mae)

        logger.info(f"Combination {i}/{n_combos}:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Test:  {len(X_test)} samples")
        logger.info(f"  MAE: {mae:.4f}")

    logger.info("")
    logger.info(f"Average MAE across {len(scores)} combinations: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    logger.info("")


def demonstrate_bias_comparison():
    """Compare biased (no purge) vs unbiased (with purge) validation"""

    logger.info("=" * 80)
    logger.info("BIAS COMPARISON: No Purge vs With Purge")
    logger.info("=" * 80)

    # Generate autocorrelated data (simulating financial time series)
    np.random.seed(42)
    n_samples = 1000

    # Create autocorrelated features
    X = np.zeros((n_samples, 5))
    for i in range(1, n_samples):
        X[i] = 0.9 * X[i-1] + 0.1 * np.random.randn(5)  # AR(1) process

    # Target depends on recent features
    y = X[:, 0] + 0.3 * np.roll(X[:, 1], -5) + 0.1 * np.random.randn(n_samples)

    # Validator WITHOUT purge (biased)
    validator_biased = WalkForwardValidator(
        n_splits=5,
        test_size=0.2,
        anchored=True,
        purge_pct=0.0,  # NO purge
        embargo_pct=0.0,  # NO embargo
    )

    # Validator WITH purge (unbiased)
    validator_unbiased = WalkForwardValidator(
        n_splits=5,
        test_size=0.2,
        anchored=True,
        purge_pct=0.02,  # 2% purge
        embargo_pct=0.01,  # 1% embargo
    )

    logger.info(f"Dataset: {n_samples} samples (autocorrelated)")
    logger.info("")

    # Run biased validation
    logger.info("WITHOUT Purge/Embargo (BIASED):")
    scores_biased = []
    for split in validator_biased.split(X, y):
        X_train = X[split.train_indices]
        y_train = y[split.train_indices]
        X_test = X[split.test_indices]
        y_test = y[split.test_indices]

        y_pred = np.full_like(y_test, y_train.mean())
        mae = np.abs(y_test - y_pred).mean()
        scores_biased.append(mae)

    logger.info(f"  Average MAE: {np.mean(scores_biased):.4f} ± {np.std(scores_biased):.4f}")
    logger.info("")

    # Run unbiased validation
    logger.info("WITH Purge/Embargo (UNBIASED):")
    scores_unbiased = []
    for split in validator_unbiased.split(X, y):
        X_train = X[split.train_indices]
        y_train = y[split.train_indices]
        X_test = X[split.test_indices]
        y_test = y[split.test_indices]

        y_pred = np.full_like(y_test, y_train.mean())
        mae = np.abs(y_test - y_pred).mean()
        scores_unbiased.append(mae)

    logger.info(f"  Average MAE: {np.mean(scores_unbiased):.4f} ± {np.std(scores_unbiased):.4f}")
    logger.info("")

    # Compare
    bias_difference = np.mean(scores_biased) - np.mean(scores_unbiased)
    logger.info(f"Bias Effect: {bias_difference:.4f}")
    logger.info(f"  → Biased validation appears {abs(bias_difference)/np.mean(scores_unbiased)*100:.1f}% "
               f"{'better' if bias_difference < 0 else 'worse'} than reality")
    logger.info("")


def main():
    """Run all demonstrations"""

    logger.info("Walk-Forward Validation Examples")
    logger.info("=" * 80)
    logger.info("")

    try:
        # 1. Anchored (expanding window)
        demonstrate_walk_forward_anchored()

        # 2. Rolling (sliding window)
        demonstrate_walk_forward_rolling()

        # 3. Combinatorial Purged CV
        demonstrate_cpcv()

        # 4. Bias comparison
        demonstrate_bias_comparison()

        logger.info("=" * 80)
        logger.info("✅ All demonstrations completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"❌ Demonstration failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
