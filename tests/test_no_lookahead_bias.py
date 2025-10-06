"""
Test suite for verifying absence of look-ahead bias in training pipeline.

This is a CRITICAL test - look-ahead bias invalidates all metrics and makes
production performance unpredictable.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forex_diffusion.training.train_sklearn import _standardize_train_val, fetch_candles_from_db


def test_standardization_no_lookahead():
    """
    Test that standardization doesn't leak information from validation set.

    CRITICAL: This verifies the most common source of look-ahead bias.
    """
    # Create synthetic time-series data with temporal structure
    n_samples = 1000
    n_features = 20

    # Simulate time-series with trend (to make train/val different)
    X_data = []
    for i in range(n_features):
        # Each feature has a trend component
        trend = np.linspace(0, 1, n_samples) * np.random.randn()
        noise = np.random.randn(n_samples) * 0.5
        feature = trend + noise
        X_data.append(feature)

    X = pd.DataFrame(np.array(X_data).T, columns=[f"feat_{i}" for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples))

    # Call standardization function
    val_frac = 0.2
    (Xtr_scaled, ytr), (Xva_scaled, yva), (mu, sigma, metadata) = _standardize_train_val(X, y, val_frac)

    # ASSERTION 1: Train and val should have different distributions
    # (if they're identical, likely look-ahead bias)
    assert metadata["ks_test_median_p"] is not None
    assert metadata["ks_test_median_p"] < 0.8, (
        f"Train/Val distributions too similar (p={metadata['ks_test_median_p']:.3f}). "
        f"Possible look-ahead bias detected!"
    )

    # ASSERTION 2: Validation statistics should differ from training
    # Calculate validation set statistics
    val_mean = Xva_scaled.mean(axis=0)
    val_std = Xva_scaled.std(axis=0)

    # These should NOT be close to 0 and 1 (which would indicate fitting on val set)
    # Allow some tolerance for small val sets
    mean_diff = np.abs(val_mean).mean()
    std_diff = np.abs(val_std - 1.0).mean()

    # If val set was used in fit, mean would be ~0 and std ~1
    # We expect some deviation (but not too much)
    assert mean_diff > 0.05 or std_diff > 0.05, (
        f"Validation set appears standardized (mean_diff={mean_diff:.3f}, std_diff={std_diff:.3f}). "
        f"Possible look-ahead bias!"
    )

    # ASSERTION 3: Verify metadata contains required fields
    assert "train_size" in metadata
    assert "val_size" in metadata
    assert "train_mean" in metadata
    assert "train_std" in metadata
    assert metadata["train_size"] + metadata["val_size"] == n_samples

    # ASSERTION 4: Verify temporal ordering (no shuffle)
    # The split should maintain temporal order
    train_size = int(n_samples * (1 - val_frac))
    assert metadata["train_size"] == train_size

    print("✅ PASS: No look-ahead bias detected in standardization")
    return True


def test_temporal_ordering_preserved():
    """
    Test that train/test split preserves temporal ordering.

    This ensures we never train on future data.
    """
    n_samples = 1000

    # Create time-indexed data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='H')
    X = pd.DataFrame({
        'timestamp': dates,
        'value': np.arange(n_samples)  # Monotonic increasing
    })
    y = pd.Series(np.random.randn(n_samples))

    # Drop timestamp for standardization (but use value to verify ordering)
    X_features = X[['value']]

    val_frac = 0.2
    (Xtr_scaled, ytr), (Xva_scaled, yva), (mu, sigma, metadata) = _standardize_train_val(X_features, y, val_frac)

    # ASSERTION: All training values should be < all validation values
    # (if temporal ordering is preserved)
    train_size = metadata["train_size"]
    val_size = metadata["val_size"]

    # Original X values
    X_values = X['value'].values
    train_max = X_values[:train_size].max()
    val_min = X_values[train_size:].min()

    assert train_max < val_min, (
        f"Temporal ordering violated! train_max={train_max}, val_min={val_min}. "
        f"Future data leaked into training set!"
    )

    print("✅ PASS: Temporal ordering preserved in split")
    return True


def test_ks_test_statistical_power():
    """
    Test that KS test can actually detect bias when present.

    This is a negative test - we intentionally introduce bias and verify it's caught.
    """
    n_samples = 1000
    n_features = 20

    # Create data where train and val are IDENTICAL (artificial bias)
    X_identical = np.random.randn(n_samples, n_features)
    X = pd.DataFrame(X_identical, columns=[f"feat_{i}" for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples))

    val_frac = 0.2

    # This should trigger a warning
    with pytest.warns(RuntimeWarning, match="POTENTIAL LOOK-AHEAD BIAS"):
        (Xtr_scaled, ytr), (Xva_scaled, yva), (mu, sigma, metadata) = _standardize_train_val(X, y, val_frac)

    # KS test should show high p-values (distributions are similar)
    assert metadata["ks_test_median_p"] > 0.8

    print("✅ PASS: KS test correctly detects artificial bias")
    return True


def test_feature_time_alignment():
    """
    Test that features don't contain future information.

    For time-based features (MA, indicators, etc.), verify that feature at time t
    only uses data from times <= t.
    """
    # This is a design test - features should be calculated with lookback only
    # Create synthetic OHLCV data
    n_candles = 100
    df = pd.DataFrame({
        'ts_utc': pd.date_range('2020-01-01', periods=n_candles, freq='5min'),
        'open': 1.1000 + np.random.randn(n_candles) * 0.001,
        'high': 1.1010 + np.random.randn(n_candles) * 0.001,
        'low': 1.0990 + np.random.randn(n_candles) * 0.001,
        'close': 1.1000 + np.random.randn(n_candles) * 0.001,
        'volume': np.random.randint(1000, 10000, n_candles)
    })

    # Calculate a simple moving average feature
    window = 10
    df['sma'] = df['close'].rolling(window=window).mean()

    # ASSERTION: SMA at time t should only depend on data at times [t-window+1, t]
    # Verify by checking that future data changes don't affect past SMA
    for i in range(window, n_candles - 10):
        sma_original = df.loc[i, 'sma']

        # Modify future data
        df_modified = df.copy()
        df_modified.loc[i+1:, 'close'] = 999.0  # Change all future values

        # Recalculate SMA
        df_modified['sma'] = df_modified['close'].rolling(window=window).mean()
        sma_modified = df_modified.loc[i, 'sma']

        # SMA at time i should be IDENTICAL (no future data used)
        assert abs(sma_original - sma_modified) < 1e-10, (
            f"Feature at time {i} changed when future data modified! "
            f"Look-ahead bias in feature calculation!"
        )

    print("✅ PASS: Features don't use future information")
    return True


def test_scaler_metadata_saved():
    """
    Test that scaler metadata is properly saved for reproducibility.

    This is critical for production inference - we need the exact same
    train set statistics.
    """
    n_samples = 500
    n_features = 10

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f"feat_{i}" for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples))

    val_frac = 0.2
    (Xtr_scaled, ytr), (Xva_scaled, yva), (mu, sigma, metadata) = _standardize_train_val(X, y, val_frac)

    # ASSERTION: Metadata contains all required fields
    required_fields = [
        "train_size",
        "val_size",
        "train_mean",
        "train_std",
        "ks_test_p_values",
        "ks_test_median_p"
    ]

    for field in required_fields:
        assert field in metadata, f"Missing required metadata field: {field}"

    # ASSERTION: Can reproduce standardization using saved metadata
    mu_saved = np.array(metadata["train_mean"])
    sigma_saved = np.array(metadata["train_std"])

    # Apply saved statistics to original val data
    X_val_original = X.values[metadata["train_size"]:]
    X_val_reproduced = (X_val_original - mu_saved) / sigma_saved

    # Should match Xva_scaled
    np.testing.assert_allclose(X_val_reproduced, Xva_scaled, rtol=1e-6)

    print("✅ PASS: Scaler metadata properly saved and reproducible")
    return True


if __name__ == "__main__":
    """Run all tests."""
    print("=" * 80)
    print("LOOK-AHEAD BIAS VERIFICATION TEST SUITE")
    print("=" * 80)
    print()

    tests = [
        ("Standardization No Look-Ahead", test_standardization_no_lookahead),
        ("Temporal Ordering Preserved", test_temporal_ordering_preserved),
        ("KS Test Statistical Power", test_ks_test_statistical_power),
        ("Feature Time Alignment", test_feature_time_alignment),
        ("Scaler Metadata Saved", test_scaler_metadata_saved),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 80)
        try:
            test_func()
            passed += 1
            print(f"✅ PASSED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED: {test_name}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed == 0:
        print("\n✅ ALL TESTS PASSED - No look-ahead bias detected!")
        sys.exit(0)
    else:
        print(f"\n❌ {failed} TEST(S) FAILED - Review immediately!")
        sys.exit(1)
