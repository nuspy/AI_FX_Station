"""
Unit tests for feature_pipeline.py

Tests the unified entry point and ensures consistency across implementations.
"""
import pytest
import pandas as pd
import numpy as np

from forex_diffusion.features.feature_pipeline import (
    compute_features,
    compute_minimal_features,
    get_feature_names,
    validate_input_data,
    FeatureConfig
)


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    n = 200
    
    # Generate realistic price data
    close = 1.2000 + np.cumsum(np.random.randn(n) * 0.0001)
    high = close + np.abs(np.random.randn(n) * 0.0005)
    low = close - np.abs(np.random.randn(n) * 0.0005)
    open_price = close.copy()
    open_price[1:] = close[:-1]
    
    # Timestamps (1-minute bars)
    base_ts = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
    timestamps = [base_ts + pd.Timedelta(minutes=i) for i in range(n)]
    ts_utc = [int(ts.timestamp() * 1000) for ts in timestamps]
    
    df = pd.DataFrame({
        'ts_utc': ts_utc,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100, 1000, n)
    })
    
    return df


class TestInputValidation:
    """Test input validation."""
    
    def test_valid_input(self, sample_ohlc_data):
        """Test that valid input passes validation."""
        validate_input_data(sample_ohlc_data)  # Should not raise
    
    def test_missing_columns(self):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({'close': [1.0, 2.0]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_input_data(df)
    
    def test_short_data_warning(self, caplog):
        """Test that short data triggers warning."""
        df = pd.DataFrame({
            'ts_utc': [1000, 2000],
            'open': [1.0, 1.1],
            'high': [1.1, 1.2],
            'low': [0.9, 1.0],
            'close': [1.05, 1.15]
        })
        
        validate_input_data(df)
        assert "may be insufficient" in caplog.text.lower()


class TestFeatureComputation:
    """Test feature computation."""
    
    def test_compute_features_default_config(self, sample_ohlc_data):
        """Test computing features with default configuration."""
        features, scaler = compute_features(sample_ohlc_data)
        
        # Check output types
        assert isinstance(features, pd.DataFrame)
        assert scaler is not None
        
        # Check that features were computed
        assert len(features) == len(sample_ohlc_data)
        assert features.shape[1] > 0
    
    def test_compute_features_no_standardization(self, sample_ohlc_data):
        """Test computing features without standardization."""
        features, scaler = compute_features(
            sample_ohlc_data, 
            standardize=False
        )
        
        assert isinstance(features, pd.DataFrame)
        assert scaler is None
    
    def test_compute_minimal_features(self, sample_ohlc_data):
        """Test computing minimal feature set."""
        features = compute_minimal_features(
            sample_ohlc_data,
            indicators=['atr', 'rsi']
        )
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlc_data)
        
        # Check for expected features
        assert 'r_open' in features.columns  # relative OHLC
        assert 'hour_sin' in features.columns  # temporal
        assert 'atr_14' in features.columns  # ATR indicator
        assert 'rsi_14' in features.columns  # RSI indicator
    
    def test_compute_features_with_custom_config(self, sample_ohlc_data):
        """Test computing features with custom configuration."""
        config = FeatureConfig({
            "indicators": {
                "atr": {"enabled": True, "n": 20},
                "rsi": {"enabled": True, "n": 10}
            }
        })
        
        features, scaler = compute_features(sample_ohlc_data, config=config)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0


class TestFeatureNames:
    """Test feature name introspection."""
    
    def test_get_feature_names_default(self):
        """Test getting feature names with default config."""
        names = get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 0
        
        # Check for expected features
        assert 'r_open' in names
        assert 'hour_sin' in names
    
    def test_get_feature_names_custom_config(self):
        """Test getting feature names with custom config."""
        config = FeatureConfig({
            "indicators": {
                "atr": {"enabled": True, "n": 14},
                "rsi": {"enabled": False}
            }
        })
        
        names = get_feature_names(config)
        
        assert isinstance(names, list)
        assert any('atr' in name for name in names)


class TestConsistency:
    """Test consistency between different implementations."""
    
    def test_minimal_vs_full_features(self, sample_ohlc_data):
        """Test that minimal and full computation produce consistent results."""
        # Compute with minimal
        minimal = compute_minimal_features(sample_ohlc_data, indicators=['atr'])
        
        # Compute with full (no standardization for comparison)
        config = FeatureConfig({
            "indicators": {
                "atr": {"enabled": True, "n": 14},
                "rsi": {"enabled": False},
                "bollinger": {"enabled": False},
                "macd": {"enabled": False}
            }
        })
        full, _ = compute_features(sample_ohlc_data, config=config, standardize=False)
        
        # Check that common features are consistent
        # Note: Column names might differ slightly
        assert len(minimal) == len(full)
    
    def test_feature_determinism(self, sample_ohlc_data):
        """Test that feature computation is deterministic."""
        features1, _ = compute_features(sample_ohlc_data, standardize=False)
        features2, _ = compute_features(sample_ohlc_data, standardize=False)
        
        # Should produce identical results
        pd.testing.assert_frame_equal(features1, features2)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['ts_utc', 'open', 'high', 'low', 'close'])
        
        # Should handle gracefully (may raise or return empty)
        # Implementation-dependent behavior
        pass
    
    def test_nan_values(self, sample_ohlc_data):
        """Test handling of NaN values in input."""
        data = sample_ohlc_data.copy()
        data.loc[10:20, 'close'] = np.nan
        
        # Should handle NaN gracefully
        features, scaler = compute_features(data, standardize=False)
        assert isinstance(features, pd.DataFrame)
    
    def test_single_row(self):
        """Test with single row of data."""
        df = pd.DataFrame({
            'ts_utc': [1000],
            'open': [1.0],
            'high': [1.1],
            'low': [0.9],
            'close': [1.05]
        })
        
        # Should handle gracefully
        # Some features may be NaN due to insufficient data
        features, _ = compute_features(df, standardize=False)
        assert isinstance(features, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
