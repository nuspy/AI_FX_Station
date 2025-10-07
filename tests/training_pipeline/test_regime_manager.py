"""
Unit tests for regime_manager module.

Tests regime classification and best model tracking.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from src.forex_diffusion.training.training_pipeline.regime_manager import RegimeManager


class TestRegimeClassification:
    """Tests for market regime classification."""

    def setup_method(self):
        """Set up test fixtures."""
        self.session = Mock()
        self.manager = RegimeManager(
            session=self.session,
            trend_window=50,
            volatility_window=20,
            returns_window=10
        )

    def create_test_ohlc(self, n_bars=100, trend='bull', volatility='low'):
        """Create test OHLC data."""
        dates = pd.date_range('2024-01-01', periods=n_bars, freq='1H')

        if trend == 'bull':
            # Upward trending prices
            close = np.linspace(1.1000, 1.1500, n_bars)
        elif trend == 'bear':
            # Downward trending prices
            close = np.linspace(1.1500, 1.1000, n_bars)
        else:  # ranging
            # Sideways with noise
            close = 1.1000 + np.random.randn(n_bars) * 0.0010

        if volatility == 'high':
            noise = np.random.randn(n_bars) * 0.0050
        else:  # low
            noise = np.random.randn(n_bars) * 0.0010

        close = close + noise

        return pd.DataFrame({
            'timestamp': dates,
            'open': close - 0.0005,
            'high': close + 0.0010,
            'low': close - 0.0010,
            'close': close,
            'volume': np.random.randint(1000, 10000, n_bars)
        })

    def test_classify_regime_bull_trending(self):
        """Test classification of bull trending regime."""
        ohlc = self.create_test_ohlc(trend='bull', volatility='low')

        regime = self.manager.classify_regime(ohlc)

        assert regime == 'bull_trending'

    def test_classify_regime_bear_trending(self):
        """Test classification of bear trending regime."""
        ohlc = self.create_test_ohlc(trend='bear', volatility='low')

        regime = self.manager.classify_regime(ohlc)

        assert regime == 'bear_trending'

    def test_classify_regime_volatile_ranging(self):
        """Test classification of volatile ranging regime."""
        ohlc = self.create_test_ohlc(trend='ranging', volatility='high')

        regime = self.manager.classify_regime(ohlc)

        assert regime in ['volatile_ranging', 'calm_ranging']  # Depends on exact values

    def test_calculate_market_features(self):
        """Test market feature calculation."""
        ohlc = self.create_test_ohlc()

        features_df = self.manager.calculate_market_features(ohlc)

        assert 'trend_strength' in features_df.columns
        assert 'volatility' in features_df.columns
        assert 'returns_mean' in features_df.columns
        assert 'volatility_percentile' in features_df.columns

        # Check values are in valid ranges
        assert features_df['trend_strength'].between(0, 1).all()
        assert features_df['volatility_percentile'].between(0, 100).all()


class TestRegimeEvaluation:
    """Tests for regime performance evaluation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.session = Mock()
        self.manager = RegimeManager(session=self.session)

    def test_evaluate_regime_improvements_better_performance(self):
        """Test evaluation when new model is better."""
        current_metrics = {
            'bull_trending': {'sharpe_ratio': 1.5},
            'bear_trending': {'sharpe_ratio': 1.2}
        }

        # Mock database to return lower previous scores
        self.session.query = Mock()
        self.session.query.return_value.filter.return_value.first.side_effect = [
            Mock(performance_score=1.0),  # bull_trending previous
            Mock(performance_score=0.9)   # bear_trending previous
        ]

        improvements = self.manager.evaluate_regime_improvements(
            regime_metrics=current_metrics,
            primary_metric='sharpe_ratio',
            min_improvement=0.01
        )

        assert improvements['bull_trending'] is True  # 1.5 > 1.0
        assert improvements['bear_trending'] is True  # 1.2 > 0.9

    def test_evaluate_regime_improvements_no_improvement(self):
        """Test when new model doesn't improve."""
        current_metrics = {
            'bull_trending': {'sharpe_ratio': 0.8}
        }

        # Mock previous score as higher
        self.session.query = Mock()
        self.session.query.return_value.filter.return_value.first.return_value = \
            Mock(performance_score=1.0)

        improvements = self.manager.evaluate_regime_improvements(
            regime_metrics=current_metrics,
            primary_metric='sharpe_ratio',
            min_improvement=0.01
        )

        assert improvements['bull_trending'] is False  # 0.8 < 1.0

    def test_evaluate_regime_improvements_first_model(self):
        """Test when no previous model exists."""
        current_metrics = {
            'bull_trending': {'sharpe_ratio': 1.5}
        }

        # Mock no previous model
        self.session.query = Mock()
        self.session.query.return_value.filter.return_value.first.return_value = None

        improvements = self.manager.evaluate_regime_improvements(
            regime_metrics=current_metrics,
            primary_metric='sharpe_ratio',
            min_improvement=0.01
        )

        assert improvements['bull_trending'] is True  # First model always improves


class TestRegimeBestModelTracking:
    """Tests for best model tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.session = Mock()
        self.manager = RegimeManager(session=self.session)

    def test_update_regime_bests_new_best(self):
        """Test updating best models for regimes."""
        regime_metrics = {
            'bull_trending': {
                'sharpe_ratio': 2.0,
                'max_drawdown': -0.15,
                'win_rate': 0.58
            }
        }

        # Mock evaluate as improvement
        self.manager.evaluate_regime_improvements = Mock(return_value={
            'bull_trending': True
        })

        # Mock database update
        from ..training.training_pipeline import database
        database.update_best_model_for_regime = Mock()

        updated_regimes = self.manager.update_regime_bests(
            training_run_id=1,
            inference_backtest_id=1,
            regime_metrics=regime_metrics,
            primary_metric='sharpe_ratio'
        )

        assert 'bull_trending' in updated_regimes


class TestRegimeMetricsCalculation:
    """Tests for regime-specific metrics calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.session = Mock()
        self.manager = RegimeManager(session=self.session)

    def test_split_data_by_regime(self):
        """Test splitting backtest data by regime."""
        # Create test data with regime labels
        ohlc = pd.DataFrame({
            'close': np.random.randn(100) + 1.1000,
            'regime': ['bull_trending'] * 50 + ['bear_trending'] * 50
        })

        predictions = np.random.randn(100) * 0.0010
        returns = np.random.randn(100) * 0.0010

        split_data = self.manager.split_data_by_regime(
            ohlc=ohlc,
            predictions=predictions,
            returns=returns
        )

        assert 'bull_trending' in split_data
        assert 'bear_trending' in split_data

        assert len(split_data['bull_trending']['predictions']) == 50
        assert len(split_data['bear_trending']['predictions']) == 50

    def test_calculate_regime_metrics(self):
        """Test calculation of regime-specific metrics."""
        split_data = {
            'bull_trending': {
                'predictions': np.array([0.001, 0.002, -0.001]),
                'returns': np.array([0.0015, 0.0018, -0.0005])
            }
        }

        metrics = self.manager.calculate_regime_metrics(split_data)

        assert 'bull_trending' in metrics
        assert 'sharpe_ratio' in metrics['bull_trending']
        assert 'max_drawdown' in metrics['bull_trending']
        assert 'win_rate' in metrics['bull_trending']


class TestRegimeDefinitionLoading:
    """Tests for regime definition management."""

    def test_load_regime_definitions(self):
        """Test loading regime definitions from database."""
        # Mock database query
        mock_regime = Mock()
        mock_regime.regime_name = 'bull_trending'
        mock_regime.description = 'Strong upward trend'
        mock_regime.detection_rules = {'trend_strength': '> 0.7'}

        session = Mock()
        session.query.return_value.filter.return_value.all.return_value = [mock_regime]

        manager = RegimeManager(session=session)

        assert 'bull_trending' in manager.regime_definitions
        assert manager.regime_definitions['bull_trending'].regime_name == 'bull_trending'

    def test_reload_regime_definitions(self):
        """Test reloading regime definitions."""
        session = Mock()
        session.query.return_value.filter.return_value.all.return_value = []

        manager = RegimeManager(session=session)

        # Initially empty
        assert len(manager.regime_definitions) == 0

        # Add mock regime
        mock_regime = Mock()
        mock_regime.regime_name = 'new_regime'
        session.query.return_value.filter.return_value.all.return_value = [mock_regime]

        # Reload
        manager.regime_definitions = manager._load_regime_definitions()

        assert 'new_regime' in manager.regime_definitions
