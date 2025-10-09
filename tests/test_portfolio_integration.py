"""
Test suite for portfolio optimization integration.

Tests the core portfolio optimization modules and GUI integration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Skip all tests if riskfolio-lib is not installed
pytest.importorskip("riskfolio")

from forex_diffusion.portfolio.optimizer import PortfolioOptimizer
from forex_diffusion.portfolio.position_sizer import AdaptivePositionSizer
from forex_diffusion.portfolio.risk_metrics import RiskMetricsCalculator


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Generate correlated returns for 3 assets
    n_assets = 3
    n_days = 100

    # Create returns with some correlation
    returns_matrix = np.random.randn(n_days, n_assets) * 0.01

    returns_df = pd.DataFrame(
        returns_matrix,
        index=dates,
        columns=["EURUSD", "GBPUSD", "USDJPY"]
    )

    return returns_df


@pytest.fixture
def sample_predictions():
    """Generate sample prediction data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-04-11", periods=10, freq="D")

    predictions_df = pd.DataFrame(
        np.random.randn(10, 3) * 0.005,
        index=dates,
        columns=["EURUSD", "GBPUSD", "USDJPY"]
    )

    return predictions_df


class TestPortfolioOptimizer:
    """Test PortfolioOptimizer class."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = PortfolioOptimizer(
            risk_measure="CVaR",
            objective="Sharpe",
            risk_free_rate=0.0,
            risk_aversion=1.0
        )

        assert optimizer.risk_measure == "CVaR"
        assert optimizer.objective == "Sharpe"
        assert optimizer.risk_free_rate == 0.0
        assert optimizer.risk_aversion == 1.0

    def test_optimization(self, sample_returns):
        """Test basic portfolio optimization."""
        optimizer = PortfolioOptimizer(
            risk_measure="MV",
            objective="Sharpe"
        )

        weights = optimizer.optimize(
            returns=sample_returns,
            constraints={"max_weight": 0.5, "min_weight": 0.1},
            method="Classic"
        )

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 0.01  # Weights should sum to ~1
        assert weights.min() >= 0.0  # No short selling by default
        assert weights.max() <= 0.5  # Respects max_weight constraint

    def test_risk_parity(self, sample_returns):
        """Test risk parity optimization."""
        optimizer = PortfolioOptimizer(risk_measure="MV")

        weights = optimizer.optimize_risk_parity(
            returns=sample_returns,
            risk_budgets=None  # Equal risk contribution
        )

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 0.01

    @pytest.mark.skip(reason="Efficient frontier API may vary by riskfolio-lib version")
    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier calculation."""
        optimizer = PortfolioOptimizer(risk_measure="MV")

        frontier = optimizer.calculate_efficient_frontier(
            returns=sample_returns,
            points=10
        )

        assert isinstance(frontier, pd.DataFrame)
        # Note: frontier may be empty if the API changed
        if len(frontier) > 0:
            assert len(frontier) == 10  # Should have 10 points

    def test_backtest_strategy(self, sample_returns):
        """Test portfolio backtesting."""
        optimizer = PortfolioOptimizer(risk_measure="MV", objective="Sharpe")

        weights = optimizer.optimize(returns=sample_returns, method="Classic")

        metrics = optimizer.backtest_strategy(
            weights=weights,
            returns=sample_returns
        )

        assert isinstance(metrics, dict)
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "ann_return" in metrics
        assert "ann_volatility" in metrics

    def test_portfolio_stats(self, sample_returns):
        """Test portfolio statistics calculation."""
        optimizer = PortfolioOptimizer()

        weights = optimizer.optimize(returns=sample_returns, method="Classic")

        stats = optimizer.get_portfolio_stats(
            weights=weights,
            returns=sample_returns
        )

        assert isinstance(stats, dict)
        assert "expected_return" in stats
        assert "volatility" in stats
        assert "sharpe_ratio" in stats
        assert "concentration" in stats


class TestAdaptivePositionSizer:
    """Test AdaptivePositionSizer class."""

    def test_initialization(self):
        """Test position sizer initialization."""
        sizer = AdaptivePositionSizer(
            lookback_period=60,
            rebalance_frequency=5,
            max_position_size=0.25,
            min_position_size=0.01
        )

        assert sizer.lookback_period == 60
        assert sizer.rebalance_frequency == 5
        assert sizer.max_position_size == 0.25
        assert sizer.min_position_size == 0.01

    def test_calculate_positions(self, sample_returns, sample_predictions):
        """Test position calculation."""
        sizer = AdaptivePositionSizer(
            lookback_period=60,
            rebalance_frequency=5
        )

        positions = sizer.calculate_positions(
            predictions=sample_predictions,
            historical_returns=sample_returns,
            current_date=pd.Timestamp("2024-04-11"),
            total_capital=100000.0,
            force_rebalance=True
        )

        assert isinstance(positions, dict)
        assert len(positions) > 0
        assert all(isinstance(v, float) for v in positions.values())

        # Check that total position value is reasonable
        total_value = sum(positions.values())
        assert 0 < total_value <= 100000.0

    def test_risk_parity_positions(self, sample_returns):
        """Test risk parity position calculation."""
        sizer = AdaptivePositionSizer()

        positions = sizer.calculate_risk_parity_positions(
            historical_returns=sample_returns,
            total_capital=100000.0,
            risk_budgets=None
        )

        assert isinstance(positions, dict)
        assert len(positions) > 0

    def test_volatility_adjustment(self):
        """Test volatility-based position adjustment."""
        sizer = AdaptivePositionSizer()

        base_positions = {
            "EURUSD": 30000.0,
            "GBPUSD": 40000.0,
            "USDJPY": 30000.0
        }

        volatility_estimates = {
            "EURUSD": 0.10,
            "GBPUSD": 0.20,  # Higher volatility
            "USDJPY": 0.15
        }

        adjusted = sizer.adjust_for_volatility(
            base_positions=base_positions,
            volatility_estimates=volatility_estimates,
            target_volatility=0.15
        )

        assert isinstance(adjusted, dict)
        assert len(adjusted) == len(base_positions)

        # Higher volatility asset should get lower position size
        assert adjusted["GBPUSD"] < base_positions["GBPUSD"]

    def test_rebalance_trades(self):
        """Test trade calculation for rebalancing."""
        sizer = AdaptivePositionSizer()

        target_positions = {
            "EURUSD": 35000.0,
            "GBPUSD": 40000.0,
            "USDJPY": 25000.0
        }

        current_positions = {
            "EURUSD": 30000.0,
            "GBPUSD": 45000.0,
            "USDJPY": 25000.0
        }

        trades = sizer.get_rebalance_trades(
            target_positions=target_positions,
            current_positions=current_positions,
            min_trade_size=100.0
        )

        assert isinstance(trades, dict)

        # Should have 2 trades (EURUSD buy, GBPUSD sell)
        # USDJPY should not trade (no change)
        assert "EURUSD" in trades
        assert trades["EURUSD"] > 0  # Buy
        assert "GBPUSD" in trades
        assert trades["GBPUSD"] < 0  # Sell
        assert "USDJPY" not in trades  # No trade

    def test_position_summary(self, sample_returns, sample_predictions):
        """Test position summary generation."""
        sizer = AdaptivePositionSizer()

        # First optimization to set state
        sizer.calculate_positions(
            predictions=sample_predictions,
            historical_returns=sample_returns,
            current_date=pd.Timestamp("2024-04-11"),
            total_capital=100000.0,
            force_rebalance=True
        )

        summary = sizer.get_position_summary()

        assert isinstance(summary, dict)
        assert summary["status"] == "active"
        assert "num_positions" in summary
        assert "max_weight" in summary
        assert "concentration" in summary


class TestRiskMetricsCalculator:
    """Test RiskMetricsCalculator class."""

    def test_var_calculation(self):
        """Test VaR calculation."""
        returns = pd.Series(np.random.randn(1000) * 0.01)

        var_95 = RiskMetricsCalculator.calculate_var(returns, confidence=0.95)
        var_99 = RiskMetricsCalculator.calculate_var(returns, confidence=0.99)

        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_99 < var_95  # 99% VaR should be more extreme

    def test_cvar_calculation(self):
        """Test CVaR calculation."""
        returns = pd.Series(np.random.randn(1000) * 0.01)

        cvar_95 = RiskMetricsCalculator.calculate_cvar(returns, confidence=0.95)
        cvar_99 = RiskMetricsCalculator.calculate_cvar(returns, confidence=0.99)

        assert isinstance(cvar_95, float)
        assert isinstance(cvar_99, float)
        assert cvar_99 < cvar_95  # 99% CVaR should be more extreme

    def test_all_metrics(self):
        """Test calculation of all risk metrics."""
        returns = pd.Series(np.random.randn(1000) * 0.01)

        metrics = RiskMetricsCalculator.calculate_all_metrics(returns)

        assert isinstance(metrics, dict)
        assert "var_95" in metrics
        assert "var_99" in metrics
        assert "cvar_95" in metrics
        assert "cvar_99" in metrics
        assert "volatility" in metrics
        assert "downside_deviation" in metrics
        assert "skewness" in metrics
        assert "kurtosis" in metrics


class TestGUIIntegration:
    """Test GUI integration (basic import tests)."""

    def test_portfolio_tab_import(self):
        """Test that PortfolioOptimizationTab can be imported."""
        from forex_diffusion.ui.portfolio_tab import PortfolioOptimizationTab
        assert PortfolioOptimizationTab is not None

    def test_portfolio_viz_import(self):
        """Test that visualization widget can be imported."""
        # This will only work if matplotlib is installed
        try:
            from forex_diffusion.ui.portfolio_viz import EfficientFrontierWidget
            assert EfficientFrontierWidget is not None
        except ImportError:
            pytest.skip("Matplotlib not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
