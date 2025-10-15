"""
Unit tests for Position Sizer

Tests all position sizing methods and safety constraints.
Part of PROC-001 - automated testing implementation.
"""
import pytest
import numpy as np

from forex_diffusion.risk.position_sizer import PositionSizer, BacktestTradeHistory


class TestPositionSizerBasic:
    """Basic position sizing tests"""
    
    def test_fixed_fractional_basic(self):
        """Test fixed fractional sizing with basic inputs"""
        sizer = PositionSizer(base_risk_pct=2.0)
        
        size = sizer.calculate_position_size(
            method='fixed_fractional',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.0950,  # 50 pip stop
            pip_size=0.0001
        )
        
        # Risk = $200 (2% of $10,000)
        # Stop = 50 pips = $500 per standard lot
        # Size = $200 / $500 = 0.4 lots
        assert size == pytest.approx(0.4, rel=0.01)
    
    def test_fixed_fractional_zero_risk(self):
        """Test fixed fractional with zero risk"""
        sizer = PositionSizer(base_risk_pct=0.0)
        
        size = sizer.calculate_position_size(
            method='fixed_fractional',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.0950
        )
        
        assert size == 0.0
    
    def test_max_position_constraint(self):
        """Test maximum position size constraint"""
        sizer = PositionSizer(
            base_risk_pct=10.0,  # Very high risk
            max_position_size_pct=5.0  # But capped at 5%
        )
        
        size = sizer.calculate_position_size(
            method='fixed_fractional',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.0990  # Small stop = large position
        )
        
        # Should be capped at 5% of balance = $500
        # At 1.1000 price = 0.45 lots max
        assert size <= 0.5  # Rough check
    
    def test_min_position_constraint(self):
        """Test minimum position size constraint"""
        sizer = PositionSizer(
            base_risk_pct=0.01,  # Very small risk
            min_position_size_pct=0.1  # But minimum 0.1%
        )
        
        size = sizer.calculate_position_size(
            method='fixed_fractional',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.0900  # Large stop
        )
        
        # Should respect minimum
        assert size >= 0.01  # At least 0.01 lots


class TestKellyCriterion:
    """Kelly Criterion position sizing tests"""
    
    def test_kelly_basic(self):
        """Test Kelly criterion with winning system"""
        trade_history = BacktestTradeHistory(
            wins=[0.02, 0.03, 0.025],  # 3 wins
            losses=[0.015, 0.01],  # 2 losses
            total_trades=5,
            win_rate=0.6,
            avg_win=0.025,
            avg_loss=0.0125,
            max_consecutive_losses=2
        )
        
        sizer = PositionSizer(kelly_fraction=0.25)
        
        size = sizer.calculate_position_size(
            method='kelly',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            trade_history=trade_history
        )
        
        # Kelly = (p * b - q) / b
        # p = 0.6, q = 0.4, b = 0.025/0.0125 = 2
        # Kelly = (0.6 * 2 - 0.4) / 2 = 0.4
        # Quarter Kelly = 0.1 (10% of capital)
        assert size > 0.0
    
    def test_kelly_losing_system(self):
        """Test Kelly with losing system (negative expectancy)"""
        trade_history = BacktestTradeHistory(
            wins=[0.01],
            losses=[0.02, 0.025, 0.03],
            total_trades=4,
            win_rate=0.25,  # Low win rate
            avg_win=0.01,
            avg_loss=0.025,
            max_consecutive_losses=3
        )
        
        sizer = PositionSizer(kelly_fraction=0.25)
        
        size = sizer.calculate_position_size(
            method='kelly',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            trade_history=trade_history
        )
        
        # Negative expectancy should return minimum or zero
        assert size <= 0.01  # Very small or zero


class TestVolatilityAdjusted:
    """Volatility-adjusted position sizing tests"""
    
    def test_volatility_basic(self):
        """Test volatility-based sizing"""
        sizer = PositionSizer(base_risk_pct=2.0)
        
        # Normal volatility
        size_normal = sizer.calculate_position_size(
            method='volatility_adjusted',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            atr=0.0015  # 15 pips ATR
        )
        
        # High volatility
        size_high_vol = sizer.calculate_position_size(
            method='volatility_adjusted',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            atr=0.0030  # 30 pips ATR (2x normal)
        )
        
        # High volatility should result in smaller position
        assert size_high_vol < size_normal


class TestDrawdownProtection:
    """Drawdown protection tests"""
    
    def test_drawdown_reduction(self):
        """Test position size reduction during drawdown"""
        sizer = PositionSizer(
            base_risk_pct=2.0,
            drawdown_reduction_enabled=True,
            drawdown_size_multiplier=0.5  # Half size in drawdown
        )
        
        # Normal situation
        size_normal = sizer.calculate_position_size(
            method='fixed_fractional',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            current_drawdown_pct=0.0  # No drawdown
        )
        
        # During drawdown
        size_drawdown = sizer.calculate_position_size(
            method='fixed_fractional',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            current_drawdown_pct=12.0  # 12% drawdown
        )
        
        # Size should be reduced
        assert size_drawdown == pytest.approx(size_normal * 0.5, rel=0.01)


class TestPortfolioConstraints:
    """Portfolio-level constraint tests (STRUCT-002)"""
    
    def test_max_sector_exposure(self):
        """Test maximum sector exposure constraint"""
        sizer = PositionSizer(
            base_risk_pct=2.0,
            max_sector_exposure_pct=30.0
        )
        
        # TODO: Implement when portfolio constraints are fully integrated
        pass
    
    def test_correlation_adjustment(self):
        """Test correlation-based position adjustment"""
        sizer = PositionSizer(
            base_risk_pct=2.0,
            correlation_threshold=0.7,
            use_correlation_adjustment=True
        )
        
        # TODO: Implement when correlation logic is added
        pass


class TestEdgeCases:
    """Edge case and error handling tests"""
    
    def test_invalid_method(self):
        """Test invalid method raises error"""
        sizer = PositionSizer()
        
        with pytest.raises((ValueError, KeyError)):
            sizer.calculate_position_size(
                method='invalid_method',
                account_balance=10000,
                entry_price=1.1000,
                stop_loss_price=1.0950
            )
    
    def test_zero_stop_distance(self):
        """Test zero stop distance (entry == stop)"""
        sizer = PositionSizer(base_risk_pct=2.0)
        
        size = sizer.calculate_position_size(
            method='fixed_fractional',
            account_balance=10000,
            entry_price=1.1000,
            stop_loss_price=1.1000  # Same as entry
        )
        
        # Should return 0 or minimal size
        assert size == 0.0 or size == pytest.approx(0.0)
    
    def test_negative_account_balance(self):
        """Test negative account balance"""
        sizer = PositionSizer()
        
        with pytest.raises((ValueError, AssertionError)):
            sizer.calculate_position_size(
                method='fixed_fractional',
                account_balance=-1000,  # Invalid
                entry_price=1.1000,
                stop_loss_price=1.0950
            )


@pytest.mark.slow
class TestPerformance:
    """Performance tests"""
    
    def test_calculation_speed(self):
        """Test position size calculation is fast"""
        import time
        
        sizer = PositionSizer()
        
        start = time.perf_counter()
        
        for _ in range(1000):
            sizer.calculate_position_size(
                method='fixed_fractional',
                account_balance=10000,
                entry_price=1.1000,
                stop_loss_price=1.0950
            )
        
        elapsed = time.perf_counter() - start
        
        # Should complete 1000 calculations in < 100ms
        assert elapsed < 0.1, f"Too slow: {elapsed:.3f}s for 1000 calculations"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
