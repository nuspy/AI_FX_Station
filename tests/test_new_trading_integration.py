"""
Integration tests for new trading system components.

Tests the integration of:
- ParameterLoaderService
- AdaptiveStopLossManager
- PositionSizer
- AdvancedMetricsCalculator
- RiskProfileLoader
"""

import sys
from pathlib import Path
import pytest
import tempfile
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from forex_diffusion.training_pipeline.database_models import (
    Base,
    OptimizedParameters,
    RiskProfile,
    AdvancedMetrics,
)
from forex_diffusion.services.parameter_loader import ParameterLoaderService
from forex_diffusion.services.risk_profile_loader import RiskProfileLoader
from forex_diffusion.risk.adaptive_stop_loss_manager import (
    AdaptiveStopLossManager,
    AdaptationFactors,
)
from forex_diffusion.risk.position_sizer import (
    PositionSizer,
    BacktestTradeHistory,
)
from forex_diffusion.backtest.advanced_metrics_calculator import (
    AdvancedMetricsCalculator,
)


@pytest.fixture
def test_db():
    """Create temporary test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


def test_parameter_loader_integration(test_db):
    """Test ParameterLoaderService with database."""
    # Create test parameters
    engine = create_engine(f"sqlite:///{test_db}")
    Session = sessionmaker(bind=engine)
    session = Session()

    params = OptimizedParameters(
        pattern_type='harmonic',
        symbol='EURUSD',
        timeframe='15m',
        market_regime='trending_up',
        form_params='{"tolerance_pct": 2.0}',
        action_params='{"sl_atr_multiplier": 2.5, "tp_atr_multiplier": 4.0}',
        performance_metrics='{"sharpe": 2.3}',
        optimization_timestamp=datetime.now(),
        data_range_start=datetime.now() - timedelta(days=365),
        data_range_end=datetime.now(),
        sample_count=500,
        validation_status='validated',
    )
    session.add(params)
    session.commit()
    session.close()

    # Test loader
    loader = ParameterLoaderService(test_db, cache_ttl_seconds=60)
    loaded = loader.load_parameters('harmonic', 'EURUSD', '15m', 'trending_up')

    assert loaded is not None
    assert loaded.pattern_type == 'harmonic'
    assert loaded.action_params['sl_atr_multiplier'] == 2.5
    assert loaded.source == 'database'

    # Test cache
    loaded2 = loader.load_parameters('harmonic', 'EURUSD', '15m', 'trending_up')
    assert loaded2.source == 'cache'


def test_adaptive_sl_manager():
    """Test AdaptiveStopLossManager calculations."""
    manager = AdaptiveStopLossManager(
        base_sl_atr_multiplier=2.0,
        base_tp_atr_multiplier=3.0,
        trailing_enabled=True,
    )

    # Test initial stops calculation
    sl, tp, levels = manager.calculate_initial_stops(
        symbol='EURUSD',
        direction='long',
        entry_price=1.0850,
        atr=0.0015,
        current_spread=0.00002,
        avg_spread=0.00002,
        regime='trending_up',
    )

    assert sl < 1.0850  # Stop below entry for long
    assert tp > 1.0850  # Take profit above entry
    assert len(levels) == 2  # Hard and volatility stops

    # Test adaptation
    factors = AdaptationFactors(
        atr=0.0015,
        current_spread=0.00002,
        avg_spread=0.00002,
        spread_ratio=1.0,
        news_risk_level='medium',
        regime='trending_up',
        time_in_position_hours=2.0,
        unrealized_pnl_pct=5.0,
    )

    new_sl, new_tp, triggered, reason = manager.update_stops(
        symbol='EURUSD',
        direction='long',
        entry_price=1.0850,
        entry_time=datetime.now() - timedelta(hours=2),
        current_price=1.0900,
        current_stop_loss=sl,
        current_take_profit=tp,
        factors=factors,
    )

    # Should tighten due to news risk
    assert new_sl >= sl or not triggered


def test_position_sizer_kelly():
    """Test PositionSizer with Kelly Criterion."""
    sizer = PositionSizer(
        base_risk_pct=1.0,
        kelly_fraction=0.25,
    )

    # Create backtest history
    history = BacktestTradeHistory(
        wins=[0.02, 0.03, 0.025],  # 3 wins
        losses=[0.01, 0.015],  # 2 losses
        total_trades=5,
        win_rate=0.6,
        avg_win=0.025,
        avg_loss=0.0125,
        max_consecutive_losses=2,
    )

    # Calculate position size
    size, metadata = sizer.calculate_position_size(
        method='kelly',
        account_balance=10000,
        entry_price=1.0850,
        stop_loss_price=1.0820,
        backtest_history=history,
    )

    assert size > 0
    assert metadata['method'] == 'kelly'
    assert 0.1 <= metadata['size_pct'] <= 5.0  # Within constraints


def test_position_sizer_drawdown_protection():
    """Test position sizer drawdown protection."""
    sizer = PositionSizer(
        base_risk_pct=1.0,
        drawdown_reduction_enabled=True,
        drawdown_threshold_pct=10.0,
        drawdown_size_multiplier=0.5,
    )

    # Normal situation
    size1, meta1 = sizer.calculate_position_size(
        method='fixed_fractional',
        account_balance=10000,
        entry_price=1.0850,
        stop_loss_price=1.0820,
        current_drawdown_pct=5.0,
    )

    # In drawdown
    size2, meta2 = sizer.calculate_position_size(
        method='fixed_fractional',
        account_balance=10000,
        entry_price=1.0850,
        stop_loss_price=1.0820,
        current_drawdown_pct=15.0,
    )

    # Size should be reduced in drawdown
    assert size2 < size1
    assert meta2['drawdown_protection_active'] == True


def test_advanced_metrics_calculator():
    """Test AdvancedMetricsCalculator."""
    calculator = AdvancedMetricsCalculator(risk_free_rate=0.02)

    # Create synthetic equity curve
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    equity = pd.Series(
        10000 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01)),
        index=dates
    )
    returns = equity.pct_change().fillna(0)

    # Create synthetic trades
    trades = [
        {'pnl': 100, 'entry_time': dates[i], 'exit_time': dates[i+1]}
        for i in range(0, 100, 10)
    ]
    # Add some losses
    trades.extend([
        {'pnl': -50, 'entry_time': dates[i], 'exit_time': dates[i+1]}
        for i in range(5, 50, 15)
    ])

    # Calculate metrics
    metrics = calculator.calculate(
        equity_curve=equity,
        returns=returns,
        trades=trades,
        period_start=dates[0].to_pydatetime(),
        period_end=dates[-1].to_pydatetime(),
    )

    # Verify metrics are calculated
    assert metrics.total_trades == len(trades)
    assert metrics.sharpe_ratio != 0
    assert metrics.sortino_ratio != 0
    assert metrics.max_drawdown_pct <= 0
    assert -100 <= metrics.return_skewness <= 100
    assert metrics.system_quality_number != 0


def test_risk_profile_integration(test_db):
    """Test RiskProfileLoader with database."""
    # Create test profile
    engine = create_engine(f"sqlite:///{test_db}")
    Session = sessionmaker(bind=engine)
    session = Session()

    profile = RiskProfile(
        profile_name='TestProfile',
        profile_type='custom',
        is_active=True,
        max_risk_per_trade_pct=1.5,
        max_portfolio_risk_pct=7.0,
        position_sizing_method='kelly',
        kelly_fraction=0.25,
        base_sl_atr_multiplier=2.0,
        base_tp_atr_multiplier=3.0,
        use_trailing_stop=True,
        trailing_activation_pct=50.0,
        regime_adjustment_enabled=True,
        volatility_adjustment_enabled=True,
        news_awareness_enabled=True,
        max_correlated_positions=2,
        correlation_threshold=0.7,
        max_positions_per_symbol=2,
        max_total_positions=5,
        max_daily_loss_pct=2.0,
        max_drawdown_pct=10.0,
        recovery_mode_threshold_pct=5.0,
        recovery_risk_multiplier=0.5,
    )
    session.add(profile)
    session.commit()
    session.close()

    # Test loader
    loader = RiskProfileLoader(test_db)
    loaded = loader.load_active_profile()

    assert loaded is not None
    assert loaded.profile_name == 'TestProfile'
    assert loaded.max_risk_per_trade_pct == 1.5
    assert loaded.position_sizing_method == 'kelly'

    # Test list
    all_profiles = loader.list_all_profiles()
    assert 'TestProfile' in all_profiles


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
