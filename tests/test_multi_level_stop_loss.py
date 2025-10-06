"""
Test suite for Multi-Level Stop Loss system.

Tests all stop loss types and their priority ordering.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forex_diffusion.risk.multi_level_stop_loss import (
    MultiLevelStopLoss,
    StopLossType,
    StopLossLevel
)


def test_volatility_stop_long():
    """Test ATR-based volatility stop for long position."""
    risk_manager = MultiLevelStopLoss(atr_multiplier=2.0)

    position = {
        'entry_price': 1.1000,
        'direction': 'long',
        'entry_time': pd.Timestamp.now(),
        'account_balance': 10000,
        'size': 10000
    }

    atr = 0.0015
    # Price drops below entry - 2*ATR
    current_price = 1.0970  # 1.1000 - 2*0.0015 = 1.0970

    triggered, stop_type, reason = risk_manager.check_stop_triggered(
        position, current_price, atr
    )

    assert triggered == True
    assert stop_type == StopLossType.VOLATILITY
    assert "Volatility stop" in reason
    print(f"✅ PASS: Volatility stop triggered correctly - {reason}")


def test_volatility_stop_short():
    """Test ATR-based volatility stop for short position."""
    risk_manager = MultiLevelStopLoss(atr_multiplier=2.0)

    position = {
        'entry_price': 1.1000,
        'direction': 'short',
        'entry_time': pd.Timestamp.now(),
        'account_balance': 10000,
        'size': 10000
    }

    atr = 0.0015
    # Price rises above entry + 2*ATR
    current_price = 1.1030  # 1.1000 + 2*0.0015 = 1.1030

    triggered, stop_type, reason = risk_manager.check_stop_triggered(
        position, current_price, atr
    )

    assert triggered == True
    assert stop_type == StopLossType.VOLATILITY
    print(f"✅ PASS: Volatility stop (short) triggered correctly - {reason}")


def test_time_stop():
    """Test maximum holding period stop."""
    risk_manager = MultiLevelStopLoss(max_holding_hours=48)

    # Position held for 50 hours
    position = {
        'entry_price': 1.1000,
        'direction': 'long',
        'entry_time': pd.Timestamp.now() - pd.Timedelta(hours=50),
        'account_balance': 10000,
        'size': 10000
    }

    atr = 0.0015
    current_price = 1.1020  # Profitable position

    triggered, stop_type, reason = risk_manager.check_stop_triggered(
        position, current_price, atr
    )

    assert triggered == True
    assert stop_type == StopLossType.TIME
    assert "50" in reason  # Should mention hours held
    print(f"✅ PASS: Time stop triggered correctly - {reason}")


def test_technical_stop():
    """Test pattern invalidation stop."""
    risk_manager = MultiLevelStopLoss()

    position = {
        'entry_price': 1.1000,
        'direction': 'long',
        'entry_time': pd.Timestamp.now(),
        'pattern_invalidation_price': 1.0980,
        'account_balance': 10000,
        'size': 10000
    }

    atr = 0.0015
    current_price = 1.0975  # Below pattern invalidation

    triggered, stop_type, reason = risk_manager.check_stop_triggered(
        position, current_price, atr
    )

    assert triggered == True
    assert stop_type == StopLossType.TECHNICAL
    print(f"✅ PASS: Technical stop triggered correctly - {reason}")


def test_correlation_stop():
    """Test correlation-based systemic risk stop."""
    risk_manager = MultiLevelStopLoss(correlation_threshold=0.85)

    position = {
        'entry_price': 1.1000,
        'direction': 'long',
        'entry_time': pd.Timestamp.now(),
        'account_balance': 10000,
        'size': 10000
    }

    atr = 0.0015
    current_price = 1.1020  # Profitable
    market_correlation = 0.90  # High correlation (systemic risk)

    triggered, stop_type, reason = risk_manager.check_stop_triggered(
        position, current_price, atr, market_correlation
    )

    assert triggered == True
    assert stop_type == StopLossType.CORRELATION
    assert "0.90" in reason
    print(f"✅ PASS: Correlation stop triggered correctly - {reason}")


def test_daily_loss_limit():
    """Test daily loss limit protection."""
    risk_manager = MultiLevelStopLoss(daily_loss_limit_pct=3.0)

    # Simulate daily loss approaching limit
    risk_manager.daily_pnl = -240  # 2.4% of 10k account (80% of 3% limit)

    position = {
        'entry_price': 1.1000,
        'direction': 'long',
        'entry_time': pd.Timestamp.now(),
        'account_balance': 10000,
        'size': 10000
    }

    atr = 0.0015
    current_price = 1.1020  # Any price

    triggered, stop_type, reason = risk_manager.check_stop_triggered(
        position, current_price, atr
    )

    assert triggered == True
    assert stop_type == StopLossType.DAILY_LOSS
    print(f"✅ PASS: Daily loss limit triggered correctly - {reason}")


def test_trailing_stop_long():
    """Test trailing stop for long position."""
    risk_manager = MultiLevelStopLoss(trailing_stop_pct=2.0)

    position = {
        'entry_price': 1.1000,
        'direction': 'long',
        'entry_time': pd.Timestamp.now(),
        'highest_price': 1.1100,  # Position went to profit
        'account_balance': 10000,
        'size': 10000
    }

    atr = 0.0015
    # Price drops below trailing stop (1.1100 * 0.98 = 1.0878)
    current_price = 1.0870

    triggered, stop_type, reason = risk_manager.check_stop_triggered(
        position, current_price, atr
    )

    assert triggered == True
    assert stop_type == StopLossType.TRAILING
    print(f"✅ PASS: Trailing stop triggered correctly - {reason}")


def test_no_stop_triggered():
    """Test that stops don't trigger when all conditions normal."""
    risk_manager = MultiLevelStopLoss()

    position = {
        'entry_price': 1.1000,
        'direction': 'long',
        'entry_time': pd.Timestamp.now(),
        'account_balance': 10000,
        'size': 10000
    }

    atr = 0.0015
    current_price = 1.1050  # Profitable, all stops clear

    triggered, stop_type, reason = risk_manager.check_stop_triggered(
        position, current_price, atr
    )

    assert triggered == False
    assert stop_type is None
    assert reason is None
    print("✅ PASS: No stop triggered when all conditions normal")


def test_priority_ordering():
    """Test that stops are checked in correct priority order."""
    risk_manager = MultiLevelStopLoss(
        atr_multiplier=2.0,
        max_holding_hours=48
    )

    # Position that triggers multiple stops
    position = {
        'entry_price': 1.1000,
        'direction': 'long',
        'entry_time': pd.Timestamp.now() - pd.Timedelta(hours=50),  # Time stop
        'pattern_invalidation_price': 1.0980,  # Technical stop
        'account_balance': 10000,
        'size': 10000
    }

    atr = 0.0015
    current_price = 1.0970  # Below technical and volatility stops

    triggered, stop_type, reason = risk_manager.check_stop_triggered(
        position, current_price, atr
    )

    # TIME should trigger first (higher priority than TECHNICAL/VOLATILITY)
    assert triggered == True
    assert stop_type == StopLossType.TIME
    print(f"✅ PASS: Priority ordering correct - TIME before TECHNICAL/VOLATILITY")


def test_update_trailing_stops():
    """Test trailing stop updates with price movement."""
    risk_manager = MultiLevelStopLoss()

    # Long position
    position = {
        'entry_price': 1.1000,
        'direction': 'long',
        'entry_time': pd.Timestamp.now(),
        'account_balance': 10000,
        'size': 10000
    }

    # Price moves up - should update highest_price
    position = risk_manager.update_trailing_stops(position, 1.1050)
    assert position['highest_price'] == 1.1050

    # Price moves higher
    position = risk_manager.update_trailing_stops(position, 1.1100)
    assert position['highest_price'] == 1.1100

    # Price moves down - highest_price should NOT change
    position = risk_manager.update_trailing_stops(position, 1.1080)
    assert position['highest_price'] == 1.1100

    print("✅ PASS: Trailing stops update correctly")


def test_risk_metrics():
    """Test risk metrics calculation."""
    risk_manager = MultiLevelStopLoss(atr_multiplier=2.0)

    position = {
        'entry_price': 1.1000,
        'direction': 'long',
        'entry_time': pd.Timestamp.now(),
        'account_balance': 10000,
        'size': 10000
    }

    atr = 0.0015
    current_price = 1.1050  # +50 pips profit

    metrics = risk_manager.get_risk_metrics(position, current_price, atr)

    # Check metrics structure
    assert 'unrealized_pnl' in metrics
    assert 'daily_pnl' in metrics
    assert 'position_risk_pips' in metrics
    assert 'stop_distances' in metrics
    assert 'active_stops' in metrics

    # Verify PnL calculation
    expected_pnl = (1.1050 - 1.1000) * 10000
    assert abs(metrics['unrealized_pnl'] - expected_pnl) < 0.01

    print(f"✅ PASS: Risk metrics calculated correctly")
    print(f"  Unrealized P&L: ${metrics['unrealized_pnl']:.2f}")
    print(f"  Position Risk: {metrics['position_risk_pips']:.1f} pips")
    print(f"  Active Stops: {metrics['active_stops']}")


def test_daily_pnl_reset():
    """Test that daily P&L resets at midnight."""
    risk_manager = MultiLevelStopLoss()

    # Set daily PnL
    risk_manager.daily_pnl = -100
    risk_manager.daily_pnl_reset_time = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)

    # Update with new PnL - should trigger reset
    risk_manager.update_daily_pnl(-50)

    # Daily PnL should be reset + new value
    assert risk_manager.daily_pnl == -50
    assert risk_manager.daily_pnl_reset_time == pd.Timestamp.now().normalize()

    print("✅ PASS: Daily P&L resets correctly at midnight")


if __name__ == "__main__":
    """Run all tests."""
    print("=" * 80)
    print("MULTI-LEVEL STOP LOSS TEST SUITE")
    print("=" * 80)
    print()

    tests = [
        ("Volatility Stop (Long)", test_volatility_stop_long),
        ("Volatility Stop (Short)", test_volatility_stop_short),
        ("Time Stop", test_time_stop),
        ("Technical Stop", test_technical_stop),
        ("Correlation Stop", test_correlation_stop),
        ("Daily Loss Limit", test_daily_loss_limit),
        ("Trailing Stop (Long)", test_trailing_stop_long),
        ("No Stop Triggered", test_no_stop_triggered),
        ("Priority Ordering", test_priority_ordering),
        ("Update Trailing Stops", test_update_trailing_stops),
        ("Risk Metrics", test_risk_metrics),
        ("Daily P&L Reset", test_daily_pnl_reset),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 80)
        try:
            test_func()
            passed += 1
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
        print("\n✅ ALL TESTS PASSED - Multi-level stop loss working correctly!")
        sys.exit(0)
    else:
        print(f"\n❌ {failed} TEST(S) FAILED - Review immediately!")
        sys.exit(1)
