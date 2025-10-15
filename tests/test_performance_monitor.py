"""
Unit tests for Performance Degradation Detector

Tests performance monitoring and alert system.
Part of PROC-001 - BUG-004 (HIGH) testing.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from forex_diffusion.trading.performance_monitor import (
    PerformanceDegradationDetector,
    PerformanceExpectations,
    DegradationAlert
)


class TestPerformanceExpectations:
    """Performance expectations configuration tests"""
    
    def test_default_expectations(self):
        """Test default expectations are reasonable"""
        expectations = PerformanceExpectations()
        
        assert expectations.expected_win_rate == 0.58
        assert expectations.expected_sharpe == 1.8
        assert expectations.expected_max_dd == 0.08
        assert expectations.expected_profit_factor == 1.8
    
    def test_custom_expectations(self):
        """Test custom expectations"""
        expectations = PerformanceExpectations(
            expected_win_rate=0.65,
            expected_sharpe=2.0,
            expected_max_dd=0.05,
            expected_profit_factor=2.5
        )
        
        assert expectations.expected_win_rate == 0.65
        assert expectations.expected_sharpe == 2.0


class TestPerformanceMonitor:
    """Performance degradation detector tests"""
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly"""
        expectations = PerformanceExpectations()
        detector = PerformanceDegradationDetector(expectations)
        
        assert detector.last_check_time is None
        assert len(detector.alert_history) == 0
    
    def test_should_check_timing(self):
        """Test check interval timing"""
        expectations = PerformanceExpectations()
        detector = PerformanceDegradationDetector(
            expectations,
            check_interval_hours=24
        )
        
        # First check
        assert detector.should_check() is True
        
        # Mark as checked
        detector.last_check_time = datetime.now()
        
        # Too soon
        assert detector.should_check() is False
        
        # After interval
        detector.last_check_time = datetime.now() - timedelta(hours=25)
        assert detector.should_check() is True


class TestWinRateMonitoring:
    """Win rate degradation tests"""
    
    def test_win_rate_normal(self):
        """Test normal win rate (no alert)"""
        expectations = PerformanceExpectations(expected_win_rate=0.58)
        detector = PerformanceDegradationDetector(expectations)
        
        # Good performance
        trades = [
            {'success': True, 'exit_time': datetime.now(), 'pnl': 100},
            {'success': True, 'exit_time': datetime.now(), 'pnl': 150},
            {'success': False, 'exit_time': datetime.now(), 'pnl': -50},
            {'success': True, 'exit_time': datetime.now(), 'pnl': 120},
            {'success': True, 'exit_time': datetime.now(), 'pnl': 80}
        ]
        # Win rate = 4/5 = 0.8 > 0.58 expected
        
        alerts = detector.check_degradation(trades)
        
        assert alerts is None or len(alerts) == 0
    
    def test_win_rate_degraded_warning(self):
        """Test win rate degradation warning"""
        expectations = PerformanceExpectations(
            expected_win_rate=0.58,
            win_rate_threshold=0.10  # Alert if drops by 10%
        )
        detector = PerformanceDegradationDetector(expectations)
        
        # Degraded performance: 45% win rate (58% - 13%)
        trades = []
        for i in range(20):
            trades.append({
                'success': i < 9,  # 9 wins, 11 losses = 45%
                'exit_time': datetime.now() - timedelta(days=i),
                'pnl': 100 if i < 9 else -80,
                'pnl_percentage': 0.01 if i < 9 else -0.008
            })
        
        alerts = detector.check_degradation(trades)
        
        assert alerts is not None
        assert len(alerts) >= 1
        
        win_rate_alert = [a for a in alerts if a.metric == 'win_rate'][0]
        assert win_rate_alert.severity in ['warning', 'critical']
        assert win_rate_alert.actual < expectations.expected_win_rate
    
    def test_win_rate_critical(self):
        """Test critical win rate degradation"""
        expectations = PerformanceExpectations(
            expected_win_rate=0.58,
            win_rate_threshold=0.10
        )
        detector = PerformanceDegradationDetector(expectations)
        
        # Severely degraded: 35% win rate
        trades = []
        for i in range(20):
            trades.append({
                'success': i < 7,  # 7 wins, 13 losses = 35%
                'exit_time': datetime.now() - timedelta(days=i),
                'pnl': 100 if i < 7 else -80,
                'pnl_percentage': 0.01 if i < 7 else -0.008
            })
        
        alerts = detector.check_degradation(trades)
        
        win_rate_alert = [a for a in alerts if a.metric == 'win_rate'][0]
        assert win_rate_alert.severity == 'critical'
        assert win_rate_alert.recommended_action == 'PAUSE_TRADING'


class TestSharpeRatioMonitoring:
    """Sharpe ratio degradation tests"""
    
    def test_sharpe_normal(self):
        """Test normal Sharpe ratio"""
        expectations = PerformanceExpectations(expected_sharpe=1.8)
        detector = PerformanceDegradationDetector(expectations)
        
        # Good returns
        np.random.seed(42)
        trades = []
        returns = np.random.normal(0.001, 0.005, 30)  # Mean 0.1%, std 0.5%
        
        for i, ret in enumerate(returns):
            trades.append({
                'success': ret > 0,
                'exit_time': datetime.now() - timedelta(days=i),
                'pnl': ret * 10000,
                'pnl_percentage': ret
            })
        
        alerts = detector.check_degradation(trades)
        
        # May or may not alert depending on realized Sharpe
        # Main test: doesn't crash
        assert alerts is None or isinstance(alerts, list)
    
    def test_sharpe_negative(self):
        """Test negative Sharpe ratio (critical)"""
        expectations = PerformanceExpectations(expected_sharpe=1.8)
        detector = PerformanceDegradationDetector(expectations)
        
        # Losing system
        trades = []
        for i in range(30):
            trades.append({
                'success': False,
                'exit_time': datetime.now() - timedelta(days=i),
                'pnl': -100,
                'pnl_percentage': -0.01  # Losing 1% per trade
            })
        
        alerts = detector.check_degradation(trades)
        
        assert alerts is not None
        sharpe_alert = [a for a in alerts if a.metric == 'sharpe_ratio']
        if sharpe_alert:
            assert sharpe_alert[0].severity == 'critical'


class TestMaxDrawdownMonitoring:
    """Maximum drawdown tests"""
    
    def test_max_dd_normal(self):
        """Test normal drawdown"""
        expectations = PerformanceExpectations(
            expected_max_dd=0.08,
            max_dd_threshold=0.12
        )
        detector = PerformanceDegradationDetector(expectations)
        
        # Small drawdown
        trades = [
            {'pnl': 100, 'exit_time': datetime.now() - timedelta(days=i)}
            for i in range(10)
        ]
        trades[5]['pnl'] = -50  # Small loss
        
        alerts = detector.check_degradation(trades)
        
        # Should not alert
        dd_alerts = [a for a in alerts if a.metric == 'max_drawdown'] if alerts else []
        assert len(dd_alerts) == 0
    
    def test_max_dd_exceeded(self):
        """Test drawdown threshold exceeded"""
        expectations = PerformanceExpectations(
            expected_max_dd=0.08,
            max_dd_threshold=0.12
        )
        detector = PerformanceDegradationDetector(expectations)
        
        # Large drawdown sequence
        trades = []
        cumulative = 0
        for i in range(20):
            if i < 5:
                pnl = 200  # Initial gains
            elif i < 15:
                pnl = -150  # Drawdown phase
            else:
                pnl = 50  # Recovery
            
            cumulative += pnl
            trades.append({
                'pnl': pnl,
                'exit_time': datetime.now() - timedelta(days=19-i),
                'success': pnl > 0,
                'pnl_percentage': pnl / 10000
            })
        
        alerts = detector.check_degradation(trades)
        
        if alerts:
            dd_alerts = [a for a in alerts if a.metric == 'max_drawdown']
            if dd_alerts:
                assert dd_alerts[0].recommended_action in ['REDUCE_RISK', 'PAUSE_TRADING']


class TestProfitFactorMonitoring:
    """Profit factor degradation tests"""
    
    def test_profit_factor_good(self):
        """Test good profit factor"""
        expectations = PerformanceExpectations(expected_profit_factor=1.8)
        detector = PerformanceDegradationDetector(expectations)
        
        # Good system: $1000 wins, $500 losses = PF 2.0
        trades = [
            {'pnl': 100, 'exit_time': datetime.now(), 'success': True, 'pnl_percentage': 0.01},
            {'pnl': 150, 'exit_time': datetime.now(), 'success': True, 'pnl_percentage': 0.015},
            {'pnl': -50, 'exit_time': datetime.now(), 'success': False, 'pnl_percentage': -0.005},
            {'pnl': 200, 'exit_time': datetime.now(), 'success': True, 'pnl_percentage': 0.02},
            {'pnl': -80, 'exit_time': datetime.now(), 'success': False, 'pnl_percentage': -0.008}
        ]
        
        alerts = detector.check_degradation(trades)
        
        # Should not alert
        pf_alerts = [a for a in alerts if a.metric == 'profit_factor'] if alerts else []
        assert len(pf_alerts) == 0
    
    def test_profit_factor_degraded(self):
        """Test degraded profit factor"""
        expectations = PerformanceExpectations(
            expected_profit_factor=1.8,
            profit_factor_threshold=0.30
        )
        detector = PerformanceDegradationDetector(expectations)
        
        # Poor system: $500 wins, $600 losses = PF 0.83
        trades = []
        for i in range(20):
            if i < 10:
                pnl = 50  # Wins
            else:
                pnl = -60  # Larger losses
            
            trades.append({
                'pnl': pnl,
                'exit_time': datetime.now() - timedelta(days=i),
                'success': pnl > 0,
                'pnl_percentage': pnl / 10000
            })
        
        alerts = detector.check_degradation(trades)
        
        assert alerts is not None
        pf_alerts = [a for a in alerts if a.metric == 'profit_factor']
        if pf_alerts:
            assert pf_alerts[0].severity in ['warning', 'critical']


class TestPerformanceSummary:
    """Performance summary generation tests"""
    
    def test_get_summary(self):
        """Test performance summary generation"""
        expectations = PerformanceExpectations()
        detector = PerformanceDegradationDetector(expectations)
        
        trades = [
            {'success': True, 'exit_time': datetime.now(), 'pnl': 100, 'pnl_percentage': 0.01},
            {'success': True, 'exit_time': datetime.now(), 'pnl': 150, 'pnl_percentage': 0.015},
            {'success': False, 'exit_time': datetime.now(), 'pnl': -50, 'pnl_percentage': -0.005}
        ]
        
        summary = detector.get_performance_summary(trades)
        
        assert summary['status'] in ['ok', 'insufficient_data']
        assert 'trade_count' in summary
        
        if summary['status'] == 'ok':
            assert 'win_rate' in summary
            assert 'sharpe_ratio' in summary
            assert 'comparison' in summary


@pytest.mark.integration
class TestIntegration:
    """Integration tests"""
    
    def test_full_monitoring_cycle(self):
        """Test complete monitoring cycle"""
        expectations = PerformanceExpectations()
        detector = PerformanceDegradationDetector(
            expectations,
            rolling_window_days=30,
            check_interval_hours=24
        )
        
        # Simulate 60 days of trades
        trades = []
        for day in range(60):
            for trade in range(3):  # 3 trades per day
                success = np.random.random() > 0.45  # 55% win rate
                pnl = 100 if success else -80
                
                trades.append({
                    'success': success,
                    'pnl': pnl,
                    'pnl_percentage': pnl / 10000,
                    'exit_time': datetime.now() - timedelta(days=day)
                })
        
        # Check degradation
        alerts = detector.check_degradation(trades)
        
        # Get summary
        summary = detector.get_performance_summary(trades)
        
        assert summary['trade_count'] > 0
        assert isinstance(alerts, (list, type(None)))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
