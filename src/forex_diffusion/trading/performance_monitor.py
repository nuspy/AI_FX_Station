"""
Performance Degradation Detection for Live Trading

Monitors live trading performance and compares against backtest expectations.
Provides early warning when system performance degrades significantly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from loguru import logger


@dataclass
class PerformanceExpectations:
    """Expected performance metrics from backtest"""
    expected_win_rate: float = 0.58
    expected_sharpe: float = 1.8
    expected_max_dd: float = 0.08
    expected_profit_factor: float = 1.8
    
    # Alert thresholds (percentage degradation)
    win_rate_threshold: float = 0.10  # Alert if 10% drop
    sharpe_threshold: float = 0.30    # Alert if 30% drop
    max_dd_threshold: float = 0.12    # Alert if exceeds 12%
    profit_factor_threshold: float = 0.30  # Alert if 30% drop


@dataclass
class DegradationAlert:
    """Alert for performance degradation"""
    timestamp: datetime
    metric: str
    expected: float
    actual: float
    degradation_pct: float
    severity: str  # "warning", "critical"
    recommended_action: str


class PerformanceDegradationDetector:
    """
    Detects when live performance degrades vs backtest expectations.
    
    Features:
    - Rolling window metrics calculation
    - Comparison against backtest expectations
    - Multi-metric monitoring
    - Severity-based alerting
    - Automatic action recommendations
    """
    
    def __init__(
        self,
        expectations: PerformanceExpectations,
        rolling_window_days: int = 30,
        check_interval_hours: int = 24,
        min_trades_for_check: int = 10
    ):
        self.expectations = expectations
        self.rolling_window_days = rolling_window_days
        self.check_interval_hours = check_interval_hours
        self.min_trades_for_check = min_trades_for_check
        
        self.last_check_time: Optional[datetime] = None
        self.alert_history: List[DegradationAlert] = []
    
    def should_check(self) -> bool:
        """Check if enough time has passed for next check"""
        if self.last_check_time is None:
            return True
        
        hours_since_check = (datetime.now() - self.last_check_time).total_seconds() / 3600
        return hours_since_check >= self.check_interval_hours
    
    def check_degradation(self, trade_history: List[Dict]) -> Optional[List[DegradationAlert]]:
        """
        Check for performance degradation.
        
        Args:
            trade_history: List of completed trades with keys:
                - pnl: P&L for trade
                - pnl_percentage: Return percentage
                - exit_time: Trade exit datetime
                - success: Boolean success flag
                
        Returns:
            List of alerts if degradation detected, None otherwise
        """
        if not self.should_check():
            return None
        
        self.last_check_time = datetime.now()
        
        # Filter recent trades
        recent_trades = self._filter_recent_trades(
            trade_history,
            days=self.rolling_window_days
        )
        
        if len(recent_trades) < self.min_trades_for_check:
            logger.debug(
                f"Insufficient trades for degradation check: "
                f"{len(recent_trades)} < {self.min_trades_for_check}"
            )
            return None
        
        logger.info(
            f"Performance degradation check: {len(recent_trades)} trades "
            f"in last {self.rolling_window_days} days"
        )
        
        alerts = []
        
        # Check win rate
        alert = self._check_win_rate(recent_trades)
        if alert:
            alerts.append(alert)
        
        # Check Sharpe ratio
        alert = self._check_sharpe(recent_trades)
        if alert:
            alerts.append(alert)
        
        # Check max drawdown
        alert = self._check_max_drawdown(recent_trades)
        if alert:
            alerts.append(alert)
        
        # Check profit factor
        alert = self._check_profit_factor(recent_trades)
        if alert:
            alerts.append(alert)
        
        if alerts:
            logger.warning(f"Performance degradation detected: {len(alerts)} metric(s) degraded")
            self.alert_history.extend(alerts)
        else:
            logger.info("Performance check passed - no degradation detected")
        
        return alerts if alerts else None
    
    def _filter_recent_trades(
        self,
        trades: List[Dict],
        days: int
    ) -> List[Dict]:
        """Filter trades from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent = []
        for trade in trades:
            exit_time = trade.get('exit_time')
            if exit_time and exit_time >= cutoff:
                recent.append(trade)
        
        return recent
    
    def _check_win_rate(self, trades: List[Dict]) -> Optional[DegradationAlert]:
        """Check win rate degradation"""
        if not trades:
            return None
        
        # Calculate current win rate
        successful = sum(1 for t in trades if t.get('success', False))
        current_win_rate = successful / len(trades)
        
        # Check degradation
        degradation = self.expectations.expected_win_rate - current_win_rate
        
        if degradation > self.expectations.win_rate_threshold:
            severity = "critical" if degradation > self.expectations.win_rate_threshold * 1.5 else "warning"
            
            return DegradationAlert(
                timestamp=datetime.now(),
                metric="win_rate",
                expected=self.expectations.expected_win_rate,
                actual=current_win_rate,
                degradation_pct=degradation,
                severity=severity,
                recommended_action="PAUSE_TRADING" if severity == "critical" else "REVIEW_SYSTEM"
            )
        
        return None
    
    def _check_sharpe(self, trades: List[Dict]) -> Optional[DegradationAlert]:
        """Check Sharpe ratio degradation"""
        returns = [t.get('pnl_percentage', 0.0) for t in trades]
        
        if len(returns) < 2:
            return None
        
        # Calculate Sharpe (simplified, annualized)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return None
        
        # Annualize (assume daily frequency for simplicity)
        current_sharpe = (mean_return / std_return) * np.sqrt(252)
        
        # Check degradation (proportional)
        if current_sharpe <= 0:
            # Negative Sharpe is always critical
            return DegradationAlert(
                timestamp=datetime.now(),
                metric="sharpe_ratio",
                expected=self.expectations.expected_sharpe,
                actual=current_sharpe,
                degradation_pct=1.0,  # 100% degradation
                severity="critical",
                recommended_action="PAUSE_TRADING"
            )
        
        degradation = (self.expectations.expected_sharpe - current_sharpe) / self.expectations.expected_sharpe
        
        if degradation > self.expectations.sharpe_threshold:
            severity = "critical" if degradation > self.expectations.sharpe_threshold * 1.5 else "warning"
            
            return DegradationAlert(
                timestamp=datetime.now(),
                metric="sharpe_ratio",
                expected=self.expectations.expected_sharpe,
                actual=current_sharpe,
                degradation_pct=degradation,
                severity=severity,
                recommended_action="PAUSE_TRADING" if severity == "critical" else "REVIEW_SYSTEM"
            )
        
        return None
    
    def _check_max_drawdown(self, trades: List[Dict]) -> Optional[DegradationAlert]:
        """Check max drawdown"""
        if not trades:
            return None
        
        # Calculate equity curve
        cumulative_pnl = 0
        equity = [0]
        
        for trade in trades:
            cumulative_pnl += trade.get('pnl', 0.0)
            equity.append(cumulative_pnl)
        
        equity = np.array(equity)
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-12)
        current_max_dd = np.max(drawdown)
        
        # Check if exceeds threshold
        if current_max_dd > self.expectations.max_dd_threshold:
            severity = "critical" if current_max_dd > self.expectations.max_dd_threshold * 1.5 else "warning"
            
            return DegradationAlert(
                timestamp=datetime.now(),
                metric="max_drawdown",
                expected=self.expectations.expected_max_dd,
                actual=current_max_dd,
                degradation_pct=(current_max_dd - self.expectations.expected_max_dd),
                severity=severity,
                recommended_action="PAUSE_TRADING" if severity == "critical" else "REDUCE_RISK"
            )
        
        return None
    
    def _check_profit_factor(self, trades: List[Dict]) -> Optional[DegradationAlert]:
        """Check profit factor degradation"""
        if not trades:
            return None
        
        total_profit = sum(t.get('pnl', 0.0) for t in trades if t.get('pnl', 0.0) > 0)
        total_loss = abs(sum(t.get('pnl', 0.0) for t in trades if t.get('pnl', 0.0) < 0))
        
        if total_loss == 0:
            return None  # Can't calculate profit factor
        
        current_profit_factor = total_profit / total_loss
        
        # Check degradation
        degradation = (self.expectations.expected_profit_factor - current_profit_factor) / self.expectations.expected_profit_factor
        
        if degradation > self.expectations.profit_factor_threshold:
            severity = "critical" if degradation > self.expectations.profit_factor_threshold * 1.5 else "warning"
            
            return DegradationAlert(
                timestamp=datetime.now(),
                metric="profit_factor",
                expected=self.expectations.expected_profit_factor,
                actual=current_profit_factor,
                degradation_pct=degradation,
                severity=severity,
                recommended_action="REVIEW_SYSTEM" if severity == "warning" else "PAUSE_TRADING"
            )
        
        return None
    
    def get_performance_summary(self, trades: List[Dict]) -> Dict:
        """Get current performance summary"""
        recent_trades = self._filter_recent_trades(trades, self.rolling_window_days)
        
        if not recent_trades:
            return {
                'status': 'insufficient_data',
                'trade_count': 0
            }
        
        # Calculate metrics
        successful = sum(1 for t in recent_trades if t.get('success', False))
        win_rate = successful / len(recent_trades)
        
        returns = [t.get('pnl_percentage', 0.0) for t in recent_trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        total_profit = sum(t.get('pnl', 0.0) for t in recent_trades if t.get('pnl', 0.0) > 0)
        total_loss = abs(sum(t.get('pnl', 0.0) for t in recent_trades if t.get('pnl', 0.0) < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            'status': 'ok',
            'trade_count': len(recent_trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'avg_return': mean_return,
            'comparison': {
                'win_rate_vs_expected': win_rate - self.expectations.expected_win_rate,
                'sharpe_vs_expected': sharpe - self.expectations.expected_sharpe,
                'profit_factor_vs_expected': profit_factor - self.expectations.expected_profit_factor
            }
        }
