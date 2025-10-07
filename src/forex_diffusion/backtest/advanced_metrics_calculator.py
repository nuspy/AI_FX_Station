"""
Advanced Metrics Calculator

Calculates extended performance metrics beyond basic backtest results:
- Sortino Ratio (downside deviation)
- Calmar Ratio (return / max drawdown)
- MAR Ratio (CAGR / max drawdown)
- Omega Ratio (probability-weighted gains/losses)
- Gain-to-Pain Ratio
- Ulcer Index (drawdown depth and duration)
- Recovery Time Analysis
- Return Distribution (skewness, kurtosis, VaR, CVaR)
- System Quality Number (SQN)
- K-Ratio

Integrates with backtest engine and stores results in advanced_metrics table.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from loguru import logger


@dataclass
class AdvancedMetrics:
    """Container for advanced performance metrics."""

    # Basic metrics (for reference)
    total_return_pct: float
    total_trades: int
    win_rate_pct: float
    sharpe_ratio: float

    # Advanced risk-adjusted
    sortino_ratio: float
    calmar_ratio: float
    mar_ratio: float
    omega_ratio: float
    gain_to_pain_ratio: float

    # Drawdown metrics
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    avg_drawdown_pct: float
    recovery_time_days: int
    ulcer_index: float

    # Return distribution
    return_skewness: float
    return_kurtosis: float
    var_95_pct: float  # Value at Risk
    cvar_95_pct: float  # Conditional VaR (Expected Shortfall)

    # Win/Loss analysis
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    win_loss_ratio: float
    profit_factor: float

    # Consistency
    win_streak_max: int
    loss_streak_max: int
    monthly_win_rate_pct: float
    expectancy_per_trade: float

    # System quality
    system_quality_number: float  # SQN
    k_ratio: float


class AdvancedMetricsCalculator:
    """
    Calculates advanced performance metrics from trade history.

    Usage:
        calculator = AdvancedMetricsCalculator()
        metrics = calculator.calculate(equity_curve, returns, trades)
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"AdvancedMetricsCalculator initialized (risk_free_rate={risk_free_rate})")

    def calculate(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
        trades: List[Dict[str, Any]],
        period_start: datetime,
        period_end: datetime,
    ) -> AdvancedMetrics:
        """
        Calculate all advanced metrics.

        Args:
            equity_curve: Time series of equity values
            returns: Time series of returns (% change)
            trades: List of trade dictionaries
            period_start: Start of measurement period
            period_end: End of measurement period

        Returns:
            AdvancedMetrics object
        """
        logger.info(f"Calculating advanced metrics for {len(trades)} trades")

        # Basic metrics
        total_return_pct = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100
        total_trades = len(trades)
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        win_rate_pct = (len(wins) / total_trades * 100) if total_trades > 0 else 0

        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe(returns)

        # Advanced risk-adjusted metrics
        sortino_ratio = self._calculate_sortino(returns)
        calmar_ratio = self._calculate_calmar(equity_curve, returns)
        mar_ratio = self._calculate_mar(equity_curve, returns, period_start, period_end)
        omega_ratio = self._calculate_omega(returns)
        gain_to_pain_ratio = self._calculate_gain_to_pain(returns)

        # Drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(equity_curve)

        # Return distribution
        distribution_metrics = self._calculate_distribution_metrics(returns)

        # Win/Loss analysis
        win_loss_metrics = self._calculate_win_loss_metrics(trades)

        # Consistency metrics
        consistency_metrics = self._calculate_consistency_metrics(trades, equity_curve)

        # System quality
        sqn = self._calculate_sqn(trades)
        k_ratio = self._calculate_k_ratio(equity_curve)

        return AdvancedMetrics(
            # Basic
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            win_rate_pct=win_rate_pct,
            sharpe_ratio=sharpe_ratio,
            # Advanced risk-adjusted
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            mar_ratio=mar_ratio,
            omega_ratio=omega_ratio,
            gain_to_pain_ratio=gain_to_pain_ratio,
            # Drawdown
            **drawdown_metrics,
            # Distribution
            **distribution_metrics,
            # Win/Loss
            **win_loss_metrics,
            # Consistency
            **consistency_metrics,
            # System quality
            system_quality_number=sqn,
            k_ratio=k_ratio,
        )

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe Ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
        return sharpe

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """
        Calculate Sortino Ratio.

        Like Sharpe but only considers downside deviation (negative returns).
        Better than Sharpe for asymmetric return distributions.
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
        return sortino

    def _calculate_calmar(self, equity_curve: pd.Series, returns: pd.Series) -> float:
        """
        Calculate Calmar Ratio.

        Calmar = Annual Return / Max Drawdown
        """
        if len(returns) == 0:
            return 0.0

        annual_return = returns.mean() * 252  # Annualized
        max_dd = self._get_max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        calmar = annual_return / abs(max_dd)
        return calmar

    def _calculate_mar(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """
        Calculate MAR Ratio (similar to Calmar).

        MAR = CAGR / Max Drawdown
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate CAGR
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0])
        years = (end_date - start_date).days / 365.25
        cagr = (total_return ** (1 / years)) - 1 if years > 0 else 0.0

        max_dd = self._get_max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        mar = cagr / abs(max_dd)
        return mar

    def _calculate_omega(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio.

        Omega = Sum of gains above threshold / Sum of losses below threshold
        """
        if len(returns) == 0:
            return 0.0

        gains = returns[returns > threshold]
        losses = returns[returns < threshold]

        sum_gains = gains.sum()
        sum_losses = abs(losses.sum())

        if sum_losses == 0:
            return float('inf') if sum_gains > 0 else 0.0

        omega = sum_gains / sum_losses
        return omega

    def _calculate_gain_to_pain(self, returns: pd.Series) -> float:
        """
        Calculate Gain-to-Pain Ratio.

        Gain-to-Pain = Sum of all returns / Sum of absolute negative returns
        """
        if len(returns) == 0:
            return 0.0

        total_return = returns.sum()
        negative_returns = returns[returns < 0]
        pain = abs(negative_returns.sum())

        if pain == 0:
            return float('inf') if total_return > 0 else 0.0

        return total_return / pain

    def _calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive drawdown metrics."""
        if len(equity_curve) < 2:
            return {
                'max_drawdown_pct': 0.0,
                'max_drawdown_duration_days': 0,
                'avg_drawdown_pct': 0.0,
                'recovery_time_days': 0,
                'ulcer_index': 0.0,
            }

        # Calculate drawdown series
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100

        # Max drawdown
        max_dd = drawdown.min()

        # Drawdown duration
        dd_duration = self._calculate_dd_duration(drawdown)

        # Average drawdown
        negative_dd = drawdown[drawdown < 0]
        avg_dd = negative_dd.mean() if len(negative_dd) > 0 else 0.0

        # Recovery time (average time to recover from drawdown)
        recovery_time = self._calculate_recovery_time(equity_curve)

        # Ulcer Index (considers depth AND duration of drawdowns)
        ulcer_index = self._calculate_ulcer_index(drawdown)

        return {
            'max_drawdown_pct': max_dd,
            'max_drawdown_duration_days': dd_duration,
            'avg_drawdown_pct': avg_dd,
            'recovery_time_days': recovery_time,
            'ulcer_index': ulcer_index,
        }

    def _get_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Get maximum drawdown as decimal."""
        if len(equity_curve) < 2:
            return 0.0

        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()

    def _calculate_dd_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        in_drawdown = drawdown < 0
        dd_periods = []
        current_period = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    dd_periods.append(current_period)
                current_period = 0

        if current_period > 0:
            dd_periods.append(current_period)

        return max(dd_periods) if dd_periods else 0

    def _calculate_recovery_time(self, equity_curve: pd.Series) -> int:
        """Calculate average recovery time from drawdowns."""
        running_max = equity_curve.cummax()
        drawdown = equity_curve < running_max

        recovery_times = []
        in_dd = False
        dd_start = 0

        for i, is_dd in enumerate(drawdown):
            if is_dd and not in_dd:
                in_dd = True
                dd_start = i
            elif not is_dd and in_dd:
                recovery_times.append(i - dd_start)
                in_dd = False

        return int(np.mean(recovery_times)) if recovery_times else 0

    def _calculate_ulcer_index(self, drawdown: pd.Series) -> float:
        """
        Calculate Ulcer Index.

        Measures both depth and duration of drawdowns.
        UI = sqrt(mean(drawdown^2))
        """
        if len(drawdown) == 0:
            return 0.0

        ulcer = np.sqrt((drawdown ** 2).mean())
        return ulcer

    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate return distribution metrics."""
        if len(returns) < 2:
            return {
                'return_skewness': 0.0,
                'return_kurtosis': 0.0,
                'var_95_pct': 0.0,
                'cvar_95_pct': 0.0,
            }

        # Skewness (asymmetry)
        skew = stats.skew(returns.dropna())

        # Kurtosis (tail risk - excess kurtosis, where 0 = normal distribution)
        kurt = stats.kurtosis(returns.dropna())

        # Value at Risk (VaR) at 95% confidence
        var_95 = np.percentile(returns.dropna(), 5)  # 5th percentile

        # Conditional VaR (CVaR / Expected Shortfall)
        # Average of returns worse than VaR
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0.0

        return {
            'return_skewness': float(skew),
            'return_kurtosis': float(kurt),
            'var_95_pct': float(var_95),
            'cvar_95_pct': float(cvar_95),
        }

    def _calculate_win_loss_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate win/loss analysis metrics."""
        if not trades:
            return {
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
                'largest_win_pct': 0.0,
                'largest_loss_pct': 0.0,
                'win_loss_ratio': 0.0,
                'profit_factor': 0.0,
            }

        wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
        losses = [abs(t['pnl']) for t in trades if t.get('pnl', 0) < 0]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = max(losses) if losses else 0.0

        # Win/Loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Profit factor
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'largest_win_pct': largest_win,
            'largest_loss_pct': largest_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
        }

    def _calculate_consistency_metrics(
        self,
        trades: List[Dict],
        equity_curve: pd.Series
    ) -> Dict[str, Any]:
        """Calculate consistency metrics."""
        if not trades:
            return {
                'win_streak_max': 0,
                'loss_streak_max': 0,
                'monthly_win_rate_pct': 0.0,
                'expectancy_per_trade': 0.0,
            }

        # Win/loss streaks
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for trade in trades:
            if trade.get('pnl', 0) > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)

        # Monthly win rate (simplified - assumes daily equity curve)
        # Group by month and check if month was profitable
        if len(equity_curve) > 30:
            monthly_returns = equity_curve.resample('M').last().pct_change()
            monthly_wins = (monthly_returns > 0).sum()
            monthly_total = len(monthly_returns.dropna())
            monthly_win_rate = (monthly_wins / monthly_total * 100) if monthly_total > 0 else 0.0
        else:
            monthly_win_rate = 0.0

        # Expectancy per trade
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        expectancy = total_pnl / len(trades) if trades else 0.0

        return {
            'win_streak_max': max_win_streak,
            'loss_streak_max': max_loss_streak,
            'monthly_win_rate_pct': monthly_win_rate,
            'expectancy_per_trade': expectancy,
        }

    def _calculate_sqn(self, trades: List[Dict]) -> float:
        """
        Calculate System Quality Number (SQN).

        SQN = sqrt(N) * (mean(R) / stdev(R))
        Where R is the R-multiple (profit/risk) for each trade

        SQN > 2.5 = good system
        SQN > 3.0 = excellent system
        """
        if not trades or len(trades) < 2:
            return 0.0

        # Calculate R-multiples (simplified - use PnL as proxy)
        r_multiples = [t.get('pnl', 0) for t in trades]

        mean_r = np.mean(r_multiples)
        std_r = np.std(r_multiples)

        if std_r == 0:
            return 0.0

        sqn = np.sqrt(len(trades)) * (mean_r / std_r)
        return sqn

    def _calculate_k_ratio(self, equity_curve: pd.Series) -> float:
        """
        Calculate K-Ratio.

        Measures consistency of equity curve growth.
        K-Ratio = Slope / (StdError * sqrt(N))

        Higher = more consistent growth
        """
        if len(equity_curve) < 3:
            return 0.0

        # Log equity for linear regression
        log_equity = np.log(equity_curve)
        x = np.arange(len(log_equity))

        # Linear regression
        slope, intercept = np.polyfit(x, log_equity, 1)

        # Calculate residuals and standard error
        predicted = slope * x + intercept
        residuals = log_equity - predicted
        std_error = np.std(residuals)

        if std_error == 0:
            return 0.0

        k_ratio = slope / (std_error / np.sqrt(len(equity_curve)))
        return k_ratio
