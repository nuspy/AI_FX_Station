"""
Advanced Position Sizing Strategy

Extends TradingStrategy with integration to PositionSizer for Kelly/Optimal f sizing.
Builds BacktestTradeHistory dynamically and recalculates optimal sizing as trades accumulate.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from forex_diffusion.backtesting.advanced_backtest_engine import TradingStrategy
from forex_diffusion.risk.position_sizer import PositionSizer, BacktestTradeHistory


class AdaptivePositionSizingStrategy(TradingStrategy):
    """
    Strategy that uses advanced position sizing (Kelly, Optimal f).

    Dynamically builds trade history and recalculates optimal sizing
    as the backtest progresses.
    """

    def __init__(
        self,
        base_strategy: TradingStrategy,
        position_sizing_method: str = 'kelly',
        kelly_fraction: float = 0.25,
        initial_history_trades: int = 30,
        recalculate_every_n_trades: int = 10,
    ):
        """
        Initialize adaptive position sizing strategy.

        Args:
            base_strategy: Underlying strategy for signal generation
            position_sizing_method: 'kelly', 'optimal_f', 'fixed_fractional', 'volatility_adjusted'
            kelly_fraction: Kelly fraction (0.25 = quarter Kelly)
            initial_history_trades: Min trades before enabling Kelly/Optimal f
            recalculate_every_n_trades: Recalculate sizing every N trades
        """
        self.base_strategy = base_strategy
        self.position_sizing_method = position_sizing_method
        self.kelly_fraction = kelly_fraction
        self.initial_history_trades = initial_history_trades
        self.recalculate_every_n_trades = recalculate_every_n_trades

        # Position sizer
        self.position_sizer = PositionSizer(
            base_risk_pct=1.0,  # Default fallback
            kelly_fraction=kelly_fraction,
            max_position_size_pct=5.0,
            min_position_size_pct=0.1,
            drawdown_reduction_enabled=True,
        )

        # Trade tracking
        self.completed_trades: List[Dict] = []
        self.trade_history: Optional[BacktestTradeHistory] = None

        logger.info(
            f"AdaptivePositionSizingStrategy initialized: "
            f"method={position_sizing_method}, kelly_fraction={kelly_fraction}"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Delegate to base strategy for signal generation."""
        return self.base_strategy.generate_signals(data)

    def get_position_size(
        self,
        data: pd.DataFrame,
        signal: int,
        current_capital: float,
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate position size using advanced sizing method.

        Args:
            data: Market data
            signal: Trading signal (1 long, -1 short, 0 hold)
            current_capital: Current account balance
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            atr: Average True Range

        Returns:
            Position size in units of base currency
        """
        if signal == 0:
            return 0.0

        # Get entry and stop prices
        if entry_price is None:
            entry_price = data['close'].iloc[-1]

        if stop_loss_price is None:
            # Use simple ATR-based stop if not provided
            if atr is None:
                atr = self._calculate_atr(data)
            stop_loss_price = (
                entry_price - (2.0 * atr) if signal > 0
                else entry_price + (2.0 * atr)
            )

        # Update trade history if enough trades completed
        if (
            len(self.completed_trades) >= self.initial_history_trades
            and len(self.completed_trades) % self.recalculate_every_n_trades == 0
        ):
            self._update_trade_history()

        # Calculate current drawdown
        current_drawdown_pct = self._calculate_current_drawdown(current_capital)

        # Calculate position size
        size, metadata = self.position_sizer.calculate_position_size(
            method=self.position_sizing_method,
            account_balance=current_capital,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            backtest_history=self.trade_history,
            atr=atr,
            current_drawdown_pct=current_drawdown_pct,
            existing_exposure_pct=0.0,  # Single position assumption
        )

        logger.debug(
            f"Position size: {size:.2f} units "
            f"({metadata['size_pct']:.2f}% of capital), "
            f"method={metadata['method']}"
        )

        return size

    def record_trade(self, trade_pnl: float, trade_size: float, entry_capital: float):
        """
        Record completed trade for history tracking.

        Args:
            trade_pnl: Trade profit/loss in base currency
            trade_size: Position size
            entry_capital: Capital at trade entry
        """
        # Calculate return as fraction of capital risked
        pnl_fraction = trade_pnl / entry_capital

        self.completed_trades.append({
            'pnl': trade_pnl,
            'pnl_fraction': pnl_fraction,
            'size': trade_size,
            'entry_capital': entry_capital,
        })

        logger.debug(f"Trade recorded: P&L={trade_pnl:.2f}, fraction={pnl_fraction:.4f}")

    def _update_trade_history(self):
        """Build BacktestTradeHistory from completed trades."""
        if not self.completed_trades:
            return

        # Separate wins and losses
        wins = []
        losses = []

        for trade in self.completed_trades:
            pnl_frac = trade['pnl_fraction']
            if pnl_frac > 0:
                wins.append(pnl_frac)
            elif pnl_frac < 0:
                losses.append(abs(pnl_frac))

        # Calculate statistics
        total_trades = len(self.completed_trades)
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        # Calculate max consecutive losses
        max_consecutive_losses = self._calculate_max_consecutive_losses()

        # Create history object
        self.trade_history = BacktestTradeHistory(
            wins=wins,
            losses=losses,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_losses=max_consecutive_losses,
        )

        logger.info(
            f"Trade history updated: {total_trades} trades, "
            f"win_rate={win_rate*100:.1f}%, "
            f"avg_win={avg_win*100:.2f}%, avg_loss={avg_loss*100:.2f}%"
        )

    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades."""
        if not self.completed_trades:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in self.completed_trades:
            if trade['pnl_fraction'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max(max_consecutive, 1)  # At least 1

    def _calculate_current_drawdown(self, current_capital: float) -> float:
        """Calculate current drawdown percentage."""
        if not self.completed_trades:
            return 0.0

        # Find peak capital
        peak_capital = max(
            trade['entry_capital'] for trade in self.completed_trades
        )
        peak_capital = max(peak_capital, current_capital)

        # Calculate drawdown
        drawdown_pct = ((peak_capital - current_capital) / peak_capital) * 100
        return max(0.0, drawdown_pct)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(data) < period:
            return data['close'].iloc[-1] * 0.01  # 1% fallback

        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return atr if not np.isnan(atr) else data['close'].iloc[-1] * 0.01

    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        return {
            'total_trades': len(self.completed_trades),
            'history_initialized': self.trade_history is not None,
            'current_method': self.position_sizing_method,
            'kelly_fraction': self.kelly_fraction,
            'trade_history': (
                {
                    'win_rate': self.trade_history.win_rate,
                    'avg_win': self.trade_history.avg_win,
                    'avg_loss': self.trade_history.avg_loss,
                    'max_consecutive_losses': self.trade_history.max_consecutive_losses,
                }
                if self.trade_history
                else None
            ),
        }
