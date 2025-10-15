"""
Integrated Backtest System

Comprehensive backtesting integrating all advanced components:
- Multi-timeframe ensemble predictions
- Multi-model stacked ensemble
- Regime detection and adaptation
- Multi-level risk management
- Regime-aware position sizing
- Smart execution optimization
- Transaction cost modeling

This is a complete end-to-end backtest that validates the entire trading system.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import json

try:
    from ..models.multi_timeframe_ensemble import MultiTimeframeEnsemble, Timeframe
    from ..models.ml_stacked_ensemble import StackedMLEnsemble
    from ..regime.hmm_detector import HMMRegimeDetector
    from ..risk.multi_level_stop_loss import MultiLevelStopLoss, StopLossType
    from ..risk.regime_position_sizer import RegimePositionSizer, MarketRegime
    from ..execution.smart_execution import SmartExecutionOptimizer
except ImportError:
    pass


@dataclass
class Trade:
    """Individual trade record."""
    trade_id: int
    symbol: str
    direction: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    size: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Context
    entry_regime: Optional[str] = None
    exit_regime: Optional[str] = None
    entry_signal_confidence: float = 0.0

    # Execution costs
    entry_cost: float = 0.0
    exit_cost: float = 0.0
    slippage: float = 0.0

    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    # Exit details
    exit_reason: str = ""
    stop_type_triggered: Optional[str] = None

    # Analytics
    mae: float = 0.0  # Maximum adverse excursion
    mfe: float = 0.0  # Maximum favorable excursion
    holding_time_hours: float = 0.0


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    # Data
    symbol: str
    start_date: datetime
    end_date: datetime
    timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '1h'])

    # Models
    use_multi_timeframe: bool = True
    use_stacked_ensemble: bool = True
    use_regime_detection: bool = True
    use_smart_execution: bool = True

    # Capital management
    initial_capital: float = 10000.0
    max_positions: int = 3
    base_risk_per_trade_pct: float = 1.0

    # Risk management
    use_multi_level_stops: bool = True
    max_holding_hours: int = 24
    daily_loss_limit_pct: float = 3.0

    # Costs
    spread_pct: float = 0.0002  # 2 pips for EURUSD
    commission_pct: float = 0.0001  # 0.01%
    slippage_pct: float = 0.0001  # 0.01%

    # Execution
    min_signal_confidence: float = 0.6

    # Walk-forward
    train_size_days: int = 30
    test_size_days: int = 7
    step_size_days: int = 7


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    # Config
    config: BacktestConfig

    # Trades
    trades: List[Trade]
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Performance
    initial_capital: float = 10000.0
    final_capital: float = 10000.0
    total_return: float = 0.0
    total_return_pct: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0

    # Trade metrics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Time metrics
    avg_holding_time_hours: float = 0.0

    # Regime performance
    regime_performance: Dict[str, Dict] = field(default_factory=dict)

    # Cost analysis
    total_costs: float = 0.0
    avg_cost_per_trade: float = 0.0

    # Equity curve
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Monthly returns
    monthly_returns: pd.Series = field(default_factory=pd.Series)


class IntegratedBacktester:
    """
    Integrated backtesting system with all advanced components.

    Features:
    - Walk-forward validation
    - Multi-timeframe ensemble
    - Multi-model stacking
    - Regime detection
    - Multi-level risk management
    - Smart execution optimization
    - Comprehensive performance analysis

    Example:
        >>> config = BacktestConfig(
        ...     symbol='EURUSD',
        ...     start_date=datetime(2023, 1, 1),
        ...     end_date=datetime(2024, 1, 1),
        ...     initial_capital=10000
        ... )
        >>> backtester = IntegratedBacktester(config)
        >>> result = backtester.run(data, features, labels)
        >>> backtester.print_report(result)
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize integrated backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config

        # Components
        self.mtf_ensemble: Optional[MultiTimeframeEnsemble] = None
        self.ml_ensemble: Optional[StackedMLEnsemble] = None
        self.regime_detector: Optional[HMMRegimeDetector] = None
        self.risk_manager = MultiLevelStopLoss()
        self.position_sizer = RegimePositionSizer(
            base_risk_per_trade_pct=config.base_risk_per_trade_pct
        )
        self.execution_optimizer = SmartExecutionOptimizer()

        # State
        self.trades: List[Trade] = []
        self.open_positions: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.current_capital: float = config.initial_capital
        self.trade_counter: int = 0
        self.daily_pnl: float = 0.0
        self.last_date: Optional[datetime] = None

        logger.info("Integrated backtester initialized")

    def run(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        labels: pd.Series,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run comprehensive backtest.

        Args:
            data: OHLCV data
            features: Calculated features
            labels: Target labels
            verbose: Print progress

        Returns:
            BacktestResult with comprehensive metrics
        """
        if verbose:
            logger.info("=" * 80)
            logger.info("INTEGRATED BACKTEST STARTING")
            logger.info("=" * 80)
            logger.info(f"Symbol: {self.config.symbol}")
            logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
            logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
            logger.info("=" * 80)

        # Reset state
        self._reset_state()

        # Calculate walk-forward windows
        train_size = self.config.train_size_days * 288  # Assuming 5min data (288 candles/day)
        test_size = self.config.test_size_days * 288
        step_size = self.config.step_size_days * 288

        n_windows = (len(data) - train_size - test_size) // step_size + 1

        if verbose:
            logger.info(f"Walk-forward windows: {n_windows}")

        # Walk-forward loop
        for window_idx in range(n_windows):
            # Define window
            train_start = window_idx * step_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size

            if test_end > len(data):
                break

            if verbose:
                logger.info(f"\n[Window {window_idx + 1}/{n_windows}] "
                           f"Train: {train_start}-{train_end}, Test: {test_start}-{test_end}")

            # Split data
            X_train = features.iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]
            X_test = features.iloc[test_start:test_end]
            y_test = labels.iloc[test_start:test_end]
            data_test = data.iloc[test_start:test_end]

            # Train models
            if self.config.use_stacked_ensemble:
                self.ml_ensemble = StackedMLEnsemble(n_folds=5)
                self.ml_ensemble.fit(X_train, y_train, verbose=False)
                if verbose:
                    logger.info("  ✓ Trained stacked ensemble")

            # Train regime detector
            if self.config.use_regime_detection:
                self.regime_detector = HMMRegimeDetector(n_regimes=4)
                self.regime_detector.fit(data.iloc[train_start:train_end])
                if verbose:
                    logger.info("  ✓ Trained regime detector")

            # Run backtest on test window
            self._backtest_window(data_test, X_test, y_test, verbose=False)

        # Generate results
        result = self._generate_results()

        if verbose:
            self._print_summary(result)

        return result

    def _reset_state(self):
        """Reset backtester state."""
        self.trades = []
        self.open_positions = []
        self.equity_history = []
        self.current_capital = self.config.initial_capital
        self.trade_counter = 0
        self.daily_pnl = 0.0
        self.last_date = None

    def _backtest_window(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        labels: pd.Series,
        verbose: bool = False
    ):
        """
        Backtest a single window.

        Args:
            data: OHLCV data for window
            features: Features for window
            labels: Labels for window
            verbose: Print progress
        """
        for i in range(len(data)):
            current_time = data.index[i] if isinstance(data.index[i], datetime) else datetime.now()
            current_price = data.iloc[i]['close']

            # Reset daily P&L
            if self.last_date is None or current_time.date() != self.last_date.date():
                self.daily_pnl = 0.0
                self.last_date = current_time

            # Check daily loss limit
            if abs(self.daily_pnl) >= self.config.initial_capital * (self.config.daily_loss_limit_pct / 100):
                # Close all positions
                self._close_all_positions(current_time, current_price, "Daily loss limit")
                continue

            # Manage existing positions
            self._manage_positions(i, data, current_time, current_price)

            # Check for new entries
            if len(self.open_positions) < self.config.max_positions:
                self._check_entry(i, data, features, current_time, current_price)

            # Record equity
            self.equity_history.append((current_time, self.current_capital))

    def _manage_positions(
        self,
        bar_index: int,
        data: pd.DataFrame,
        current_time: datetime,
        current_price: float
    ):
        """
        Manage open positions (check stops, update trailing).

        Args:
            bar_index: Current bar index
            data: OHLCV data
            current_time: Current timestamp
            current_price: Current price
        """
        positions_to_close = []

        for position in self.open_positions:
            # Update MAE/MFE
            if position.direction == 'long':
                position.mfe = max(position.mfe, current_price - position.entry_price)
                position.mae = max(position.mae, position.entry_price - current_price)
            else:
                position.mfe = max(position.mfe, position.entry_price - current_price)
                position.mae = max(position.mae, current_price - position.entry_price)

            # Check multi-level stops
            if self.config.use_multi_level_stops:
                atr = self._calculate_atr(data.iloc[max(0, bar_index-14):bar_index+1])

                # Update trailing stops
                position_dict = self._trade_to_dict(position)
                updated = self.risk_manager.update_trailing_stops(position_dict, current_price)
                position.stop_loss = updated.get('stop_loss', position.stop_loss)

                # Check if stopped
                triggered, stop_type, reason = self.risk_manager.check_stop_triggered(
                    position_dict,
                    current_price,
                    atr
                )

                if triggered:
                    positions_to_close.append((position, reason, stop_type.value))
                    continue

            # Simple stop check (fallback)
            if position.stop_loss:
                if position.direction == 'long' and current_price <= position.stop_loss:
                    positions_to_close.append((position, "Stop loss", "simple_stop"))
                elif position.direction == 'short' and current_price >= position.stop_loss:
                    positions_to_close.append((position, "Stop loss", "simple_stop"))

            # Take profit check
            if position.take_profit:
                if position.direction == 'long' and current_price >= position.take_profit:
                    positions_to_close.append((position, "Take profit", "take_profit"))
                elif position.direction == 'short' and current_price <= position.take_profit:
                    positions_to_close.append((position, "Take profit", "take_profit"))

        # Close positions
        for position, reason, stop_type in positions_to_close:
            self._close_position(position, current_time, current_price, reason, stop_type)

    def _check_entry(
        self,
        bar_index: int,
        data: pd.DataFrame,
        features: pd.DataFrame,
        current_time: datetime,
        current_price: float
    ):
        """
        Check for new entry signals.

        Args:
            bar_index: Current bar index
            data: OHLCV data
            features: Features
            current_time: Current timestamp
            current_price: Current price
        """
        # Get prediction
        if self.ml_ensemble:
            X = features.iloc[bar_index:bar_index+1]
            prediction = self.ml_ensemble.predict(X)[0]
            probabilities = self.ml_ensemble.predict_proba(X)[0]
            confidence = np.max(probabilities)
        else:
            return

        # Check signal strength
        if prediction == 0 or confidence < self.config.min_signal_confidence:
            return

        # Detect regime
        regime = None
        if self.regime_detector and self.config.use_regime_detection:
            regime_data = data.iloc[max(0, bar_index-100):bar_index+1]
            if len(regime_data) >= 50:
                try:
                    regime_state = self.regime_detector.predict_current(regime_data)
                    regime = regime_state.regime.value if regime_state else None
                except:
                    pass

        # Calculate position size
        direction = 'long' if prediction > 0 else 'short'
        stop_distance = current_price * 0.02  # 2% initial stop
        stop_price = current_price - stop_distance if direction == 'long' else current_price + stop_distance

        position_size = self._calculate_position_size(
            current_price,
            stop_price,
            regime,
            confidence
        )

        # Check if we have enough capital
        required_capital = position_size * current_price
        if required_capital > self.current_capital * 0.9:  # Don't use more than 90%
            return

        # Calculate execution cost
        if self.config.use_smart_execution:
            exec_cost = self.execution_optimizer.estimate_execution_cost(
                order_size=position_size,
                current_price=current_price,
                direction='buy' if direction == 'long' else 'sell',
                current_spread=current_price * self.config.spread_pct
            )
            entry_cost = exec_cost.total_cost
            execution_price = exec_cost.execution_price
        else:
            entry_cost = position_size * current_price * (self.config.spread_pct + self.config.commission_pct)
            execution_price = current_price * (1 + self.config.spread_pct) if direction == 'long' else current_price * (1 - self.config.spread_pct)

        # Create trade
        take_profit = current_price + stop_distance * 2 if direction == 'long' else current_price - stop_distance * 2

        trade = Trade(
            trade_id=self.trade_counter,
            symbol=self.config.symbol,
            direction=direction,
            entry_time=current_time,
            entry_price=execution_price,
            size=position_size,
            stop_loss=stop_price,
            take_profit=take_profit,
            entry_regime=regime,
            entry_signal_confidence=confidence,
            entry_cost=entry_cost
        )

        self.open_positions.append(trade)
        self.current_capital -= entry_cost
        self.trade_counter += 1

        logger.debug(f"  → ENTRY {direction.upper()} @ {execution_price:.5f}, "
                    f"size={position_size:.2f}, regime={regime}, conf={confidence:.2%}")

    def _close_position(
        self,
        position: Trade,
        exit_time: datetime,
        exit_price: float,
        reason: str,
        stop_type: str
    ):
        """Close a position and record trade."""
        # Calculate execution cost
        if self.config.use_smart_execution:
            exec_cost = self.execution_optimizer.estimate_execution_cost(
                order_size=position.size,
                current_price=exit_price,
                direction='sell' if position.direction == 'long' else 'buy',
                current_spread=exit_price * self.config.spread_pct
            )
            exit_cost = exec_cost.total_cost
            final_exit_price = exec_cost.execution_price
        else:
            exit_cost = position.size * exit_price * (self.config.spread_pct + self.config.commission_pct)
            final_exit_price = exit_price * (1 - self.config.spread_pct) if position.direction == 'long' else exit_price * (1 + self.config.spread_pct)

        # Calculate P&L
        if position.direction == 'long':
            gross_pnl = (final_exit_price - position.entry_price) * position.size
        else:
            gross_pnl = (position.entry_price - final_exit_price) * position.size

        net_pnl = gross_pnl - position.entry_cost - exit_cost

        # Update position
        position.exit_time = exit_time
        position.exit_price = final_exit_price
        position.exit_cost = exit_cost
        position.gross_pnl = gross_pnl
        position.net_pnl = net_pnl
        position.exit_reason = reason
        position.stop_type_triggered = stop_type
        position.holding_time_hours = (exit_time - position.entry_time).total_seconds() / 3600

        # Update capital
        self.current_capital += gross_pnl - exit_cost
        self.daily_pnl += net_pnl

        # Record trade
        self.trades.append(position)
        self.open_positions.remove(position)

        logger.debug(f"  ← EXIT {position.direction.upper()} @ {final_exit_price:.5f}, "
                    f"P&L=${net_pnl:.2f}, reason={reason}")

    def _close_all_positions(self, current_time: datetime, current_price: float, reason: str):
        """Close all open positions."""
        for position in list(self.open_positions):
            self._close_position(position, current_time, current_price, reason, "forced_close")

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        regime: Optional[str],
        confidence: float
    ) -> float:
        """Calculate position size using regime-aware sizing."""
        # Map regime
        regime_map = {
            'trending_up': MarketRegime.TRENDING_UP,
            'trending_down': MarketRegime.TRENDING_DOWN,
            'ranging': MarketRegime.RANGING,
            'volatile': MarketRegime.VOLATILE
        }
        market_regime = regime_map.get(regime, MarketRegime.RANGING)

        # Calculate size
        sizing = self.position_sizer.calculate_position_size(
            account_balance=self.current_capital,
            entry_price=entry_price,
            stop_loss_price=stop_price,
            current_regime=market_regime,
            pattern_confidence=confidence
        )

        return sizing['position_size']

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        if len(data) < 2:
            return 0.001

        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return atr if not np.isnan(atr) else 0.001

    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade to dict for risk manager."""
        return {
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'stop_loss': trade.stop_loss,
            'entry_time': trade.entry_time,
            'highest_price': trade.entry_price + trade.mfe,
            'lowest_price': trade.entry_price - trade.mae
        }

    def _generate_results(self) -> BacktestResult:
        """Generate comprehensive backtest results."""
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.net_pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.net_pnl <= 0)

        if total_trades == 0:
            return BacktestResult(
                config=self.config,
                trades=self.trades,
                initial_capital=self.config.initial_capital,
                final_capital=self.current_capital
            )

        # Performance
        total_return = self.current_capital - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100

        # Trade metrics
        win_rate = winning_trades / total_trades
        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in self.trades if t.net_pnl <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Risk metrics
        returns = [t.net_pnl for t in self.trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            downside = [r for r in returns if r < 0]
            if downside:
                sortino_ratio = (np.mean(returns) / np.std(downside)) * np.sqrt(252)
            else:
                sortino_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Drawdown
        equity = [self.config.initial_capital] + [self.config.initial_capital + sum(returns[:i+1]) for i in range(len(returns))]
        running_max = np.maximum.accumulate(equity)
        drawdowns = running_max - equity
        max_drawdown = np.max(drawdowns)
        max_drawdown_pct = (max_drawdown / self.config.initial_capital) * 100 if self.config.initial_capital > 0 else 0

        calmar_ratio = (total_return_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0

        # Time metrics
        avg_holding_time = np.mean([t.holding_time_hours for t in self.trades])

        # Regime performance
        regime_perf = {}
        for regime in set(t.entry_regime for t in self.trades if t.entry_regime):
            regime_trades = [t for t in self.trades if t.entry_regime == regime]
            regime_pnls = [t.net_pnl for t in regime_trades]
            regime_perf[regime] = {
                'trades': len(regime_trades),
                'total_pnl': sum(regime_pnls),
                'avg_pnl': np.mean(regime_pnls),
                'win_rate': sum(1 for p in regime_pnls if p > 0) / len(regime_pnls)
            }

        # Cost analysis
        total_costs = sum(t.entry_cost + t.exit_cost for t in self.trades)
        avg_cost_per_trade = total_costs / total_trades

        # Equity curve
        equity_curve = pd.DataFrame(self.equity_history, columns=['timestamp', 'equity'])
        equity_curve.set_index('timestamp', inplace=True)

        # Monthly returns
        trades_df = pd.DataFrame([{
            'timestamp': t.exit_time,
            'pnl': t.net_pnl
        } for t in self.trades if t.exit_time])

        if not trades_df.empty:
            trades_df.set_index('timestamp', inplace=True)
            monthly_returns = trades_df.resample('M')['pnl'].sum()
        else:
            monthly_returns = pd.Series()

        return BacktestResult(
            config=self.config,
            trades=self.trades,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            initial_capital=self.config.initial_capital,
            final_capital=self.current_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_holding_time_hours=avg_holding_time,
            regime_performance=regime_perf,
            total_costs=total_costs,
            avg_cost_per_trade=avg_cost_per_trade,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns
        )

    def _print_summary(self, result: BacktestResult):
        """Print backtest summary."""
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)

        logger.info(f"\n💰 CAPITAL:")
        logger.info(f"  Initial: ${result.initial_capital:,.2f}")
        logger.info(f"  Final: ${result.final_capital:,.2f}")
        logger.info(f"  Return: ${result.total_return:,.2f} ({result.total_return_pct:+.2f}%)")

        logger.info(f"\n📊 TRADES:")
        logger.info(f"  Total: {result.total_trades}")
        logger.info(f"  Wins: {result.winning_trades} ({result.win_rate:.2%})")
        logger.info(f"  Losses: {result.losing_trades}")
        logger.info(f"  Avg holding time: {result.avg_holding_time_hours:.1f} hours")

        logger.info(f"\n📈 PERFORMANCE:")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"  Expectancy: ${result.expectancy:.2f}")
        logger.info(f"  Avg Win: ${result.avg_win:.2f}")
        logger.info(f"  Avg Loss: ${result.avg_loss:.2f}")

        logger.info(f"\n⚠️ RISK:")
        logger.info(f"  Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2f}%)")
        logger.info(f"  Calmar Ratio: {result.calmar_ratio:.2f}")

        logger.info(f"\n💵 COSTS:")
        logger.info(f"  Total: ${result.total_costs:,.2f}")
        logger.info(f"  Avg per trade: ${result.avg_cost_per_trade:.2f}")

        if result.regime_performance:
            logger.info(f"\n🔄 REGIME PERFORMANCE:")
            for regime, perf in result.regime_performance.items():
                logger.info(f"  {regime}:")
                logger.info(f"    Trades: {perf['trades']}")
                logger.info(f"    Win Rate: {perf['win_rate']:.2%}")
                logger.info(f"    Avg P&L: ${perf['avg_pnl']:.2f}")

        logger.info("=" * 80)

    def save_results(self, result: BacktestResult, output_path: Path):
        """
        Save backtest results to disk.

        Args:
            result: Backtest result
            output_path: Output directory
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary = {
            'config': {
                'symbol': result.config.symbol,
                'start_date': result.config.start_date.isoformat(),
                'end_date': result.config.end_date.isoformat(),
                'initial_capital': result.config.initial_capital
            },
            'performance': {
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_return': result.total_return,
                'total_return_pct': result.total_return_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'max_drawdown_pct': result.max_drawdown_pct,
                'profit_factor': result.profit_factor
            }
        }

        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save trades
        trades_df = pd.DataFrame([{
            'trade_id': t.trade_id,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'size': t.size,
            'gross_pnl': t.gross_pnl,
            'net_pnl': t.net_pnl,
            'entry_regime': t.entry_regime,
            'exit_reason': t.exit_reason
        } for t in result.trades])

        trades_df.to_csv(output_path / 'trades.csv', index=False)

        # Save equity curve
        result.equity_curve.to_csv(output_path / 'equity_curve.csv')

        logger.info(f"Results saved to {output_path}")
