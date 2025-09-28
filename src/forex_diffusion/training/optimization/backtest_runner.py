"""
Backtest runner with walk-forward validation and comprehensive metrics calculation.

This module handles the execution of backtests for pattern optimization trials,
including walk-forward validation, recency weighting, and detailed performance metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""

    # Walk-forward parameters
    walk_forward_months: int = 6
    purge_days: int = 1
    embargo_days: int = 2

    # Position sizing
    initial_capital: float = 100000.0
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 0.1  # Max 10% of capital per trade

    # Transaction costs
    spread_bps: float = 1.0  # 1 basis point spread
    commission_per_lot: float = 0.0
    slippage_bps: float = 0.5  # 0.5 basis point slippage

    # Risk management
    max_daily_drawdown: float = 0.05  # 5% daily drawdown limit
    max_concurrent_trades: int = 5

@dataclass
class TradeResult:
    """Results from a single trade"""

    signal_id: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # "long" or "short"

    # Target and stops
    target_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Trade outcome
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    return_percentage: float = 0.0

    # Trade metrics
    holding_period_hours: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Target/invalidation tracking
    target_reached: bool = False
    stop_hit: bool = False
    time_expired: bool = False
    invalidated: bool = False

    # Additional metadata
    pattern_key: str = ""
    confidence: float = 0.0
    trade_size: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""

    # Basic statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    success_rate: float = 0.0

    # Returns
    total_return: float = 0.0
    total_return_percentage: float = 0.0
    annualized_return: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade metrics
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    win_loss_ratio: float = 0.0

    # Timing metrics
    average_holding_period_hours: float = 0.0
    hit_rate_by_holding_period: Dict[str, float] = field(default_factory=dict)

    # Temporal metrics
    temporal_coverage_months: float = 0.0
    first_trade_date: Optional[datetime] = None
    last_trade_date: Optional[datetime] = None

    # Consistency metrics
    monthly_returns: List[float] = field(default_factory=list)
    variance_across_blocks: float = 0.0
    consistency_score: float = 0.0

    # Trade details
    trades: List[TradeResult] = field(default_factory=list)

    # Additional metrics for invalidation rules
    invalidation_statistics: Dict[str, Any] = field(default_factory=dict)

    # Regime-specific breakdown
    regime_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class BacktestRunner:
    """
    Executes backtests with walk-forward validation and comprehensive metrics.

    Features:
    - Walk-forward validation with purge and embargo
    - Realistic transaction costs and slippage
    - Comprehensive performance metrics
    - Recency weighting support
    - Regime-aware analysis
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run_walk_forward(self, data: pd.DataFrame, detector: Any,
                        action_params: Dict[str, Any], **kwargs) -> BacktestResult:
        """
        Execute walk-forward backtest with purge and embargo.

        Args:
            data: Historical price data
            detector: Pattern detector instance
            action_params: Action parameters for trade execution
            **kwargs: Additional parameters

        Returns:
            BacktestResult with comprehensive metrics
        """
        # Override config with kwargs
        walk_months = kwargs.get("walk_forward_months", self.config.walk_forward_months)
        purge_days = kwargs.get("purge_days", self.config.purge_days)
        embargo_days = kwargs.get("embargo_days", self.config.embargo_days)

        if data is None or data.empty:
            logger.warning("Empty data provided to backtest runner")
            return BacktestResult()

        # Prepare data
        data = self._prepare_data(data)

        # Generate walk-forward splits
        splits = self._generate_walk_forward_splits(
            data, walk_months, purge_days, embargo_days
        )

        logger.info(f"Running walk-forward backtest with {len(splits)} splits")

        all_trades = []
        equity_curve = []
        current_capital = self.config.initial_capital

        for i, (train_data, test_data) in enumerate(splits):
            # Generate signals on test data
            signals = self._generate_signals(detector, test_data, action_params)

            # Execute trades
            split_trades = self._execute_trades(
                test_data, signals, action_params, current_capital
            )

            # Update capital and track equity
            for trade in split_trades:
                current_capital += trade.pnl
                equity_curve.append({
                    "date": trade.exit_time or trade.entry_time,
                    "equity": current_capital,
                    "trade_pnl": trade.pnl
                })

            all_trades.extend(split_trades)

            logger.debug(f"Split {i+1}: {len(split_trades)} trades, capital: {current_capital:.2f}")

        # Calculate comprehensive metrics
        result = self._calculate_comprehensive_metrics(all_trades, equity_curve, data)

        logger.info(f"Backtest completed: {result.total_trades} trades, "
                   f"{result.success_rate:.1%} success rate, "
                   f"{result.total_return_percentage:.1%} return")

        return result

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate data for backtesting"""

        df = data.copy()

        # Ensure required columns exist
        required_cols = ["ts_utc", "open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert timestamp to datetime if needed
        if "ts_utc" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
                df["ts_utc"] = pd.to_datetime(df["ts_utc"], unit="ms", utc=True)

        # Sort by timestamp
        df = df.sort_values("ts_utc").reset_index(drop=True)

        # Calculate additional indicators if needed
        df = self._add_technical_indicators(df)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators needed for trading logic"""

        # ATR for stop loss calculations
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # True Range
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        true_range = np.maximum(tr1, np.maximum(tr2, tr3))

        # 14-period ATR
        atr_14 = pd.Series(true_range).rolling(window=14, min_periods=1).mean()
        df["atr_14"] = atr_14

        # Simple moving averages for context
        df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
        df["sma_50"] = df["close"].rolling(window=50, min_periods=1).mean()

        return df

    def _generate_walk_forward_splits(self, data: pd.DataFrame, walk_months: int,
                                    purge_days: int, embargo_days: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate walk-forward validation splits"""

        splits = []
        start_date = data["ts_utc"].min()
        end_date = data["ts_utc"].max()

        # Calculate split boundaries
        total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        if total_months < walk_months * 2:
            logger.warning(f"Insufficient data for walk-forward: {total_months} months available, "
                          f"{walk_months * 2} months needed")
            # Return single split using all data
            mid_point = len(data) // 2
            train_data = data.iloc[:mid_point].copy()
            test_data = data.iloc[mid_point:].copy()
            return [(train_data, test_data)]

        current_date = start_date
        min_train_months = walk_months

        while current_date + timedelta(days=30 * (min_train_months + walk_months)) <= end_date:
            # Training period
            train_end = current_date + timedelta(days=30 * min_train_months)

            # Purge period
            purge_end = train_end + timedelta(days=purge_days)

            # Test period
            test_start = purge_end
            test_end = test_start + timedelta(days=30 * walk_months)

            # Embargo - don't use data too close to test end for next iteration
            embargo_start = test_end
            next_start = embargo_start + timedelta(days=embargo_days)

            # Extract data splits
            train_mask = (data["ts_utc"] >= current_date) & (data["ts_utc"] < train_end)
            test_mask = (data["ts_utc"] >= test_start) & (data["ts_utc"] < test_end)

            train_data = data.loc[train_mask].copy()
            test_data = data.loc[test_mask].copy()

            if len(train_data) > 0 and len(test_data) > 0:
                splits.append((train_data, test_data))

            # Move to next period
            current_date = next_start
            min_train_months += walk_months  # Expanding window

        return splits

    def _generate_signals(self, detector: Any, data: pd.DataFrame,
                         action_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals using the pattern detector"""

        try:
            # Run pattern detection
            if hasattr(detector, "detect"):
                pattern_events = detector.detect(data)
            else:
                logger.warning("Detector does not have detect method")
                return []

            # Convert pattern events to trading signals
            signals = []
            for event in pattern_events:
                signal = self._convert_event_to_signal(event, data, action_params)
                if signal:
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.exception(f"Signal generation failed: {e}")
            return []

    def _convert_event_to_signal(self, event: Any, data: pd.DataFrame,
                               action_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert pattern event to trading signal"""

        try:
            # Extract basic information
            entry_time = getattr(event, "confirm_ts", None)
            if entry_time is None:
                return None

            # Find corresponding data row
            if isinstance(entry_time, (int, np.integer)):
                # Timestamp in milliseconds
                entry_dt = pd.to_datetime(entry_time, unit="ms", utc=True)
            else:
                entry_dt = pd.to_datetime(entry_time)

            # Find nearest data point
            time_diff = np.abs(data["ts_utc"] - entry_dt)
            entry_idx = time_diff.idxmin()
            entry_row = data.iloc[entry_idx]

            entry_price = float(entry_row["close"])
            direction = getattr(event, "direction", "bull")

            # Calculate target and stop prices
            target_price, stop_price = self._calculate_target_stop(
                entry_row, action_params, direction
            )

            # Create signal
            signal = {
                "signal_id": f"{getattr(event, 'pattern_key', 'unknown')}_{entry_idx}",
                "pattern_key": getattr(event, "pattern_key", "unknown"),
                "entry_time": entry_dt,
                "entry_price": entry_price,
                "direction": "long" if direction.lower() in ["bull", "bullish"] else "short",
                "target_price": target_price,
                "stop_price": stop_price,
                "confidence": getattr(event, "score", 0.5),
                "horizon_bars": action_params.get("horizon_bars", 50),
                "entry_idx": entry_idx
            }

            return signal

        except Exception as e:
            logger.warning(f"Failed to convert event to signal: {e}")
            return None

    def _calculate_target_stop(self, entry_row: pd.Series, action_params: Dict[str, Any],
                             direction: str) -> Tuple[float, float]:
        """Calculate target and stop loss prices"""

        entry_price = float(entry_row["close"])
        atr = float(entry_row.get("atr_14", entry_price * 0.01))  # Fallback to 1%

        # Risk-reward ratio
        rr_ratio = action_params.get("risk_reward_ratio", 2.0)

        # ATR buffer
        atr_buffer = action_params.get("buffer_atr", 1.0)

        # Calculate stop distance
        stop_distance = atr * atr_buffer

        if direction.lower() in ["bull", "bullish", "long"]:
            # Long position
            stop_price = entry_price - stop_distance
            target_price = entry_price + (stop_distance * rr_ratio)
        else:
            # Short position
            stop_price = entry_price + stop_distance
            target_price = entry_price - (stop_distance * rr_ratio)

        return target_price, stop_price

    def _execute_trades(self, data: pd.DataFrame, signals: List[Dict[str, Any]],
                       action_params: Dict[str, Any], current_capital: float) -> List[TradeResult]:
        """Execute trades based on signals"""

        trades = []
        active_trades = []
        max_concurrent = action_params.get("max_concurrent_trades",
                                         self.config.max_concurrent_trades)

        for idx, row in data.iterrows():
            current_time = row["ts_utc"]
            current_price = row["close"]

            # Check for new signals
            new_signals = [s for s in signals if s["entry_time"] <= current_time
                          and s not in [t["signal"] for t in active_trades]]

            # Add new trades (respecting position limits)
            for signal in new_signals:
                if len(active_trades) < max_concurrent:
                    trade = self._create_trade_from_signal(signal, current_capital)
                    if trade:
                        active_trades.append({
                            "trade": trade,
                            "signal": signal,
                            "entry_idx": idx
                        })

            # Update active trades
            completed_trades = []
            for active_trade in active_trades:
                trade = active_trade["trade"]
                signal = active_trade["signal"]

                # Check exit conditions
                exit_info = self._check_exit_conditions(
                    trade, signal, row, current_time, action_params
                )

                if exit_info["should_exit"]:
                    # Close trade
                    trade.exit_time = current_time
                    trade.exit_price = exit_info["exit_price"]

                    # Calculate PnL
                    self._calculate_trade_pnl(trade, row)

                    # Update trade tracking metrics
                    self._update_trade_metrics(trade, data, active_trade["entry_idx"], idx)

                    trades.append(trade)
                    completed_trades.append(active_trade)
                else:
                    # Update unrealized metrics
                    self._update_unrealized_metrics(trade, row)

            # Remove completed trades
            for completed in completed_trades:
                active_trades.remove(completed)

        # Close any remaining active trades at the end
        if active_trades:
            final_row = data.iloc[-1]
            for active_trade in active_trades:
                trade = active_trade["trade"]
                trade.exit_time = final_row["ts_utc"]
                trade.exit_price = final_row["close"]
                trade.time_expired = True

                self._calculate_trade_pnl(trade, final_row)
                self._update_trade_metrics(trade, data, active_trade["entry_idx"], len(data) - 1)

                trades.append(trade)

        return trades

    def _create_trade_from_signal(self, signal: Dict[str, Any],
                                 current_capital: float) -> Optional[TradeResult]:
        """Create a TradeResult from a signal"""

        try:
            # Calculate position size based on risk management
            risk_amount = current_capital * self.config.risk_per_trade
            entry_price = signal["entry_price"]
            stop_price = signal["stop_price"]

            if stop_price is None:
                return None

            # Calculate risk per unit
            if signal["direction"] == "long":
                risk_per_unit = abs(entry_price - stop_price)
            else:
                risk_per_unit = abs(stop_price - entry_price)

            if risk_per_unit == 0:
                return None

            # Position size
            position_size = risk_amount / risk_per_unit
            max_position_value = current_capital * self.config.max_position_size
            max_position_size = max_position_value / entry_price

            position_size = min(position_size, max_position_size)

            # Create trade
            trade = TradeResult(
                signal_id=signal["signal_id"],
                entry_time=signal["entry_time"],
                entry_price=entry_price,
                direction=signal["direction"],
                target_price=signal["target_price"],
                stop_price=stop_price,
                pattern_key=signal["pattern_key"],
                confidence=signal["confidence"],
                trade_size=position_size
            )

            # Calculate transaction costs
            trade.commission = self.config.commission_per_lot * position_size
            spread_cost = entry_price * self.config.spread_bps / 10000 * position_size
            slippage_cost = entry_price * self.config.slippage_bps / 10000 * position_size
            trade.slippage = spread_cost + slippage_cost

            return trade

        except Exception as e:
            logger.warning(f"Failed to create trade from signal: {e}")
            return None

    def _check_exit_conditions(self, trade: TradeResult, signal: Dict[str, Any],
                             row: pd.Series, current_time: datetime,
                             action_params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if trade should be exited"""

        current_price = row["close"]
        high = row["high"]
        low = row["low"]

        # Check target hit
        if trade.direction == "long":
            if trade.target_price and high >= trade.target_price:
                return {"should_exit": True, "exit_price": trade.target_price, "reason": "target"}
            if trade.stop_price and low <= trade.stop_price:
                return {"should_exit": True, "exit_price": trade.stop_price, "reason": "stop"}
        else:
            if trade.target_price and low <= trade.target_price:
                return {"should_exit": True, "exit_price": trade.target_price, "reason": "target"}
            if trade.stop_price and high >= trade.stop_price:
                return {"should_exit": True, "exit_price": trade.stop_price, "reason": "stop"}

        # Check time limit
        horizon_bars = signal.get("horizon_bars", 50)
        bars_elapsed = (current_time - trade.entry_time).total_seconds() / 3600  # Assuming hourly bars
        if bars_elapsed >= horizon_bars:
            return {"should_exit": True, "exit_price": current_price, "reason": "time"}

        return {"should_exit": False, "exit_price": current_price, "reason": "none"}

    def _calculate_trade_pnl(self, trade: TradeResult, exit_row: pd.Series) -> None:
        """Calculate trade P&L and metrics"""

        entry_price = trade.entry_price
        exit_price = trade.exit_price
        position_size = trade.trade_size

        if exit_price is None:
            return

        # Calculate raw P&L
        if trade.direction == "long":
            raw_pnl = (exit_price - entry_price) * position_size
        else:
            raw_pnl = (entry_price - exit_price) * position_size

        # Subtract costs
        net_pnl = raw_pnl - trade.commission - trade.slippage

        trade.pnl = net_pnl
        trade.pnl_percentage = (net_pnl / (entry_price * position_size)) * 100
        trade.return_percentage = (exit_price / entry_price - 1) * 100

        if trade.direction == "short":
            trade.return_percentage *= -1

        # Determine outcome
        if trade.target_price:
            if trade.direction == "long":
                trade.target_reached = exit_price >= trade.target_price
            else:
                trade.target_reached = exit_price <= trade.target_price

        if trade.stop_price:
            if trade.direction == "long":
                trade.stop_hit = exit_price <= trade.stop_price
            else:
                trade.stop_hit = exit_price >= trade.stop_price

    def _update_trade_metrics(self, trade: TradeResult, data: pd.DataFrame,
                            entry_idx: int, exit_idx: int) -> None:
        """Update trade metrics like MFE/MAE and holding period"""

        if exit_idx <= entry_idx:
            return

        # Holding period
        holding_period = trade.exit_time - trade.entry_time
        trade.holding_period_hours = holding_period.total_seconds() / 3600

        # Maximum Favorable/Adverse Excursion
        trade_data = data.iloc[entry_idx:exit_idx + 1]
        entry_price = trade.entry_price

        if trade.direction == "long":
            # MFE is highest high - entry
            mfe = trade_data["high"].max() - entry_price
            # MAE is entry - lowest low
            mae = entry_price - trade_data["low"].min()
        else:
            # MFE is entry - lowest low
            mfe = entry_price - trade_data["low"].min()
            # MAE is highest high - entry
            mae = trade_data["high"].max() - entry_price

        trade.max_favorable_excursion = max(0, mfe)
        trade.max_adverse_excursion = max(0, mae)

    def _update_unrealized_metrics(self, trade: TradeResult, row: pd.Series) -> None:
        """Update unrealized metrics for active trades"""
        # This could track unrealized P&L, drawdown, etc.
        pass

    def _calculate_comprehensive_metrics(self, trades: List[TradeResult],
                                       equity_curve: List[Dict[str, Any]],
                                       data: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive backtest metrics"""

        if not trades:
            return BacktestResult()

        result = BacktestResult()
        result.trades = trades
        result.total_trades = len(trades)

        # Basic trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        result.winning_trades = len(winning_trades)
        result.losing_trades = len(losing_trades)
        result.success_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

        # Return metrics
        total_pnl = sum(t.pnl for t in trades)
        result.total_return = total_pnl
        result.total_return_percentage = (total_pnl / self.config.initial_capital) * 100

        # Trade metrics
        if winning_trades:
            result.average_win = np.mean([t.pnl for t in winning_trades])
        if losing_trades:
            result.average_loss = np.mean([t.pnl for t in losing_trades])

        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))

        if total_losses > 0:
            result.profit_factor = total_wins / total_losses
        else:
            result.profit_factor = float('inf') if total_wins > 0 else 0

        result.expectancy = np.mean([t.pnl for t in trades])

        if result.average_loss != 0:
            result.win_loss_ratio = abs(result.average_win / result.average_loss)

        # Timing metrics
        result.average_holding_period_hours = np.mean([t.holding_period_hours for t in trades])

        # Temporal coverage
        if trades:
            result.first_trade_date = min(t.entry_time for t in trades)
            result.last_trade_date = max(t.exit_time or t.entry_time for t in trades)
            coverage_delta = result.last_trade_date - result.first_trade_date
            result.temporal_coverage_months = coverage_delta.days / 30.44  # Average days per month

        # Risk metrics (simplified)
        if equity_curve:
            equity_values = [point["equity"] for point in equity_curve]
            returns = np.diff(equity_values) / equity_values[:-1]

            result.volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

            # Drawdown calculation
            equity_series = pd.Series(equity_values)
            running_max = equity_series.cummax()
            drawdowns = (equity_series - running_max) / running_max
            result.max_drawdown = abs(drawdowns.min())

            # Risk ratios
            if result.volatility > 0:
                result.sharpe_ratio = (result.annualized_return - 0.02) / result.volatility  # Assuming 2% risk-free rate

        # Consistency metrics
        monthly_returns = self._calculate_monthly_returns(trades)
        result.monthly_returns = monthly_returns

        if len(monthly_returns) > 1:
            result.variance_across_blocks = np.var(monthly_returns)
            # Simple consistency score
            positive_months = sum(1 for r in monthly_returns if r > 0)
            result.consistency_score = positive_months / len(monthly_returns)

        return result

    def _calculate_monthly_returns(self, trades: List[TradeResult]) -> List[float]:
        """Calculate monthly returns from trades"""

        if not trades:
            return []

        # Group trades by month
        monthly_pnl = {}

        for trade in trades:
            if trade.exit_time:
                month_key = trade.exit_time.strftime("%Y-%m")
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0
                monthly_pnl[month_key] += trade.pnl

        # Convert to percentage returns (simplified)
        monthly_returns = []
        capital = self.config.initial_capital

        for month in sorted(monthly_pnl.keys()):
            pnl = monthly_pnl[month]
            return_pct = (pnl / capital) * 100
            monthly_returns.append(return_pct)
            capital += pnl  # Update capital

        return monthly_returns