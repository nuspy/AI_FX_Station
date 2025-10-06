"""
Automated Trading Engine

Real-time trading engine that integrates all components:
- Multi-timeframe ensemble predictions
- Multi-model stacked ensemble
- Regime detection
- Multi-level risk management
- Regime-aware position sizing
- Smart execution optimization

Connects to broker API and executes trades automatically.
"""
from __future__ import annotations

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import threading
from enum import Enum

try:
    from ..models.multi_timeframe_ensemble import MultiTimeframeEnsemble, Timeframe
    from ..models.ml_stacked_ensemble import StackedMLEnsemble
    from ..regime.hmm_detector import HMMRegimeDetector
    from ..risk.multi_level_stop_loss import MultiLevelStopLoss
    from ..risk.regime_position_sizer import RegimePositionSizer, MarketRegime
    from ..execution.smart_execution import SmartExecutionOptimizer
except ImportError:
    pass


class TradingState(Enum):
    """Trading engine states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class Position:
    """Active trading position."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float
    take_profit: Optional[float]
    regime: Optional[str]
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None


@dataclass
class TradingConfig:
    """Trading engine configuration."""
    symbols: List[str]
    timeframes: List[str]
    update_interval_seconds: int = 60
    max_positions: int = 5
    account_balance: float = 10000.0
    risk_per_trade_pct: float = 1.0
    use_multi_timeframe: bool = True
    use_stacked_ensemble: bool = True
    use_regime_detection: bool = True
    use_smart_execution: bool = True


class AutomatedTradingEngine:
    """
    Automated trading engine with all advanced components.

    Features:
    - Real-time market data monitoring
    - Multi-timeframe ensemble predictions
    - Regime-aware position sizing
    - Multi-level risk management
    - Smart execution optimization
    - Broker API integration

    Example:
        >>> config = TradingConfig(
        ...     symbols=['EURUSD', 'GBPUSD'],
        ...     timeframes=['5m', '15m', '1h'],
        ...     max_positions=3
        ... )
        >>> engine = AutomatedTradingEngine(config)
        >>> engine.start()
    """

    def __init__(
        self,
        config: TradingConfig,
        broker_api: Optional[Any] = None
    ):
        """
        Initialize trading engine.

        Args:
            config: Trading configuration
            broker_api: Broker API instance (optional)
        """
        self.config = config
        self.broker_api = broker_api

        # State
        self.state = TradingState.STOPPED
        self.positions: Dict[str, Position] = {}
        self.account_balance = config.account_balance

        # Components
        self.mtf_ensemble: Optional[MultiTimeframeEnsemble] = None
        self.ml_ensemble: Optional[StackedMLEnsemble] = None
        self.regime_detector: Optional[HMMRegimeDetector] = None
        self.risk_manager = MultiLevelStopLoss()
        self.position_sizer = RegimePositionSizer(
            base_risk_per_trade_pct=config.risk_per_trade_pct
        )
        self.execution_optimizer = SmartExecutionOptimizer()

        # Threading
        self.trading_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Performance tracking
        self.trades_history: List[Dict] = []
        self.daily_pnl = 0.0

        logger.info("Automated Trading Engine initialized")

    def load_models(
        self,
        mtf_ensemble: Optional[MultiTimeframeEnsemble] = None,
        ml_ensemble: Optional[StackedMLEnsemble] = None,
        regime_detector: Optional[HMMRegimeDetector] = None
    ):
        """
        Load trained models into engine.

        Args:
            mtf_ensemble: Multi-timeframe ensemble
            ml_ensemble: Stacked ML ensemble
            regime_detector: Regime detector
        """
        self.mtf_ensemble = mtf_ensemble
        self.ml_ensemble = ml_ensemble
        self.regime_detector = regime_detector

        logger.info("Models loaded into trading engine")

    def start(self):
        """Start automated trading."""
        if self.state == TradingState.RUNNING:
            logger.warning("Trading engine already running")
            return

        logger.info("🚀 Starting automated trading engine...")
        self.state = TradingState.RUNNING
        self.stop_event.clear()

        # Start trading loop in separate thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()

        logger.info("✅ Trading engine started")

    def stop(self):
        """Stop automated trading."""
        logger.info("🛑 Stopping trading engine...")
        self.state = TradingState.STOPPED
        self.stop_event.set()

        if self.trading_thread:
            self.trading_thread.join(timeout=10)

        # Close all positions
        self._close_all_positions("Engine stopped")

        logger.info("✅ Trading engine stopped")

    def pause(self):
        """Pause trading (keep positions open)."""
        logger.info("⏸️  Pausing trading engine...")
        self.state = TradingState.PAUSED

    def resume(self):
        """Resume trading."""
        logger.info("▶️  Resuming trading engine...")
        self.state = TradingState.RUNNING

    def _trading_loop(self):
        """Main trading loop."""
        logger.info("Trading loop started")

        while not self.stop_event.is_set():
            try:
                if self.state == TradingState.RUNNING:
                    # 1. Update market data
                    market_data = self._fetch_market_data()

                    # 2. Manage existing positions
                    self._manage_positions(market_data)

                    # 3. Check for new trading opportunities
                    if len(self.positions) < self.config.max_positions:
                        self._check_new_opportunities(market_data)

                    # 4. Update performance metrics
                    self._update_metrics()

                # Sleep until next update
                time.sleep(self.config.update_interval_seconds)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self.state = TradingState.ERROR
                time.sleep(60)  # Wait before retry

    def _fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch current market data from broker.

        Returns:
            Dict mapping symbol to OHLCV dataframe
        """
        market_data = {}

        for symbol in self.config.symbols:
            if self.broker_api:
                # Fetch from broker API
                data = self.broker_api.get_ohlcv(
                    symbol=symbol,
                    timeframe='5m',
                    limit=500
                )
            else:
                # Simulated data (for testing)
                data = self._generate_simulated_data(symbol)

            market_data[symbol] = data

        return market_data

    def _generate_simulated_data(self, symbol: str) -> pd.DataFrame:
        """Generate simulated market data for testing."""
        # Simple random walk for testing
        n_candles = 500
        base_price = 1.1000 if 'EUR' in symbol else 1.2500

        prices = [base_price]
        for _ in range(n_candles - 1):
            change = np.random.randn() * 0.0001
            prices.append(prices[-1] + change)

        df = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=n_candles, freq='5min'),
            'open': prices,
            'high': [p * 1.0001 for p in prices],
            'low': [p * 0.9999 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_candles)
        })

        return df

    def _manage_positions(self, market_data: Dict[str, pd.DataFrame]):
        """
        Manage existing positions (check stops, update trailing).

        Args:
            market_data: Current market data
        """
        positions_to_close = []

        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue

            current_price = market_data[symbol]['close'].iloc[-1]

            # Calculate ATR for volatility stops
            atr = self._calculate_atr(market_data[symbol])

            # Update trailing stops
            position = self.risk_manager.update_trailing_stops(
                position.__dict__,
                current_price
            )
            self.positions[symbol] = Position(**position)

            # Check if stop triggered
            triggered, stop_type, reason = self.risk_manager.check_stop_triggered(
                position.__dict__,
                current_price,
                atr
            )

            if triggered:
                logger.info(f"🛑 Stop triggered for {symbol}: {reason}")
                positions_to_close.append((symbol, reason))

        # Close triggered positions
        for symbol, reason in positions_to_close:
            self._close_position(symbol, reason)

    def _check_new_opportunities(self, market_data: Dict[str, pd.DataFrame]):
        """
        Check for new trading opportunities.

        Args:
            market_data: Current market data
        """
        for symbol in self.config.symbols:
            # Skip if already have position
            if symbol in self.positions:
                continue

            if symbol not in market_data:
                continue

            # Get prediction
            signal, confidence = self._get_trading_signal(symbol, market_data)

            # Check if signal strong enough
            if signal != 0 and confidence > 0.6:
                # Calculate position size
                current_price = market_data[symbol]['close'].iloc[-1]

                # Detect regime
                regime = self._detect_regime(market_data[symbol])

                # Calculate position size
                position_size = self._calculate_position_size(
                    symbol,
                    current_price,
                    signal,
                    confidence,
                    regime
                )

                # Execute trade
                if position_size > 0:
                    self._open_position(
                        symbol,
                        signal,
                        current_price,
                        position_size,
                        regime
                    )

    def _get_trading_signal(
        self,
        symbol: str,
        market_data: Dict[str, pd.DataFrame]
    ) -> tuple[int, float]:
        """
        Get trading signal from models.

        Returns:
            (signal, confidence) where signal is -1/0/+1
        """
        # Use multi-timeframe ensemble if available
        if self.mtf_ensemble and self.config.use_multi_timeframe:
            # Prepare data for multiple timeframes
            data_by_tf = {}
            # For now, use same data (in production, fetch different TFs)
            for tf in [Timeframe.M5, Timeframe.H1]:
                data_by_tf[tf] = market_data[symbol]

            result = self.mtf_ensemble.predict_ensemble(data_by_tf)
            return result['final_signal'], result['confidence']

        # Fallback: simple prediction
        return 0, 0.0

    def _detect_regime(self, data: pd.DataFrame) -> Optional[str]:
        """Detect current market regime."""
        if self.regime_detector and self.config.use_regime_detection:
            try:
                self.regime_detector.fit(data.tail(200))
                state = self.regime_detector.predict_current(data)
                return state.regime.value if state else None
            except:
                return None
        return None

    def _calculate_position_size(
        self,
        symbol: str,
        price: float,
        signal: int,
        confidence: float,
        regime: Optional[str]
    ) -> float:
        """Calculate optimal position size."""
        # Map regime
        regime_map = {
            'trending_up': MarketRegime.TRENDING_UP,
            'trending_down': MarketRegime.TRENDING_DOWN,
            'ranging': MarketRegime.RANGING,
            'volatile': MarketRegime.VOLATILE
        }
        market_regime = regime_map.get(regime, MarketRegime.RANGING)

        # Calculate stop distance (2% default)
        stop_distance = price * 0.02
        stop_price = price - stop_distance if signal > 0 else price + stop_distance

        # Calculate size
        sizing = self.position_sizer.calculate_position_size(
            account_balance=self.account_balance,
            entry_price=price,
            stop_loss_price=stop_price,
            current_regime=market_regime,
            pattern_confidence=confidence
        )

        return sizing['position_size']

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return atr if not np.isnan(atr) else 0.001

    def _open_position(
        self,
        symbol: str,
        signal: int,
        price: float,
        size: float,
        regime: Optional[str]
    ):
        """Open new position."""
        direction = 'long' if signal > 0 else 'short'

        # Calculate stops
        stop_distance = price * 0.02
        stop_loss = price - stop_distance if signal > 0 else price + stop_distance
        take_profit = price + stop_distance * 2 if signal > 0 else price - stop_distance * 2

        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            entry_time=datetime.now(),
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime=regime
        )

        # Execute via broker API
        if self.broker_api:
            order_id = self.broker_api.place_order(
                symbol=symbol,
                side=direction,
                size=size,
                order_type='market'
            )
            logger.info(f"✅ Order placed: {order_id}")

        # Store position
        self.positions[symbol] = position

        logger.info(f"🟢 OPENED {direction.upper()} {symbol} @ {price:.5f}, size={size:.2f}, regime={regime}")

    def _close_position(self, symbol: str, reason: str):
        """Close existing position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Get current price (simulated)
        current_price = position.entry_price * 1.001  # Simulated

        # Calculate P&L
        if position.direction == 'long':
            pnl = (current_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - current_price) * position.size

        # Execute close via broker API
        if self.broker_api:
            order_id = self.broker_api.close_position(symbol)
            logger.info(f"✅ Position closed: {order_id}")

        # Update account
        self.account_balance += pnl

        # Record trade
        trade = {
            'symbol': symbol,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': current_price,
            'size': position.size,
            'pnl': pnl,
            'regime': position.regime,
            'reason': reason,
            'entry_time': position.entry_time,
            'exit_time': datetime.now()
        }
        self.trades_history.append(trade)

        # Remove position
        del self.positions[symbol]

        logger.info(f"🔴 CLOSED {symbol} @ {current_price:.5f}, P&L=${pnl:.2f}, reason={reason}")

    def _close_all_positions(self, reason: str):
        """Close all open positions."""
        symbols = list(self.positions.keys())
        for symbol in symbols:
            self._close_position(symbol, reason)

    def _update_metrics(self):
        """Update performance metrics."""
        # Daily P&L reset
        if datetime.now().hour == 0 and datetime.now().minute == 0:
            self.daily_pnl = 0.0

    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            'state': self.state.value,
            'account_balance': self.account_balance,
            'open_positions': len(self.positions),
            'total_trades': len(self.trades_history),
            'daily_pnl': self.daily_pnl,
            'positions': [
                {
                    'symbol': p.symbol,
                    'direction': p.direction,
                    'entry_price': p.entry_price,
                    'size': p.size
                }
                for p in self.positions.values()
            ]
        }
