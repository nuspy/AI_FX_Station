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
    from ..risk.adaptive_stop_loss_manager import AdaptiveStopLossManager, AdaptationFactors
    from ..risk.position_sizer import PositionSizer, BacktestTradeHistory
    from ..execution.smart_execution import SmartExecutionOptimizer
    from ..services.parameter_loader import ParameterLoaderService
    from ..services.dom_aggregator import DOMAggreg atorService
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
    database_path: str = "forex_data.db"  # Path to database for parameter loading
    use_optimized_parameters: bool = True  # Use optimized parameters from database
    use_adaptive_stops: bool = True  # Use adaptive stop loss manager
    position_sizing_method: str = 'kelly'  # Position sizing: fixed_fractional, kelly, optimal_f, volatility_adjusted
    kelly_fraction: float = 0.25  # Kelly fraction (0.25 = quarter Kelly)
    use_dom_data: bool = True  # Use real-time DOM data for spread and liquidity
    use_sentiment_data: bool = True  # Use sentiment for signal filtering and position sizing
    db_engine: Optional[Any] = None  # Database engine for DOM and sentiment services


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

        # Old position sizer (keeping for backward compatibility)
        self.position_sizer = RegimePositionSizer(
            base_risk_per_trade_pct=config.risk_per_trade_pct
        )

        # New advanced position sizer
        self.advanced_position_sizer = PositionSizer(
            base_risk_pct=config.risk_per_trade_pct,
            kelly_fraction=config.kelly_fraction,
            max_position_size_pct=5.0,
            min_position_size_pct=0.1,
            max_total_exposure_pct=20.0,
            drawdown_reduction_enabled=True
        )

        self.execution_optimizer = SmartExecutionOptimizer()

        logger.info(f"✅ Advanced Position Sizer initialized: method={config.position_sizing_method}")

        # Adaptive Stop Loss Manager
        self.adaptive_sl_manager: Optional[AdaptiveStopLossManager] = None
        if config.use_adaptive_stops:
            self.adaptive_sl_manager = AdaptiveStopLossManager(
                base_sl_atr_multiplier=2.0,
                base_tp_atr_multiplier=3.0,
                trailing_enabled=True,
                trailing_activation_pct=50.0
            )
            logger.info("✅ Adaptive Stop Loss Manager initialized")

        # Parameter Loader Service
        self.parameter_loader: Optional[ParameterLoaderService] = None
        if config.use_optimized_parameters:
            try:
                self.parameter_loader = ParameterLoaderService(
                    db_path=config.database_path,
                    cache_ttl_seconds=3600,  # 1 hour cache
                    require_validation=True
                )
                logger.info("✅ Parameter Loader Service initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Parameter Loader: {e}. Using defaults.")

        # DOM Aggregator Service
        self.dom_service: Optional[DOMAggreg atorService] = None
        if config.use_dom_data and config.db_engine:
            try:
                self.dom_service = DOMAggreg atorService(
                    engine=config.db_engine,
                    symbols=config.symbols
                )
                logger.info("✅ DOM Aggregator Service initialized")
            except Exception as e:
                logger.warning(f"Could not initialize DOM Service: {e}. Using statistical spreads.")

        # Sentiment Aggregator Service
        self.sentiment_service: Optional[Any] = None
        if config.use_sentiment_data and config.db_engine:
            try:
                from ..services.sentiment_aggregator import SentimentAggregatorService
                self.sentiment_service = SentimentAggregatorService(
                    engine=config.db_engine,
                    symbols=config.symbols,
                    interval_seconds=30
                )
                self.sentiment_service.start()
                logger.info("✅ Sentiment Aggregator Service initialized and started")
            except Exception as e:
                logger.warning(f"Could not initialize Sentiment Service: {e}. Trading without sentiment.")

        # Threading
        self.trading_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Performance tracking
        self.trades_history: List[Dict] = []
        self.daily_pnl = 0.0

        # ATR cache for parameter loading
        self._last_atr: Dict[str, float] = {}

        # Spread history cache for anomaly detection (symbol -> list of spreads)
        self._spread_history: Dict[str, List[float]] = {}
        self._spread_history_size = 720  # 1 hour of data at 5-second intervals

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

        # Stop sentiment service
        if self.sentiment_service:
            try:
                self.sentiment_service.stop()
                logger.info("✅ Sentiment service stopped")
            except Exception as e:
                logger.warning(f"Error stopping sentiment service: {e}")

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

            # Cache ATR for use in _open_position
            self._last_atr[symbol] = atr

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
        Get trading signal from models with sentiment filtering.

        Returns:
            (signal, confidence) where signal is -1/0/+1
        """
        # Get base signal from models
        signal = 0
        confidence = 0.0

        if self.mtf_ensemble and self.config.use_multi_timeframe:
            # Prepare data for multiple timeframes
            data_by_tf = {}
            # For now, use same data (in production, fetch different TFs)
            for tf in [Timeframe.M5, Timeframe.H1]:
                data_by_tf[tf] = market_data[symbol]

            result = self.mtf_ensemble.predict_ensemble(data_by_tf)
            signal = result['final_signal']
            confidence = result['confidence']
        else:
            # Fallback: no prediction available
            return 0, 0.0

        # Apply sentiment filtering (contrarian strategy)
        if self.sentiment_service:
            try:
                sentiment_metrics = self.sentiment_service.get_latest_sentiment_metrics(symbol)
                if sentiment_metrics:
                    contrarian_signal = sentiment_metrics.get('contrarian_signal', 0.0)

                    # Contrarian strategy: extreme positioning = fade the crowd
                    if contrarian_signal > 0 and signal < 0:
                        # Crowd is short (bearish), our signal is bearish too = reduce confidence
                        confidence *= 0.7
                        logger.info(
                            f"🔄 Sentiment conflict for {symbol}: crowd bearish, signal bearish, "
                            f"reduced confidence to {confidence:.2f}"
                        )
                    elif contrarian_signal < 0 and signal > 0:
                        # Crowd is long (bullish), our signal is bullish too = reduce confidence
                        confidence *= 0.7
                        logger.info(
                            f"🔄 Sentiment conflict for {symbol}: crowd bullish, signal bullish, "
                            f"reduced confidence to {confidence:.2f}"
                        )
                    elif contrarian_signal > 0 and signal > 0:
                        # Crowd is short, our signal is bullish = boost confidence (contrarian)
                        confidence *= 1.3
                        logger.info(
                            f"✅ Sentiment alignment for {symbol}: contrarian bullish, "
                            f"boosted confidence to {confidence:.2f}"
                        )
                    elif contrarian_signal < 0 and signal < 0:
                        # Crowd is long, our signal is bearish = boost confidence (contrarian)
                        confidence *= 1.3
                        logger.info(
                            f"✅ Sentiment alignment for {symbol}: contrarian bearish, "
                            f"boosted confidence to {confidence:.2f}"
                        )

            except Exception as e:
                logger.debug(f"Could not apply sentiment filtering for {symbol}: {e}")

        return signal, confidence

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
        """
        Calculate optimal position size with multiple constraints.

        Implements multiple sizing constraints:
        - Risk-based sizing (Kelly, fixed fractional)
        - Regime adjustments
        - Liquidity constraints (from DOM)
        - Order flow alignment
        - Spread cost penalty
        - Sentiment-based adjustments (contrarian strategy)
        """
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

        # Calculate base size from existing position sizer
        sizing = self.position_sizer.calculate_position_size(
            account_balance=self.account_balance,
            entry_price=price,
            stop_loss_price=stop_price,
            current_regime=market_regime,
            pattern_confidence=confidence
        )

        base_size = sizing['position_size']
        final_size = base_size

        # Apply liquidity constraints if DOM data available
        if self.dom_service:
            try:
                # Get DOM metrics
                dom_metrics = self.dom_service.get_latest_dom_metrics(symbol)
                if dom_metrics:
                    # 1. LIQUIDITY CONSTRAINT
                    # Get full DOM snapshot for depth calculation
                    from sqlalchemy import text
                    with self.dom_service.engine.connect() as conn:
                        query = text(
                            "SELECT bids, asks FROM market_depth "
                            "WHERE symbol = :symbol "
                            "ORDER BY ts_utc DESC LIMIT 1"
                        )
                        row = conn.execute(query, {"symbol": symbol}).fetchone()

                        if row:
                            import json
                            bids = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                            asks = json.loads(row[1]) if isinstance(row[1], str) else row[1]

                            # Calculate total depth (top 20 levels)
                            max_levels = min(20, len(bids), len(asks))
                            bid_depth = sum(vol for _, vol in bids[:max_levels]) if bids else 0.0
                            ask_depth = sum(vol for _, vol in asks[:max_levels]) if asks else 0.0
                            total_depth = bid_depth + ask_depth

                            if total_depth > 0:
                                # Maximum 50% of available depth
                                max_liquidity_size = total_depth * 0.5

                                if final_size > max_liquidity_size:
                                    logger.warning(
                                        f"💧 Position size reduced from {final_size:.2f} to {max_liquidity_size:.2f} "
                                        f"due to liquidity constraint for {symbol} (depth={total_depth:.2f})"
                                    )
                                    final_size = max_liquidity_size

                            # 2. ORDER FLOW ADJUSTMENT
                            imbalance = dom_metrics.get('imbalance', 0.0)
                            if imbalance != 0:
                                # Calculate flow alignment
                                # Long + strong bids (>0.3): favorable, boost 1.2x
                                # Long + strong asks (<-0.3): unfavorable, reduce 0.7x
                                # Short + strong asks (<-0.3): favorable, boost 1.2x
                                # Short + strong bids (>0.3): unfavorable, reduce 0.7x

                                flow_adjustment = 1.0

                                if signal > 0:  # Long
                                    if imbalance > 0.3:
                                        flow_adjustment = 1.2
                                        logger.info(f"📈 Position size boosted 1.2x due to favorable order flow (bid-heavy: {imbalance:.2f})")
                                    elif imbalance < -0.3:
                                        flow_adjustment = 0.7
                                        logger.info(f"📉 Position size reduced 0.7x due to unfavorable order flow (ask-heavy: {imbalance:.2f})")
                                else:  # Short
                                    if imbalance < -0.3:
                                        flow_adjustment = 1.2
                                        logger.info(f"📈 Position size boosted 1.2x due to favorable order flow (ask-heavy: {imbalance:.2f})")
                                    elif imbalance > 0.3:
                                        flow_adjustment = 0.7
                                        logger.info(f"📉 Position size reduced 0.7x due to unfavorable order flow (bid-heavy: {imbalance:.2f})")

                                final_size *= flow_adjustment

                            # 3. SPREAD COST PENALTY
                            spread = dom_metrics.get('spread', 0.0)
                            if spread > 0:
                                # Convert spread to pips (assuming 4-decimal pairs)
                                spread_pips = spread * 10000
                                spread_penalty = 1.0

                                if spread_pips > 3.0:  # Wide spread (>3 pips)
                                    spread_penalty = 0.7
                                    logger.warning(
                                        f"📊 Position size reduced 0.7x due to wide spread "
                                        f"({spread_pips:.1f} pips) for {symbol}"
                                    )

                                final_size *= spread_penalty

            except Exception as e:
                logger.debug(f"Could not apply DOM-based position sizing for {symbol}: {e}")

        # 4. SENTIMENT ADJUSTMENT
        if self.sentiment_service:
            try:
                sentiment_metrics = self.sentiment_service.get_latest_sentiment_metrics(symbol)
                if sentiment_metrics:
                    contrarian_signal = sentiment_metrics.get('contrarian_signal', 0.0)
                    sentiment_confidence = sentiment_metrics.get('confidence', 0.0)

                    # Apply sentiment-based sizing adjustment (contrarian strategy)
                    sentiment_adjustment = 1.0

                    # Strong sentiment (confidence > 0.6)
                    if sentiment_confidence > 0.6:
                        if (contrarian_signal > 0 and signal > 0) or (contrarian_signal < 0 and signal < 0):
                            # Sentiment agrees with signal = boost size
                            sentiment_adjustment = 1.2
                            logger.info(
                                f"💪 Position size boosted 1.2x due to strong sentiment alignment "
                                f"(confidence={sentiment_confidence:.2f})"
                            )
                        elif (contrarian_signal > 0 and signal < 0) or (contrarian_signal < 0 and signal > 0):
                            # Sentiment conflicts with signal = reduce size
                            sentiment_adjustment = 0.8
                            logger.info(
                                f"⚠️  Position size reduced 0.8x due to sentiment conflict "
                                f"(confidence={sentiment_confidence:.2f})"
                            )
                    # Moderate sentiment (0.4 < confidence <= 0.6)
                    elif sentiment_confidence > 0.4:
                        if (contrarian_signal > 0 and signal > 0) or (contrarian_signal < 0 and signal < 0):
                            # Moderate agreement = slight boost
                            sentiment_adjustment = 1.1
                            logger.info(
                                f"📊 Position size boosted 1.1x due to moderate sentiment alignment "
                                f"(confidence={sentiment_confidence:.2f})"
                            )
                        elif (contrarian_signal > 0 and signal < 0) or (contrarian_signal < 0 and signal > 0):
                            # Moderate conflict = slight reduction
                            sentiment_adjustment = 0.9
                            logger.info(
                                f"📊 Position size reduced 0.9x due to moderate sentiment conflict "
                                f"(confidence={sentiment_confidence:.2f})"
                            )

                    final_size *= sentiment_adjustment

            except Exception as e:
                logger.debug(f"Could not apply sentiment-based position sizing for {symbol}: {e}")

        # Ensure final size is positive and reasonable
        if final_size < 0:
            final_size = 0.0

        # Log final decision if size was adjusted
        if abs(final_size - base_size) > 0.01:
            adjustment_pct = ((final_size - base_size) / base_size * 100) if base_size > 0 else 0
            logger.info(
                f"✅ Final position size for {symbol}: {final_size:.2f} "
                f"(base={base_size:.2f}, adjustment={adjustment_pct:+.1f}%)"
            )

        return final_size

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

    def _get_real_spread(self, symbol: str, price: float) -> Dict[str, Any]:
        """
        Get real-time spread from DOM service with anomaly detection.

        Args:
            symbol: Trading symbol
            price: Current price (for fallback calculation)

        Returns:
            Dictionary with:
                - current_spread: Real-time spread
                - avg_spread: Historical average spread
                - is_anomaly: Whether spread is abnormally wide
                - anomaly_level: 'normal', 'elevated', 'high', 'extreme'
                - used_fallback: Whether fallback was used
        """
        # Try to get real spread from DOM service
        if self.dom_service:
            try:
                dom_metrics = self.dom_service.get_latest_dom_metrics(symbol)
                if dom_metrics and dom_metrics.get('spread') is not None:
                    current_spread = dom_metrics['spread']

                    # Update spread history for this symbol
                    if symbol not in self._spread_history:
                        self._spread_history[symbol] = []

                    self._spread_history[symbol].append(current_spread)

                    # Keep only recent history
                    if len(self._spread_history[symbol]) > self._spread_history_size:
                        self._spread_history[symbol] = self._spread_history[symbol][-self._spread_history_size:]

                    # Calculate historical average
                    if len(self._spread_history[symbol]) >= 10:
                        avg_spread = sum(self._spread_history[symbol]) / len(self._spread_history[symbol])
                    else:
                        # Not enough history, use current as average
                        avg_spread = current_spread

                    # Detect anomaly
                    ratio = current_spread / avg_spread if avg_spread > 0 else 1.0
                    is_anomaly = ratio > 2.0
                    anomaly_level = 'normal'

                    if ratio > 3.0:
                        anomaly_level = 'extreme'
                        logger.warning(
                            f"⚠️  EXTREME spread for {symbol}: {current_spread:.5f} "
                            f"({ratio:.1f}x avg {avg_spread:.5f})"
                        )
                    elif ratio > 2.0:
                        anomaly_level = 'high'
                        logger.warning(
                            f"⚠️  HIGH spread for {symbol}: {current_spread:.5f} "
                            f"({ratio:.1f}x avg {avg_spread:.5f})"
                        )
                    elif ratio > 1.5:
                        anomaly_level = 'elevated'
                        logger.info(
                            f"📊 Elevated spread for {symbol}: {current_spread:.5f} "
                            f"({ratio:.1f}x avg {avg_spread:.5f})"
                        )

                    return {
                        'current_spread': current_spread,
                        'avg_spread': avg_spread,
                        'is_anomaly': is_anomaly,
                        'anomaly_level': anomaly_level,
                        'ratio': ratio,
                        'used_fallback': False
                    }

            except Exception as e:
                logger.debug(f"Could not get DOM spread for {symbol}: {e}")

        # Fallback: Use statistical model based on symbol
        # Different symbols have different typical spreads
        fallback_spread = self._get_statistical_spread(symbol, price)

        logger.debug(f"Using fallback spread for {symbol}: {fallback_spread:.5f}")

        return {
            'current_spread': fallback_spread,
            'avg_spread': fallback_spread,
            'is_anomaly': False,
            'anomaly_level': 'normal',
            'ratio': 1.0,
            'used_fallback': True
        }

    def _get_statistical_spread(self, symbol: str, price: float) -> float:
        """
        Calculate statistical spread based on symbol and time-of-day.

        Args:
            symbol: Trading symbol
            price: Current price

        Returns:
            Estimated spread
        """
        # Typical spreads for major forex pairs (in absolute price terms)
        # These are reasonable estimates based on market averages
        typical_spreads = {
            'EURUSD': 0.00015,  # 1.5 pips
            'GBPUSD': 0.00020,  # 2.0 pips
            'USDJPY': 0.002,    # 0.2 pips (JPY quoted)
            'AUDUSD': 0.00020,  # 2.0 pips
            'USDCAD': 0.00020,  # 2.0 pips
            'USDCHF': 0.00020,  # 2.0 pips
            'NZDUSD': 0.00025,  # 2.5 pips
            'EURGBP': 0.00020,  # 2.0 pips
            'EURJPY': 0.002,    # 0.2 pips (JPY quoted)
            'GBPJPY': 0.003,    # 0.3 pips (JPY quoted)
        }

        # Try to find exact match
        if symbol in typical_spreads:
            return typical_spreads[symbol]

        # Default fallback: 0.01% of price (1 pip for most pairs)
        return price * 0.0001

    def _open_position(
        self,
        symbol: str,
        signal: int,
        price: float,
        size: float,
        regime: Optional[str],
        pattern_type: str = 'pattern',
        timeframe: str = '5m'
    ):
        """
        Open new position using optimized parameters.

        Args:
            symbol: Trading symbol
            signal: Trading signal (>0 long, <0 short)
            price: Entry price
            size: Position size
            regime: Market regime
            pattern_type: Pattern type for parameter loading
            timeframe: Timeframe for parameter loading
        """
        direction = 'long' if signal > 0 else 'short'

        # Load optimized parameters
        sl_multiplier = 2.0  # Default
        tp_multiplier = 3.0  # Default
        params_source = 'default'

        if self.parameter_loader:
            try:
                params = self.parameter_loader.load_parameters(
                    pattern_type=pattern_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    market_regime=regime
                )
                sl_multiplier = params.action_params.get('sl_atr_multiplier', 2.0)
                tp_multiplier = params.action_params.get('tp_atr_multiplier', 3.0)
                params_source = params.source

                logger.info(
                    f"📊 Loaded params for {pattern_type}/{symbol}/{timeframe}/{regime}: "
                    f"SL={sl_multiplier:.1f}x, TP={tp_multiplier:.1f}x (source={params_source})"
                )
            except Exception as e:
                logger.warning(f"Could not load optimized params: {e}. Using defaults.")

        # Calculate stops using adaptive SL manager if available
        atr = 0.001  # Fallback
        if hasattr(self, '_last_atr') and symbol in getattr(self, '_last_atr', {}):
            atr = self._last_atr[symbol]
        else:
            # Use percentage-based fallback
            atr = price * 0.01

        # Get current spread from DOM service (with fallback to statistical model)
        spread_data = self._get_real_spread(symbol, price)
        current_spread = spread_data['current_spread']
        avg_spread = spread_data['avg_spread']

        # Log spread anomaly warnings
        if spread_data['is_anomaly']:
            logger.warning(
                f"⚠️  Spread anomaly detected for {symbol}: {spread_data['anomaly_level']}, "
                f"ratio={spread_data['ratio']:.2f}x"
            )

        if self.adaptive_sl_manager:
            # Use adaptive stop loss manager
            stop_loss, take_profit, stop_levels = self.adaptive_sl_manager.calculate_initial_stops(
                symbol=symbol,
                direction=direction,
                entry_price=price,
                atr=atr,
                current_spread=current_spread,
                avg_spread=avg_spread,
                regime=regime,
                sl_multiplier_override=sl_multiplier,
                tp_multiplier_override=tp_multiplier,
            )
            logger.info(
                f"Adaptive SL calculated: {len(stop_levels)} levels, "
                f"SL={stop_loss:.5f}, TP={take_profit:.5f}"
            )
        else:
            # Fallback to simple calculation
            stop_distance = atr * sl_multiplier
            stop_loss = price - stop_distance if signal > 0 else price + stop_distance
            take_profit = price + (atr * tp_multiplier) if signal > 0 else price - (atr * tp_multiplier)

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
