"""
Order Flow Analysis Engine

Generates signals from microstructure and order book dynamics.
Analyzes bid/ask spread, depth, volume imbalance, and large orders.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque


class OrderFlowSignalType(Enum):
    """Types of order flow signals"""
    LIQUIDITY_IMBALANCE = "liquidity_imbalance"
    ABSORPTION = "absorption"
    EXHAUSTION = "exhaustion"
    LARGE_PLAYER = "large_player"
    SPREAD_ANOMALY = "spread_anomaly"
    HIDDEN_ORDER = "hidden_order"


@dataclass
class OrderFlowMetrics:
    """Real-time order flow metrics"""
    timestamp: int
    symbol: str
    timeframe: str

    # Order book depth
    bid_ask_spread: float
    bid_depth: float
    ask_depth: float
    depth_imbalance: float  # (bid - ask) / (bid + ask)

    # Volume metrics
    buy_volume: float
    sell_volume: float
    volume_imbalance: float  # (buy - sell) / (buy + sell)
    large_order_count: int

    # Statistical measures
    spread_zscore: float
    imbalance_zscore: float
    absorption_detected: bool
    exhaustion_detected: bool

    # Context
    regime: Optional[str] = None
    price: Optional[float] = None


@dataclass
class OrderFlowSignal:
    """Order flow-based trading signal"""
    signal_type: OrderFlowSignalType
    direction: str  # 'bull' or 'bear'
    strength: float  # 0-1
    confidence: float  # 0-1
    timestamp: int
    symbol: str
    timeframe: str
    entry_price: float
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class OrderFlowAnalyzer:
    """
    Analyzes order flow and microstructure for trading signals.

    Detects:
    - Liquidity imbalances (bid/ask depth skew)
    - Absorption patterns (large orders absorbed without price movement)
    - Exhaustion patterns (declining volume with continued price movement)
    - Large player activity (unusual order sizes)
    - Spread anomalies (abnormal widening/tightening)
    """

    def __init__(
        self,
        rolling_window: int = 20,
        imbalance_threshold: float = 0.3,
        zscore_threshold: float = 2.0,
        large_order_percentile: float = 0.95,
        absorption_threshold: float = 0.4,
        exhaustion_volume_decline: float = 0.3
    ):
        """
        Initialize order flow analyzer.

        Args:
            rolling_window: Bars for rolling statistics
            imbalance_threshold: Threshold for significant imbalance (0-1)
            zscore_threshold: Z-score threshold for anomalies
            large_order_percentile: Percentile for large order detection
            absorption_threshold: Threshold for absorption detection
            exhaustion_volume_decline: Volume decline for exhaustion (0-1)
        """
        self.rolling_window = rolling_window
        self.imbalance_threshold = imbalance_threshold
        self.zscore_threshold = zscore_threshold
        self.large_order_percentile = large_order_percentile
        self.absorption_threshold = absorption_threshold
        self.exhaustion_volume_decline = exhaustion_volume_decline

        # State tracking
        self.spread_history: deque = deque(maxlen=rolling_window)
        self.imbalance_history: deque = deque(maxlen=rolling_window)
        self.volume_history: deque = deque(maxlen=rolling_window)
        self.order_size_history: deque = deque(maxlen=100)

    def compute_metrics(
        self,
        timestamp: int,
        symbol: str,
        timeframe: str,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float,
        buy_volume: float,
        sell_volume: float,
        order_sizes: Optional[List[float]] = None,
        regime: Optional[str] = None,
        current_price: Optional[float] = None
    ) -> OrderFlowMetrics:
        """
        Compute order flow metrics from market data.

        Args:
            timestamp: Unix timestamp in milliseconds
            symbol: Trading symbol
            timeframe: Timeframe
            bid_price: Current bid price
            ask_price: Current ask price
            bid_size: Total bid depth
            ask_size: Total ask depth
            buy_volume: Buy volume in period
            sell_volume: Sell volume in period
            order_sizes: List of individual order sizes
            regime: Current market regime
            current_price: Current market price

        Returns:
            Order flow metrics
        """
        # Basic calculations
        spread = ask_price - bid_price
        total_depth = bid_size + ask_size
        depth_imbalance = (bid_size - ask_size) / total_depth if total_depth > 0 else 0.0

        total_volume = buy_volume + sell_volume
        volume_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0

        # Detect large orders
        large_order_count = 0
        if order_sizes:
            self.order_size_history.extend(order_sizes)
            if len(self.order_size_history) >= 20:
                large_threshold = np.percentile(list(self.order_size_history), self.large_order_percentile * 100)
                large_order_count = sum(1 for size in order_sizes if size >= large_threshold)

        # Update histories
        self.spread_history.append(spread)
        self.imbalance_history.append(volume_imbalance)
        self.volume_history.append(total_volume)

        # Calculate z-scores
        if len(self.spread_history) >= 10:
            spread_mean = np.mean(self.spread_history)
            spread_std = np.std(self.spread_history)
            spread_zscore = (spread - spread_mean) / spread_std if spread_std > 0 else 0.0
        else:
            spread_zscore = 0.0

        if len(self.imbalance_history) >= 10:
            imbalance_mean = np.mean(self.imbalance_history)
            imbalance_std = np.std(self.imbalance_history)
            imbalance_zscore = (volume_imbalance - imbalance_mean) / imbalance_std if imbalance_std > 0 else 0.0
        else:
            imbalance_zscore = 0.0

        # Detect absorption
        absorption_detected = self._detect_absorption(
            depth_imbalance, volume_imbalance, large_order_count
        )

        # Detect exhaustion
        exhaustion_detected = self._detect_exhaustion()

        return OrderFlowMetrics(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            bid_ask_spread=spread,
            bid_depth=bid_size,
            ask_depth=ask_size,
            depth_imbalance=depth_imbalance,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            volume_imbalance=volume_imbalance,
            large_order_count=large_order_count,
            spread_zscore=spread_zscore,
            imbalance_zscore=imbalance_zscore,
            absorption_detected=absorption_detected,
            exhaustion_detected=exhaustion_detected,
            regime=regime,
            price=current_price
        )

    def _detect_absorption(
        self,
        depth_imbalance: float,
        volume_imbalance: float,
        large_order_count: int
    ) -> bool:
        """
        Detect absorption pattern.

        Absorption occurs when large orders are filled without significant price movement,
        indicating strong support/resistance.
        """
        # Strong imbalance + large orders = absorption
        strong_depth_imbalance = abs(depth_imbalance) > self.absorption_threshold
        strong_volume_imbalance = abs(volume_imbalance) > self.absorption_threshold
        has_large_orders = large_order_count > 0

        return strong_depth_imbalance and strong_volume_imbalance and has_large_orders

    def _detect_exhaustion(self) -> bool:
        """
        Detect exhaustion pattern.

        Exhaustion occurs when volume declines while price continues trending,
        suggesting weakening momentum.
        """
        if len(self.volume_history) < self.rolling_window:
            return False

        volumes = list(self.volume_history)
        recent_volume = np.mean(volumes[-5:])
        earlier_volume = np.mean(volumes[-self.rolling_window:-5])

        if earlier_volume == 0:
            return False

        volume_decline = (earlier_volume - recent_volume) / earlier_volume
        return volume_decline > self.exhaustion_volume_decline

    def generate_signals(
        self,
        metrics: OrderFlowMetrics,
        current_price: float,
        atr: float
    ) -> List[OrderFlowSignal]:
        """
        Generate trading signals from order flow metrics.

        Args:
            metrics: Computed order flow metrics
            current_price: Current market price
            atr: Average True Range for stops/targets

        Returns:
            List of order flow signals
        """
        signals: List[OrderFlowSignal] = []

        # 1. Liquidity imbalance signal
        if abs(metrics.imbalance_zscore) > self.zscore_threshold:
            direction = 'bull' if metrics.volume_imbalance > 0 else 'bear'
            strength = min(abs(metrics.imbalance_zscore) / (self.zscore_threshold * 2), 1.0)
            confidence = min(abs(metrics.volume_imbalance), 1.0)

            target_mult = 2.0 if abs(metrics.volume_imbalance) > 0.5 else 1.5
            stop_mult = 1.0

            signals.append(OrderFlowSignal(
                signal_type=OrderFlowSignalType.LIQUIDITY_IMBALANCE,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timestamp=metrics.timestamp,
                symbol=metrics.symbol,
                timeframe=metrics.timeframe,
                entry_price=current_price,
                target_price=current_price + (atr * target_mult if direction == 'bull' else -atr * target_mult),
                stop_price=current_price - (atr * stop_mult if direction == 'bull' else -atr * stop_mult),
                metadata={
                    'volume_imbalance': metrics.volume_imbalance,
                    'imbalance_zscore': metrics.imbalance_zscore
                }
            ))

        # 2. Absorption signal
        if metrics.absorption_detected:
            # Absorption suggests reversal
            direction = 'bull' if metrics.depth_imbalance > 0 else 'bear'
            strength = min(abs(metrics.depth_imbalance) + 0.3, 1.0)
            confidence = 0.7 + (0.3 * (metrics.large_order_count / 10))

            signals.append(OrderFlowSignal(
                signal_type=OrderFlowSignalType.ABSORPTION,
                direction=direction,
                strength=strength,
                confidence=min(confidence, 1.0),
                timestamp=metrics.timestamp,
                symbol=metrics.symbol,
                timeframe=metrics.timeframe,
                entry_price=current_price,
                target_price=current_price + (atr * 2.5 if direction == 'bull' else -atr * 2.5),
                stop_price=current_price - (atr * 1.2 if direction == 'bull' else -atr * 1.2),
                metadata={
                    'depth_imbalance': metrics.depth_imbalance,
                    'large_orders': metrics.large_order_count
                }
            ))

        # 3. Exhaustion signal
        if metrics.exhaustion_detected:
            # Exhaustion suggests reversal
            # Infer direction from recent volume imbalance
            direction = 'bear' if metrics.volume_imbalance > 0 else 'bull'  # Reverse
            strength = 0.7
            confidence = 0.65

            signals.append(OrderFlowSignal(
                signal_type=OrderFlowSignalType.EXHAUSTION,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timestamp=metrics.timestamp,
                symbol=metrics.symbol,
                timeframe=metrics.timeframe,
                entry_price=current_price,
                target_price=current_price + (atr * 2.0 if direction == 'bull' else -atr * 2.0),
                stop_price=current_price - (atr * 1.5 if direction == 'bull' else -atr * 1.5),
                metadata={'exhaustion_detected': True}
            ))

        # 4. Large player activity
        if metrics.large_order_count > 2:
            # Large orders suggest institutional activity
            direction = 'bull' if metrics.volume_imbalance > 0 else 'bear'
            strength = min(metrics.large_order_count / 5, 1.0)
            confidence = 0.75

            signals.append(OrderFlowSignal(
                signal_type=OrderFlowSignalType.LARGE_PLAYER,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timestamp=metrics.timestamp,
                symbol=metrics.symbol,
                timeframe=metrics.timeframe,
                entry_price=current_price,
                target_price=current_price + (atr * 3.0 if direction == 'bull' else -atr * 3.0),
                stop_price=current_price - (atr * 1.0 if direction == 'bull' else -atr * 1.0),
                metadata={'large_order_count': metrics.large_order_count}
            ))

        # 5. Spread anomaly
        if abs(metrics.spread_zscore) > self.zscore_threshold:
            # Unusual spread suggests upcoming move
            direction = 'bull' if metrics.spread_zscore > 0 else 'bear'  # Wide spread = volatility
            strength = min(abs(metrics.spread_zscore) / 3, 1.0)
            confidence = 0.6

            signals.append(OrderFlowSignal(
                signal_type=OrderFlowSignalType.SPREAD_ANOMALY,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timestamp=metrics.timestamp,
                symbol=metrics.symbol,
                timeframe=metrics.timeframe,
                entry_price=current_price,
                target_price=current_price + (atr * 1.5 if direction == 'bull' else -atr * 1.5),
                stop_price=current_price - (atr * 1.2 if direction == 'bull' else -atr * 1.2),
                metadata={'spread_zscore': metrics.spread_zscore}
            ))

        return signals

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        regime_col: Optional[str] = None
    ) -> Tuple[List[OrderFlowMetrics], List[OrderFlowSignal]]:
        """
        Analyze a DataFrame of market data.

        Args:
            df: DataFrame with columns: timestamp, bid, ask, bid_size, ask_size, buy_volume, sell_volume, atr
            symbol: Trading symbol
            timeframe: Timeframe
            regime_col: Optional column name for regime

        Returns:
            (metrics_list, signals_list)
        """
        metrics_list: List[OrderFlowMetrics] = []
        signals_list: List[OrderFlowSignal] = []

        for idx, row in df.iterrows():
            regime = row[regime_col] if regime_col and regime_col in df.columns else None

            metrics = self.compute_metrics(
                timestamp=int(row['timestamp']),
                symbol=symbol,
                timeframe=timeframe,
                bid_price=row['bid'],
                ask_price=row['ask'],
                bid_size=row.get('bid_size', 1000.0),
                ask_size=row.get('ask_size', 1000.0),
                buy_volume=row.get('buy_volume', row.get('volume', 0) / 2),
                sell_volume=row.get('sell_volume', row.get('volume', 0) / 2),
                order_sizes=None,  # Would need tick data
                regime=regime,
                current_price=row['close']
            )

            metrics_list.append(metrics)

            # Generate signals
            atr_val = row.get('atr', row['close'] * 0.01)  # Default 1% if no ATR
            signals = self.generate_signals(metrics, row['close'], atr_val)
            signals_list.extend(signals)

        return metrics_list, signals_list

    def reset(self):
        """Reset internal state"""
        self.spread_history.clear()
        self.imbalance_history.clear()
        self.volume_history.clear()
        self.order_size_history.clear()
