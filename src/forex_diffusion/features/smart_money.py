"""
Smart Money Detection

Identifies institutional order flow patterns:
- Unusual Volume Detection (volume spikes indicating institutional activity)
- Volume Absorption (large volume with minimal price movement)
- Imbalance Detection (buy/sell pressure imbalances)
- Order Block Identification (zones where smart money entered)

Reference: ICT (Inner Circle Trader) concepts + institutional order flow analysis
"""
from __future__ import annotations

from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import zscore
from loguru import logger


@dataclass
class SmartMoneySignal:
    """Smart money detection result"""
    unusual_volume: float  # 0-1, strength of volume anomaly
    absorption_detected: bool  # True if absorption pattern found
    buy_pressure: float  # -1 to 1, negative = selling, positive = buying
    order_block_strength: float  # 0-1, strength of order block
    institutional_footprint: float  # 0-1, aggregate institutional activity score


class SmartMoneyDetector:
    """
    Detects institutional money movement patterns.

    Identifies when professional traders (banks, funds, institutions)
    are actively entering or exiting positions.
    """

    def __init__(
        self,
        volume_ma_period: int = 20,
        volume_std_threshold: float = 2.0,  # Unusual volume = 2 std deviations
        absorption_threshold: float = 0.3,  # Max price movement for absorption
        imbalance_period: int = 5,  # Period for pressure calculation
    ):
        """
        Initialize Smart Money detector.

        Args:
            volume_ma_period: Period for volume moving average
            volume_std_threshold: Z-score threshold for unusual volume
            absorption_threshold: Max price range % for absorption detection
            imbalance_period: Window for buy/sell pressure calculation
        """
        self.volume_ma_period = volume_ma_period
        self.volume_std_threshold = volume_std_threshold
        self.absorption_threshold = absorption_threshold
        self.imbalance_period = imbalance_period

    def detect_unusual_volume(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> float:
        """
        Detect unusual volume spikes (potential institutional activity).

        Returns:
            Score 0-1 indicating strength of volume anomaly
        """
        if index < self.volume_ma_period:
            return 0.0

        # Get volume window
        window = df.iloc[max(0, index - self.volume_ma_period):index]
        current_volume = df.iloc[index]["volume"]

        # Calculate z-score
        volumes = window["volume"].values
        if len(volumes) < 2:
            return 0.0

        mean_vol = volumes.mean()
        std_vol = volumes.std()

        if std_vol == 0:
            return 0.0

        z = (current_volume - mean_vol) / std_vol

        # Convert to 0-1 score (capped at threshold)
        score = min(1.0, max(0.0, z / self.volume_std_threshold))

        return float(score)

    def detect_absorption(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> Tuple[bool, float]:
        """
        Detect volume absorption (high volume, low price movement).

        Absorption indicates smart money accumulating/distributing quietly.

        Returns:
            (is_absorption, strength)
        """
        if index < 1:
            return False, 0.0

        current = df.iloc[index]
        open_price = current["open"]
        high = current["high"]
        low = current["low"]
        close = current["close"]
        volume = current["volume"]

        # Calculate price range relative to open
        price_range = (high - low) / (open_price + 1e-10)

        # Check for unusual volume
        unusual_vol = self.detect_unusual_volume(df, index)

        # Absorption: high volume + narrow range
        is_absorption = (
            unusual_vol > 0.5 and
            price_range < self.absorption_threshold
        )

        # Strength based on volume spike and range tightness
        strength = unusual_vol * (1.0 - price_range / self.absorption_threshold)
        strength = min(1.0, max(0.0, strength))

        return is_absorption, float(strength)

    def calculate_buy_sell_pressure(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> float:
        """
        Calculate buy/sell pressure from recent bars.

        Uses close position in range as proxy for pressure:
        - Close near high = buying pressure
        - Close near low = selling pressure

        Returns:
            Pressure score from -1 (sell) to +1 (buy)
        """
        if index < self.imbalance_period:
            return 0.0

        # Get recent bars
        window = df.iloc[max(0, index - self.imbalance_period + 1):index + 1]

        pressures = []
        for i in range(len(window)):
            bar = window.iloc[i]
            high = bar["high"]
            low = bar["low"]
            close = bar["close"]
            volume = bar["volume"]

            # Close position in range (0 = low, 1 = high)
            range_size = high - low
            if range_size == 0:
                close_pos = 0.5
            else:
                close_pos = (close - low) / range_size

            # Convert to pressure (-1 to +1)
            pressure = (close_pos - 0.5) * 2.0

            # Weight by volume
            pressures.append(pressure * volume)

        # Volume-weighted average pressure
        total_volume = window["volume"].sum()
        if total_volume == 0:
            return 0.0

        avg_pressure = sum(pressures) / total_volume

        return float(np.clip(avg_pressure, -1.0, 1.0))

    def detect_order_block(
        self,
        df: pd.DataFrame,
        index: int,
        lookback: int = 10,
    ) -> float:
        """
        Detect order blocks (zones where smart money entered).

        Order block characteristics:
        - Strong move away from the zone
        - High volume at the zone
        - Price respect of the zone (support/resistance)

        Returns:
            Order block strength 0-1
        """
        if index < lookback + 1:
            return 0.0

        current_close = df.iloc[index]["close"]
        window = df.iloc[max(0, index - lookback):index + 1]

        # Find strong moves (>2% price change)
        returns = window["close"].pct_change()
        strong_moves = abs(returns) > 0.02

        if not strong_moves.any():
            return 0.0

        # For each strong move, check if current price is testing that zone
        max_strength = 0.0

        for i in range(len(window) - 1):
            if not strong_moves.iloc[i]:
                continue

            bar = window.iloc[i]
            bar_high = bar["high"]
            bar_low = bar["low"]
            bar_volume = bar["volume"]

            # Check if current price is near this zone
            distance_to_zone = min(
                abs(current_close - bar_high) / (current_close + 1e-10),
                abs(current_close - bar_low) / (current_close + 1e-10)
            )

            # Close to zone (within 0.5%)
            if distance_to_zone < 0.005:
                # Calculate strength based on volume and move size
                vol_strength = self.detect_unusual_volume(df, index - lookback + i)
                move_strength = min(1.0, abs(returns.iloc[i]) / 0.05)  # Cap at 5%

                strength = (vol_strength + move_strength) / 2.0
                max_strength = max(max_strength, strength)

        return float(max_strength)

    def analyze_bar(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> SmartMoneySignal:
        """
        Analyze a single bar for smart money activity.

        Args:
            df: DataFrame with OHLCV data
            index: Index of bar to analyze

        Returns:
            SmartMoneySignal with detection results
        """
        if index < self.volume_ma_period:
            return SmartMoneySignal(
                unusual_volume=0.0,
                absorption_detected=False,
                buy_pressure=0.0,
                order_block_strength=0.0,
                institutional_footprint=0.0,
            )

        # 1. Unusual volume detection
        unusual_vol = self.detect_unusual_volume(df, index)

        # 2. Absorption detection
        absorption, absorption_strength = self.detect_absorption(df, index)

        # 3. Buy/sell pressure
        pressure = self.calculate_buy_sell_pressure(df, index)

        # 4. Order block detection
        order_block = self.detect_order_block(df, index)

        # 5. Aggregate institutional footprint
        # Combine signals: high volume + absorption + order blocks = institutional activity
        footprint = (
            unusual_vol * 0.3 +
            absorption_strength * 0.3 +
            order_block * 0.4
        )

        return SmartMoneySignal(
            unusual_volume=unusual_vol,
            absorption_detected=absorption,
            buy_pressure=pressure,
            order_block_strength=order_block,
            institutional_footprint=min(1.0, footprint),
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze entire DataFrame for smart money patterns.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with smart money features:
            - sm_unusual_volume: Volume anomaly score (0-1)
            - sm_absorption: Absorption detected (binary)
            - sm_absorption_strength: Absorption strength (0-1)
            - sm_buy_pressure: Buy/sell pressure (-1 to 1)
            - sm_order_block: Order block strength (0-1)
            - sm_footprint: Institutional footprint (0-1)
            - sm_footprint_ema: Smoothed footprint
        """
        results = []

        for i in range(len(df)):
            signal = self.analyze_bar(df, i)
            results.append(signal)

        # Convert to features
        features = pd.DataFrame(index=df.index)

        features["sm_unusual_volume"] = [r.unusual_volume for r in results]
        features["sm_absorption"] = [1 if r.absorption_detected else 0 for r in results]

        # Calculate absorption strength separately for continuous feature
        absorption_strengths = []
        for i in range(len(df)):
            _, strength = self.detect_absorption(df, i)
            absorption_strengths.append(strength)
        features["sm_absorption_strength"] = absorption_strengths

        features["sm_buy_pressure"] = [r.buy_pressure for r in results]
        features["sm_order_block"] = [r.order_block_strength for r in results]
        features["sm_footprint"] = [r.institutional_footprint for r in results]

        # Add smoothed footprint (EMA)
        features["sm_footprint_ema"] = features["sm_footprint"].ewm(span=10, adjust=False).mean()

        # Directional signals (bullish/bearish institutional activity)
        features["sm_bullish"] = (
            (features["sm_footprint"] > 0.5) &
            (features["sm_buy_pressure"] > 0.2)
        ).astype(int)

        features["sm_bearish"] = (
            (features["sm_footprint"] > 0.5) &
            (features["sm_buy_pressure"] < -0.2)
        ).astype(int)

        return features

    def get_signal_summary(
        self,
        df: pd.DataFrame,
        window: int = 20,
    ) -> Dict[str, any]:
        """
        Get summary of smart money activity over a window.

        Args:
            df: DataFrame with OHLCV data
            window: Lookback window for summary

        Returns:
            Dictionary with activity metrics
        """
        if len(df) < window:
            window = len(df)

        recent_df = df.iloc[-window:]
        features = self.analyze_dataframe(recent_df)

        # Count signals
        absorption_count = int(features["sm_absorption"].sum())
        bullish_count = int(features["sm_bullish"].sum())
        bearish_count = int(features["sm_bearish"].sum())

        # Average metrics
        avg_footprint = float(features["sm_footprint"].mean())
        avg_pressure = float(features["sm_buy_pressure"].mean())
        avg_unusual_vol = float(features["sm_unusual_volume"].mean())

        # Current state
        current_footprint = float(features["sm_footprint_ema"].iloc[-1])
        current_pressure = float(features["sm_buy_pressure"].iloc[-1])

        # Determine bias
        bias = "neutral"
        if current_pressure > 0.3 and current_footprint > 0.5:
            bias = "bullish_institutional"
        elif current_pressure < -0.3 and current_footprint > 0.5:
            bias = "bearish_institutional"

        return {
            "absorption_count": absorption_count,
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
            "avg_footprint": avg_footprint,
            "avg_pressure": avg_pressure,
            "avg_unusual_volume": avg_unusual_vol,
            "current_footprint": current_footprint,
            "current_pressure": current_pressure,
            "bias": bias,
        }


# Convenience function
def analyze_smart_money(
    df: pd.DataFrame,
    volume_ma: int = 20,
    volume_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Quick smart money analysis on DataFrame.

    Usage:
        sm_features = analyze_smart_money(df)
        df = pd.concat([df, sm_features], axis=1)
    """
    detector = SmartMoneyDetector(
        volume_ma_period=volume_ma,
        volume_std_threshold=volume_threshold,
    )
    return detector.analyze_dataframe(df)
