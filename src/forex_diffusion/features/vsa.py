"""
Volume Spread Analysis (VSA)

Analyzes the relationship between volume and price spread to identify:
- Accumulation/Distribution patterns
- Buying/Selling Climax
- No Demand/No Supply zones
- Effort vs Result analysis

Reference: "Master the Markets" by Tom Williams
"""
from __future__ import annotations

from typing import Dict
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class VSASignal(Enum):
    """VSA signal types"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    BUYING_CLIMAX = "buying_climax"
    SELLING_CLIMAX = "selling_climax"
    NO_DEMAND = "no_demand"
    NO_SUPPLY = "no_supply"
    UPTHRUST = "upthrust"
    SPRING = "spring"
    NEUTRAL = "neutral"


@dataclass
class VSABar:
    """VSA analysis for a single bar"""
    signal: VSASignal
    strength: float  # 0-1, strength of signal
    volume_ratio: float  # Volume relative to average
    spread_ratio: float  # Spread relative to average
    close_position: float  # 0-1, where close is in the range (0=low, 1=high)


class VSAAnalyzer:
    """
    Volume Spread Analysis analyzer.

    Identifies professional money movement patterns by analyzing
    the relationship between volume and price spread.
    """

    def __init__(
        self,
        volume_ma_period: int = 20,
        spread_ma_period: int = 20,
        volume_threshold_high: float = 1.5,  # High volume = 1.5x average
        volume_threshold_ultra: float = 2.0,  # Ultra high = 2.0x average
        volume_threshold_low: float = 0.7,   # Low volume = 0.7x average
        spread_threshold_narrow: float = 0.7,  # Narrow spread = 0.7x average
        spread_threshold_wide: float = 1.5,    # Wide spread = 1.5x average
    ):
        """
        Initialize VSA analyzer.

        Args:
            volume_ma_period: Period for volume moving average
            spread_ma_period: Period for spread moving average
            volume_threshold_high: Threshold for high volume detection
            volume_threshold_ultra: Threshold for ultra-high volume
            volume_threshold_low: Threshold for low volume detection
            spread_threshold_narrow: Threshold for narrow spread
            spread_threshold_wide: Threshold for wide spread
        """
        self.volume_ma_period = volume_ma_period
        self.spread_ma_period = spread_ma_period
        self.vol_high = volume_threshold_high
        self.vol_ultra = volume_threshold_ultra
        self.vol_low = volume_threshold_low
        self.spread_narrow = spread_threshold_narrow
        self.spread_wide = spread_threshold_wide

    def analyze_bar(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> VSABar:
        """
        Analyze a single bar for VSA signals.

        Args:
            df: DataFrame with OHLCV data
            index: Index of bar to analyze

        Returns:
            VSABar with signal and strength
        """
        if index < max(self.volume_ma_period, self.spread_ma_period):
            return VSABar(
                signal=VSASignal.NEUTRAL,
                strength=0.0,
                volume_ratio=1.0,
                spread_ratio=1.0,
                close_position=0.5,
            )

        # Current bar data
        open_price = df.iloc[index]["open"]
        high = df.iloc[index]["high"]
        low = df.iloc[index]["low"]
        close = df.iloc[index]["close"]
        volume = df.iloc[index]["volume"]

        # Previous bar data
        prev_close = df.iloc[index - 1]["close"]
        prev_high = df.iloc[index - 1]["high"]
        prev_low = df.iloc[index - 1]["low"]

        # Calculate metrics
        spread = high - low
        close_position = (close - low) / (spread + 1e-10)  # 0-1 scale

        # Calculate averages
        lookback = df.iloc[max(0, index - self.volume_ma_period):index]
        avg_volume = lookback["volume"].mean()
        avg_spread = (lookback["high"] - lookback["low"]).mean()

        # Volume and spread ratios
        vol_ratio = volume / (avg_volume + 1e-10)
        spread_ratio = spread / (avg_spread + 1e-10)

        # Determine bar direction
        is_up_bar = close > open_price
        is_down_bar = close < open_price

        # Volume classification
        is_high_volume = vol_ratio >= self.vol_high
        is_ultra_high_volume = vol_ratio >= self.vol_ultra
        is_low_volume = vol_ratio <= self.vol_low

        # Spread classification
        is_narrow_spread = spread_ratio <= self.spread_narrow
        is_wide_spread = spread_ratio >= self.spread_wide

        # VSA Signal Detection
        signal = VSASignal.NEUTRAL
        strength = 0.0

        # 1. BUYING CLIMAX: Up bar, ultra-high volume, wide spread, close near high
        # Indicates end of uptrend (professionals selling to public)
        if (is_up_bar and is_ultra_high_volume and is_wide_spread and
            close_position > 0.7 and close > prev_close):
            signal = VSASignal.BUYING_CLIMAX
            strength = min(1.0, vol_ratio / self.vol_ultra)

        # 2. SELLING CLIMAX: Down bar, ultra-high volume, wide spread, close near low
        # Indicates end of downtrend (professionals buying from public)
        elif (is_down_bar and is_ultra_high_volume and is_wide_spread and
              close_position < 0.3 and close < prev_close):
            signal = VSASignal.SELLING_CLIMAX
            strength = min(1.0, vol_ratio / self.vol_ultra)

        # 3. ACCUMULATION: Down bar, high volume, narrow spread, close mid-high
        # Professionals absorbing supply
        elif (is_down_bar and is_high_volume and is_narrow_spread and
              close_position > 0.4):
            signal = VSASignal.ACCUMULATION
            strength = vol_ratio / self.vol_high * (1.0 - spread_ratio)

        # 4. DISTRIBUTION: Up bar, high volume, narrow spread, close mid-low
        # Professionals distributing to public
        elif (is_up_bar and is_high_volume and is_narrow_spread and
              close_position < 0.6):
            signal = VSASignal.DISTRIBUTION
            strength = vol_ratio / self.vol_high * (1.0 - spread_ratio)

        # 5. NO DEMAND: Up bar, low volume, narrow spread
        # Lack of buying interest (bearish sign)
        elif is_up_bar and is_low_volume and is_narrow_spread:
            signal = VSASignal.NO_DEMAND
            strength = (1.0 - vol_ratio) * (1.0 - spread_ratio)

        # 6. NO SUPPLY: Down bar, low volume, narrow spread
        # Lack of selling interest (bullish sign)
        elif is_down_bar and is_low_volume and is_narrow_spread:
            signal = VSASignal.NO_SUPPLY
            strength = (1.0 - vol_ratio) * (1.0 - spread_ratio)

        # 7. UPTHRUST: Bar closes near low, after pushing into previous resistance
        # False breakout upward (bearish)
        elif high > prev_high and close < (high - spread * 0.5) and is_high_volume:
            signal = VSASignal.UPTHRUST
            strength = vol_ratio / self.vol_high * (1.0 - close_position)

        # 8. SPRING: Bar closes near high, after pushing below previous support
        # False breakout downward (bullish)
        elif low < prev_low and close > (low + spread * 0.5) and is_high_volume:
            signal = VSASignal.SPRING
            strength = vol_ratio / self.vol_high * close_position

        return VSABar(
            signal=signal,
            strength=float(strength),
            volume_ratio=float(vol_ratio),
            spread_ratio=float(spread_ratio),
            close_position=float(close_position),
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze entire DataFrame for VSA signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with VSA features:
            - vsa_signal_[type]: Binary indicators for each signal type
            - vsa_strength: Overall signal strength
            - vsa_volume_ratio: Volume relative to average
            - vsa_spread_ratio: Spread relative to average
            - vsa_close_position: Close position in bar range
            - vsa_bullish_score: Aggregate bullish VSA score
            - vsa_bearish_score: Aggregate bearish VSA score
        """
        results = []

        for i in range(len(df)):
            bar = self.analyze_bar(df, i)
            results.append(bar)

        # Convert to features
        features = pd.DataFrame(index=df.index)

        # Binary signal indicators
        for signal_type in VSASignal:
            col_name = f"vsa_{signal_type.value}"
            features[col_name] = [
                1 if r.signal == signal_type else 0
                for r in results
            ]

        # Continuous features
        features["vsa_strength"] = [r.strength for r in results]
        features["vsa_volume_ratio"] = [r.volume_ratio for r in results]
        features["vsa_spread_ratio"] = [r.spread_ratio for r in results]
        features["vsa_close_position"] = [r.close_position for r in results]

        # Aggregate bullish/bearish scores
        bullish_signals = [
            VSASignal.SELLING_CLIMAX,
            VSASignal.ACCUMULATION,
            VSASignal.NO_SUPPLY,
            VSASignal.SPRING,
        ]

        bearish_signals = [
            VSASignal.BUYING_CLIMAX,
            VSASignal.DISTRIBUTION,
            VSASignal.NO_DEMAND,
            VSASignal.UPTHRUST,
        ]

        features["vsa_bullish_score"] = [
            r.strength if r.signal in bullish_signals else 0.0
            for r in results
        ]

        features["vsa_bearish_score"] = [
            r.strength if r.signal in bearish_signals else 0.0
            for r in results
        ]

        # Add smoothed scores (EMA)
        features["vsa_bullish_ema"] = features["vsa_bullish_score"].ewm(span=10, adjust=False).mean()
        features["vsa_bearish_ema"] = features["vsa_bearish_score"].ewm(span=10, adjust=False).mean()

        return features

    def get_signal_summary(
        self,
        df: pd.DataFrame,
        window: int = 20,
    ) -> Dict[str, any]:
        """
        Get summary of VSA signals over a window.

        Args:
            df: DataFrame with OHLCV data
            window: Lookback window for summary

        Returns:
            Dictionary with signal counts and dominant signal
        """
        if len(df) < window:
            window = len(df)

        recent_df = df.iloc[-window:]
        features = self.analyze_dataframe(recent_df)

        # Count each signal type
        signal_counts = {}
        for signal_type in VSASignal:
            col_name = f"vsa_{signal_type.value}"
            if col_name in features.columns:
                signal_counts[signal_type.value] = int(features[col_name].sum())

        # Get dominant signal
        dominant = max(signal_counts.items(), key=lambda x: x[1])

        # Calculate average strength
        avg_strength = float(features["vsa_strength"].mean())

        # Current bullish/bearish bias
        bullish_score = float(features["vsa_bullish_ema"].iloc[-1])
        bearish_score = float(features["vsa_bearish_ema"].iloc[-1])

        bias = "neutral"
        if bullish_score > bearish_score * 1.2:
            bias = "bullish"
        elif bearish_score > bullish_score * 1.2:
            bias = "bearish"

        return {
            "signal_counts": signal_counts,
            "dominant_signal": dominant[0],
            "dominant_count": dominant[1],
            "avg_strength": avg_strength,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "bias": bias,
        }


# Convenience function
def analyze_vsa(
    df: pd.DataFrame,
    volume_ma: int = 20,
    spread_ma: int = 20,
) -> pd.DataFrame:
    """
    Quick VSA analysis on DataFrame.

    Usage:
        vsa_features = analyze_vsa(df)
        df = pd.concat([df, vsa_features], axis=1)
    """
    analyzer = VSAAnalyzer(
        volume_ma_period=volume_ma,
        spread_ma_period=spread_ma,
    )
    return analyzer.analyze_dataframe(df)
