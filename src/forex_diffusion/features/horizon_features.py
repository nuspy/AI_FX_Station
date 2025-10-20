"""
Horizon-Specific Feature Engineering

Generates features optimized for different prediction horizons.
Short horizons need fast indicators, long horizons need slow indicators.

Example:
- H1 (5 bars): RSI(7), MACD(6,12,5), BB(10), fast momentum
- H2 (20 bars): RSI(14), MACD(12,26,9), BB(20), medium momentum
- H3 (50 bars): RSI(21), MACD(24,52,18), BB(50), slow momentum

Reference: "Evidence-Based Technical Analysis" by Aronson (2007)
"""
from __future__ import annotations

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class HorizonConfig:
    """Configuration for a specific horizon"""
    horizon_bars: int
    horizon_name: str

    # Indicator periods
    rsi_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    bb_period: int
    bb_std: float
    ema_period: int
    atr_period: int

    # Momentum lookback
    momentum_period: int
    roc_period: int  # Rate of change

    # Volume analysis
    volume_ma_period: int


# Pre-defined horizon configurations
HORIZON_CONFIGS = {
    "short": HorizonConfig(
        horizon_bars=5,
        horizon_name="short",
        rsi_period=7,
        macd_fast=6,
        macd_slow=12,
        macd_signal=5,
        bb_period=10,
        bb_std=2.0,
        ema_period=8,
        atr_period=7,
        momentum_period=5,
        roc_period=5,
        volume_ma_period=10,
    ),
    "medium": HorizonConfig(
        horizon_bars=20,
        horizon_name="medium",
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bb_period=20,
        bb_std=2.0,
        ema_period=20,
        atr_period=14,
        momentum_period=14,
        roc_period=12,
        volume_ma_period=20,
    ),
    "long": HorizonConfig(
        horizon_bars=50,
        horizon_name="long",
        rsi_period=21,
        macd_fast=24,
        macd_slow=52,
        macd_signal=18,
        bb_period=50,
        bb_std=2.0,
        ema_period=50,
        atr_period=21,
        momentum_period=28,
        roc_period=25,
        volume_ma_period=50,
    ),
}


class HorizonFeatureEngineer:
    """
    Generates horizon-specific technical indicators.

    Features are optimized for the prediction horizon:
    - Short horizon: Fast, responsive indicators
    - Long horizon: Slow, stable indicators
    """

    def __init__(self, horizons: Optional[List[str]] = None):
        """
        Initialize horizon feature engineer.

        Args:
            horizons: List of horizon names (default: ["short", "medium", "long"])
        """
        self.horizons = horizons or ["short", "medium", "long"]
        self.configs = {h: HORIZON_CONFIGS[h] for h in self.horizons}

    def generate_features(
        self,
        df: pd.DataFrame,
        include_base: bool = True,
    ) -> pd.DataFrame:
        """
        Generate all horizon-specific features.

        Args:
            df: OHLCV DataFrame
            include_base: Include non-horizon-specific base features

        Returns:
            DataFrame with horizon-specific features
        """
        result = df.copy()

        # Generate features for each horizon
        for horizon_name, config in self.configs.items():
            logger.info(f"Generating features for horizon: {horizon_name}")

            horizon_feats = self._generate_horizon_features(df, config)

            # Prefix with horizon name
            horizon_feats = horizon_feats.add_prefix(f"{horizon_name}_")

            # Merge with result
            result = pd.concat([result, horizon_feats], axis=1)

        # Optionally add base features (not horizon-specific)
        if include_base:
            base_feats = self._generate_base_features(df)
            result = pd.concat([result, base_feats], axis=1)

        logger.info(
            f"Generated {len(result.columns) - len(df.columns)} horizon-specific features"
        )

        return result

    def _generate_horizon_features(
        self,
        df: pd.DataFrame,
        config: HorizonConfig,
    ) -> pd.DataFrame:
        """
        Generate features for a specific horizon.

        Args:
            df: OHLCV DataFrame
            config: Horizon configuration

        Returns:
            DataFrame with features for this horizon
        """
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df.get("volume", pd.Series(1, index=df.index))

        # 1. RSI (Relative Strength Index)
        rsi = self._calculate_rsi(close, config.rsi_period)
        features["rsi"] = rsi
        features["rsi_overbought"] = (rsi > 70).astype(int)
        features["rsi_oversold"] = (rsi < 30).astype(int)

        # 2. MACD (Moving Average Convergence Divergence)
        macd, signal, histogram = self._calculate_macd(
            close,
            config.macd_fast,
            config.macd_slow,
            config.macd_signal,
        )
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_histogram"] = histogram
        features["macd_cross_bullish"] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)
        features["macd_cross_bearish"] = ((macd < signal) & (macd.shift(1) >= signal.shift(1))).astype(int)

        # 3. Bollinger Bands
        bb_middle, bb_upper, bb_lower = self._calculate_bollinger_bands(
            close,
            config.bb_period,
            config.bb_std,
        )
        features["bb_middle"] = bb_middle
        features["bb_upper"] = bb_upper
        features["bb_lower"] = bb_lower
        features["bb_width"] = (bb_upper - bb_lower) / bb_middle
        features["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)  # 0-1
        features["bb_squeeze"] = (features["bb_width"] < features["bb_width"].rolling(20).quantile(0.2)).astype(int)

        # 4. EMA (Exponential Moving Average)
        ema = close.ewm(span=config.ema_period, adjust=False).mean()
        features["ema"] = ema
        features["price_vs_ema"] = (close - ema) / ema
        features["ema_slope"] = (ema - ema.shift(5)) / ema.shift(5)

        # 5. ATR (Average True Range)
        atr = self._calculate_atr(df, config.atr_period)
        features["atr"] = atr
        features["atr_normalized"] = atr / close

        # 6. Momentum
        momentum = close.diff(config.momentum_period) / close.shift(config.momentum_period)
        features["momentum"] = momentum

        # 7. Rate of Change (ROC)
        roc = (close - close.shift(config.roc_period)) / close.shift(config.roc_period)
        features["roc"] = roc

        # 8. Volume features
        vol_ma = volume.rolling(config.volume_ma_period).mean()
        features["volume_ma"] = vol_ma
        features["volume_ratio"] = volume / vol_ma
        features["volume_surge"] = (volume > vol_ma * 1.5).astype(int)

        # 9. Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(df, config.momentum_period)
        features["stoch_k"] = stoch_k
        features["stoch_d"] = stoch_d
        features["stoch_overbought"] = (stoch_k > 80).astype(int)
        features["stoch_oversold"] = (stoch_k < 20).astype(int)

        # 10. Trend strength
        # ADX-like indicator
        adx = self._calculate_adx(df, config.momentum_period)
        features["adx"] = adx
        features["trending"] = (adx > 25).astype(int)

        return features

    def _generate_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate base features (not horizon-specific).

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with base features
        """
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # Price relationships
        features["high_low_range"] = (high - low) / close
        features["close_open_diff"] = (close - open_) / open_
        features["upper_shadow"] = (high - np.maximum(open_, close)) / close
        features["lower_shadow"] = (np.minimum(open_, close) - low) / close

        # Candle body
        features["body_size"] = abs(close - open_) / open_
        features["is_green"] = (close > open_).astype(int)

        return features

    # Technical indicator calculations

    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        close: pd.Series,
        fast: int,
        slow: int,
        signal: int,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return macd, signal_line, histogram

    def _calculate_bollinger_bands(
        self,
        close: pd.Series,
        period: int,
        std_multiplier: float,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = middle + (std * std_multiplier)
        lower = middle - (std * std_multiplier)

        return middle, upper, lower

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def _calculate_stochastic(
        self,
        df: pd.DataFrame,
        period: int,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()

        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(3).mean()  # 3-period smoothing

        return stoch_k, stoch_d

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # True range
        tr = self._calculate_atr(df, 1)

        # Smooth DM and TR
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()
        tr_smooth = tr.ewm(span=period, adjust=False).mean()

        # Directional indicators
        plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
        minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated"""
        feature_names = []

        for horizon_name in self.horizons:
            # RSI features
            feature_names.extend([
                f"{horizon_name}_rsi",
                f"{horizon_name}_rsi_overbought",
                f"{horizon_name}_rsi_oversold",
            ])

            # MACD features
            feature_names.extend([
                f"{horizon_name}_macd",
                f"{horizon_name}_macd_signal",
                f"{horizon_name}_macd_histogram",
                f"{horizon_name}_macd_cross_bullish",
                f"{horizon_name}_macd_cross_bearish",
            ])

            # Bollinger Bands features
            feature_names.extend([
                f"{horizon_name}_bb_middle",
                f"{horizon_name}_bb_upper",
                f"{horizon_name}_bb_lower",
                f"{horizon_name}_bb_width",
                f"{horizon_name}_bb_position",
                f"{horizon_name}_bb_squeeze",
            ])

            # EMA features
            feature_names.extend([
                f"{horizon_name}_ema",
                f"{horizon_name}_price_vs_ema",
                f"{horizon_name}_ema_slope",
            ])

            # ATR features
            feature_names.extend([
                f"{horizon_name}_atr",
                f"{horizon_name}_atr_normalized",
            ])

            # Momentum features
            feature_names.extend([
                f"{horizon_name}_momentum",
                f"{horizon_name}_roc",
            ])

            # Volume features
            feature_names.extend([
                f"{horizon_name}_volume_ma",
                f"{horizon_name}_volume_ratio",
                f"{horizon_name}_volume_surge",
            ])

            # Stochastic features
            feature_names.extend([
                f"{horizon_name}_stoch_k",
                f"{horizon_name}_stoch_d",
                f"{horizon_name}_stoch_overbought",
                f"{horizon_name}_stoch_oversold",
            ])

            # ADX features
            feature_names.extend([
                f"{horizon_name}_adx",
                f"{horizon_name}_trending",
            ])

        # Base features
        feature_names.extend([
            "high_low_range",
            "close_open_diff",
            "upper_shadow",
            "lower_shadow",
            "body_size",
            "is_green",
        ])

        return feature_names


# Convenience function
def generate_horizon_features(
    df: pd.DataFrame,
    horizons: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Quick horizon feature generation.

    Args:
        df: OHLCV DataFrame
        horizons: List of horizon names (default: ["short", "medium", "long"])

    Returns:
        DataFrame with horizon-specific features
    """
    engineer = HorizonFeatureEngineer(horizons=horizons)
    return engineer.generate_features(df, include_base=True)
