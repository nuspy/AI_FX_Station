"""
Adaptive Window Sizing for Regime Detection

Dynamically adjusts the lookback window for regime detection based on
current market conditions (volatility, momentum, microstructure).

In high volatility → shorter window (more reactive)
In low volatility → longer window (more stable)

Reference: "Adaptive Filtering" by Haykin (2002)
"""
from __future__ import annotations

from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class WindowConfig:
    """Adaptive window configuration"""
    base_window: int = 100  # Default lookback period
    min_window: int = 50    # Minimum window size
    max_window: int = 200   # Maximum window size

    # Adjustment parameters
    volatility_weight: float = 0.5
    momentum_weight: float = 0.3
    range_weight: float = 0.2

    # Smoothing
    smoothing_alpha: float = 0.3  # EMA smoothing for window changes


@dataclass
class MarketConditions:
    """Current market conditions for window sizing"""
    volatility: float          # Recent volatility (std of returns)
    normalized_volatility: float  # Volatility relative to historical
    momentum: float            # Directional momentum strength
    range_expansion: float     # Average true range trend
    microstructure_quality: float  # Price stability metric


class AdaptiveWindowSizer:
    """
    Adaptively sizes regime detection window based on market conditions.

    Workflow:
    1. Calculate current market metrics (volatility, momentum, range)
    2. Normalize metrics relative to historical distribution
    3. Compute adaptive window size via weighted formula
    4. Smooth window changes to avoid jumps
    5. Apply constraints (min/max bounds)
    """

    def __init__(self, config: Optional[WindowConfig] = None):
        """
        Initialize adaptive window sizer.

        Args:
            config: WindowConfig with parameters
        """
        self.config = config or WindowConfig()

        # Historical state for normalization
        self.volatility_history: list = []
        self.momentum_history: list = []
        self.range_history: list = []

        # Smoothed window size
        self.current_window: float = self.config.base_window
        self.previous_window: int = self.config.base_window

        # History lookback for normalization
        self.normalization_lookback = 500

    def calculate_adaptive_window(
        self,
        df: pd.DataFrame,
        current_idx: Optional[int] = None,
    ) -> int:
        """
        Calculate adaptive window size for current market conditions.

        Args:
            df: OHLCV DataFrame
            current_idx: Index to calculate from (default: latest)

        Returns:
            Adaptive window size (integer)
        """
        if current_idx is None:
            current_idx = len(df) - 1

        # Need minimum history for calculation
        if current_idx < 50:
            logger.warning(f"Insufficient history at idx {current_idx}, using base window")
            return self.config.base_window

        # Step 1: Calculate current market conditions
        conditions = self._calculate_market_conditions(df, current_idx)

        # Step 2: Compute raw adaptive window
        raw_window = self._compute_window_from_conditions(conditions)

        # Step 3: Smooth window changes (EMA)
        smoothed_window = (
            self.config.smoothing_alpha * raw_window +
            (1 - self.config.smoothing_alpha) * self.current_window
        )

        # Step 4: Apply constraints
        final_window = int(np.clip(
            smoothed_window,
            self.config.min_window,
            self.config.max_window,
        ))

        # Update state
        self.previous_window = int(self.current_window)
        self.current_window = smoothed_window

        # Log significant changes
        if abs(final_window - self.previous_window) > 10:
            logger.info(
                f"Window size adjusted: {self.previous_window} → {final_window} "
                f"(vol={conditions.normalized_volatility:.2f}, "
                f"momentum={conditions.momentum:.2f})"
            )

        return final_window

    def _calculate_market_conditions(
        self,
        df: pd.DataFrame,
        current_idx: int,
        lookback: int = 50,
    ) -> MarketConditions:
        """
        Calculate current market conditions.

        Args:
            df: OHLCV DataFrame
            current_idx: Current index
            lookback: Lookback period for metrics

        Returns:
            MarketConditions with calculated metrics
        """
        # Extract recent data
        start_idx = max(0, current_idx - lookback)
        recent = df.iloc[start_idx:current_idx + 1]

        if len(recent) < 10:
            # Insufficient data, return neutral conditions
            return MarketConditions(
                volatility=0.01,
                normalized_volatility=0.5,
                momentum=0.0,
                range_expansion=0.0,
                microstructure_quality=0.5,
            )

        # 1. Volatility (std of log returns)
        returns = np.log(recent["close"] / recent["close"].shift(1)).dropna()
        volatility = float(returns.std())

        # Normalize volatility relative to historical
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > self.normalization_lookback:
            self.volatility_history.pop(0)

        if len(self.volatility_history) > 20:
            vol_pct_rank = (
                sum(1 for v in self.volatility_history if v < volatility) /
                len(self.volatility_history)
            )
        else:
            vol_pct_rank = 0.5  # Neutral if insufficient history

        # 2. Momentum (directional strength)
        # Use linear regression slope of close prices
        x = np.arange(len(recent))
        y = recent["close"].values

        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            # Normalize by price level and volatility
            momentum = slope / (recent["close"].iloc[-1] * volatility + 1e-10)
            momentum = float(np.tanh(momentum * 10))  # Squash to [-1, 1]
        else:
            momentum = 0.0

        self.momentum_history.append(abs(momentum))
        if len(self.momentum_history) > self.normalization_lookback:
            self.momentum_history.pop(0)

        # 3. Range expansion (ATR trend)
        # Calculate ATR
        high_low = recent["high"] - recent["low"]
        high_close = abs(recent["high"] - recent["close"].shift(1))
        low_close = abs(recent["low"] - recent["close"].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=1).mean()

        if len(atr) > 2:
            # Trend of ATR (expanding or contracting)
            atr_slope = (atr.iloc[-1] - atr.iloc[-10]) / (atr.iloc[-10] + 1e-10)
            range_expansion = float(np.tanh(atr_slope * 5))  # Squash to [-1, 1]
        else:
            range_expansion = 0.0

        self.range_history.append(abs(range_expansion))
        if len(self.range_history) > self.normalization_lookback:
            self.range_history.pop(0)

        # 4. Microstructure quality (price stability)
        # Measure consistency of price changes
        # High quality = low noise, clear trends
        # Low quality = choppy, noisy

        if len(returns) > 5:
            # Hurst exponent approximation
            # H > 0.5: trending (high quality)
            # H < 0.5: mean-reverting (low quality)
            # H = 0.5: random walk (neutral)

            lags = [2, 4, 8]
            variances = []

            for lag in lags:
                if len(returns) > lag:
                    lagged_var = ((returns - returns.shift(lag)).dropna()).var()
                    variances.append(lagged_var)

            if len(variances) >= 2:
                # Log-log regression
                log_lags = np.log(lags[:len(variances)])
                log_vars = np.log(variances)
                hurst_approx = np.polyfit(log_lags, log_vars, 1)[0] / 2

                # Normalize to [0, 1] where 0.5 = random walk
                microstructure_quality = float(np.clip((hurst_approx - 0.3) / 0.4, 0, 1))
            else:
                microstructure_quality = 0.5
        else:
            microstructure_quality = 0.5

        return MarketConditions(
            volatility=volatility,
            normalized_volatility=float(vol_pct_rank),
            momentum=float(abs(momentum)),
            range_expansion=range_expansion,
            microstructure_quality=microstructure_quality,
        )

    def _compute_window_from_conditions(
        self,
        conditions: MarketConditions,
    ) -> float:
        """
        Compute window size from market conditions.

        Logic:
        - High volatility → shorter window (reactive)
        - Low volatility → longer window (stable)
        - High momentum → moderate window (trending)
        - Range expansion → shorter window (breakout)

        Args:
            conditions: MarketConditions

        Returns:
            Raw window size (float, before constraints)
        """
        base = self.config.base_window

        # Volatility component
        # High vol (percentile > 0.8) → reduce window by up to 50%
        # Low vol (percentile < 0.2) → increase window by up to 100%
        vol_factor = 1.0 + (0.5 - conditions.normalized_volatility)  # Range: [0.5, 1.5]

        # Momentum component
        # High momentum → moderate window (don't overreact)
        # Low momentum → longer window (more stable)
        momentum_factor = 1.0 + 0.3 * (1 - conditions.momentum)  # Range: [1.0, 1.3]

        # Range expansion component
        # Expanding range → shorter window (capture change)
        # Contracting range → longer window (wait for clarity)
        if conditions.range_expansion > 0:
            range_factor = 1.0 - 0.2 * min(conditions.range_expansion, 1.0)
        else:
            range_factor = 1.0 + 0.2 * min(abs(conditions.range_expansion), 1.0)

        # Weighted combination
        total_weight = (
            self.config.volatility_weight +
            self.config.momentum_weight +
            self.config.range_weight
        )

        combined_factor = (
            self.config.volatility_weight * vol_factor +
            self.config.momentum_weight * momentum_factor +
            self.config.range_weight * range_factor
        ) / total_weight

        raw_window = base * combined_factor

        logger.debug(
            f"Window factors: vol={vol_factor:.2f}, momentum={momentum_factor:.2f}, "
            f"range={range_factor:.2f} → combined={combined_factor:.2f}, "
            f"raw_window={raw_window:.1f}"
        )

        return raw_window

    def get_window_statistics(self) -> dict:
        """Get statistics about window sizing decisions"""
        return {
            "current_window": int(self.current_window),
            "previous_window": self.previous_window,
            "base_window": self.config.base_window,
            "min_window": self.config.min_window,
            "max_window": self.config.max_window,
            "volatility_samples": len(self.volatility_history),
            "momentum_samples": len(self.momentum_history),
            "range_samples": len(self.range_history),
        }

    def reset(self) -> None:
        """Reset adaptive state"""
        self.volatility_history.clear()
        self.momentum_history.clear()
        self.range_history.clear()
        self.current_window = self.config.base_window
        self.previous_window = self.config.base_window
        logger.info("Adaptive window sizer reset to base state")


# Convenience function
def calculate_adaptive_window(
    df: pd.DataFrame,
    base_window: int = 100,
    min_window: int = 50,
    max_window: int = 200,
) -> int:
    """
    Quick adaptive window calculation.

    Args:
        df: OHLCV DataFrame
        base_window: Default window size
        min_window: Minimum allowed window
        max_window: Maximum allowed window

    Returns:
        Adaptive window size
    """
    config = WindowConfig(
        base_window=base_window,
        min_window=min_window,
        max_window=max_window,
    )

    sizer = AdaptiveWindowSizer(config)
    return sizer.calculate_adaptive_window(df)
