"""
Volume Profile Analysis

Calculates volume distribution across price levels to identify:
- POC (Point of Control): Price level with highest volume
- Value Area (VA): Price range containing 70% of volume
- High Volume Nodes (HVN): Local volume maxima (support/resistance)
- Low Volume Nodes (LVN): Local volume minima (breakout zones)

Reference: "Mind Over Markets" by James Dalton
"""
from __future__ import annotations

from typing import Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from loguru import logger


@dataclass
class VolumeProfileResult:
    """Volume profile analysis result for a single window"""
    poc_price: float  # Point of Control (highest volume price)
    vah: float  # Value Area High (upper bound of 70% volume)
    val: float  # Value Area Low (lower bound of 70% volume)
    hvn_prices: List[float]  # High Volume Nodes (local maxima)
    lvn_prices: List[float]  # Low Volume Nodes (local minima)
    total_volume: float  # Total volume in window
    price_bins: np.ndarray  # Price bin edges
    volume_bins: np.ndarray  # Volume per price bin


class VolumeProfile:
    """
    Volume Profile calculator.

    Analyzes volume distribution across price levels to identify
    key zones for trading decisions.
    """

    def __init__(
        self,
        n_bins: int = 50,
        value_area_pct: float = 0.70,
        hvn_prominence: float = 0.2,
        lvn_prominence: float = 0.1,
    ):
        """
        Initialize Volume Profile calculator.

        Args:
            n_bins: Number of price bins for volume distribution
            value_area_pct: Percentage of volume for Value Area (default 70%)
            hvn_prominence: Minimum prominence for HVN detection (relative to max volume)
            lvn_prominence: Minimum prominence for LVN detection (relative to max volume)
        """
        self.n_bins = n_bins
        self.value_area_pct = value_area_pct
        self.hvn_prominence = hvn_prominence
        self.lvn_prominence = lvn_prominence

    def calculate(
        self,
        df: pd.DataFrame,
        window: int = 100,
        price_col: str = "close",
        volume_col: str = "volume",
    ) -> VolumeProfileResult:
        """
        Calculate volume profile for a window of data.

        Args:
            df: DataFrame with OHLCV data
            window: Number of bars to include in profile
            price_col: Column name for price (default: close)
            volume_col: Column name for volume

        Returns:
            VolumeProfileResult with POC, VA, HVN, LVN
        """
        if len(df) < window:
            raise ValueError(f"DataFrame has {len(df)} rows, need at least {window}")

        # Use last 'window' bars
        window_data = df.iloc[-window:]

        # Get price range
        high_prices = window_data["high"].values
        low_prices = window_data["low"].values
        volumes = window_data[volume_col].values

        price_min = np.min(low_prices)
        price_max = np.max(high_prices)

        # Create price bins
        price_bins = np.linspace(price_min, price_max, self.n_bins + 1)
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2

        # Distribute volume across bins
        # For each bar, distribute its volume across price bins it spans
        volume_distribution = np.zeros(self.n_bins)

        for i in range(len(window_data)):
            bar_high = high_prices[i]
            bar_low = low_prices[i]
            bar_volume = volumes[i]

            # Find bins that overlap with this bar's price range
            overlapping_bins = (bin_centers >= bar_low) & (bin_centers <= bar_high)
            n_overlapping = overlapping_bins.sum()

            if n_overlapping > 0:
                # Distribute volume evenly across overlapping bins
                volume_distribution[overlapping_bins] += bar_volume / n_overlapping

        total_volume = volume_distribution.sum()

        if total_volume == 0:
            # No volume data, return neutral profile
            mid_price = (price_min + price_max) / 2
            return VolumeProfileResult(
                poc_price=mid_price,
                vah=price_max,
                val=price_min,
                hvn_prices=[],
                lvn_prices=[],
                total_volume=0.0,
                price_bins=price_bins,
                volume_bins=volume_distribution,
            )

        # 1. Find POC (Point of Control) - bin with maximum volume
        poc_idx = np.argmax(volume_distribution)
        poc_price = bin_centers[poc_idx]

        # 2. Calculate Value Area (70% of volume around POC)
        vah, val = self._calculate_value_area(
            bin_centers, volume_distribution, poc_idx, total_volume
        )

        # 3. Find High Volume Nodes (local maxima)
        hvn_prices = self._find_hvn(
            bin_centers, volume_distribution, poc_idx
        )

        # 4. Find Low Volume Nodes (local minima)
        lvn_prices = self._find_lvn(
            bin_centers, volume_distribution
        )

        return VolumeProfileResult(
            poc_price=float(poc_price),
            vah=float(vah),
            val=float(val),
            hvn_prices=[float(p) for p in hvn_prices],
            lvn_prices=[float(p) for p in lvn_prices],
            total_volume=float(total_volume),
            price_bins=price_bins,
            volume_bins=volume_distribution,
        )

    def _calculate_value_area(
        self,
        bin_centers: np.ndarray,
        volume_distribution: np.ndarray,
        poc_idx: int,
        total_volume: float,
    ) -> Tuple[float, float]:
        """
        Calculate Value Area High and Low.

        Value Area contains {value_area_pct}% of volume, centered around POC.
        """
        target_volume = total_volume * self.value_area_pct

        # Start from POC and expand outward
        left_idx = poc_idx
        right_idx = poc_idx
        accumulated_volume = volume_distribution[poc_idx]

        # Expand alternately left and right to accumulate volume
        while accumulated_volume < target_volume:
            # Check which side has more volume
            left_volume = volume_distribution[left_idx - 1] if left_idx > 0 else 0
            right_volume = volume_distribution[right_idx + 1] if right_idx < len(bin_centers) - 1 else 0

            if left_volume == 0 and right_volume == 0:
                break

            # Expand to side with more volume
            if left_volume >= right_volume and left_idx > 0:
                left_idx -= 1
                accumulated_volume += left_volume
            elif right_idx < len(bin_centers) - 1:
                right_idx += 1
                accumulated_volume += right_volume
            else:
                break

        vah = bin_centers[right_idx]
        val = bin_centers[left_idx]

        return vah, val

    def _find_hvn(
        self,
        bin_centers: np.ndarray,
        volume_distribution: np.ndarray,
        poc_idx: int,
    ) -> List[float]:
        """
        Find High Volume Nodes (local volume maxima).

        HVNs are support/resistance zones where significant volume accumulated.
        """
        if len(volume_distribution) < 3:
            return []

        # Normalize volume for prominence calculation
        max_volume = volume_distribution.max()
        if max_volume == 0:
            return []

        # Find peaks with minimum prominence
        prominence_threshold = max_volume * self.hvn_prominence
        peaks, properties = find_peaks(
            volume_distribution,
            prominence=prominence_threshold,
            distance=2  # At least 2 bins apart
        )

        # Convert peak indices to prices
        hvn_prices = bin_centers[peaks].tolist()

        # Always include POC if not already in list
        poc_price = bin_centers[poc_idx]
        if poc_price not in hvn_prices:
            hvn_prices.append(poc_price)

        return sorted(hvn_prices)

    def _find_lvn(
        self,
        bin_centers: np.ndarray,
        volume_distribution: np.ndarray,
    ) -> List[float]:
        """
        Find Low Volume Nodes (local volume minima).

        LVNs are potential breakout zones with low volume (fast price movement).
        """
        if len(volume_distribution) < 3:
            return []

        # Invert volume distribution to find minima
        inverted = -volume_distribution

        max_inv = inverted.max()
        if max_inv == 0:
            return []

        # Find valleys with minimum prominence
        prominence_threshold = max_inv * self.lvn_prominence
        valleys, properties = find_peaks(
            inverted,
            prominence=prominence_threshold,
            distance=2
        )

        # Convert valley indices to prices
        lvn_prices = bin_centers[valleys].tolist()

        return sorted(lvn_prices)

    def calculate_rolling(
        self,
        df: pd.DataFrame,
        window: int = 100,
        step: int = 1,
    ) -> pd.DataFrame:
        """
        Calculate rolling volume profile features for entire DataFrame.

        Args:
            df: DataFrame with OHLCV data
            window: Window size for volume profile calculation
            step: Step size between calculations (1 = every bar)

        Returns:
            DataFrame with volume profile features:
            - poc_distance: Distance from current price to POC (%)
            - vah_distance: Distance from current price to VAH (%)
            - val_distance: Distance from current price to VAL (%)
            - in_value_area: Boolean, is price in Value Area
            - closest_hvn_distance: Distance to nearest HVN (%)
            - closest_lvn_distance: Distance to nearest LVN (%)
        """
        features = []

        for i in range(window, len(df) + 1, step):
            window_df = df.iloc[max(0, i - window):i]
            current_price = df.iloc[i - 1]["close"]

            try:
                profile = self.calculate(window_df, window=min(window, len(window_df)))

                # Calculate feature distances (as percentages)
                poc_dist = (current_price - profile.poc_price) / profile.poc_price * 100
                vah_dist = (current_price - profile.vah) / profile.vah * 100
                val_dist = (current_price - profile.val) / profile.val * 100
                in_va = 1 if profile.val <= current_price <= profile.vah else 0

                # Find closest HVN and LVN
                closest_hvn = None
                if profile.hvn_prices:
                    closest_hvn = min(profile.hvn_prices, key=lambda p: abs(p - current_price))
                    closest_hvn_dist = (current_price - closest_hvn) / closest_hvn * 100
                else:
                    closest_hvn_dist = 0.0

                closest_lvn = None
                if profile.lvn_prices:
                    closest_lvn = min(profile.lvn_prices, key=lambda p: abs(p - current_price))
                    closest_lvn_dist = (current_price - closest_lvn) / closest_lvn * 100
                else:
                    closest_lvn_dist = 0.0

                features.append({
                    "poc_distance": poc_dist,
                    "vah_distance": vah_dist,
                    "val_distance": val_dist,
                    "in_value_area": in_va,
                    "closest_hvn_distance": closest_hvn_dist,
                    "closest_lvn_distance": closest_lvn_dist,
                })

            except Exception as e:
                logger.warning(f"Volume profile calculation failed at index {i}: {e}")
                # Fill with neutral values
                features.append({
                    "poc_distance": 0.0,
                    "vah_distance": 0.0,
                    "val_distance": 0.0,
                    "in_value_area": 0,
                    "closest_hvn_distance": 0.0,
                    "closest_lvn_distance": 0.0,
                })

        # Create DataFrame with proper index alignment
        result_df = pd.DataFrame(features)

        # Align with original DataFrame index
        # Pad beginning with NaN for warmup period
        padding = pd.DataFrame(
            np.nan,
            index=range(window - 1),
            columns=result_df.columns
        )

        result_df = pd.concat([padding, result_df], ignore_index=True)
        result_df.index = df.index[:len(result_df)]

        return result_df
