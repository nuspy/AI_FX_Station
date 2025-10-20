"""
Incremental Feature Update System for real-time feature computation.

Enables efficient updating of features when new candles arrive without
recomputing the entire feature set from scratch.
"""
from __future__ import annotations

import time
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger

from .feature_cache import get_feature_cache


class IncrementalUpdateError(Exception):
    """Raised when incremental update fails."""
    pass


class FeatureWindow:
    """
    Manages a rolling window of features with incremental updates.

    Tracks the state of computed features and allows efficient updates
    when new data becomes available.
    """

    def __init__(self, window_size: int = 512, symbol: str = "", timeframe: str = ""):
        self.window_size = window_size
        self.symbol = symbol
        self.timeframe = timeframe

        # Feature storage
        self._features_df: Optional[pd.DataFrame] = None
        self._last_update_ts: Optional[int] = None
        self._feature_config: Optional[Dict[str, Any]] = None

        # Incremental computation state
        self._indicator_states: Dict[str, Any] = {}
        self._cached_computations: Dict[str, Any] = {}

        # Update tracking
        self._is_initialized = False
        self._update_count = 0

    def initialize(self, df_candles: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Initialize the feature window with initial data.

        Args:
            df_candles: Initial candle data
            feature_config: Configuration for feature computation

        Returns:
            Computed features dataframe
        """
        self._feature_config = feature_config.copy()

        # Compute initial features
        features_df = self._compute_full_features(df_candles, feature_config)

        # Store in window
        self._features_df = features_df.tail(self.window_size).copy()
        self._last_update_ts = int(df_candles["ts_utc"].iat[-1]) if len(df_candles) > 0 else None

        # Initialize indicator states for incremental updates
        self._initialize_indicator_states(df_candles)

        self._is_initialized = True
        self._update_count = 0

        logger.debug(f"Feature window initialized for {self.symbol} {self.timeframe} "
                    f"with {len(features_df)} features")

        return self._features_df.copy()

    def update(self, new_candles: pd.DataFrame) -> pd.DataFrame:
        """
        Incrementally update features with new candle data.

        Args:
            new_candles: New candle data to incorporate

        Returns:
            Updated features dataframe
        """
        if not self._is_initialized:
            raise IncrementalUpdateError("Feature window not initialized")

        if new_candles.empty:
            return self._features_df.copy()

        start_time = time.time()

        try:
            # Check if we can do incremental update
            if self._can_update_incrementally(new_candles):
                updated_features = self._incremental_update(new_candles)
            else:
                # Fall back to full recomputation
                logger.debug(f"Falling back to full recomputation for {self.symbol} {self.timeframe}")
                updated_features = self._full_recomputation(new_candles)

            # Update window
            self._features_df = updated_features.tail(self.window_size).copy()
            self._last_update_ts = int(new_candles["ts_utc"].iat[-1]) if len(new_candles) > 0 else self._last_update_ts
            self._update_count += 1

            elapsed = time.time() - start_time
            logger.debug(f"Feature update completed for {self.symbol} {self.timeframe} "
                        f"in {elapsed:.3f}s (update #{self._update_count})")

            return self._features_df.copy()

        except Exception as e:
            logger.error(f"Incremental update failed for {self.symbol} {self.timeframe}: {e}")
            raise IncrementalUpdateError(f"Feature update failed: {e}") from e

    def _can_update_incrementally(self, new_candles: pd.DataFrame) -> bool:
        """Check if incremental update is possible."""
        if self._features_df is None or self._last_update_ts is None:
            return False

        # Check timestamp continuity
        first_new_ts = int(new_candles["ts_utc"].iat[0])

        # Allow for small gaps (e.g., missing 1-2 candles)
        timeframe_minutes = self._get_timeframe_minutes()
        max_gap_ms = timeframe_minutes * 60 * 1000 * 3  # 3 candles max gap

        time_gap = first_new_ts - self._last_update_ts

        if time_gap > max_gap_ms:
            logger.debug(f"Time gap too large for incremental update: {time_gap}ms > {max_gap_ms}ms")
            return False

        # Check if new candles count is reasonable for incremental update
        if len(new_candles) > self.window_size // 4:
            logger.debug(f"Too many new candles for incremental update: {len(new_candles)}")
            return False

        return True

    def _incremental_update(self, new_candles: pd.DataFrame) -> pd.DataFrame:
        """Perform incremental feature update."""

        # Get existing data to combine with new data
        existing_window = self._get_computation_window()

        # Combine existing and new data
        combined_data = pd.concat([existing_window, new_candles], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=['ts_utc'], keep='last')
        combined_data = combined_data.sort_values('ts_utc').reset_index(drop=True)

        # Compute features on combined data
        # Note: Some features (like indicators) require historical context,
        # so we compute on the combined dataset but only keep the new portions

        new_features = self._compute_incremental_features(
            combined_data,
            new_candles,
            self._feature_config
        )

        # Combine with existing features
        if self._features_df is not None:
            # Remove overlapping timestamps and append new features
            last_existing_ts = self._last_update_ts
            non_overlapping_new = new_features[
                new_features.index.map(
                    lambda i: combined_data.loc[i, 'ts_utc'] > last_existing_ts
                )
            ]

            updated_features = pd.concat([self._features_df, non_overlapping_new], ignore_index=True)
        else:
            updated_features = new_features

        return updated_features

    def _compute_incremental_features(self, combined_data: pd.DataFrame,
                                    new_candles: pd.DataFrame,
                                    feature_config: Dict[str, Any]) -> pd.DataFrame:
        """Compute features incrementally where possible."""

        from .feature_engineering import relative_ohlc, temporal_features, realized_volatility_feature
        from .feature_utils import coerce_indicator_tfs
        from .indicator_pipeline import compute_indicators  # ISSUE-001b: Use centralized indicator computation

        feats_list = []

        # Compute features that can be done incrementally

        # Relative OHLC - can be computed on just new data
        if feature_config.get("use_relative_ohlc", True):
            # For incremental OHLC, we need some context for relative computation
            context_size = min(50, len(combined_data))
            context_data = combined_data.tail(context_size)
            ohlc_feats = relative_ohlc(context_data)

            # Extract only the new portions
            new_start_idx = len(context_data) - len(new_candles)
            if new_start_idx >= 0:
                new_ohlc_feats = ohlc_feats.tail(len(new_candles)).reset_index(drop=True)
                feats_list.append(new_ohlc_feats)

        # Temporal features - can be computed on new data only
        if feature_config.get("use_temporal_features", True):
            temp_feats = temporal_features(new_candles)
            feats_list.append(temp_feats)

        # Realized volatility - needs historical context
        rv_window = feature_config.get("rv_window", 60)
        if rv_window > 1:
            # Need sufficient history for realized vol computation
            rv_context_size = min(rv_window * 2, len(combined_data))
            rv_context_data = combined_data.tail(rv_context_size)
            rv_feats = realized_volatility_feature(rv_context_data, rv_window)

            # Extract new portions
            new_start_idx = len(rv_context_data) - len(new_candles)
            if new_start_idx >= 0:
                new_rv_feats = rv_feats.tail(len(new_candles)).reset_index(drop=True)
                feats_list.append(new_rv_feats)

        # Indicators - need historical context for proper computation
        indicator_tfs_raw = feature_config.get("indicator_tfs", {})
        indicator_tfs = coerce_indicator_tfs(indicator_tfs_raw)

        # Advanced features
        is_advanced = feature_config.get("advanced", False) or feature_config.get("use_advanced_features", False)

        if is_advanced:
            if not indicator_tfs:
                indicator_tfs = {}

            if feature_config.get("enable_ema_features", False):
                indicator_tfs.setdefault("ema", [self.timeframe])
            if feature_config.get("enable_donchian", False):
                indicator_tfs.setdefault("donchian", [self.timeframe])
            if feature_config.get("enable_keltner", False):
                indicator_tfs.setdefault("keltner", [self.timeframe])
            if feature_config.get("enable_hurst_advanced", False):
                indicator_tfs.setdefault("hurst", [self.timeframe])

        if indicator_tfs:
            # Indicators need sufficient history for proper calculation
            indicator_context_size = min(200, len(combined_data))  # Usually enough for most indicators
            indicator_context_data = combined_data.tail(indicator_context_size)

            # Build indicator config
            ind_cfg = self._build_indicator_config(feature_config, indicator_tfs)

            if ind_cfg:
                indicator_feats = compute_indicators(
                    indicator_context_data,
                    ind_cfg,
                    indicator_tfs,
                    self.timeframe
                )

                # Extract new portions
                new_start_idx = len(indicator_context_data) - len(new_candles)
                if new_start_idx >= 0:
                    new_indicator_feats = indicator_feats.tail(len(new_candles)).reset_index(drop=True)
                    feats_list.append(new_indicator_feats)

        if not feats_list:
            raise IncrementalUpdateError("No features computed in incremental update")

        # Combine features
        new_features = pd.concat(feats_list, axis=1)
        new_features = new_features.replace([np.inf, -np.inf], np.nan)

        return new_features

    def _full_recomputation(self, new_candles: pd.DataFrame) -> pd.DataFrame:
        """Fall back to full feature recomputation."""

        # Get sufficient historical context
        context_window = self._get_computation_window()

        # Combine with new data
        combined_data = pd.concat([context_window, new_candles], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=['ts_utc'], keep='last')
        combined_data = combined_data.sort_values('ts_utc').reset_index(drop=True)

        # Recompute all features
        full_features = self._compute_full_features(combined_data, self._feature_config)

        # Re-initialize indicator states
        self._initialize_indicator_states(combined_data)

        return full_features

    def _compute_full_features(self, df_candles: pd.DataFrame,
                             feature_config: Dict[str, Any]) -> pd.DataFrame:
        """Compute full feature set from scratch."""

        from .feature_engineering import relative_ohlc, temporal_features, realized_volatility_feature
        from .feature_utils import coerce_indicator_tfs
        from .indicator_pipeline import compute_indicators  # ISSUE-001b: Use centralized indicator computation

        feats_list = []

        # Relative OHLC
        if feature_config.get("use_relative_ohlc", True):
            feats_list.append(relative_ohlc(df_candles))

        # Temporal features
        if feature_config.get("use_temporal_features", True):
            feats_list.append(temporal_features(df_candles))

        # Realized volatility
        rv_window = feature_config.get("rv_window", 60)
        if rv_window > 1:
            feats_list.append(realized_volatility_feature(df_candles, rv_window))

        # Indicators
        indicator_tfs_raw = feature_config.get("indicator_tfs", {})
        indicator_tfs = coerce_indicator_tfs(indicator_tfs_raw)

        # Advanced features
        is_advanced = feature_config.get("advanced", False) or feature_config.get("use_advanced_features", False)

        if is_advanced:
            if not indicator_tfs:
                indicator_tfs = {}

            if feature_config.get("enable_ema_features", False):
                indicator_tfs.setdefault("ema", [self.timeframe])
            if feature_config.get("enable_donchian", False):
                indicator_tfs.setdefault("donchian", [self.timeframe])
            if feature_config.get("enable_keltner", False):
                indicator_tfs.setdefault("keltner", [self.timeframe])
            if feature_config.get("enable_hurst_advanced", False):
                indicator_tfs.setdefault("hurst", [self.timeframe])

        if indicator_tfs:
            ind_cfg = self._build_indicator_config(feature_config, indicator_tfs)
            if ind_cfg:
                feats_list.append(compute_indicators(df_candles, ind_cfg, indicator_tfs, self.timeframe))

        if not feats_list:
            raise IncrementalUpdateError("No features configured for computation")

        # Combine features
        feats_df = pd.concat(feats_list, axis=1)
        feats_df = feats_df.replace([np.inf, -np.inf], np.nan)

        return feats_df

    def _build_indicator_config(self, feature_config: Dict[str, Any],
                               indicator_tfs: Dict[str, List[str]]) -> Dict[str, Any]:
        """Build indicator configuration from feature config."""

        ind_cfg = {}

        if "atr" in indicator_tfs:
            ind_cfg["atr"] = {"n": feature_config.get("atr_n", 14)}
        if "rsi" in indicator_tfs:
            ind_cfg["rsi"] = {"n": feature_config.get("rsi_n", 14)}
        if "bollinger" in indicator_tfs:
            ind_cfg["bollinger"] = {"n": feature_config.get("bb_n", 20), "dev": 2.0}
        if "macd" in indicator_tfs:
            ind_cfg["macd"] = {"fast": 12, "slow": 26, "signal": 9}
        if "donchian" in indicator_tfs:
            ind_cfg["donchian"] = {"n": feature_config.get("don_n", 20)}
        if "keltner" in indicator_tfs:
            ind_cfg["keltner"] = {
                "ema": feature_config.get("keltner_ema", 20),
                "atr": feature_config.get("keltner_atr", 10),
                "mult": feature_config.get("keltner_k", 1.5)
            }
        if "hurst" in indicator_tfs:
            ind_cfg["hurst"] = {"window": feature_config.get("hurst_window", 64)}
        if "ema" in indicator_tfs:
            ind_cfg["ema"] = {
                "fast": feature_config.get("ema_fast", 12),
                "slow": feature_config.get("ema_slow", 26)
            }

        return ind_cfg

    def _initialize_indicator_states(self, df_candles: pd.DataFrame):
        """Initialize states for incremental indicator computation."""
        # This could store EMAs, previous ATR values, etc. for efficient updates
        # For now, we'll keep it simple and recompute when needed
        self._indicator_states = {
            "last_close": float(df_candles["close"].iat[-1]) if len(df_candles) > 0 else 0.0,
            "last_volume": float(df_candles["volume"].iat[-1]) if len(df_candles) > 0 else 0.0,
        }

    def _get_computation_window(self) -> pd.DataFrame:
        """Get the data window needed for computation context."""
        # This should return recent candle data for context
        # For now, we'll use a simple approach
        if hasattr(self, '_last_candles_context'):
            return self._last_candles_context
        return pd.DataFrame()

    def _get_timeframe_minutes(self) -> int:
        """Convert timeframe to minutes."""
        tf = self.timeframe.lower()
        if tf.endswith("m"):
            return int(tf[:-1])
        elif tf.endswith("h"):
            return int(tf[:-1]) * 60
        elif tf.endswith("d"):
            return int(tf[:-1]) * 24 * 60
        else:
            return 1  # Default to 1 minute

    def get_latest_features(self) -> Optional[pd.DataFrame]:
        """Get the latest computed features."""
        return self._features_df.copy() if self._features_df is not None else None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the feature window."""
        return {
            "initialized": self._is_initialized,
            "window_size": self.window_size,
            "current_size": len(self._features_df) if self._features_df is not None else 0,
            "last_update_ts": self._last_update_ts,
            "update_count": self._update_count,
            "feature_count": len(self._features_df.columns) if self._features_df is not None else 0,
            "symbol": self.symbol,
            "timeframe": self.timeframe
        }


class IncrementalFeatureManager:
    """
    Manages multiple feature windows for different symbol/timeframe combinations.

    Provides a high-level interface for incremental feature updates across
    multiple trading pairs and timeframes.
    """

    def __init__(self, max_windows: int = 20, default_window_size: int = 512):
        self.max_windows = max_windows
        self.default_window_size = default_window_size
        self.windows: Dict[str, FeatureWindow] = {}
        self.feature_cache = get_feature_cache()

    def get_window_key(self, symbol: str, timeframe: str) -> str:
        """Generate a unique key for symbol/timeframe combination."""
        return f"{symbol}_{timeframe}"

    def get_or_create_window(self, symbol: str, timeframe: str,
                           window_size: Optional[int] = None) -> FeatureWindow:
        """Get existing feature window or create a new one."""
        key = self.get_window_key(symbol, timeframe)

        if key not in self.windows:
            # Check if we need to remove old windows
            if len(self.windows) >= self.max_windows:
                self._cleanup_old_windows()

            size = window_size or self.default_window_size
            self.windows[key] = FeatureWindow(size, symbol, timeframe)
            logger.debug(f"Created new feature window for {symbol} {timeframe}")

        return self.windows[key]

    def update_features(self, symbol: str, timeframe: str,
                       df_candles: pd.DataFrame,
                       feature_config: Dict[str, Any],
                       new_candles: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Update features for a symbol/timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            df_candles: Full candle history for initialization
            feature_config: Feature computation configuration
            new_candles: New candles for incremental update (optional)

        Returns:
            Updated features dataframe
        """
        window = self.get_or_create_window(symbol, timeframe)

        if not window._is_initialized:
            # Initialize with full data
            return window.initialize(df_candles, feature_config)
        elif new_candles is not None and not new_candles.empty:
            # Incremental update
            return window.update(new_candles)
        else:
            # Return existing features
            existing = window.get_latest_features()
            return existing if existing is not None else pd.DataFrame()

    def get_features(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get latest features for a symbol/timeframe."""
        key = self.get_window_key(symbol, timeframe)
        if key in self.windows:
            return self.windows[key].get_latest_features()
        return None

    def clear_window(self, symbol: str, timeframe: str):
        """Clear a specific feature window."""
        key = self.get_window_key(symbol, timeframe)
        if key in self.windows:
            del self.windows[key]
            logger.debug(f"Cleared feature window for {symbol} {timeframe}")

    def clear_all_windows(self):
        """Clear all feature windows."""
        count = len(self.windows)
        self.windows.clear()
        logger.info(f"Cleared {count} feature windows")

    def _cleanup_old_windows(self):
        """Remove least recently used windows to make space."""
        if not self.windows:
            return

        # Sort by last update time (oldest first)
        sorted_windows = sorted(
            self.windows.items(),
            key=lambda x: x[1]._last_update_ts or 0
        )

        # Remove oldest windows until we're under the limit
        remove_count = len(self.windows) - self.max_windows + 1
        for i in range(min(remove_count, len(sorted_windows))):
            key, window = sorted_windows[i]
            del self.windows[key]
            logger.debug(f"Removed old feature window: {key}")

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get statistics about the feature manager."""
        window_stats = {}
        for key, window in self.windows.items():
            window_stats[key] = window.get_stats()

        return {
            "total_windows": len(self.windows),
            "max_windows": self.max_windows,
            "default_window_size": self.default_window_size,
            "windows": window_stats
        }


# Global instance for convenience
_global_manager: Optional[IncrementalFeatureManager] = None

def get_incremental_manager(max_windows: Optional[int] = None,
                          default_window_size: Optional[int] = None) -> IncrementalFeatureManager:
    """Get the global incremental feature manager."""
    global _global_manager
    if _global_manager is None or (max_windows is not None and max_windows != _global_manager.max_windows):
        _global_manager = IncrementalFeatureManager(
            max_windows or 20,
            default_window_size or 512
        )
    return _global_manager