"""
Feature caching system for avoiding redundant computations.

Caches computed features to speed up inference when the same data is processed multiple times.
"""
from __future__ import annotations

import hashlib
import pickle
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from loguru import logger


class FeatureCache:
    """
    Cache for computed features to avoid redundant calculations.
    """

    def __init__(self, cache_dir: Optional[Path] = None, max_cache_size_mb: int = 500):
        self.cache_dir = cache_dir or Path(".cache/features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_mb = max_cache_size_mb
        self._memory_cache: Dict[str, Tuple[pd.DataFrame, Any]] = {}

    def _generate_cache_key(self,
                          df_hash: str,
                          config_dict: Dict[str, Any],
                          timeframe: str) -> str:
        """Generate unique cache key for feature computation."""
        cache_data = {
            'df_hash': df_hash,
            'config': config_dict,
            'timeframe': timeframe,
            'version': '1.0'
        }

        cache_str = str(sorted(cache_data.items()))
        return hashlib.sha256(cache_str.encode()).hexdigest()[:16]

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate hash for dataframe content."""
        # Use last timestamp, length, and sample of OHLC values
        try:
            last_ts = df['ts_utc'].iloc[-1] if len(df) > 0 else 0
            content = f"{last_ts}_{len(df)}"

            if 'close' in df.columns and len(df) > 0:
                # Sample values from beginning, middle, end
                close_vals = df['close'].iloc[[0, len(df)//2, -1]].values
                content += f"_{hash(tuple(close_vals))}"

            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception:
            # Fallback to basic hash
            return hashlib.sha256(str(df.shape).encode()).hexdigest()[:16]

    def get_cached_features(self,
                          df: pd.DataFrame,
                          config_dict: Dict[str, Any],
                          timeframe: str) -> Optional[Tuple[pd.DataFrame, Any]]:
        """
        Retrieve cached features if available.

        Returns:
            Tuple of (features_df, metadata) or None if not cached
        """
        try:
            df_hash = self._hash_dataframe(df)
            cache_key = self._generate_cache_key(df_hash, config_dict, timeframe)

            # Check memory cache first
            if cache_key in self._memory_cache:
                logger.debug(f"Features found in memory cache: {cache_key}")
                return self._memory_cache[cache_key]

            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Add to memory cache
                self._memory_cache[cache_key] = cached_data
                logger.debug(f"Features loaded from disk cache: {cache_key}")
                return cached_data

            return None

        except Exception as e:
            logger.warning(f"Error retrieving cached features: {e}")
            return None

    def cache_features(self,
                      df: pd.DataFrame,
                      features_df: pd.DataFrame,
                      metadata: Any,
                      config_dict: Dict[str, Any],
                      timeframe: str) -> None:
        """
        Cache computed features.

        Args:
            df: Original input dataframe
            features_df: Computed features
            metadata: Associated metadata (e.g., standardizer)
            config_dict: Configuration used for computation
            timeframe: Timeframe of the data
        """
        try:
            df_hash = self._hash_dataframe(df)
            cache_key = self._generate_cache_key(df_hash, config_dict, timeframe)

            cached_data = (features_df.copy(), metadata)

            # Store in memory cache
            self._memory_cache[cache_key] = cached_data

            # Store on disk
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.debug(f"Features cached: {cache_key}")

            # Clean up old cache files if needed
            self._cleanup_cache()

        except Exception as e:
            logger.warning(f"Error caching features: {e}")

    def _cleanup_cache(self) -> None:
        """Clean up old cache files to keep cache size under limit."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))

            if not cache_files:
                return

            # Calculate total cache size
            total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)

            if total_size_mb <= self.max_cache_size_mb:
                return

            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda f: f.stat().st_mtime)

            # Remove oldest files until under limit
            while total_size_mb > self.max_cache_size_mb and cache_files:
                old_file = cache_files.pop(0)
                file_size_mb = old_file.stat().st_size / (1024 * 1024)
                old_file.unlink()
                total_size_mb -= file_size_mb
                logger.debug(f"Removed old cache file: {old_file.name}")

        except Exception as e:
            logger.warning(f"Error cleaning up cache: {e}")

    def clear_cache(self) -> None:
        """Clear all cached features."""
        try:
            # Clear memory cache
            self._memory_cache.clear()

            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

            logger.info("Feature cache cleared")

        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)

            return {
                'memory_entries': len(self._memory_cache),
                'disk_entries': len(cache_files),
                'total_size_mb': round(total_size_mb, 2),
                'max_size_mb': self.max_cache_size_mb,
                'cache_dir': str(self.cache_dir)
            }

        except Exception as e:
            return {'error': str(e)}


# Global cache instance
_global_cache: Optional[FeatureCache] = None

def get_feature_cache() -> FeatureCache:
    """Get the global feature cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = FeatureCache()
    return _global_cache