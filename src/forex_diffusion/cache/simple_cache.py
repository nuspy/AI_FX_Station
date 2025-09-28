"""
Simple in-memory caching system for pattern detection.

Compatible with Windows and provides basic caching functionality
without external dependencies.
"""

from __future__ import annotations

import json
import pickle
import hashlib
import threading
import weakref
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    hit_ratio: float = 0.0
    total_keys: int = 0

    def update_hit_ratio(self):
        total = self.hits + self.misses
        self.hit_ratio = (self.hits / total) if total > 0 else 0.0


class SimpleCache:
    """Simple in-memory cache with LRU eviction"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 256):
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
        self._max_memory_mb = max_memory_mb
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                self._stats.hits += 1
                return self._cache[key]
            else:
                self._stats.misses += 1
                return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        with self._lock:
            # If key exists, remove from access order
            if key in self._cache:
                self._access_order.remove(key)

            # Add to cache
            self._cache[key] = value
            self._access_order.append(key)
            self._stats.sets += 1

            # Check if we need to evict
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            return key in self._cache

    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern (simplified - only supports * wildcard)"""
        with self._lock:
            if pattern == "*":
                return list(self._cache.keys())
            # Simple pattern matching
            import fnmatch
            return [k for k in self._cache.keys() if fnmatch.fnmatch(k, pattern)]

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is too large"""
        # Evict by size
        while len(self._cache) > self._max_size:
            if not self._access_order:
                break
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
                self._stats.evictions += 1

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            self._stats.total_keys = len(self._cache)
            self._stats.update_hit_ratio()
            return self._stats


class PatternCache:
    """Pattern-specific caching with simple in-memory backend"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cache_config = config.get('cache', {}).get('redis', {})  # Keep same config structure
        self._enabled = self._cache_config.get('enabled', True)

        if not self._enabled:
            logger.info("Pattern cache disabled in configuration")
            self._cache = None
            return

        # Initialize simple cache
        max_size = self._cache_config.get('max_keys', 10000)
        max_memory_mb = self._cache_config.get('max_memory_mb', 256)

        self._cache = SimpleCache(max_size=max_size, max_memory_mb=max_memory_mb)
        self._key_ttl = self._cache_config.get('key_ttl', 3600)

        logger.info(f"Pattern cache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")

    def get_pattern_results(self, pattern_type: str, symbol: str, timeframe: str,
                          data_hash: str) -> Optional[List]:
        """Get cached pattern detection results"""
        if not self._enabled or not self._cache:
            return None

        cache_key = f"pattern:{pattern_type}:{symbol}:{timeframe}:{data_hash}"
        return self._cache.get(cache_key)

    def set_pattern_results(self, pattern_type: str, symbol: str, timeframe: str,
                          data_hash: str, results: List) -> None:
        """Cache pattern detection results"""
        if not self._enabled or not self._cache:
            return

        cache_key = f"pattern:{pattern_type}:{symbol}:{timeframe}:{data_hash}"
        self._cache.set(cache_key, results, ttl=self._key_ttl)

    def get_strength_calculation(self, pattern_id: str, data_hash: str) -> Optional[Dict]:
        """Get cached pattern strength calculation"""
        if not self._enabled or not self._cache:
            return None

        cache_key = f"strength:{pattern_id}:{data_hash}"
        return self._cache.get(cache_key)

    def set_strength_calculation(self, pattern_id: str, data_hash: str,
                               calculation: Dict) -> None:
        """Cache pattern strength calculation"""
        if not self._enabled or not self._cache:
            return

        cache_key = f"strength:{pattern_id}:{data_hash}"
        self._cache.set(cache_key, calculation, ttl=self._key_ttl)

    def get_regime_detection(self, symbol: str, timeframe: str, data_hash: str) -> Optional[Dict]:
        """Get cached regime detection results"""
        if not self._enabled or not self._cache:
            return None

        cache_key = f"regime:{symbol}:{timeframe}:{data_hash}"
        return self._cache.get(cache_key)

    def set_regime_detection(self, symbol: str, timeframe: str, data_hash: str,
                           result: Dict) -> None:
        """Cache regime detection results"""
        if not self._enabled or not self._cache:
            return

        cache_key = f"regime:{symbol}:{timeframe}:{data_hash}"
        self._cache.set(cache_key, result, ttl=self._key_ttl)

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics"""
        if not self._enabled or not self._cache:
            return {"enabled": False}

        stats = self._cache.get_stats()
        return {
            "enabled": True,
            "total_keys": stats.total_keys,
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_ratio": stats.hit_ratio,
            "memory_usage_mb": stats.memory_usage_mb,
            "memory_limit_mb": self._cache._max_memory_mb,
            "evictions": stats.evictions
        }

    def clear_pattern_cache(self, pattern_type: Optional[str] = None) -> None:
        """Clear pattern cache entries"""
        if not self._enabled or not self._cache:
            return

        if pattern_type:
            # Clear specific pattern type
            keys_to_delete = self._cache.keys(f"pattern:{pattern_type}:*")
            for key in keys_to_delete:
                self._cache.delete(key)
        else:
            # Clear all pattern entries
            keys_to_delete = self._cache.keys("pattern:*")
            for key in keys_to_delete:
                self._cache.delete(key)

    def invalidate_symbol(self, symbol: str) -> None:
        """Invalidate all cache entries for a symbol"""
        if not self._enabled or not self._cache:
            return

        keys_to_delete = self._cache.keys(f"*:{symbol}:*")
        for key in keys_to_delete:
            self._cache.delete(key)


# Global cache instances
_pattern_cache: Optional[PatternCache] = None
_general_cache: Optional[SimpleCache] = None


def get_pattern_cache(config: Optional[Dict[str, Any]] = None) -> PatternCache:
    """Get global pattern cache instance"""
    global _pattern_cache
    if _pattern_cache is None:
        if config is None:
            # Default config
            config = {
                'cache': {
                    'redis': {
                        'enabled': True,
                        'max_keys': 10000,
                        'max_memory_mb': 256,
                        'key_ttl': 3600
                    }
                }
            }
        _pattern_cache = PatternCache(config)
    return _pattern_cache


def get_cache(config: Optional[Dict[str, Any]] = None) -> SimpleCache:
    """Get global general cache instance"""
    global _general_cache
    if _general_cache is None:
        max_size = 5000
        max_memory_mb = 128
        if config:
            cache_config = config.get('cache', {}).get('general', {})
            max_size = cache_config.get('max_keys', max_size)
            max_memory_mb = cache_config.get('max_memory_mb', max_memory_mb)

        _general_cache = SimpleCache(max_size=max_size, max_memory_mb=max_memory_mb)
    return _general_cache


def cache_decorator(key_func: Callable = None, ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_cache()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result

        return wrapper
    return decorator