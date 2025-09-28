"""
Disk-based caching system for pattern detection and analysis.

Provides embedded caching without external dependencies.
Supports LRU eviction, configurable memory limits, and performance monitoring.
"""

from __future__ import annotations

import json
import pickle
import hashlib
import psutil
import threading
import tempfile
import os
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

try:
    import diskcache as dc
except ImportError:
    dc = None


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0
    total_keys: int = 0


class RedisLiteCache:
    """
    Embedded Redis cache using redislite for pattern detection optimization.

    Features:
    - Embedded Redis (no external server needed)
    - Configurable memory limits with LRU eviction
    - Performance monitoring and statistics
    - Thread-safe operations
    - Automatic memory management
    """

    def __init__(self, config: Dict[str, Any]):
        if redislite is None:
            raise ImportError("redislite package not available. Install with: pip install redislite")

        self.config = config
        self._cache_config = config.get('cache', {}).get('redis', {})
        self._enabled = self._cache_config.get('enabled', False)

        if not self._enabled:
            logger.info("Redis cache disabled in configuration")
            self._redis = None
            return

        # Initialize Redis-lite
        self._db_path = self._cache_config.get('path', 'data/redis.db')
        self._max_memory_percent = self._cache_config.get('max_memory_percent', 70)
        self._memory_policy = self._cache_config.get('memory_policy', 'allkeys-lru')
        self._key_ttl = self._cache_config.get('key_ttl', 3600)

        # Calculate dynamic max memory (starts small, expands as needed up to limit)
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self._max_memory_mb = int(total_memory_gb * 1024 * self._max_memory_percent / 100)
        self._current_memory_limit_mb = min(256, self._max_memory_mb)  # Start with 256MB

        # Statistics and monitoring
        self._stats = CacheStats()
        self._stats_lock = threading.Lock()

        # Initialize Redis connection
        self._redis = None
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis-lite connection with configuration"""
        try:
            # Create Redis-lite instance
            self._redis = redislite.Redis(
                self._db_path,
                decode_responses=False  # We'll handle encoding manually
            )

            # Configure initial memory policy (dynamic expansion)
            self._redis.config_set('maxmemory', f'{self._current_memory_limit_mb}mb')
            self._redis.config_set('maxmemory-policy', self._memory_policy)

            # Test connection
            self._redis.ping()
            logger.info(f"Redis-lite initialized: {self._db_path} (initial: {self._current_memory_limit_mb}MB, max: {self._max_memory_mb}MB)")

        except Exception as e:
            logger.error(f"Failed to initialize Redis-lite: {e}")
            self._redis = None
            self._enabled = False

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key from arguments"""
        # Create deterministic hash from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_hash = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:12]
        return f"{prefix}:{key_hash}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        try:
            # Try JSON first for simple types (faster)
            if isinstance(value, (str, int, float, bool, list, dict)):
                return json.dumps(value).encode('utf-8')
            else:
                # Fall back to pickle for complex objects
                return b'pickle:' + pickle.dumps(value)
        except Exception:
            # Last resort: pickle everything
            return b'pickle:' + pickle.dumps(value)

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from Redis storage"""
        try:
            if data.startswith(b'pickle:'):
                return pickle.loads(data[7:])  # Remove 'pickle:' prefix
            else:
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to deserialize cached value: {e}")
            return None

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._enabled or not self._redis:
            return None

        try:
            data = self._redis.get(key)
            if data is None:
                with self._stats_lock:
                    self._stats.misses += 1
                return None

            value = self._deserialize_value(data)
            with self._stats_lock:
                self._stats.hits += 1
            return value

        except Exception as e:
            logger.debug(f"Cache get error for key {key}: {e}")
            with self._stats_lock:
                self._stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL and dynamic memory expansion"""
        if not self._enabled or not self._redis:
            return False

        try:
            serialized = self._serialize_value(value)
            ttl = ttl or self._key_ttl

            # Check if we need to expand memory before setting
            self._check_and_expand_memory()

            if ttl > 0:
                self._redis.setex(key, ttl, serialized)
            else:
                self._redis.set(key, serialized)

            return True

        except Exception as e:
            # If out of memory, try expanding and retry once
            if "OOM" in str(e) or "memory" in str(e).lower():
                if self._expand_memory():
                    try:
                        if ttl > 0:
                            self._redis.setex(key, ttl, serialized)
                        else:
                            self._redis.set(key, serialized)
                        return True
                    except Exception:
                        pass

            logger.debug(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._enabled or not self._redis:
            return False

        try:
            return bool(self._redis.delete(key))
        except Exception as e:
            logger.debug(f"Cache delete error for key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        if not self._enabled or not self._redis:
            return False

        try:
            self._redis.flushdb()
            with self._stats_lock:
                self._stats = CacheStats()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        if not self._enabled or not self._redis:
            return CacheStats()

        try:
            with self._stats_lock:
                # Update memory usage and key count
                info = self._redis.info('memory')
                memory_mb = info.get('used_memory', 0) / (1024 * 1024)
                key_count = self._redis.dbsize()

                # Calculate hit rate
                total_requests = self._stats.hits + self._stats.misses
                hit_rate = self._stats.hits / total_requests if total_requests > 0 else 0.0

                # Update stats
                self._stats.memory_usage_mb = memory_mb
                self._stats.total_keys = key_count
                self._stats.hit_rate = hit_rate

                return CacheStats(
                    hits=self._stats.hits,
                    misses=self._stats.misses,
                    evictions=self._stats.evictions,
                    memory_usage_mb=memory_mb,
                    hit_rate=hit_rate,
                    total_keys=key_count
                )

        except Exception as e:
            logger.debug(f"Error getting cache stats: {e}")
            return CacheStats()

    def is_enabled(self) -> bool:
        """Check if cache is enabled and available"""
        return self._enabled and self._redis is not None

    def _check_and_expand_memory(self):
        """Check memory usage and expand if approaching limit"""
        try:
            info = self._redis.info('memory')
            used_memory_mb = info.get('used_memory', 0) / (1024 * 1024)

            # If using more than 85% of current limit, expand
            if used_memory_mb > self._current_memory_limit_mb * 0.85:
                self._expand_memory()

        except Exception as e:
            logger.debug(f"Error checking memory usage: {e}")

    def _expand_memory(self) -> bool:
        """Expand memory limit dynamically up to maximum"""
        try:
            if self._current_memory_limit_mb >= self._max_memory_mb:
                logger.debug("Cache already at maximum memory limit")
                return False

            # Expand by 50% or 512MB, whichever is smaller
            expansion = min(
                int(self._current_memory_limit_mb * 0.5),
                512,
                self._max_memory_mb - self._current_memory_limit_mb
            )

            new_limit = min(self._current_memory_limit_mb + expansion, self._max_memory_mb)

            # Update Redis memory limit
            self._redis.config_set('maxmemory', f'{new_limit}mb')

            old_limit = self._current_memory_limit_mb
            self._current_memory_limit_mb = new_limit

            logger.info(f"Cache memory expanded: {old_limit}MB → {new_limit}MB")
            return True

        except Exception as e:
            logger.error(f"Failed to expand cache memory: {e}")
            return False

    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information"""
        try:
            info = self._redis.info('memory')
            used_mb = info.get('used_memory', 0) / (1024 * 1024)

            return {
                'used_mb': used_mb,
                'current_limit_mb': self._current_memory_limit_mb,
                'max_limit_mb': self._max_memory_mb,
                'usage_percent': (used_mb / self._current_memory_limit_mb * 100) if self._current_memory_limit_mb > 0 else 0,
                'max_usage_percent': (self._current_memory_limit_mb / self._max_memory_mb * 100) if self._max_memory_mb > 0 else 0,
                'can_expand': self._current_memory_limit_mb < self._max_memory_mb
            }
        except Exception:
            return {
                'used_mb': 0,
                'current_limit_mb': self._current_memory_limit_mb,
                'max_limit_mb': self._max_memory_mb,
                'usage_percent': 0,
                'max_usage_percent': 0,
                'can_expand': False
            }

    def close(self):
        """Close Redis connection"""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
            self._redis = None


class PatternCache:
    """
    High-level caching interface for pattern detection.

    Provides specialized caching methods for:
    - Pattern detection results
    - Parameter combinations
    - Multi-timeframe analysis
    - Regime detection data
    """

    def __init__(self, redis_cache: RedisLiteCache):
        self.redis = redis_cache
        self._pattern_config = redis_cache.config.get('cache', {}).get('pattern_cache', {})
        self._ttl_minutes = self._pattern_config.get('ttl_minutes', 30)
        self._ttl_seconds = self._ttl_minutes * 60

    def cache_pattern_detection(self,
                               asset: str,
                               timeframe: str,
                               pattern_type: str,
                               params: Dict[str, Any],
                               result: Any) -> bool:
        """Cache pattern detection result"""
        key = self.redis._generate_key(
            'pattern_detection',
            asset, timeframe, pattern_type,
            **params
        )
        return self.redis.set(key, result, ttl=self._ttl_seconds)

    def get_pattern_detection(self,
                             asset: str,
                             timeframe: str,
                             pattern_type: str,
                             params: Dict[str, Any]) -> Optional[Any]:
        """Get cached pattern detection result"""
        key = self.redis._generate_key(
            'pattern_detection',
            asset, timeframe, pattern_type,
            **params
        )
        return self.redis.get(key)

    def cache_multi_timeframe_analysis(self,
                                      asset: str,
                                      timeframes: List[str],
                                      analysis_type: str,
                                      result: Any) -> bool:
        """Cache multi-timeframe analysis result"""
        key = self.redis._generate_key(
            'multi_tf_analysis',
            asset, tuple(sorted(timeframes)), analysis_type
        )
        return self.redis.set(key, result, ttl=self._ttl_seconds)

    def get_multi_timeframe_analysis(self,
                                    asset: str,
                                    timeframes: List[str],
                                    analysis_type: str) -> Optional[Any]:
        """Get cached multi-timeframe analysis result"""
        key = self.redis._generate_key(
            'multi_tf_analysis',
            asset, tuple(sorted(timeframes)), analysis_type
        )
        return self.redis.get(key)

    def cache_regime_detection(self,
                              asset: str,
                              timeframe: str,
                              regime_data: Any) -> bool:
        """Cache regime detection result"""
        key = self.redis._generate_key(
            'regime_detection',
            asset, timeframe
        )
        # Longer TTL for regime data (1 hour)
        return self.redis.set(key, regime_data, ttl=3600)

    def get_regime_detection(self,
                           asset: str,
                           timeframe: str) -> Optional[Any]:
        """Get cached regime detection result"""
        key = self.redis._generate_key(
            'regime_detection',
            asset, timeframe
        )
        return self.redis.get(key)

    def cache_parameter_optimization(self,
                                   asset: str,
                                   timeframe: str,
                                   pattern_type: str,
                                   regime: str,
                                   optimal_params: Dict[str, Any]) -> bool:
        """Cache optimized parameters for specific context"""
        key = self.redis._generate_key(
            'optimal_params',
            asset, timeframe, pattern_type, regime
        )
        # Very long TTL for optimized parameters (24 hours)
        return self.redis.set(key, optimal_params, ttl=86400)

    def get_parameter_optimization(self,
                                 asset: str,
                                 timeframe: str,
                                 pattern_type: str,
                                 regime: str) -> Optional[Dict[str, Any]]:
        """Get cached optimized parameters"""
        key = self.redis._generate_key(
            'optimal_params',
            asset, timeframe, pattern_type, regime
        )
        return self.redis.get(key)

    def invalidate_pattern_cache(self, asset: str, timeframe: str):
        """Invalidate all pattern cache for specific asset/timeframe"""
        if not self.redis.is_enabled():
            return

        try:
            # Find and delete all keys matching pattern
            pattern = f"pattern_detection:*{asset}*{timeframe}*"
            keys = self.redis._redis.keys(pattern)
            if keys:
                self.redis._redis.delete(*keys)
                logger.debug(f"Invalidated {len(keys)} pattern cache entries for {asset}/{timeframe}")
        except Exception as e:
            logger.debug(f"Error invalidating pattern cache: {e}")


# Global cache instance
_cache_instance: Optional[RedisLiteCache] = None
_pattern_cache_instance: Optional[PatternCache] = None


def initialize_cache(config: Dict[str, Any]) -> tuple[RedisLiteCache, PatternCache]:
    """Initialize global cache instances"""
    global _cache_instance, _pattern_cache_instance

    _cache_instance = RedisLiteCache(config)
    _pattern_cache_instance = PatternCache(_cache_instance)

    return _cache_instance, _pattern_cache_instance


def get_cache() -> Optional[RedisLiteCache]:
    """Get global cache instance"""
    return _cache_instance


def get_pattern_cache() -> Optional[PatternCache]:
    """Get global pattern cache instance"""
    return _pattern_cache_instance


def cache_decorator(prefix: str, ttl: Optional[int] = None):
    """
    Decorator for caching function results.

    Usage:
        @cache_decorator('my_function', ttl=3600)
        def expensive_function(arg1, arg2):
            return complex_calculation(arg1, arg2)
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            cache = get_cache()
            if not cache or not cache.is_enabled():
                return func(*args, **kwargs)

            # Generate cache key
            key = cache._generate_key(prefix, *args, **kwargs)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl=ttl)
            return result

        return wrapper
    return decorator