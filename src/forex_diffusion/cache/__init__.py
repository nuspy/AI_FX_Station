"""
Caching system for ForexGPT pattern detection and analysis.

Provides Redis-lite based caching with LRU eviction, performance monitoring,
and specialized interfaces for pattern detection optimization.
"""

from .redis_cache import (
    RedisLiteCache,
    PatternCache,
    CacheStats,
    initialize_cache,
    get_cache,
    get_pattern_cache,
    cache_decorator
)

__all__ = [
    'RedisLiteCache',
    'PatternCache',
    'CacheStats',
    'initialize_cache',
    'get_cache',
    'get_pattern_cache',
    'cache_decorator'
]