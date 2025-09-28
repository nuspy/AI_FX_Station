"""
Caching system for ForexGPT pattern detection and analysis.

Provides simple in-memory caching with LRU eviction, performance monitoring,
and specialized interfaces for pattern detection optimization.
"""

try:
    # Try to use redis cache first (if available)
    from .redis_cache import (
        PatternCache,
        CacheStats,
        get_cache,
        get_pattern_cache,
        cache_decorator
    )
except ImportError:
    # Fallback to simple cache for Windows compatibility
    from .simple_cache import (
        PatternCache,
        CacheStats,
        get_cache,
        get_pattern_cache,
        cache_decorator
    )

__all__ = [
    'PatternCache',
    'CacheStats',
    'get_cache',
    'get_pattern_cache',
    'cache_decorator'
]