"""
Spread Analytics Service

Monitors and analyzes bid-ask spreads for anomaly detection and contextual alerts.
Tracks spread history, calculates percentiles, and provides visual indicators.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Deque
from collections import deque
from datetime import datetime, timedelta
import statistics
from loguru import logger


class SpreadAnalytics:
    """
    Service for real-time spread monitoring and anomaly detection.

    Features:
    - Rolling spread history tracking
    - Statistical anomaly detection
    - Spread percentile calculation
    - Alert generation for abnormal conditions
    - Per-symbol analytics
    """

    def __init__(self, history_size: int = 720, alert_threshold_multiplier: float = 2.0):
        """
        Initialize spread analytics service.

        Args:
            history_size: Number of spread samples to keep (720 = 1 hour at 5s intervals)
            alert_threshold_multiplier: Multiplier for anomaly detection (2.0 = 2x average)
        """
        self.history_size = history_size
        self.alert_threshold_multiplier = alert_threshold_multiplier

        # Per-symbol spread history
        self._spread_history: Dict[str, Deque[float]] = {}

        # Cache for computed statistics
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(seconds=5)  # Cache stats for 5 seconds

    def record_spread(self, symbol: str, spread: float, timestamp: Optional[datetime] = None):
        """
        Record a spread observation.

        Args:
            symbol: Trading symbol
            spread: Current spread value
            timestamp: Optional timestamp (defaults to now)
        """
        if symbol not in self._spread_history:
            self._spread_history[symbol] = deque(maxlen=self.history_size)

        self._spread_history[symbol].append(spread)

        # Invalidate cache
        if symbol in self._stats_cache:
            del self._stats_cache[symbol]

        logger.debug(f"Recorded spread for {symbol}: {spread:.5f}")

    def get_spread_statistics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get statistical metrics for a symbol's spread.

        Returns:
            Dictionary with:
            - current: Latest spread
            - mean: Average spread
            - median: Median spread
            - std: Standard deviation
            - min: Minimum spread
            - max: Maximum spread
            - p25, p50, p75, p90, p95: Percentiles
            - sample_count: Number of samples
        """
        if symbol not in self._spread_history or len(self._spread_history[symbol]) == 0:
            return None

        # Check cache
        now = datetime.now()
        if symbol in self._stats_cache:
            if symbol in self._cache_timestamp:
                if now - self._cache_timestamp[symbol] < self._cache_ttl:
                    return self._stats_cache[symbol]

        # Compute statistics
        spreads = list(self._spread_history[symbol])

        try:
            stats = {
                'current': spreads[-1] if spreads else 0.0,
                'mean': statistics.mean(spreads) if spreads else 0.0,
                'median': statistics.median(spreads) if spreads else 0.0,
                'std': statistics.stdev(spreads) if len(spreads) > 1 else 0.0,
                'min': min(spreads) if spreads else 0.0,
                'max': max(spreads) if spreads else 0.0,
                'sample_count': len(spreads)
            }

            # Calculate percentiles
            if spreads:
                sorted_spreads = sorted(spreads)
                n = len(sorted_spreads)
                stats['p25'] = sorted_spreads[int(n * 0.25)]
                stats['p50'] = sorted_spreads[int(n * 0.50)]
                stats['p75'] = sorted_spreads[int(n * 0.75)]
                stats['p90'] = sorted_spreads[int(n * 0.90)]
                stats['p95'] = sorted_spreads[int(n * 0.95)]
            else:
                stats.update({'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p90': 0.0, 'p95': 0.0})

            # Cache results
            self._stats_cache[symbol] = stats
            self._cache_timestamp[symbol] = now

            return stats

        except Exception as e:
            logger.error(f"Error computing spread statistics for {symbol}: {e}")
            return None

    def detect_anomaly(self, symbol: str, current_spread: float) -> Dict[str, Any]:
        """
        Detect spread anomalies.

        Args:
            symbol: Trading symbol
            current_spread: Current spread to check

        Returns:
            Dictionary with:
            - is_anomaly: Whether spread is anomalous
            - severity: 'normal', 'elevated', 'high', 'critical'
            - ratio: Spread/average ratio
            - threshold: Threshold that was exceeded
            - message: Human-readable description
        """
        stats = self.get_spread_statistics(symbol)

        if not stats or stats['sample_count'] < 10:
            return {
                'is_anomaly': False,
                'severity': 'unknown',
                'ratio': 1.0,
                'threshold': 0.0,
                'message': 'Insufficient data for anomaly detection'
            }

        avg_spread = stats['mean']
        if avg_spread == 0:
            return {
                'is_anomaly': False,
                'severity': 'normal',
                'ratio': 1.0,
                'threshold': 0.0,
                'message': 'Average spread is zero'
            }

        ratio = current_spread / avg_spread

        # Determine severity based on ratio
        if ratio >= 3.0:
            severity = 'critical'
            is_anomaly = True
            message = f"🔴 CRITICAL: Spread is {ratio:.1f}x average (current={current_spread:.5f}, avg={avg_spread:.5f})"
        elif ratio >= 2.0:
            severity = 'high'
            is_anomaly = True
            message = f"🟠 HIGH: Spread is {ratio:.1f}x average (current={current_spread:.5f}, avg={avg_spread:.5f})"
        elif ratio >= 1.5:
            severity = 'elevated'
            is_anomaly = True
            message = f"🟡 ELEVATED: Spread is {ratio:.1f}x average (current={current_spread:.5f}, avg={avg_spread:.5f})"
        else:
            severity = 'normal'
            is_anomaly = False
            message = f"✅ Normal spread ({ratio:.2f}x average)"

        return {
            'is_anomaly': is_anomaly,
            'severity': severity,
            'ratio': ratio,
            'threshold': avg_spread * self.alert_threshold_multiplier,
            'message': message,
            'stats': stats
        }

    def get_spread_percentile(self, symbol: str, spread: float) -> Optional[float]:
        """
        Calculate which percentile a given spread falls into.

        Args:
            symbol: Trading symbol
            spread: Spread value to check

        Returns:
            Percentile (0-100) or None if insufficient data
        """
        if symbol not in self._spread_history or len(self._spread_history[symbol]) < 10:
            return None

        spreads = sorted(list(self._spread_history[symbol]))
        count_below = sum(1 for s in spreads if s <= spread)
        percentile = (count_below / len(spreads)) * 100

        return percentile

    def get_alert_level(self, symbol: str, spread: float) -> str:
        """
        Get visual alert level for a spread.

        Args:
            symbol: Trading symbol
            spread: Spread value

        Returns:
            Alert level: 'normal', 'warning', 'danger'
        """
        anomaly = self.detect_anomaly(symbol, spread)

        if anomaly['severity'] in ['critical', 'high']:
            return 'danger'
        elif anomaly['severity'] == 'elevated':
            return 'warning'
        else:
            return 'normal'

    def get_contextual_display(self, symbol: str, current_spread: float, price: float = 1.0) -> str:
        """
        Get contextual display string for spread.

        Args:
            symbol: Trading symbol
            current_spread: Current spread
            price: Current price (for basis points calculation)

        Returns:
            Formatted string with spread context
        """
        stats = self.get_spread_statistics(symbol)
        if not stats:
            return f"{current_spread:.5f} (no history)"

        # Calculate basis points
        spread_bps = (current_spread / price) * 10000 if price > 0 else 0

        # Get percentile
        percentile = self.get_spread_percentile(symbol, current_spread)
        percentile_str = f"{percentile:.0f}th pctl" if percentile else "N/A"

        # Get anomaly status
        anomaly = self.detect_anomaly(symbol, current_spread)

        # Build contextual string
        if anomaly['is_anomaly']:
            icon = {'elevated': '🟡', 'high': '🟠', 'critical': '🔴'}.get(anomaly['severity'], '⚠️')
            return (
                f"{icon} {current_spread:.5f} ({spread_bps:.1f}bps) - "
                f"{anomaly['ratio']:.1f}x avg ({percentile_str})"
            )
        else:
            return (
                f"✅ {current_spread:.5f} ({spread_bps:.1f}bps) - "
                f"avg={stats['mean']:.5f} ({percentile_str})"
            )

    def clear_history(self, symbol: Optional[str] = None):
        """
        Clear spread history.

        Args:
            symbol: Symbol to clear, or None to clear all
        """
        if symbol:
            if symbol in self._spread_history:
                self._spread_history[symbol].clear()
            if symbol in self._stats_cache:
                del self._stats_cache[symbol]
        else:
            self._spread_history.clear()
            self._stats_cache.clear()
            self._cache_timestamp.clear()

    def get_summary(self, symbol: str) -> Optional[str]:
        """
        Get human-readable summary of spread analytics.

        Args:
            symbol: Trading symbol

        Returns:
            Summary string
        """
        stats = self.get_spread_statistics(symbol)
        if not stats:
            return None

        return (
            f"Spread Analytics for {symbol}:\n"
            f"  Current: {stats['current']:.5f}\n"
            f"  Average: {stats['mean']:.5f}\n"
            f"  Median: {stats['median']:.5f}\n"
            f"  Range: {stats['min']:.5f} - {stats['max']:.5f}\n"
            f"  Std Dev: {stats['std']:.5f}\n"
            f"  Percentiles: P25={stats['p25']:.5f}, P50={stats['p50']:.5f}, "
            f"P75={stats['p75']:.5f}, P95={stats['p95']:.5f}\n"
            f"  Samples: {stats['sample_count']}"
        )
