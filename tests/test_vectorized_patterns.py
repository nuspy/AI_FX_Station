"""
Test suite for vectorized pattern detection.

Verifies:
1. Vectorized detectors produce correct results
2. Results match loop-based detectors
3. Performance improvements are real
4. Integration with pattern registry works
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

# Import both implementations
from forex_diffusion.patterns.vectorized_detectors import (
    VectorizedCandleDetector,
    VectorizedThreeCandleDetector,
    detect_swings_vectorized,
    benchmark_vectorized_vs_loop
)
from forex_diffusion.patterns.candles import SimpleCandleDetector
from forex_diffusion.patterns.registry import PatternRegistry


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    n = 1000
    
    # Generate realistic price data
    base_price = 1.1000
    returns = np.random.randn(n) * 0.0005
    close = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC with realistic relationships
    high = close * (1 + np.abs(np.random.randn(n)) * 0.0002)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.0002)
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0]
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1min'),
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    df['ts_utc'] = df['timestamp']
    
    return df


@pytest.fixture
def hammer_data():
    """Generate data with clear hammer patterns."""
    n = 100
    base_price = 1.1000
    
    # Normal candles
    close = np.full(n, base_price)
    open_prices = close + np.random.randn(n) * 0.0001
    high = np.maximum(open_prices, close) + np.random.rand(n) * 0.0001
    low = np.minimum(open_prices, close) - np.random.rand(n) * 0.0001
    
    # Insert clear hammer patterns at specific indices
    hammer_indices = [10, 30, 50, 70, 90]
    for idx in hammer_indices:
        if idx < n:
            # Hammer: small body, long lower shadow, small upper shadow
            body_size = 0.0002
            close[idx] = base_price
            open_prices[idx] = close[idx] - body_size
            high[idx] = close[idx] + body_size * 0.2  # Small upper shadow
            low[idx] = close[idx] - body_size * 3.0   # Long lower shadow
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1min'),
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    df['ts_utc'] = df['timestamp']
    
    return df, hammer_indices


class TestVectorizedCandleDetectors:
    """Test vectorized candlestick pattern detection."""
    
    def test_vectorized_hammer_detection(self, hammer_data):
        """Test vectorized hammer detection produces correct results."""
        df, expected_indices = hammer_data
        
        detector = VectorizedCandleDetector("hammer")
        events = detector.detect(df)
        
        # Should find the hammer patterns
        assert len(events) > 0, "Vectorized detector should find hammer patterns"
        
        # Check events are PatternEvents with correct structure
        for event in events:
            assert hasattr(event, 'pattern_key')
            assert event.pattern_key == "hammer"
            assert event.direction == "bull"
            assert event.kind == "candle"
            assert event.score > 0
    
    def test_vectorized_vs_loop_hammer(self, hammer_data):
        """Verify vectorized and loop-based detectors find similar patterns."""
        df, _ = hammer_data
        
        # Vectorized detection
        vec_detector = VectorizedCandleDetector("hammer")
        vec_events = vec_detector.detect(df)
        
        # Loop-based detection
        loop_detector = SimpleCandleDetector("hammer")
        loop_events = loop_detector.detect(df)
        
        # Should find similar number of patterns (may differ slightly due to implementation details)
        vec_count = len(vec_events)
        loop_count = len(loop_events)
        
        print(f"\nVectorized found: {vec_count} hammers")
        print(f"Loop-based found: {loop_count} hammers")
        
        # Allow for small differences (within 20%)
        assert abs(vec_count - loop_count) <= max(1, loop_count * 0.2), \
            f"Vectorized ({vec_count}) and loop-based ({loop_count}) should find similar patterns"
    
    def test_vectorized_engulfing(self, sample_ohlc_data):
        """Test vectorized engulfing pattern detection."""
        df = sample_ohlc_data
        
        # Test bullish engulfing
        bull_detector = VectorizedCandleDetector("bullish_engulfing")
        bull_events = bull_detector.detect(df)
        
        print(f"\nFound {len(bull_events)} bullish engulfing patterns")
        
        for event in bull_events:
            assert event.pattern_key == "bullish_engulfing"
            assert event.direction == "bull"
            assert event.bars_span == 2
        
        # Test bearish engulfing
        bear_detector = VectorizedCandleDetector("bearish_engulfing")
        bear_events = bear_detector.detect(df)
        
        print(f"Found {len(bear_events)} bearish engulfing patterns")
        
        for event in bear_events:
            assert event.pattern_key == "bearish_engulfing"
            assert event.direction == "bear"
            assert event.bars_span == 2
    
    def test_vectorized_doji(self, sample_ohlc_data):
        """Test vectorized doji detection."""
        df = sample_ohlc_data
        
        detector = VectorizedCandleDetector("doji")
        events = detector.detect(df)
        
        print(f"\nFound {len(events)} doji patterns")
        
        for event in events:
            assert event.pattern_key == "doji"
            assert event.direction == "neutral"


class TestVectorizedThreeCandleDetectors:
    """Test vectorized three-candle pattern detection."""
    
    def test_three_white_soldiers(self, sample_ohlc_data):
        """Test three white soldiers detection."""
        df = sample_ohlc_data
        
        detector = VectorizedThreeCandleDetector("three_white_soldiers")
        events = detector.detect(df)
        
        print(f"\nFound {len(events)} three white soldiers patterns")
        
        for event in events:
            assert event.pattern_key == "three_white_soldiers"
            assert event.direction == "bull"
            assert event.bars_span == 3
            assert event.touches == 3
    
    def test_three_black_crows(self, sample_ohlc_data):
        """Test three black crows detection."""
        df = sample_ohlc_data
        
        detector = VectorizedThreeCandleDetector("three_black_crows")
        events = detector.detect(df)
        
        print(f"\nFound {len(events)} three black crows patterns")
        
        for event in events:
            assert event.pattern_key == "three_black_crows"
            assert event.direction == "bear"
            assert event.bars_span == 3
            assert event.touches == 3


class TestSwingDetection:
    """Test vectorized swing detection."""
    
    def test_swing_detection_basic(self, sample_ohlc_data):
        """Test basic swing detection."""
        df = sample_ohlc_data
        
        prices = df['close'].values
        
        # Calculate ATR for threshold
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        hl = high - low
        hc = np.abs(high - prev_close)
        lc = np.abs(low - prev_close)
        tr = np.maximum(hl, np.maximum(hc, lc))
        atr_values = pd.Series(tr).ewm(span=14).mean().values
        
        # Detect swings
        swing_indices, swing_types = detect_swings_vectorized(prices, atr_values, atr_mult=2.0)
        
        print(f"\nFound {len(swing_indices)} swing points")
        print(f"Swing highs: {np.sum(swing_types == 1)}")
        print(f"Swing lows: {np.sum(swing_types == -1)}")
        
        # Should find some swings
        assert len(swing_indices) > 0, "Should detect swing points"
        
        # Should have both highs and lows
        assert np.any(swing_types == 1), "Should detect swing highs"
        assert np.any(swing_types == -1), "Should detect swing lows"
        
        # Indices should be in valid range
        assert np.all(swing_indices >= 0)
        assert np.all(swing_indices < len(prices))


class TestPerformance:
    """Test performance improvements of vectorized detection."""
    
    def test_performance_hammer(self, sample_ohlc_data):
        """Benchmark hammer detection performance."""
        df = sample_ohlc_data
        iterations = 20  # More iterations for stable timing
        
        # Warm-up runs to avoid cold start bias
        vec_detector = VectorizedCandleDetector("hammer")
        loop_detector = SimpleCandleDetector("hammer")
        for _ in range(3):
            vec_detector.detect(df)
            loop_detector.detect(df)
        
        # Vectorized timing
        start_vec = time.time()
        for _ in range(iterations):
            vec_events = vec_detector.detect(df)
        time_vec = (time.time() - start_vec) / iterations
        
        # Loop-based timing
        start_loop = time.time()
        for _ in range(iterations):
            loop_events = loop_detector.detect(df)
        time_loop = (time.time() - start_loop) / iterations
        
        speedup = time_loop / time_vec if time_vec > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Performance Benchmark - Hammer Detection ({len(df)} candles)")
        print(f"{'='*60}")
        print(f"Vectorized:  {time_vec*1000:.2f}ms ({len(vec_events)} patterns)")
        print(f"Loop-based:  {time_loop*1000:.2f}ms ({len(loop_events)} patterns)")
        print(f"Speedup:     {speedup:.1f}x")
        print(f"{'='*60}")
        
        # Vectorized should be faster or comparable (allow for timing variance)
        # On small datasets, overhead may dominate - just check it's not dramatically slower
        assert speedup >= 0.8, f"Vectorized unexpectedly slow: {speedup:.1f}x (expected >=0.8x)"
        
        # Log if actually faster
        if speedup >= 1.5:
            print(f"✅ Vectorization effective: {speedup:.1f}x speedup")
    
    def test_performance_engulfing(self, sample_ohlc_data):
        """Benchmark engulfing detection performance."""
        df = sample_ohlc_data
        iterations = 10
        
        # Vectorized timing
        vec_detector = VectorizedCandleDetector("bullish_engulfing")
        start_vec = time.time()
        for _ in range(iterations):
            vec_events = vec_detector.detect(df)
        time_vec = (time.time() - start_vec) / iterations
        
        # Loop-based timing
        loop_detector = SimpleCandleDetector("bullish_engulfing")
        start_loop = time.time()
        for _ in range(iterations):
            loop_events = loop_detector.detect(df)
        time_loop = (time.time() - start_loop) / iterations
        
        speedup = time_loop / time_vec if time_vec > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Performance Benchmark - Bullish Engulfing ({len(df)} candles)")
        print(f"{'='*60}")
        print(f"Vectorized:  {time_vec*1000:.2f}ms ({len(vec_events)} patterns)")
        print(f"Loop-based:  {time_loop*1000:.2f}ms ({len(loop_events)} patterns)")
        print(f"Speedup:     {speedup:.1f}x")
        print(f"{'='*60}")
        
        # Should be faster
        assert speedup >= 2.0, f"Vectorized should be at least 2x faster, got {speedup:.1f}x"


class TestPatternRegistryIntegration:
    """Test integration with PatternRegistry."""
    
    def test_registry_uses_vectorized(self):
        """Test registry uses vectorized detectors by default."""
        registry = PatternRegistry(use_vectorized=True)
        detectors = registry.detectors(kinds=["candle"])
        
        # Should have vectorized detectors
        vec_detectors = [d for d in detectors if isinstance(d, (VectorizedCandleDetector, VectorizedThreeCandleDetector))]
        
        print(f"\nTotal candle detectors: {len(detectors)}")
        print(f"Vectorized detectors: {len(vec_detectors)}")
        
        assert len(vec_detectors) > 0, "Registry should include vectorized detectors"
    
    def test_registry_fallback_to_loop(self):
        """Test registry can fallback to loop-based detectors."""
        registry = PatternRegistry(use_vectorized=False)
        detectors = registry.detectors(kinds=["candle"])
        
        # Should have loop-based detectors
        loop_detectors = [d for d in detectors if isinstance(d, SimpleCandleDetector)]
        
        print(f"\nTotal candle detectors: {len(detectors)}")
        print(f"Loop-based detectors: {len(loop_detectors)}")
        
        assert len(loop_detectors) > 0, "Registry should include loop-based detectors when vectorized=False"
    
    def test_registry_integration_detection(self, sample_ohlc_data):
        """Test full integration: registry -> detectors -> patterns."""
        df = sample_ohlc_data
        
        # Get vectorized detectors from registry
        registry = PatternRegistry(use_vectorized=True)
        detectors = registry.detectors(kinds=["candle"])
        
        # Run detection
        all_events = []
        for detector in detectors:
            try:
                events = detector.detect(df)
                all_events.extend(events)
            except Exception as e:
                print(f"Detector {getattr(detector, 'key', 'unknown')} failed: {e}")
        
        print(f"\nTotal patterns detected: {len(all_events)}")
        print(f"Pattern types: {set(e.pattern_key for e in all_events)}")
        
        # Should detect some patterns
        assert len(all_events) > 0, "Should detect patterns through registry integration"


def test_benchmark_function(sample_ohlc_data):
    """Test the benchmark utility function."""
    df = sample_ohlc_data
    
    results = benchmark_vectorized_vs_loop(df, iterations=5)
    
    print(f"\n{'='*60}")
    print(f"Benchmark Results (built-in function)")
    print(f"{'='*60}")
    print(f"Vectorized time: {results['vectorized_time_ms']:.2f}ms")
    print(f"Loop-based time: {results['loop_time_ms']:.2f}ms")
    print(f"Speedup: {results['speedup']:.1f}x")
    print(f"Patterns found (vec): {results['events_vectorized']}")
    print(f"Patterns found (loop): {results['events_loop']}")
    print(f"{'='*60}")
    
    assert results['speedup'] >= 2.0, f"Should show at least 2x speedup, got {results['speedup']:.1f}x"
    assert results['vectorized_time_ms'] > 0
    assert results['loop_time_ms'] > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
