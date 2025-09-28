"""
Elliott Wave pattern detection for forex markets.

Implements simplified Elliott Wave pattern recognition including:
- 5-wave impulse patterns (bull/bear)
- 3-wave corrective patterns (ABC)
- Wave degree analysis
- Fibonacci ratios validation

Note: This is a simplified implementation focusing on clear wave structures
without complex wave counting algorithms.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import time_array, safe_tz_convert, fit_line_indices, atr


class ElliottWaveDetector(DetectorBase):
    """Elliott Wave pattern detector"""

    def __init__(self, key: str, min_span: int = 50, max_span: int = 200,
                 min_wave_size: float = 0.5, fibonacci_tolerance: float = 0.2):
        self.key = key
        self.kind = "chart"
        self.min_span = min_span
        self.max_span = max_span
        self.min_wave_size = min_wave_size  # Minimum wave size as ratio of ATR
        self.fibonacci_tolerance = fibonacci_tolerance

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        if df is None or len(df) < self.min_span:
            return []

        evs: List[PatternEvent] = []
        ts = time_array(df)
        ts = safe_tz_convert(ts, None)

        hi = df["high"].astype(float).to_numpy()
        lo = df["low"].astype(float).to_numpy()
        cl = df["close"].astype(float).to_numpy()
        atr_values = atr(df, 14).to_numpy()

        n = len(df)

        if self.key in ("elliott_impulse_bull", "elliott_impulse_bear"):
            evs.extend(self._detect_impulse_waves(hi, lo, cl, atr_values, ts, n))
        elif self.key in ("elliott_corrective_abc", "elliott_corrective_wxy"):
            evs.extend(self._detect_corrective_waves(hi, lo, cl, atr_values, ts, n))

        return evs

    def _detect_impulse_waves(self, hi, lo, cl, atr_values, ts, n):
        """Detect 5-wave impulse patterns"""
        evs = []

        for end in range(self.min_span, n):
            for span in range(self.min_span, min(self.max_span, end) + 1, 10):
                start = end - span

                # Find potential wave points for bullish impulse
                if self.key == "elliott_impulse_bull":
                    waves = self._find_bull_impulse_waves(hi, lo, start, end)
                else:
                    waves = self._find_bear_impulse_waves(hi, lo, start, end)

                if waves and self._validate_impulse_waves(waves, atr_values, start):
                    confidence = self._calculate_wave_confidence(waves, atr_values, start)
                    if confidence > 0.6:
                        direction = "bull" if self.key == "elliott_impulse_bull" else "bear"
                        magnitude = abs(waves[-1][1] - waves[0][1])  # Total move

                        evs.append(PatternEvent(
                            self.key, "chart", direction, ts[waves[0][0]], ts[waves[-1][0]],
                            "confirmed", confidence, magnitude, 5, span, None, 25,
                            {"waves": waves, "wave_count": 5}
                        ))

        return evs

    def _find_bull_impulse_waves(self, hi, lo, start, end):
        """Find 5-wave bullish impulse pattern"""
        # Look for pattern: Low -> High -> Low -> High -> Low -> High
        # Waves: 1(up), 2(down), 3(up), 4(down), 5(up)

        # Find major swing points
        swing_highs = []
        swing_lows = []

        # Simple swing detection with lookback/lookahead of 5
        lookback = 5
        for i in range(start + lookback, end - lookback):
            # Check for swing high
            if all(hi[i] >= hi[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_highs.append((i, hi[i]))

            # Check for swing low
            if all(lo[i] <= lo[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_lows.append((i, lo[i]))

        # Try to construct 5-wave pattern
        if len(swing_lows) >= 3 and len(swing_highs) >= 2:
            # Sort by index
            swing_lows.sort()
            swing_highs.sort()

            # Pattern: low(0) -> high(1) -> low(2) -> high(3) -> low(4) -> high(5)
            for i in range(len(swing_lows) - 2):
                for j in range(len(swing_highs) - 1):
                    low0_idx, low0_val = swing_lows[i]

                    # Find first high after low0
                    high1_candidates = [h for h in swing_highs if h[0] > low0_idx]
                    if not high1_candidates:
                        continue
                    high1_idx, high1_val = high1_candidates[0]

                    # Find second low after high1
                    low2_candidates = [l for l in swing_lows if l[0] > high1_idx and l[1] > low0_val]
                    if not low2_candidates:
                        continue
                    low2_idx, low2_val = low2_candidates[0]

                    # Find second high after low2
                    high3_candidates = [h for h in swing_highs if h[0] > low2_idx and h[1] > high1_val]
                    if not high3_candidates:
                        continue
                    high3_idx, high3_val = high3_candidates[0]

                    # Find third low after high3
                    low4_candidates = [l for l in swing_lows if l[0] > high3_idx and l[1] > low2_val]
                    if not low4_candidates:
                        continue
                    low4_idx, low4_val = low4_candidates[0]

                    # Find final high after low4
                    high5_candidates = [h for h in swing_highs if h[0] > low4_idx and h[1] > high3_val]
                    if not high5_candidates:
                        continue
                    high5_idx, high5_val = high5_candidates[0]

                    # Construct wave pattern
                    waves = [
                        (low0_idx, low0_val),   # Start
                        (high1_idx, high1_val), # Wave 1 top
                        (low2_idx, low2_val),   # Wave 2 bottom
                        (high3_idx, high3_val), # Wave 3 top
                        (low4_idx, low4_val),   # Wave 4 bottom
                        (high5_idx, high5_val)  # Wave 5 top
                    ]

                    return waves

        return None

    def _find_bear_impulse_waves(self, hi, lo, start, end):
        """Find 5-wave bearish impulse pattern"""
        # Look for pattern: High -> Low -> High -> Low -> High -> Low
        # Waves: 1(down), 2(up), 3(down), 4(up), 5(down)

        swing_highs = []
        swing_lows = []

        lookback = 5
        for i in range(start + lookback, end - lookback):
            if all(hi[i] >= hi[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_highs.append((i, hi[i]))
            if all(lo[i] <= lo[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_lows.append((i, lo[i]))

        if len(swing_highs) >= 3 and len(swing_lows) >= 2:
            swing_highs.sort()
            swing_lows.sort()

            # Pattern: high(0) -> low(1) -> high(2) -> low(3) -> high(4) -> low(5)
            for i in range(len(swing_highs) - 2):
                high0_idx, high0_val = swing_highs[i]

                # Find first low after high0
                low1_candidates = [l for l in swing_lows if l[0] > high0_idx]
                if not low1_candidates:
                    continue
                low1_idx, low1_val = low1_candidates[0]

                # Find second high after low1
                high2_candidates = [h for h in swing_highs if h[0] > low1_idx and h[1] < high0_val]
                if not high2_candidates:
                    continue
                high2_idx, high2_val = high2_candidates[0]

                # Find second low after high2
                low3_candidates = [l for l in swing_lows if l[0] > high2_idx and l[1] < low1_val]
                if not low3_candidates:
                    continue
                low3_idx, low3_val = low3_candidates[0]

                # Find third high after low3
                high4_candidates = [h for h in swing_highs if h[0] > low3_idx and h[1] < high2_val]
                if not high4_candidates:
                    continue
                high4_idx, high4_val = high4_candidates[0]

                # Find final low after high4
                low5_candidates = [l for l in swing_lows if l[0] > high4_idx and l[1] < low3_val]
                if not low5_candidates:
                    continue
                low5_idx, low5_val = low5_candidates[0]

                waves = [
                    (high0_idx, high0_val), # Start
                    (low1_idx, low1_val),   # Wave 1 bottom
                    (high2_idx, high2_val), # Wave 2 top
                    (low3_idx, low3_val),   # Wave 3 bottom
                    (high4_idx, high4_val), # Wave 4 top
                    (low5_idx, low5_val)    # Wave 5 bottom
                ]

                return waves

        return None

    def _detect_corrective_waves(self, hi, lo, cl, atr_values, ts, n):
        """Detect 3-wave corrective patterns (ABC)"""
        evs = []

        for end in range(self.min_span, n):
            for span in range(self.min_span, min(self.max_span, end) + 1, 10):
                start = end - span

                # Find ABC corrective waves
                abc_waves = self._find_abc_correction(hi, lo, start, end)

                if abc_waves and self._validate_abc_waves(abc_waves, atr_values, start):
                    confidence = self._calculate_abc_confidence(abc_waves, atr_values, start)
                    if confidence > 0.5:
                        # Determine direction based on overall move
                        total_move = abc_waves[-1][1] - abc_waves[0][1]
                        direction = "bull" if total_move > 0 else "bear"
                        magnitude = abs(total_move)

                        evs.append(PatternEvent(
                            self.key, "chart", direction, ts[abc_waves[0][0]], ts[abc_waves[-1][0]],
                            "confirmed", confidence, magnitude, 3, span, None, 15,
                            {"waves": abc_waves, "wave_count": 3}
                        ))

        return evs

    def _find_abc_correction(self, hi, lo, start, end):
        """Find ABC corrective wave pattern"""
        swing_highs = []
        swing_lows = []

        lookback = 5
        for i in range(start + lookback, end - lookback):
            if all(hi[i] >= hi[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_highs.append((i, hi[i]))
            if all(lo[i] <= lo[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_lows.append((i, lo[i]))

        # Try to form ABC pattern (3 waves)
        all_swings = swing_highs + swing_lows
        all_swings.sort()

        if len(all_swings) >= 4:
            # Take first 4 swings that form alternating pattern
            for i in range(len(all_swings) - 3):
                point_a = all_swings[i]
                point_b = all_swings[i + 1]
                point_c = all_swings[i + 2]
                point_d = all_swings[i + 3]

                # Check if it's alternating high-low or low-high pattern
                a_is_high = point_a in swing_highs
                b_is_high = point_b in swing_highs
                c_is_high = point_c in swing_highs
                d_is_high = point_d in swing_highs

                if a_is_high != b_is_high and b_is_high != c_is_high and c_is_high != d_is_high:
                    return [point_a, point_b, point_c, point_d]

        return None

    def _validate_impulse_waves(self, waves, atr_values, start):
        """Validate impulse wave structure"""
        if not waves or len(waves) != 6:
            return False

        # Check minimum wave sizes
        for i in range(len(waves) - 1):
            wave_size = abs(waves[i + 1][1] - waves[i][1])
            min_size = self.min_wave_size * atr_values[waves[i][0]]
            if wave_size < min_size:
                return False

        # Elliott Wave rules:
        # 1. Wave 2 never retraces more than 100% of wave 1
        wave1_size = abs(waves[1][1] - waves[0][1])
        wave2_retrace = abs(waves[2][1] - waves[1][1])
        if wave2_retrace >= wave1_size:
            return False

        # 2. Wave 3 is never the shortest
        wave3_size = abs(waves[3][1] - waves[2][1])
        wave5_size = abs(waves[5][1] - waves[4][1])
        if wave3_size < wave1_size and wave3_size < wave5_size:
            return False

        # 3. Wave 4 doesn't overlap with wave 1 territory
        if self.key == "elliott_impulse_bull":
            if waves[4][1] <= waves[1][1]:  # Wave 4 low shouldn't go below wave 1 high
                return False
        else:
            if waves[4][1] >= waves[1][1]:  # Wave 4 high shouldn't go above wave 1 low
                return False

        return True

    def _validate_abc_waves(self, waves, atr_values, start):
        """Validate ABC corrective wave structure"""
        if not waves or len(waves) != 4:
            return False

        # Check minimum wave sizes
        for i in range(len(waves) - 1):
            wave_size = abs(waves[i + 1][1] - waves[i][1])
            min_size = self.min_wave_size * atr_values[waves[i][0]]
            if wave_size < min_size:
                return False

        return True

    def _calculate_wave_confidence(self, waves, atr_values, start):
        """Calculate confidence based on Elliott Wave guidelines"""
        confidence = 0.5

        # Fibonacci relationships
        wave1_size = abs(waves[1][1] - waves[0][1])
        wave3_size = abs(waves[3][1] - waves[2][1])
        wave5_size = abs(waves[5][1] - waves[4][1])

        # Wave 3 extension (1.618 of wave 1)
        if self._check_fibonacci_ratio(wave3_size, wave1_size, 1.618):
            confidence += 0.2

        # Wave 5 equality or 0.618 of wave 1+3
        wave13_combined = wave1_size + wave3_size
        if self._check_fibonacci_ratio(wave5_size, wave1_size, 1.0) or \
           self._check_fibonacci_ratio(wave5_size, wave13_combined, 0.618):
            confidence += 0.15

        # Wave structure quality
        if wave3_size > wave1_size and wave3_size > wave5_size:
            confidence += 0.1

        return min(confidence, 1.0)

    def _calculate_abc_confidence(self, waves, atr_values, start):
        """Calculate confidence for ABC correction"""
        confidence = 0.4

        wave_a = abs(waves[1][1] - waves[0][1])
        wave_c = abs(waves[3][1] - waves[2][1])

        # C wave equality or Fibonacci relationship to A wave
        if self._check_fibonacci_ratio(wave_c, wave_a, 1.0):
            confidence += 0.2
        elif self._check_fibonacci_ratio(wave_c, wave_a, 0.618) or \
             self._check_fibonacci_ratio(wave_c, wave_a, 1.618):
            confidence += 0.15

        return min(confidence, 1.0)

    def _check_fibonacci_ratio(self, value1, value2, target_ratio):
        """Check if ratio is within tolerance of Fibonacci level"""
        if value2 == 0:
            return False
        actual_ratio = value1 / value2
        return abs(actual_ratio - target_ratio) <= self.fibonacci_tolerance * target_ratio


def make_elliott_wave_detectors() -> List[ElliottWaveDetector]:
    """Create Elliott Wave pattern detectors"""
    patterns = [
        "elliott_impulse_bull",
        "elliott_impulse_bear",
        "elliott_corrective_abc"
    ]

    return [ElliottWaveDetector(pattern) for pattern in patterns]