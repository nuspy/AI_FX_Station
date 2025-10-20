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
from typing import List
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import time_array, safe_tz_convert, atr


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

        # PERFORMANCE FIX: Pre-compute swing points once for entire dataset
        # This reduces complexity from O(n^6) to O(n^2)
        swing_highs, swing_lows = self._find_all_swing_points(hi, lo, n)

        # Use sliding window with larger steps to reduce iterations
        step_size = max(10, self.min_span // 5)  # Adaptive step size

        for end in range(self.min_span, n, step_size):
            # Limit span iterations to avoid O(n^2) behavior
            max_spans_to_check = 5  # Only check a few span sizes
            span_step = max(20, (self.max_span - self.min_span) // max_spans_to_check)

            for span in range(self.min_span, min(self.max_span, end) + 1, span_step):
                start = end - span

                # Find potential wave points using pre-computed swings
                if self.key == "elliott_impulse_bull":
                    waves = self._find_bull_impulse_waves_optimized(swing_highs, swing_lows, start, end)
                else:
                    waves = self._find_bear_impulse_waves_optimized(swing_highs, swing_lows, start, end)

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

    def _find_all_swing_points(self, hi, lo, n):
        """Pre-compute all swing points once for better performance"""
        swing_highs = []
        swing_lows = []
        lookback = 5

        for i in range(lookback, n - lookback):
            # Check for swing high
            if all(hi[i] >= hi[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_highs.append((i, hi[i]))

            # Check for swing low
            if all(lo[i] <= lo[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_lows.append((i, lo[i]))

        return swing_highs, swing_lows

    def _find_bull_impulse_waves_optimized(self, swing_highs, swing_lows, start, end):
        """Optimized bullish wave finding using pre-computed swing points"""
        # Filter swing points to the current window
        window_highs = [(idx, val) for idx, val in swing_highs if start <= idx <= end]
        window_lows = [(idx, val) for idx, val in swing_lows if start <= idx <= end]

        if len(window_lows) < 3 or len(window_highs) < 2:
            return None

        # Sort by index
        window_lows.sort()
        window_highs.sort()

        # Simple greedy approach - find first valid 5-wave pattern
        # This is much faster than the nested O(n^4) approach
        for i, (low0_idx, low0_val) in enumerate(window_lows[:-2]):
            # Find first high after low0
            high1 = next(((idx, val) for idx, val in window_highs if idx > low0_idx), None)
            if not high1:
                continue
            high1_idx, high1_val = high1

            # Find second low after high1 (wave 2)
            low2 = next(((idx, val) for idx, val in window_lows[i+1:] if idx > high1_idx and val > low0_val * 0.8), None)
            if not low2:
                continue
            low2_idx, low2_val = low2

            # Find second high after low2 (wave 3) - should be highest
            high3 = next(((idx, val) for idx, val in window_highs if idx > low2_idx and val > high1_val), None)
            if not high3:
                continue
            high3_idx, high3_val = high3

            # Find third low after high3 (wave 4)
            low4 = next(((idx, val) for idx, val in window_lows if idx > high3_idx and val > low2_val * 0.8), None)
            if not low4:
                continue
            low4_idx, low4_val = low4

            # Find final high after low4 (wave 5)
            high5 = next(((idx, val) for idx, val in window_highs if idx > low4_idx and val > high3_val * 0.8), None)
            if not high5:
                continue
            high5_idx, high5_val = high5

            # Return first valid pattern found
            return [
                (low0_idx, low0_val),   # Start
                (high1_idx, high1_val), # Wave 1 top
                (low2_idx, low2_val),   # Wave 2 bottom
                (high3_idx, high3_val), # Wave 3 top
                (low4_idx, low4_val),   # Wave 4 bottom
                (high5_idx, high5_val)  # Wave 5 top
            ]

        return None

    def _find_bear_impulse_waves_optimized(self, swing_highs, swing_lows, start, end):
        """Optimized bearish wave finding using pre-computed swing points"""
        # Filter swing points to the current window
        window_highs = [(idx, val) for idx, val in swing_highs if start <= idx <= end]
        window_lows = [(idx, val) for idx, val in swing_lows if start <= idx <= end]

        if len(window_highs) < 3 or len(window_lows) < 2:
            return None

        # Sort by index
        window_highs.sort()
        window_lows.sort()

        # Simple greedy approach for bearish pattern
        for i, (high0_idx, high0_val) in enumerate(window_highs[:-2]):
            # Find first low after high0
            low1 = next(((idx, val) for idx, val in window_lows if idx > high0_idx), None)
            if not low1:
                continue
            low1_idx, low1_val = low1

            # Find second high after low1 (wave 2)
            high2 = next(((idx, val) for idx, val in window_highs[i+1:] if idx > low1_idx and val < high0_val * 1.2), None)
            if not high2:
                continue
            high2_idx, high2_val = high2

            # Find second low after high2 (wave 3) - should be lowest
            low3 = next(((idx, val) for idx, val in window_lows if idx > high2_idx and val < low1_val), None)
            if not low3:
                continue
            low3_idx, low3_val = low3

            # Find third high after low3 (wave 4)
            high4 = next(((idx, val) for idx, val in window_highs if idx > low3_idx and val < high2_val * 1.2), None)
            if not high4:
                continue
            high4_idx, high4_val = high4

            # Find final low after high4 (wave 5)
            low5 = next(((idx, val) for idx, val in window_lows if idx > high4_idx and val < low3_val * 1.2), None)
            if not low5:
                continue
            low5_idx, low5_val = low5

            # Return first valid pattern found
            return [
                (high0_idx, high0_val), # Start
                (low1_idx, low1_val),   # Wave 1 bottom
                (high2_idx, high2_val), # Wave 2 top
                (low3_idx, low3_val),   # Wave 3 bottom
                (high4_idx, high4_val), # Wave 4 top
                (low5_idx, low5_val)    # Wave 5 bottom
            ]

        return None

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

        # PERFORMANCE FIX: Pre-compute swing points once
        swing_highs, swing_lows = self._find_all_swing_points(hi, lo, n)

        # Use larger steps to reduce iterations
        step_size = max(15, self.min_span // 4)

        for end in range(self.min_span, n, step_size):
            # Limit span iterations
            max_spans_to_check = 4
            span_step = max(25, (self.max_span - self.min_span) // max_spans_to_check)

            for span in range(self.min_span, min(self.max_span, end) + 1, span_step):
                start = end - span

                # Find ABC corrective waves using optimized method
                abc_waves = self._find_abc_correction_optimized(swing_highs, swing_lows, start, end)

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

    def _find_abc_correction_optimized(self, swing_highs, swing_lows, start, end):
        """Optimized ABC corrective wave finding"""
        # Filter swing points to window
        window_highs = [(idx, val) for idx, val in swing_highs if start <= idx <= end]
        window_lows = [(idx, val) for idx, val in swing_lows if start <= idx <= end]

        all_swings = window_highs + window_lows
        all_swings.sort()

        if len(all_swings) < 4:
            return None

        # Simple greedy approach - find first valid ABC pattern
        for i in range(len(all_swings) - 3):
            point_a = all_swings[i]
            point_b = all_swings[i + 1]
            point_c = all_swings[i + 2]
            point_d = all_swings[i + 3]

            # Check if pattern forms valid ABC correction
            # A should be start, B peak/trough, C partial retracement, D end
            if self._is_valid_abc_pattern(point_a, point_b, point_c, point_d):
                return [point_a, point_b, point_c, point_d]

        return None

    def _is_valid_abc_pattern(self, a, b, c, d):
        """Check if 4 points form valid ABC correction"""
        a_idx, a_val = a
        b_idx, b_val = b
        c_idx, c_val = c
        d_idx, d_val = d

        # Basic structure checks
        if not (a_idx < b_idx < c_idx < d_idx):
            return False

        # Check for alternating high/low pattern
        if a_val < b_val:  # A to B is up
            if c_val >= b_val or d_val <= c_val:  # B to C should be down, C to D should be up
                return False
        else:  # A to B is down
            if c_val <= b_val or d_val >= c_val:  # B to C should be up, C to D should be down
                return False

        # Check retracement ratios (simple validation)
        ab_move = abs(b_val - a_val)
        bc_move = abs(c_val - b_val)
        cd_move = abs(d_val - c_val)

        # B to C should be significant retracement (30-80% of AB)
        bc_ratio = bc_move / ab_move if ab_move > 0 else 0
        if not (0.3 <= bc_ratio <= 0.8):
            return False

        # C to D should extend beyond C but not too far
        if cd_move < ab_move * 0.2:  # Minimum extension
            return False

        return True

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