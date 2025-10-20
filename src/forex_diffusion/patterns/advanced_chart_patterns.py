"""
Advanced chart patterns for forex markets.

Implements complex chart formations including:
- Rounding bottom/top (Saucer patterns)
- Island reversals
- Gap patterns
- Price channels with breakouts
- Measured moves
- Three drives pattern
- Five-wave patterns
"""

from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import time_array, atr, safe_tz_convert


class RoundingPatternDetector(DetectorBase):
    """Detect rounding bottom and rounding top patterns"""

    def __init__(self, key: str, min_span: int = 30, max_span: int = 100):
        self.key = key
        self.kind = "chart"
        self.min_span = min_span
        self.max_span = max_span

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

        for end in range(self.min_span, n):
            for span in range(self.min_span, min(self.max_span, end) + 1, 5):
                start = end - span

                if self.key == "rounding_bottom":
                    pattern_found = self._detect_rounding_bottom(lo, start, end, atr_values)
                else:  # rounding_top
                    pattern_found = self._detect_rounding_top(hi, start, end, atr_values)

                if pattern_found:
                    confidence, lowest_point = pattern_found
                    direction = "bull" if self.key == "rounding_bottom" else "bear"
                    magnitude = atr_values[end] * 3

                    evs.append(PatternEvent(
                        self.key, "chart", direction, ts[start], ts[end],
                        "confirmed", confidence, magnitude, 1, span, None, 15,
                        {"lowest_point": lowest_point}
                    ))

        return evs

    def _detect_rounding_bottom(self, lo, start, end, atr_values):
        """Detect U-shaped rounding bottom"""
        data_slice = lo[start:end + 1]
        n = len(data_slice)

        if n < 20:
            return None

        # Find the lowest point
        min_idx = np.argmin(data_slice)
        min_val = data_slice[min_idx]

        # Check for gradual decline and rise
        left_side = data_slice[:min_idx + 1]
        right_side = data_slice[min_idx:]

        if len(left_side) < 5 or len(right_side) < 5:
            return None

        # Check left side decline
        left_slope = self._calculate_average_slope(left_side)
        right_slope = self._calculate_average_slope(right_side)

        # Rounding bottom should have negative left slope and positive right slope
        if left_slope >= 0 or right_slope <= 0:
            return None

        # Check for gradual curvature (not sharp V)
        curvature_score = self._calculate_curvature_score(data_slice, min_idx)

        confidence = 0.4 + curvature_score * 0.4
        if confidence > 0.6:
            return confidence, start + min_idx

        return None

    def _detect_rounding_top(self, hi, start, end, atr_values):
        """Detect inverted U-shaped rounding top"""
        data_slice = hi[start:end + 1]
        n = len(data_slice)

        if n < 20:
            return None

        # Find the highest point
        max_idx = np.argmax(data_slice)
        max_val = data_slice[max_idx]

        # Check for gradual rise and decline
        left_side = data_slice[:max_idx + 1]
        right_side = data_slice[max_idx:]

        if len(left_side) < 5 or len(right_side) < 5:
            return None

        # Check slopes
        left_slope = self._calculate_average_slope(left_side)
        right_slope = self._calculate_average_slope(right_side)

        # Rounding top should have positive left slope and negative right slope
        if left_slope <= 0 or right_slope >= 0:
            return None

        # Check for gradual curvature
        curvature_score = self._calculate_curvature_score(-data_slice, max_idx)  # Invert for top

        confidence = 0.4 + curvature_score * 0.4
        if confidence > 0.6:
            return confidence, start + max_idx

        return None

    def _calculate_average_slope(self, data):
        """Calculate average slope of data series"""
        if len(data) < 2:
            return 0
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        return slope

    def _calculate_curvature_score(self, data, center_idx):
        """Calculate how smooth/rounded the pattern is"""
        if center_idx < 2 or center_idx >= len(data) - 2:
            return 0

        # Check if adjacent points create smooth curve
        smoothness = 0
        window = min(5, center_idx, len(data) - center_idx - 1)

        for i in range(1, window):
            left_diff = abs(data[center_idx - i] - data[center_idx - i + 1])
            right_diff = abs(data[center_idx + i] - data[center_idx + i - 1])

            # Reward gradual changes
            if left_diff < np.std(data) and right_diff < np.std(data):
                smoothness += 1

        return smoothness / (window - 1) if window > 1 else 0


class IslandReversalDetector(DetectorBase):
    """Detect island reversal patterns"""

    def __init__(self, key: str, min_gap_size: float = 0.5):
        self.key = key
        self.kind = "chart"
        self.min_gap_size = min_gap_size  # Minimum gap size as ratio of ATR

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        if df is None or len(df) < 10:
            return []

        evs: List[PatternEvent] = []
        ts = time_array(df)
        ts = safe_tz_convert(ts, None)

        hi = df["high"].astype(float).to_numpy()
        lo = df["low"].astype(float).to_numpy()
        atr_values = atr(df, 14).to_numpy()

        n = len(df)

        for i in range(5, n - 5):
            if self.key == "island_reversal_top":
                island_found = self._detect_island_top(hi, lo, i, atr_values)
            else:  # island_reversal_bottom
                island_found = self._detect_island_bottom(hi, lo, i, atr_values)

            if island_found:
                confidence, start_idx, end_idx = island_found
                direction = "bear" if self.key == "island_reversal_top" else "bull"
                magnitude = atr_values[i] * 2

                evs.append(PatternEvent(
                    self.key, "chart", direction, ts[start_idx], ts[end_idx],
                    "confirmed", confidence, magnitude, 1, end_idx - start_idx, None, 20,
                    {"island_center": i}
                ))

        return evs

    def _detect_island_top(self, hi, lo, center, atr_values):
        """Detect island reversal top"""
        # Look for gap up followed by gap down
        if center < 2 or center >= len(hi) - 2:
            return None

        # Gap up: current low > previous high
        gap_up = lo[center - 1] > hi[center - 2]

        # Gap down: next high < current low
        gap_down = hi[center + 1] < lo[center]

        if gap_up and gap_down:
            # Check gap sizes
            gap_up_size = lo[center - 1] - hi[center - 2]
            gap_down_size = lo[center] - hi[center + 1]

            min_gap = self.min_gap_size * atr_values[center]

            if gap_up_size >= min_gap and gap_down_size >= min_gap:
                confidence = 0.6 + min(gap_up_size, gap_down_size) / (atr_values[center] * 2)
                return min(confidence, 1.0), center - 2, center + 2

        return None

    def _detect_island_bottom(self, hi, lo, center, atr_values):
        """Detect island reversal bottom"""
        if center < 2 or center >= len(hi) - 2:
            return None

        # Gap down: current high < previous low
        gap_down = hi[center - 1] < lo[center - 2]

        # Gap up: next low > current high
        gap_up = lo[center + 1] > hi[center]

        if gap_down and gap_up:
            # Check gap sizes
            gap_down_size = lo[center - 2] - hi[center - 1]
            gap_up_size = lo[center + 1] - hi[center]

            min_gap = self.min_gap_size * atr_values[center]

            if gap_down_size >= min_gap and gap_up_size >= min_gap:
                confidence = 0.6 + min(gap_down_size, gap_up_size) / (atr_values[center] * 2)
                return min(confidence, 1.0), center - 2, center + 2

        return None


class MeasuredMoveDetector(DetectorBase):
    """Detect measured move patterns (AB=CD type moves)"""

    def __init__(self, key: str, min_span: int = 20, max_span: int = 80):
        self.key = key
        self.kind = "chart"
        self.min_span = min_span
        self.max_span = max_span

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        if df is None or len(df) < self.min_span:
            return []

        evs: List[PatternEvent] = []
        ts = time_array(df)
        ts = safe_tz_convert(ts, None)

        hi = df["high"].astype(float).to_numpy()
        lo = df["low"].astype(float).to_numpy()
        atr_values = atr(df, 14).to_numpy()

        n = len(df)

        for end in range(self.min_span, n):
            for span in range(self.min_span, min(self.max_span, end) + 1, 3):
                start = end - span

                measured_move = self._detect_measured_move(hi, lo, start, end)

                if measured_move:
                    confidence, direction, points = measured_move
                    magnitude = abs(points[-1][1] - points[0][1])

                    evs.append(PatternEvent(
                        self.key, "chart", direction, ts[points[0][0]], ts[points[-1][0]],
                        "confirmed", confidence, magnitude, 4, span, None, 18,
                        {"move_points": points}
                    ))

        return evs

    def _detect_measured_move(self, hi, lo, start, end):
        """Detect AB=CD measured move pattern"""
        # Find swing points
        swing_highs = []
        swing_lows = []

        lookback = 3
        for i in range(start + lookback, end - lookback):
            if all(hi[i] >= hi[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_highs.append((i, hi[i]))
            if all(lo[i] <= lo[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_lows.append((i, lo[i]))

        # Try to form ABCD pattern
        all_swings = swing_highs + swing_lows
        all_swings.sort()

        if len(all_swings) >= 4:
            for i in range(len(all_swings) - 3):
                points = all_swings[i:i + 4]

                # Check alternating pattern
                is_alternating = True
                for j in range(len(points) - 1):
                    curr_is_high = points[j] in swing_highs
                    next_is_high = points[j + 1] in swing_highs
                    if curr_is_high == next_is_high:
                        is_alternating = False
                        break

                if is_alternating:
                    # Check measured move relationship
                    A, B, C, D = points

                    AB_move = abs(B[1] - A[1])
                    CD_move = abs(D[1] - C[1])

                    # AB should approximately equal CD
                    if AB_move > 0:
                        ratio = CD_move / AB_move
                        if 0.8 <= ratio <= 1.25:  # Allow some tolerance
                            confidence = 0.5 + (1 - abs(ratio - 1.0)) * 0.3
                            direction = "bull" if D[1] > A[1] else "bear"
                            return confidence, direction, points

        return None


class ThreeDrivesDetector(DetectorBase):
    """Detect Three Drives pattern"""

    def __init__(self, key: str, min_span: int = 40, max_span: int = 120):
        self.key = key
        self.kind = "chart"
        self.min_span = min_span
        self.max_span = max_span

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        if df is None or len(df) < self.min_span:
            return []

        evs: List[PatternEvent] = []
        ts = time_array(df)
        ts = safe_tz_convert(ts, None)

        hi = df["high"].astype(float).to_numpy()
        lo = df["low"].astype(float).to_numpy()
        atr_values = atr(df, 14).to_numpy()

        n = len(df)

        for end in range(self.min_span, n):
            for span in range(self.min_span, min(self.max_span, end) + 1, 5):
                start = end - span

                three_drives = self._detect_three_drives(hi, lo, start, end)

                if three_drives:
                    confidence, direction, drives = three_drives
                    magnitude = abs(drives[-1][1] - drives[0][1])

                    evs.append(PatternEvent(
                        self.key, "chart", direction, ts[drives[0][0]], ts[drives[-1][0]],
                        "confirmed", confidence, magnitude, 7, span, None, 25,
                        {"drives": drives}
                    ))

        return evs

    def _detect_three_drives(self, hi, lo, start, end):
        """Detect three drives pattern (3 successive waves)"""
        # Find swing points
        swing_highs = []
        swing_lows = []

        lookback = 4
        for i in range(start + lookback, end - lookback):
            if all(hi[i] >= hi[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_highs.append((i, hi[i]))
            if all(lo[i] <= lo[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_lows.append((i, lo[i]))

        all_swings = swing_highs + swing_lows
        all_swings.sort()

        # Need 7 points for three drives pattern
        if len(all_swings) >= 7:
            for i in range(len(all_swings) - 6):
                points = all_swings[i:i + 7]

                # Check alternating pattern
                is_alternating = True
                for j in range(len(points) - 1):
                    curr_is_high = points[j] in swing_highs
                    next_is_high = points[j + 1] in swing_highs
                    if curr_is_high == next_is_high:
                        is_alternating = False
                        break

                if is_alternating:
                    # Validate three drives structure
                    if self._validate_three_drives_structure(points):
                        confidence = 0.7
                        direction = "bear" if points[0] in swing_highs else "bull"
                        return confidence, direction, points

        return None

    def _validate_three_drives_structure(self, points):
        """Validate that points form proper three drives structure"""
        if len(points) != 7:
            return False

        # Three drives should show progressive extension
        # For bullish: higher highs and higher lows
        # For bearish: lower lows and lower highs

        # Extract the drive points (1st, 3rd, 5th, 7th)
        drive1, drive2, drive3 = points[0], points[2], points[4]
        final_drive = points[6]

        # Check for progression in drives
        if points[0][1] < points[2][1] < points[4][1] < points[6][1]:  # Bullish drives
            return True
        elif points[0][1] > points[2][1] > points[4][1] > points[6][1]:  # Bearish drives
            return True

        return False


def make_advanced_chart_pattern_detectors() -> List[DetectorBase]:
    """Create all advanced chart pattern detectors."""
    patterns = [
        RoundingPatternDetector("rounding_bottom"),
        RoundingPatternDetector("rounding_top"),
        IslandReversalDetector("island_reversal_top"),
        IslandReversalDetector("island_reversal_bottom"),
        MeasuredMoveDetector("measured_move"),
        ThreeDrivesDetector("three_drives")
    ]

    return patterns