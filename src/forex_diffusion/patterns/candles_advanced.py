"""
Advanced candlestick patterns without volume dependency.

Implements complex multi-candle formations including Morning/Evening Star,
Abandoned Baby, Three Line Strike, Three Inside/Outside patterns, and others.
All patterns are designed to work without volume data.
"""

from __future__ import annotations
from typing import List
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import time_array, safe_tz_convert


def _body(o, c):
    """Calculate body size"""
    return abs(float(c) - float(o))


def _tr(h, l, pc):
    """Calculate true range"""
    return max(float(h) - float(l), abs(float(h) - float(pc)), abs(float(l) - float(pc)))


def _is_doji(o, c, h, l, doji_threshold=0.1):
    """Check if candle is a doji"""
    body = _body(o, c)
    total_range = h - l
    return body <= doji_threshold * total_range if total_range > 0 else False


def _is_bullish(o, c):
    """Check if candle is bullish"""
    return c > o


def _is_bearish(o, c):
    """Check if candle is bearish"""
    return c < o


def _upper_shadow(h, o, c):
    """Calculate upper shadow length"""
    return h - max(o, c)


def _lower_shadow(o, c, l):
    """Calculate lower shadow length"""
    return min(o, c) - l


class AdvancedCandleDetector(DetectorBase):
    """Advanced multi-candle pattern detector"""

    def __init__(self, key: str):
        self.key = key
        self.kind = "candle"

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        if df is None or len(df) < 5:
            return []

        evs: List[PatternEvent] = []
        ts = time_array(df)
        ts = safe_tz_convert(ts, None)

        o = df["open"].astype(float).to_numpy()
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        c = df["close"].astype(float).to_numpy()

        n = len(df)

        if self.key in ("morning_star", "evening_star"):
            evs.extend(self._detect_star_patterns(o, h, l, c, ts, n))
        elif self.key in ("abandoned_baby_bull", "abandoned_baby_bear"):
            evs.extend(self._detect_abandoned_baby(o, h, l, c, ts, n))
        elif self.key in ("three_line_strike_bull", "three_line_strike_bear"):
            evs.extend(self._detect_three_line_strike(o, h, l, c, ts, n))
        elif self.key in ("three_inside_up", "three_inside_down"):
            evs.extend(self._detect_three_inside(o, h, l, c, ts, n))
        elif self.key in ("three_outside_up", "three_outside_down"):
            evs.extend(self._detect_three_outside(o, h, l, c, ts, n))
        elif self.key in ("unique_three_river_bottom", "unique_three_river_top"):
            evs.extend(self._detect_unique_three_river(o, h, l, c, ts, n))
        elif self.key in ("concealing_baby_swallow", "identical_three_crows"):
            evs.extend(self._detect_special_patterns(o, h, l, c, ts, n))

        return evs

    def _detect_star_patterns(self, o, h, l, c, ts, n):
        """Detect Morning Star and Evening Star patterns"""
        evs = []

        for i in range(2, n):
            # Three candle pattern: i-2, i-1, i
            i1, i2, i3 = i-2, i-1, i

            # Check for morning star (bullish reversal)
            if self.key == "morning_star":
                # First candle: long bearish
                first_bearish = _is_bearish(o[i1], c[i1]) and _body(o[i1], c[i1]) > 0.6 * (h[i1] - l[i1])

                # Second candle: small body (doji or spinning top), gaps down
                second_small = _body(o[i2], c[i2]) < 0.3 * _body(o[i1], c[i1])
                second_gaps_down = max(o[i2], c[i2]) < min(o[i1], c[i1])

                # Third candle: bullish, closes above midpoint of first candle
                third_bullish = _is_bullish(o[i3], c[i3])
                third_closes_high = c[i3] > (o[i1] + c[i1]) / 2

                if first_bearish and second_small and second_gaps_down and third_bullish and third_closes_high:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bull", ts[i1], ts[i3], "confirmed",
                        0.7, tr, 3, 3, None, 15, {"marker": i3}
                    ))

            # Check for evening star (bearish reversal)
            elif self.key == "evening_star":
                # First candle: long bullish
                first_bullish = _is_bullish(o[i1], c[i1]) and _body(o[i1], c[i1]) > 0.6 * (h[i1] - l[i1])

                # Second candle: small body, gaps up
                second_small = _body(o[i2], c[i2]) < 0.3 * _body(o[i1], c[i1])
                second_gaps_up = min(o[i2], c[i2]) > max(o[i1], c[i1])

                # Third candle: bearish, closes below midpoint of first candle
                third_bearish = _is_bearish(o[i3], c[i3])
                third_closes_low = c[i3] < (o[i1] + c[i1]) / 2

                if first_bullish and second_small and second_gaps_up and third_bearish and third_closes_low:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bear", ts[i1], ts[i3], "confirmed",
                        0.7, tr, 3, 3, None, 15, {"marker": i3}
                    ))

        return evs

    def _detect_abandoned_baby(self, o, h, l, c, ts, n):
        """Detect Abandoned Baby patterns"""
        evs = []

        for i in range(2, n):
            i1, i2, i3 = i-2, i-1, i

            # Middle candle must be a doji
            if not _is_doji(o[i2], c[i2], h[i2], l[i2]):
                continue

            if self.key == "abandoned_baby_bull":
                # Bearish first candle, doji gaps down, bullish third candle gaps up
                first_bearish = _is_bearish(o[i1], c[i1])
                doji_gaps_down = h[i2] < l[i1]  # Complete gap
                third_bullish = _is_bullish(o[i3], c[i3])
                third_gaps_up = l[i3] > h[i2]  # Complete gap

                if first_bearish and doji_gaps_down and third_bullish and third_gaps_up:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bull", ts[i1], ts[i3], "confirmed",
                        0.8, tr, 3, 3, None, 20, {"marker": i3}
                    ))

            elif self.key == "abandoned_baby_bear":
                # Bullish first candle, doji gaps up, bearish third candle gaps down
                first_bullish = _is_bullish(o[i1], c[i1])
                doji_gaps_up = l[i2] > h[i1]  # Complete gap
                third_bearish = _is_bearish(o[i3], c[i3])
                third_gaps_down = h[i3] < l[i2]  # Complete gap

                if first_bullish and doji_gaps_up and third_bearish and third_gaps_down:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bear", ts[i1], ts[i3], "confirmed",
                        0.8, tr, 3, 3, None, 20, {"marker": i3}
                    ))

        return evs

    def _detect_three_line_strike(self, o, h, l, c, ts, n):
        """Detect Three Line Strike patterns"""
        evs = []

        for i in range(3, n):
            i1, i2, i3, i4 = i-3, i-2, i-1, i

            if self.key == "three_line_strike_bull":
                # First three candles: consecutive bearish with lower closes
                three_bearish = all(_is_bearish(o[j], c[j]) for j in [i1, i2, i3])
                descending = c[i1] > c[i2] > c[i3]

                # Fourth candle: bullish, opens below third close, closes above first open
                fourth_bullish = _is_bullish(o[i4], c[i4])
                opens_below = o[i4] < c[i3]
                closes_above = c[i4] > o[i1]

                if three_bearish and descending and fourth_bullish and opens_below and closes_above:
                    tr = _tr(h[i4], l[i4], c[i3])
                    evs.append(PatternEvent(
                        self.key, "candle", "bull", ts[i1], ts[i4], "confirmed",
                        0.75, tr, 4, 4, None, 18, {"marker": i4}
                    ))

            elif self.key == "three_line_strike_bear":
                # First three candles: consecutive bullish with higher closes
                three_bullish = all(_is_bullish(o[j], c[j]) for j in [i1, i2, i3])
                ascending = c[i1] < c[i2] < c[i3]

                # Fourth candle: bearish, opens above third close, closes below first open
                fourth_bearish = _is_bearish(o[i4], c[i4])
                opens_above = o[i4] > c[i3]
                closes_below = c[i4] < o[i1]

                if three_bullish and ascending and fourth_bearish and opens_above and closes_below:
                    tr = _tr(h[i4], l[i4], c[i3])
                    evs.append(PatternEvent(
                        self.key, "candle", "bear", ts[i1], ts[i4], "confirmed",
                        0.75, tr, 4, 4, None, 18, {"marker": i4}
                    ))

        return evs

    def _detect_three_inside(self, o, h, l, c, ts, n):
        """Detect Three Inside Up/Down patterns"""
        evs = []

        for i in range(2, n):
            i1, i2, i3 = i-2, i-1, i

            if self.key == "three_inside_up":
                # First candle: bearish
                # Second candle: bullish harami (inside first candle)
                # Third candle: bullish, closes above first candle high
                first_bearish = _is_bearish(o[i1], c[i1])
                second_bullish = _is_bullish(o[i2], c[i2])
                second_inside = min(o[i2], c[i2]) > min(o[i1], c[i1]) and max(o[i2], c[i2]) < max(o[i1], c[i1])
                third_bullish = _is_bullish(o[i3], c[i3])
                third_closes_above = c[i3] > h[i1]

                if first_bearish and second_bullish and second_inside and third_bullish and third_closes_above:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bull", ts[i1], ts[i3], "confirmed",
                        0.65, tr, 3, 3, None, 12, {"marker": i3}
                    ))

            elif self.key == "three_inside_down":
                # First candle: bullish
                # Second candle: bearish harami (inside first candle)
                # Third candle: bearish, closes below first candle low
                first_bullish = _is_bullish(o[i1], c[i1])
                second_bearish = _is_bearish(o[i2], c[i2])
                second_inside = min(o[i2], c[i2]) > min(o[i1], c[i1]) and max(o[i2], c[i2]) < max(o[i1], c[i1])
                third_bearish = _is_bearish(o[i3], c[i3])
                third_closes_below = c[i3] < l[i1]

                if first_bullish and second_bearish and second_inside and third_bearish and third_closes_below:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bear", ts[i1], ts[i3], "confirmed",
                        0.65, tr, 3, 3, None, 12, {"marker": i3}
                    ))

        return evs

    def _detect_three_outside(self, o, h, l, c, ts, n):
        """Detect Three Outside Up/Down patterns"""
        evs = []

        for i in range(2, n):
            i1, i2, i3 = i-2, i-1, i

            if self.key == "three_outside_up":
                # First two candles form bullish engulfing
                # Third candle: bullish, closes higher than second
                first_bearish = _is_bearish(o[i1], c[i1])
                second_bullish = _is_bullish(o[i2], c[i2])
                engulfing = o[i2] <= c[i1] and c[i2] >= o[i1]
                third_bullish = _is_bullish(o[i3], c[i3])
                third_higher = c[i3] > c[i2]

                if first_bearish and second_bullish and engulfing and third_bullish and third_higher:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bull", ts[i1], ts[i3], "confirmed",
                        0.6, tr, 3, 3, None, 10, {"marker": i3}
                    ))

            elif self.key == "three_outside_down":
                # First two candles form bearish engulfing
                # Third candle: bearish, closes lower than second
                first_bullish = _is_bullish(o[i1], c[i1])
                second_bearish = _is_bearish(o[i2], c[i2])
                engulfing = o[i2] >= c[i1] and c[i2] <= o[i1]
                third_bearish = _is_bearish(o[i3], c[i3])
                third_lower = c[i3] < c[i2]

                if first_bullish and second_bearish and engulfing and third_bearish and third_lower:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bear", ts[i1], ts[i3], "confirmed",
                        0.6, tr, 3, 3, None, 10, {"marker": i3}
                    ))

        return evs

    def _detect_unique_three_river(self, o, h, l, c, ts, n):
        """Detect Unique Three River patterns"""
        evs = []

        for i in range(2, n):
            i1, i2, i3 = i-2, i-1, i

            if self.key == "unique_three_river_bottom":
                # Three consecutive bearish candles with specific characteristics
                three_bearish = all(_is_bearish(o[j], c[j]) for j in [i1, i2, i3])

                # Second candle should be a hammer-like formation
                second_body = _body(o[i2], c[i2])
                second_lower_shadow = _lower_shadow(o[i2], c[i2], l[i2])
                second_hammer_like = second_lower_shadow > 2 * second_body

                # Third candle should close above second candle's close
                third_higher = c[i3] > c[i2]

                if three_bearish and second_hammer_like and third_higher:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bull", ts[i1], ts[i3], "confirmed",
                        0.55, tr, 3, 3, None, 8, {"marker": i3}
                    ))

            elif self.key == "unique_three_river_top":
                # Three consecutive bullish candles with specific characteristics
                three_bullish = all(_is_bullish(o[j], c[j]) for j in [i1, i2, i3])

                # Second candle should be a shooting star-like formation
                second_body = _body(o[i2], c[i2])
                second_upper_shadow = _upper_shadow(h[i2], o[i2], c[i2])
                second_star_like = second_upper_shadow > 2 * second_body

                # Third candle should close below second candle's close
                third_lower = c[i3] < c[i2]

                if three_bullish and second_star_like and third_lower:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bear", ts[i1], ts[i3], "confirmed",
                        0.55, tr, 3, 3, None, 8, {"marker": i3}
                    ))

        return evs

    def _detect_special_patterns(self, o, h, l, c, ts, n):
        """Detect special rare patterns"""
        evs = []

        if self.key == "concealing_baby_swallow":
            # Rare bearish continuation pattern
            for i in range(3, n):
                i1, i2, i3, i4 = i-3, i-2, i-1, i

                # Four consecutive bearish candles
                four_bearish = all(_is_bearish(o[j], c[j]) for j in [i1, i2, i3, i4])

                # Second and third candles should gap down
                gap_down_2 = h[i2] < l[i1]
                gap_down_3 = h[i3] < l[i2]

                # Fourth candle engulfs third
                fourth_engulfs = o[i4] > o[i3] and c[i4] < c[i3]

                if four_bearish and gap_down_2 and gap_down_3 and fourth_engulfs:
                    tr = _tr(h[i4], l[i4], c[i3])
                    evs.append(PatternEvent(
                        self.key, "candle", "bear", ts[i1], ts[i4], "confirmed",
                        0.4, tr, 4, 4, None, 5, {"marker": i4}
                    ))

        elif self.key == "identical_three_crows":
            # Bearish pattern with three similar bearish candles
            for i in range(2, n):
                i1, i2, i3 = i-2, i-1, i

                # Three bearish candles
                three_bearish = all(_is_bearish(o[j], c[j]) for j in [i1, i2, i3])

                # Each opens within previous body and closes lower
                second_opens_in_first = min(o[i1], c[i1]) < o[i2] < max(o[i1], c[i1])
                third_opens_in_second = min(o[i2], c[i2]) < o[i3] < max(o[i2], c[i2])

                # Descending closes
                descending = c[i1] > c[i2] > c[i3]

                if three_bearish and second_opens_in_first and third_opens_in_second and descending:
                    tr = _tr(h[i3], l[i3], c[i2])
                    evs.append(PatternEvent(
                        self.key, "candle", "bear", ts[i1], ts[i3], "confirmed",
                        0.65, tr, 3, 3, None, 12, {"marker": i3}
                    ))

        return evs


def make_advanced_candle_detectors() -> List[AdvancedCandleDetector]:
    """Create all advanced candlestick pattern detectors"""
    patterns = [
        "morning_star",
        "evening_star",
        "abandoned_baby_bull",
        "abandoned_baby_bear",
        "three_line_strike_bull",
        "three_line_strike_bear",
        "three_inside_up",
        "three_inside_down",
        "three_outside_up",
        "three_outside_down",
        "unique_three_river_bottom",
        "unique_three_river_top",
        "concealing_baby_swallow",
        "identical_three_crows"
    ]

    return [AdvancedCandleDetector(pattern) for pattern in patterns]