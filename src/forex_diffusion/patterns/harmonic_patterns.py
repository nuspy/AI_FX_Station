"""
Harmonic pattern detection for forex markets.

Implements advanced harmonic patterns including:
- Gartley pattern (222)
- Butterfly pattern
- Bat pattern
- Crab pattern
- Cypher pattern
- Shark pattern

All patterns use Fibonacci ratios for validation and work without volume data.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import time_array, atr, safe_tz_convert


class HarmonicPatternDetector(DetectorBase):
    """Harmonic pattern detector using XABCD structure"""

    def __init__(self, key: str, min_span: int = 40, max_span: int = 150,
                 fibonacci_tolerance: float = 0.05):
        self.key = key
        self.kind = "chart"
        self.min_span = min_span
        self.max_span = max_span
        self.fibonacci_tolerance = fibonacci_tolerance

        # Define Fibonacci ratios for each pattern
        self.pattern_ratios = {
            "gartley_bull": {
                "AB_XA": (0.618, 0.786),  # AB retracement of XA
                "BC_AB": (0.382, 0.886),  # BC retracement of AB
                "CD_BC": (1.13, 1.618),   # CD extension of BC
                "AD_XA": (0.786, 0.786)   # AD retracement of XA (precise)
            },
            "gartley_bear": {
                "AB_XA": (0.618, 0.786),
                "BC_AB": (0.382, 0.886),
                "CD_BC": (1.13, 1.618),
                "AD_XA": (0.786, 0.786)
            },
            "butterfly_bull": {
                "AB_XA": (0.786, 0.786),
                "BC_AB": (0.382, 0.886),
                "CD_BC": (1.618, 2.618),
                "AD_XA": (1.27, 1.618)
            },
            "butterfly_bear": {
                "AB_XA": (0.786, 0.786),
                "BC_AB": (0.382, 0.886),
                "CD_BC": (1.618, 2.618),
                "AD_XA": (1.27, 1.618)
            },
            "bat_bull": {
                "AB_XA": (0.382, 0.50),
                "BC_AB": (0.382, 0.886),
                "CD_BC": (1.618, 2.618),
                "AD_XA": (0.886, 0.886)
            },
            "bat_bear": {
                "AB_XA": (0.382, 0.50),
                "BC_AB": (0.382, 0.886),
                "CD_BC": (1.618, 2.618),
                "AD_XA": (0.886, 0.886)
            },
            "crab_bull": {
                "AB_XA": (0.382, 0.618),
                "BC_AB": (0.382, 0.886),
                "CD_BC": (2.24, 3.618),
                "AD_XA": (1.618, 1.618)
            },
            "crab_bear": {
                "AB_XA": (0.382, 0.618),
                "BC_AB": (0.382, 0.886),
                "CD_BC": (2.24, 3.618),
                "AD_XA": (1.618, 1.618)
            },
            "cypher_bull": {
                "AB_XA": (0.382, 0.618),
                "BC_AB": (1.13, 1.414),
                "CD_BC": (0.618, 0.786),
                "AD_XA": (0.786, 0.786)
            },
            "cypher_bear": {
                "AB_XA": (0.382, 0.618),
                "BC_AB": (1.13, 1.414),
                "CD_BC": (0.618, 0.786),
                "AD_XA": (0.786, 0.786)
            },
            "shark_bull": {
                "AB_XA": (0.382, 0.618),
                "BC_AB": (1.13, 1.618),
                "CD_BC": (1.618, 2.24),
                "AD_XA": (0.886, 1.13)
            },
            "shark_bear": {
                "AB_XA": (0.382, 0.618),
                "BC_AB": (1.13, 1.618),
                "CD_BC": (1.618, 2.24),
                "AD_XA": (0.886, 1.13)
            }
        }

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

        # Find XABCD patterns
        for end in range(self.min_span, n):
            for span in range(self.min_span, min(self.max_span, end) + 1, 5):
                start = end - span

                xabcd_points = self._find_xabcd_points(hi, lo, start, end)

                if xabcd_points:
                    pattern_match = self._validate_harmonic_pattern(xabcd_points)
                    if pattern_match and pattern_match == self.key:
                        confidence = self._calculate_harmonic_confidence(xabcd_points)
                        if confidence > 0.6:
                            direction = "bull" if "_bull" in self.key else "bear"
                            magnitude = abs(xabcd_points[4][1] - xabcd_points[0][1])

                            evs.append(PatternEvent(
                                self.key, "chart", direction,
                                ts[xabcd_points[0][0]], ts[xabcd_points[4][0]],
                                "confirmed", confidence, magnitude, 5, span, None, 30,
                                {"xabcd_points": xabcd_points}
                            ))

        return evs

    def _find_xabcd_points(self, hi, lo, start, end) -> Optional[List[Tuple[int, float]]]:
        """Find XABCD pivot points"""
        # Simple swing detection
        swing_highs = []
        swing_lows = []

        lookback = 3
        for i in range(start + lookback, end - lookback):
            # Swing high
            if all(hi[i] >= hi[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_highs.append((i, hi[i]))

            # Swing low
            if all(lo[i] <= lo[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_lows.append((i, lo[i]))

        # Need at least 5 swing points for XABCD
        all_swings = swing_highs + swing_lows
        all_swings.sort()

        if len(all_swings) < 5:
            return None

        # Try to form XABCD pattern with alternating highs and lows
        for i in range(len(all_swings) - 4):
            points = all_swings[i:i + 5]

            # Check if pattern alternates between highs and lows
            is_alternating = True
            for j in range(len(points) - 1):
                curr_is_high = points[j] in swing_highs
                next_is_high = points[j + 1] in swing_highs
                if curr_is_high == next_is_high:
                    is_alternating = False
                    break

            if is_alternating:
                return points

        return None

    def _validate_harmonic_pattern(self, xabcd_points) -> Optional[str]:
        """Validate if points form a specific harmonic pattern"""
        if not xabcd_points or len(xabcd_points) != 5:
            return None

        X, A, B, C, D = xabcd_points

        # Calculate moves
        XA = abs(A[1] - X[1])
        AB = abs(B[1] - A[1])
        BC = abs(C[1] - B[1])
        CD = abs(D[1] - C[1])
        AD = abs(D[1] - A[1])

        if XA == 0 or AB == 0 or BC == 0:
            return None

        # Calculate ratios
        AB_XA = AB / XA
        BC_AB = BC / AB
        CD_BC = CD / BC if BC > 0 else 0
        AD_XA = AD / XA

        # Determine if bullish or bearish
        is_bullish = D[1] < A[1]  # D is below A for bullish patterns

        # Check against pattern definitions
        for pattern_name, ratios in self.pattern_ratios.items():
            if is_bullish and "_bull" not in pattern_name:
                continue
            if not is_bullish and "_bear" not in pattern_name:
                continue

            if self._check_ratio_match(AB_XA, ratios["AB_XA"]) and \
               self._check_ratio_match(BC_AB, ratios["BC_AB"]) and \
               self._check_ratio_match(CD_BC, ratios["CD_BC"]) and \
               self._check_ratio_match(AD_XA, ratios["AD_XA"]):
                return pattern_name

        return None

    def _check_ratio_match(self, actual_ratio, target_range):
        """Check if actual ratio matches target range within tolerance"""
        min_ratio, max_ratio = target_range
        tolerance = self.fibonacci_tolerance

        return (min_ratio * (1 - tolerance) <= actual_ratio <= max_ratio * (1 + tolerance))

    def _calculate_harmonic_confidence(self, xabcd_points):
        """Calculate confidence based on Fibonacci precision"""
        X, A, B, C, D = xabcd_points

        XA = abs(A[1] - X[1])
        AB = abs(B[1] - A[1])
        BC = abs(C[1] - B[1])
        CD = abs(D[1] - C[1])
        AD = abs(D[1] - A[1])

        if XA == 0 or AB == 0 or BC == 0:
            return 0.0

        AB_XA = AB / XA
        BC_AB = BC / AB
        CD_BC = CD / BC if BC > 0 else 0
        AD_XA = AD / XA

        confidence = 0.5
        ratios = self.pattern_ratios[self.key]

        # Add confidence based on ratio precision
        for ratio_name, target_range in ratios.items():
            if ratio_name == "AB_XA":
                actual = AB_XA
            elif ratio_name == "BC_AB":
                actual = BC_AB
            elif ratio_name == "CD_BC":
                actual = CD_BC
            elif ratio_name == "AD_XA":
                actual = AD_XA
            else:
                continue

            target_mid = (target_range[0] + target_range[1]) / 2
            precision = 1 - abs(actual - target_mid) / target_mid
            confidence += precision * 0.1

        # Bonus for precise key ratios
        if self.key.startswith("gartley") and abs(AD_XA - 0.786) < 0.02:
            confidence += 0.1
        elif self.key.startswith("bat") and abs(AD_XA - 0.886) < 0.02:
            confidence += 0.1
        elif self.key.startswith("crab") and abs(AD_XA - 1.618) < 0.05:
            confidence += 0.1

        return min(confidence, 1.0)


class ABCDPatternDetector(DetectorBase):
    """Simple ABCD pattern detector"""

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

                abcd_points = self._find_abcd_points(hi, lo, start, end)

                if abcd_points and self._validate_abcd_pattern(abcd_points):
                    confidence = self._calculate_abcd_confidence(abcd_points)
                    if confidence > 0.5:
                        direction = "bull" if abcd_points[3][1] > abcd_points[0][1] else "bear"
                        magnitude = abs(abcd_points[3][1] - abcd_points[0][1])

                        evs.append(PatternEvent(
                            self.key, "chart", direction,
                            ts[abcd_points[0][0]], ts[abcd_points[3][0]],
                            "confirmed", confidence, magnitude, 4, span, None, 20,
                            {"abcd_points": abcd_points}
                        ))

        return evs

    def _find_abcd_points(self, hi, lo, start, end):
        """Find ABCD points"""
        swing_highs = []
        swing_lows = []

        lookback = 2
        for i in range(start + lookback, end - lookback):
            if all(hi[i] >= hi[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_highs.append((i, hi[i]))
            if all(lo[i] <= lo[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                swing_lows.append((i, lo[i]))

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
                    return points

        return None

    def _validate_abcd_pattern(self, abcd_points):
        """Validate ABCD pattern with time and price symmetry"""
        A, B, C, D = abcd_points

        # Calculate moves
        AB = abs(B[1] - A[1])
        BC = abs(C[1] - B[1])
        CD = abs(D[1] - C[1])

        # Calculate times
        time_AB = B[0] - A[0]
        time_BC = C[0] - B[0]
        time_CD = D[0] - C[0]

        if AB == 0 or BC == 0 or time_AB == 0 or time_BC == 0:
            return False

        # Price relationships (CD should be 1.0-1.618 of AB)
        cd_ab_ratio = CD / AB
        price_symmetry = 0.618 <= cd_ab_ratio <= 1.618

        # Time relationships (CD time should be similar to AB time)
        cd_ab_time_ratio = time_CD / time_AB if time_AB > 0 else 0
        time_symmetry = 0.618 <= cd_ab_time_ratio <= 1.618

        return price_symmetry and time_symmetry

    def _calculate_abcd_confidence(self, abcd_points):
        """Calculate ABCD confidence based on symmetry"""
        A, B, C, D = abcd_points

        AB = abs(B[1] - A[1])
        CD = abs(D[1] - C[1])

        time_AB = B[0] - A[0]
        time_CD = D[0] - C[0]

        confidence = 0.4

        # Price symmetry
        if AB > 0:
            cd_ab_ratio = CD / AB
            if abs(cd_ab_ratio - 1.0) < 0.1:  # Nearly equal
                confidence += 0.2
            elif abs(cd_ab_ratio - 1.272) < 0.1:  # 1.272 Fibonacci
                confidence += 0.15
            elif abs(cd_ab_ratio - 1.618) < 0.1:  # Golden ratio
                confidence += 0.15

        # Time symmetry
        if time_AB > 0:
            time_ratio = time_CD / time_AB
            if abs(time_ratio - 1.0) < 0.2:
                confidence += 0.15

        return min(confidence, 1.0)


def make_harmonic_pattern_detectors() -> List[DetectorBase]:
    """Create all harmonic pattern detectors"""
    harmonic_patterns = [
        "gartley_bull", "gartley_bear",
        "butterfly_bull", "butterfly_bear",
        "bat_bull", "bat_bear",
        "crab_bull", "crab_bear",
        "cypher_bull", "cypher_bear",
        "shark_bull", "shark_bear"
    ]

    detectors = []
    for pattern in harmonic_patterns:
        detectors.append(HarmonicPatternDetector(pattern))

    # Add simple ABCD patterns
    detectors.append(ABCDPatternDetector("abcd_bull"))
    detectors.append(ABCDPatternDetector("abcd_bear"))

    return detectors