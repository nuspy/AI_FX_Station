from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import atr, zigzag_pivots, fit_line_indices

class BroadeningDetector(DetectorBase):
    """
    Generic 'broadening' detector: envelope with diverging upper/lower trendlines.
    Subclasses specify right-angle or wedge variants via parameters.
    """
    key = "broadening_generic"
    kind = "chart"

    def __init__(self,
                 key: str,
                 right_angle: Optional[str] = None,  # "upper" or "lower" or None
                 wedge: Optional[str] = None,        # "ascending"/"descending" or None
                 min_touches: int = 4,
                 divergence_min_ratio: float = 0.02,
                 atr_period: int = 14) -> None:
        self.key = key
        self.ra = right_angle
        self.wedge = wedge
        self.min_touches = min_touches
        self.divergence_min_ratio = divergence_min_ratio
        self.atr_period = atr_period

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        events: List[PatternEvent] = []
        if df is None or df.empty or len(df) < 40:
            return events
        highs = df["high"].astype(float).to_numpy()
        lows = df["low"].astype(float).to_numpy()
        close = df["close"].astype(float).to_numpy()
        ts = pd.to_datetime(df["ts_utc"], unit="ms", utc=True).tz_convert(None)

        # pivots and envelope over a rolling window
        win = min(200, len(df))
        start = max(0, len(df) - win)
        atr_vals = atr(df.iloc[start:], self.atr_period).to_numpy()
        idx0 = start
        # Fit lines to highs and lows over the window
        slope_hi, icp_hi = fit_line_indices(highs, start, len(df)-1)
        slope_lo, icp_lo = fit_line_indices(lows, start, len(df)-1)

        # Divergence test: opposite signs
        diverging = (slope_hi > 0 and slope_lo < 0) or (slope_hi < 0 and slope_lo > 0)
        if not diverging:
            return events

        # Increasing width test
        x0, x1 = start, len(df)-1
        width_0 = (slope_hi*x0 + icp_hi) - (slope_lo*x0 + icp_lo)
        width_1 = (slope_hi*x1 + icp_hi) - (slope_lo*x1 + icp_lo)
        if width_1 <= width_0 * (1.0 + self.divergence_min_ratio):
            return events

        # Right-angle constraint if requested
        if self.ra == "upper":
            if abs(slope_hi) > abs(slope_lo) or abs(slope_hi) > 1e-6:
                return events
        if self.ra == "lower":
            if abs(slope_lo) > abs(slope_hi) or abs(slope_lo) > 1e-6:
                return events

        # Wedge hint: both up or both down slopes but widening—here skip, broadening wedge handled elsewhere
        # Touches (approximate): count crossings near fitted lines
        touches = 0
        tol = np.nanmedian(atr_vals[-50:]) if len(atr_vals) else 0.0
        for i in range(start, len(df)):
            up = slope_hi*i + icp_hi
            lo = slope_lo*i + icp_lo
            if abs(highs[i]-up) <= 1.5*tol or abs(lows[i]-lo) <= 1.5*tol:
                touches += 1
        if touches < self.min_touches:
            return events

        # Determine breakout side at the last bar: close beyond envelope?
        up_last = slope_hi*(len(df)-1) + icp_hi
        lo_last = slope_lo*(len(df)-1) + icp_lo
        direction = "bull" if close[-1] > up_last else ("bear" if close[-1] < lo_last else "neutral")
        if direction == "neutral":
            # forming
            state = "forming"
            target = None
        else:
            state = "confirmed"
            height = float(width_1)
            mult = 1.0
            target = float(close[-1] + (mult*height if direction=="bull" else -mult*height))

        events.append(PatternEvent(
            pattern_key=self.key,
            kind="chart",
            direction=direction,
            start_ts=ts.iloc[start],
            confirm_ts=ts.iloc[-1],
            state=state, score=0.6, scale_atr=float(width_1/max(tol,1e-9)),
            touches=touches, bars_span=len(df)-start,
            target_price=target, horizon_bars=min(60, len(df)-start),
            overlay={
                "upper_line": (start, len(df)-1, float(slope_hi), float(icp_hi)),
                "lower_line": (start, len(df)-1, float(slope_lo), float(icp_lo)),
            }
        ))
        return events

def make_broadening_detectors() -> List[BroadeningDetector]:
    return [
        BroadeningDetector(key="broadening_bottom"),
        BroadeningDetector(key="broadening_top"),
        BroadeningDetector(key="right_angle_broadening_ascending", right_angle="lower"),
        BroadeningDetector(key="right_angle_broadening_descending", right_angle="upper"),
        BroadeningDetector(key="broadening_wedge_ascending", wedge="ascending"),
        BroadeningDetector(key="broadening_wedge_descending", wedge="descending"),
    ]
