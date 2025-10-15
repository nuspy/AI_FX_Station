from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import time_array, safe_tz_convert
from .primitives import atr, fit_line_indices

class BroadeningDetector(DetectorBase):
    """
    Rilevatore generico per famiglie di pattern "broadening".
    Usa una finestra scorrevole, fit lineare sui massimi/minimi e verifica divergenza delle trendline.
    """
    key = "broadening_generic"
    kind = "chart"
    def __init__(self, key: str, right_angle: Optional[str]=None, wedge: Optional[str]=None,
                 min_touches:int=4, divergence_min_ratio:float=0.02, atr_period:int=14,
                 window:int=400, stride:int=50, max_events:int=20) -> None:
        self.key=key; self.ra=right_angle; self.wedge=wedge
        self.min_touches=min_touches; self.divergence_min_ratio=divergence_min_ratio
        self.atr_period=atr_period; self.window=window; self.stride=stride; self.max_events=max_events

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        events: List[PatternEvent] = []
        if df is None or df.empty or len(df) < max(60, self.window//2): return events
        highs = df["high"].astype(float).to_numpy()
        lows = df["low"].astype(float).to_numpy()
        close = df["close"].astype(float).to_numpy()


        ts = time_array(df)
        ts = safe_tz_convert(ts, None)


        n = len(df)
        win = min(self.window, n)
        starts = list(range(0, n-win, self.stride)) + [max(0, n-win)]
        atr_vals = atr(df, self.atr_period).to_numpy()

        for start in starts[::-1]:
            end = start + win - 1
            slope_hi, icp_hi = fit_line_indices(highs, start, end)
            slope_lo, icp_lo = fit_line_indices(lows, start, end)
            diverging = (slope_hi>0 and slope_lo<0) or (slope_hi<0 and slope_lo>0)
            if not diverging: continue

            width_0 = (slope_hi*start + icp_hi) - (slope_lo*start + icp_lo)
            width_1 = (slope_hi*end + icp_hi) - (slope_lo*end + icp_lo)
            if width_1 <= width_0 * (1.0 + self.divergence_min_ratio): continue

            if self.ra == "upper" and (abs(slope_hi) > 1e-6 or abs(slope_hi) > abs(slope_lo)): continue
            if self.ra == "lower" and (abs(slope_lo) > 1e-6 or abs(slope_lo) > abs(slope_hi)): continue

            touches = 0; tol = float(np.nanmedian(atr_vals[max(0,start):end+1]))
            for i in range(start, end+1):
                up = slope_hi*i + icp_hi; lo = slope_lo*i + icp_lo
                if abs(highs[i]-up) <= 1.5*tol or abs(lows[i]-lo) <= 1.5*tol:
                    touches += 1
            if touches < self.min_touches: continue

            up_last = slope_hi*end + icp_hi; lo_last = slope_lo*end + icp_lo
            direction = "bull" if close[end] > up_last else ("bear" if close[end] < lo_last else "neutral")
            state = "confirmed" if direction != "neutral" else "forming"
            target = None
            if state == "confirmed":
                height = float((up_last - lo_last)); mult=1.0
                target = float(close[end] + (mult*height if direction=='bull' else -mult*height))

            events.append(PatternEvent(self.key,"chart",direction,ts[start],ts[end],state,0.6,
                                       float((up_last - lo_last)/max(tol,1e-9)),touches,win,target,min(120,win),
                                       {"upper_line":(start,end,float(slope_hi),float(icp_hi)),
                                        "lower_line":(start,end,float(slope_lo),float(icp_lo))}))
            if len(events) >= self.max_events: break
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
