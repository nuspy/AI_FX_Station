from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import fit_line_indices, atr

class TriangleDetector(DetectorBase):
    key = "triangle_generic"
    kind = "chart"

    def __init__(self, key: str, mode: str, min_span:int=20, max_span:int=120, min_touches:int=4, atr_period:int=14, max_events:int=50) -> None:
        self.key=key; self.mode=mode
        self.min_span=min_span; self.max_span=max_span; self.min_touches=min_touches
        self.atr_period=atr_period; self.max_events=max_events

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        hi = df["high"].astype(float).to_numpy()
        lo = df["low"].astype(float).to_numpy()
        ts = pd.to_datetime(df["time"] if "time" in df.columns else df.index)
        a = atr(df, self.atr_period).to_numpy()
        events: List[PatternEvent] = []
        n = len(df)
        for end in range(self.min_span, n):
            for span in range(self.min_span, min(self.max_span, end)+1, max(5, self.min_span//2)):
                start = end - span
                # fit lines on highs and lows
                s_hi, b_hi = fit_line_indices(hi, start, end)
                s_lo, b_lo = fit_line_indices(lo, start, end)

                # require converging lines
                converge = ( (hi[end] - (s_hi*end+b_hi))**2 + (lo[end] - (s_lo*end+b_lo))**2 ) < 10*a[end]**2

                # classify
                up_flat   = abs(s_hi) < 1e-6 and s_lo > 0
                down_flat = abs(s_lo) < 1e-6 and s_hi < 0
                symm      = s_hi < 0 and s_lo > 0

                ok = False
                if self.mode=="ascending" and up_flat:
                    ok=True
                elif self.mode=="descending" and down_flat:
                    ok=True
                elif self.mode=="symmetrical" and symm:
                    ok=True

                if ok:
                    # touches heuristic: count points near each line
                    idx = np.arange(start, end+1)
                    upper = s_hi*idx + b_hi
                    lower = s_lo*idx + b_lo
                    tol = a[end]*0.5
                    touch_u = int(np.sum(np.abs(hi[start:end+1]-upper) <= tol))
                    touch_l = int(np.sum(np.abs(lo[start:end+1]-lower) <= tol))
                    if (touch_u + touch_l) >= self.min_touches and converge:
                        direction = "bull" if self.mode=="ascending" else ("bear" if self.mode=="descending" else "neutral")
                        events.append(PatternEvent(self.key,"chart",direction, ts.iloc[start], ts.iloc[end], "confirmed", 0.55, float(a[end]), touch_u+touch_l, span, None, span//2, {"upper_line":(start,end,float(s_hi),float(b_hi)),"lower_line":(start,end,float(s_lo),float(b_lo))}))
                        if len(events)>=self.max_events: return events
        return events

def make_triangle_detectors() -> List[TriangleDetector]:
    return [
        TriangleDetector("ascending_triangle","ascending"),
        TriangleDetector("descending_triangle","descending"),
        TriangleDetector("symmetrical_triangle","symmetrical"),
    ]
