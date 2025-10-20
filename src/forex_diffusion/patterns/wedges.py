from __future__ import annotations
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import time_array
from .primitives import fit_line_indices, atr

class WedgeDetector(DetectorBase):
    kind="chart"
    def __init__(self, key:str, ascending:bool, min_span:int=30, max_span:int=200, min_touches:int=4, max_events:int=40) -> None:
        self.key=key; self.ascending=ascending; self.min_span=min_span; self.max_span=max_span; self.min_touches=min_touches; self.max_events=max_events

    def detect(self, df: pd.DataFrame):
        hi = df["high"].astype(float).to_numpy()
        lo = df["low"].astype(float).to_numpy()
        ts = time_array(df)
        a = atr(df,14).to_numpy()
        events=[]
        n=len(df)
        for end in range(self.min_span, n):
            for span in range(self.min_span, min(self.max_span, end)+1, 10):
                start = end - span
                s_hi, b_hi = fit_line_indices(hi, start, end)
                s_lo, b_lo = fit_line_indices(lo, start, end)
                # converging and both sloping up or both sloping down
                if self.ascending and s_hi>0 and s_lo>0 and abs(s_hi) > abs(s_lo):
                    pass
                elif (not self.ascending) and s_hi<0 and s_lo<0 and abs(s_lo) > abs(s_hi):
                    pass
                else:
                    continue
                # touches
                idx = np.arange(start,end+1)
                upper = s_hi*idx + b_hi; lower = s_lo*idx + b_lo
                tol = a[end]*0.5
                touch = int(np.sum(np.abs(hi[start:end+1]-upper)<=tol) + np.sum(np.abs(lo[start:end+1]-lower)<=tol))
                if touch >= self.min_touches:
                    direction = "bear" if self.ascending else "bull"
                    events.append(PatternEvent(self.key,"chart",direction, ts[start], ts[end], "confirmed", 0.53, float(a[end]), touch, span, None, span//2, {"upper_line":(start,end,float(s_hi),float(b_hi)),"lower_line":(start,end,float(s_lo),float(b_lo))}))
                    if len(events)>=self.max_events: return events
        return events

def make_wedge_detectors():
    return [
        WedgeDetector("wedge_ascending", True),
        WedgeDetector("wedge_descending", False),
    ]
