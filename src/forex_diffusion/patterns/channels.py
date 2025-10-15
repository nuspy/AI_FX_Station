from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import time_array
from .primitives import fit_line_indices, atr

class ChannelDetector(DetectorBase):
    key="channel_generic"
    kind="chart"
    def __init__(self, key:str, rising:bool, min_span:int=30, max_span:int=180, max_events:int=50) -> None:
        self.key=key; self.rising=rising; self.min_span=min_span; self.max_span=max_span; self.max_events=max_events

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        hi = df["high"].astype(float).to_numpy()
        lo = df["low"].astype(float).to_numpy()
        ts = time_array(df)
        a = atr(df, 14).to_numpy()
        events: List[PatternEvent]=[]
        n=len(df)

        # PERFORMANCE FIX: Use much larger step sizes to reduce O(n^3) complexity
        end_step = max(20, self.min_span // 3)  # Skip many end points
        span_step = max(25, (self.max_span - self.min_span) // 5)  # Only check few span sizes

        for end in range(self.min_span, n, end_step):
            # Limit number of spans to check per end point
            spans_to_check = 3  # Only check 3 different span sizes
            for i in range(spans_to_check):
                span = self.min_span + i * span_step
                if span > min(self.max_span, end):
                    break

                start = end - span
                s_hi, b_hi = fit_line_indices(hi, start, end)
                s_lo, b_lo = fit_line_indices(lo, start, end)

                # slopes roughly equal (parallel)
                if abs(s_hi - s_lo) < abs(s_hi)*0.25 + 1e-6:
                    slope = (s_hi + s_lo)/2.0
                    if (self.rising and slope>0) or ((not self.rising) and slope<0):
                        # touches check
                        idx = np.arange(start, end+1)
                        upper = s_hi*idx + b_hi
                        lower = s_lo*idx + b_lo
                        tol = a[end]*0.5
                        touch = int(np.sum(np.abs(hi[start:end+1]-upper)<=tol) + np.sum(np.abs(lo[start:end+1]-lower)<=tol))
                        if touch >= 4:
                            direction = "bull" if self.rising else "bear"
                            events.append(PatternEvent(self.key,"chart",direction, ts[start], ts[end], "confirmed", 0.5, float(a[end]), touch, span, None, span//2, {"upper_line":(start,end,float(s_hi),float(b_hi)),"lower_line":(start,end,float(s_lo),float(b_lo))}))
                            if len(events)>=self.max_events: return events
        return events

def make_channel_detectors() -> List[ChannelDetector]:
    return [
        ChannelDetector("rising_channel", True),
        ChannelDetector("falling_channel", False),
    ]
