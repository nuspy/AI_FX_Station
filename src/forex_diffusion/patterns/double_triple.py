from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import atr

class _PeaksBase(DetectorBase):
    kind="chart"
    def __init__(self, key:str, peaks:int=2, top:bool=True, window:int=120, tol:float=0.005, max_events:int=40) -> None:
        self.key=key; self.peaks=peaks; self.top=top; self.window=window; self.tol=tol; self.max_events=max_events

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        ts = pd.to_datetime(df["time"] if "time" in df.columns else df.index).to_numpy()
        a = atr(df, 14).to_numpy()
        events: List[PatternEvent] = []
        n=len(df)
        series = h if self.top else -l
        for end in range(self.window, n):
            start = end - self.window
            seg = series[start:end+1]
            locs = [i for i in range(1,len(seg)-1) if seg[i]>seg[i-1] and seg[i]>seg[i+1]]
            if len(locs) < self.peaks: 
                continue
            # scan groups of 'peaks' with approx equal heights
            for i in range(len(locs)-self.peaks+1):
                idxs = locs[i:i+self.peaks]
                heights = seg[idxs]
                if (heights.max()-heights.min()) <= heights.mean()*self.tol*10:
                    direction = "bear" if self.top else "bull"
                    events.append(PatternEvent(self.key,"chart",direction, ts[start+idxs[0]], ts[end], "confirmed", 0.52, float(a[end]), self.peaks, self.window, None, self.window//3, {"peaks":[start+j for j in idxs]}))
                    if len(events)>=self.max_events: return events
        return events

def make_double_triple_detectors():
    return [
        _PeaksBase("double_top", peaks=2, top=True),
        _PeaksBase("double_bottom", peaks=2, top=False),
        _PeaksBase("triple_top", peaks=3, top=True),
        _PeaksBase("triple_bottom", peaks=3, top=False),
    ]
