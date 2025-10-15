from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import time_array
from .primitives import atr

class _PeaksBase(DetectorBase):
    kind="chart"
    def __init__(self, key:str, peaks:int=2, top:bool=True, window:int=120, tol:float=0.005, max_events:int=40) -> None:
        self.key=key; self.peaks=peaks; self.top=top; self.window=window; self.tol=tol; self.max_events=max_events

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        ts = time_array(df)
        a = atr(df, 14).to_numpy()
        events: List[PatternEvent] = []
        n=len(df)
        series = h if self.top else -l

        # PERFORMANCE FIX: Pre-compute all peaks once for entire series
        all_peaks = self._find_all_peaks(series)
        if len(all_peaks) < self.peaks:
            return events

        # Use much larger step size to reduce iterations
        step_size = max(15, self.window // 8)

        for end in range(self.window, n, step_size):
            start = end - self.window

            # Filter peaks to current window
            window_peaks = [i for i in all_peaks if start <= i <= end]

            if len(window_peaks) < self.peaks:
                continue

            # scan groups of 'peaks' with approx equal heights
            for i in range(len(window_peaks) - self.peaks + 1):
                idxs = window_peaks[i:i+self.peaks]
                heights = np.array([series[j] for j in idxs])
                if (heights.max()-heights.min()) <= heights.mean()*self.tol*10:
                    direction = "bear" if self.top else "bull"
                    events.append(PatternEvent(self.key,"chart",direction, ts[idxs[0]], ts[end], "confirmed", 0.52, float(a[end]), self.peaks, self.window, None, self.window//3, {"peaks":idxs}))
                    if len(events)>=self.max_events: return events
        return events

    def _find_all_peaks(self, series):
        """Pre-compute all peaks in the series once"""
        peaks = []
        for i in range(1, len(series) - 1):
            if series[i] > series[i-1] and series[i] > series[i+1]:
                peaks.append(i)
        return peaks

def make_double_triple_detectors():
    return [
        _PeaksBase("double_top", peaks=2, top=True),
        _PeaksBase("double_bottom", peaks=2, top=False),
        _PeaksBase("triple_top", peaks=3, top=True),
        _PeaksBase("triple_bottom", peaks=3, top=False),
    ]
