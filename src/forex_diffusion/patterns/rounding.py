from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

from .engine import PatternEvent, DetectorBase
from .primitives import atr, time_array

class RoundingDetector(DetectorBase):
    kind = "chart"
    def __init__(self, key: str, top: bool, window: int = 200, max_events: int = 20) -> None:
        self.key = key; self.top = top; self.window = window; self.max_events = max_events

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        ts = time_array(df); c = df["close"].astype(float).to_numpy(); a = atr(df, 14).to_numpy()
        n = len(df); out: List[PatternEvent] = []
        for end in range(self.window, n):
            start = end - self.window; seg = c[start:end+1]
            s1 = np.gradient(seg); s2 = np.gradient(s1); curv = float(np.mean(s2))
            if self.top and curv < 0:
                out.append(PatternEvent(self.key, "chart", "bear", pd.to_datetime(ts[start]), pd.to_datetime(ts[end]), float(seg[-1]), 0.52, float(a[end]), 3, self.window, None, 40.0, {}))
            elif (not self.top) and curv > 0:
                out.append(PatternEvent(self.key, "chart", "bull", pd.to_datetime(ts[start]), pd.to_datetime(ts[end]), float(seg[-1]), 0.52, float(a[end]), 3, self.window, None, 40.0, {}))
            if len(out) >= self.max_events: break
        return out

def make_rounding_detectors() -> List[RoundingDetector]:
    return [RoundingDetector("rounding_top", True), RoundingDetector("rounding_bottom", False)]
