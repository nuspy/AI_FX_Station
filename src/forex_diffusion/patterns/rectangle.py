from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import time_array
from .primitives import atr

class RectangleDetector(DetectorBase):
    kind="chart"
    def __init__(self, key:str="rectangle_range", window:int=80, tightness:float=0.8, max_events:int=60) -> None:
        self.key=key; self.window=window; self.tightness=tightness; self.max_events=max_events
    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        ts = time_array(df)
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        a = atr(df,14).to_numpy()
        events=[]
        n=len(df)
        for end in range(self.window, n):
            start = end - self.window
            hi = h[start:end+1]; lo = l[start:end+1]
            height = hi.max() - lo.min()
            # rettangolo "tight": range relativamente stretto vs ATR cumulato
            atr_mean = float(np.mean(a[start:end+1]))
            if height <= self.tightness*atr_mean:
                events.append(PatternEvent(self.key,"chart","neutral", ts[start], ts[end], "confirmed", 0.5, float(a[end]), 4, self.window, None, self.window//2, {"box":(start,end,float(lo.min()),float(hi.max()))}))
                if len(events)>=self.max_events: break
        return events

def make_rectangle_detectors():
    return [RectangleDetector()]
