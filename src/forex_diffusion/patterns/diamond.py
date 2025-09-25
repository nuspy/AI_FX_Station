from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import atr

class DiamondDetector(DetectorBase):
    kind="chart"
    def __init__(self, key:str, top:bool=True, window:int=120, max_events:int=30) -> None:
        self.key=key; self.top=top; self.window=window; self.max_events=max_events
    def detect(self, df: pd.DataFrame):
        # Heuristica: fase di allargamento (volatilità in aumento) seguita da contrazione (in calo)
        import numpy as np
        ts = pd.to_datetime(df["time"] if "time" in df.columns else df.index)
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        a = atr(df,14).to_numpy()
        events=[]
        n=len(df)
        rng = h - l
        vol = pd.Series(rng).rolling(10,min_periods=1).std().to_numpy()
        for end in range(self.window, n):
            start = end - self.window
            seg = vol[start:end+1]
            if len(seg)<30: continue
            if seg[:len(seg)//2].mean() > seg[len(seg)//2:].mean()*1.1:  # da ampio a stretto (contrazione)
                direction = "bear" if self.top else "bull"
                events.append(PatternEvent(self.key,"chart",direction, ts.iloc[start], ts.iloc[end], "confirmed", 0.51, float(a[end]), 0, self.window, None, self.window//2, {}))
                if len(events)>=self.max_events: break
        return events

def make_diamond_detectors():
    return [DiamondDetector("diamond_top", True), DiamondDetector("diamond_bottom", False)]
