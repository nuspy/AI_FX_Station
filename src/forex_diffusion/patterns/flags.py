from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import atr

class FlagDetector(DetectorBase):
    key = "flag_generic"
    kind = "chart"
    def __init__(self, key: str, direction: str, impulse_mult: float=2.0, window:int=40, max_events:int=50) -> None:
        self.key=key; self.dir=direction; self.impulse_mult=impulse_mult; self.window=window; self.max_events=max_events

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        c = df["close"].astype(float).to_numpy()
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        ts = pd.to_datetime(df["time"] if "time" in df.columns else df.index).to_numpy()
        a = atr(df, 14).to_numpy()
        events: List[PatternEvent] = []
        n = len(df)
        for i in range(30, n):
            w = min(self.window, i)
            # impulse: cumulative move vs ATR
            move = c[i] - c[i-w]
            amean = np.mean(a[i-w:i])
            if amean==0: 
                continue
            if self.dir=="bull" and move < self.impulse_mult*amean: 
                continue
            if self.dir=="bear" and -move < self.impulse_mult*amean: 
                continue

            # consolidation: small channel over last w/2 bars
            j0 = i - w//2
            sub_h = h[j0:i]; sub_l = l[j0:i]
            height = sub_h.max() - sub_l.min()
            if height <= 0.8*amean:  # tight consolidation
                events.append(PatternEvent(self.key,"chart",self.dir, ts[j0], ts[i], "confirmed", 0.52, float(a[i]), 3, w//2, None, w//2, {"box":(j0,i,float(sub_l.min()),float(sub_h.max()))}))
                if len(events)>=self.max_events: break
        return events

def make_flag_detectors() -> List[FlagDetector]:
    return [
        FlagDetector("bull_flag","bull"),
        FlagDetector("bear_flag","bear"),
        FlagDetector("bull_pennant","bull"),
        FlagDetector("bear_pennant","bear"),
    ]
