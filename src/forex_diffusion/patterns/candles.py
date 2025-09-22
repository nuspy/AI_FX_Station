from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase

def _body(o,c):
    return abs(float(c)-float(o))

def _tr(h,l,pc):
    return max(float(h)-float(l), abs(float(h)-float(pc)), abs(float(l)-float(pc)))

class SimpleCandleDetector(DetectorBase):
    key = "candle_generic"
    kind = "candle"

    def __init__(self, key: str) -> None:
        self.key = key

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        evs: List[PatternEvent] = []
        if df is None or len(df) < 3:
            return evs
        ts = pd.to_datetime(df["ts_utc"], unit="ms", utc=True).tz_convert(None)
        o = df["open"].astype(float).to_numpy()
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        c = df["close"].astype(float).to_numpy()
        pc = np.roll(c,1); pc[0]=c[0]

        i = len(df)-1  # last bar (forming/confirmed)
        body = _body(o[i], c[i])
        tr = _tr(h[i], l[i], pc[i])
        upper = float(h[i]-max(o[i],c[i]))
        lower = float(min(o[i],c[i])-l[i])

        if self.key in ("hammer","shooting_star"):
            # scale-invariant rules
            body_ok = body >= 0.25*tr
            if self.key=="hammer":
                cond = body_ok and lower >= 2.0*body and upper <= 0.3*body
                direction = "bull"
            else:
                cond = body_ok and upper >= 2.0*body and lower <= 0.3*body
                direction = "bear"
            if cond:
                evs.append(PatternEvent(
                    pattern_key=self.key, kind="candle", direction=direction,
                    start_ts=ts.iloc[i], confirm_ts=ts.iloc[i], state="forming",
                    score=0.5, scale_atr=tr, touches=1, bars_span=1,
                    target_price=None, horizon_bars=10,
                    overlay={"marker": i}
                ))
        elif self.key in ("bullish_engulfing","bearish_engulfing"):
            i2 = i-1
            if i2>=0:
                cond_bull = (c[i]>o[i]) and (c[i2]<o[i2]) and (c[i]>=o[i2]) and (o[i]<=c[i2])
                cond_bear = (c[i]<o[i]) and (c[i2]>o[i2]) and (c[i]<=o[i2]) and (o[i]>=c[i2])
                if (self.key=="bullish_engulfing" and cond_bull) or (self.key=="bearish_engulfing" and cond_bear):
                    direction = "bull" if self.key.startswith("bull") else "bear"
                    evs.append(PatternEvent(self.key,"candle",direction,ts.iloc[i2],ts.iloc[i],"forming",0.5,tr,2,2,None,10,{"marker": i}))
        elif self.key in ("harami_bull","harami_bear"):
            i2 = i-1
            if i2>=0:
                # inside body
                big_first = _body(o[i2], c[i2]) >= 0.6*_tr(h[i2], l[i2], pc[i2])
                inside = (min(o[i],c[i]) >= min(o[i2],c[i2])) and (max(o[i],c[i]) <= max(o[i2],c[i2]))
                if big_first and inside:
                    direction = "bull" if self.key.endswith("bull") else "bear"
                    evs.append(PatternEvent(self.key,"candle",direction,ts.iloc[i2],ts.iloc[i],"forming",0.45,tr,2,2,None,10,{"marker": i}))
        return evs

def make_candle_detectors() -> List[SimpleCandleDetector]:
    keys = ["hammer","shooting_star","bullish_engulfing","bearish_engulfing","harami_bull","harami_bear"]
    return [SimpleCandleDetector(k) for k in keys]
