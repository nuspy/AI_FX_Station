from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase

def _body(o,c): return abs(float(c)-float(o))
def _tr(h,l,pc): return max(float(h)-float(l), abs(float(h)-float(pc)), abs(float(l)-float(pc)))

class SimpleCandleDetector(DetectorBase):
    """Sottoinsieme di pattern candlestick: hammer, shooting_star, engulfing, harami."""
    key = "candle_generic"
    kind = "candle"
    def __init__(self, key: str) -> None:
        self.key = key
    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        evs: List[PatternEvent] = []
        if df is None or len(df) < 3: return evs

        ts = pd.to_datetime(df["ts_utc"], unit="ms", utc=True)
        try:
            ts = ts.dt.tz_convert(None)
        except AttributeError:
            ts = ts.tz_convert(None)

        o = df["open"].astype(float).to_numpy()
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        c = df["close"].astype(float).to_numpy()
        pc = np.roll(c,1); pc[0]=c[0]
        for i in range(1, len(df)):
            body = _body(o[i], c[i]); tr = _tr(h[i], l[i], pc[i])
            upper = float(h[i]-max(o[i],c[i])); lower = float(min(o[i],c[i])-l[i])
            # Hammer / Shooting star
            if self.key in ("hammer","shooting_star"):
                body_ok = body >= 0.25*tr
                if self.key=="hammer":
                    cond = body_ok and lower >= 2.0*body and upper <= 0.3*body; direction = "bull"
                else:
                    cond = body_ok and upper >= 2.0*body and lower <= 0.3*body; direction = "bear"
                if cond:
                    evs.append(PatternEvent(self.key,"candle",direction,ts[i],ts[i],"confirmed",0.5,tr,1,1,None,10,{"marker": i}))
            # Engulfing
            elif self.key in ("bullish_engulfing","bearish_engulfing"):
                i2 = i-1
                cond_bull = (c[i]>o[i]) and (c[i2]<o[i2]) and (c[i]>=o[i2]) and (o[i]<=c[i2])
                cond_bear = (c[i]<o[i]) and (c[i2]>o[i2]) and (c[i]<=o[i2]) and (o[i]>=c[i2])
                if (self.key=="bullish_engulfing" and cond_bull) or (self.key=="bearish_engulfing" and cond_bear):
                    direction = "bull" if self.key.startswith("bull") else "bear"
                    evs.append(PatternEvent(self.key,"candle",direction,ts[i2],ts[i],"confirmed",0.5,tr,2,2,None,10,{"marker": i}))
            # Harami
            elif self.key in ("harami_bull","harami_bear"):
                i2 = i-1
                big_first = _body(o[i2], c[i2]) >= 0.6*_tr(h[i2], l[i2], pc[i2])
                inside = (min(o[i],c[i]) >= min(o[i2],c[i2])) and (max(o[i],c[i]) <= max(o[i2],c[i2]))
                if big_first and inside:
                    direction = "bull" if self.key.endswith("bull") else "bear"
                    evs.append(PatternEvent(self.key,"candle",direction,ts[i2],ts[i],"confirmed",0.45,tr,2,2,None,10,{"marker": i}))
        return evs

def make_candle_detectors() -> List[SimpleCandleDetector]:
    keys = ["hammer","shooting_star","bullish_engulfing","bearish_engulfing","harami_bull","harami_bear"]+EXTRA_CANDLE_KEYS
    return [SimpleCandleDetector(k) for k in keys]


# --- Extra candlestick keys (batch 2) ---
EXTRA_CANDLE_KEYS = ['three_white_soldiers', 'three_black_crows', 'dark_cloud_cover', 'piercing_line', 'dragonfly_doji', 'gravestone_doji', 'tweezer_top', 'tweezer_bottom', 'rising_three_methods', 'falling_three_methods']


# --- Extended Candlestick Patterns (skeleton heuristics) ---
# Nota: euristiche compatte per prima iterazione; parametri fino a esposizione via patterns.yaml

def _bool(v): return bool(v)

class BeltHoldDetector(SimpleCandleDetector):
    def __init__(self, bull: bool):
        super().__init__("belt_hold_bull" if bull else "belt_hold_bear")
    # uses inherited detect with key suffix

class KickerDetector(SimpleCandleDetector):
    def __init__(self, bull: bool):
        super().__init__("kicker_bull" if bull else "kicker_bear")


class TasukiDetector(SimpleCandleDetector):
    def __init__(self, bull: bool):
        super().__init__("tasuki_bull" if bull else "tasuki_bear")


EXTRA_CANDLE_KEYS.extend([
    'harami_cross_bull','harami_cross_bear',
    'belt_hold_bull','belt_hold_bear',
    'mat_hold_bull','mat_hold_bear',
    'kicker_bull','kicker_bear',
    'tasuki_bull','tasuki_bear'
])
