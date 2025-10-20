
from __future__ import annotations
from typing import List, Any, Optional
import numpy as np
import pandas as pd

def _to_ms(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, (int, np.integer)):
            v = int(x)
            if v > 10**16:  # ns
                return v // 1_000_000
            if v > 10**11:  # ms
                return v
            if v > 10**8:   # s
                return v * 1000
            return v * 1000
        ts = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return int(ts.value // 1_000_000)
    except Exception:
        return None

def _nearest_close_at_ts(df: pd.DataFrame, ts_ms: int) -> Optional[float]:
    try:
        ts_arr = df["ts_utc"].to_numpy(dtype="int64")
        i = np.searchsorted(ts_arr, ts_ms, side="left")
        if i <= 0:
            j = 0
        elif i >= len(ts_arr):
            j = len(ts_arr) - 1
        else:
            j = i if (ts_ms - ts_arr[i - 1]) > (ts_arr[i] - ts_ms) else i - 1
        return float(df["close"].iloc[j])
    except Exception:
        return None

def _atr_default(df: pd.DataFrame, n: int = 14) -> Optional[float]:
    try:
        h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
        prev_c = np.r_[c[0], c[:-1]]
        tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
        atr = pd.Series(tr).rolling(n, min_periods=max(3, n//2)).mean().to_numpy()
        tail = atr[-100:] if atr.size >= 100 else atr
        val = np.nanmedian(tail)
        return float(val) if np.isfinite(val) else None
    except Exception:
        return None

def _amplitude_guess(df: pd.DataFrame, e: Any) -> Optional[float]:
    for a, b in (("upper", "lower"), ("high", "low")):
        up, lo = getattr(e, a, None), getattr(e, b, None)
        try:
            if up is not None and lo is not None:
                amp = abs(float(up) - float(lo))
                if np.isfinite(amp) and amp > 0:
                    return amp
        except Exception:
            pass
    for s_name, e_name in (("start_idx", "end_idx"), ("i_start", "i_end"), ("confirm_idx", "end_idx")):
        s = getattr(e, s_name, None)
        t = getattr(e, e_name, None)
        if isinstance(s, int) and isinstance(t, int) and 0 <= s < t < len(df):
            try:
                w_hi = float(df["high"].iloc[s:t + 1].max())
                w_lo = float(df["low"].iloc[s:t + 1].min())
                amp = abs(w_hi - w_lo)
                if np.isfinite(amp) and amp > 0:
                    return amp
            except Exception:
                pass
    atr = _atr_default(df, 14)
    if atr is not None and atr > 0:
        return atr
    try:
        w_hi = float(df["high"].tail(60).max())
        w_lo = float(df["low"].tail(60).min())
        amp = 0.5 * abs(w_hi - w_lo)
        return amp if amp > 0 else None
    except Exception:
        return None

def _infer_direction(e: Any) -> str:
    d = getattr(e, "direction", None)
    if isinstance(d, str) and d:
        return d.lower()
    key = str(getattr(e, "pattern_key", getattr(e, "key", getattr(e, "name", "")))).lower()
    if any(w in key for w in ("bear", "top", "descending", "inverted", "down")):
        return "down"
    if any(w in key for w in ("bull", "bottom", "ascending", "up")):
        return "up"
    return "neutral"

def _infer_kind(e: Any) -> str:
    k = getattr(e, "kind", None)
    if isinstance(k, str) and k:
        return k
    key = str(getattr(e, "pattern_key", getattr(e, "key", getattr(e, "name", "")))).lower()
    if "candle" in key:
        return "candle"
    return "chart"

def _infer_name(e: Any) -> str:
    n = getattr(e, "name", None) or getattr(e, "pattern_key", None) or getattr(e, "key", None) or e.__class__.__name__
    return str(n)

def enrich_events(df: pd.DataFrame, events: List[Any], default_lookback: int = 50, dom_service=None, symbol: str = None) -> List[Any]:
    """
    Enrich pattern events with timestamps, targets, and optional DOM confirmation.

    Args:
        df: OHLC dataframe
        events: List of pattern events
        default_lookback: Default lookback bars for patterns without start_ts
        dom_service: Optional DOM aggregator service for confidence adjustment
        symbol: Trading symbol for DOM lookup

    Returns:
        List of enriched events with adjusted confidence scores
    """
    out: List[Any] = []
    for e in events:
        try:
            name = _infer_name(e)
            kind = _infer_kind(e)
            direction = _infer_direction(e)
            for attr, val in (("name", name), ("kind", kind), ("direction", direction)):
                try:
                    setattr(e, attr, val)
                except Exception:
                    pass

            ts_ms = None
            for cand in (getattr(e, "confirm_ts", None),
                         getattr(e, "end_ts", None),
                         getattr(e, "ts", None)):
                ts_ms = _to_ms(cand)
                if ts_ms is not None:
                    break
            if ts_ms is None:
                idx = getattr(e, "confirm_idx", None)
                if isinstance(idx, int) and 0 <= idx < len(df):
                    ts_ms = int(df["ts_utc"].iloc[idx])

            st_ms = _to_ms(getattr(e, "start_ts", None))
            en_ms = _to_ms(getattr(e, "end_ts", None))
            if st_ms is None:
                sidx = getattr(e, "start_idx", None)
                if isinstance(sidx, int) and 0 <= sidx < len(df):
                    st_ms = int(df["ts_utc"].iloc[sidx])
            if en_ms is None:
                eidx = getattr(e, "end_idx", None)
                if isinstance(eidx, int) and 0 <= eidx < len(df):
                    en_ms = int(df["ts_utc"].iloc[eidx])

            if st_ms is None or en_ms is None:
                if ts_ms is None:
                    continue
                ts_arr = df["ts_utc"].to_numpy(dtype="int64")
                i = np.searchsorted(ts_arr, ts_ms, side="left")
                L = max(0, i - default_lookback)
                st_ms = int(ts_arr[L]) if st_ms is None else st_ms
                en_ms = int(ts_arr[max(0, i - 1)]) if en_ms is None else en_ms

            for attr, val in (("confirm_ts", ts_ms), ("start_ts", st_ms), ("end_ts", en_ms)):
                try:
                    setattr(e, attr, int(val))
                except Exception:
                    pass

            px = getattr(e, "confirm_price", None)
            try:
                px = float(px)
            except Exception:
                if ts_ms is not None:
                    px = _nearest_close_at_ts(df, ts_ms)
            if px is None:
                continue
            try:
                setattr(e, "confirm_price", float(px))
            except Exception:
                pass

            tgt = getattr(e, "target_price", None)
            if tgt is None:
                amp = _amplitude_guess(df, e)
                if amp is not None:
                    d = str(direction).lower()
                    # negative move for bearish / down patterns
                    sign = -1.0 if d in ("down", "bear", "bearish", "short", "sell") else 1.0
                    tgt = float(px) + sign * float(amp)
            try:
                if tgt is not None:
                    setattr(e, "target_price", float(tgt))
            except Exception:
                pass

            # Calculate failure_price (stop loss / invalidation point) if not set
            fail = getattr(e, "failure_price", None)
            # Check if failure_price is actually a price (not a number of bars like 40.0)
            # If it's close to the confirm_price, it's a real price; otherwise it's likely bars
            if fail is not None:
                try:
                    fail_val = float(fail)
                    # Check if it looks like a price value (within reasonable range of confirm_price)
                    # If it's less than 1000 and px > 1000 (pip format), or if it's very far from px, it's probably not a price
                    if abs(fail_val - float(px)) > (float(px) * 10):  # More than 10x away = not a price
                        fail = None  # Reset and recalculate
                except Exception:
                    fail = None

            if fail is None:
                # Calculate stop loss based on direction and amplitude
                amp = _amplitude_guess(df, e)
                if amp is not None:
                    d = str(direction).lower()
                    # Stop loss is opposite direction from target
                    # For bull patterns: stop below entry
                    # For bear patterns: stop above entry
                    # Use 0.5 * amplitude as stop distance (risk/reward = 1:2)
                    stop_distance = 0.5 * float(amp)
                    if d in ("down", "bear", "bearish", "short", "sell"):
                        fail = float(px) + stop_distance  # Stop above for bear patterns
                    else:
                        fail = float(px) - stop_distance  # Stop below for bull patterns
            try:
                if fail is not None:
                    setattr(e, "failure_price", float(fail))
            except Exception:
                pass

            # Apply DOM confirmation if available
            if dom_service and symbol:
                try:
                    from .....patterns.dom_confirmation import DOMPatternConfirmation
                    confirmer = DOMPatternConfirmation(dom_service)

                    # Get original score
                    original_score = getattr(e, 'score', 0.5)

                    # Get DOM confirmation
                    confirmation = confirmer.confirm_pattern(direction, symbol, original_score)

                    # Update score and add confirmation metadata
                    setattr(e, 'score', confirmation['adjusted_score'])
                    setattr(e, 'dom_confirmation', confirmation)
                    setattr(e, 'dom_aligned', confirmation['dom_aligned'])

                except Exception:
                    # Silently continue if DOM confirmation fails
                    pass

            out.append(e)
        except Exception:
            continue
    return out
