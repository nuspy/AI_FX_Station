# scripts/test_patterns.py
import sys, pathlib, pandas as pd, numpy as np
from collections import Counter

# Assicura che 'src' sia nel path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.forex_diffusion.patterns.registry import PatternRegistry

def make_dummy_df_hammer(n=200):
    ts = pd.date_range("2025-06-01", periods=n, freq="min", tz="UTC")
    price = np.cumsum(np.random.randn(n) * 0.0002) + 1.10
    o = price + np.random.randn(n)*0.00005
    c = price + np.random.randn(n)*0.00005
    h = np.maximum(o, c) + np.abs(np.random.randn(n))*0.0001
    l = np.minimum(o, c) - np.abs(np.random.randn(n))*0.0001
    # forza un hammer sull'ultima barra
    i = n-1
    body = 0.00005
    o[i] = price[i]
    c[i] = price[i] + body
    h[i] = c[i] + 0.00002
    l[i] = o[i] - (body*3.5)
    return pd.DataFrame({
        "ts_utc": (ts.view("int64") // 10**6).astype("int64"),
        "open": o, "high": h, "low": l, "close": c,
    })

def _getattr_fallback(obj, *names, default=None):
    """Ritorna il primo attributo presente tra 'names'. Supporta anche dict."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
        if isinstance(obj, dict) and n in obj:
            return obj[n]
    return default

def normalize_event(e):
    """Converte un evento in dict standard, con fallback su varianti di naming."""
    return {
        "key": _getattr_fallback(e, "key", "name", "pattern", default=type(e).__name__),
        "kind": _getattr_fallback(e, "kind", "type", default="unknown"),
        "direction": _getattr_fallback(e, "direction", "dir", default="neutral"),
        "state": _getattr_fallback(e, "state", "status", default="unknown"),
        "confirm_ts": _getattr_fallback(e, "confirm_ts", "t_confirm", "ts", "time", default=None),
        "target_price": _getattr_fallback(e, "target_price", "target", "tp", default=None),
        "raw": e,
    }

if __name__ == "__main__":
    reg = PatternRegistry()
    df = make_dummy_df_hammer()
    events = []
    for det in reg.detectors(kinds=["candle","chart"]):
        try:
            events.extend(det.detect(df))
        except Exception as ex:
            print(f"[WARN] detector {getattr(det, 'key', det)} failed: {ex}")

    print(f"Total events: {len(events)}")

    if not events:
        sys.exit(0)

    # Normalizza e stampa ultimi 10
    norm = [normalize_event(e) for e in events]
    print("\nLast 10 events (normalized):")
    for e in norm[-10:]:
        ts = e["confirm_ts"]
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        print(f"- {e['key']:30s} | {e['kind']:7s} | {e['direction']:5s} | {e['state']:9s} | ts={ts} | tgt={e['target_price']}")

    # Riepilogo
    by_kind = Counter(e["kind"] for e in norm)
    by_key  = Counter(e["key"] for e in norm)
    print("\nSummary by kind:", dict(by_kind))
    print("Top keys:", by_key.most_common(10))

    # Se serve debug più profondo: mostra i nomi attributi disponibili del primo evento
    sample = events[-1]
    print("\nSample event type:", type(sample))
    print("Has attributes:", [n for n in dir(sample) if not n.startswith("_")][:30])
