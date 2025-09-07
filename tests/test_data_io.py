import pandas as pd
import numpy as np
import pytest
from src.forex_diffusion.data import io as data_io
import pandas as pd
import numpy as np
import pytest
from src.forex_diffusion.data import io as data_io


def make_minute_candles(start_ts_ms: int, n: int):
    """
    Helper: create n consecutive 1-minute candles starting at start_ts_ms (ms).
    Returns DataFrame with ts_utc, open, high, low, close, volume
    """
    rows = []
    price = 1.1000
    for i in range(n):
        ts = start_ts_ms + i * 60_000
        open_p = price + np.random.randn() * 1e-4
        close_p = open_p + np.random.randn() * 1e-4
        high_p = max(open_p, close_p) + abs(np.random.randn() * 5e-5)
        low_p = min(open_p, close_p) - abs(np.random.randn() * 5e-5)
        vol = float(max(1.0, np.abs(np.random.randn() * 10.0)))
        rows.append({"ts_utc": int(ts), "open": float(open_p), "high": float(high_p), "low": float(low_p), "close": float(close_p), "volume": vol})
        price = close_p
    return pd.DataFrame(rows)


def test_validate_candles_df_dup_and_price_checks():
    start = int(pd.Timestamp("2023-01-01T00:00:00Z").value // 1_000_000)
    df = make_minute_candles(start, 6)
    # introduce a duplicate ts and an invalid price row
    dup_row = df.iloc[2].copy()
    df = pd.concat([df.iloc[:3], pd.DataFrame([dup_row]), df.iloc[3:]]).reset_index(drop=True)
    # invalid price: set high < max(open, close)
    df.loc[4, "high"] = min(df.loc[4, "open"], df.loc[4, "close"]) - 0.00001
    cleaned, report = data_io.validate_candles_df(df.copy(), symbol="EUR/USD", timeframe="1m")
    # report should indicate duplicates removed and invalid price relation flagged
    assert report["n_dups"] >= 1
    assert "n_invalid_price_relation" in report
    assert report["n_invalid_price_relation"] >= 1
    # cleaned DataFrame should have no duplicate timestamps
    assert cleaned["ts_utc"].duplicated().sum() == 0
    # all price relations valid
    assert (cleaned["high"] >= cleaned[["open", "close"]].max(axis=1)).all()
    assert (cleaned["low"] <= cleaned[["open", "close"]].min(axis=1)).all()


def test_resample_candles_1m_to_2m():
    # build deterministic candles for 4 minutes
    start = int(pd.Timestamp("2023-01-02T00:00:00Z").value // 1_000_000)
    rows = []
    # minute 0
    rows.append({"ts_utc": start, "open": 1.0, "high": 1.02, "low": 0.99, "close": 1.01, "volume": 10.0})
    # minute 1
    rows.append({"ts_utc": start + 60_000, "open": 1.01, "high": 1.03, "low": 1.0, "close": 1.02, "volume": 5.0})
    # minute 2
    rows.append({"ts_utc": start + 2 * 60_000, "open": 1.02, "high": 1.04, "low": 1.01, "close": 1.03, "volume": 7.0})
    # minute 3
    rows.append({"ts_utc": start + 3 * 60_000, "open": 1.03, "high": 1.05, "low": 1.02, "close": 1.04, "volume": 3.0})
    df = pd.DataFrame(rows)
    # resample to 2m
    r = data_io.resample_candles(df, src_tf="1m", tgt_tf="2m")
    # expect 2 rows
    assert len(r) == 2
    # first 2-minute bar: open = first open (1.0), close = last close in period (1.02), high = max(1.02,1.03)=1.03, low = min(0.99,1.0)=0.99, volume = 15.0
    first = r.iloc[0]
    assert pytest.approx(first["open"], rel=1e-6) == 1.0
    assert pytest.approx(first["close"], rel=1e-6) == 1.02
    assert pytest.approx(first["high"], rel=1e-6) == 1.03
    assert pytest.approx(first["low"], rel=1e-6) == 0.99
    assert pytest.approx(first["volume"], rel=1e-6) == 15.0
    # second 2-minute bar
    second = r.iloc[1]
    assert pytest.approx(second["open"], rel=1e-6) == 1.02
    assert pytest.approx(second["close"], rel=1e-6) == 1.04
    assert pytest.approx(second["high"], rel=1e-6) == 1.05
    assert pytest.approx(second["low"], rel=1e-6) == 1.01
    assert pytest.approx(second["volume"], rel=1e-6) == 10.0

def make_minute_candles(start_ts_ms: int, n: int):
    """
    Helper: create n consecutive 1-minute candles starting at start_ts_ms (ms).
    Returns DataFrame with ts_utc, open, high, low, close, volume
    """
    rows = []
    price = 1.1000
    for i in range(n):
        ts = start_ts_ms + i * 60_000
        open_p = price + np.random.randn() * 1e-4
        close_p = open_p + np.random.randn() * 1e-4
        high_p = max(open_p, close_p) + abs(np.random.randn() * 5e-5)
        low_p = min(open_p, close_p) - abs(np.random.randn() * 5e-5)
        vol = float(max(1.0, np.abs(np.random.randn() * 10.0)))
        rows.append({"ts_utc": int(ts), "open": float(open_p), "high": float(high_p), "low": float(low_p), "close": float(close_p), "volume": vol})
        price = close_p
    return pd.DataFrame(rows)


def test_validate_candles_df_dup_and_price_checks():
    start = int(pd.Timestamp("2023-01-01T00:00:00Z").value // 1_000_000)
    df = make_minute_candles(start, 6)
    # introduce a duplicate ts and an invalid price row
    dup_row = df.iloc[2].copy()
    df = pd.concat([df.iloc[:3], pd.DataFrame([dup_row]), df.iloc[3:]]).reset_index(drop=True)
    # invalid price: set high < max(open, close)
    df.loc[4, "high"] = min(df.loc[4, "open"], df.loc[4, "close"]) - 0.00001
    cleaned, report = data_io.validate_candles_df(df.copy(), symbol="EUR/USD", timeframe="1m")
    # report should indicate duplicates removed and invalid price relation flagged
    assert report["n_dups"] >= 1
    assert "n_invalid_price_relation" in report
    assert report["n_invalid_price_relation"] >= 1
    # cleaned DataFrame should have no duplicate timestamps
    assert cleaned["ts_utc"].duplicated().sum() == 0
    # all price relations valid
    assert (cleaned["high"] >= cleaned[["open", "close"]].max(axis=1)).all()
    assert (cleaned["low"] <= cleaned[["open", "close"]].min(axis=1)).all()


def test_resample_candles_1m_to_2m():
    # build deterministic candles for 4 minutes
    start = int(pd.Timestamp("2023-01-02T00:00:00Z").value // 1_000_000)
    rows = []
    # minute 0
    rows.append({"ts_utc": start, "open": 1.0, "high": 1.02, "low": 0.99, "close": 1.01, "volume": 10.0})
    # minute 1
    rows.append({"ts_utc": start + 60_000, "open": 1.01, "high": 1.03, "low": 1.0, "close": 1.02, "volume": 5.0})
    # minute 2
    rows.append({"ts_utc": start + 2 * 60_000, "open": 1.02, "high": 1.04, "low": 1.01, "close": 1.03, "volume": 7.0})
    # minute 3
    rows.append({"ts_utc": start + 3 * 60_000, "open": 1.03, "high": 1.05, "low": 1.02, "close": 1.04, "volume": 3.0})
    df = pd.DataFrame(rows)
    # resample to 2m
    r = data_io.resample_candles(df, src_tf="1m", tgt_tf="2m")
    # expect 2 rows
    assert len(r) == 2
    # first 2-minute bar: open = first open (1.0), close = last close in period (1.02), high = max(1.02,1.03)=1.03, low = min(0.99,1.0)=0.99, volume = 15.0
    first = r.iloc[0]
    assert pytest.approx(first["open"], rel=1e-6) == 1.0
    assert pytest.approx(first["close"], rel=1e-6) == 1.02
    assert pytest.approx(first["high"], rel=1e-6) == 1.03
    assert pytest.approx(first["low"], rel=1e-6) == 0.99
    assert pytest.approx(first["volume"], rel=1e-6) == 15.0
    # second 2-minute bar
    second = r.iloc[1]
    assert pytest.approx(second["open"], rel=1e-6) == 1.02
    assert pytest.approx(second["close"], rel=1e-6) == 1.04
    assert pytest.approx(second["high"], rel=1e-6) == 1.05
    assert pytest.approx(second["low"], rel=1e-6) == 1.01
    assert pytest.approx(second["volume"], rel=1e-6) == 10.0
