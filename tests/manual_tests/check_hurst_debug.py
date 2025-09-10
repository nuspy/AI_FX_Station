#!/usr/bin/env python3
"""
tests/manual_tests/check_hurst_debug.py

Debug Hurst discrepancies: for a given symbol/timeframe and ts_utc prints:
 - pipeline stored (reconstructed) Hurst raw value (using Standardizer mu/sigma if available)
 - recomputed Hurst via hurst_aggvar, _rs_hurst, hurst_feature on multiple windows
 - series length and basic stats

Usage (PowerShell):
  .\scripts\check_hurst_debug.ps1 -Symbol "EUR/USD" -Timeframe "1m" -TsUtc 1750376100000

"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--ts_utc", type=int, required=True, help="Timestamp (ms UTC) to inspect")
    p.add_argument("--window_max", type=int, default=1024, help="Max lookback samples")
    args = p.parse_args()

    try:
        from forex_diffusion.services.db_service import DBService
        from forex_diffusion.features.pipeline import hurst_aggvar, _rs_hurst, hurst_feature
        from forex_diffusion.features.pipeline import to_datetime_index
    except Exception as e:
        print("Failed to import project modules:", e)
        raise SystemExit(1)

    db = DBService()
    eng = db.engine

    # load candles around ts_utc (latest window_max bars ending at ts)
    with eng.connect() as conn:
        q = f"SELECT ts_utc, open, high, low, close FROM market_data_candles WHERE symbol=:s AND timeframe=:tf AND ts_utc <= :t ORDER BY ts_utc DESC LIMIT :lim"
        rows = conn.execute(q, {"s": args.symbol, "tf": args.timeframe, "t": args.ts_utc, "lim": args.window_max}).fetchall()
        if not rows:
            print("No candles found for query; abort.")
            raise SystemExit(2)
        # rows are in descending ts order: convert to ascending
        rows = rows[::-1]
        import pandas as pd
        df = pd.DataFrame([dict(r._mapping) if hasattr(r, "_mapping") else {"ts_utc": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4]} for r in rows])

    print("Loaded candles:", len(df))
    print("First ts:", df["ts_utc"].iat[0], "Last ts:", df["ts_utc"].iat[-1])

    # Compute log returns series
    import numpy as np
    df["r"] = np.log(df["close"]).diff().fillna(0.0)

    # Compute multiple hurst estimators
    rseries = df["r"].dropna()
    print("Return series length:", len(rseries))
    # aggregated variance
    try:
        h_agg = hurst_aggvar(rseries)
    except Exception as e:
        h_agg = float("nan")
        print("hurst_aggvar failed:", e)
    # rs_hurst on full series
    try:
        rs_h = _rs_hurst(rseries.to_numpy())
    except Exception as e:
        rs_h = float("nan")
        print("_rs_hurst failed:", e)

    print(f"Aggregated-variance H: {h_agg}")
    print(f"R/S H (full series): {rs_h}")

    # Rolling hurst over different windows
    for w in [32, 64, 128, 256]:
        try:
            if len(rseries) >= w:
                hf = hurst_feature(pd.DataFrame({"r": rseries.values}), window=w, out_col="htmp")
                val = float(hf["h_tmp"].iloc[-1]) if "h_tmp" in hf.columns else float("nan")
            else:
                val = float("nan")
        except Exception:
            val = float("nan")
        print(f"Rolling hurst (window={w}): {val}")

    # Show last 10 returns
    print("Last 10 returns:", rseries.tail(10).tolist())

    # Compare to stored pipeline value in features table (if exists)
    with eng.connect() as conn:
        # attempt to read latest features entry for this ts (features_df is not persisted, but latents exist)
        # find latent for exact ts
        lat = conn.execute("SELECT id, ts_utc, latent_json, regime_label FROM latents WHERE symbol=:s AND timeframe=:tf AND ts_utc = :t", {"s": args.symbol, "tf": args.timeframe, "t": args.ts_utc}).fetchone()
        if lat:
            print("Latent row found id:", lat[0], "ts_utc:", lat[1], "regime_label:", lat[3])
        else:
            print("No latent with exact ts_utc found; showing most recent latent for symbol/timeframe")
            lat = conn.execute("SELECT id, ts_utc, latent_json, regime_label FROM latents WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc DESC LIMIT 1", {"s": args.symbol, "tf": args.timeframe}).fetchone()
            if lat:
                print("Most recent latent id:", lat[0], "ts_utc:", lat[1], "regime_label:", lat[3])

    print("Done.")

if __name__ == '__main__':
    main()
