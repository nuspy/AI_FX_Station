#!/usr/bin/env python3
"""
tests/manual_tests/run_backfill_and_check.py

Run a backfill for symbol/timeframe (1m) using MarketDataService, then verify candles present in the same
market_data_candles table used by the GUI (counts, range, recent samples).

Usage (PowerShell):
  python tests/manual_tests/run_backfill_and_check.py --symbol "EUR/USD" --days 30

Notes:
  - Ensure .venv active and provider API key available (e.g., TIINGO_API_KEY).
  - This script calls existing backfill implementation (no duplication).
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import time
import datetime

ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="EUR/USD")
    p.add_argument("--days", type=int, default=30, help="Approx days to request for backfill")
    p.add_argument("--timeframe", default="1m", choices=["1m", "5m", "15m", "30m", "60m", "1d"])
    args = p.parse_args()

    try:
        from forex_diffusion.services.marketdata import MarketDataService
        from forex_diffusion.services.db_service import DBService
        from sqlalchemy import text, MetaData
    except Exception as e:
        print("Import failed:", e)
        raise SystemExit(1)

    print(f"Starting backfill+check for {args.symbol} {args.timeframe} (days={args.days})")

    # instantiate services
    try:
        ms = MarketDataService()
    except Exception as e:
        print("Failed to create MarketDataService:", e)
        raise SystemExit(1)

    try:
        dbs = DBService()
    except Exception as e:
        print("Failed to create DBService:", e)
        raise SystemExit(1)

    # run backfill using existing service (this performs provider fetch + upsert)
    try:
        print("Requesting backfill (this may take while for intraday data)...")
        # Use force_full=False to fetch recent intraday; providers may interpret differently
        report = ms.backfill_symbol_timeframe(args.symbol, args.timeframe, force_full=False)
        print("Backfill report:", report)
    except Exception as e:
        print("Backfill call failed:", e)
        raise SystemExit(2)

    # small pause to ensure DB transactions flushed
    time.sleep(1.0)

    # verify candles in market_data_candles via DBService (same table used by GUI)
    try:
        engine = dbs.engine
        meta = MetaData()
        meta.reflect(bind=engine, only=["market_data_candles"])
        tbl = meta.tables.get("market_data_candles")
        if tbl is None:
            print("market_data_candles table not found in DB after backfill.")
            raise SystemExit(3)
        with engine.connect() as conn:
            # count entries that match symbol variants
            sym = args.symbol
            alt = sym.replace("/", "") if "/" in sym else (f"{sym[0:3]}/{sym[3:6]}" if len(sym) == 6 else sym)
            params = {"s1": sym, "s2": alt, "tf": args.timeframe}
            q_count = text("SELECT COUNT(*) FROM market_data_candles WHERE (symbol = :s1 OR symbol = :s2) AND timeframe = :tf")
            cnt = conn.execute(q_count, params).scalar() or 0
            print(f"Rows for {args.symbol} {args.timeframe}: {cnt}")

            # range
            q_rng = text("SELECT MIN(ts_utc), MAX(ts_utc) FROM market_data_candles WHERE (symbol = :s1 OR symbol = :s2) AND timeframe = :tf")
            r = conn.execute(q_rng, params).fetchone()
            min_ts, max_ts = r[0], r[1]
            if min_ts and max_ts:
                dt_min = datetime.datetime.fromtimestamp(min_ts/1000, tz=datetime.timezone.utc).astimezone()
                dt_max = datetime.datetime.fromtimestamp(max_ts/1000, tz=datetime.timezone.utc).astimezone()
                print(f"Range: {min_ts} -> {max_ts} ({dt_min} -> {dt_max})")
            else:
                print("Range: None")

            # show recent 20 samples
            q_sample = text("SELECT id,symbol,timeframe,ts_utc,open,high,low,close,volume,resampled FROM market_data_candles WHERE (symbol = :s1 OR symbol = :s2) AND timeframe = :tf ORDER BY ts_utc DESC LIMIT 20")
            rows = conn.execute(q_sample, params).fetchall()
            print("Recent rows (up to 20):")
            for rr in rows:
                try:
                    mapping = getattr(rr, "_mapping", None)
                    if mapping is not None:
                        rec = dict(mapping)
                    else:
                        rec = {
                            "id": rr[0], "symbol": rr[1], "timeframe": rr[2], "ts_utc": rr[3],
                            "open": rr[4], "high": rr[5], "low": rr[6], "close": rr[7],
                            "volume": rr[8], "resampled": rr[9]
                        }
                except Exception:
                    rec = {"raw": str(rr)}
                t = rec.get("ts_utc")
                try:
                    dt = datetime.datetime.fromtimestamp(int(t)/1000, tz=datetime.timezone.utc).astimezone()
                    ts_str = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    ts_str = str(t)
                print(f" id={rec.get('id')} sym={rec.get('symbol')} ts={rec.get('ts_utc')} ({ts_str}) open={rec.get('open')} close={rec.get('close')}")
    except Exception as e:
        print("Verification failed:", e)
        raise SystemExit(4)

    print("Backfill + check completed successfully.")

if __name__ == "__main__":
    main()
