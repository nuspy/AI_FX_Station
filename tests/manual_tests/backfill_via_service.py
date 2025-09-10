#!/usr/bin/env python3
"""
tests/manual_tests/backfill_via_service.py

Trigger provider backfill using MarketDataService.backfill_symbol_timeframe and verify results
directly against the same 'market_data_candles' table used by the GUI (via DBService).

Usage (PowerShell):
  python tests/manual_tests/backfill_via_service.py --symbol "EUR/USD" --timeframe "1d" --days 30
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
    p.add_argument("--timeframe", default="1m", choices=["1m","5m","15m","30m","60m","1d"])
    p.add_argument("--days", type=int, default=365, help="Days to backfill (for 1d this is days, for 1m approximate)")
    args = p.parse_args()

    try:
        from forex_diffusion.services.marketdata import MarketDataService
        from forex_diffusion.services.db_service import DBService
        from forex_diffusion.data import io as data_io
        from sqlalchemy import select, MetaData, text
    except Exception as e:
        print("Import failed:", e)
        raise SystemExit(1)

    # Use MarketDataService for provider/backfill logic
    ms = MarketDataService()
    print("Provider:", ms.provider_name())

    # Use DBService to ensure we target the same database/table as the GUI
    dbs = DBService()
    engine = dbs.engine

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - args.days * 24 * 3600 * 1000

    print(f"Requesting backfill for {args.symbol} {args.timeframe} from {datetime.datetime.fromtimestamp(start_ms/1000)} to {datetime.datetime.fromtimestamp(now_ms/1000)}")
    try:
        report = ms.backfill_symbol_timeframe(args.symbol, args.timeframe, force_full=False)
        print("Backfill report:", report)
    except Exception as e:
        print("Backfill failed:", e)
        raise SystemExit(2)

    # Verify directly against market_data_candles table used by GUI
    try:
        meta = MetaData()
        meta.reflect(bind=engine, only=["market_data_candles"])
        tbl = meta.tables.get("market_data_candles")
        if tbl is None:
            print("market_data_candles table not found after backfill.")
            raise SystemExit(3)
        with engine.connect() as conn:
            # count matching symbol/timeframe (normalize both symbol variants)
            sym = args.symbol
            alt = sym.replace("/", "") if "/" in sym else f"{sym[0:3]}/{sym[3:6]}" if len(sym)==6 else sym
            params = {"s1": sym, "s2": alt, "tf": args.timeframe}
            q_count = text("SELECT COUNT(*) FROM market_data_candles WHERE (symbol = :s1 OR symbol = :s2) AND timeframe = :tf")
            cnt = conn.execute(q_count, params).scalar() or 0
            print(f"After backfill: rows for {args.symbol} {args.timeframe}: {cnt}")
            q_sample = text("SELECT id, symbol, timeframe, ts_utc, open, high, low, close, volume, resampled FROM market_data_candles WHERE (symbol = :s1 OR symbol = :s2) AND timeframe = :tf ORDER BY ts_utc DESC LIMIT 5")
            rows = conn.execute(q_sample, params).fetchall()
            print("Recent rows (up to 5):")
            for rr in rows:
                t = rr[3]
                try:
                    dt = datetime.datetime.fromtimestamp(int(t)/1000, tz=datetime.timezone.utc).astimezone()
                    ts_str = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    ts_str = str(int(t))
                print(f"  id={rr[0]} symbol={rr[1]} tf={rr[2]} ts={t} ({ts_str}) open={rr[4]} high={rr[5]} low={rr[6]} close={rr[7]} vol={rr[8]} resampled={rr[9]}")
    except Exception as e:
        print("Verification query failed:", e)
        raise SystemExit(4)

if __name__ == "__main__":
    main()
