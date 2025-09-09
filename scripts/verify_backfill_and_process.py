#!/usr/bin/env python3
"""
verify_backfill_and_process.py

Run a backfill for a given symbol/timeframe, then inspect DB tables to verify persistence.

Usage (PowerShell):
  python .\scripts\verify_backfill_and_process.py --symbol "EUR/USD" --timeframe "1d" --force-full --show 5

Outputs:
 - Backfill report (actions with upsert reports)
 - Last N rows from market_data_candles, ticks_aggregate, signals
"""
from __future__ import annotations

import sys
import json
import argparse
from typing import Optional

# Ensure project src is importable
sys.path.insert(0, ".")

from forex_diffusion.services.marketdata import MarketDataService
from forex_diffusion.services.db_service import DBService
from sqlalchemy import MetaData, select

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="EUR/USD")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--force-full", dest="force_full", action="store_true", help="Force full backfill")
    p.add_argument("--show", type=int, default=5, help="How many rows to display per table")
    return p.parse_args()

def print_section(title: str):
    print("\n" + ("-" * 8) + f" {title} " + ("-" * 8))

def show_table_rows(db, table_name: str, n: int = 5):
    meta = MetaData()
    try:
        meta.reflect(bind=db.engine, only=[table_name])
    except Exception as e:
        print(f"Could not reflect {table_name}: {e}")
        return
    tbl = meta.tables.get(table_name)
    if tbl is None:
        print(f"Table {table_name} not found")
        return
    with db.engine.connect() as conn:
        # choose ordering column
        order_col = tbl.c.ts_utc if "ts_utc" in tbl.c else (tbl.c.ts_created_ms if "ts_created_ms" in tbl.c else None)
        stmt = select(tbl).order_by(order_col.desc() if order_col is not None else tbl.c.id.desc()).limit(n)
        try:
            rows = conn.execute(stmt).fetchall()
            print(f"Last {len(rows)} rows from {table_name}:")
            for r in rows:
                try:
                    print(dict(r))
                except Exception:
                    print(r)
        except Exception as e:
            print(f"Query failed for {table_name}: {e}")

def main():
    args = parse_args()
    svc = MarketDataService(provider_name="tiingo")
    print_section("Provider")
    print("Using provider:", svc.provider_name())
    print("Poll interval (s):", svc.poll_interval())

    print_section("Running backfill")
    try:
        report = svc.backfill_symbol_timeframe(args.symbol, args.timeframe, force_full=args.force_full)
        print("Backfill report:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except Exception as e:
        print("Backfill failed:", e)
        return

    print_section("DB inspection")
    db = DBService()
    # show candles, ticks, signals
    for t in ["market_data_candles", "ticks_aggregate", "signals"]:
        show_table_rows(db, t, args.show)

    print_section("Done")
    print("If candles were upserted, verify GUI Refresh to see signals/processing results.")

if __name__ == "__main__":
    main()
