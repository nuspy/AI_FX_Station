#!/usr/bin/env python3
"""
Insert test signals and print recent rows from the DB.

Usage:
  python .\scripts\send_and_check_signals.py --count 5 --interval 0.5 --show 10

This script uses the project's DBService to persist signals and then reads back the latest rows.
"""
from __future__ import annotations

import argparse
import time
import json
from typing import Any, Dict

# ensure project src is importable when run from repo root
import sys
sys.path.insert(0, ".")

from forex_diffusion.services.db_service import DBService
from sqlalchemy import MetaData, select

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=5, help="Number of test signals to write")
    p.add_argument("--interval", type=float, default=0.2, help="Seconds between inserts")
    p.add_argument("--show", type=int, default=10, help="How many recent rows to display after inserts")
    p.add_argument("--symbol", default="EUR/USD")
    p.add_argument("--timeframe", default="1m")
    return p.parse_args()

def main():
    args = parse_args()
    db = DBService()
    print(f"Using DB engine: {db.engine}")
    for i in range(args.count):
        payload: Dict[str, Any] = {
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "entry_price": round(1.2000 + i * 0.0001, 6),
            "target_price": round(1.2050 + i * 0.0001, 6),
            "stop_price": round(1.1950 + i * 0.0001, 6),
            "metrics": {"test_index": i},
        }
        ts = int(time.time() * 1000)
        db.write_signal(payload)
        print(f"Written signal #{i+1} at ts={ts}: {payload}")
        time.sleep(args.interval)

    # read back recent rows
    meta = MetaData()
    meta.reflect(bind=db.engine, only=["signals"])
    tbl = meta.tables.get("signals")
    if tbl is None:
        print("signals table not found.")
        return

    with db.engine.connect() as conn:
        stmt = select(
            tbl.c.id, tbl.c.ts_created_ms, tbl.c.symbol, tbl.c.timeframe,
            tbl.c.entry_price, tbl.c.target_price, tbl.c.stop_price, tbl.c.metrics
        ).order_by(tbl.c.ts_created_ms.desc()).limit(args.show)
        rows = conn.execute(stmt).fetchall()
        print(f"\nLast {len(rows)} signals (most recent first):")
        for r in rows:
            print({
                "id": int(r[0]),
                "ts_created_ms": int(r[1]) if r[1] is not None else None,
                "symbol": r[2],
                "timeframe": r[3],
                "entry": r[4],
                "target": r[5],
                "stop": r[6],
                "metrics": json.loads(r[7]) if r[7] else {},
            })

if __name__ == "__main__":
    main()
