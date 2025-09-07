#!/usr/bin/env python3
"""
Benchmark bulk insert for features table.

Usage:
    python scripts/benchmark_bulk.py --db-url sqlite:///./tmp.db --rows 10000 --batch 500
    python scripts/benchmark_bulk.py --db-url postgresql://user:pass@localhost/dbname --rows 200000 --batch 1000

This script measures time to insert N rows using:
 - direct DBService.write_features_bulk
 - DBWriter (enqueue + flush wait)

It prints throughput rows/sec for both approaches.
"""

import argparse
import json
import time
import random
import string
from datetime import datetime
from math import ceil

from sqlalchemy import create_engine
from src.forex_diffusion.services.db_service import DBService
from src.forex_diffusion.services.db_writer import DBWriter


def random_features():
    return {
        "r": random.uniform(-0.001, 0.001),
        "atr": random.uniform(0.0001, 0.001),
        "rsi": random.uniform(10, 90),
        "don_pos": random.uniform(0, 1),
    }


def generate_rows(symbol: str, timeframe: str, start_ts: int, n: int):
    rows = []
    for i in range(n):
        rows.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "ts_utc": start_ts + i * 60_000,
            "features": random_features(),
            "pipeline_version": "v1",
        })
    return rows


def bench_direct(engine_url: str, rows, batch_size):
    engine = create_engine(engine_url, future=True)
    dbs = DBService(engine=engine)
    t0 = time.time()
    # insert in chunks
    for i in range(0, len(rows), batch_size):
        dbs.write_features_bulk(rows[i:i+batch_size])
    dt = time.time() - t0
    return dt


def bench_dbwriter(engine_url: str, rows, batch_size):
    engine = create_engine(engine_url, future=True)
    dbs = DBService(engine=engine)
    dbw = DBWriter(db_service=dbs, batch_size=batch_size)
    dbw.start()
    t0 = time.time()
    for r in rows:
        enq = dbw.write_features_async(symbol=r["symbol"], timeframe=r["timeframe"], ts_utc=r["ts_utc"], features=r["features"], pipeline_version=r["pipeline_version"])
        if not enq:
            # if queue full, small sleep
            time.sleep(0.01)
    # wait for queue to drain
    start_wait = time.time()
    while not dbw.queue.empty() and (time.time() - start_wait) < 60:
        time.sleep(0.1)
    dbw.stop(flush=True, timeout=10.0)
    dt = time.time() - t0
    return dt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db-url", required=True)
    p.add_argument("--rows", type=int, default=10000)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--symbol", default="EUR/USD")
    p.add_argument("--timeframe", default="1m")
    return p.parse_args()


def main():
    args = parse_args()
    rows = generate_rows(args.symbol, args.timeframe, int(datetime.utcnow().timestamp() * 1000), args.rows)
    print("Running direct bulk benchmark: rows={}, batch={}".format(args.rows, args.batch))
    dt_direct = bench_direct(args.db_url, rows, args.batch)
    print("Direct bulk: time_sec={:.3f}, rows/s={:.1f}".format(dt_direct, args.rows / dt_direct if dt_direct > 0 else float("inf")))
    print("Running DBWriter benchmark (async enqueue->flush): rows={}, batch_est={}".format(args.rows, args.batch))
    dt_writer = bench_dbwriter(args.db_url, rows, args.batch)
    print("DBWriter: time_sec={:.3f}, rows/s={:.1f}".format(dt_writer, args.rows / dt_writer if dt_writer > 0 else float("inf")))


if __name__ == "__main__":
    main()
