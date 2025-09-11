#!/usr/bin/env python3
"""
scripts/count_candle_duplicates.py

Count how many times a candle (ts_utc) appears in market_data_candles for a given symbol/timeframe.

Usage examples:
  # single timestamps
  python scripts/count_candle_duplicates.py --symbol "EUR/USD" --timeframe 1m --ts-list 1752792900000,1752793200000

  # ranges JSON string
  python scripts/count_candle_duplicates.py --symbol "EUR/USD" --timeframe 1m --ranges '[{"prev_ts":1752792900000,"next_ts":1752793200000,"delta_ms":300000}]'

  # ranges file
  python scripts/count_candle_duplicates.py --symbol "EUR/USD" --timeframe 1m --ranges-file ./ranges.json

  # show top duplicate timestamps
  python scripts/count_candle_duplicates.py --symbol "EUR/USD" --timeframe 1m --top-duplicates 50

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("count_candle_duplicates")


def make_engine(url_or_path: str):
    # If looks like a sqlite file path, convert to sqlite:///...
    if url_or_path.startswith("sqlite://") or "://" in url_or_path:
        return create_engine(url_or_path)
    # treat as local path
    path = os.path.expanduser(url_or_path)
    if not path.startswith("/"):
        path = os.path.abspath(path)
    return create_engine(f"sqlite:///{path}")


def parse_ranges_arg(s: str) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
        raise ValueError("ranges JSON must be a list")
    except Exception as e:
        raise ValueError(f"Failed to parse ranges JSON: {e}")


def load_ranges_file(p: str) -> List[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as fh:
        return parse_ranges_arg(fh.read())


def count_ts(engine, symbol: str, timeframe: str, ts: int) -> int:
    q = text("SELECT COUNT(1) FROM market_data_candles WHERE symbol = :s AND timeframe = :tf AND ts_utc = :t")
    with engine.connect() as conn:
        return int(conn.execute(q, {"s": symbol, "tf": timeframe, "t": int(ts)}).scalar() or 0)


def counts_in_range(engine, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> Dict[int, int]:
    """
    Return mapping ts_utc -> count for rows in [start_ts, end_ts).
    """
    q = text(
        "SELECT ts_utc, COUNT(1) as c FROM market_data_candles "
        "WHERE symbol = :s AND timeframe = :tf AND ts_utc >= :a AND ts_utc < :b "
        "GROUP BY ts_utc ORDER BY ts_utc ASC"
    )
    out = {}
    with engine.connect() as conn:
        rows = conn.execute(q, {"s": symbol, "tf": timeframe, "a": int(start_ts), "b": int(end_ts)}).fetchall()
        for r in rows:
            try:
                ts = int(r[0])
                c = int(r[1])
            except Exception:
                continue
            out[ts] = c
    return out


def top_duplicates(engine, symbol: str, timeframe: str, limit: int = 100):
    q = text(
        "SELECT ts_utc, COUNT(1) AS c FROM market_data_candles "
        "WHERE symbol = :s AND timeframe = :tf "
        "GROUP BY ts_utc HAVING c > 1 ORDER BY c DESC LIMIT :lim"
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"s": symbol, "tf": timeframe, "lim": int(limit)}).fetchall()
        return [(int(r[0]), int(r[1])) for r in rows]


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Count candle occurrences to detect duplicate inserts from backfill.")
    p.add_argument("--db", default="./data/forex_diffusion.db", help="Database path or SQLAlchemy URL (default ./data/forex_diffusion.db)")
    p.add_argument("--symbol", required=True, help="Symbol, e.g. 'EUR/USD'")
    p.add_argument("--timeframe", required=True, help="Timeframe string, e.g. '1m'")
    p.add_argument("--ts-list", help="Comma-separated list of ts_utc integers to check")
    p.add_argument("--ranges", help="JSON string with list of ranges [{'prev_ts':..., 'next_ts':..., 'delta_ms':...}, ...]")
    p.add_argument("--ranges-file", help="Path to JSON file containing ranges")
    p.add_argument("--top-duplicates", type=int, default=0, help="Show top N duplicate timestamps (count>1)")
    args = p.parse_args(argv)

    engine = make_engine(args.db)
    logger.info("Using DB: %s", args.db)
    logger.info("Checking symbol=%s timeframe=%s", args.symbol, args.timeframe)

    # single timestamps
    if args.ts_list:
        ts_items = [t.strip() for t in args.ts_list.split(",") if t.strip()]
        for t in ts_items:
            try:
                ts = int(t)
            except Exception:
                logger.error("Invalid ts value: %s", t)
                continue
            c = count_ts(engine, args.symbol, args.timeframe, ts)
            logger.info("ts=%d count=%d", ts, c)

    # ranges
    ranges = []
    if args.ranges_file:
        ranges = load_ranges_file(args.ranges_file)
    elif args.ranges:
        ranges = parse_ranges_arg(args.ranges)

    for r in ranges:
        prev = int(r.get("prev_ts"))
        nxt = int(r.get("next_ts"))
        delta = int(r.get("delta_ms", 0))
        logger.info("Checking range prev=%d next=%d delta_ms=%d", prev, nxt, delta)
        mapping = counts_in_range(engine, args.symbol, args.timeframe, prev, nxt)
        total_rows = sum(mapping.values())
        distinct_ts = len(mapping)
        duplicates = {ts: c for ts, c in mapping.items() if c > 1}
        logger.info("Range rows=%d distinct_ts=%d duplicates=%d", total_rows, distinct_ts, len(duplicates))
        if duplicates:
            logger.info("Duplicates in range (ts -> count):")
            for ts, c in sorted(duplicates.items()):
                logger.info("  %d -> %d", ts, c)
        else:
            logger.info("No duplicates found in specified range.")

    # top duplicates global
    if args.top_duplicates and args.top_duplicates > 0:
        tups = top_duplicates(engine, args.symbol, args.timeframe, limit=args.top_duplicates)
        if tups:
            logger.info("Top %d duplicate timestamps (global):", args.top_duplicates)
            for ts, c in tups:
                logger.info("  ts=%d count=%d", ts, c)
        else:
            logger.info("No duplicate timestamps found globally for %s %s", args.symbol, args.timeframe)


if __name__ == "__main__":
    main()
