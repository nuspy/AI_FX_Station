#!/usr/bin/env python3
"""
Auto-tuning script for features bulk insert and DBWriter batching.

- Runs grid search over bulk_batch_size and db_writer batch_size values using scripts/benchmark_bulk.py functions.
- Records rows/sec for direct bulk and DBWriter paths.
- Emits CSV in artifacts/bench_tuning_<ts>.csv and optionally updates configs/default.yaml with best bulk_batch_size.

Usage:
  python scripts/auto_tune.py --db-url postgresql://fx:fxpass@127.0.0.1:5432/magicforex --rows 50000 --grid "250,500,1000" --repeats 2 --update-config
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

# allow importing scripts.benchmark_bulk
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_bulk import bench_direct, bench_dbwriter, generate_rows  # type: ignore
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db-url", required=True)
    p.add_argument("--rows", type=int, default=50000)
    p.add_argument("--grid", default="250,500,1000", help="comma-separated batch sizes to test")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--symbol", default="EUR/USD")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--update-config", action="store_true", help="update configs/default.yaml with best bulk_batch_size")
    return p.parse_args()


def run_grid(db_url: str, rows: int, grid: List[int], repeats: int, batch_for_writer: List[int], symbol: str, timeframe: str):
    results = []
    for bulk in grid:
        for bwriter in batch_for_writer:
            rates = []
            for rep in range(repeats):
                # generate rows to keep identical workload across tests
                rows_list = generate_rows(symbol, timeframe, int(datetime.utcnow().timestamp() * 1000), rows)
                # direct bulk benchmark with batch size = bulk
                t_direct = bench_direct(db_url, rows_list, bulk)
                r_direct = rows / t_direct if t_direct > 0 else 0.0
                # DBWriter benchmark with writer batch roughly bwriter
                t_writer = bench_dbwriter(db_url, rows_list, bwriter)
                r_writer = rows / t_writer if t_writer > 0 else 0.0
                rates.append((r_direct, r_writer, t_direct, t_writer))
                # small pause
                time.sleep(0.5)
            # aggregate
            avg_direct = sum(r[0] for r in rates) / len(rates)
            avg_writer = sum(r[1] for r in rates) / len(rates)
            results.append({
                "bulk_batch_size": bulk,
                "dbwriter_batch_size": bwriter,
                "avg_rows_sec_direct": avg_direct,
                "avg_rows_sec_dbwriter": avg_writer,
                "repeats": repeats,
            })
            print(f"Tested bulk={bulk} writer={bwriter} -> direct={avg_direct:.1f} r/s writer={avg_writer:.1f} r/s")
    return results


def save_results(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"bench_tuning_{ts}.csv"
    keys = ["bulk_batch_size", "dbwriter_batch_size", "avg_rows_sec_direct", "avg_rows_sec_dbwriter", "repeats"]
    with out_file.open("w", newline='', encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in keys})
    return out_file


def update_config_with_best(out_file: Path, config_path: Path):
    # read CSV, pick best avg_rows_sec_dbwriter
    best = None
    with out_file.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            val = float(r.get("avg_rows_sec_dbwriter", 0.0))
            if best is None or val > best[0]:
                best = (val, int(r["bulk_batch_size"]), int(r["dbwriter_batch_size"]))
    if best is None:
        return None
    bs = best[1]
    # update configs/default.yaml -> features.bulk_batch_size
    cfg = {}
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "features" not in cfg:
        cfg["features"] = {}
    cfg["features"]["bulk_batch_size"] = bs
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return bs


def main():
    args = parse_args()
    grid = [int(x) for x in args.grid.split(",") if x.strip()]
    # for writer batch, test same grid (can be parameterized)
    writer_grid = grid
    print("Auto-tune grid:", grid)
    results = run_grid(args.db_url, args.rows, grid, args.repeats, writer_grid, args.symbol, args.timeframe)
    out = save_results(results, ROOT / "artifacts")
    print("Results saved to:", out)
    if args.update_config:
        cfg_path = ROOT / "configs" / "default.yaml"
        bs = update_config_with_best(out, cfg_path)
        if bs is not None:
            print("Updated configs/default.yaml with bulk_batch_size =", bs)
        else:
            print("No best found; config not updated.")


if __name__ == "__main__":
    main()
