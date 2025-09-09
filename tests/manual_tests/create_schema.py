#!/usr/bin/env python3
"""
Create minimal schema required by the GUI: table market_data_candles.
Reads DB URL from configs/default.yaml if contains sqlite:///..., else uses ./data/forex_diffusion.db as fallback.

Usage:
  python tests/manual_tests/create_schema.py
  python tests/manual_tests/create_schema.py --db-path ./data/forex_diffusion.db
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

def find_sqlite_path_in_yaml(yaml_path: Path) -> str | None:
    try:
        import yaml
    except Exception:
        return None
    if not yaml_path.exists():
        return None
    raw = yaml_path.read_text(encoding="utf-8")
    m = re.search(r"(sqlite:/{3,4}[^\s'\"\n]+)", raw)
    if m:
        url = m.group(1)
        if url.startswith("sqlite:///"):
            return url[len("sqlite:///"):]
        if url.startswith("sqlite:////"):
            return url[len("sqlite:////"):]
        return url
    try:
        y = yaml.safe_load(raw)
    except Exception:
        return None
    def dfs(n):
        if isinstance(n, dict):
            for v in n.values():
                r = dfs(v)
                if r:
                    return r
        elif isinstance(n, list):
            for it in n:
                r = dfs(it)
                if r:
                    return r
        elif isinstance(n, str):
            if "sqlite" in n:
                if n.startswith("sqlite:///"):
                    return n[len("sqlite:///"):]
                return n
        return None
    return dfs(y)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", help="Explicit sqlite file path to use/create")
    p.add_argument("--config", default="configs/default.yaml", help="Config YAML to probe for sqlite URL")
    args = p.parse_args()

    db_path = None
    if args.db_path:
        db_path = Path(args.db_path)
    else:
        cfg = Path(args.config)
        found = find_sqlite_path_in_yaml(cfg)
        if found:
            fp = found
            if fp.startswith("/") and re.match(r"^/[A-Za-z]:", fp):
                fp = fp[1:]
            db_path = Path(fp)

    if db_path is None:
        db_path = Path("data") / "forex_diffusion.db"
        print(f"No DB path in config; using fallback: {db_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, BigInteger, Float, Boolean, Text, DateTime
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        meta = MetaData()
        # define minimal table matching expected schema
        tbl = Table(
            "market_data_candles",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False, index=False),
            Column("timeframe", String(16), nullable=False),
            Column("ts_utc", BigInteger, nullable=False, index=False),
            Column("open", Float),
            Column("high", Float),
            Column("low", Float),
            Column("close", Float),
            Column("volume", Float),
            Column("resampled", Boolean, default=False),
        )
        # create table if not exists
        meta.create_all(engine)
        print("Schema ensured; table 'market_data_candles' created if missing in", db_path)
    except Exception as e:
        print("Failed to create schema:", e)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
