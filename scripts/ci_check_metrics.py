#!/usr/bin/env python3
"""
CI helper: run RegimeScheduler for a short period and verify metrics persisted in DB.

- Use DATABASE_URL env var (sqlite or postgres). Defaults to sqlite://./ci_metrics.db
- Runs alembic upgrade head, starts RegimeScheduler (interval_seconds=2), waits ~6s, stops, then queries metrics DB.
Exit code 0 on success, non-zero on failure.
"""
import os
import sys
import time
import subprocess
from sqlalchemy import create_engine
from loguru import logger

# ensure project root is importable
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forex_diffusion.services.scheduler import RegimeScheduler
from src.forex_diffusion.services.db_service import DBService

def run_alembic(db_url: str):
    env = os.environ.copy()
    env["DATABASE_URL"] = db_url
    print("Running alembic upgrade head with DATABASE_URL=", db_url)
    subprocess.run(["alembic", "upgrade", "head"], check=True, env=env, cwd=str(ROOT))

def main():
    db_url = os.environ.get("DATABASE_URL", f"sqlite:///{ROOT / 'ci_metrics.db'}")
    # apply migrations
    try:
        run_alembic(db_url)
    except Exception as e:
        print("Alembic upgrade failed:", e)
        raise

    engine = create_engine(db_url, future=True)
    dbs = DBService(engine=engine)

    # Remove any recent metrics to avoid false positives (best-effort)
    # Not all DBs support delete in the same way via DBService, so keep simple.

    scheduler = RegimeScheduler(engine=engine, interval_seconds=2, batch_size=1)
    scheduler.start()
    print("Scheduler started for CI test; waiting 6 seconds to allow one run...")
    time.sleep(6.0)
    scheduler.stop(timeout=5.0)
    print("Scheduler stopped; checking metrics in DB...")

    # Query metrics table for recent entries
    try:
        rows = dbs.query_metrics(name="regime_incremental_updated", since_ms=int((time.time() - 3600) * 1000), limit=10)
        print("Metrics rows found:", len(rows))
        if len(rows) == 0:
            print("CI check failed: no regime_incremental_updated metrics found.")
            sys.exit(2)
        else:
            print("CI check succeeded: metric sample:", rows[0])
            sys.exit(0)
    except Exception as e:
        print("Failed to query metrics:", e)
        sys.exit(3)

if __name__ == "__main__":
    main()
