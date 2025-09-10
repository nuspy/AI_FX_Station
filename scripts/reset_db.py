#!/usr/bin/env python3
"""
scripts/reset_db.py

Delete the local SQLite DB file (as configured) and recreate schema via DBService.
Optional: run alembic upgrade head after recreation.

Usage:
  # with .venv activated
  python scripts/reset_db.py        # delete sqlite file and recreate tables
  python scripts/reset_db.py --alembic  # also run alembic upgrade head
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
import argparse

# ensure project src importable
ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def extract_sqlite_path(db_url: str) -> Path | None:
    if not db_url:
        return None
    db_url = str(db_url)
    # handle sqlite:///relative/path and sqlite:////absolute/path
    if db_url.startswith("sqlite:///"):
        path = db_url.split("sqlite:///")[-1]
        return Path(path).resolve()
    if db_url.startswith("sqlite://"):
        # fallback
        path = db_url.split("sqlite://")[-1]
        return Path(path).resolve()
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alembic", action="store_true", help="Run alembic -c alembic.ini upgrade head after recreate")
    args = p.parse_args()

    # Load config to find DB url; fallback to env DATABASE_URL
    try:
        from forex_diffusion.utils.config import get_config
        cfg = get_config()
        db_url = getattr(cfg, "db", None)
        if isinstance(db_url, dict):
            db_url = db_url.get("database_url")
        elif hasattr(db_url, "database_url"):
            db_url = getattr(db_url, "database_url", None)
        # if still None, check env
        if not db_url:
            db_url = os.environ.get("DATABASE_URL") or os.environ.get("DB_URL") or os.environ.get("DB")
    except Exception:
        db_url = os.environ.get("DATABASE_URL") or os.environ.get("DB_URL") or os.environ.get("DB")

    sqlite_path = extract_sqlite_path(db_url) if db_url else None

    if sqlite_path is None:
        print("No sqlite DB URL detected (not in config or env). Aborting.")
        print("Detected DB URL:", db_url)
        raise SystemExit(1)

    print("Detected sqlite DB file:", sqlite_path)

    # Stop if running? We simply delete file if user confirms
    try:
        if sqlite_path.exists():
            print("Removing existing DB file:", sqlite_path)
            sqlite_path.unlink()
        else:
            print("DB file does not exist, will create new one.")
        # Ensure parent dir exists
        parent = sqlite_path.parent
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print("Failed to remove/create DB file:", e)
        raise SystemExit(2)

    # Recreate tables by instantiating DBService (which calls _ensure_tables)
    try:
        from forex_diffusion.services.db_service import DBService
        DBService()  # creates engine and tables
        print("DBService invoked; tables should be created.")
    except Exception as e:
        print("Failed to instantiate DBService and create tables:", e)
        raise SystemExit(3)

    # Optionally run alembic upgrade head
    if args.alembic:
        try:
            print("Running alembic upgrade head...")
            subprocess.check_call(["alembic", "-c", "alembic.ini", "upgrade", "head"])
            print("Alembic upgrade head completed.")
        except subprocess.CalledProcessError as e:
            print("Alembic upgrade failed:", e)
            raise SystemExit(4)
        except Exception as e:
            print("Failed to run alembic:", e)
            raise SystemExit(4)

    print("Database reset complete.")

if __name__ == "__main__":
    main()
