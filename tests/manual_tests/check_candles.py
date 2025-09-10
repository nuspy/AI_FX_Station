#!/usr/bin/env python3
"""
tests/manual_tests/check_candles.py

Check candles present in DB for a given symbol/timeframe and print count, min/max ts and recent samples.

This script uses DBService and the same 'market_data_candles' table as the UI HistoryTab,
and normalizes common symbol forms (EUR/USD <-> EURUSD) to avoid mismatches.

It also prints the DB URL and the exact SQL used by the GUI to enable direct comparison.

Usage (PowerShell):
  python tests/manual_tests/check_candles.py --symbol "EUR/USD" --timeframe "1m"
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import datetime
import inspect

ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def _normalize_symbol(s: str) -> list[str]:
    """
    Return possible normalized variants to query the DB.
    E.g., 'EUR/USD' -> ['EUR/USD','EURUSD']; 'EURUSD' -> ['EURUSD','EUR/USD'].
    """
    s = (s or "").strip()
    if "/" in s:
        alt = s.replace("/", "")
        return [s, alt]
    elif len(s) == 6 and s.isalpha():
        alt = f"{s[0:3]}/{s[3:6]}"
        return [s, alt]
    else:
        return [s]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="EUR/USD")
    p.add_argument("--timeframe", default="1m")
    args = p.parse_args()

    try:
        from forex_diffusion.services.db_service import DBService
        from sqlalchemy import select, text
        from forex_diffusion.ui.history_tab import HistoryTab
    except Exception as e:
        print("Import failed:", e)
        raise SystemExit(1)

    # Print DB URL used by DBService (so you can compare with UI)
    try:
        dbs = DBService()
        cfg = None
        try:
            from forex_diffusion.utils.config import get_config
            cfg = get_config()
        except Exception:
            cfg = None
        db_url = None
        if cfg is not None:
            try:
                db_url = getattr(cfg.db, "database_url", None) if hasattr(cfg, "db") else (cfg.get("db", {}).get("database_url") if isinstance(cfg, dict) else None)
            except Exception:
                db_url = None
        if not db_url:
            import os
            db_url = os.environ.get("DATABASE_URL") or os.environ.get("DB") or None
        print("DB URL (used by DBService):", db_url)
    except Exception as e:
        print("Failed to instantiate DBService to read DB URL:", e)
        raise SystemExit(1)

    eng = dbs.engine
    sym_variants = _normalize_symbol(args.symbol)

    # Print the exact SQL used by the GUI's HistoryTab.refresh (for comparison)
    try:
        src = inspect.getsource(HistoryTab.refresh)
        print("\n--- GUI HistoryTab.refresh source (snippet) ---\n")
        print("\n".join(src.splitlines()[:40]))
        print("\n--- end snippet ---\n")
    except Exception:
        pass

    try:
        with eng.connect() as conn:
            # Use same WHERE clause as GUI: symbol == exact text and timeframe == exact text
            # But also show variants count to detect stored formats
            # Exact GUI query:
            stmt_gui = text("SELECT * FROM market_data_candles WHERE symbol = :sym AND timeframe = :tf ORDER BY ts_utc DESC LIMIT :lim")
            # Count using GUI exact symbol
            exact_cnt = conn.execute(text("SELECT COUNT(*) FROM market_data_candles WHERE symbol = :sym AND timeframe = :tf"), {"sym": args.symbol, "tf": args.timeframe}).scalar() or 0
            # Count using normalized variants (in)
            placeholders = ", ".join([f":s{i}" for i in range(len(sym_variants))])
            params = {f"s{i}": sym_variants[i] for i in range(len(sym_variants))}
            params["tf"] = args.timeframe
            stmt_count_variants = text(f"SELECT COUNT(*) FROM market_data_candles WHERE symbol IN ({placeholders}) AND timeframe=:tf")
            cnt_variants = conn.execute(stmt_count_variants, params).scalar() or 0

            print(f"Requested symbol='{args.symbol}' timeframe='{args.timeframe}'")
            print(f"Count exact match (symbol='{args.symbol}'): {exact_cnt}")
            print(f"Count variants match {sym_variants}: {cnt_variants}")

            # Show sample rows using exact GUI query (same ordering as GUI)
            rows_gui = conn.execute(stmt_gui.bindparams(sym=args.symbol, tf=args.timeframe, lim=200)).fetchall()
            print(f"\nRows returned by exact GUI query (limit 200): {len(rows_gui)}")
            for rr in rows_gui[:10]:
                # attempt to map row to dict similarly to HistoryTab._row_to_dict
                try:
                    mapping = getattr(rr, "_mapping", None)
                    if mapping is not None:
                        rec = dict(mapping)
                    else:
                        # tuple fallback: id,symbol,timeframe,ts_utc,open,high,low,close,volume,resampled
                        rec = {
                            "id": rr[0],
                            "symbol": rr[1],
                            "timeframe": rr[2],
                            "ts_utc": rr[3],
                            "open": rr[4],
                            "high": rr[5],
                            "low": rr[6] if len(rr) > 6 else None,
                            "close": rr[7] if len(rr) > 7 else None,
                            "volume": rr[8] if len(rr) > 8 else None,
                            "resampled": rr[9] if len(rr) > 9 else False,
                        }
                except Exception:
                    rec = {"raw": str(rr)}
                t = rec.get("ts_utc")
                try:
                    dt = datetime.datetime.fromtimestamp(int(t)/1000, tz=datetime.timezone.utc).astimezone()
                    ts_str = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    ts_str = str(t)
                print(f"  id={rec.get('id')} sym={rec.get('symbol')} tf={rec.get('timeframe')} ts={rec.get('ts_utc')} ({ts_str}) open={rec.get('open')} close={rec.get('close')}")

            # Also show rows for normalized variants
            rows_var = conn.execute(text(f"SELECT id,symbol,timeframe,ts_utc,open,high,low,close,volume,resampled FROM market_data_candles WHERE symbol IN ({placeholders}) AND timeframe=:tf ORDER BY ts_utc DESC LIMIT 200"), params).fetchall()
            print(f"\nRows returned by variants query (limit 200): {len(rows_var)}")
            for rr in rows_var[:10]:
                try:
                    mapping = getattr(rr, "_mapping", None)
                    if mapping is not None:
                        rec = dict(mapping)
                    else:
                        rec = {
                            "id": rr[0],
                            "symbol": rr[1],
                            "timeframe": rr[2],
                            "ts_utc": rr[3],
                            "open": rr[4],
                            "high": rr[5],
                            "low": rr[6] if len(rr) > 6 else None,
                            "close": rr[7] if len(rr) > 7 else None,
                            "volume": rr[8] if len(rr) > 8 else None,
                            "resampled": rr[9] if len(rr) > 9 else False,
                        }
                except Exception:
                    rec = {"raw": str(rr)}
                t = rec.get("ts_utc")
                try:
                    dt = datetime.datetime.fromtimestamp(int(t)/1000, tz=datetime.timezone.utc).astimezone()
                    ts_str = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    ts_str = str(t)
                print(f"  id={rec.get('id')} sym={rec.get('symbol')} tf={rec.get('timeframe')} ts={rec.get('ts_utc')} ({ts_str}) open={rec.get('open')} close={rec.get('close')}")

    except Exception as e:
        print("DB query failed:", e)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
