#!/usr/bin/env python3
"""
Check AlphaVantage connectivity and sample data retrieval.

Usage (PowerShell):
  # use config symbols if present:
  python .\scripts\check_alphavantage.py

  # single symbol/timeframe:
  python .\scripts\check_alphavantage.py --symbol "EUR/USD" --timeframe "1m"

Interpretation:
 - If 'api_key' is missing -> set ALPHAVANTAGE_KEY env var or config/providers.alpha_vantage.key
 - If get_current_price returns empty or raises -> network/key/rate-limit issue
 - If get_historical returns empty -> provider returned no data for the timeframe/window
 - Look for "Note" or error messages indicating rate-limit in output
"""
from __future__ import annotations

import sys
import json
import argparse
from typing import Optional

# ensure project src is importable when run from repo root
sys.path.insert(0, ".")

from forex_diffusion.services.marketdata import AlphaVantageClient, MarketDataService
from forex_diffusion.utils.config import get_config
from loguru import logger

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", help='Symbol like "EUR/USD"', default=None)
    p.add_argument("--timeframe", help='timeframe like 1m or 1d', default="1m")
    p.add_argument("--history_seconds", type=int, default=3600, help="history window (seconds) to request for intraday")
    p.add_argument("--do_backfill", action="store_true", help="OPTIONAL: run backfill_symbol_timeframe (will write to DB)")
    return p.parse_args()

def pretty_print(title: str, obj):
    print(f"--- {title} ---")
    try:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(repr(obj))
    print("")

def main():
    args = parse_args()
    cfg = get_config()
    # determine symbols list
    cfg_symbols = []
    try:
        cfg_symbols = getattr(cfg.data, "symbols", None) or (cfg.data.get("symbols", []) if isinstance(cfg.data, dict) else [])
    except Exception:
        cfg_symbols = []
    symbols = [args.symbol] if args.symbol else (cfg_symbols or ["EUR/USD"])

    # instantiate client and service
    client = AlphaVantageClient()
    msvc = None
    try:
        msvc = MarketDataService()
    except Exception:
        msvc = None

    # report API key presence
    api_key_info = {
        "api_key_present": bool(getattr(client, "api_key", None)),
        "api_key_value_masked": (("****" + str(getattr(client, "api_key", ""))[-4:]) if getattr(client, "api_key", None) else None),
        "base_url": getattr(client, "base_url", None),
        "rate_limit_per_minute": getattr(client, "rate_limit", None),
        "_HAS_ALPHA_VANTAGE": None
    }
    try:
        # try to introspect internal flag set in module import
        api_key_info["_HAS_ALPHA_VANTAGE"] = "__" not in getattr(client, "_fx", None).__class__.__name__ if getattr(client, "_fx", None) is not None else False
    except Exception:
        api_key_info["_HAS_ALPHA_VANTAGE"] = False

    pretty_print("AlphaVantage client info", api_key_info)

    for sym in symbols:
        print(f"Checking symbol: {sym} timeframe: {args.timeframe}")
        # current price
        try:
            cp = client.get_current_price(sym)
            pretty_print("current_price_raw", cp or {})
        except Exception as e:
            pretty_print("current_price_error", {"error": str(e)})
        # historical window: request end = now, start = now - history_seconds
        try:
            import time
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - int(args.history_seconds) * 1000
            hist = client.get_historical(symbol=sym, timeframe=args.timeframe, start_ts_ms=start_ms, end_ts_ms=now_ms)
            if hist is None:
                pretty_print("historical_result", {"note": "None returned"})
            else:
                try:
                    # convert to minimal summary
                    rows = []
                    # pandas DataFrame case
                    if hasattr(hist, "shape"):
                        rows = hist.tail(5).to_dict(orient="records") if not hist.empty else []
                        pretty_print("historical_tail", {"rows": rows, "rows_count": int(hist.shape[0]) if hasattr(hist, "shape") else None})
                    else:
                        pretty_print("historical_raw", hist)
                except Exception as e:
                    pretty_print("historical_parse_error", {"error": str(e)})
        except Exception as e:
            pretty_print("historical_error", {"error": str(e)})

        # optional: backfill (writes to DB)
        if args.do_backfill and msvc is not None:
            try:
                print("Running backfill_symbol_timeframe (may write to DB)...")
                res = msvc.backfill_symbol_timeframe(sym, args.timeframe, force_full=False)
                pretty_print("backfill_report", res)
            except Exception as e:
                pretty_print("backfill_error", {"error": str(e)})
        print("")

if __name__ == "__main__":
    main()
