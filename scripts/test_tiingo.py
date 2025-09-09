#!/usr/bin/env python3
"""
Quick test for Tiingo provider connectivity.

Usage (PowerShell):
  python .\\scripts\\test_tiingo.py

Will print:
 - MarketDataService.provider_name
 - get_current_price() raw result
 - get_historical() tail (if any)
"""
import sys
import json
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from forex_diffusion.services.marketdata import MarketDataService
except ImportError:
    # Try alternative import path for local development
    from src.forex_diffusion.services.marketdata import MarketDataService
from loguru import logger
import time

def main():
    svc = MarketDataService(provider_name="tiingo")
    print("Provider:", svc.provider_name())
    sym = "EUR/USD"
    print("Testing get_current_price for", sym)
    try:
        cur = svc.provider.get_current_price(sym)
        print("get_current_price raw:")
        print(json.dumps(cur, indent=2, ensure_ascii=False))
    except Exception as e:
        print("get_current_price exception:", e)

    print("\nTesting get_historical recent (1d) for", sym)
    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - 24 * 3600 * 1000
        df = svc.provider.get_historical(symbol=sym, timeframe="1d", start_ts_ms=start_ms, end_ts_ms=now_ms)
        try:
            print("historical rows:", len(df))
            if hasattr(df, "tail"):
                print(df.tail(3).to_dict(orient="records"))
            else:
                print(df)
        except Exception:
            print("historic raw:", df)
    except Exception as e:
        print("get_historical exception:", e)

if __name__ == "__main__":
    main()
