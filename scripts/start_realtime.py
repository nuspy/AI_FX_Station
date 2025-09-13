#!/usr/bin/env python3
"""
Start a local RealTimeIngestionService for testing.

Usage (PowerShell):
  python .\\scripts\\start_realtime.py
Press Ctrl+C to stop.

This script starts DBWriter (if available), a MarketDataService and RealTimeIngestionService
which polls provider current price and upserts 1m candles + tick aggregates.
"""
from __future__ import annotations

import sys
import time
import signal

sys.path.insert(0, ".")
from loguru import logger

try:
    from src.forex_diffusion.services.db_service import DBService
    from src.forex_diffusion.services.marketdata import MarketDataService
    from src.forex_diffusion.services.realtime import RealTimeIngestionService
    from src.forex_diffusion.services.db_writer import DBWriter
except Exception as e:
    print("Failed to import project services:", e)
    raise

def main():
    print("Initializing DBService and MarketDataService (provider=tiingo)...")
    db = DBService()
    # explicitly use Tiingo as default provider for realtime helper
    msvc = MarketDataService(provider_name="tiingo")
    # start DBWriter if available
    db_writer = None
    try:
        db_writer = DBWriter(db_service=db)
        db_writer.start()
        print("DBWriter started.")
    except Exception as e:
        print("DBWriter not started (optional):", e)
        db_writer = None

    # Create RealTimeIngestionService (uses config for symbols by default)
    rt = RealTimeIngestionService(engine=db.engine, market_service=msvc, timeframe="1m", poll_interval=1.0, db_writer=db_writer)
    try:
        rt.start()
        print("RealTimeIngestionService started. Poll interval:", rt.poll_interval, "s")
    except Exception as e:
        print("Failed to start RealTimeIngestionService:", e)
        if db_writer:
            db_writer.stop()
        return

    # graceful stop on Ctrl+C
    def _sigint(sig, frame):
        print("Stopping services...")
        try:
            rt.stop()
        except Exception:
            pass
        try:
            if db_writer:
                db_writer.stop()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    print("Running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _sigint(None, None)

if __name__ == "__main__":
    main()
