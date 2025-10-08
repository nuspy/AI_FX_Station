#!/usr/bin/env python
"""
Test Tiingo API timeframes to verify correct resampleFreq values.
Downloads 1 day of data for each timeframe without saving to DB.
"""

from forex_diffusion.services.marketdata import TiingoClient
from datetime import datetime, timezone, timedelta
from loguru import logger

def test_timeframes():
    """Test all timeframes we want to use"""

    # Initialize Tiingo client (will read API key from config/env)
    client = TiingoClient()

    # Symbol to test
    symbol = "EUR/USD"

    # Timeframes to test with their Tiingo resampleFreq values
    timeframes_to_test = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "4h": "4hour",
        "1d": "1day"
    }

    # Date range: yesterday to today (1 day)
    end_date = datetime.now(tz=timezone.utc).date().isoformat()
    start_date = (datetime.now(tz=timezone.utc) - timedelta(days=1)).date().isoformat()

    logger.info(f"Testing Tiingo API for {symbol}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("-" * 80)

    results = {}

    for tf_label, resample_freq in timeframes_to_test.items():
        logger.info(f"\nTesting timeframe: {tf_label} (resampleFreq={resample_freq})")

        try:
            df = client.get_candles(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                resample_freq=resample_freq
            )

            if df is not None and not df.empty:
                row_count = len(df)
                first_ts = df.iloc[0]["ts_utc"] if "ts_utc" in df.columns else None
                last_ts = df.iloc[-1]["ts_utc"] if "ts_utc" in df.columns else None

                # Convert timestamps to readable format
                if first_ts:
                    first_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
                    last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
                    logger.info(f"  ✓ SUCCESS: {row_count} candles")
                    logger.info(f"    First: {first_dt.isoformat()}")
                    logger.info(f"    Last:  {last_dt.isoformat()}")
                    results[tf_label] = {"status": "SUCCESS", "rows": row_count}
                else:
                    logger.warning(f"  ⚠ Data received but no ts_utc column")
                    results[tf_label] = {"status": "WARNING", "rows": row_count}
            else:
                logger.warning(f"  ✗ EMPTY: Tiingo returned empty data")
                results[tf_label] = {"status": "EMPTY", "rows": 0}

        except Exception as e:
            logger.error(f"  ✗ ERROR: {e}")
            results[tf_label] = {"status": "ERROR", "error": str(e)}

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    success_count = sum(1 for r in results.values() if r.get("status") == "SUCCESS")

    for tf_label, result in results.items():
        status = result.get("status", "UNKNOWN")
        if status == "SUCCESS":
            logger.info(f"  ✓ {tf_label:4s} -> {status:10s} ({result.get('rows', 0)} rows)")
        elif status == "EMPTY":
            logger.warning(f"  ⚠ {tf_label:4s} -> {status:10s}")
        else:
            logger.error(f"  ✗ {tf_label:4s} -> {status:10s} ({result.get('error', 'Unknown error')})")

    logger.info("-" * 80)
    logger.info(f"Total: {success_count}/{len(timeframes_to_test)} timeframes working")

    if success_count == len(timeframes_to_test):
        logger.info("✓ All timeframes are valid and working!")
    else:
        logger.warning(f"⚠ {len(timeframes_to_test) - success_count} timeframe(s) failed - may need different resampleFreq values")

if __name__ == "__main__":
    test_timeframes()
