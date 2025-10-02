#!/usr/bin/env python
"""
Test if Tiingo API automatically excludes weekends when requesting multi-day ranges.
This would simplify our backfill logic significantly.
"""

from forex_diffusion.services.marketdata import TiingoClient
from datetime import datetime, timezone, timedelta
from loguru import logger

def test_weekend_exclusion():
    """Test if requesting Fri-Mon range excludes Sat-Sun automatically"""

    client = TiingoClient()
    symbol = "EUR/USD"

    # Request range that includes a full weekend: Friday -> Monday
    # Example: 2025-09-26 (Fri) to 2025-09-29 (Mon) - includes Sat 27, Sun 28
    start_date = "2025-09-26"  # Friday
    end_date = "2025-09-29"    # Monday

    logger.info(f"Testing weekend exclusion: {start_date} to {end_date}")
    logger.info("This range includes Saturday 27 and Sunday 28")
    logger.info("-" * 80)

    # Test with 1min data
    df = client.get_candles(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        resample_freq="1min"
    )

    if df is None or df.empty:
        logger.error("No data returned!")
        return False

    # Convert timestamps to datetime
    df["datetime"] = df["ts_utc"].apply(lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc))

    # Check for weekend data
    weekend_data = df[df["datetime"].dt.dayofweek >= 5]  # Saturday=5, Sunday=6

    # Analyze results
    logger.info(f"Total candles received: {len(df)}")
    logger.info(f"First candle: {df.iloc[0]['datetime'].isoformat()}")
    logger.info(f"Last candle:  {df.iloc[-1]['datetime'].isoformat()}")
    logger.info("")
    logger.info(f"Weekend candles (Sat/Sun): {len(weekend_data)}")

    if len(weekend_data) > 0:
        logger.warning("❌ Provider INCLUDES weekend data - we need to filter manually")
        logger.info("Sample weekend timestamps:")
        for idx in weekend_data.head(5).index:
            dt = weekend_data.loc[idx, "datetime"]
            logger.info(f"  {dt.isoformat()} ({dt.strftime('%A')})")
        return False
    else:
        logger.info("✓ Provider EXCLUDES weekend data automatically!")
        logger.info("We can simplify backfill logic by removing weekend splitting")
        return True

if __name__ == "__main__":
    result = test_weekend_exclusion()

    if result:
        logger.info("\n" + "=" * 80)
        logger.info("RESULT: Provider handles weekends automatically ✓")
        logger.info("RECOMMENDATION: Simplify backfill by removing split_range_avoid_weekend()")
        logger.info("=" * 80)
    else:
        logger.info("\n" + "=" * 80)
        logger.info("RESULT: Provider includes weekends ✗")
        logger.info("RECOMMENDATION: Keep current weekend filtering logic")
        logger.info("=" * 80)
