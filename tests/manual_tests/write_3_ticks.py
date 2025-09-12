import os
import random
import sys
import time

# Add src to path to allow imports from the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from loguru import logger
from sqlalchemy import text

from forex_diffusion.services.db_service import DBService


def main():
    """
    Tests the DBService.write_tick method by writing and verifying 3 random ticks.
    """
    logger.info("Starting tick writing test...")
    try:
        db_service = DBService()
        logger.info("DBService instantiated successfully.")
    except Exception as e:
        logger.error(f"Failed to instantiate DBService: {e}")
        return

    ticks_to_write = []
    base_ts = int(time.time() * 1000)
    base_price = 1.08500

    for i in range(3):
        tick = {
            "symbol": "EUR/USD",
            "timeframe": "tick",  # Represents raw tick data
            "ts_utc": base_ts + i * 1000,  # 1 second apart
            "price": base_price + (random.random() - 0.5) * 0.0001,
            "bid": base_price + (random.random() - 0.5) * 0.0002,
            "ask": base_price + (random.random() - 0.5) * 0.0002 + 0.00005,
            "volume": None,
            "ts_created_ms": int(time.time() * 1000),
        }
        ticks_to_write.append(tick)

    logger.info(f"Attempting to write {len(ticks_to_write)} ticks...")
    success_count = 0
    for tick in ticks_to_write:
        if db_service.write_tick(tick):
            success_count += 1

    if success_count == len(ticks_to_write):
        logger.info("All ticks were sent to write_tick successfully.")
    else:
        logger.warning(
            f"{len(ticks_to_write) - success_count} ticks may have failed to write."
        )

    # --- Verification Step ---
    logger.info("Verifying written ticks in the database...")
    try:
        with db_service.engine.connect() as conn:
            stmt = text(
                "SELECT * FROM market_data_ticks WHERE ts_utc >= :start_ts ORDER BY ts_utc DESC LIMIT 5"
            )
            results = conn.execute(stmt, {"start_ts": base_ts}).fetchall()

            logger.info(f"Found {len(results)} ticks in the database for this test run.")
            if len(results) >= len(ticks_to_write):
                logger.success("Verification successful! Ticks are being persisted.")
                for row in results:
                    logger.debug(f"  - DB Row: {dict(row._mapping)}")
            else:
                logger.error(
                    "Verification FAILED! Number of ticks in DB does not match number written."
                )

    except Exception as e:
        logger.error(f"Verification failed with an exception: {e}")


if __name__ == "__main__":
    main()