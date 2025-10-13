"""
Test cTrader historical data availability for specific dates
"""
import asyncio
from datetime import datetime, timezone
from loguru import logger

async def test_date_range(symbol: str, start_date: str, end_date: str):
    """Test if data is available for a specific date range."""
    try:
        from src.forex_diffusion.providers.ctrader_provider import CTraderProvider
        from src.forex_diffusion.utils.user_settings import get_setting

        # Get config from settings
        config = {
            'client_id': get_setting('ctrader_client_id', ''),
            'client_secret': get_setting('ctrader_client_secret', ''),
            'access_token': get_setting('ctrader_access_token', ''),
            'account_id': get_setting('ctrader_account_id', None),
            'environment': get_setting('ctrader_environment', 'demo'),
        }

        # Create provider
        provider = CTraderProvider(config=config)

        # Connect
        logger.info(f"Connecting to cTrader...")
        success = await provider.connect()
        if not success:
            logger.error("Failed to connect to cTrader")
            return None

        logger.info(f"Connected successfully")

        # Convert dates to timestamps
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Request 1-minute bars
        logger.info(f"Requesting data for {symbol} from {start_date} to {end_date}")
        logger.info(f"Timestamps: {start_ms} to {end_ms}")

        df = await provider.get_historical_bars(symbol, "1m", start_ms, end_ms)

        if df is not None and not df.empty:
            logger.success(f"✅ Data available: {len(df)} bars")
            logger.info(f"First timestamp: {df['ts_utc'].iloc[0]}")
            logger.info(f"Last timestamp: {df['ts_utc'].iloc[-1]}")
            logger.info(f"First 3 rows:\n{df.head(3)}")
        else:
            logger.warning(f"❌ No data returned for {start_date} to {end_date}")

        # Disconnect
        await provider.disconnect()
        return df

    except Exception as e:
        logger.error(f"Error testing date range: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    symbol = "EUR/USD"

    logger.info("=" * 80)
    logger.info("TEST 1: 9 ottobre 2025 (4 giorni fa)")
    logger.info("=" * 80)
    await test_date_range(symbol, "2025-10-09", "2025-10-10")

    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: 15 maggio 2025 (~5 mesi fa)")
    logger.info("=" * 80)
    await test_date_range(symbol, "2025-05-15", "2025-05-16")

if __name__ == "__main__":
    asyncio.run(main())
