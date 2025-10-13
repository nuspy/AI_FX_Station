"""
Test cTrader with different date ranges to find what data is actually available
"""
import asyncio
from datetime import datetime, timezone, timedelta
from loguru import logger

async def test_date_range(symbol: str, start_date: str, end_date: str, description: str):
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

        # Convert dates to timestamps
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Request 1-minute bars
        logger.info(f"{description}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Timestamps: {start_ms} to {end_ms}")

        df = await provider.get_historical_bars(symbol, "1m", start_ms, end_ms)

        if df is not None and not df.empty:
            logger.success(f"✅ Data available: {len(df)} bars")
            logger.info(f"First timestamp: {df['ts_utc'].iloc[0]}")
            logger.info(f"Last timestamp: {df['ts_utc'].iloc[-1]}")
            logger.info(f"First 3 rows:\n{df.head(3)}")
        else:
            logger.warning(f"❌ No data returned")

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

    # Today's date (Oct 13, 2025)
    today = datetime(2025, 10, 13, tzinfo=timezone.utc)

    tests = [
        # Last 7 days
        (
            (today - timedelta(days=7)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
            "TEST 1: Ultimi 7 giorni (6-13 ottobre)"
        ),
        # Last 30 days
        (
            (today - timedelta(days=30)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
            "TEST 2: Ultimi 30 giorni (13 settembre - 13 ottobre)"
        ),
        # Last 60 days
        (
            (today - timedelta(days=60)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
            "TEST 3: Ultimi 60 giorni (14 agosto - 13 ottobre)"
        ),
        # Last 90 days
        (
            (today - timedelta(days=90)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
            "TEST 4: Ultimi 90 giorni (15 luglio - 13 ottobre)"
        ),
    ]

    for start_date, end_date, description in tests:
        logger.info("\n" + "=" * 80)
        logger.info(description)
        logger.info("=" * 80)
        result = await test_date_range(symbol, start_date, end_date, description)

        # Wait a bit between requests
        await asyncio.sleep(2)

        # If we got data, no need to continue
        if result is not None and not result.empty:
            logger.success(f"\n🎯 TROVATO! I dati sono disponibili per questo range.")
            break
        else:
            logger.warning(f"\n⚠️  Nessun dato per questo range, provo range più lungo...")

if __name__ == "__main__":
    asyncio.run(main())
