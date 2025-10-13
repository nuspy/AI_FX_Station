"""
Test Order Book (DOM) functionality with cTrader WebSocket

Verifies:
1. WebSocket connection to cTrader
2. Order book streaming and storage
3. Database market_depth table
4. Real-time bid/ask/spread data
"""
import asyncio
import time
from datetime import datetime
from loguru import logger
from sqlalchemy import create_engine, text

# Configure logging
logger.add("test_order_books.log", rotation="10 MB")


def check_database_setup():
    """Check if market_depth table exists and is accessible."""
    try:
        from src.forex_diffusion.utils.user_settings import get_setting

        # Get database URL
        db_url = get_setting('database_url', 'sqlite:///./data/forex_diffusion.db')
        engine = create_engine(db_url)

        # Check if market_depth table exists
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='market_depth'"
            ))
            table_exists = result.fetchone() is not None

            if table_exists:
                # Count existing records
                count_result = conn.execute(text("SELECT COUNT(*) FROM market_depth"))
                count = count_result.scalar()
                logger.success(f"✅ Table market_depth exists with {count} records")

                # Show sample data
                if count > 0:
                    sample = conn.execute(text(
                        "SELECT symbol, ts_utc, mid_price, spread, imbalance "
                        "FROM market_depth ORDER BY ts_utc DESC LIMIT 3"
                    ))
                    logger.info("Sample order book data:")
                    for row in sample:
                        logger.info(f"  {row.symbol}: mid={row.mid_price:.5f}, spread={row.spread:.5f}, imb={row.imbalance:.3f}")

                return True, engine
            else:
                logger.error("❌ Table market_depth does not exist")
                logger.info("Run migrations: alembic upgrade head")
                return False, None

    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False, None


def test_websocket_connection():
    """Test WebSocket connection and order book streaming."""
    try:
        from src.forex_diffusion.services.ctrader_websocket import CTraderWebSocketService
        from src.forex_diffusion.utils.user_settings import get_setting

        # Check if cTrader is configured
        client_id = get_setting('ctrader_client_id', '')
        client_secret = get_setting('ctrader_client_secret', '')
        access_token = get_setting('ctrader_access_token', '')
        account_id = get_setting('ctrader_account_id', None)

        if not all([client_id, client_secret, access_token, account_id]):
            logger.error("❌ cTrader credentials not configured in settings")
            logger.info("Configure in: Settings > Data Sources > cTrader")
            return False

        logger.info(f"✅ cTrader credentials found (account: {account_id})")

        # Get database
        db_url = get_setting('database_url', 'sqlite:///./data/forex_diffusion.db')
        engine = create_engine(db_url)

        # Create WebSocket service
        ws_service = CTraderWebSocketService(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            account_id=account_id,
            db_engine=engine,
            environment='demo',
            symbols=['EURUSD', 'GBPUSD']
        )

        # Track updates
        order_book_updates = []
        volume_updates = []
        sentiment_updates = []

        def on_order_book(data):
            order_book_updates.append(data)
            logger.info(
                f"📊 Order Book: {data['symbol']} - "
                f"Mid={data.get('mid_price', 0):.5f}, Spread={data.get('spread', 0):.5f}"
            )

        def on_volume(data):
            volume_updates.append(data)
            logger.debug(f"📈 Volume: {data['symbol']} - {data.get('tick_volume', 0)}")

        def on_sentiment(data):
            sentiment_updates.append(data)
            logger.info(
                f"💭 Sentiment: {data['symbol']} - {data['sentiment']} "
                f"(ratio={data['ratio']:.2f})"
            )

        # Set callbacks
        ws_service.on_order_book_update = on_order_book
        ws_service.on_volume_update = on_volume
        ws_service.on_sentiment_update = on_sentiment

        # Start service
        logger.info("Starting WebSocket service...")
        ws_service.start()

        # Wait for data (30 seconds)
        logger.info("Collecting order book data for 30 seconds...")
        start_time = time.time()

        while time.time() - start_time < 30:
            time.sleep(1)

            if len(order_book_updates) > 0 and (time.time() - start_time) % 5 < 1:
                logger.info(f"  Updates so far: OrderBooks={len(order_book_updates)}, "
                          f"Volumes={len(volume_updates)}, Sentiment={len(sentiment_updates)}")

        # Stop service
        logger.info("Stopping WebSocket service...")
        ws_service.stop()

        # Results
        logger.info("\n" + "=" * 80)
        logger.info("TEST RESULTS")
        logger.info("=" * 80)
        logger.success(f"✅ Order Book Updates: {len(order_book_updates)}")
        logger.success(f"✅ Volume Updates: {len(volume_updates)}")
        logger.success(f"✅ Sentiment Updates: {len(sentiment_updates)}")

        # Check latest order books
        for symbol in ['EURUSD', 'GBPUSD']:
            latest = ws_service.get_latest_order_book(symbol)
            if latest:
                logger.info(f"\nLatest Order Book for {symbol}:")
                logger.info(f"  Mid Price: {latest.get('mid_price', 0):.5f}")
                logger.info(f"  Spread: {latest.get('spread', 0):.5f}")
                logger.info(f"  Imbalance: {latest.get('imbalance', 0):.3f}")
                logger.info(f"  Bids: {len(latest.get('bids', []))} levels")
                logger.info(f"  Asks: {len(latest.get('asks', []))} levels")

        # Check database
        with engine.connect() as conn:
            recent_count = conn.execute(text(
                f"SELECT COUNT(*) FROM market_depth "
                f"WHERE ts_utc > {int((time.time() - 60) * 1000)}"
            ))
            db_count = recent_count.scalar()
            logger.success(f"\n✅ Database records (last minute): {db_count}")

        success = (
            len(order_book_updates) > 0 and
            len(volume_updates) > 0 and
            db_count > 0
        )

        if success:
            logger.success("\n✅✅✅ ORDER BOOKS TEST PASSED ✅✅✅")
        else:
            logger.error("\n❌ ORDER BOOKS TEST FAILED")
            logger.info("  Check: cTrader connection, WebSocket streaming, database writes")

        return success

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("Install required packages: pip install ctrader-open-api twisted")
        return False
    except Exception as e:
        logger.exception(f"❌ Test failed: {e}")
        return False


def main():
    """Run all order book tests."""
    logger.info("=" * 80)
    logger.info("TESTING ORDER BOOKS (DOM) FUNCTIONALITY")
    logger.info("=" * 80)

    # Test 1: Database setup
    logger.info("\n[TEST 1] Checking database setup...")
    db_ok, engine = check_database_setup()

    if not db_ok:
        logger.error("Database not ready. Fix issues and try again.")
        return

    # Test 2: WebSocket connection and streaming
    logger.info("\n[TEST 2] Testing WebSocket order book streaming...")
    ws_ok = test_websocket_connection()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Database Setup: {'✅ PASS' if db_ok else '❌ FAIL'}")
    logger.info(f"WebSocket Streaming: {'✅ PASS' if ws_ok else '❌ FAIL'}")

    if db_ok and ws_ok:
        logger.success("\n🎉 ALL ORDER BOOK TESTS PASSED 🎉")
    else:
        logger.error("\n⚠️  SOME TESTS FAILED - CHECK LOGS ABOVE")


if __name__ == "__main__":
    main()
