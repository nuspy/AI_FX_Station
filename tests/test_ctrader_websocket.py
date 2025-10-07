"""
Test script for cTrader WebSocket integration.

IMPORTANT: This script requires cTrader OAuth credentials.
Set environment variables before running:

On Windows (PowerShell):
$env:CTRADER_CLIENT_ID = "your_client_id"
$env:CTRADER_CLIENT_SECRET = "your_client_secret"
$env:CTRADER_ACCESS_TOKEN = "your_access_token"
$env:CTRADER_ACCOUNT_ID = "your_account_id"

Or create a .env file with:
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_client_secret
CTRADER_ACCESS_TOKEN=your_access_token
CTRADER_ACCOUNT_ID=your_account_id
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from sqlalchemy import create_engine

# Try to load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded .env file")
except ImportError:
    logger.warning("python-dotenv not installed, using environment variables only")

from forex_diffusion.services.ctrader_websocket import CTraderWebSocketService


def get_credentials():
    """Get credentials from environment variables."""
    client_id = os.getenv('CTRADER_CLIENT_ID')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET')
    access_token = os.getenv('CTRADER_ACCESS_TOKEN')
    account_id = os.getenv('CTRADER_ACCOUNT_ID')

    if not all([client_id, client_secret, access_token, account_id]):
        logger.error("Missing cTrader credentials in environment variables")
        logger.info("Required: CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID")
        return None

    try:
        account_id = int(account_id)
    except ValueError:
        logger.error(f"Invalid CTRADER_ACCOUNT_ID: {account_id} (must be integer)")
        return None

    return {
        'client_id': client_id,
        'client_secret': client_secret,
        'access_token': access_token,
        'account_id': account_id
    }


async def test_websocket():
    """Test cTrader WebSocket connection and data streaming."""

    # Get credentials
    creds = get_credentials()
    if not creds:
        logger.error("Cannot proceed without credentials")
        return

    logger.info("=" * 60)
    logger.info("cTrader WebSocket Test")
    logger.info("=" * 60)

    # Create in-memory database for testing
    db_engine = create_engine("sqlite:///:memory:", echo=False)

    # Create market_depth table
    with db_engine.begin() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_depth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20),
                ts_utc INTEGER,
                bids TEXT,
                asks TEXT,
                mid_price FLOAT,
                spread FLOAT,
                imbalance FLOAT,
                provider VARCHAR(20)
            )
        """)

    logger.info("✓ Created test database")

    # Tracking variables
    order_book_updates = []
    volume_updates = []
    sentiment_updates = []

    # Callbacks
    def on_order_book(data):
        order_book_updates.append(data)
        logger.info(f"📊 Order Book: {data['symbol']} - Spread: {data.get('spread', 0):.5f}")

    def on_volume(data):
        volume_updates.append(data)
        logger.info(f"📈 Volume: {data['symbol']} - Tick Volume: {data.get('tick_volume', 0)}")

    def on_sentiment(data):
        sentiment_updates.append(data)
        logger.info(
            f"💭 Sentiment: {data['symbol']} - {data['sentiment'].upper()} "
            f"(ratio={data['ratio']:.2f}, confidence={data['confidence']:.2f})"
        )

    # Create WebSocket service
    logger.info(f"Connecting to cTrader demo account {creds['account_id']}...")

    ws_service = CTraderWebSocketService(
        client_id=creds['client_id'],
        client_secret=creds['client_secret'],
        access_token=creds['access_token'],
        account_id=creds['account_id'],
        db_engine=db_engine,
        environment='demo',
        symbols=['EURUSD', 'GBPUSD', 'USDJPY']
    )

    # Set callbacks
    ws_service.on_order_book_update = on_order_book
    ws_service.on_volume_update = on_volume
    ws_service.on_sentiment_update = on_sentiment

    # Start service
    ws_service.start()

    logger.info("⏳ Waiting for connection and authentication...")
    await asyncio.sleep(5)

    if not ws_service.connected:
        logger.error("❌ Failed to connect to cTrader")
        ws_service.stop()
        return

    logger.info("✓ Connected to cTrader")

    if not ws_service.authenticated:
        logger.warning("⚠️ Not authenticated yet, waiting...")
        await asyncio.sleep(5)

    if ws_service.authenticated:
        logger.info("✓ Authenticated successfully")
    else:
        logger.error("❌ Authentication failed")
        ws_service.stop()
        return

    # Stream data for 30 seconds
    logger.info("\n" + "=" * 60)
    logger.info("Streaming data for 30 seconds...")
    logger.info("=" * 60 + "\n")

    start_time = time.time()
    last_report_time = start_time

    while time.time() - start_time < 30:
        await asyncio.sleep(1)

        # Report every 5 seconds
        if time.time() - last_report_time >= 5:
            logger.info(f"\n--- Status Report ({int(time.time() - start_time)}s elapsed) ---")
            logger.info(f"  Order Book Updates: {len(order_book_updates)}")
            logger.info(f"  Volume Updates: {len(volume_updates)}")
            logger.info(f"  Sentiment Updates: {len(sentiment_updates)}")
            logger.info(f"  Subscribed Symbols: {list(ws_service.subscribed_symbols.keys())}")

            # Show latest data
            for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
                ob = ws_service.get_latest_order_book(symbol)
                if ob:
                    logger.info(
                        f"  {symbol}: "
                        f"Mid={ob.get('mid_price', 0):.5f}, "
                        f"Spread={ob.get('spread', 0):.5f}, "
                        f"Imbalance={ob.get('imbalance', 0):.2f}"
                    )

                sent = ws_service.get_latest_sentiment(symbol)
                if sent:
                    logger.info(
                        f"    Sentiment: {sent['sentiment']} "
                        f"(Buy: {sent['buy_volume']}, Sell: {sent['sell_volume']})"
                    )

            logger.info("-" * 40 + "\n")
            last_report_time = time.time()

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Complete - Final Summary")
    logger.info("=" * 60)
    logger.info(f"✓ Total Order Book Updates: {len(order_book_updates)}")
    logger.info(f"✓ Total Volume Updates: {len(volume_updates)}")
    logger.info(f"✓ Total Sentiment Updates: {len(sentiment_updates)}")

    # Check database
    with db_engine.connect() as conn:
        result = conn.execute("SELECT COUNT(*) FROM market_depth").fetchone()
        db_count = result[0]
        logger.info(f"✓ Database Records: {db_count}")

        if db_count > 0:
            # Show sample records
            sample = conn.execute(
                "SELECT symbol, ts_utc, mid_price, spread, imbalance "
                "FROM market_depth ORDER BY ts_utc DESC LIMIT 5"
            ).fetchall()

            logger.info("\nSample Records:")
            for row in sample:
                symbol, ts, mid, spread, imb = row
                dt = time.strftime('%H:%M:%S', time.localtime(ts / 1000))
                logger.info(
                    f"  {dt} - {symbol}: Mid={mid:.5f}, Spread={spread:.5f}, Imbalance={imb:.2f}"
                )

    # Performance metrics
    if order_book_updates:
        duration = time.time() - start_time
        rate = len(order_book_updates) / duration
        logger.info(f"\n✓ Average Update Rate: {rate:.2f} updates/second")

    # Stop service
    logger.info("\nStopping WebSocket service...")
    ws_service.stop()
    await asyncio.sleep(2)

    logger.info("✓ Test completed successfully")

    # Return results for assertions if needed
    return {
        'order_book_updates': len(order_book_updates),
        'volume_updates': len(volume_updates),
        'sentiment_updates': len(sentiment_updates),
        'db_records': db_count,
        'success': db_count > 0 and len(order_book_updates) > 0
    }


async def test_oauth_required():
    """
    Test if OAuth credentials are needed.

    Note: cTrader requires OAuth 2.0 flow to get access token.
    This is a one-time process that requires browser interaction.
    """
    logger.info("\n" + "=" * 60)
    logger.info("OAuth Flow Information")
    logger.info("=" * 60)
    logger.info("""
To use cTrader API, you need:

1. Create a cTrader Open API application:
   https://openapi.ctrader.com/

2. Get your Client ID and Client Secret from the app

3. Run OAuth flow to get access token:

   from forex_diffusion.credentials.oauth import OAuth2Flow

   oauth = OAuth2Flow(client_id, client_secret)
   token_data = await oauth.authorize()  # Opens browser

   access_token = token_data['access_token']
   refresh_token = token_data['refresh_token']

4. Set environment variables:
   CTRADER_CLIENT_ID = your_client_id
   CTRADER_CLIENT_SECRET = your_client_secret
   CTRADER_ACCESS_TOKEN = access_token
   CTRADER_ACCOUNT_ID = your_trading_account_id

5. Run this test script

Note: Access tokens expire. Use refresh_token to get new tokens.
    """)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Check for credentials
    if not os.getenv('CTRADER_ACCESS_TOKEN'):
        logger.warning("No access token found in environment")
        asyncio.run(test_oauth_required())
        sys.exit(1)

    # Run test
    try:
        result = asyncio.run(test_websocket())

        if result and result['success']:
            logger.info("\n✅ All tests passed!")
            sys.exit(0)
        else:
            logger.error("\n❌ Tests failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"\n❌ Test failed with error: {e}")
        sys.exit(1)
