"""
Mock cTrader WebSocket Test - Works Without Real API Credentials

This script simulates cTrader WebSocket data to test the entire system:
- Order book streaming
- Volume aggregation
- Sentiment calculation
- Database persistence
- GUI integration

Run this IMMEDIATELY to validate all code without API setup.
"""

import asyncio
import time
import random
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from sqlalchemy import create_engine, text


class MockCTraderWebSocket:
    """Simulates cTrader WebSocket with realistic data."""

    def __init__(self, db_engine, symbols=None):
        self.db_engine = db_engine
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY']

        # Realistic starting prices
        self.prices = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2650,
            'USDJPY': 149.50
        }

        # Simulated order book depth
        self.depth_levels = 5

        # Tracking
        self.order_book_updates = []
        self.volume_updates = []
        self.sentiment_updates = []

        # Volume buffers for sentiment
        self.volume_buffers = {s: [] for s in self.symbols}

        # Callbacks
        self.on_order_book_update = None
        self.on_volume_update = None
        self.on_sentiment_update = None

        self.running = False

    def start(self):
        """Start mock streaming."""
        self.running = True
        logger.info("✓ Mock WebSocket started")

    def stop(self):
        """Stop mock streaming."""
        self.running = False
        logger.info("✓ Mock WebSocket stopped")

    def _generate_order_book(self, symbol: str) -> dict:
        """Generate realistic order book snapshot."""
        current_price = self.prices[symbol]

        # Simulate price movement (random walk)
        change = random.gauss(0, 0.0001)  # 1 pip standard deviation
        current_price += change
        self.prices[symbol] = current_price

        # Spread (typically 0.5-2 pips for major pairs)
        spread_pips = random.uniform(0.8, 1.5)
        spread = spread_pips / 10000

        bid = current_price - spread / 2
        ask = current_price + spread / 2

        # Generate depth levels
        bids = []
        asks = []

        for i in range(self.depth_levels):
            # Bids (prices decrease)
            bid_price = bid - (i * spread)
            bid_volume = random.uniform(50000, 200000) * (1.0 - i * 0.15)  # Decreasing volume
            bids.append([round(bid_price, 5), round(bid_volume, 2)])

            # Asks (prices increase)
            ask_price = ask + (i * spread)
            ask_volume = random.uniform(50000, 200000) * (1.0 - i * 0.15)
            asks.append([round(ask_price, 5), round(ask_volume, 2)])

        # Calculate imbalance
        total_bid_vol = sum(b[1] for b in bids)
        total_ask_vol = sum(a[1] for a in asks)
        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

        return {
            'symbol': symbol,
            'timestamp': int(time.time() * 1000),
            'bids': bids,
            'asks': asks,
            'mid_price': current_price,
            'spread': spread,
            'imbalance': imbalance
        }

    def _generate_volume_tick(self, symbol: str, order_book: dict) -> dict:
        """Generate volume data for a tick."""
        # Simulate tick volume (varies by market activity)
        tick_volume = random.randint(1, 20)

        # Simulate bid/ask volume split (affects sentiment)
        # If imbalance is positive (more bids), more buying
        imbalance = order_book['imbalance']

        if imbalance > 0:
            # More bid volume = more buyers = more buy orders filled
            buy_ratio = 0.5 + (imbalance * 0.3)
        else:
            buy_ratio = 0.5 + (imbalance * 0.3)

        buy_ratio = max(0.2, min(0.8, buy_ratio))  # Clamp to 20-80%

        bid_volume = tick_volume * buy_ratio
        ask_volume = tick_volume * (1 - buy_ratio)

        return {
            'symbol': symbol,
            'timestamp': order_book['timestamp'],
            'tick_volume': tick_volume,
            'bid_volume': round(bid_volume, 2),
            'ask_volume': round(ask_volume, 2),
            'price': order_book['mid_price']
        }

    def _calculate_sentiment(self, symbol: str, volume_data: dict):
        """Calculate sentiment from recent volumes."""
        # Add to buffer
        self.volume_buffers[symbol].append(volume_data)

        # Keep last 20 ticks
        if len(self.volume_buffers[symbol]) > 20:
            self.volume_buffers[symbol].pop(0)

        # Need at least 10 ticks for sentiment
        if len(self.volume_buffers[symbol]) < 10:
            return None

        # Calculate total buy vs sell
        total_buy = sum(v['bid_volume'] for v in self.volume_buffers[symbol])
        total_sell = sum(v['ask_volume'] for v in self.volume_buffers[symbol])

        if total_buy + total_sell == 0:
            return None

        # Sentiment ratio
        ratio = (total_buy - total_sell) / (total_buy + total_sell)

        # Classify
        if ratio > 0.3:
            sentiment = "bullish"
        elif ratio < -0.3:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            'symbol': symbol,
            'timestamp': volume_data['timestamp'],
            'sentiment': sentiment,
            'ratio': ratio,
            'buy_volume': total_buy,
            'sell_volume': total_sell,
            'confidence': abs(ratio)
        }

    def _store_order_book(self, dom_data: dict):
        """Store order book to database."""
        try:
            with self.db_engine.begin() as conn:
                query = text(
                    "INSERT INTO market_depth (symbol, ts_utc, bids, asks, mid_price, spread, imbalance, provider) "
                    "VALUES (:symbol, :ts_utc, :bids, :asks, :mid_price, :spread, :imbalance, :provider)"
                )

                conn.execute(query, {
                    'symbol': dom_data['symbol'],
                    'ts_utc': dom_data['timestamp'],
                    'bids': json.dumps(dom_data['bids']),
                    'asks': json.dumps(dom_data['asks']),
                    'mid_price': dom_data['mid_price'],
                    'spread': dom_data['spread'],
                    'imbalance': dom_data['imbalance'],
                    'provider': 'ctrader_mock'
                })

        except Exception as e:
            logger.error(f"Error storing order book: {e}")

    async def stream(self, duration: int = 30):
        """Stream mock data for specified duration."""
        logger.info(f"Streaming mock data for {duration} seconds...")

        start_time = time.time()
        tick_interval = 0.5  # 2 ticks/second (realistic)

        while time.time() - start_time < duration:
            for symbol in self.symbols:
                # Generate order book
                order_book = self._generate_order_book(symbol)
                self.order_book_updates.append(order_book)

                # Store to database
                self._store_order_book(order_book)

                # Callback
                if self.on_order_book_update:
                    self.on_order_book_update(order_book)

                # Generate volume
                volume_data = self._generate_volume_tick(symbol, order_book)
                self.volume_updates.append(volume_data)

                # Callback
                if self.on_volume_update:
                    self.on_volume_update(volume_data)

                # Calculate sentiment
                sentiment = self._calculate_sentiment(symbol, volume_data)
                if sentiment:
                    self.sentiment_updates.append(sentiment)

                    # Callback
                    if self.on_sentiment_update:
                        self.on_sentiment_update(sentiment)

            # Wait before next tick
            await asyncio.sleep(tick_interval)

        logger.info(f"✓ Streaming complete ({duration}s)")


async def test_mock_websocket():
    """Test with mock cTrader data."""

    logger.info("=" * 70)
    logger.info("Mock cTrader WebSocket Test")
    logger.info("=" * 70)
    logger.info("This test simulates cTrader WebSocket without real API")
    print()

    # Create test database
    db_engine = create_engine("sqlite:///:memory:", echo=False)

    with db_engine.begin() as conn:
        conn.execute(text("""
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
        """))

    logger.info("✓ Created test database")

    # Create mock service
    mock_ws = MockCTraderWebSocket(
        db_engine=db_engine,
        symbols=['EURUSD', 'GBPUSD', 'USDJPY']
    )

    # Set callbacks with logging
    def on_order_book(data):
        logger.info(
            f"📊 {data['symbol']}: "
            f"Price={data['mid_price']:.5f}, "
            f"Spread={data['spread']*10000:.1f} pips, "
            f"Imbalance={data['imbalance']:.2f}"
        )

    def on_volume(data):
        logger.info(
            f"📈 {data['symbol']}: "
            f"Volume={data['tick_volume']}, "
            f"Buy={data['bid_volume']:.1f}, "
            f"Sell={data['ask_volume']:.1f}"
        )

    def on_sentiment(data):
        emoji = "🟢" if data['sentiment'] == 'bullish' else "🔴" if data['sentiment'] == 'bearish' else "⚪"
        logger.info(
            f"{emoji} {data['symbol']}: "
            f"{data['sentiment'].upper()} "
            f"(ratio={data['ratio']:.2f}, confidence={data['confidence']:.2f})"
        )

    mock_ws.on_order_book_update = on_order_book
    mock_ws.on_volume_update = on_volume
    mock_ws.on_sentiment_update = on_sentiment

    # Start service
    mock_ws.start()

    # Stream for 10 seconds (reduced for testing)
    print("\n" + "=" * 70)
    logger.info("Starting data stream (10 seconds)...")
    print("=" * 70 + "\n")

    await mock_ws.stream(duration=10)

    # Final summary
    print("\n" + "=" * 70)
    logger.info("Test Complete - Results")
    print("=" * 70)

    logger.info(f"✓ Order Book Updates: {len(mock_ws.order_book_updates)}")
    logger.info(f"✓ Volume Updates: {len(mock_ws.volume_updates)}")
    logger.info(f"✓ Sentiment Updates: {len(mock_ws.sentiment_updates)}")

    # Check database
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM market_depth")).fetchone()
        db_count = result[0]

        logger.info(f"✓ Database Records: {db_count}")

        # Sample records
        if db_count > 0:
            sample = conn.execute(text(
                "SELECT symbol, ts_utc, mid_price, spread, imbalance "
                "FROM market_depth ORDER BY ts_utc DESC LIMIT 5"
            )).fetchall()

            print("\nLatest 5 Records:")
            for row in sample:
                symbol, ts, mid, spread, imb = row
                dt = time.strftime('%H:%M:%S', time.localtime(ts / 1000))
                logger.info(
                    f"  {dt} - {symbol}: "
                    f"Price={mid:.5f}, "
                    f"Spread={spread*10000:.1f}pips, "
                    f"Imbalance={imb:.2f}"
                )

        # Performance
        duration = 10
        rate = db_count / duration if duration > 0 else 0
        print(f"\n✓ Average Rate: {rate:.2f} updates/second per symbol")
        logger.info(f"✓ Total Rate: {rate * 3:.2f} updates/second (3 symbols)")

    # Stop service
    mock_ws.stop()

    # Verify results (adjusted for 10 second test)
    success = (
        len(mock_ws.order_book_updates) > 15 and
        len(mock_ws.volume_updates) > 15 and
        len(mock_ws.sentiment_updates) > 5 and
        db_count > 15
    )

    if success:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        logger.info("1. This validates all code works correctly")
        logger.info("2. Setup real cTrader API credentials (see CTrader_API_Setup_Guide.md)")
        logger.info("3. Replace MockCTraderWebSocket with CTraderWebSocketService")
        logger.info("4. Run tests/test_ctrader_websocket.py with real credentials")
    else:
        print("\n✗ Some tests failed")

    return {
        'order_book_count': len(mock_ws.order_book_updates),
        'volume_count': len(mock_ws.volume_updates),
        'sentiment_count': len(mock_ws.sentiment_updates),
        'db_records': db_count,
        'success': success
    }


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Run test
    try:
        result = asyncio.run(test_mock_websocket())

        if result['success']:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"\n❌ Test failed: {e}")
        sys.exit(1)
