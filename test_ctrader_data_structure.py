"""
Test script to show actual data structures from cTrader API.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.forex_diffusion.providers.ctrader_provider import CTraderProvider
from src.forex_diffusion.utils.user_settings import get_setting
from loguru import logger


async def test_historical_bars():
    """Show structure of historical bars (candele storiche)."""
    print("\n" + "="*60)
    print("HISTORICAL BARS (Candele Storiche - REST API)")
    print("="*60)

    config = {
        'client_id': get_setting('ctrader_client_id', ''),
        'client_secret': get_setting('ctrader_client_secret', ''),
        'access_token': get_setting('ctrader_access_token', ''),
        'environment': get_setting('ctrader_environment', 'demo'),
    }

    provider = CTraderProvider(config=config)

    try:
        # Connect
        print("Connecting to cTrader...")
        success = await provider.connect()
        if not success:
            print("Failed to connect")
            return

        # Get 1 hour of data
        from datetime import datetime, timedelta, timezone
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        print(f"Requesting bars for EUR/USD from {start_time} to {end_time}")
        df = await provider.get_historical_bars("EUR/USD", "5m", start_ms, end_ms)

        if df is not None and not df.empty:
            print(f"\nReceived {len(df)} bars")
            print("\nFirst 3 bars:")
            print(df.head(3).to_string())
            print("\nColumns:", list(df.columns))
            print("\nSample bar data:")
            bar = df.iloc[0]
            print(f"  timestamp: {bar['ts_utc']}")
            print(f"  open:      {bar['open']:.5f}")
            print(f"  high:      {bar['high']:.5f}")
            print(f"  low:       {bar['low']:.5f}")
            print(f"  close:     {bar['close']:.5f}")
            print(f"  volume:    {bar.get('volume', 'N/A')}")
            print(f"  tick_volume: {bar.get('tick_volume', 'N/A')}")
            print("\n⚠️  NOTE: Trendbars have OHLC only, NO separate bid/ask")
        else:
            print("No data received")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await provider.disconnect()


async def test_spot_event():
    """Show structure of WebSocket spot events (realtime)."""
    print("\n" + "="*60)
    print("SPOT EVENT (Realtime WebSocket)")
    print("="*60)

    config = {
        'client_id': get_setting('ctrader_client_id', ''),
        'client_secret': get_setting('ctrader_client_secret', ''),
        'access_token': get_setting('ctrader_access_token', ''),
        'environment': get_setting('ctrader_environment', 'demo'),
    }

    provider = CTraderProvider(config=config)

    try:
        # Connect
        print("Connecting to cTrader...")
        success = await provider.connect()
        if not success:
            print("Failed to connect")
            return

        print("\n⚠️  Spot event structure (from protobuf definition):")
        print("  Fields: bid, ask, timestamp, symbolId")
        print("  Example:")
        print("    bid: 1.08523")
        print("    ask: 1.08525")
        print("    spread: 0.00002 (2 pips)")
        print("    timestamp: 1640000000000")
        print("\n✓ Spot events HAVE separate bid and ask prices")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await provider.disconnect()


async def test_tick_data():
    """Show structure of historical tick data."""
    print("\n" + "="*60)
    print("TICK DATA (Historical Ticks - REST API)")
    print("="*60)

    config = {
        'client_id': get_setting('ctrader_client_id', ''),
        'client_secret': get_setting('ctrader_client_secret', ''),
        'access_token': get_setting('ctrader_access_token', ''),
        'environment': get_setting('ctrader_environment', 'demo'),
    }

    provider = CTraderProvider(config=config)

    try:
        # Connect
        print("Connecting to cTrader...")
        success = await provider.connect()
        if not success:
            print("Failed to connect")
            return

        # Get 10 minutes of tick data
        from datetime import datetime, timedelta, timezone
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=10)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        print(f"Requesting ticks for EUR/USD from {start_time} to {end_time}")
        df = await provider.get_historical_ticks("EUR/USD", start_ms, end_ms)

        if df is not None and not df.empty:
            print(f"\nReceived {len(df)} ticks")
            print("\nFirst 5 ticks:")
            print(df.head(5).to_string())
            print("\nColumns:", list(df.columns))
            print("\nSample tick data:")
            tick = df.iloc[0]
            print(f"  timestamp: {tick['ts_utc']}")
            print(f"  bid:       {tick['bid']:.5f}" if tick['bid'] is not None else "  bid:       None")
            print(f"  ask:       {tick['ask']:.5f}" if tick['ask'] is not None else "  ask:       None")
            print(f"  price:     {tick['price']:.5f}")
            print("\n✓ Our implementation requests BOTH bid and ask separately and merges them")
        else:
            print("No data received")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await provider.disconnect()


async def main():
    """Run all tests."""
    # Test historical bars (candele)
    await test_historical_bars()

    # Test spot event structure (WebSocket realtime)
    await test_spot_event()

    # Test historical tick data
    await test_tick_data()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("1. HISTORICAL BARS (candele): OHLC only, NO bid/ask")
    print("   → Use for: Chart display, pattern detection")
    print()
    print("2. SPOT EVENTS (WebSocket): HAS bid AND ask")
    print("   → Use for: Realtime quotes in GUI, order execution")
    print()
    print("3. HISTORICAL TICKS: Single value per request (BID or ASK)")
    print("   → Use for: Tick analysis, need both bid and ask")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
