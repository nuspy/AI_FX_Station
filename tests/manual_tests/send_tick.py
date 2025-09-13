import asyncio
import json
import time
import random

try:
    import websockets
except ImportError:
    raise SystemExit("Missing dependency 'websockets'. Install with: pip install websockets")


async def send():
    """Connects to the local WebSocket and sends a simulated tick every second."""
    uri = "ws://127.0.0.1:8765"

    # --- Simulation Parameters ---
    # Initial price for EUR/USD
    price = 1.17300
    # Small positive drift to simulate a slight upward trend
    drift = 0.000001
    # The difference between bid and ask prices
    spread = 0.00005

    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        print("Connection successful. Starting to send simulated EUR/USD ticks...")
        while True:
            # 1. Calculate a small random price variation
            # This creates the up/down fluctuation of the market
            random_step = random.uniform(-0.00005, 0.00005)

            # 2. Apply the variation and the drift to the price
            price += random_step + drift

            # 3. Calculate bid and ask based on the new price and spread
            bid = price - (spread / 2)
            ask = price + (spread / 2)

            # 4. Create the message payload
            msg = {
                "symbol": "EUR/USD",
                "ts_utc": int(time.time() * 1000),
                "price": round(price, 5),
                "bid": round(bid, 5),
                "ask": round(ask, 5),
                "volume": round(random.uniform(0.1, 5.0), 2)  # Add some random volume
            }

            # 5. Send the message and print to console
            await ws.send(json.dumps(msg))
            print(f"Sent: Price={msg['price']:.5f}, Bid={msg['bid']:.5f}, Ask={msg['ask']:.5f}")

            # 6. Wait for one second before sending the next tick
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(send())
    except KeyboardInterrupt:
        print("\nScript stopped by user.")
    except ConnectionRefusedError:
        print(f"Connection refused. Is the main application running and listening on ws://127.0.0.1:8765?")
