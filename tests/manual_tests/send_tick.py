import asyncio, json, time
try:
    import websockets
except Exception:
    raise SystemExit("Missing dependency 'websockets'. Install with: pip install websockets")

async def send():
    uri = "ws://127.0.0.1:8765"
    async with websockets.connect(uri) as ws:
        msg = {
            "symbol": "EUR/USD",
            "timeframe": "1m",
            "ts_utc": int(time.time() * 1000),
            "price": 1.23456,
            "bid": 1.23450,
            "ask": 1.23462
        }
        await ws.send(json.dumps(msg))
        print("sent", msg)

if __name__ == "__main__":
    asyncio.run(send())