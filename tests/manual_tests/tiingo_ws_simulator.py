#!/usr/bin/env python3
"""
tests/manual_tests/tiingo_ws_simulator.py

A WebSocket server that simulates the Tiingo FX data feed.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime, timezone
import argparse

try:
    import websockets
except ImportError:
    raise SystemExit("Missing dependency 'websockets'. Install with: pip install websockets")

CONNECTED_CLIENTS = set()
SUBSCRIPTIONS = {}

class MarketSimulator:
    def __init__(self, initial_price=1.17300):
        self.price = initial_price
        self.drift = 0.000001
        self.volatility = 0.00005
        self.spread = 0.00005

    def next_tick(self) -> dict:
        random_step = random.uniform(-self.volatility, self.volatility)
        self.price += random_step + self.drift
        bid = self.price - (self.spread / 2)
        ask = self.price + (self.spread / 2)
        return {"price": round(self.price, 5), "bid": round(bid, 5), "ask": round(ask, 5)}

async def data_sender(websocket):
    client_id = id(websocket)
    simulators = {}
    last_heartbeat_time = time.time()

    while True:
        try:
            await asyncio.sleep(1)
            if client_id not in SUBSCRIPTIONS:
                continue

            # Create simulators for newly subscribed tickers
            for ticker in SUBSCRIPTIONS.get(client_id, []):
                if ticker not in simulators:
                    simulators[ticker] = MarketSimulator()

            # Generate and send a data message for each subscribed ticker
            for ticker, simulator in simulators.items():
                tick_data = simulator.next_tick()
                message = {
                    "service": "fx", "messageType": "A",
                    "data": ["Q", ticker, datetime.now(timezone.utc).isoformat(), 1000000.0, tick_data["bid"], tick_data["ask"], tick_data["price"]]
                }
                await websocket.send(json.dumps(message))

            # Send a heartbeat every 15 seconds
            if time.time() - last_heartbeat_time > 15:
                heartbeat = {"response": {"code": 200, "message": "HeartBeat"}, "messageType": "H"}
                await websocket.send(json.dumps(heartbeat))
                last_heartbeat_time = time.time()

        except websockets.ConnectionClosed:
            print(f"Connection closed for client {client_id}. Stopping data sender.")
            break
        except Exception as e:
            print(f"Error in data_sender for client {client_id}: {e}")
            break

async def message_receiver(websocket):
    client_id = id(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("eventName") == "subscribe":
                    tickers = data.get("eventData", {}).get("tickers", [])
                    SUBSCRIPTIONS[client_id] = tickers
                    print(f"Client {client_id} subscribed to tickers: {tickers}")
                    ack_message = {
                        "response": {"code": 200, "message": "Success"},
                        "data": {"subscriptionId": random.randint(1000, 9999)},
                        "messageType": "I"
                    }
                    await websocket.send(json.dumps(ack_message))
            except Exception as e:
                print(f"Error processing message from client {client_id}: {e}")
    finally:
        if client_id in SUBSCRIPTIONS:
            del SUBSCRIPTIONS[client_id]

async def handler(websocket):
    client_id = id(websocket)
    print(f"Client {client_id} connected.")
    CONNECTED_CLIENTS.add(websocket)
    
    receiver_task = asyncio.create_task(message_receiver(websocket))
    sender_task = asyncio.create_task(data_sender(websocket))
    
    try:
        done, pending = await asyncio.wait([receiver_task, sender_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
    finally:
        print(f"Client {client_id} disconnected.")
        if websocket in CONNECTED_CLIENTS:
            CONNECTED_CLIENTS.remove(websocket)

async def main():
    parser = argparse.ArgumentParser(description="Tiingo WebSocket Simulator")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8766, help="Port to bind the server to")
    args = parser.parse_args()

    print(f"Starting Tiingo WebSocket simulator on ws://{args.host}:{args.port}")
    async with websockets.serve(handler, args.host, args.port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user.")
