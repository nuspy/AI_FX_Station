# src/forex_diffusion/services/local_ws.py
from __future__ import annotations

import asyncio
import json
import threading
from typing import Optional

import pandas as pd
from loguru import logger

# import websockets lazily / defensively
try:
    import websockets  # type: ignore
    _HAS_WEBSOCKETS = True
except Exception:
    websockets = None
    _HAS_WEBSOCKETS = False

from ..utils.event_bus import publish
from ..services.db_service import DBService
from ..data import io as data_io


class LocalWebsocketServer:
    """
    Simple WebSocket server that accepts JSON messages with fields:
      { "symbol": "EUR/USD", "timeframe": "1m", "ts_utc": 1660000000000, "price": 1.12345 }
    On message it:
      - creates a 1-row candle (open/high/low/close=price) and upserts into market_data_candles
      - publishes an internal event 'tick' with dict(payload)
    Runs in a background thread with its own asyncio loop.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, db_service: Optional[DBService] = None):
        self.host = host
        self.port = int(port)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.db_service = db_service or DBService()

    async def _handler(self, websocket, path):
        async for message in websocket:
            try:
                data = json.loads(message)
                # normalize expected fields
                symbol = data.get("symbol")
                timeframe = data.get("timeframe", "1m")
                ts = int(data.get("ts_utc", 0))
                price = float(data.get("price", 0.0))
                # build candle df
                df = pd.DataFrame([{
                    "ts_utc": int(ts),
                    "open": float(price),
                    "high": float(price),
                    "low": float(price),
                    "close": float(price),
                    "volume": None
                }])
                try:
                    # upsert into DB
                    data_io.upsert_candles(self.db_service.engine, df, symbol, timeframe, resampled=False)
                except Exception as e:
                    logger.exception("local_ws: failed to upsert candle: {}", e)
                # publish internal event for UI subscribers
                publish("tick", {"symbol": symbol, "timeframe": timeframe, "ts_utc": ts, "price": price})
            except Exception as e:
                logger.exception("local_ws: failed to process message: {}", e)

    def _start_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server = websockets.serve(self._handler, self.host, self.port, loop=loop)
        srv = loop.run_until_complete(server)
        logger.info("LocalWebsocketServer started on ws://{}:{}", self.host, self.port)
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(srv.wait_closed())
            loop.close()

    def start(self) -> bool:
        """
        Start websocket server in background thread. Returns True if server thread started, False otherwise.
        """
        if not _HAS_WEBSOCKETS:
            logger.warning("LocalWebsocketServer not started: 'websockets' package is not installed.")
            return False
        if self._thread and self._thread.is_alive():
            logger.debug("LocalWebsocketServer already running (thread alive).")
            return True
        try:
            self._stop.clear()
            self._thread = threading.Thread(target=self._start_loop, daemon=True)
            self._thread.start()
            logger.debug("LocalWebsocketServer thread launched.")
            return True
        except Exception as e:
            logger.exception("LocalWebsocketServer failed to start thread: {}", e)
            return False

    def stop(self):
        self._stop.set()
        # Stopping asyncio loop gracefully is handled by thread exit
