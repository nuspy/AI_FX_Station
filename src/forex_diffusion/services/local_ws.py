# src/forex_diffusion/services/local_ws.py
from __future__ import annotations

import asyncio
import json
import threading
from typing import Optional

from loguru import logger

try:
    import websockets
except ImportError:
    websockets = None

from ..services.db_service import DBService

class LocalWebsocketServer:
    """
    A simple WebSocket server that listens for tick data, and directly persists it
    to the database using DBService.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, db_service: Optional[DBService] = None):
        self.host = host
        self.port = int(port)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.db_service = db_service or DBService()

    async def _handler(self, websocket, path):
        """Handles incoming messages, persisting them directly."""
        logger.critical(f"--- LOCAL_WS: NEW CLIENT CONNECTED FROM {path} ---")
        async for message in websocket:
            try:
                data = json.loads(message)
                if "symbol" not in data or "ts_utc" not in data:
                    logger.warning(f"local_ws: received message without required fields: {data}")
                    continue
                
                self.db_service.write_tick(data)
                logger.debug(f"local_ws: Persisted tick for {data.get('symbol')}")

            except Exception as e:
                logger.exception(f"local_ws: Failed to process message: {e}")

    def _start_loop(self):
        """Runs the asyncio event loop in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def server_main():
            async with websockets.serve(self._handler, self.host, self.port):
                logger.info(f"LocalWebsocketServer started on ws://{self.host}:{self.port}")
                await self._stop_event.wait()

        try:
            loop.run_until_complete(server_main())
        finally:
            loop.close()
            logger.info("LocalWebsocketServer event loop closed.")

    def start(self) -> bool:
        if not websockets:
            logger.warning("LocalWebsocketServer not started: 'websockets' package not installed.")
            return False
        if self._thread and self._thread.is_alive():
            return True
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()
        logger.debug("LocalWebsocketServer thread launched.")
        return True

    def stop(self):
        logger.info("Stopping LocalWebsocketServer...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("LocalWebsocketServer stopped.")
