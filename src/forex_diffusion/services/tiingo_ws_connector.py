# src/forex_diffusion/services/tiingo_ws_connector.py
from __future__ import annotations

import threading
import time
import json
import os
from typing import Optional, Iterable, List, Callable, Any
import pandas as pd
from loguru import logger

try:
    import websocket
    _HAS_WS_CLIENT = True
except ImportError:
    websocket = None
    _HAS_WS_CLIENT = False

class TiingoWSConnector:
    """
    Connects to a WebSocket and streams data by directly calling registered handlers.
    """
    def __init__(
        self, 
        uri: str, 
        api_key: Optional[str] = None, 
        tickers: Optional[Iterable[str]] = None, 
        chart_handler: Optional[Callable[[Any], None]] = None,
        db_handler: Optional[Callable[[Any], None]] = None,
        status_handler: Optional[Callable[[str], None]] = None
    ):
        self.uri = uri
        self.api_key = api_key
        self.tickers = [str(t).lower() for t in (tickers or ["eurusd"])]
        self.chart_handler = chart_handler
        self.db_handler = db_handler
        self.status_handler = status_handler
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ws_app: Optional[websocket.WebSocketApp] = None
        self._was_down = False

    @property
    def running(self) -> bool:
        """Check if WebSocket connector is currently running."""
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def start(self):
        if not _HAS_WS_CLIENT:
            logger.warning("TiingoWSConnector not started: 'websocket' package not installed.")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("TiingoWSConnector thread launched.")

    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        if self._ws_app:
            self._ws_app.close()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("TiingoWSConnector stopped.")

    def _run(self):
        while not self._stop_event.is_set():
            self._ws_app = websocket.WebSocketApp(
                self.uri,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            self._ws_app.run_forever(ping_interval=20, ping_timeout=10)
            
            if not self._stop_event.is_set():
                logger.warning("WebSocket connection closed. Reconnecting in 5 seconds...")
                time.sleep(5)

    def _on_open(self, ws):
        logger.info("TiingoWSConnector connection opened.")
        try:
            if self._was_down and self.status_handler:
                try:
                    self.status_handler("ws_restored")
                except Exception:
                    pass
            self._was_down = False
            sub_payload = {
                "eventName": "subscribe",
                "authorization": self.api_key or "",
                "eventData": {"thresholdLevel": "5", "tickers": self.tickers}
                }
            ws.send(json.dumps(sub_payload))
            logger.info(f"TiingoWSConnector subscribe sent: tickers={self.tickers}")
        except Exception as e:
            logger.warning(f"TiingoWSConnector failed to send subscribe: {e}")



    def _on_message(self, ws, message):
        msg = json.loads(message)
        mtype = msg.get("messageType")
        data = msg.get("data")
        
        if mtype != "A" or not isinstance(data, list) or len(data) < 6:
            return

        pair, iso_ts, bid, ask = data[1], data[2], data[4], data[5]
        price = (float(bid) + float(ask)) / 2.0
        ts_ms = int(pd.to_datetime(iso_ts).value // 1_000_000)
        norm_symbol = f"{pair[:3].upper()}/{pair[3:].upper()}"
        
        payload = {
            "symbol": norm_symbol, "ts_utc": ts_ms,
            "price": price, "bid": float(bid), "ask": float(ask), "volume": None
        }

        if self.chart_handler:
            self.chart_handler(payload)
        
        if self.db_handler:
            self.db_handler(payload)

    def _on_error(self, ws, error):
        logger.error(f"TiingoWSConnector error: {error}")
        try:
            if not self._was_down and self.status_handler:
                self.status_handler("ws_down")
            self._was_down = True
        except Exception:
            pass

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"TiingoWSConnector connection closed: {close_status_code} - {close_msg}")
        try:
            if not self._was_down and self.status_handler:
                self.status_handler("ws_down")
            self._was_down = True
        except Exception:
            pass
