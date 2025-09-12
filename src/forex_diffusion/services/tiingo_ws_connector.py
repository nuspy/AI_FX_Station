# src/forex_diffusion/services/tiingo_ws_connector.py
from __future__ import annotations

import threading
import time
import json
import os
from typing import Optional, Iterable
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
    Connects to Tiingo WebSocket using an event-driven approach with WebSocketApp.
    Handles automatic keep-alive and reconnects.
    """
    def __init__(self, uri: str = "wss://api.tiingo.com/fx", api_key: Optional[str] = None, tickers: Optional[Iterable[str]] = None, threshold: str = "5"):
        self.uri = uri
        self.api_key = api_key or os.environ.get("TIINGO_APIKEY") or os.environ.get("TIINGO_API_KEY")
        self.tickers = [str(t).lower() for t in (list(tickers) if tickers else ["eurusd"])]
        self.threshold = str(threshold)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ws_app: Optional[websocket.WebSocketApp] = None
        try:
            from ..utils.event_bus import publish
            self._publish = publish
        except ImportError:
            self._publish = lambda *a, **k: None

    def start(self):
        if not _HAS_WS_CLIENT:
            logger.warning("TiingoWSConnector not started: missing 'websocket' package")
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
            # run_forever handles the connection loop, including pings and reading messages.
            self._ws_app.run_forever(ping_interval=20, ping_timeout=10)
            
            if not self._stop_event.is_set():
                logger.warning("WebSocket connection closed. Reconnecting in 5 seconds...")
                time.sleep(5)

    def _on_open(self, ws):
        logger.info("TiingoWSConnector connection opened.")
        try:
            sub_payload = {
                "eventName": "subscribe",
                "authorization": self.api_key or "",
                "eventData": {
                    "thresholdLevel": self.threshold,
                    "tickers": self.tickers
                }
            }
            ws.send(json.dumps(sub_payload))
            logger.info(f"TiingoWSConnector subscribe sent: tickers={self.tickers}")
        except Exception as e:
            logger.exception(f"Failed to send subscribe message: {e}")

    def _on_message(self, ws, message):
        try:
            logger.info(f"TiingoWSConnector RAW: {message}")
            msg = json.loads(message)
            mtype = msg.get("messageType")
            data = msg.get("data")
            payload = None

            if mtype == "A" and isinstance(data, list) and len(data) >= 6:
                pair, iso_ts, bid, ask = data[1], data[2], data[4], data[5]
                price = bid if bid is not None else ask
                ts_ms = int(pd.to_datetime(iso_ts).value // 1_000_000)
                norm_symbol = f"{pair[:3].upper()}/{pair[3:].upper()}"
                payload = {
                    "symbol": norm_symbol, "ts_utc": ts_ms,
                    "price": float(price) if price is not None else None,
                    "bid": float(bid) if bid is not None else None,
                    "ask": float(ask) if ask is not None else None, "volume": None
                }
            elif mtype == "T" and isinstance(data, list) and len(data) >= 5:
                pair, iso_ts, price, size = data[1], data[2], data[3], data[4]
                ts_ms = int(pd.to_datetime(iso_ts).value // 1_000_000)
                norm_symbol = f"{pair[:3].upper()}/{pair[3:].upper()}"
                payload = {
                    "symbol": norm_symbol, "ts_utc": ts_ms,
                    "price": float(price) if price is not None else None,
                    "bid": None, "ask": None, "volume": float(size) if size is not None else None
                }
            elif mtype in ("I", "H"):
                logger.debug(f"TiingoWSConnector info/heartbeat: {msg.get('response', msg)}")

            if payload:
                self._publish("tick", payload)

        except Exception as e:
            logger.exception(f"Error processing message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"TiingoWSConnector error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"TiingoWSConnector connection closed: {close_status_code} - {close_msg}")
