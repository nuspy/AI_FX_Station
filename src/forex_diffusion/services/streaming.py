# streaming.py - simple WebSocket provider wrapper (best-effort)
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

try:
    import websocket  # type: ignore
    _HAS_WS = True
except Exception:
    _HAS_WS = False

class WebSocketProvider:
    """
    Minimal WebSocket wrapper that connects to a configured ws_url and calls a callback on each message.
    Non-blocking: runs in a background thread.
    """
    def __init__(self, ws_url: str, on_message: Callable[[str], None], on_open: Optional[Callable]=None, on_close: Optional[Callable]=None):
        if not _HAS_WS:
            raise RuntimeError("websocket-client is not installed")
        self.ws_url = ws_url
        self.on_message = on_message
        self.on_open = on_open
        self.on_close = on_close
        self._ws = None
        self._thread = None
        self._stop = threading.Event()

    def _run(self):
        def _on_msg(ws, message):
            try:
                self.on_message(message)
            except Exception:
                pass
        def _on_open(ws):
            if self.on_open:
                try:
                    self.on_open()
                except Exception:
                    pass
        def _on_close(ws, *args):
            if self.on_close:
                try:
                    self.on_close()
                except Exception:
                    pass
        self._ws = websocket.WebSocketApp(self.ws_url, on_message=_on_msg, on_open=_on_open, on_close=_on_close)
        while not self._stop.is_set():
            try:
                self._ws.run_forever()
            except Exception:
                time.sleep(1)
        try:
            self._ws.close()
        except Exception:
            pass

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
