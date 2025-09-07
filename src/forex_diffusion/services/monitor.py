"""
RegimeMonitor: periodically check ANN index metrics and expose them in-memory for admin queries.

- Non-blocking background thread.
- Query RegimeService.get_index_metrics() periodically.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from loguru import logger

from .regime_service import RegimeService


class RegimeMonitor:
    def __init__(self, engine=None, interval_seconds: int = 60):
        self.rs = RegimeService(engine=engine)
        self.interval = int(interval_seconds)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.latest_metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="RegimeMonitor", daemon=True)
        self._thread.start()
        logger.info("RegimeMonitor started (interval={}s)", self.interval)

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("RegimeMonitor stopped")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                metrics = self.rs.get_index_metrics()
                with self._lock:
                    self.latest_metrics = metrics
            except Exception as e:
                logger.exception("RegimeMonitor run error: {}", e)
            # sleep with stop-check
            for _ in range(self.interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.latest_metrics)
