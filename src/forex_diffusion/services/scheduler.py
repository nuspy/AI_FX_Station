"""
RegimeScheduler: scheduled job to perform periodic incremental updates of ANN index and export Prometheus metrics.

- Uses RegimeService.incremental_update(batch_size) to add new latents.
- Exposes Prometheus metrics: counter for updates, last update duration, last_indexed_id gauge.
- Start/stop safe and registered by inference lifespan.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from loguru import logger
from prometheus_client import Counter, Gauge, Histogram

from .regime_service import RegimeService

# Prometheus metrics
REGIME_INCREMENTAL_COUNTER = Counter("magicforex_regime_incremental_runs_total", "Number of incremental regime update runs")
REGIME_INCREMENTAL_DURATION = Histogram("magicforex_regime_incremental_duration_seconds", "Duration of incremental update (s)")
REGIME_LAST_INDEXED = Gauge("magicforex_regime_last_indexed_id", "Last indexed latent id (for ANN index)")

class RegimeScheduler:
    def __init__(self, engine=None, interval_seconds: int = 600, batch_size: int = 1000):
        self.rs = RegimeService(engine=engine)
        self.interval = int(interval_seconds)
        self.batch_size = int(batch_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="RegimeScheduler", daemon=True)
        self._thread.start()
        logger.info("RegimeScheduler started (interval={}s batch_size={})", self.interval, self.batch_size)

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("RegimeScheduler stopped")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                start = time.time()
                REGIME_INCREMENTAL_COUNTER.inc()
                with REGIME_INCREMENTAL_DURATION.time():
                    res = self.rs.incremental_update(batch_size=self.batch_size)
                # update gauge if returned last_indexed_id
                lid = res.get("last_indexed_id")
                if lid is not None:
                    REGIME_LAST_INDEXED.set(float(lid))
                logger.info("RegimeScheduler incremental update result: {}", res)
                # persist metric to DB (best-effort)
                try:
                    from .db_service import DBService
                    db = DBService()
                    db.write_metric("regime_incremental_updated", float(res.get("updated", 0)), labels={"last_indexed_id": lid})
                except Exception as e:
                    logger.exception("RegimeScheduler: failed to persist metric to DB: {}", e)
            except Exception as e:
                logger.exception("RegimeScheduler error during incremental update: {}", e)
            # sleep with interruption check
            for _ in range(self.interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
