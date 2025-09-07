"""
FeatureCompactor: scheduled compaction job that periodically deletes old features.

- Starts a background thread that wakes up every compaction_interval_hours and invokes DBService.compact_features.
- Safe to start/stop from application lifespan.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from loguru import logger

from .db_service import DBService
from ..utils.config import get_config


class FeatureCompactor:
    def __init__(self, engine=None):
        cfg = get_config()
        self.engine = engine or DBService().engine
        self.db = DBService(engine=self.engine)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.interval_hours = int(getattr(cfg, "features", {}).get("compaction_interval_hours", 24) if isinstance(getattr(cfg, "features", {}), dict) else 24)
        self.retention_days = int(getattr(cfg, "features", {}).get("retention_days", 365) if isinstance(getattr(cfg, "features", {}), dict) else 365)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="FeatureCompactor", daemon=True)
        self._thread.start()
        logger.info("FeatureCompactor started: interval_hours={} retention_days={}", self.interval_hours, self.retention_days)

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("FeatureCompactor stopped")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                # perform compaction
                try:
                    deleted = self.db.compact_features(older_than_days=self.retention_days)
                    logger.info("FeatureCompactor: compact_features removed {} rows older than {} days", deleted, self.retention_days)
                except Exception as e:
                    logger.exception("FeatureCompactor: compact_features failed: {}", e)
            except Exception as exc:
                logger.exception("FeatureCompactor run error: {}", exc)
            # sleep until next run
            for _ in range(int(self.interval_hours * 3600)):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
