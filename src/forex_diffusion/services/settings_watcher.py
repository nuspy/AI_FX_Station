"""
SettingsWatcher: monitor file-based settings and call registered callbacks on change.
Lightweight thread checking mtime.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from ..utils.user_settings import SETTINGS_FILE, load_settings
from loguru import logger

class SettingsWatcher:
    def __init__(self, path: Optional[Path] = None, poll_interval: float = 2.0):
        self.path = path or SETTINGS_FILE
        self.poll = float(poll_interval)
        self._callbacks: List[Callable[[dict], None]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_mtime = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        try:
            if self.path.exists():
                self._last_mtime = self.path.stat().st_mtime
        except Exception:
            self._last_mtime = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="SettingsWatcher", daemon=True)
        self._thread.start()
        logger.info("SettingsWatcher started, watching {}", self.path)

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("SettingsWatcher stopped")

    def register_callback(self, cb: Callable[[dict], None]):
        self._callbacks.append(cb)

    def _run(self):
        while not self._stop.is_set():
            try:
                if self.path.exists():
                    mtime = self.path.stat().st_mtime
                    if self._last_mtime is None or mtime > self._last_mtime:
                        # reload and call callbacks
                        try:
                            cfg = load_settings()
                            for cb in self._callbacks:
                                try:
                                    cb(cfg)
                                except Exception as e:
                                    logger.exception("SettingsWatcher callback failed: {}", e)
                        except Exception as e:
                            logger.exception("SettingsWatcher failed to load settings: {}", e)
                        self._last_mtime = mtime
            except Exception as e:
                logger.exception("SettingsWatcher loop error: {}", e)
            # sleep
            for _ in range(int(max(1, self.poll))):
                if self._stop.is_set():
                    break
                time.sleep(1)
