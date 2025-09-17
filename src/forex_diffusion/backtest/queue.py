from __future__ import annotations

import threading
import time
from typing import List

from loguru import logger

from .db import BacktestDB
from .worker import Worker, TrialConfig


class BacktestQueue:
    """Simple background worker that polls for pending jobs and executes them asynchronously."""

    def __init__(self, poll_interval: float = 1.0):
        self.db = BacktestDB()
        self.worker = Worker(db=self.db)
        self.poll_interval = float(poll_interval)
        self._thr: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self):
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()
        logger.info("BacktestQueue started")

    def stop(self, timeout: float | None = None):
        self._stop.set()
        if self._thr and self._thr.is_alive():
            self._thr.join(timeout=timeout)
        logger.info("BacktestQueue stopped")

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self.db.next_pending_job()
                if not job:
                    time.sleep(self.poll_interval)
                    continue
                job_id = int(job["id"]) if isinstance(job, dict) else int(getattr(job, "id"))
                self.db.set_job_status(job_id, "running")
                # Build TrialConfig list from stored bt_config rows
                cfg_rows = self.db.configs_for_job(job_id)
                configs: List[TrialConfig] = []
                for row in cfg_rows:
                    payload = row.get("payload_json") or {}
                    configs.append(TrialConfig(
                        model_name=str(payload.get("model", "baseline_rw")),
                        prediction_type=str(payload.get("ptype", "Baseline")),
                        timeframe=str(payload.get("timeframe", "1m")),
                        horizons_sec=list(payload.get("horizons_sec", [])),
                        samples_range=tuple(payload.get("samples_range", (200,1500,200))),
                        indicators=dict(payload.get("indicators", {})),
                        interval=dict(payload.get("interval", {})),
                        data_version=payload.get("data_version"),
                        extra=dict(payload.get("extra", {})),
                    ))
                if configs:
                    # ensure market services do not trigger REST backfill during job
                    try:
                        from ..services.marketdata import MarketDataService
                        # not strictly needed here; worker reads directly from DB
                    except Exception:
                        pass
                    self.worker.run_job(job_id=job_id, configs=configs)
                self.db.set_job_status(job_id, "done")
            except Exception as e:
                logger.exception("BacktestQueue error: {}", e)
                time.sleep(self.poll_interval)


