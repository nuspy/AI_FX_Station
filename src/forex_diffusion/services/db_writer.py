import queue
import threading
from typing import Any, Dict, List, Optional

from loguru import logger

from .db_service import DBService


class DBWriter:
    """
    Asynchronously writes data to a database using a queue and a worker thread.
    """

    def __init__(self, db_service: DBService):
        self.db_service = db_service
        self._task_queue = queue.Queue()
        self._worker_thread = None
        self._is_running = False

    def _process_tasks(self):
        """The worker function that processes tasks from the queue."""
        logger.info("DBWriter worker thread started.")
        while self._is_running or not self._task_queue.empty():
            try:
                task_type, payload = self._task_queue.get(timeout=1)

                handler_name = f"write_{task_type}"
                handler = getattr(self.db_service, handler_name, None)

                if handler and callable(handler):
                    try:
                        logger.debug(f"Executing DB task: {task_type}")
                        handler(**payload)
                    except Exception:
                        logger.exception(f"Error executing DB task '{task_type}'.")
                else:
                    logger.warning(f"No handler '{handler_name}' found in DBService for task type '{task_type}'.")
                
                self._task_queue.task_done()

            except queue.Empty:
                # Queue is empty, loop again if still running
                continue
            except Exception:
                logger.exception("Exception in DBWriter worker thread.")
        
        logger.info("DBWriter worker thread stopped.")

    def start(self):
        """
        Starts the DBWriter worker thread.
        """
        if self._is_running:
            logger.warning("DBWriter is already running.")
            return

        logger.info("Starting DBWriter...")
        self._is_running = True
        self._worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self._worker_thread.start()

    def stop(self, wait: bool = True):
        """
        Stops the DBWriter worker thread.

        Args:
            wait: If True, waits for the worker thread to finish processing the queue.
        """
        if not self._is_running:
            return

        logger.info("Stopping DBWriter...")
        self._is_running = False
        if wait and self._worker_thread:
            self._worker_thread.join()
        logger.info("DBWriter stopped.")

    def enqueue_task(self, task_type: str, payload: Dict) -> bool:
        """
        Enqueues a task for the worker to process.
        """
        if not self._is_running:
            # In a real scenario, you might want to handle this differently,
            # but for now, we'll just log a warning.
            logger.warning("DBWriter is not running. Task not enqueued.")
            return False
        
        self._task_queue.put((task_type, payload))
        return True

    def write_prediction_async(
        self,
        symbol: str,
        timeframe: str,
        horizon: str,
        q05: float,
        q50: float,
        q95: float,
        meta: Optional[Dict] = None,
    ) -> bool:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": horizon,
            "q05": float(q05),
            "q50": float(q50),
            "q95": float(q95),
            "meta": meta or {},
        }
        return self.enqueue_task("prediction", payload)

    def write_signal_async(
        self,
        symbol: str,
        timeframe: str,
        entry_price: float,
        target_price: float,
        stop_price: float,
        metrics: Optional[Dict] = None,
    ) -> bool:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "entry_price": float(entry_price),
            "target_price": float(target_price),
            "stop_price": float(stop_price),
            "metrics": metrics or {},
        }
        return self.enqueue_task("signal", payload)

    def write_calibration_async(
        self,
        symbol: str,
        timeframe: str,
        ts_created_ms: int,
        alpha: float,
        half_life_days: float,
        delta_global: float,
        cov_hat: float,
        details: Optional[Dict] = None,
    ) -> bool:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "ts_created_ms": ts_created_ms,
            "alpha": float(alpha),
            "half_life_days": float(half_life_days),
            "delta_global": float(delta_global),
            "cov_hat": float(cov_hat),
            "details": details or {},
        }
        return self.enqueue_task("calibration", payload)

    def write_features_async(
        self,
        symbol: str,
        timeframe: str,
        ts_utc: int,
        features: Dict[str, Any],
        pipeline_version: Optional[str] = None,
    ) -> bool:
        """
        Enqueue a features row for async persistence.
        """
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "ts_utc": int(ts_utc),
            "features": features,
            "pipeline_version": pipeline_version,
        }
        return self.enqueue_task("features", payload)

    def write_latents_async(
        self,
        symbol: Optional[str],
        timeframe: Optional[str],
        ts_utc: int,
        latent: List[float],
        model_version: Optional[str] = None,
    ) -> bool:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "ts_utc": int(ts_utc),
            "latent": latent,
            "model_version": model_version,
        }
        return self.enqueue_task("latents", payload)

    def write_tick_async(
        self, symbol: str, timeframe: str, ts_utc: int, tick_count: int
    ) -> bool:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "ts_utc": int(ts_utc),
            "tick_count": int(tick_count),
        }
        return self.enqueue_task("ticks", payload)
