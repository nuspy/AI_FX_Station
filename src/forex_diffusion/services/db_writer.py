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
                # logger.critical(f"--- DBWRITER DEQUEUED TASK: {task_type} ---")

                handler_name = f"write_{task_type}"
                handler = getattr(self.db_service, handler_name, None)

                if handler and callable(handler):
                    try:
                        # logger.debug(f"Executing DB task: {task_type}")
                        handler(payload)
                    except Exception as e:
                        logger.exception(f"Error executing DB task '{task_type}' with payload {payload}. Error: {e}")
                else:
                    logger.warning(f"No handler '{handler_name}' found in DBService for task type '{task_type}'.")
                
                self._task_queue.task_done()

            except queue.Empty:
                continue
            except Exception:
                logger.exception("Exception in DBWriter worker thread.")
        
        logger.info("DBWriter worker thread stopped.")

    def start(self):
        """
        Starts the DBWriter worker thread.
        """
        if self._is_running:
            return

        logger.info("Starting DBWriter...")
        self._is_running = True
        self._worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self._worker_thread.start()

    def stop(self, wait: bool = True):
        """
        Stops the DBWriter worker thread.
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
            logger.warning("DBWriter is not running. Task not enqueued.")
            return False
        
        self._task_queue.put((task_type, payload))
        return True

    def write_prediction_async(self, payload: Dict[str, Any]) -> bool:
        return self.enqueue_task("prediction", payload)

    def write_signal_async(self, payload: Dict[str, Any]) -> bool:
        return self.enqueue_task("signal", payload)

    def write_calibration_async(self, payload: Dict[str, Any]) -> bool:
        return self.enqueue_task("calibration", payload)

    def write_features_async(self, payload: Dict[str, Any]) -> bool:
        return self.enqueue_task("features", payload)

    def write_latents_async(self, payload: Dict[str, Any]) -> bool:
        return self.enqueue_task("latents", payload)

    def write_tick_async(self, payload: Dict[str, Any]) -> bool:
        return self.enqueue_task("tick", payload)
