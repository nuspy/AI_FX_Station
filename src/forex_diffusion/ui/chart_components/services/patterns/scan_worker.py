# ui/chart_components/services/patterns/scan_worker.py
# Worker for running pattern scans with dynamic interval adjustment
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot, QTimer
from loguru import logger


class ScanWorker(QObject):
    produced = Signal(list)  # List[PatternEvent]

    def __init__(self, parent, kind: str, interval_ms: int) -> None:
        super().__init__()
        self._parent = parent
        self._kind = kind  # "chart" or "candle"
        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._original_interval = int(interval_ms)  # Store original interval for dynamic adjustment
        self._timer.timeout.connect(self._tick)
        self._enabled = False

    @Slot()
    def start(self):
        self._enabled = True
        self._timer.start()

    @Slot()
    def stop(self):
        self._enabled = False
        self._timer.stop()

    @Slot()
    def _tick(self):
        if not self._enabled:
            return
        try:
            # Check if market is likely closed and adjust interval accordingly
            if hasattr(self._parent, '_is_market_likely_closed'):
                if self._parent._is_market_likely_closed():
                    # If market is closed, increase interval significantly to reduce unnecessary work
                    current_interval = self._timer.interval()
                    market_closed_interval = max(300000, current_interval * 2)  # At least 5 minutes
                    if current_interval < market_closed_interval:
                        self._timer.setInterval(int(market_closed_interval))
                        logger.debug(f"Increased {self._kind} scan interval to {market_closed_interval}ms - market closed")
                    return  # Skip scanning when market is closed
                else:
                    # Market is open, restore normal interval if it was increased
                    original_interval = getattr(self, '_original_interval', self._timer.interval())
                    current_interval = self._timer.interval()
                    if current_interval > original_interval * 2:  # If interval was increased for market closure
                        self._timer.setInterval(original_interval)
                        logger.debug(f"Restored {self._kind} scan interval to {original_interval}ms - market open")

            # Check resource usage and adjust interval dynamically
            if hasattr(self._parent, '_check_resource_limits'):
                if not self._parent._check_resource_limits():
                    # If resources are constrained, increase interval
                    current_interval = self._timer.interval()
                    new_interval = min(current_interval * 1.5, 300000)  # Max 5 minutes
                    self._timer.setInterval(int(new_interval))
                    logger.debug(f"Increased {self._kind} scan interval to {new_interval}ms due to resource constraints")
                    return
                else:
                    # If resources are available, gradually decrease interval back to normal
                    original_interval = getattr(self, '_original_interval', self._timer.interval())
                    current_interval = self._timer.interval()
                    if current_interval > original_interval:
                        new_interval = max(current_interval * 0.9, original_interval)
                        self._timer.setInterval(int(new_interval))

            # Call the parent's _scan_once method from the worker thread
            evs = self._parent._scan_once(kind=self._kind) or []
            self.produced.emit(evs)
        except Exception as e:
            logger.debug(f"Error in scan worker tick ({self._kind}): {e}")