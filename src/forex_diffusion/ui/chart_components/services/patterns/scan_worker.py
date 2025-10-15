# ui/chart_components/services/patterns/scan_worker.py
# Worker for running pattern scans with dynamic interval adjustment
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from loguru import logger

if TYPE_CHECKING:
    from .patterns_service import PatternsService


class ScanWorker(QObject):
    produced = Signal(list)  # List[PatternEvent]
    error_threshold_exceeded = Signal(str, str)  # kind, error_message

    def __init__(self, parent: 'PatternsService', kind: str, interval_ms: int) -> None:
        super().__init__()
        self._parent: PatternsService = parent
        self._kind = kind  # "chart" or "candle"
        self._timer: Optional[QTimer] = None  # Create lazily after moveToThread
        self._interval_ms = int(interval_ms)
        self._original_interval = int(interval_ms)  # Store original interval for dynamic adjustment
        self._enabled = False
        
        # Error tracking and recovery
        self._error_count = 0
        self._max_errors = 5
        self._backoff_multiplier = 1.0

    @Slot()
    def start(self):
        # Create timer on this thread (after moveToThread)
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.setInterval(self._interval_ms)
            self._timer.timeout.connect(self._tick)
        self._enabled = True
        self._timer.start()

    @Slot()
    def stop(self):
        self._enabled = False
        if self._timer is not None:
            self._timer.stop()

    @Slot()
    def _tick(self):
        if not self._enabled or self._timer is None:
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
            
            # Reset error tracking on success
            if self._error_count > 0:
                logger.info(f"{self._kind} scan recovered after {self._error_count} errors")
            self._error_count = 0
            self._backoff_multiplier = 1.0
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in scan worker tick ({self._kind}): {e}, count={self._error_count}/{self._max_errors}")
            
            if self._error_count >= self._max_errors:
                # Stop worker after repeated failures
                logger.critical(f"Stopping {self._kind} scan worker after {self._max_errors} consecutive errors")
                self.stop()
                # Emit signal to notify GUI
                self.error_threshold_exceeded.emit(self._kind, str(e))
            else:
                # Exponential backoff
                self._backoff_multiplier *= 1.5
                new_interval = int(self._original_interval * self._backoff_multiplier)
                if self._timer:
                    self._timer.setInterval(min(new_interval, 300000))  # Max 5 min
                    logger.warning(f"Increased {self._kind} interval to {new_interval}ms due to errors (backoff x{self._backoff_multiplier:.1f})")