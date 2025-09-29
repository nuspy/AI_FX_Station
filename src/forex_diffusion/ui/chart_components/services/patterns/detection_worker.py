# ui/chart_components/services/patterns/detection_worker.py
# Worker for running pattern detection batches in background thread
from __future__ import annotations

import time
from PySide6.QtCore import QObject, Signal, Slot
from loguru import logger


class DetectionWorker(QObject):
    """Worker for running pattern detection batches in background thread"""
    batch_completed = Signal(int, list)  # batch_number, events
    detection_finished = Signal(list)    # all_events
    progress_updated = Signal(int, str)  # percentage, status_message

    def __init__(self):
        super().__init__()
        self._active = False

    @Slot(object, list, int)
    def process_detection_batch(self, df, detectors, batch_size):
        """Process detection in background thread to keep GUI responsive"""
        if self._active:
            return  # Already processing

        self._active = True
        try:
            # Import boundary config
            from .....patterns.boundary_config import get_boundary_config
            boundary_config = get_boundary_config()

            all_events = []
            total_batches = (len(detectors) + batch_size - 1) // batch_size

            # Try to determine current timeframe from dataframe or default to 5m
            timeframe = self._detect_timeframe_from_df(df)

            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(detectors))
                batch_detectors = detectors[start_idx:end_idx]

                # Update progress in GUI thread
                progress_percent = int((batch_num / total_batches) * 100)
                status = f"Scanning patterns... {progress_percent}% ({batch_num + 1}/{total_batches})"
                self.progress_updated.emit(progress_percent, status)

                # Process detectors in this batch
                batch_events = []
                for i, det in enumerate(batch_detectors):
                    try:
                        detector_key = getattr(det, 'key', f'unknown_{i}')

                        start_time = time.time()

                        # Apply boundary-specific dataframe limitation
                        detector_df = self._apply_boundary_to_df(df, detector_key, timeframe, boundary_config)

                        events = det.detect(detector_df)
                        elapsed = time.time() - start_time

                        if elapsed > 1.0:  # Log slow detectors
                            logger.debug(f"Detector {detector_key} took {elapsed:.2f}s")

                        if events:
                            batch_events.extend(events)

                    except Exception as e:
                        detector_key = getattr(det, 'key', f'unknown_{i}')
                        logger.error(f"Detector {detector_key} failed: {e}")

                # Emit batch completion
                self.batch_completed.emit(batch_num, batch_events)
                all_events.extend(batch_events)

                # Allow other threads to run (yield)
                time.sleep(0.001)  # 1ms pause to keep GUI responsive

            # Emit final completion
            self.detection_finished.emit(all_events)

        except Exception as e:
            logger.error(f"Detection worker error: {e}")
            self.detection_finished.emit([])
        finally:
            self._active = False

    def _detect_timeframe_from_df(self, df):
        """Try to detect timeframe from dataframe timestamps"""
        try:
            if len(df) < 2:
                return "5m"  # Default

            # Calculate average interval between timestamps
            timestamps = df.index if hasattr(df.index, 'to_pydatetime') else df['timestamp']
            if len(timestamps) < 2:
                return "5m"

            # Get time difference between first two rows
            time_diff = timestamps[1] - timestamps[0]
            total_seconds = time_diff.total_seconds()

            # Map to standard timeframes
            if total_seconds <= 60:  # <= 1 minute
                return "1m"
            elif total_seconds <= 300:  # <= 5 minutes
                return "5m"
            elif total_seconds <= 900:  # <= 15 minutes
                return "15m"
            elif total_seconds <= 3600:  # <= 1 hour
                return "1h"
            elif total_seconds <= 14400:  # <= 4 hours
                return "4h"
            elif total_seconds <= 86400:  # <= 1 day
                return "1d"
            else:
                return "1w"

        except Exception:
            return "5m"  # Safe fallback

    def _apply_boundary_to_df(self, df, detector_key, timeframe, boundary_config):
        """Apply pattern-specific boundary to limit dataframe"""
        try:
            # Get boundary for this pattern/timeframe
            boundary_candles = boundary_config.get_boundary(detector_key, timeframe)

            # Limit dataframe to last N candles
            if len(df) > boundary_candles:
                limited_df = df.tail(boundary_candles).copy()
                return limited_df
            else:
                return df

        except Exception as e:
            logger.debug(f"Error applying boundary for {detector_key}: {e}")
            return df  # Return original on error