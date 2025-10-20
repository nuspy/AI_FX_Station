"""
Advanced multithread pattern detection system.

Implements parallel detection across multiple worker threads to maximize
performance on high-core systems (up to 32+ threads).
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import queue
import time
import concurrent.futures
from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal, Slot
from loguru import logger
import pandas as pd


@dataclass
class DetectionTask:
    """Single detection task for a worker thread"""
    detector: Any
    detector_key: str
    dataframe: pd.DataFrame
    task_id: int
    batch_id: int


@dataclass
class DetectionResult:
    """Result from a detection task"""
    task_id: int
    batch_id: int
    detector_key: str
    events: List[Any]
    execution_time: float
    error: Optional[str] = None


class MultithreadDetectionWorker(QObject):
    """
    Advanced multithread detection worker that distributes pattern detection
    across multiple parallel threads for maximum performance.
    """

    # Signals for thread-safe communication
    progress_updated = Signal(int, str)  # percentage, status_message
    batch_completed = Signal(int, list, float)  # batch_id, events, execution_time
    detection_finished = Signal(list, dict)  # all_events, performance_stats
    error_occurred = Signal(str)  # error_message

    def __init__(self, max_workers: int = 8):
        super().__init__()
        self.max_workers = max_workers
        self._active = False
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._results_queue = queue.Queue()
        self._performance_stats = {
            'total_detectors': 0,
            'total_execution_time': 0.0,
            'successful_detectors': 0,
            'failed_detectors': 0,
            'events_found': 0,
            'parallel_efficiency': 0.0
        }

    @Slot(object, list, int)
    def process_detection_multithread(self, df: pd.DataFrame, detectors: List[Any], max_workers: int):
        """
        Process detection using multiple parallel threads.

        Args:
            df: Source dataframe
            detectors: List of pattern detectors
            max_workers: Number of parallel threads to use
        """
        if self._active:
            logger.warning("Detection already in progress, ignoring new request")
            return

        self._active = True
        self.max_workers = min(max_workers, len(detectors))  # Don't use more threads than detectors

        try:
            # Import boundary config
            from ....patterns.boundary_config import get_boundary_config
            boundary_config = get_boundary_config()

            # Detect timeframe
            timeframe = self._detect_timeframe_from_df(df)

            start_time = time.time()
            logger.info(f"Starting multithread detection: {len(detectors)} detectors on {self.max_workers} threads")

            # Initialize performance stats
            self._performance_stats = {
                'total_detectors': len(detectors),
                'total_execution_time': 0.0,
                'successful_detectors': 0,
                'failed_detectors': 0,
                'events_found': 0,
                'parallel_efficiency': 0.0,
                'threads_used': self.max_workers
            }

            # Create detection tasks with boundary-limited dataframes
            tasks = []
            for i, detector in enumerate(detectors):
                detector_key = getattr(detector, 'key', f'detector_{i}')

                # Apply boundary-specific dataframe limitation
                detector_df = self._apply_boundary_to_df(df, detector_key, timeframe, boundary_config)

                task = DetectionTask(
                    detector=detector,
                    detector_key=detector_key,
                    dataframe=detector_df,
                    task_id=i,
                    batch_id=i // self.max_workers  # Group tasks into batches
                )
                tasks.append(task)

            # Process tasks in parallel using ThreadPoolExecutor
            self._process_tasks_parallel(tasks)

            # Calculate final performance stats
            total_time = time.time() - start_time
            self._performance_stats['total_execution_time'] = total_time

            # Calculate parallel efficiency (ideal vs actual time)
            avg_detector_time = total_time / len(detectors) if len(detectors) > 0 else 0
            sequential_time = avg_detector_time * len(detectors)
            self._performance_stats['parallel_efficiency'] = (sequential_time / total_time) if total_time > 0 else 0

            logger.info(f"Multithread detection completed in {total_time:.2f}s with {self.max_workers} threads")

        except Exception as e:
            logger.error(f"Multithread detection error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self._active = False

    def _process_tasks_parallel(self, tasks: List[DetectionTask]):
        """Process detection tasks in parallel using thread pool"""

        # Group tasks into batches for progress reporting
        batch_groups = {}
        for task in tasks:
            if task.batch_id not in batch_groups:
                batch_groups[task.batch_id] = []
            batch_groups[task.batch_id].append(task)

        total_batches = len(batch_groups)
        completed_batches = 0
        all_events = []

        # Create thread pool executor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self._executor = executor

            # Submit all tasks to thread pool
            future_to_task = {}
            for task in tasks:
                future = executor.submit(self._execute_detection_task, task)
                future_to_task[future] = task

            # Process results as they complete
            batch_results = {}

            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    result = future.result()

                    # Update performance stats
                    if result.error:
                        self._performance_stats['failed_detectors'] += 1
                    else:
                        self._performance_stats['successful_detectors'] += 1
                        self._performance_stats['events_found'] += len(result.events)
                        if result.events:
                            all_events.extend(result.events)

                    # Group results by batch
                    if result.batch_id not in batch_results:
                        batch_results[result.batch_id] = []
                    batch_results[result.batch_id].append(result)

                    # Check if batch is complete
                    if len(batch_results[result.batch_id]) == len(batch_groups[result.batch_id]):
                        # Batch completed
                        completed_batches += 1
                        batch_events = []
                        batch_time = 0.0

                        for batch_result in batch_results[result.batch_id]:
                            if not batch_result.error and batch_result.events:
                                batch_events.extend(batch_result.events)
                            batch_time += batch_result.execution_time

                        # Emit batch completion
                        self.batch_completed.emit(result.batch_id, batch_events, batch_time)

                        # Update progress
                        progress = int((completed_batches / total_batches) * 100)
                        status = f"Parallel detection: {progress}% ({completed_batches}/{total_batches} batches, {self.max_workers} threads)"
                        self.progress_updated.emit(progress, status)

                except Exception as e:
                    logger.error(f"Task execution error for {task.detector_key}: {e}")
                    self._performance_stats['failed_detectors'] += 1

        # Emit final completion
        self.detection_finished.emit(all_events, self._performance_stats)

    def _execute_detection_task(self, task: DetectionTask) -> DetectionResult:
        """Execute a single detection task in worker thread"""
        start_time = time.time()

        try:
            # Execute detector
            events = task.detector.detect(task.dataframe)
            execution_time = time.time() - start_time

            return DetectionResult(
                task_id=task.task_id,
                batch_id=task.batch_id,
                detector_key=task.detector_key,
                events=events or [],
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Detector {task.detector_key} failed: {str(e)}"
            logger.debug(error_msg)

            return DetectionResult(
                task_id=task.task_id,
                batch_id=task.batch_id,
                detector_key=task.detector_key,
                events=[],
                execution_time=execution_time,
                error=error_msg
            )

    def _detect_timeframe_from_df(self, df: pd.DataFrame) -> str:
        """Detect timeframe from dataframe timestamps"""
        try:
            if len(df) < 2:
                return "5m"

            # Get timestamps
            if hasattr(df.index, 'to_pydatetime'):
                timestamps = df.index
            elif 'timestamp' in df.columns:
                timestamps = df['timestamp']
            elif 'ts_utc' in df.columns:
                timestamps = df['ts_utc']
            else:
                return "5m"

            if len(timestamps) < 2:
                return "5m"

            # Calculate time difference
            time_diff = timestamps[1] - timestamps[0]
            total_seconds = time_diff.total_seconds()

            # Map to timeframes
            if total_seconds <= 60:
                return "1m"
            elif total_seconds <= 300:
                return "5m"
            elif total_seconds <= 900:
                return "15m"
            elif total_seconds <= 3600:
                return "1h"
            elif total_seconds <= 14400:
                return "4h"
            elif total_seconds <= 86400:
                return "1d"
            else:
                return "1w"

        except Exception:
            return "5m"

    def _apply_boundary_to_df(self, df: pd.DataFrame, detector_key: str, timeframe: str, boundary_config) -> pd.DataFrame:
        """Apply pattern-specific boundary to limit dataframe"""
        try:
            boundary_candles = boundary_config.get_boundary(detector_key, timeframe)

            if len(df) > boundary_candles:
                return df.tail(boundary_candles).copy()
            else:
                return df

        except Exception as e:
            logger.debug(f"Error applying boundary for {detector_key}: {e}")
            return df

    def stop_detection(self):
        """Stop detection if running"""
        if self._active and self._executor:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
        self._active = False

    def start_detection(self, detection_tasks: List[Dict]):
        """
        Start detection with a list of task dictionaries.

        Args:
            detection_tasks: List of dicts with keys: 'detector', 'dataframe', 'detector_key', optional 'historical'
        """
        if self._active:
            logger.warning("Detection already in progress, ignoring new request")
            return

        try:
            # Extract detectors and dataframes from tasks
            detectors = []
            dataframes = []

            for task in detection_tasks:
                detector = task.get('detector')
                dataframe = task.get('dataframe')

                if detector is not None and dataframe is not None:
                    detectors.append(detector)
                    # For now, use the first dataframe as the base
                    # In the future, we could optimize this to handle different dataframes per detector
                    if len(dataframes) == 0:
                        dataframes = [dataframe]

            if detectors and dataframes:
                # Use the existing process_detection_multithread method
                self.process_detection_multithread(dataframes[0], detectors, self.max_workers)
            else:
                logger.warning("No valid detection tasks provided")

        except Exception as e:
            logger.error(f"Error starting detection: {e}")
            self.error_occurred.emit(str(e))


class HistoricalDetectionWorker(MultithreadDetectionWorker):
    """
    Specialized worker for historical pattern detection.
    Handles larger datasets and longer time ranges with optimized chunking.
    """

    def __init__(self, max_workers: int = 8):
        super().__init__(max_workers)
        self.chunk_size = 5000  # Process in chunks for historical data

    @Slot(object, list, int, str, str)
    def process_historical_detection(self, df: pd.DataFrame, detectors: List[Any],
                                   max_workers: int, start_time: str, end_time: str):
        """
        Process historical detection with time range filtering and chunking.

        Args:
            df: Source dataframe
            detectors: List of detectors
            max_workers: Number of threads
            start_time: Historical start time (e.g., "30d")
            end_time: Historical end time (e.g., "7d")
        """

        # Filter dataframe to historical range
        filtered_df = self._filter_historical_range(df, start_time, end_time)

        if filtered_df.empty:
            logger.warning("No historical data in specified range")
            self.detection_finished.emit([], {'error': 'No historical data'})
            return

        logger.info(f"Processing historical detection: {len(filtered_df)} rows, {len(detectors)} detectors")

        # Use standard multithread detection on filtered data
        self.process_detection_multithread(filtered_df, detectors, max_workers)

    def _filter_historical_range(self, df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
        """Filter dataframe to historical time range"""
        try:
            from datetime import datetime, timedelta

            # Parse time strings (e.g., "30d", "7d")
            def parse_time_delta(time_str: str) -> timedelta:
                time_str = time_str.strip().lower()
                if time_str.endswith('d'):
                    return timedelta(days=int(time_str[:-1]))
                elif time_str.endswith('h'):
                    return timedelta(hours=int(time_str[:-1]))
                elif time_str.endswith('m'):
                    return timedelta(minutes=int(time_str[:-1]))
                else:
                    return timedelta(days=int(time_str))

            now = datetime.now()
            start_delta = parse_time_delta(start_time)
            end_delta = parse_time_delta(end_time)

            start_date = now - start_delta
            end_date = now - end_delta

            # Filter dataframe
            if hasattr(df.index, 'to_pydatetime'):
                mask = (df.index >= start_date) & (df.index <= end_date)
            elif 'timestamp' in df.columns:
                df_ts = pd.to_datetime(df['timestamp'])
                mask = (df_ts >= start_date) & (df_ts <= end_date)
            elif 'ts_utc' in df.columns:
                df_ts = pd.to_datetime(df['ts_utc'])
                mask = (df_ts >= start_date) & (df_ts <= end_date)
            else:
                logger.warning("No timestamp column found for historical filtering")
                return df

            return df[mask].copy()

        except Exception as e:
            logger.error(f"Error filtering historical range: {e}")
            return df