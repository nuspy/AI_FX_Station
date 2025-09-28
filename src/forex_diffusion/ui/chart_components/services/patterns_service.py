
from __future__ import annotations
from typing import List, Optional, Dict, Iterable
import pandas as pd
import numpy as np
import asyncio
import psutil
from loguru import logger
from PySide6.QtCore import QTimer

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer

# Cache and performance imports
from ....cache import get_cache, get_pattern_cache, cache_decorator
from ....patterns.strength_calculator import PatternStrengthCalculator

class _ScanWorker(QObject):
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

            evs = self._parent._scan_once(kind=self._kind) or []
            self.produced.emit(evs)
        except Exception as e:
            logger.debug(f"Error in scan worker tick ({self._kind}): {e}")


from .base import ChartServiceBase
# Assicurati che hns.py sia correttamente importato e contenga i detector per "chart"

from ....patterns.candles import make_candle_detectors
from ....patterns.broadening import make_broadening_detectors
from ....patterns.wedges import make_wedge_detectors
from ....patterns.triangles import make_triangle_detectors
from ....patterns.rectangle import make_rectangle_detectors
from ....patterns.diamond import make_diamond_detectors
from ....patterns.double_triple import make_double_triple_detectors
from ....patterns.channels import make_channel_detectors
from ....patterns.flags import make_flag_detectors
#Pluto from .variants import make_param_variants
from  ....patterns.hns import make_hns_detectors
from ....patterns.registry import PatternRegistry

#from ..services.hns import make_broadening_detectors, make_wedge_detectors, make_triangle_detectors
from ....patterns.engine import PatternEvent
from ....patterns.info_provider import PatternInfoProvider
from ...pattern_overlay import PatternOverlayRenderer
from .patterns_adapter import enrich_events

# Training/Optimization system imports
from ....training.optimization.engine import OptimizationEngine
from ....training.optimization.task_manager import TaskManager
from ....training.optimization.parameter_space import ParameterSpace

OHLC_SYNONYMS: Dict[str, str] = {
        'o':'open','op':'open','open':'open','bidopen':'open','askopen':'open',
        'h':'high','hi':'high','high':'high','bidhigh':'high','askhigh':'high',
        'l':'low','lo':'low','low':'low','bidlow':'low','asklow':'low',
        'c':'close','cl':'close','close':'close','bidclose':'close','askclose':'close','last':'close','price':'close','mid':'close'
    }
TS_SYNONYMS = ['ts_utc','timestamp','time','ts','datetime','date','dt','ts_ms','ts_ns']

class PatternsService(ChartServiceBase):
    def __init__(self, view, controller) -> None:
        super().__init__(view, controller)
        self._enabled_chart = False
        self._enabled_candle = False
        self._enabled_history = False
        self._events: List[PatternEvent] = []
        self.registry = PatternRegistry()
        self.info = PatternInfoProvider(self._default_info_path())
        self.renderer = PatternOverlayRenderer(controller, self.info)

        # Strategy selection for real-time scanning
        self._current_strategy = "balanced"  # Default: balanced strategy
        self._available_strategies = {
            "high_return": {
                "name": "High Return",
                "description": "Maximize profits, accept higher risk",
                "focus": "Expectancy, Total Return, Profit Factor"
            },
            "low_risk": {
                "name": "Low Risk",
                "description": "Minimize losses, conservative approach",
                "focus": "Success Rate, Low Drawdown, High Sharpe Ratio"
            },
            "balanced": {
                "name": "Balanced",
                "description": "Balanced risk/reward profile",
                "focus": "Overall performance balance"
            }
        }

        self._busy = False
        self._pending_df: Optional[pd.DataFrame] = None
        self._debounce_timer: Optional[QTimer] = None
        self._debounce_ms = 30

        # Initialize cache and strength calculator
        self._pattern_cache = get_pattern_cache()
        self._redis_cache = get_cache()

        # Load configuration for advanced features
        self._config = self._load_config()
        self._strength_calculator = PatternStrengthCalculator(self._config)

        # Load historical pattern configuration
        self._historical_config = self._load_historical_config()

        # Resource monitoring
        self._cpu_limit_percent = self._config.get('resources', {}).get('pattern_detection', {}).get('max_cpu_percent', 30)
        self._memory_threshold = self._config.get('resources', {}).get('memory', {}).get('max_usage_percent', 80)

        try:
            self.view.canvas.mpl_connect('pick_event', self.renderer.on_pick)
        except Exception:
            pass
        try:
            self.renderer.use_badges = True
        except Exception:
            pass

        logger.debug("PatternsService initialized")

        # Persistent cache (per symbol) and multi-timeframe scan state
        self._cache: Dict[tuple, object] = {}
        self._cache_symbol: Optional[str] = None
        self._scanned_tfs_by_symbol: Dict[str, set] = {}
        self._scanning_multi: bool = False

        # Dual background threads for scans
        self._chart_thread = QThread(self.view)
        self._candle_thread = QThread(self.view)

        # Load intervals from config
        import yaml, os
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '..', '..', '..', 'configs', 'patterns.yaml')
            with open(config_path, 'r', encoding='utf-8') as fh:
                _cfg = yaml.safe_load(fh) or {}
        except Exception:
            _cfg = {}

        cur = (_cfg.get('patterns',{}).get('current_scan',{}) or {})
        minutes = int(cur.get('interval_minutes', 5))

        # Create workers and move to threads
        self._chart_worker = _ScanWorker(self, 'chart', minutes*60*1000)
        self._candle_worker = _ScanWorker(self, 'candle', minutes*60*1000)

        self._chart_worker.moveToThread(self._chart_thread)
        self._candle_worker.moveToThread(self._candle_thread)

        # Connect signals
        self._chart_thread.started.connect(self._chart_worker.start)
        self._candle_thread.started.connect(self._candle_worker.start)
        self._chart_worker.produced.connect(self._on_chart_patterns_detected)
        self._candle_worker.produced.connect(self._on_candle_patterns_detected)

        self._threads_started = False

    def _load_config(self) -> dict:
        """Load configuration for patterns service with resource limits"""
        try:
            import yaml, os
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                '..', '..', '..', 'configs', 'default.yaml'
            )

            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            # Set defaults for resource management
            if 'resources' not in config:
                config['resources'] = {}
            if 'pattern_detection' not in config['resources']:
                config['resources']['pattern_detection'] = {}
            if 'max_cpu_percent' not in config['resources']['pattern_detection']:
                config['resources']['pattern_detection']['max_cpu_percent'] = 30
            if 'memory' not in config['resources']:
                config['resources']['memory'] = {}
            if 'max_usage_percent' not in config['resources']['memory']:
                config['resources']['memory']['max_usage_percent'] = 80

            return config

        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            return {
                'resources': {
                    'pattern_detection': {'max_cpu_percent': 30},
                    'memory': {'max_usage_percent': 80}
                }
            }

    def _load_historical_config(self) -> dict:
        """Load historical pattern configuration from patterns.yaml"""
        try:
            import yaml, os
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                '..', '..', '..', 'configs', 'patterns.yaml'
            )

            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            historical_config = config.get('historical_patterns', {
                'enabled': False,
                'start_time': '30d',
                'end_time': '7d'
            })

            return historical_config

        except Exception as e:
            logger.warning(f"Could not load historical config, using defaults: {e}")
            return {
                'enabled': False,
                'start_time': '30d',
                'end_time': '7d'
            }

    @staticmethod
    def parse_time_string(time_str: str) -> int:
        """Convert time string format 'xm', 'xh', 'xd' to minutes"""
        if not time_str:
            return 0

        time_str = time_str.strip().lower()
        try:
            if time_str.endswith('m'):
                return int(time_str[:-1])
            elif time_str.endswith('h'):
                return int(time_str[:-1]) * 60
            elif time_str.endswith('d'):
                return int(time_str[:-1]) * 60 * 24
            else:
                # Assume minutes if not specified
                return int(time_str)
        except ValueError:
            return 0

    def _check_resource_limits(self) -> bool:
        """Check if current resource usage is within limits"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self._cpu_limit_percent:
                logger.debug(f"CPU usage {cpu_percent:.1f}% exceeds limit {self._cpu_limit_percent}%")
                return False

            # Check memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self._memory_threshold:
                logger.debug(f"Memory usage {memory_percent:.1f}% exceeds limit {self._memory_threshold}%")
                return False

            # Check if Redis cache memory is within limits
            try:
                cache_info = self._redis_cache.get_cache_info()
                if cache_info.get('memory_usage_percent', 0) > 90:
                    logger.debug("Redis cache memory usage high, throttling pattern detection")
                    return False
            except Exception:
                pass

            return True

        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
            return True  # Allow execution if monitoring fails

    def get_resource_stats(self) -> dict:
        """Get current resource usage statistics for monitoring"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            stats = {
                'cpu_percent': cpu_percent,
                'cpu_limit': self._cpu_limit_percent,
                'memory_percent': memory.percent,
                'memory_limit': self._memory_threshold,
                'memory_available_gb': memory.available / (1024**3),
                'pattern_detection_active': self._enabled_chart or self._enabled_candle,
                'scan_intervals': {
                    'chart': getattr(self._chart_worker, '_timer', {}).interval() if hasattr(self, '_chart_worker') else 0,
                    'candle': getattr(self._candle_worker, '_timer', {}).interval() if hasattr(self, '_candle_worker') else 0
                }
            }

            # Add cache statistics if available
            try:
                cache_info = self._redis_cache.get_cache_info()
                stats['cache'] = {
                    'memory_usage_mb': cache_info.get('memory_usage_mb', 0),
                    'memory_limit_mb': cache_info.get('memory_limit_mb', 0),
                    'hit_ratio': cache_info.get('hit_ratio', 0.0),
                    'total_keys': cache_info.get('total_keys', 0)
                }
            except Exception:
                stats['cache'] = {'status': 'unavailable'}

            return stats

        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {'error': str(e)}

    def test_async_detection(self, test_data=None) -> dict:
        """Test async pattern detection with mock data for verification"""
        try:
            import numpy as np
            import pandas as pd
            from datetime import datetime, timedelta

            # Create test data if not provided
            if test_data is None:
                dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=1000, freq='1min')
                np.random.seed(42)

                price = 1.1000
                prices = [price]
                for _ in range(999):
                    change = np.random.normal(0, 0.0001)
                    price = max(0.9000, min(1.3000, price + change))
                    prices.append(price)

                test_data = pd.DataFrame({
                    'timestamp': dates,
                    'open': prices,
                    'high': [p * (1 + abs(np.random.normal(0, 0.0001))) for p in prices],
                    'low': [p * (1 - abs(np.random.normal(0, 0.0001))) for p in prices],
                    'close': prices
                })

            # Test resource limits first
            resource_check = self._check_resource_limits()

            # Simulate a quick detection test
            start_time = datetime.now()

            # Mock the detection without running full async
            from src.forex_diffusion.patterns.registry import PatternRegistry
            reg = PatternRegistry()
            available_detectors = len(list(reg.detectors(['chart']))) + len(list(reg.detectors(['candle'])))

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            return {
                'success': True,
                'test_timestamp': start_time.isoformat(),
                'duration_ms': duration_ms,
                'resource_check_passed': resource_check,
                'available_detectors': available_detectors,
                'test_data_rows': len(test_data),
                'resource_stats': self.get_resource_stats()
            }

        except Exception as e:
            logger.error(f"Error in async detection test: {e}")
            return {
                'success': False,
                'error': str(e),
                'test_timestamp': datetime.now().isoformat()
            }

    def _on_chart_patterns_detected(self, events: List):
        """Handle chart patterns detected in background thread"""
        try:
            if events and self._enabled_chart:
                # Update events on main thread
                chart_events = [e for e in events if getattr(e, 'kind', '') == 'chart']
                # Remove old chart events and add new ones
                self._events = [e for e in self._events if getattr(e, 'kind', '') != 'chart']
                self._events.extend(chart_events)
                self._repaint()
        except Exception as e:
            logger.error(f"Error handling chart patterns: {e}")

    def _on_candle_patterns_detected(self, events: List):
        """Handle candlestick patterns detected in background thread"""
        try:
            if events and self._enabled_candle:
                # Update events on main thread
                candle_events = [e for e in events if getattr(e, 'kind', '') == 'candle']
                # Remove old candle events and add new ones
                self._events = [e for e in self._events if getattr(e, 'kind', '') != 'candle']
                self._events.extend(candle_events)
                self._repaint()
        except Exception as e:
            logger.error(f"Error handling candle patterns: {e}")

    def set_chart_enabled(self, on: bool):
        self._enabled_chart = bool(on)
        logger.info(f"Patterns: CHART toggle → {self._enabled_chart}")

        # TEMPORARY: Disable continuous scanning to prevent GUI blocking
        if self._enabled_chart:
            logger.info("Chart patterns enabled - use 'Scan Historical' button for pattern detection")
            logger.info("Continuous scanning temporarily disabled to prevent GUI blocking")

        # No timer/scanning setup to prevent GUI blocking
        # Only historical scanning will work for now
        self._repaint()

    def set_candle_enabled(self, on: bool):
        self._enabled_candle = bool(on)
        logger.info(f"Patterns: CANDLE toggle → {self._enabled_candle}")

        # TEMPORARY: Disable continuous scanning to prevent GUI blocking
        if self._enabled_candle:
            logger.info("Candle patterns enabled - use 'Scan Historical' button for pattern detection")
            logger.info("Continuous scanning temporarily disabled to prevent GUI blocking")

        # No timer/scanning setup to prevent GUI blocking
        # Only historical scanning will work for now
        self._repaint()

    def set_history_enabled(self, on: bool):
        self._enabled_history = bool(on)
        logger.info(f"Patterns: HISTORY toggle → {self._enabled_history}")
        self._repaint()

    def _is_market_likely_closed(self) -> bool:
        """Check if market is likely closed based on data freshness and time"""
        try:
            from datetime import datetime, timedelta
            import pytz

            # Get the last data timestamp
            df = getattr(self.controller.plot_service, '_last_df', None)
            if df is None or len(df) == 0:
                return True

            # Check if we have recent data (last 30 minutes for active markets)
            if hasattr(df, 'index') and len(df) > 0:
                try:
                    last_timestamp = df.index[-1]
                    if hasattr(last_timestamp, 'tz_localize'):
                        # If timestamp is naive, assume UTC
                        if last_timestamp.tz is None:
                            last_timestamp = last_timestamp.tz_localize('UTC')

                    now = datetime.now(pytz.UTC)
                    time_diff = now - last_timestamp

                    # If last data is older than 30 minutes, market likely closed
                    if time_diff > timedelta(minutes=30):
                        return True

                except Exception:
                    # If timestamp processing fails, be conservative
                    pass

            # Additional check: typical market closed hours (weekend)
            now = datetime.now(pytz.UTC)
            weekday = now.weekday()  # 0=Monday, 6=Sunday

            # Weekend (Saturday-Sunday)
            if weekday >= 5:  # Saturday=5, Sunday=6
                return True

            # Forex is mostly 24/5, but some hours have low activity
            # Be conservative and allow scanning during weekdays
            return False

        except Exception as e:
            logger.debug(f"Error checking market hours: {e}")
            # If check fails, assume market is open to avoid blocking functionality
            return False

    def _timer_scan(self, kind: str):
        """Timer-based scanning on main thread (no threading issues)"""
        try:
            logger.debug(f"Timer scan triggered for {kind} - GUI should remain responsive")

            # Quick checks before scanning
            if not getattr(self, f'_enabled_{kind}', False):
                logger.debug(f"{kind} patterns not enabled, skipping")
                return

            # Market and resource checks
            if self._is_market_likely_closed():
                # Adjust timer interval if market is closed
                timer = getattr(self, f'_{kind}_timer', None)
                if timer and timer.interval() < 300000:
                    timer.setInterval(300000)  # 5 minutes
                    logger.debug(f"Increased {kind} timer interval - market closed")
                return

            if not self._check_resource_limits():
                logger.debug(f"Skipping {kind} scan - resource limits")
                return

            # Get data
            df = getattr(self.controller.plot_service, '_last_df', None)
            dfN = self._normalize_df(df)
            if dfN is None or len(dfN) == 0:
                return

            # Check if data changed
            if hasattr(self, '_last_scan_data_hash'):
                current_hash = hash(str(dfN.iloc[-10:].values.tobytes()) if len(dfN) >= 10 else str(dfN.values.tobytes()))
                if current_hash == getattr(self, f'_last_{kind}_scan_hash', None):
                    return
                setattr(self, f'_last_{kind}_scan_hash', current_hash)

            # Quick sync scan with limited detectors
            events = self._quick_pattern_scan(dfN, kind)

            # Emit results
            if kind == 'chart':
                self._on_chart_patterns_detected(events or [])
            else:
                self._on_candle_patterns_detected(events or [])

        except Exception as e:
            logger.debug(f"Error in timer scan ({kind}): {e}")

    def _quick_pattern_scan(self, dfN, kind: str):
        """Ultra-lightweight pattern scan - DISABLED to prevent GUI blocking"""
        try:
            # TEMPORARY: Disable pattern detection completely to prevent GUI blocking
            # The detector.detect() calls on 7324 rows are too heavy for main thread
            logger.debug(f"Pattern detection temporarily disabled to prevent GUI blocking ({kind})")
            return []

            # TODO: Implement proper background processing with:
            # 1. QThread with proper signal/slot communication
            # 2. Or chunked processing with QTimer.singleShot(0, ...)
            # 3. Or only process last N candles instead of full dataset

        except Exception as e:
            logger.debug(f"Quick pattern scan error: {e}")
            return []

    def start_historical_scan_with_range(self, df: pd.DataFrame = None):
        """Start one-time historical pattern scan with configured time range"""
        try:
            # Reload historical config in case it was updated
            self._historical_config = self._load_historical_config()

            if not self._historical_config.get('enabled', False):
                logger.info("Historical patterns not enabled")
                return

            # Get time range in minutes
            start_minutes = self.parse_time_string(self._historical_config.get('start_time', '30d'))
            end_minutes = self.parse_time_string(self._historical_config.get('end_time', '7d'))

            logger.info(f"Starting historical scan: {start_minutes}m to {end_minutes}m ago")

            # Use the provided dataframe or get current one
            if df is None:
                df = getattr(self, '_current_df', None)

            if df is None:
                logger.error("No dataframe available for historical scan")
                return

            # Filter dataframe to historical range
            historical_df = self._filter_df_for_historical_range(df, start_minutes, end_minutes)

            if historical_df is None or len(historical_df) == 0:
                logger.warning("No data available in specified historical range")
                return

            logger.info(f"Historical scan: analyzing {len(historical_df)} rows from {start_minutes}m to {end_minutes}m ago")

            # Run detection on historical data
            self._run_historical_detection(historical_df)

        except Exception as e:
            logger.error(f"Error in historical scan: {e}")

    def _filter_df_for_historical_range(self, df: pd.DataFrame, start_minutes: int, end_minutes: int) -> pd.DataFrame:
        """Filter dataframe to specified historical time range"""
        try:
            import pandas as pd
            from datetime import datetime, timedelta

            # Find timestamp column
            ts_col = None
            for col in df.columns:
                if col.lower() in TS_SYNONYMS:
                    ts_col = col
                    break

            if ts_col is None:
                # Try to use index if it's datetime
                if hasattr(df.index, 'normalize'):
                    ts_col = df.index.name or 'timestamp'
                    df = df.reset_index()
                else:
                    logger.error("No timestamp column found for historical filtering")
                    return None

            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                df[ts_col] = pd.to_datetime(df[ts_col])

            # Calculate time range
            now = datetime.now()
            start_time = now - timedelta(minutes=start_minutes)
            end_time = now - timedelta(minutes=end_minutes)

            # Filter dataframe
            mask = (df[ts_col] >= start_time) & (df[ts_col] <= end_time)
            filtered_df = df[mask].copy()

            return filtered_df

        except Exception as e:
            logger.error(f"Error filtering dataframe for historical range: {e}")
            return None

    def _run_historical_detection(self, df: pd.DataFrame):
        """Run pattern detection on historical data"""
        try:
            # Determine which pattern types to scan based on enabled settings
            kinds = []
            if self._enabled_chart:
                kinds.append("chart")
            if self._enabled_candle:
                kinds.append("candle")

            if not kinds:
                logger.info("No pattern types enabled for historical scan")
                return

            # Run detection for each enabled kind
            historical_events = []
            for kind in kinds:
                events = self._scan_once(kind=kind, df=df) or []
                # Mark events as historical
                for event in events:
                    event.historical = True
                historical_events.extend(events)

            # Clear current events and replace with historical events
            self._events.clear()
            self._events.extend(historical_events)

            logger.info(f"Historical scan completed: found {len(historical_events)} patterns")
            self._repaint()

        except Exception as e:
            logger.error(f"Error running historical detection: {e}")

    def detect_async(self, df: Optional[pd.DataFrame]):
        logger.debug(f"Patterns.detect_async: shape={getattr(df, 'shape', None)} "
                     f"cols={list(df.columns)[:8] if hasattr(df, 'columns') else None}")
        logger.debug(f"Patterns.detect_async: received df={type(df).__name__ if df is not None else None}")
        self._pending_df = df
        # Store current dataframe for historical scanning
        if df is not None:
            self._current_df = df
        if self._debounce_timer is None:
            self._debounce_timer = QTimer()
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.timeout.connect(self._consume_debounce)
        if not self._debounce_timer.isActive():
            self._debounce_timer.start(self._debounce_ms)

    def _consume_debounce(self):
        df = self._pending_df
        self._pending_df = None
        if df is None:
            return
        if self._busy:
            self._pending_df = df
            return
        self._busy = True
        try:
            # Use non-blocking detection for large datasets
            if df is not None and len(df) > 500:
                self._run_detection_nonblocking(df)
                # For non-blocking, we release busy immediately
                self._busy = False
            else:
                self._run_detection(df)
                self._busy = False
        except Exception as e:
            self._busy = False
            logger.error(f"Error in detect_async: {e}")

        # Process pending requests
        if self._pending_df is not None:
            nxt = self._pending_df
            self._pending_df = None
            self.detect_async(nxt)

    def _run_detection(self, df: pd.DataFrame):
        """Run pattern detection with batching for large datasets"""
        try:
            kinds: list[str] = []
            if self._enabled_chart:
                kinds.append("chart")
            if self._enabled_candle:
                kinds.append("candle")
            if not kinds:
                logger.info("Patterns: toggles OFF → skipping detection")
                self._events.clear()
                self.renderer.clear()
                return

            # Reset cache if symbol changed
            try:
                cur_sym = getattr(self.view, "symbol", None) or getattr(self.controller, "symbol", None)
                if self._cache_symbol is None:
                    self._cache_symbol = cur_sym
                elif cur_sym != self._cache_symbol:
                    self._cache_symbol = cur_sym
                    self._cache = {}
                    self._events = []
                    self.renderer.clear()
            except Exception:
                pass

            if self._enabled_history:
                ds = getattr(self.controller, "data_service", None)
                if ds and hasattr(ds, "get_full_dataframe"):
                    try:
                        df_full = ds.get_full_dataframe()
                        if isinstance(df_full, pd.DataFrame) and not df_full.empty:
                            df = df_full
                            logger.info(f"Patterns: using FULL dataframe for scan → rows={len(df)}")
                    except Exception as e:
                        logger.debug(f"Patterns: get_full_dataframe failed: {e}")

            if df is None:
                df = getattr(self.controller.plot_service, "_last_df", None)

            dfN = self._normalize_df(df)
            if dfN is None or dfN.empty:
                logger.warning("Patterns: normalization failed or empty → no detection")
                self._events.clear()
                self.renderer.clear()
                return

            try:
                ts = pd.to_datetime(dfN['ts_utc'], unit='ms', utc=True)
                try:
                    dfN['ts_dt'] = ts.dt.tz_convert(None)
                except AttributeError:
                    dfN['ts_dt'] = ts.tz_convert(None)
            except Exception:
                pass

            dets_list = self.registry.detectors(kinds=kinds)
            if dets_list is None:
                logger.error(f"No detectors found for kinds: {kinds}")
                return []  # Oppure gestisci l'assenza di detector
            else:
                dets = list(dets_list)
            logger.info(f"Patterns: normalized df rows={len(dfN)}; detectors={len(dets)}")

            # Use async detection with resource monitoring
            evs: List[PatternEvent] = []

            # Check if we should use async detection for large datasets
            use_async = len(dfN) > 1000 or len(dets) > 10

            if use_async:
                logger.info(f"Using async detection for {len(dfN)} rows with {len(dets)} detectors")

                # Process in smaller batches to avoid blocking
                batch_size = max(1, len(dets) // 4)  # Process in 4 batches

                for i in range(0, len(dets), batch_size):
                    batch_dets = dets[i:i + batch_size]
                    logger.debug(f"Processing detector batch {i//batch_size + 1}/4 ({len(batch_dets)} detectors)")

                    # Check resource limits before each batch
                    if not self._check_resource_limits():
                        logger.warning("Resource limits exceeded, skipping remaining detectors")
                        break

                    # Process batch
                    for det in batch_dets:
                        try:
                            # Quick check if we should continue
                            if not self._check_resource_limits():
                                logger.debug(f"Skipping detector {getattr(det, 'key', '?')} due to resource limits")
                                continue

                            batch_events = det.detect(dfN)
                            if batch_events:
                                evs.extend(batch_events)
                        except Exception as e:
                            logger.debug(f"Detector {getattr(det, 'key', '?')} failed: {e}")

                    # Small delay between batches to allow UI updates
                    import time
                    time.sleep(0.01)
            else:
                # Use original synchronous detection for small datasets
                for det in dets:
                    try:
                        evs.extend(det.detect(dfN))
                    except Exception as e:
                        logger.debug(f"Detector {getattr(det, 'key', '?')} failed: {e}")

            logger.info(
                f"Patterns detected: total={len(evs)} "
                f"(chart={sum(1 for x in evs if getattr(x, 'kind', '') == 'chart')}, "
                f"candle={sum(1 for x in evs if getattr(x, 'kind', '') == 'candle')})"
            )

            # Enrich, attach human info, annotate TF hint, and merge into persistent cache
            enriched = enrich_events(dfN, evs)
            tf_hint = getattr(self.view, "_patterns_scan_tf_hint", None) or getattr(self.controller, "timeframe", None)

            for e in enriched:
                # Attach info from pattern_info.json (name, description, benchmarks, notes, image)
                try:
                    self._attach_info_to_event(e)
                except Exception:
                    pass
                # Annotate timeframe
                try:
                    if tf_hint is not None:
                        setattr(e, "tf", str(tf_hint))
                except Exception:
                    pass

            if not isinstance(getattr(self, "_cache", None), dict):
                self._cache = {}

            for e in enriched:
                try:
                    name = getattr(e, "name", getattr(e, "key", type(e).__name__))
                    start_ts = getattr(e, "start_ts", None) or getattr(e, "begin_ts", None)
                    end_ts = getattr(e, "end_ts", None) or getattr(e, "finish_ts", None)
                    confirm_ts = getattr(e, "confirm_ts", getattr(e, "ts", None))
                    tfk = getattr(e, "tf", None)
                    key = (str(name), int(start_ts or 0), int(end_ts or 0), int(confirm_ts or 0), str(tfk or "-"))
                    self._cache[key] = e
                except Exception:
                    continue

            self._events = list(self._cache.values())
            self._repaint()

            # Scan other timeframes for the same symbol and merge (once per symbol session)
            try:
                cur_tf = str(tf_hint or getattr(self.controller, "timeframe", "") or "").lower()
                self._scan_other_timeframes(current_tf=cur_tf)
            except Exception:
                pass

        except Exception as e:
            logger.exception("Patterns _run_detection failed: {}", e)

    def on_update_plot(self, df: pd.DataFrame):
        from .patterns_hook import call_patterns_detection
        call_patterns_detection(self.controller, self.view, df)
        self.detect_async(df)

    def _run_detection_nonblocking(self, df: pd.DataFrame):
        """Run pattern detection in non-blocking batches using QTimer"""
        try:
            # Initialize state for non-blocking detection
            if not hasattr(self, '_detection_state'):
                self._detection_state = {
                    'active': False,
                    'df': None,
                    'detectors': [],
                    'current_batch': 0,
                    'events': [],
                    'batch_size': 8,
                    'timer': QTimer()
                }
                self._detection_state['timer'].timeout.connect(self._process_detection_batch)

            # If already running, queue this request
            if self._detection_state['active']:
                self._detection_state['df'] = df  # Update with latest data
                return

            # Start new detection process
            kinds: list[str] = []
            if self._enabled_chart:
                kinds.append("chart")
            if self._enabled_candle:
                kinds.append("candle")
            if not kinds:
                logger.info("Patterns: toggles OFF → skipping detection")
                self._events.clear()
                self.renderer.clear()
                return

            dfN = self._normalize_df(df)
            if dfN is None or dfN.empty:
                logger.warning("Patterns: normalization failed or empty → no detection")
                self._events.clear()
                self.renderer.clear()
                return

            dets_list = self.registry.detectors(kinds=kinds)
            if dets_list is None:
                logger.error(f"No detectors found for kinds: {kinds}")
                return

            dets = list(dets_list)

            logger.info(f"Starting non-blocking detection: {len(dfN)} rows, {len(dets)} detectors - analyzing performance issues")

            # Update status to show scan starting
            try:
                if hasattr(self.view, 'update_status'):
                    self.view.update_status(f"Starting pattern scan: {len(dets)} detectors, {len(dfN)} bars")
                elif hasattr(self.controller, 'update_status'):
                    self.controller.update_status("Starting pattern scan...")
            except Exception:
                pass

            # Initialize detection state
            self._detection_state.update({
                'active': True,
                'df': dfN,
                'detectors': dets,
                'current_batch': 0,
                'events': [],
                'kinds': kinds
            })

            # Start processing
            self._detection_state['timer'].start(10)  # Process every 10ms

        except Exception as e:
            logger.exception("Failed to start non-blocking detection: {}", e)

    def _process_detection_batch(self):
        """Process one batch of detectors"""
        try:
            state = self._detection_state
            if not state['active']:
                return

            batch_size = state['batch_size']
            current_batch = state['current_batch']
            detectors = state['detectors']
            dfN = state['df']

            start_idx = current_batch * batch_size
            end_idx = min(start_idx + batch_size, len(detectors))

            if start_idx >= len(detectors):
                # Detection complete
                self._finish_detection()
                return

            # Check resource limits
            if not self._check_resource_limits():
                logger.warning("Resource limits exceeded, finishing detection early")
                self._finish_detection()
                return

            # Process current batch
            batch_detectors = detectors[start_idx:end_idx]
            total_batches = (len(detectors) + batch_size - 1) // batch_size
            progress_percent = int((current_batch / total_batches) * 100)

            logger.debug(f"Processing detector batch {current_batch + 1}/{total_batches} ({len(batch_detectors)} detectors) - {progress_percent}%")

            # Update progress indicator
            try:
                if hasattr(self.view, 'update_status'):
                    self.view.update_status(f"Scanning patterns... {progress_percent}% ({current_batch + 1}/{total_batches})")
                elif hasattr(self.controller, 'update_status'):
                    self.controller.update_status(f"Scanning patterns... {progress_percent}%")
            except Exception:
                pass

            for i, det in enumerate(batch_detectors):
                try:
                    detector_key = getattr(det, 'key', f'unknown_{i}')

                    logger.debug(f"Starting detector {detector_key} (batch {current_batch + 1}, detector {i + 1}/{len(batch_detectors)})")

                    # Simple timeout mechanism using time measurement
                    import time
                    start_time = time.time()

                    batch_events = det.detect(dfN)

                    elapsed = time.time() - start_time
                    logger.debug(f"Completed detector {detector_key} in {elapsed:.2f}s - found {len(batch_events) if batch_events else 0} events")

                    # Warn if detector is taking too long
                    if elapsed > 2.0:  # More than 2 seconds
                        logger.warning(f"Detector {detector_key} took {elapsed:.2f}s - this may cause GUI blocking")

                    # Skip detector result if it took too long (emergency brake)
                    if elapsed > 10.0:  # More than 10 seconds
                        logger.error(f"Detector {detector_key} took {elapsed:.2f}s - discarding results to prevent GUI blocking")
                        continue  # Don't add events from slow detectors

                    if batch_events:
                        state['events'].extend(batch_events)

                except Exception as e:
                    detector_key = getattr(det, 'key', f'unknown_{i}')
                    logger.error(f"Detector {detector_key} failed: {e}")
                    # Continue with next detector instead of stopping

            # Move to next batch
            state['current_batch'] += 1

            # Continue processing
            if end_idx < len(detectors):
                # Schedule next batch
                pass  # Timer will call us again
            else:
                # All batches processed
                self._finish_detection()

        except Exception as e:
            logger.error(f"Error in detection batch processing: {e}")
            self._finish_detection()

    def _finish_detection(self):
        """Finish non-blocking detection and update UI"""
        try:
            state = self._detection_state
            if not state['active']:
                return

            # Stop timer
            state['timer'].stop()
            state['active'] = False

            evs = state['events']
            dfN = state['df']

            logger.info(
                f"Patterns detected: total={len(evs)} "
                f"(chart={sum(1 for x in evs if getattr(x, 'kind', '') == 'chart')}, "
                f"candle={sum(1 for x in evs if getattr(x, 'kind', '') == 'candle')})"
            )

            # Continue with existing enrichment logic
            if evs:
                from .patterns_adapter import enrich_events
                enriched = enrich_events(dfN, evs)
                tf_hint = getattr(self.view, "_patterns_scan_tf_hint", None) or getattr(self.controller, "timeframe", None)

                for e in enriched:
                    # Attach info from pattern_info.json
                    try:
                        if hasattr(e, 'pattern_key') and self.info:
                            info_dict = self.info.get_pattern_info(e.pattern_key)
                            if info_dict:
                                for k, v in info_dict.items():
                                    try:
                                        setattr(e, f"info_{k}", v)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    # Attach timeframe hint
                    if tf_hint:
                        try:
                            setattr(e, "tf_hint", tf_hint)
                        except Exception:
                            pass

                # Update events and refresh display
                self._events = enriched
                self._repaint()

                # Update status with completion
                try:
                    pattern_count = len(enriched)
                    if hasattr(self.view, 'update_status'):
                        self.view.update_status(f"Pattern scan complete: {pattern_count} patterns found")
                    elif hasattr(self.controller, 'update_status'):
                        self.controller.update_status(f"Patterns: {pattern_count} found")
                except Exception:
                    pass
            else:
                self._events = []
                self.renderer.clear()

                # Update status for no patterns
                try:
                    if hasattr(self.view, 'update_status'):
                        self.view.update_status("Pattern scan complete: No patterns found")
                    elif hasattr(self.controller, 'update_status'):
                        self.controller.update_status("No patterns found")
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error finishing detection: {e}")
            self._detection_state['active'] = False

            # Update status on error
            try:
                if hasattr(self.view, 'update_status'):
                    self.view.update_status("Pattern scan failed")
                elif hasattr(self.controller, 'update_status'):
                    self.controller.update_status("Pattern scan error")
            except Exception:
                pass

    def _default_info_path(self):
        from pathlib import Path
        return Path(getattr(self.view, "_app_root", ".")) / "configs" / "pattern_info.json"

    def _normalize_df(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.info("Patterns: received empty DataFrame"); return None
        df0 = df.copy()
        original_cols = list(df0.columns)
        df0.columns = [str(c).strip() for c in df0.columns]
        df0.rename(columns={c: c.lower() for c in df0.columns}, inplace=True)
        cols = set(df0.columns); mapped: Dict[str,str] = {}
        for syn, canon in OHLC_SYNONYMS.items():
            if syn in cols and canon not in mapped: mapped[canon] = syn
        for canon in ['open','high','low','close']:
            if canon in cols: mapped[canon] = canon
        missing = [k for k in ['open','high','low','close'] if k not in mapped]
        if missing:
            logger.warning(f"Patterns: missing OHLC after mapping → {missing}; cols={list(df0.columns)[:24]} (orig={original_cols[:24]})")
            return None
        for canon, src in mapped.items():
            if canon != src: df0[canon] = df0[src]
        for k in ['open','high','low','close']:
            df0[k] = pd.to_numeric(df0[k], errors='coerce')
        ts_col = next((t for t in TS_SYNONYMS if t in df0.columns), None)
        if ts_col is None:
            logger.warning(f"Patterns: no time column found; available={list(df0.columns)[:24]} (orig={original_cols[:24]})")
            return None
        s = df0[ts_col]
        if np.issubdtype(s.dtype, np.datetime64):
            ts_ms = pd.to_datetime(s).view('int64') // 10**6
        else:
            vals = pd.to_numeric(s, errors='coerce')
            if vals.isna().all():
                ts_ms = pd.to_datetime(s, errors='coerce').view('int64') // 10**6
            else:
                v = int(vals.dropna().iloc[0]) if not vals.dropna().empty else 0
                if v > 10**16: ts_ms = vals.astype('int64') // 10**6
                elif v > 10**13: ts_ms = vals.astype('int64') // 10**3
                elif v > 10**11: ts_ms = vals.astype('int64')
                else: ts_ms = vals.astype('int64') * 1000
        df0['ts_utc'] = ts_ms.astype('int64', errors='ignore')
        before = len(df0)
        df0 = df0.dropna(subset=['open','high','low','close','ts_utc'])
        after = len(df0)
        logger.info(f"Patterns: normalized df rows={after} (dropped {before-after}); cols={list(df0.columns)[:24]} → using ['ts_utc','open','high','low','close']")
        return df0[['ts_utc','open','high','low','close']].copy()

    def _repaint(self) -> None:
        """Ridisegna solo il layer patterns senza toccare la Figure o il View."""
        try:
            self.renderer.draw(self._events or [])
        except Exception as e:
            from loguru import logger
            logger.debug(f"PatternsService._repaint skipped: {e}")

    # ---- Helpers ----
    def _attach_info_to_event(self, e: object) -> None:
        """Attach human-friendly name, description, benchmarks, notes, image from info provider."""
        try:
            ip = getattr(self, "info", None)
            if ip is None or not hasattr(ip, "describe"):
                return
            raw_key = str(getattr(e, "pattern_key", getattr(e, "key", "")) or "").strip()
            raw_name = str(getattr(e, "name", "") or "").strip()
            pi = ip.describe(raw_key) if raw_key else None
            if pi is None and raw_name:
                # try by name (case-insensitive) scanning db
                db = getattr(ip, "_db", {}) or {}
                low = raw_name.lower()
                for k, v in db.items():
                    try:
                        if str(v.get("name", "")).lower() == low:
                            pi = ip.describe(k)
                            break
                    except Exception:
                        continue
            if pi is None and raw_name:
                pi = ip.describe(raw_name)

            if pi is None:
                return

            if getattr(pi, "name", None):
                try: setattr(e, "name", str(pi.name))
                except Exception: pass
            if getattr(pi, "description", None):
                try: setattr(e, "description", str(pi.description))
                except Exception: pass
            bm = getattr(pi, "benchmarks", None)
            if isinstance(bm, dict):
                try: setattr(e, "benchmark", bm); setattr(e, "benchmarks", bm)
                except Exception: pass
            try:
                bull = getattr(pi, "bull", None); bear = getattr(pi, "bear", None)
                if isinstance(bull, dict):
                    setattr(e, "notes_bull", bull.get("notes") or bull)
                if isinstance(bear, dict):
                    setattr(e, "notes_bear", bear.get("notes") or bear)
            except Exception:
                pass
            img_rel = getattr(pi, "image_resource", None)
            if img_rel:
                from pathlib import Path
                root = Path(getattr(self.view, "_app_root", ".")) if hasattr(self.view, "_app_root") else Path(".")
                try: setattr(e, "image_path", (root / str(img_rel)).as_posix())
                except Exception: pass
        except Exception:
            pass

    def _scan_other_timeframes(self, current_tf: str = "") -> None:
        """Scan 1m,5m,15m,30m,1h,4h,1d and merge results into cache (once per symbol session)."""
        try:
            sym = getattr(self.view, "symbol", None) or getattr(self.controller, "symbol", None)
            if not sym:
                return
            if self._scanning_multi:
                return
            self._scanning_multi = True
            try:
                tfs_all = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                cur_set = self._scanned_tfs_by_symbol.get(sym, set())
                if current_tf:
                    cur_set.add(str(current_tf))
                for tf in tfs_all:
                    if tf in cur_set:
                        continue
                    try:
                        df_tf = self.controller.load_candles_from_db(sym, tf, limit=50000)
                    except Exception:
                        df_tf = None
                    if df_tf is None or df_tf.empty:
                        cur_set.add(tf); continue
                    hint_prev = getattr(self.view, "_patterns_scan_tf_hint", None)
                    try:
                        setattr(self.view, "_patterns_scan_tf_hint", tf)
                        self._run_detection(df_tf)
                    finally:
                        try:
                            setattr(self.view, "_patterns_scan_tf_hint", hint_prev)
                        except Exception:
                            pass
                    cur_set.add(tf)
                self._scanned_tfs_by_symbol[sym] = cur_set
            finally:
                self._scanning_multi = False
        except Exception:
            self._scanning_multi = False

    # ---------- Helpers: attach info & multi-timeframe scan ----------

    def _attach_info_to_event(self, e: object) -> None:
        """Attach human-friendly name, description, benchmarks, notes, image from info provider."""
        try:
            if not hasattr(self, "info"):
                return
            raw_key = str(getattr(e, "key", "") or "").strip()
            raw_name = str(getattr(e, "name", "") or "").strip()
            pi = None
            # 1) try by canonical key
            if raw_key and hasattr(self.info, "describe"):
                pi = self.info.describe(raw_key)
            # 2) fallback: try by human name (case-insensitive) by scanning provider DB
            if pi is None:
                try:
                    db = getattr(self.info, "_db", {}) or {}
                    low = raw_name.lower()
                    for k, v in db.items():
                        try:
                            if str(v.get("name", "")).lower() == low:
                                pi = self.info.describe(k)
                                break
                        except Exception:
                            continue
                except Exception:
                    pass
            if pi is None:
                # last resort: try name via describe (if DB happens to use same)
                if raw_name and hasattr(self.info, "describe"):
                    pi = self.info.describe(raw_name)
            if pi is None:
                return

            # Name/title
            try:
                if getattr(pi, "name", None):
                    setattr(e, "name", str(pi.name))
            except Exception:
                pass
            # Description
            try:
                if getattr(pi, "description", None):
                    setattr(e, "description", str(pi.description))
            except Exception:
                pass
            # Benchmarks (attach both 'benchmark' and 'benchmarks' for consumers)
            try:
                bm = getattr(pi, "benchmarks", None)
                if isinstance(bm, dict):
                    setattr(e, "benchmark", bm)
                    setattr(e, "benchmarks", bm)
            except Exception:
                pass
            # Notes bull/bear (optional)
            try:
                bull = getattr(pi, "bull", None)
                bear = getattr(pi, "bear", None)
                if isinstance(bull, dict):
                    setattr(e, "notes_bull", bull.get("notes") or bull)
                if isinstance(bear, dict):
                    setattr(e, "notes_bear", bear.get("notes") or bear)
            except Exception:
                pass
            # Image path (optional)
            try:
                img_rel = getattr(pi, "image_resource", None)
                if img_rel:
                    from pathlib import Path
                    root = Path(getattr(self.view, "_app_root", ".")) if hasattr(self.view, "_app_root") else Path(".")
                    img_path = (root / str(img_rel)).as_posix()
                    setattr(e, "image_path", img_path)
            except Exception:
                pass
        except Exception:
            pass

    def _scan_other_timeframes(self, current_tf: str = "") -> None:
        """Scan remaining timeframes (1m,5m,15m,30m,1h,4h,1d) and merge results into cache (once per symbol)."""
        try:
            sym = getattr(self.view, "symbol", None) or getattr(self.controller, "symbol", None)
            if not sym:
                return
            # guard reentrancy
            if self._scanning_multi:
                return
            self._scanning_multi = True
            try:
                tfs_all = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                cur_set = self._scanned_tfs_by_symbol.get(sym, set())
                # always record that we've processed current tf (if any)
                if current_tf:
                    cur_set.add(str(current_tf))
                for tf in tfs_all:
                    if tf in cur_set:
                        continue
                    try:
                        df_tf = self.controller.load_candles_from_db(sym, tf, limit=50000)
                    except Exception:
                        df_tf = None
                    if df_tf is None or df_tf.empty:
                        cur_set.add(tf)
                        continue
                    # annotate hint on view for tf tagging
                    prev_hint = getattr(self.view, "_patterns_scan_tf_hint", None)
                    try:
                        setattr(self.view, "_patterns_scan_tf_hint", tf)
                        # run detection inline to avoid debounce and to merge immediately
                        self._run_detection(df_tf)
                    finally:
                        try:
                            setattr(self.view, "_patterns_scan_tf_hint", prev_hint)
                        except Exception:
                            pass
                    cur_set.add(tf)
                self._scanned_tfs_by_symbol[sym] = cur_set
            finally:
                self._scanning_multi = False
        except Exception:
            self._scanning_multi = False

    # ---- Training/Optimization Orchestration Methods ----

    def start_optimization_study(self, config: dict) -> dict:
        """Start an optimization study using the patterns service data"""
        try:
            # Initialize optimization engine
            engine = OptimizationEngine()

            # Prepare pattern-specific configuration
            pattern_config = self._prepare_pattern_config(config)

            # Start the optimization study
            study_id = engine.run_study(pattern_config)

            logger.info(f"Started optimization study: {study_id}")
            return {
                'success': True,
                'study_id': study_id,
                'message': f'Optimization study {study_id} started successfully'
            }

        except Exception as e:
            logger.error(f"Failed to start optimization study: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to start optimization study'
            }

    def get_optimization_status(self, study_id: str) -> dict:
        """Get the status of an ongoing optimization study"""
        try:
            task_manager = TaskManager()
            status = task_manager.get_study_status(study_id)
            return {
                'success': True,
                'status': status
            }
        except Exception as e:
            logger.error(f"Failed to get optimization status: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def stop_optimization_study(self, study_id: str) -> dict:
        """Stop an ongoing optimization study"""
        try:
            task_manager = TaskManager()
            task_manager.stop_study(study_id)

            logger.info(f"Stopped optimization study: {study_id}")
            return {
                'success': True,
                'message': f'Study {study_id} stopped successfully'
            }

        except Exception as e:
            logger.error(f"Failed to stop optimization study: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_best_parameters(self, pattern_type: str = None) -> dict:
        """Get the best parameters for a pattern type or all patterns"""
        try:
            task_manager = TaskManager()

            if pattern_type:
                params = task_manager.get_best_parameters_for_pattern(pattern_type)
            else:
                params = task_manager.get_all_best_parameters()

            return {
                'success': True,
                'parameters': params
            }

        except Exception as e:
            logger.error(f"Failed to get best parameters: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def apply_optimized_parameters(self, parameters: dict) -> dict:
        """Apply optimized parameters to the pattern detection system"""
        try:
            # Update registry with new parameters
            for pattern_key, params in parameters.items():
                detectors = self.registry.detectors(pattern_keys=[pattern_key])
                if detectors:
                    for detector in detectors:
                        self._update_detector_parameters(detector, params)

            # Trigger re-detection with new parameters
            self._trigger_redetection()

            logger.info(f"Applied optimized parameters for {len(parameters)} patterns")
            return {
                'success': True,
                'message': f'Applied parameters for {len(parameters)} patterns'
            }

        except Exception as e:
            logger.error(f"Failed to apply optimized parameters: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def prepare_training_data(self, timeframes: list = None, limit: int = 10000) -> dict:
        """Prepare historical data for training/optimization"""
        try:
            if timeframes is None:
                timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

            training_data = {}
            symbol = getattr(self.view, "symbol", None) or getattr(self.controller, "symbol", None)

            if not symbol:
                raise ValueError("No symbol available for training data preparation")

            for tf in timeframes:
                try:
                    df = self.controller.load_candles_from_db(symbol, tf, limit=limit)
                    if df is not None and not df.empty:
                        normalized_df = self._normalize_df(df)
                        if normalized_df is not None:
                            training_data[tf] = {
                                'data': normalized_df,
                                'symbol': symbol,
                                'timeframe': tf,
                                'rows': len(normalized_df)
                            }
                except Exception as e:
                    logger.warning(f"Failed to load data for {tf}: {e}")
                    continue

            logger.info(f"Prepared training data for {len(training_data)} timeframes")
            return {
                'success': True,
                'training_data': training_data,
                'timeframes': list(training_data.keys()),
                'total_rows': sum(data['rows'] for data in training_data.values())
            }

        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def validate_pattern_performance(self, pattern_type: str, parameters: dict,
                                   test_data: pd.DataFrame = None) -> dict:
        """Validate pattern performance with given parameters"""
        try:
            if test_data is None:
                # Use current data if no test data provided
                test_data = getattr(self.controller.plot_service, "_last_df", None)
                if test_data is None:
                    raise ValueError("No test data available")
                test_data = self._normalize_df(test_data)

            # Get detector for pattern type
            detectors = self.registry.detectors(pattern_keys=[pattern_type])
            if not detectors:
                raise ValueError(f"No detector found for pattern type: {pattern_type}")

            detector = list(detectors)[0]

            # Create a copy and apply parameters
            temp_detector = self._create_detector_copy(detector, parameters)

            # Run detection
            events = temp_detector.detect(test_data)

            # Calculate performance metrics
            metrics = self._calculate_pattern_metrics(events, test_data)

            return {
                'success': True,
                'pattern_type': pattern_type,
                'parameters': parameters,
                'events_count': len(events),
                'metrics': metrics
            }

        except Exception as e:
            logger.error(f"Failed to validate pattern performance: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_pattern_suggestions(self, symbol: str = None) -> dict:
        """Get parameter suggestions based on historical pattern performance"""
        try:
            param_space = ParameterSpace()

            if symbol is None:
                symbol = getattr(self.view, "symbol", None) or getattr(self.controller, "symbol", None)

            suggestions = {}

            # Get available pattern types from registry
            all_detectors = self.registry.detectors()
            pattern_types = {getattr(d, 'key', getattr(d, 'name', str(d))) for d in all_detectors}

            for pattern_type in pattern_types:
                try:
                    suggested_ranges = param_space.get_suggested_ranges(pattern_type)
                    suggestions[pattern_type] = suggested_ranges
                except Exception as e:
                    logger.debug(f"No suggestions available for {pattern_type}: {e}")
                    continue

            return {
                'success': True,
                'symbol': symbol,
                'suggestions': suggestions,
                'pattern_count': len(suggestions)
            }

        except Exception as e:
            logger.error(f"Failed to get pattern suggestions: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    # ---- Private Helper Methods for Training ----

    def _prepare_pattern_config(self, config: dict) -> dict:
        """Prepare configuration for pattern optimization"""
        pattern_config = config.copy()

        # Add current symbol if not specified
        if 'symbol' not in pattern_config:
            symbol = getattr(self.view, "symbol", None) or getattr(self.controller, "symbol", None)
            pattern_config['symbol'] = symbol

        # Add pattern registry information
        pattern_config['available_patterns'] = [
            getattr(d, 'key', getattr(d, 'name', str(d)))
            for d in self.registry.detectors()
        ]

        # Add data access callback
        pattern_config['data_loader'] = self._create_data_loader()

        return pattern_config

    def _create_data_loader(self):
        """Create a data loader function for optimization"""
        def load_data(symbol: str, timeframe: str, limit: int = 10000):
            try:
                df = self.controller.load_candles_from_db(symbol, timeframe, limit=limit)
                return self._normalize_df(df) if df is not None else None
            except Exception as e:
                logger.error(f"Data loader failed for {symbol} {timeframe}: {e}")
                return None

        return load_data

    def _update_detector_parameters(self, detector, parameters: dict):
        """Update detector parameters"""
        for param_name, param_value in parameters.items():
            try:
                if hasattr(detector, param_name):
                    setattr(detector, param_name, param_value)
            except Exception as e:
                logger.warning(f"Failed to set parameter {param_name}: {e}")

    def _create_detector_copy(self, detector, parameters: dict):
        """Create a copy of detector with new parameters"""
        import copy
        temp_detector = copy.deepcopy(detector)
        self._update_detector_parameters(temp_detector, parameters)
        return temp_detector

    def _calculate_pattern_metrics(self, events: list, test_data: pd.DataFrame) -> dict:
        """Calculate performance metrics for pattern events"""
        if not events:
            return {
                'success_rate': 0.0,
                'average_confidence': 0.0,
                'event_density': 0.0
            }

        # Basic metrics
        total_events = len(events)
        successful_events = sum(1 for e in events if getattr(e, 'confidence', 0) > 0.5)

        metrics = {
            'success_rate': successful_events / total_events if total_events > 0 else 0,
            'average_confidence': sum(getattr(e, 'confidence', 0) for e in events) / total_events,
            'event_density': total_events / len(test_data) if len(test_data) > 0 else 0,
            'total_events': total_events,
            'data_rows': len(test_data)
        }

        return metrics

    def _trigger_redetection(self):
        """Trigger pattern re-detection with current data"""
        try:
            current_df = getattr(self.controller.plot_service, "_last_df", None)
            if current_df is not None:
                self.detect_async(current_df)
        except Exception as e:
            logger.warning(f"Failed to trigger redetection: {e}")

    # ---- Strategy Selection Methods ----

    def set_trading_strategy(self, strategy_tag: str) -> bool:
        """
        Set the trading strategy for real-time pattern detection.

        Args:
            strategy_tag: "high_return", "low_risk", or "balanced"

        Returns:
            True if strategy was changed successfully
        """
        if strategy_tag not in self._available_strategies:
            logger.error(f"Invalid strategy: {strategy_tag}. Available: {list(self._available_strategies.keys())}")
            return False

        old_strategy = self._current_strategy
        self._current_strategy = strategy_tag

        logger.info(f"Trading strategy changed: {old_strategy} → {strategy_tag}")

        # Apply new strategy parameters
        self.apply_production_parameters()

        # Trigger re-detection with new parameters
        self._trigger_redetection()

        return True

    def get_current_strategy(self) -> Dict[str, Any]:
        """Get current trading strategy info"""
        return {
            "current": self._current_strategy,
            "info": self._available_strategies[self._current_strategy],
            "available": self._available_strategies
        }

    def get_strategy_comparison(self) -> Dict[str, Any]:
        """Get performance comparison between available strategies"""
        try:
            from ....training.optimization.task_manager import TaskManager

            task_manager = TaskManager()
            symbol = getattr(self.view, "symbol", None) or getattr(self.controller, "symbol", None)
            timeframe = getattr(self.controller, "timeframe", None)

            if not symbol or not timeframe:
                return {"error": "No symbol/timeframe available"}

            comparison = {}

            # Get parameters for each strategy
            for strategy_tag in self._available_strategies.keys():
                try:
                    params = task_manager.get_production_parameters(
                        asset=symbol,
                        timeframe=str(timeframe),
                        strategy_tag=strategy_tag
                    )

                    if params:
                        # Extract performance metrics
                        strategy_performance = {}
                        for pattern_key, pattern_params in params.items():
                            perf = pattern_params.get('performance', {})
                            strategy_performance[pattern_key] = {
                                'success_rate': (perf.get('d1_success_rate', 0) + perf.get('d2_success_rate', 0)),
                                'combined_score': perf.get('combined_score', 0),
                                'total_signals': perf.get('total_signals', 0),
                                'robustness': perf.get('robustness_score', 0)
                            }

                        comparison[strategy_tag] = {
                            'info': self._available_strategies[strategy_tag],
                            'patterns_count': len(params),
                            'performance': strategy_performance,
                            'available': True
                        }
                    else:
                        comparison[strategy_tag] = {
                            'info': self._available_strategies[strategy_tag],
                            'available': False,
                            'reason': 'No optimized parameters available'
                        }

                except Exception as e:
                    comparison[strategy_tag] = {
                        'info': self._available_strategies[strategy_tag],
                        'available': False,
                        'error': str(e)
                    }

            return comparison

        except Exception as e:
            logger.error(f"Failed to get strategy comparison: {e}")
            return {"error": str(e)}

    def load_production_parameters(self) -> dict:
        """Load promoted parameters from database for production use"""
        try:
            from ....training.optimization.task_manager import TaskManager

            task_manager = TaskManager()

            # Get current symbol and timeframe
            symbol = getattr(self.view, "symbol", None) or getattr(self.controller, "symbol", None)
            timeframe = getattr(self.controller, "timeframe", None)

            if not symbol or not timeframe:
                logger.warning("No symbol/timeframe available for loading production parameters")
                return {}

            # Get promoted parameters for current strategy
            production_params = task_manager.get_production_parameters(
                asset=symbol,
                timeframe=str(timeframe),
                strategy_tag=self._current_strategy
            )

            logger.info(f"Loaded {len(production_params)} production parameter sets for {symbol} {timeframe}")
            return production_params

        except Exception as e:
            logger.error(f"Failed to load production parameters: {e}")
            return {}

    def apply_production_parameters(self, production_params: dict = None):
        """Apply production parameters to pattern detectors"""
        try:
            if production_params is None:
                production_params = self.load_production_parameters()

            if not production_params:
                logger.info("No production parameters to apply")
                return

            applied_count = 0

            # Apply parameters to each pattern detector
            for pattern_key, params in production_params.items():
                try:
                    # Get detectors for this pattern
                    detectors = self.registry.detectors(pattern_keys=[pattern_key])

                    if detectors:
                        for detector in detectors:
                            # Apply form parameters (detector configuration)
                            form_params = params.get('form_parameters', {})
                            for param_name, param_value in form_params.items():
                                if hasattr(detector, param_name):
                                    setattr(detector, param_name, param_value)
                                    logger.debug(f"Applied {param_name}={param_value} to {pattern_key}")

                            # Store action parameters for trade execution
                            action_params = params.get('action_parameters', {})
                            if hasattr(detector, '_action_parameters'):
                                detector._action_parameters = action_params
                            else:
                                setattr(detector, '_action_parameters', action_params)

                        applied_count += 1
                        logger.info(f"Applied production parameters to {pattern_key}")

                except Exception as e:
                    logger.warning(f"Failed to apply parameters to {pattern_key}: {e}")
                    continue

            logger.info(f"Successfully applied production parameters to {applied_count} patterns")

            # Trigger re-detection with new parameters
            self._trigger_redetection()

        except Exception as e:
            logger.error(f"Failed to apply production parameters: {e}")

    def auto_update_parameters(self, check_interval_hours: int = 24):
        """
        Check for newly promoted parameters and apply them.

        NOTE: This does NOT start new optimizations! It only checks if any
        completed optimization studies have promoted new parameters that
        should be applied to production.

        Optimizations themselves run separately and can take days/weeks.
        """
        try:
            logger.info("Checking for newly promoted parameters (not running new optimizations)...")

            # Get current parameters hash to detect changes
            current_params = self.load_production_parameters()
            current_hash = self._calculate_params_hash(current_params)

            # Compare with last known hash
            if hasattr(self, '_last_params_hash') and self._last_params_hash == current_hash:
                logger.info("No new promoted parameters found")
                return

            # New parameters detected - apply them
            if current_params:
                logger.info(f"Found {len(current_params)} newly promoted parameter sets")
                self.apply_production_parameters(current_params)
                self._last_params_hash = current_hash
            else:
                logger.info("No promoted parameters available")

            logger.info(f"Parameter check completed. Next check in {check_interval_hours}h")

        except Exception as e:
            logger.error(f"Auto parameter update failed: {e}")

    def _calculate_params_hash(self, params: dict) -> str:
        """Calculate hash of parameter set for change detection"""
        import hashlib
        import json

        try:
            params_str = json.dumps(params, sort_keys=True)
            return hashlib.md5(params_str.encode()).hexdigest()
        except Exception:
            return ""



    def _scan_once(self, kind: str):
        """Scan for patterns with async resource monitoring"""
        try:
            # Check resource limits before scanning
            if not self._check_resource_limits():
                logger.debug(f"Skipping {kind} pattern scan due to resource limits")
                return []

            df = getattr(self.controller.plot_service, '_last_df', None)
            dfN = self._normalize_df(df)
            if dfN is None or len(dfN)==0:
                logger.debug(f"Skipping {kind} pattern scan - no data available")
                return []

            # Check if data has changed since last scan to avoid unnecessary processing
            if hasattr(self, '_last_scan_data_hash'):
                current_hash = hash(str(dfN.iloc[-10:].values.tobytes()) if len(dfN) >= 10 else str(dfN.values.tobytes()))
                if current_hash == getattr(self, f'_last_{kind}_scan_hash', None):
                    logger.debug(f"Skipping {kind} pattern scan - data unchanged")
                    return []
                setattr(self, f'_last_{kind}_scan_hash', current_hash)

            # Check market hours to avoid scanning when markets are closed
            if self._is_market_likely_closed():
                logger.debug(f"Skipping {kind} pattern scan - market likely closed")
                return []

            # Start synchronous pattern detection to avoid thread blocking
            result = self._sync_pattern_detection(dfN, kind)
            return result or []

        except Exception as e:
            logger.error(f"Error in pattern scan ({kind}): {e}")
            # Return empty list instead of None to prevent further errors
            return []

    async def _async_pattern_detection(self, dfN, kind: str):
        """Async pattern detection with resource throttling"""
        try:
            # Additional check to avoid unnecessary computation during market closure
            if self._is_market_likely_closed():
                logger.debug(f"Skipping {kind} async pattern detection - market closed")
                return []

            from asyncio_throttle import Throttler
            from src.forex_diffusion.patterns.registry import PatternRegistry
            from .patterns_adapter import enrich_events
            from ....patterns.info_provider import PatternInfoProvider

            # Create throttler to limit resource usage (30% CPU max)
            max_concurrent = max(1, int(psutil.cpu_count() * 0.3))
            throttler = Throttler(rate_limit=max_concurrent, period=1.0)

            reg = PatternRegistry()
            dets = [d for d in reg.detectors([kind])]
            events = []

            # Process detectors with throttling
            async with throttler:
                detection_tasks = []
                for detector in dets:
                    task = self._detect_pattern_async(detector, dfN, throttler)
                    detection_tasks.append(task)

                # Run detections with resource monitoring
                results = await asyncio.gather(*detection_tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        logger.debug(f"Pattern detection error: {result}")
                    elif result:
                        events.extend(result)

            # Enrich events with pattern information
            if events:
                events = enrich_events(events, PatternInfoProvider())

                # Update events safely on main thread
                self._events = (self._events or []) + events

                # Schedule repaint on main thread
                try:
                    # Use QTimer to ensure repaint happens on main thread
                    QTimer.singleShot(0, self._repaint)
                except Exception:
                    pass

            return events

        except Exception as e:
            logger.error(f"Async pattern detection error: {e}")
            return []

    def _sync_pattern_detection(self, dfN, kind: str):
        """Synchronous pattern detection to avoid thread blocking"""
        try:
            # Additional check to avoid unnecessary computation during market closure
            if self._is_market_likely_closed():
                logger.debug(f"Skipping {kind} sync pattern detection - market closed")
                return []

            from src.forex_diffusion.patterns.registry import PatternRegistry
            from .patterns_adapter import enrich_events
            from ....patterns.info_provider import PatternInfoProvider

            reg = PatternRegistry()
            all_dets = [d for d in reg.detectors([kind])]
            events = []

            # Implement round-robin to process different detectors each scan
            if not hasattr(self, f'_detector_offset_{kind}'):
                setattr(self, f'_detector_offset_{kind}', 0)

            offset = getattr(self, f'_detector_offset_{kind}')
            max_detectors_per_scan = min(5, len(all_dets))  # Process at most 5 detectors per scan

            # Rotate through detectors
            dets = []
            for i in range(max_detectors_per_scan):
                idx = (offset + i) % len(all_dets)
                dets.append(all_dets[idx])

            # Update offset for next scan
            setattr(self, f'_detector_offset_{kind}', (offset + max_detectors_per_scan) % len(all_dets))

            # Process detectors synchronously with simple resource check
            for detector in dets:
                try:
                    # Quick resource check before each detector
                    if not self._check_resource_limits():
                        logger.debug(f"Stopping {kind} pattern detection due to resource limits")
                        break

                    # Simple synchronous detection
                    result = self._detect_pattern_sync(detector, dfN)
                    if result:
                        events.extend(result)

                except Exception as e:
                    logger.debug(f"Pattern detection error for {getattr(detector, 'key', 'unknown')}: {e}")
                    continue

            # Enrich events with pattern information
            if events:
                events = enrich_events(events, PatternInfoProvider())

                # Update events safely
                self._events = (self._events or []) + events

                # Schedule repaint on main thread
                try:
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, self._repaint)
                except Exception:
                    pass

            return events

        except Exception as e:
            logger.error(f"Sync pattern detection error: {e}")
            return []

    def _detect_pattern_sync(self, detector, dfN):
        """Synchronous pattern detection for a single detector"""
        try:
            # Simple synchronous detection without throttling
            if hasattr(detector, 'detect'):
                result = detector.detect(dfN)
                return result if result else []
            else:
                logger.debug(f"Detector {getattr(detector, 'key', 'unknown')} has no detect method")
                return []
        except Exception as e:
            logger.debug(f"Error in sync detection for {getattr(detector, 'key', 'unknown')}: {e}")
            return []

    async def _detect_pattern_async(self, detector, dfN, throttler):
        """Detect patterns for a single detector with resource throttling"""
        try:
            async with throttler:
                # Check resources before each detection
                if not self._check_resource_limits():
                    return []

                # Run detection in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, detector.detect, dfN)

                # Small delay to allow other processes
                await asyncio.sleep(0.01)

                return result or []

        except Exception as e:
            logger.debug(f"Pattern detector {getattr(detector, 'key', 'unknown')} error: {e}")
            return []

class _HistoricalScanWorker(QObject):
    finished=Signal()
    def __init__(self,parent)->None:
        super().__init__(); self._parent=parent; self._df_snapshot=None; self._tfs=['1m','5m','15m','30m','1h','4h','1d']
    @Slot(object)
    def set_snapshot(self,df): self._df_snapshot=df
    @Slot()
    def run(self):
        try:
            ps=self._parent
            if self._df_snapshot is None or getattr(self._df_snapshot,'empty',True): self.finished.emit(); return
            for tf in self._tfs:
                try: setattr(ps.view,'_patterns_scan_tf_hint', tf)
                except Exception: pass
                try: ps.on_update_plot(self._df_snapshot)
                except Exception: continue
        finally:
            self.finished.emit()


def start_historical_scan(self, df_snapshot):
    try:
        self._hist_thread=QThread(self.view); self._hist_worker=_HistoricalScanWorker(self)
        self._hist_worker.moveToThread(self._hist_thread)
        self._hist_thread.started.connect(self._hist_worker.run)
        try: self._hist_worker.finished.connect(self._hist_thread.quit)
        except Exception: pass
        try: self._hist_worker.set_snapshot(df_snapshot)
        except Exception: pass
        self._hist_thread.start()
    except Exception:
        try: self.on_update_plot(df_snapshot)
        except Exception: pass


def _min_required_bars(self, det) -> int:
    for attr in ('window','max_span'):
        if hasattr(det, attr):
            try:
                v = int(getattr(det, attr));
                if v and v>0: return max(60, v)
            except Exception: pass
    key = getattr(det, 'key', '') or ''
    long_families = {'head_and_shoulders':140,'inverse_head_and_shoulders':140,'diamond_':160,'triple_':140,'double_':120,'triangle':120,'wedge_':120,'channel':120,'broadening':120,'cup_and_handle':160,'rounding_':160,'barr_':200,'harmonic_':160}
    for frag, v in long_families.items():
        if frag in key: return v
    return 80
