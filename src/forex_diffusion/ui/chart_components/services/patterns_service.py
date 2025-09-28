
from __future__ import annotations
from typing import List, Optional, Dict, Iterable
import pandas as pd
import numpy as np
from loguru import logger
from PySide6.QtCore import QTimer

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer

class _ScanWorker(QObject):
    produced = Signal(list)  # List[PatternEvent]

    def __init__(self, parent, kind: str, interval_ms: int) -> None:
        super().__init__()
        self._parent = parent
        self._kind = kind  # "chart" or "candle"
        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
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
            evs = self._parent._scan_once(kind=self._kind) or []
            self.produced.emit(evs)
        except Exception:
            pass


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

        self._busy = False
        self._pending_df: Optional[pd.DataFrame] = None
        self._debounce_timer: Optional[QTimer] = None
        self._debounce_ms = 30

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
            with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '..', '..', '..', 'configs', 'patterns.yaml'), 'r', encoding='utf-8') as fh:
                _cfg = yaml.safe_load(fh) or {}
        except Exception:
            _cfg = {}
        cur = (_cfg.get('patterns',{}).get('current_scan',{}) or {})
        minutes = int(cur.get('interval_minutes', 5))
        self._chart_worker = _ScanWorker(self, 'chart', minutes*60*1000)
        self._candle_worker = _ScanWorker(self, 'candle', minutes*60*1000)
        self._chart_worker.moveToThread(self._chart_thread)
        self._candle_worker.moveToThread(self._candle_thread)
        self._chart_thread.started.connect(self._chart_worker.start)
        self._candle_thread.started.connect(self._candle_worker.start)
        self._chart_worker.produced.connect(lambda evs: None)
        self._candle_worker.produced.connect(lambda evs: None)
        self._threads_started = False
        # Persistent cache (per symbol) to keep patterns across view reloads/zoom
        self._cache: Dict[tuple, object] = {}
        self._cache_symbol: Optional[str] = None
        # Multi-timeframe scan state
        self._scanned_tfs_by_symbol: Dict[str, set] = {}
        self._scanning_multi: bool = False
        # Dual background threads for scans
        self._chart_thread = QThread(self.view)
        self._candle_thread = QThread(self.view)
        # Load intervals from config
        import yaml, os
        try:
            with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '..', '..', '..', 'configs', 'patterns.yaml'), 'r', encoding='utf-8') as fh:
                _cfg = yaml.safe_load(fh) or {}
        except Exception:
            _cfg = {}
        cur = (_cfg.get('patterns',{}).get('current_scan',{}) or {})
        minutes = int(cur.get('interval_minutes', 5))
        self._chart_worker = _ScanWorker(self, 'chart', minutes*60*1000)
        self._candle_worker = _ScanWorker(self, 'candle', minutes*60*1000)
        self._chart_worker.moveToThread(self._chart_thread)
        self._candle_worker.moveToThread(self._candle_thread)
        self._chart_thread.started.connect(self._chart_worker.start)
        self._candle_thread.started.connect(self._candle_worker.start)
        self._chart_worker.produced.connect(lambda evs: None)
        self._candle_worker.produced.connect(lambda evs: None)
        self._threads_started = False

    def set_chart_enabled(self, on: bool):
        try:
            if on and not self._chart_thread.isRunning(): self._chart_thread.start()
            if (not on) and self._chart_thread.isRunning(): self._chart_worker.stop(); self._chart_thread.quit()
        except Exception: pass

        self._enabled_chart = bool(on)
        logger.info(f"Patterns: CHART toggle → {self._enabled_chart}")
        # start/stop background thread
        try:
            if self._enabled_chart and not self._threads_started:
                self._chart_thread.start(); self._threads_started = True
            if (not self._enabled_chart) and self._chart_thread.isRunning(): self._chart_worker.stop(); self._chart_thread.quit()
        except Exception: pass
        self._repaint()

        self._enabled_chart = bool(on)
        logger.info(f"Patterns: CHART toggle → {self._enabled_chart}")
        self._repaint()

    def set_candle_enabled(self, on: bool):
        try:
            if on and not self._candle_thread.isRunning(): self._candle_thread.start()
            if (not on) and self._candle_thread.isRunning(): self._candle_worker.stop(); self._candle_thread.quit()
        except Exception: pass

        self._enabled_candle = bool(on)
        logger.info(f"Patterns: CANDLE toggle → {self._enabled_candle}")
        try:
            if self._enabled_candle and not self._threads_started:
                self._candle_thread.start(); self._threads_started = True
            if (not self._enabled_candle) and self._candle_thread.isRunning(): self._candle_worker.stop(); self._candle_thread.quit()
        except Exception: pass
        self._repaint()

        self._enabled_candle = bool(on)
        logger.info(f"Patterns: CANDLE toggle → {self._enabled_candle}")
        self._repaint()

    def set_history_enabled(self, on: bool):
        self._enabled_history = bool(on)
        logger.info(f"Patterns: HISTORY toggle → {self._enabled_history}")
        self._repaint()

    def detect_async(self, df: Optional[pd.DataFrame]):
        logger.debug(f"Patterns.detect_async: shape={getattr(df, 'shape', None)} "
                     f"cols={list(df.columns)[:8] if hasattr(df, 'columns') else None}")
        logger.debug(f"Patterns.detect_async: received df={type(df).__name__ if df is not None else None}")
        self._pending_df = df
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
            self._run_detection(df)
        finally:
            self._busy = False
            if self._pending_df is not None:
                nxt = self._pending_df
                self._pending_df = None
                self.detect_async(nxt)

    def _run_detection(self, df: pd.DataFrame):
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

            evs: List[PatternEvent] = []
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



def _scan_once(self, kind: str):
    df = getattr(self.controller.plot_service, '_last_df', None)
    dfN = self._normalize_df(df)
    if dfN is None or len(dfN)==0: return []
    from src.forex_diffusion.patterns.registry import PatternRegistry
    from .patterns_adapter import enrich_events
    from ....patterns.info_provider import PatternInfoProvider
    reg = PatternRegistry(); dets = [d for d in reg.detectors([kind])]
    events=[]
    for d in dets:
        try: events.extend(d.detect(dfN))
        except Exception: pass
    events = enrich_events(events, PatternInfoProvider())
    self._events=(self._events or [])+events
    try: self._repaint()
    except Exception: pass
    return events

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
