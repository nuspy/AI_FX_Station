
from __future__ import annotations
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from loguru import logger
from PySide6.QtCore import QTimer

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer

class _ScanWorker(QObject):
    produced = Signal(list)  # List[PatternEvent]

    def __init__(self, parent, kind: str, interval_ms: int) -> None:
        super().__init__(parent)
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
from ....patterns.registry import PatternRegistry
from ....patterns.engine import PatternEvent
from ....patterns.info_provider import PatternInfoProvider
from ...pattern_overlay import PatternOverlayRenderer

from .patterns_adapter import enrich_events

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

            dets = list(self.registry.detectors(kinds=kinds))
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

