from __future__ import annotations
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from loguru import logger
from PySide6.QtCore import QTimer

from .base import ChartServiceBase
from ....patterns.registry import PatternRegistry
from ....patterns.engine import PatternEvent
from ....patterns.info_provider import PatternInfoProvider
from ...pattern_overlay import PatternOverlayRenderer

OHLC_SYNONYMS: Dict[str, str] = {
    'o':'open','op':'open','open':'open','bidopen':'open','askopen':'open',
    'h':'high','hi':'high','high':'high','bidhigh':'high','askhigh':'high',
    'l':'low','lo':'low','low':'low','bidlow':'low','asklow':'low',
    'c':'close','cl':'close','close':'close','bidclose':'close','askclose':'close','last':'close','price':'close','mid':'close'
}
TS_SYNONYMS = ['ts_utc','timestamp','time','ts','datetime','date','dt','ts_ms','ts_ns']

class PatternsService(ChartServiceBase):
    """Service Qt: esegue detection pattern e disegna overlay (debounced, no re-entrata)."""
    def __init__(self, view, controller) -> None:
        super().__init__(view, controller)
        self._enabled_chart = False
        self._enabled_candle = False
        self._enabled_history = False
        self._events: List[PatternEvent] = []
        self.registry = PatternRegistry()
        self.info = PatternInfoProvider(self._default_info_path())
        self.renderer = PatternOverlayRenderer(controller, self.info)

        # Debounce / anti re-entrancy
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

    # ----- toggles -----
    def set_chart_enabled(self, on: bool):
        self._enabled_chart = bool(on)
        from loguru import logger
        logger.info(f"Patterns: CHART toggle → {self._enabled_chart}")
        self._repaint()

    def set_candle_enabled(self, on: bool):
        self._enabled_candle = bool(on)
        from loguru import logger;
        logger.info(f"Patterns: CANDLE toggle → {self._enabled_candle}")
        self._repaint()

    def set_history_enabled(self, on: bool):
        self._enabled_history = bool(on)
        from loguru import logger;
        logger.info(f"Patterns: HISTORY toggle → {self._enabled_history}")
        self._repaint()

    # ----- ingresso non bloccante dal hook -----
    def detect_async(self, df: Optional[pd.DataFrame]):

        # dentro detect_async, subito all’inizio
        from loguru import logger
        logger.debug(f"Patterns.detect_async: shape={getattr(df, 'shape', None)} "
                     f"cols={list(df.columns)[:8] if hasattr(df, 'columns') else None}")

        from loguru import logger
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

    # ----- detection “vera” -----
    def _run_detection(self, df: pd.DataFrame):
        from loguru import logger
        try:
            # 1) Verifica toggle
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

            # 2) Se richiesto, usa storico completo dal data_service
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

            # 3) Fallback al last_df del plot
            if df is None:
                df = getattr(self.controller.plot_service, "_last_df", None)

            # 4) Normalizzazione colonne
            dfN = self._normalize_df(df)
            if dfN is None or dfN.empty:
                logger.warning("Patterns: normalization failed or empty → no detection")
                self._events.clear()
                self.renderer.clear()
                return

            # 5) Colonna datetime (robusta su Series/Index)
            try:
                ts = pd.to_datetime(dfN['ts_utc'], unit='ms', utc=True)
                try:
                    dfN['ts_dt'] = ts.dt.tz_convert(None)
                except AttributeError:
                    dfN['ts_dt'] = ts.tz_convert(None)
            except Exception:
                pass

            # 6) Materializza i detector UNA volta e logga conteggio
            dets = list(self.registry.detectors(kinds=kinds))
            logger.info(f"Patterns: normalized df rows={len(dfN)}; detectors={len(dets)}")

            # 7) Detection
            evs = []
            for det in dets:
                try:
                    evs.extend(det.detect(dfN))
                except Exception as e:
                    logger.debug(f"Detector {getattr(det, 'key', '?')} failed: {e}")

            # 8) Risultato + disegno overlay
            logger.info(
                f"Patterns detected: total={len(evs)} "
                f"(chart={sum(1 for x in evs if getattr(x, 'kind', '') == 'chart')}, "
                f"candle={sum(1 for x in evs if getattr(x, 'kind', '') == 'candle')})"
            )

            self._events = self._enrich_events_for_plot(dfN, evs)
            self.renderer.draw(self._events)


        except Exception as e:
            logger.exception("Patterns _run_detection failed: {}", e)

    def _nearest_close_at_ts(self, dfN, ts_ms: int) -> float | None:
        import numpy as np
        try:
            ts_arr = dfN["ts_utc"].to_numpy(dtype="int64")
            i = np.searchsorted(ts_arr, ts_ms, side="left")
            j = 0 if i == 0 else (len(ts_arr) - 1 if i >= len(ts_arr) else (
                i if (ts_ms - ts_arr[i - 1]) > (ts_arr[i] - ts_ms) else i - 1))
            return float(dfN["close"].iloc[j])
        except Exception:
            return None

    def _fallback_amplitude(self, dfN, e) -> float | None:
        """Ampiezza del pattern per stimare il target."""
        import numpy as np
        # 1) upper/lower
        up = getattr(e, "upper", None)
        lo = getattr(e, "lower", None)
        try:
            if up is not None and lo is not None:
                return float(abs(up - lo))
        except Exception:
            pass
        # 2) bounds da indici
        for s_name, e_name in (("start_idx", "end_idx"), ("i_start", "i_end"), ("confirm_idx", "end_idx")):
            s = getattr(e, s_name, None)
            t = getattr(e, e_name, None)
            if isinstance(s, int) and isinstance(t, int) and 0 <= s < t < len(dfN):
                try:
                    w_hi = float(dfN["high"].iloc[s:t + 1].max())
                    w_lo = float(dfN["low"].iloc[s:t + 1].min())
                    return abs(w_hi - w_lo)
                except Exception:
                    pass
        # 3) ATR di default (se presente)
        if "atr" in dfN.columns:
            try:
                return float(np.nanmedian(dfN["atr"].tail(100)))
            except Exception:
                pass
        # 4) high-low della finestra recente
        try:
            w_hi = float(dfN["high"].tail(60).max())
            w_lo = float(dfN["low"].tail(60).min())
            return abs(w_hi - w_lo) * 0.5
        except Exception:
            return None

    def _enrich_events_for_plot(self, dfN, events: list) -> list:
        import pandas as pd
        enriched = []
        for e in events:
            # --- timestamp ms ---
            ts = getattr(e, "confirm_ts", None) or getattr(e, "end_ts", None) or getattr(e, "ts", None)
            if ts is None:
                idx = getattr(e, "confirm_idx", None)
                if isinstance(idx, int) and 0 <= idx < len(dfN): ts = int(dfN["ts_utc"].iloc[idx])
            try:
                ts_ms = int(pd.to_datetime(ts, utc=True).value // 1_000_000)
            except Exception:
                try:
                    ts_ms = int(ts)
                except Exception:
                    continue
            try:
                setattr(e, "confirm_ts", ts_ms)
            except Exception:
                pass

            # --- prezzo di conferma ---
            px = getattr(e, "confirm_price", None) or getattr(e, "price", None)
            if px is None: px = self._nearest_close_at_ts(dfN, ts_ms)
            try:
                px = float(px)
                setattr(e, "confirm_price", px)
            except Exception:
                continue

            # --- direzione ---
            direction = getattr(e, "direction", None)
            if direction is None:
                try:
                    # se pattern_key suggerisce (es. "...Top" → down, "...Bottom" → up)
                    key = str(getattr(e, "key", getattr(e, "name", ""))).lower()
                    if "top" in key or "descending" in key:
                        direction = "down"
                    elif "bottom" in key or "ascending" in key:
                        direction = "up"
                    else:
                        direction = "neutral"
                except Exception:
                    direction = "neutral"
                try:
                    setattr(e, "direction", direction)
                except Exception:
                    pass

            # --- target ---
            tgt = getattr(e, "target_price", None)
            if tgt is None:
                amp = self._fallback_amplitude(dfN, e)
                if amp is not None:
                    if str(direction).lower().startswith("down"):
                        tgt = px - amp
                    else:
                        tgt = px + amp
            if tgt is not None:
                try:
                    setattr(e, "target_price", float(tgt))
                except Exception:
                    pass

            enriched.append(e)
        return enriched

    # compat per eventuali chiamate dirette
    def on_update_plot(self, df: pd.DataFrame):
        self.detect_async(df)

    # ----- utils -----
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

    def _connect_ui_signals(self) -> None:
        # --- collegamenti UI esistenti ---
        connections = {
            # Topbar
            getattr(self, "symbol_combo", None): ("currentTextChanged", self._on_symbol_combo_changed),
            getattr(self, "tf_combo", None): ("currentTextChanged", self._on_timeframe_changed),
            getattr(self, "pred_step_combo", None): ("currentTextChanged", self._on_pred_step_changed),
            getattr(self, "years_combo", None): ("currentTextChanged", self._on_backfill_range_changed),
            getattr(self, "months_combo", None): ("currentTextChanged", self._on_backfill_range_changed),
            getattr(self, "theme_combo", None): ("currentTextChanged", self._on_theme_changed),
            getattr(self, "settings_btn", None): ("clicked", self._open_settings_dialog),
            getattr(self, "backfill_btn", None): ("clicked", self.chart_controller.on_backfill_missing_clicked),
            getattr(self, "indicators_btn", None): ("clicked", self.chart_controller.on_indicators_clicked),
            getattr(self, "build_latents_btn", None): ("clicked", self.chart_controller.on_build_latents_clicked),
            getattr(self, "forecast_settings_btn", None): ("clicked", self.chart_controller.open_forecast_settings),
            getattr(self, "forecast_btn", None): ("clicked", self.chart_controller.on_forecast_clicked),
            getattr(self, "adv_settings_btn", None): ("clicked", self.chart_controller.open_adv_forecast_settings),
            getattr(self, "adv_forecast_btn", None): ("clicked", self.chart_controller.on_advanced_forecast_clicked),
            getattr(self, "clear_forecasts_btn", None): ("clicked", self.chart_controller.clear_all_forecasts),
            getattr(self, "toggle_drawbar_btn", None): ("toggled", self._toggle_drawbar),
            getattr(self, "mode_btn", None): ("toggled", self._on_price_mode_toggled),
            getattr(self, "follow_checkbox", None): ("toggled", self._on_follow_toggled),
            getattr(self, "trade_btn", None): ("clicked", self.chart_controller.open_trade_dialog),
            # Drawbar
            getattr(self, "tb_orders", None): ("toggled", self._toggle_orders),
        }
        for widget, (signal, handler) in connections.items():
            if widget:
                getattr(widget, signal).connect(handler)

        draw_buttons = {
            getattr(self, "tb_cross", None): None,
            getattr(self, "tb_hline", None): "hline",
            getattr(self, "tb_trend", None): "trend",
            getattr(self, "tb_rect", None): "rect",
            getattr(self, "tb_fib", None): "fib",
            getattr(self, "tb_label", None): "label",
        }
        for button, mode in draw_buttons.items():
            if button:
                button.clicked.connect(lambda checked=False, m=mode: self._set_drawing_mode(m))

        for key, splitter in self._get_splitters().items():
            if splitter:
                splitter.splitterMoved.connect(
                    lambda _p, _i, k=key, s=splitter: self._persist_splitter_positions(k, s)
                )

        # --- wiring toggles "Patterns" tramite hook sicuro (nessun attributo sul controller) ---
        try:
            # lazy import per evitare dipendenze circolari in fase di import del modulo
            from ..services.patterns_hook import (
                get_patterns_service,
                set_patterns_toggle,
            )
            # inizializza lazy il service senza toccare attributi del controller
            ctrl = self.chart_controller
            get_patterns_service(ctrl, self, create=True)

            # Connetti i tre toggle, se presenti
            if getattr(self, "chart_patterns_checkbox", None):
                self.chart_patterns_checkbox.toggled.connect(
                    lambda v, c=ctrl: set_patterns_toggle(c, self, chart=bool(v))
                )
            if getattr(self, "candle_patterns_checkbox", None):
                self.candle_patterns_checkbox.toggled.connect(
                    lambda v, c=ctrl: set_patterns_toggle(c, self, candle=bool(v))
                )
            if getattr(self, "history_patterns_checkbox", None):
                self.history_patterns_checkbox.toggled.connect(
                    lambda v, c=ctrl: set_patterns_toggle(c, self, history=bool(v))
                )

            from ...chart_components.services.patterns_hook import call_patterns_detection, get_patterns_service

            try:
                # prendi un df dall'engine di plot
                ps = getattr(self, "chart_controller", None)
                plot = getattr(ps, "plot_service", None)
                df_now = getattr(plot, "_last_df", None)
                call_patterns_detection(ps, self, df_now)
            except Exception as e:
                from loguru import logger
                logger.debug(f"Patterns immediate scan failed: {e}")

            # Sincronizza immediatamente lo stato iniziale (nessun flicker, nessun freeze)
            init_chart = bool(
                getattr(self, "chart_patterns_checkbox", None) and self.chart_patterns_checkbox.isChecked())
            init_candle = bool(
                getattr(self, "candle_patterns_checkbox", None) and self.candle_patterns_checkbox.isChecked())
            init_history = bool(
                getattr(self, "history_patterns_checkbox", None) and self.history_patterns_checkbox.isChecked())
            if any([init_chart, init_candle, init_history]):
                set_patterns_toggle(ctrl, self, chart=init_chart, candle=init_candle, history=init_history)

        except Exception as e:
            try:
                from loguru import logger
                logger.debug(f"Patterns toggles wiring failed: {e}")
            except Exception:
                pass

    def _repaint(self):
        self.renderer.draw(self._events or [])
