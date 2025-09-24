
from __future__ import annotations

from loguru import logger
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ....patterns.registry import get_chart_detectors, get_candle_detectors
from ....ui.pattern_overlay import PatternEvent, PatternOverlayRenderer


class PatternsService:
    """
    Orchestrates detection and rendering of chart/candlestick patterns.
    Expect df with columns: ts_utc (ms), open, high, low, close[, volume]
    """
    def __init__(self, controller, view):
        self.ctrl = controller
        self.view = view
        self._df: Optional[pd.DataFrame] = None
        self._overlay = PatternOverlayRenderer()
        self._chart_enabled = False
        self._candle_enabled = False
        self._history_enabled = False

        # detectors
        self._chart_detectors = get_chart_detectors()
        self._candle_detectors = get_candle_detectors()

        # hook into plot axes if already available
        ax = getattr(self.ctrl, "axes", None) or getattr(self.view, "axes", None)
        if ax is not None:
            try:
                self._overlay.set_axes(ax)
            except Exception:
                pass

        self._log("PatternsService initialized")

    # ---------- toggles ----------

    def set_axes(self, ax):
        self._overlay.set_axes(ax)

    def set_chart_enabled(self, enabled: bool):
        self._chart_enabled = bool(enabled)
        self._log(f"Patterns: CHART toggle → {self._chart_enabled}")
        self._repaint()

    def set_candle_enabled(self, enabled: bool):
        self._candle_enabled = bool(enabled)
        self._log(f"Patterns: CANDLE toggle → {self._candle_enabled}")
        self._repaint()

    def set_history_enabled(self, enabled: bool):
        self._history_enabled = bool(enabled)
        self._log(f"Patterns: HISTORY toggle → {self._history_enabled}")
        self._repaint()

    # ---------- detection ----------

    def detect_async(self, df: Optional[pd.DataFrame]):
        shape = getattr(df, "shape", None)
        cols = list(df.columns) if isinstance(df, pd.DataFrame) else None
        self._log(f"Patterns.detect_async: shape={shape} cols={cols}", level="debug")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            self._df = None
            self._overlay.set_events([], mode="ms_epoch")
            return
        self._df = self._normalize_df(df)
        self._run_detection()

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # accept either ms epoch in ts_utc or datetime-like
        out = df.copy()
        if "ts_utc" not in out.columns:
            raise ValueError("PatternsService: df must contain 'ts_utc'")

        ts_col = out["ts_utc"]
        if np.issubdtype(ts_col.dtype, np.number):
            ts_ms = ts_col.astype("int64")
            out["ts_ms"] = ts_ms
            out["ts_dt"] = pd.to_datetime(ts_ms, unit="ms", utc=True)
        else:
            ts_dt = pd.to_datetime(ts_col, utc=True)
            out["ts_dt"] = ts_dt
            out["ts_ms"] = (ts_dt.view("int64") // 1_000_000).astype("int64")

        use_cols = ["ts_utc", "open", "high", "low", "close"]
        use_cols = [c for c in use_cols if c in out.columns]
        out = out[use_cols + ["ts_ms", "ts_dt"]]
        self._log(f"Patterns: normalized df rows={len(out)}; cols={use_cols} → using ['ts_utc','open','high','low','close']", level="info")
        return out

    # --- PATCH: dentro class PatternsService -----------------------------------

    def on_update_plot(self, df):
        """
        Entry-point invocato da ChartTabUI/PlotService quando viene plottato un nuovo df.
        Conserva il df, e se i toggle sono ON avvia la detection.
        """
        try:
            self._last_df = df
            if df is None:
                return
            # se nessun toggle attivo, non facciamo scanning
            if not (getattr(self, "_chart_enabled", False) or getattr(self, "_candle_enabled", False)):
                return
            # avvia lo scan con il df corrente
            self.detect_async(df)
        except Exception as ex:
            logger.error("PatternsService.on_update_plot failed: %s", ex)

    # --------------------------------------------------------------------------

    def _run_detection(self):
        if self._df is None:
            return
        if not (self._chart_enabled or self._candle_enabled):
            self._log("Patterns: toggles OFF → skipping detection")
            self._overlay.set_events([], mode="ms_epoch")
            return

        dets: List[Dict[str, Any]] = []
        kinds = []
        if self._chart_enabled:
            kinds.append("chart")
            for det in self._chart_detectors:
                try:
                    dets.extend(det.detect(self._df))
                except Exception:
                    continue
        if self._candle_enabled:
            kinds.append("candle")
            for det in self._candle_detectors:
                try:
                    dets.extend(det.detect(self._df))
                except Exception:
                    continue

        # Enrich events with trail for formation line and normalize fields
        events: List[PatternEvent] = []
        for d in dets:
            try:
                start_ts = int(d.get("start_ts") or d.get("start") or d.get("t_start"))
                confirm_ts = int(d.get("confirm_ts") or d.get("confirm") or d.get("t_confirm") or start_ts)
                end_ts = d.get("end_ts") or d.get("t_end") or None
                name = d.get("name") or d.get("pattern") or "Pattern"
                direction = (d.get("direction") or d.get("dir") or "neutral").lower()
                kind = d.get("kind") or ("candle" if d.get("is_candle") else "chart")
                confirm_price = d.get("confirm_price") or d.get("price") or d.get("p_confirm")
                target_price = d.get("target") or d.get("target_price")
                info_html = d.get("info_html")
                key = d.get("key") or f"{name}:{confirm_ts}"

                trail = d.get("trail")
                if trail is None:
                    # build trail from df closes between start and confirm
                    df = self._df
                    mask = (df["ts_ms"] >= start_ts) & (df["ts_ms"] <= confirm_ts)
                    seg = df.loc[mask, ["ts_ms", "close"]]
                    if not seg.empty:
                        trail = list(map(lambda r: (int(r["ts_ms"]), float(r["close"])), seg.to_dict("records")))

                ev = PatternEvent(
                    key=key, name=name, kind=kind, direction=direction,
                    start_ts=int(start_ts), confirm_ts=int(confirm_ts),
                    end_ts=int(end_ts) if end_ts is not None else None,
                    confirm_price=float(confirm_price) if confirm_price is not None else None,
                    target_price=float(target_price) if target_price is not None else None,
                    info_html=info_html, trail=trail
                )
                events.append(ev)
            except Exception:
                continue

        mode = "ms_epoch"  # our x is in ms
        self._log(f"Patterns: normalized df rows={len(self._df)}; detectors={len(self._chart_detectors) + len(self._candle_detectors)}", level="info")
        self._log(f"Patterns detected: total={len(events)} (chart={sum(1 for e in events if e.kind=='chart')}, candle={sum(1 for e in events if e.kind=='candle')})")
        self._overlay.set_events(events, mode=mode)

    def _repaint(self):
        # redraw last events
        self._overlay.draw(self._overlay._events or [])

    # ---------- util ----------

    def _log(self, msg: str, level: str = "info"):
        import logging
        logger = logging.getLogger(__name__)
        getattr(logger, level, logger.info)(msg)
