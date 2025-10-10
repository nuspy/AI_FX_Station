from __future__ import annotations

from typing import Optional
import time

# matplotlib removed - using finplot for all charting
import numpy as np
import pandas as pd
from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from forex_diffusion.ui.prediction_settings_dialog import PredictionSettingsDialog

from .base import ChartServiceBase


class InteractionService(ChartServiceBase):
    """Auto-generated service extracted from ChartTab."""

    def _set_drawing_mode(self, mode: Optional[str]):
        self._drawing_mode = mode
        # Update drawing_manager tool if available
        if hasattr(self.view, 'drawing_manager') and self.view.drawing_manager:
            self.view.drawing_manager.set_tool(mode)
            logger.debug(f"Drawing mode set to: {mode}")
        if hasattr(self, '_pending_points'):
            self._pending_points.clear()

    def _on_canvas_click(self, event):
        """
        Drawing tools + TestingPoint:
        - Se drawing_mode attivo: disegna H-Line, Trend, Rect, Fib, Label.
        - Con Alt+Click: TestingPoint basic; Shift+Alt+Click: advanced.
        """
        try:
            if event is None or getattr(event, "button", None) != 1:
                return

            # If drawing mode is active handle first
            if self._drawing_mode and event.xdata is not None and event.ydata is not None:
                # Drawing tools temporarily disabled - will be reimplemented with finplot
                logger.debug(f"Drawing mode {self._drawing_mode} not yet implemented in finplot")
                return
                # TODO: Implement drawing tools with finplot API
                # if self._drawing_mode == "hline":
                #     fplt.add_line(y=event.ydata, color='#9bdcff', style='--')
                # if self._drawing_mode == "trend":
                #     if not hasattr(self, "_trend_points"):
                #         self._trend_points = []
                #     self._trend_points.append((event.xdata, event.ydata))
                #     if len(self._trend_points) == 2:
                #         (x1, y1), (x2, y2) = self._trend_points
                #         fplt.add_line([(x1, y1), (x2, y2)], color='#ff9bdc', width=1.5)
                #         self._trend_points = []

            # TODO: Re-implement all drawing tools with finplot
            # All drawing modes temporarily disabled until finplot implementation
            # if self._drawing_mode == "rect":
            # if self._drawing_mode == "fib":
            # if self._drawing_mode == "label":

            # TestingPoint logic (requires Alt)
            # GUI event gives access to modifiers
            modifiers = None
            try:
                gui = getattr(event, "guiEvent", None)
                if gui is not None:
                    modifiers = gui.modifiers()
            except Exception:
                modifiers = None

            from PySide6.QtCore import Qt
            alt_pressed = False
            shift_pressed = False
            if modifiers is not None:
                try:
                    alt_pressed = bool(modifiers & Qt.AltModifier)
                    shift_pressed = bool(modifiers & Qt.ShiftModifier)
                except Exception:
                    alt_pressed = False
                    shift_pressed = False

            if not alt_pressed:
                return  # only interested in Alt+click combos

            # convert xdata to utc ms
            try:
                from datetime import timezone
                # event.xdata is already a Unix timestamp in finplot
                if isinstance(event.xdata, (int, float)):
                    clicked_ms = int(event.xdata * 1000)
                else:
                    # Fallback: try to convert as pandas Timestamp
                    dt = pd.Timestamp(event.xdata)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    clicked_ms = int(dt.timestamp() * 1000)
            except Exception:
                logger.exception("Failed to convert click xdata to datetime")
                return

            if self._last_df is None or self._last_df.empty:
                QMessageBox.information(self.view, "No data", "No chart data available to create testing forecast.")
                return

            # anchor forecast to the last timestamp <= clicked_ms (floor on X)
            try:
                arr = self._last_df["ts_utc"].astype("int64").to_numpy()
                import numpy as np
                # ensure ascending order for robust floor search
                arr_sorted = np.sort(arr.astype(np.int64))
                pos = int(np.searchsorted(arr_sorted, int(clicked_ms), side="right") - 1)
                if pos < 0:
                    pos = 0
                testing_ts = int(arr_sorted[pos])
            except Exception:
                logger.exception("Failed to locate testing point in data")
                return

            # load number of history bars from settings
            from forex_diffusion.ui.prediction_settings_dialog import PredictionSettingsDialog
            settings = PredictionSettingsDialog.get_settings_from_file() or {}
            n_bars = int(settings.get("test_history_bars", 128))

            # build candles_override: take last n_bars ending at testing_ts (inclusive)
            df = self._last_df.sort_values("ts_utc").reset_index(drop=True)
            pos = df.index[df["ts_utc"].astype("int64") == testing_ts]
            if len(pos) == 0:
                # if exact ts not found, fall back to nearest less-or-equal index
                pos = df.index[df["ts_utc"].astype("int64") <= testing_ts]
                if len(pos) == 0:
                    QMessageBox.information(self.view, "Testing point", "No suitable testing point found at clicked X.")
                    return
                idx_pos = int(pos[-1])
            else:
                idx_pos = int(pos[0])

            start_idx = max(0, idx_pos - n_bars + 1)
            df_slice = df.iloc[start_idx: idx_pos + 1].copy()
            if df_slice.empty:
                QMessageBox.information(self.view, "Testing point", "Not enough historical bars available for testing.")
                return

            # Build payload: DON'T include candles_override - let worker fetch from DB
            # This ensures enough data for multi-timeframe indicators
            payload = {
                "symbol": getattr(self, "symbol", ""),
                "timeframe": getattr(self, "timeframe", ""),
                "testing_point_ts": int(testing_ts),
                "test_history_bars": int(n_bars),
                "anchor_price": float(event.ydata) if event.ydata is not None else None,
            }
            if shift_pressed:
                payload["advanced"] = True
            else:
                payload["advanced"] = False

            # also include model_paths and other settings to allow worker to run (use saved settings)
            saved = PredictionSettingsDialog.get_settings_from_file() or {}
            payload.update({
                "model_paths": saved.get("model_paths", []),  # Use model_paths for parallel inference
                "horizons": saved.get("horizons", ["1m", "5m", "15m"]),
                "N_samples": saved.get("N_samples", 200),
                "apply_conformal": saved.get("apply_conformal", True),
                "forecast_step": (self.pred_step_combo.currentText() if hasattr(self, "pred_step_combo") and self.pred_step_combo is not None else saved.get("forecast_step", "auto")),
                "parallel_inference": True,  # Enable parallel inference for multi-model support
                "warmup_bars": saved.get("warmup_bars", 16),
                "rv_window": saved.get("rv_window", 60),
                "min_feature_coverage": saved.get("min_feature_coverage", 0.05),
            })

            # emit forecast request (will not remove existing forecasts on chart)
            self.forecastRequested.emit(payload)

            # visual feedback (optional): mark testing point with a vertical line
            try:
                line = self.ax.axvline(pd.to_datetime(testing_ts, unit="ms", utc=True), color="gray", linestyle=":", alpha=0.6)
                # remove marker after brief time (keeps chart clean); store temporarily
                self.canvas.draw()
                # keep marker as part of forecasts so it can be cleared with clear_all_forecasts
                self._forecasts.append({"id": time.time(), "created_at": time.time(), "quantiles": {}, "future_ts": None, "artists": [line], "source": "testing_marker"})
                self._trim_forecasts()
            except Exception:
                pass

        except Exception as e:
            logger.exception(f"Error in canvas click handler: {e}")

    def _on_scroll_zoom(self, event):
        try:
            if event is None or event.inaxes != self.ax:
                return
            # fattore di zoom: up=in, down=out
            step = 0.85
            factor = step if getattr(event, "button", None) == "up" else (1.0 / step)
            cx = event.xdata
            cy = event.ydata
            self._zoom_axis("x", cx, factor)
            self._zoom_axis("y", cy, factor)
            try:
                self.canvas.draw_idle()
            except Exception:
                self.canvas.draw()
            # ricarica dati coerenti con il nuovo zoom
            self._schedule_view_reload()
        except Exception:
            pass

    def _on_mouse_press(self, event):
        try:
            if event is None or event.inaxes != self.ax:
                return
            # Left button: PAN (se nessun drawing tool attivo e nessun Alt)
            if getattr(event, "button", None) == 1:
                if not getattr(self, "_drawing_mode", None):
                    # evita conflitti coi testing point (Alt)
                    try:
                        if event.guiEvent and event.guiEvent.modifiers() & Qt.AltModifier:
                            return
                    except Exception:
                        pass
                    self._lbtn_pan = True
                    self._pan_last_xy = (event.xdata, event.ydata)
                    return
            # Right button: prepara zoom assiale
            if getattr(event, "button", None) == 3:
                self._rbtn_drag = True
                self._drag_last = (event.x, event.y)  # pixel coordinates
                self._drag_axis = None  # decideremo al primo movimento
        except Exception:
            pass

    def _on_mouse_move(self, event):
        try:
            if event is None or event.inaxes != self.ax:
                return
            # update adherence badges visibility on hover
            try:
                self._update_badge_visibility(event)
            except Exception:
                pass
            try:
                self._update_hover_info(event)
            except Exception:
                pass
            # PAN con LMB: sposta i limiti in base allo spostamento del cursore in coordinate dati
            if self._lbtn_pan and self._pan_last_xy is not None:
                x_last, y_last = self._pan_last_xy
                if event.xdata is None or event.ydata is None or x_last is None or y_last is None:
                    return
                dx = event.xdata - x_last
                dy = event.ydata - y_last
                xmin, xmax = self.ax.get_xlim()
                ymin, ymax = self.ax.get_ylim()
                # sposta inverso del movimento (trascinando a destra "porti" i dati verso sinistra)
                self.ax.set_xlim(xmin - dx, xmax - dx)
                self.ax.set_ylim(ymin - dy, ymax - dy)
                self._pan_last_xy = (event.xdata, event.ydata)
                try:
                    self.canvas.draw_idle()
                except Exception:
                    self.canvas.draw()
                return

            # ZOOM con RMB (asse X o Y)
            if self._rbtn_drag and self._drag_last is not None:
                x0, y0 = self._drag_last
                dx = event.x - x0
                dy = event.y - y0
                # determina asse predominante al primo movimento significativo
                if self._drag_axis is None:
                    if abs(dx) > abs(dy) * 1.2:
                        self._drag_axis = "x"
                    elif abs(dy) > abs(dx) * 1.2:
                        self._drag_axis = "y"
                    else:
                        return
                # mappa delta pixel -> fattore di zoom
                if self._drag_axis == "x":
                    factor = (0.90 ** (dx / 20.0)) if dx != 0 else 1.0
                    cx = event.xdata if event.xdata is not None else sum(self.ax.get_xlim()) / 2.0
                    self._zoom_axis("x", cx, factor)
                else:
                    factor = (0.90 ** (-dy / 20.0)) if dy != 0 else 1.0
                    cy = event.ydata if event.ydata is not None else sum(self.ax.get_ylim()) / 2.0
                    self._zoom_axis("y", cy, factor)
                self._drag_last = (event.x, event.y)
                try:
                    self.canvas.draw_idle()
                except Exception:
                    self.canvas.draw()
        except Exception:
            pass

    def _on_mouse_release(self, event):
        try:
            btn = getattr(event, "button", None)
            if btn == 3:
                self._rbtn_drag = False
                self._drag_last = None
                self._drag_axis = None
                # reload dati dopo zoom
                self._schedule_view_reload()
            if btn == 1:
                # Check for Alt+Click forecast trigger before clearing pan state
                try:
                    gui = getattr(event, "guiEvent", None)
                    if gui is not None:
                        modifiers = gui.modifiers()
                        if modifiers & Qt.AltModifier:
                            # Trigger forecast on Alt+Click
                            self._on_canvas_click(event)
                except Exception as e:
                    logger.debug(f"Alt+Click forecast check failed: {e}")

                self._lbtn_pan = False
                self._pan_last_xy = None
                # reload dati dopo pan
                self._schedule_view_reload()
        except Exception:
            pass

    def _zoom_axis(self, axis: str, center: float, factor: float):
        """Zoom helper: scala i limiti attorno a 'center' con 'factor' (factor<1=zoom-in)."""
        try:
            if axis == "x":
                xmin, xmax = self.ax.get_xlim()
                if center is None:
                    center = (xmin + xmax) * 0.5
                w = max(1e-9, (xmax - xmin))
                new_w = max(1e-9, w * float(factor))
                left = center - (center - xmin) * (new_w / w)
                right = center + (xmax - center) * (new_w / w)
                if right - left > 1e-12:
                    self.ax.set_xlim(left, right)
            elif axis == "y":
                ymin, ymax = self.ax.get_ylim()
                if center is None:
                    center = (ymin + ymax) * 0.5
                h = max(1e-12, (ymax - ymin))
                new_h = max(1e-12, h * float(factor))
                bottom = center - (center - ymin) * (new_h / h)
                top = center + (ymax - center) * (new_h / h)
                if top - bottom > 1e-12:
                    self.ax.set_ylim(bottom, top)
        except Exception:
            pass

    def _update_badge_visibility(self, event):
        """Hide badges when cursor is inside a rectangle centered on the badge with 2x width/height; show otherwise."""
        try:
            if not hasattr(self, "_adh_badges") or not self._adh_badges:
                return
            renderer = self.canvas.get_renderer()
            if renderer is None:
                # ensure a valid renderer for correct bbox
                try:
                    self.canvas.draw()
                    renderer = self.canvas.get_renderer()
                except Exception:
                    return
            changed = False
            for art in list(self._adh_badges):
                try:
                    if art.figure is None:
                        # dropped artist; cleanup
                        self._adh_badges.remove(art)
                        continue
                    bbox = art.get_window_extent(renderer=renderer)
                    # Rectangle with same center, doubled width/height
                    x0, y0 = float(bbox.x0), float(bbox.y0)
                    x1, y1 = float(bbox.x1), float(bbox.y1)
                    w = max(1.0, x1 - x0)
                    h = max(1.0, y1 - y0)
                    cx = (x0 + x1) * 0.5
                    cy = (y0 + y1) * 0.5
                    # doubled dimensions -> half sizes equal to original full sizes
                    hx = w
                    hy = h
                    ex0, ex1 = cx - hx, cx + hx
                    ey0, ey1 = cy - hy, cy + hy
                    # strict inequalities: inside only if strictly within area, not only touching border
                    inside = (ex0 < float(event.x) < ex1) and (ey0 < float(event.y) < ey1)
                    vis = not inside
                    if art.get_visible() != vis:
                        art.set_visible(vis)
                        changed = True
                except Exception:
                    continue
            if changed:
                try:
                    self.canvas.draw_idle()
                except Exception:
                    self.canvas.draw()
        except Exception:
            pass

    def _on_nav_home(self):
        try:
            self.toolbar.home()
        except Exception:
            pass

    def _on_nav_pan(self, checked: bool):
        try:
            # disattiva zoom se pan attivo
            if checked and hasattr(self, "tb_zoom"):
                self.tb_zoom.setChecked(False)
            self.toolbar.pan()
        except Exception:
            pass

    def _on_nav_zoom(self, checked: bool):
        try:
            # disattiva pan se zoom attivo
            if checked and hasattr(self, "tb_pan"):
                self.tb_pan.setChecked(False)
            self.toolbar.zoom()
        except Exception:
            pass
