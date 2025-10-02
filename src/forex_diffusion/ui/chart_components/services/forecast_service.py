from __future__ import annotations

from typing import Any, Dict, Optional
import time
from pathlib import Path

# matplotlib removed - using finplot for all charting
import numpy as np
import pandas as pd
from loguru import logger
from PySide6.QtWidgets import QMessageBox

from forex_diffusion.ui.prediction_settings_dialog import PredictionSettingsDialog
from forex_diffusion.utils.user_settings import get_setting

from .base import ChartServiceBase
from .draggable_legend import DraggableLegend


def get_forecast_service(controller, view, create=True):
    """Get or create ForecastService instance."""
    if hasattr(view, '_forecast_service_instance'):
        return view._forecast_service_instance
    elif create:
        instance = ForecastService(view, controller)
        view._forecast_service_instance = instance
        return instance
    return None


class ForecastService(ChartServiceBase):
    """Auto-generated service extracted from ChartTab."""

    # Color palette: 100 vivid, distinct colors (excluding black)
    _COLOR_PALETTE = [
        '#2196F3', '#E91E63', '#00BCD4', '#FFC107', '#FF5722', '#9C27B0', '#4CAF50', '#FF9800',
        '#03A9F4', '#F44336', '#00E676', '#FFEB3B', '#673AB7', '#8BC34A', '#FF6F00', '#E040FB',
        '#00ACC1', '#FF1744', '#76FF03', '#FDD835', '#5E35B1', '#C0CA33', '#FF3D00', '#D500F9',
        '#0097A7', '#D32F2F', '#64DD17', '#F9A825', '#512DA8', '#AFB42B', '#DD2C00', '#AA00FF',
        '#00838F', '#C62828', '#00C853', '#F57F17', '#4527A0', '#9E9D24', '#BF360C', '#6200EA',
        '#006064', '#B71C1C', '#00E676', '#F57C00', '#311B92', '#827717', '#BF360C', '#304FFE',
        '#0091EA', '#FF6D00', '#00B8D4', '#FFD600', '#6200EA', '#C6FF00', '#DD2C00', '#2962FF',
        '#00B0FF', '#FF3D00', '#18FFFF', '#FFEA00', '#AA00FF', '#AEEA00', '#D50000', '#2979FF',
        '#00E5FF', '#FF9100', '#1DE9B6', '#FFC400', '#D500F9', '#76FF03', '#FF1744', '#448AFF',
        '#84FFFF', '#FFAB00', '#00E676', '#FFAB00', '#E040FB', '#64DD17', '#F50057', '#536DFE',
        '#A7FFEB', '#FF6D00', '#69F0AE', '#FFD740', '#D500F9', '#C6FF00', '#FF5252', '#7C4DFF',
        '#CCFF90', '#FFAB40', '#00E676', '#FFFF00', '#E040FB', '#F4FF81', '#FF1744', '#536DFE',
        '#80D8FF', '#FF9E80', '#1DE9B6', '#F4FF81'
    ]

    # Class-level mapping: model_path -> color_index (persists across instances)
    _model_color_mapping = {}

    def _ensure_legend(self):
        """Forecast items are automatically added to main chart legend via 'name' parameter."""
        # PyQtGraph automatically handles legend via plot(..., name=model_name)
        # No separate legend widget needed
        pass

    def _update_legend(self, model_path: str, model_name: str, color: str):
        """Forecast items are automatically added to main chart legend via 'name' parameter."""
        # PyQtGraph automatically handles legend via plot(..., name=model_name)
        # No manual legend update needed
        pass

    def _get_color_for_model(self, model_path: str) -> str:
        """
        Get consistent color for a model based on its path.

        Args:
            model_path: Full path to the model file

        Returns:
            Hex color string
        """
        # Check if model already has a color assigned
        if model_path in self._model_color_mapping:
            color_idx = self._model_color_mapping[model_path]
            return self._COLOR_PALETTE[color_idx % len(self._COLOR_PALETTE)]

        # Find first unused color
        used_indices = set(self._model_color_mapping.values())
        for i in range(len(self._COLOR_PALETTE)):
            if i not in used_indices:
                self._model_color_mapping[model_path] = i
                logger.debug(f"Assigned color #{i} to model {Path(model_path).name}")
                return self._COLOR_PALETTE[i]

        # If all colors used, cycle back (shouldn't happen with 100 colors)
        fallback_idx = len(self._model_color_mapping) % len(self._COLOR_PALETTE)
        self._model_color_mapping[model_path] = fallback_idx
        return self._COLOR_PALETTE[fallback_idx]

    def _plot_forecast_overlay(self, quantiles: dict, source: str = "basic"):
        """
        Plot quantiles on the chart. quantiles expected to have keys 'q50','q05','q95'
        Each value can be a list/array of floats. Optionally 'future_ts' can provide UTC ms or datetimes.
        """
        try:
            # extract and coerce to arrays
            q50 = quantiles.get("q50")
            q05 = quantiles.get("q05") or quantiles.get("q10")
            q95 = quantiles.get("q95") or quantiles.get("q90")
            label = str(quantiles.get("label", f"{source}"))

            if q50 is None:
                return

            import numpy as np
            q50_arr = np.asarray(q50, dtype=float).flatten()
            q05_arr = np.asarray(q05, dtype=float).flatten() if q05 is not None else None
            q95_arr = np.asarray(q95, dtype=float).flatten() if q95 is not None else None

            # build x positions from future_ts or from last ts + tf
            future_ts = quantiles.get("future_ts", None)
            x_vals = None
            if future_ts is not None:
                # accetta lista di ms o datetime-like
                try:
                    x_vals = pd.to_datetime(future_ts, unit="ms", utc=True)
                except Exception:
                    x_vals = pd.to_datetime(future_ts, utc=True)
            else:
                # fallback: deriva da ultimo ts e TF corrente
                if self._last_df is None or self._last_df.empty:
                    return
                try:
                    last_ts = pd.to_datetime(self._last_df["ts_utc"].astype("int64"), unit="ms", utc=True).iat[-1]
                except Exception:
                    last_ts = pd.Timestamp.utcnow().tz_localize("UTC")
                td = self._tf_to_timedelta(getattr(self, "timeframe", "1m"))
                x_vals = [last_ts + td * (i + 1) for i in range(len(q50_arr))]

            # normalize to naive datetimes for Matplotlib
            try:
                x_vals = pd.to_datetime(x_vals, utc=True).tz_localize(None)
            except Exception:
                try:
                    x_vals = pd.to_datetime(x_vals).tz_localize(None)
                except Exception:
                    pass

            # align lengths
            n = min(len(x_vals), len(q50_arr))
            if n == 0:
                return
            if len(q50_arr) != n:
                q50_arr = q50_arr[:n]
            if q05_arr is not None and len(q05_arr) != n:
                q05_arr = q05_arr[:n]
            if q95_arr is not None and len(q95_arr) != n:
                q95_arr = q95_arr[:n]
            if len(x_vals) != n:
                x_vals = x_vals[:n]

            # prepend point 0 (t0 = requested_at_ms) con anchor_price o last_close
            try:
                req_ms = quantiles.get("requested_at_ms", None)
                if req_ms is not None:
                    t0 = pd.to_datetime(int(req_ms), unit="ms", utc=True).tz_convert(None)
                    # Use anchor_price if provided (from Alt+Click Y coordinate), else use last_close
                    anchor_price = quantiles.get("anchor_price")
                    if anchor_price is not None:
                        last_close = float(anchor_price)
                    else:
                        last_close = float(self._last_df["close"].iat[-1] if "close" in self._last_df.columns else self._last_df["price"].iat[-1])
                    logger.debug(f"Prepending anchor point: t0={t0}, price={last_close}")
                    x_vals = pd.DatetimeIndex([t0]).append(pd.DatetimeIndex(x_vals))
                    import numpy as _np
                    q50_arr = _np.insert(q50_arr, 0, last_close)
                    if q05_arr is not None: q05_arr = _np.insert(q05_arr, 0, last_close)
                    if q95_arr is not None: q95_arr = _np.insert(q95_arr, 0, last_close)
                    logger.debug(f"After prepend: x_vals len={len(x_vals)}, q50 len={len(q50_arr)}")
                else:
                    logger.warning(f"No requested_at_ms in quantiles - forecast won't connect to price line")
            except Exception as e:
                logger.warning(f"Failed to prepend anchor point: {e}")

            # Get consistent color for this model
            model_path = quantiles.get("model_path_used", quantiles.get("model_path", source))
            color = self._get_color_for_model(model_path)

            # nome modello per legenda (mostra una sola volta)
            try:
                model_name = str(quantiles.get("label") or quantiles.get("source") or label)
            except Exception:
                model_name = str(label)
            if not hasattr(self, "_legend_once") or self._legend_once is None:
                self._legend_once = set()
            first_for_model = model_name not in self._legend_once
            if first_for_model:
                self._legend_once.add(model_name)

            # Convert datetime x_vals to numeric timestamps for PyQtGraph
            import pyqtgraph as pg
            if isinstance(x_vals, pd.DatetimeIndex):
                x_numeric = x_vals.astype('int64') // 10**9  # Convert to Unix timestamp (seconds)
            elif hasattr(x_vals, '__iter__'):
                try:
                    x_numeric = np.array([pd.Timestamp(x).timestamp() for x in x_vals])
                except Exception:
                    x_numeric = np.arange(len(x_vals))
            else:
                x_numeric = np.arange(len(q50_arr))

            logger.debug(f"Forecast: {len(x_numeric)} points, x range: {x_numeric[0]:.1f} to {x_numeric[-1]:.1f}")

            # Update legend with this model
            self._update_legend(model_path, model_name, color)

            # linea di previsione con marker sui punti (etichetta solo la prima volta per il modello)
            # PyQtGraph returns single PlotDataItem, not tuple like matplotlib
            line50 = self.ax.plot(
                x_numeric, q50_arr,
                pen=pg.mkPen(color, width=2),
                symbol='o', symbolSize=3.5,
                symbolBrush=pg.mkBrush(color), symbolPen=pg.mkPen(color),
                name=(model_name if first_for_model else None)
            )
            artists = [line50]

            # Se abilitato, calcola/disegna indicatori anche sulla porzione di forecast
            try:
                cfg = self._get_indicator_settings() or {}
            except Exception:
                cfg = {}
            try:
                ind_on_fcst = bool(cfg.get("ind_on_forecast", get_setting("indicators.on_forecast", False)))
            except Exception:
                ind_on_fcst = False
            if ind_on_fcst:
                try:
                    y_col = "close" if "close" in self._last_df.columns else "price"
                    hist = self._last_df[["ts_utc", y_col]].dropna().copy()
                    hist["ts_utc"] = pd.to_numeric(hist["ts_utc"], errors="coerce").astype("int64")
                    hist = hist.sort_values("ts_utc").reset_index(drop=True)
                    x_hist = pd.to_datetime(hist["ts_utc"], unit="ms", utc=True).tz_convert(None)
                    close_hist = pd.to_numeric(hist[y_col], errors="coerce").astype(float)

                    n_sma = int(cfg.get("sma_n", 20)) if cfg.get("use_sma", False) else 0
                    n_ef = int(cfg.get("ema_fast", 12)) if cfg.get("use_ema", False) else 0
                    n_es = int(cfg.get("ema_slow", 26)) if cfg.get("use_ema", False) else 0
                    n_bb = int(cfg.get("bb_n", 20)) if cfg.get("use_bollinger", False) else 0
                    n_k = int(cfg.get("bb_n", 20)) if cfg.get("use_keltner", False) else 0
                    nmax = max(1, n_sma, n_ef, n_es, n_bb, n_k)
                    tail = max(100, nmax * 4)
                    close_hist_tail = close_hist.iloc[-tail:] if len(close_hist) > tail else close_hist

                    import numpy as _np
                    close_all = _np.concatenate([close_hist_tail.values, q50_arr])
                    f_start = len(close_all) - len(q50_arr)

                    # SMA (PyQtGraph conversion)
                    if cfg.get("use_sma", False) and n_sma > 0:
                        sma_all = self._sma(pd.Series(close_all), n_sma).to_numpy()
                        try:
                            ln = self.ax.plot(x_numeric, sma_all[f_start:],
                                            pen={'color': cfg.get("color_sma", "#7f7f7f"),
                                                 'width': 1.0})  # Removed style - use default solid line
                            artists.append(ln)
                        except Exception:
                            pass
                    # EMA (PyQtGraph conversion)
                    if cfg.get("use_ema", False):
                        if n_ef > 0:
                            emaf_all = self._ema(pd.Series(close_all), n_ef).to_numpy()
                            try:
                                ln = self.ax.plot(x_numeric, emaf_all[f_start:],
                                                pen={'color': cfg.get("color_ema", "#bcbd22"),
                                                     'width': 1.0})
                                artists.append(ln)
                            except Exception:
                                pass
                        if n_es > 0:
                            emas_all = self._ema(pd.Series(close_all), n_es).to_numpy()
                            try:
                                ln = self.ax.plot(x_numeric, emas_all[f_start:],
                                                pen={'color': cfg.get("color_ema", "#bcbd22"),
                                                     'width': 1.0})
                                artists.append(ln)
                            except Exception:
                                pass
                    # Bollinger (PyQtGraph conversion - simplified, no fill_between)
                    if cfg.get("use_bollinger", False) and n_bb > 0:
                        _, bb_up_all, bb_lo_all = self._bollinger(pd.Series(close_all), n_bb, float(cfg.get("bb_k", 2.0)))
                        bu = bb_up_all.to_numpy()[f_start:]
                        bl = bb_lo_all.to_numpy()[f_start:]
                        c_bb = cfg.get("color_bollinger", "#2ca02c")
                        try:
                            l1 = self.ax.plot(x_numeric, bu, pen={'color': c_bb, 'width': 0.9})
                            l2 = self.ax.plot(x_numeric, bl, pen={'color': c_bb, 'width': 0.9})
                            artists.extend([l1, l2])
                        except Exception:
                            pass
                    # Keltner (PyQtGraph conversion)
                    if cfg.get("use_keltner", False) and n_k > 0:
                        high_all = pd.Series(close_all)
                        low_all = pd.Series(close_all)
                        ku_all, kl_all = self._keltner(high_all, low_all, pd.Series(close_all), n_k, float(cfg.get("keltner_k", 1.5)))
                        ku = (ku_all.to_numpy() if hasattr(ku_all, "to_numpy") else ku_all.values)[f_start:]
                        kl = (kl_all.to_numpy() if hasattr(kl_all, "to_numpy") else kl_all.values)[f_start:]
                        c_k = cfg.get("color_keltner", "#17becf")
                        try:
                            l1 = self.ax.plot(x_numeric, ku, pen={'color': c_k, 'width': 0.9})
                            l2 = self.ax.plot(x_numeric, kl, pen={'color': c_k, 'width': 0.9})
                            artists.extend([l1, l2])
                        except Exception:
                            pass
                except Exception:
                    pass

            # se disponibili, punti ad alta risoluzione (es. 1m) come scatter (PyQtGraph conversion)
            try:
                f_hr = quantiles.get("future_ts_hr"); q50_hr = quantiles.get("q50_hr")
                if f_hr and q50_hr and len(f_hr) == len(q50_hr):
                    x_hr_numeric = np.arange(len(f_hr))
                    import pyqtgraph as pg
                    scat = pg.ScatterPlotItem(x=x_hr_numeric, y=q50_hr, size=10,
                                            brush=pg.mkBrush(color), pen=None)
                    self.ax.addItem(scat)
                    artists.append(scat)
            except Exception:
                pass

            # evidenzia il punto 0 (istante richiesta) con marker più grande (PyQtGraph conversion)
            try:
                req_ms = quantiles.get("requested_at_ms", None)
                if req_ms is not None and len(q50_arr) > 0:
                    import pyqtgraph as pg
                    m0 = pg.ScatterPlotItem(x=[0], y=[float(q50_arr[0])], size=28,
                                          brush=pg.mkBrush(color), pen=pg.mkPen('w', width=0.8))
                    self.ax.addItem(m0)
                    artists.append(m0)
            except Exception:
                pass

            # Quantiles (PyQtGraph with filled area between q05 and q95)
            if q05_arr is not None and q95_arr is not None:
                from PySide6.QtCore import Qt
                from PySide6.QtGui import QColor

                # Parse color to get RGB values
                if isinstance(color, tuple) and len(color) >= 3:
                    r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                else:
                    # Fallback: try to parse as hex or use default
                    try:
                        if isinstance(color, str) and color.startswith('#'):
                            qcolor = QColor(color)
                            r, g, b = qcolor.red(), qcolor.green(), qcolor.blue()
                        else:
                            r, g, b = 33, 150, 243  # Default blue
                    except Exception:
                        r, g, b = 33, 150, 243

                # Create semi-transparent colors for borders (alpha=180)
                border_color = QColor(r, g, b, 180)
                # Create fill color (alpha=100)
                fill_color = QColor(r, g, b, 100)

                # Draw filled area between q05 and q95
                fill_curve = pg.FillBetweenItem(
                    pg.PlotDataItem(x_numeric, q05_arr),
                    pg.PlotDataItem(x_numeric, q95_arr),
                    brush=pg.mkBrush(fill_color)
                )
                self.ax.addItem(fill_curve)
                artists.append(fill_curve)

                # Draw border lines with semi-transparent color
                line05 = self.ax.plot(x_numeric, q05_arr, pen=pg.mkPen(border_color, width=1, style=Qt.PenStyle.DashLine))
                line95 = self.ax.plot(x_numeric, q95_arr, pen=pg.mkPen(border_color, width=1, style=Qt.PenStyle.DashLine))
                artists.extend([line05, line95])

                # Fallback: compute adherence metrics if not provided in quantiles (best-effort)
                try:
                    metrics = (quantiles or {}).get("adherence_metrics") or {}
                    if not metrics:
                        try:
                            from forex_diffusion.postproc.adherence import adherence_metrics, atr_sigma_from_df
                            anchor_ts = int(pd.to_datetime(self._last_df["ts_utc"].iloc[-1], unit="ms").value // 1_000_000) if (self._last_df is not None and not self._last_df.empty) else None
                            atr_n = 14
                            try:
                                atr_n = int((self._get_indicator_settings() or {}).get("atr_n", 14))
                            except Exception:
                                pass
                            sigma_vol = float(atr_sigma_from_df(self._last_df.rename(columns={"price":"close"}) if self._last_df is not None else pd.DataFrame(),
                                                                n=atr_n, pre_anchor_only=True, anchor_ts=anchor_ts, robust=True))
                            fut_ts_ms = list(quantiles.get("future_ts") or [])
                            actual_ts, actual_y = [], []
                            if fut_ts_ms:
                                try:
                                    controller = self.app_controller or getattr(self._main_window, "controller", None)
                                    engine = getattr(getattr(controller, "market_service", None), "engine", None) if controller else None
                                    if engine is not None:
                                        from sqlalchemy import text
                                        with engine.connect() as conn:
                                            vals = ",".join(f"({int(t)})" for t in fut_ts_ms)
                                            qsql = text(
                                                f"WITH fut(ts) AS (VALUES {vals}) "
                                                "SELECT c.ts_utc AS ts, c.close AS y "
                                                "FROM fut JOIN market_data_candles c ON c.ts_utc = fut.ts "
                                                "WHERE c.symbol = :symbol AND c.timeframe = :timeframe "
                                                "ORDER BY c.ts_utc ASC"
                                            )
                                            sym = getattr(self, 'symbol', 'EUR/USD')
                                            tf = getattr(self, 'timeframe', '1m')
                                            rows = conn.execute(qsql, {'symbol': sym, 'timeframe': tf}).fetchall()
                                            if rows:
                                                tmp = pd.DataFrame(rows, columns=['ts', 'y'])
                                                actual_ts = tmp['ts'].astype('int64').tolist()
                                                actual_y = tmp['y'].astype(float).tolist()
                                except Exception:
                                    pass
                                if not actual_ts and self._last_df is not None and not self._last_df.empty:
                                    try:
                                        ycol = 'close' if 'close' in self._last_df.columns else 'price'
                                        dfa = self._last_df[['ts_utc', ycol]].dropna().copy()
                                        dfa['ts'] = pd.to_numeric(dfa['ts_utc'], errors='coerce').astype('int64')
                                        dfa['y'] = pd.to_numeric(dfa[ycol], errors='coerce').astype(float)
                                        dfa['ts_dt'] = pd.to_datetime(dfa['ts'], unit='ms', utc=True).dt.tz_convert(None)
                                        df_fut = pd.DataFrame({'ts': pd.to_numeric(fut_ts_ms, errors='coerce').astype('int64')}).sort_values('ts')
                                        df_fut['ts_dt'] = pd.to_datetime(df_fut['ts'], unit='ms', utc=True).dt.tz_convert(None)
                                        # tolerance: ~0.51×TF
                                        def _tf_ms(tf: str) -> int:
                                            try:
                                                s = str(tf).strip().lower()
                                                if s.endswith("m"): return int(s[:-1]) * 60_000
                                                if s.endswith("h"): return int(s[:-1]) * 3_600_000
                                                if s.endswith("d"): return int(s[:-1]) * 86_400_000
                                            except Exception:
                                                pass
                                            return 60_000
                                        tol_ms = max(1_000, int(0.51 * _tf_ms(getattr(self, 'timeframe', '1m') or '1m')))
                                        merged = pd.merge_asof(
                                            df_fut.sort_values('ts_dt'),
                                            dfa.sort_values('ts_dt')[['ts_dt','ts','y']],
                                            left_on='ts_dt',
                                            right_on='ts_dt',
                                            direction='nearest',
                                            tolerance=pd.Timedelta(milliseconds=tol_ms),
                                            suffixes=('', '_real')
                                        ).dropna().reset_index(drop=True)
                                        if not merged.empty:
                                            actual_ts = (df_fut.loc[merged.index, "ts"]).astype("int64").tolist()
                                            actual_y = merged["y"].astype(float).tolist()
                                    except Exception:
                                        pass
                            if fut_ts_ms and actual_ts and actual_y:
                                metrics = adherence_metrics(
                                    fut_ts=fut_ts_ms, m=quantiles.get("q50") or [],
                                    q05=quantiles.get("q05") or [], q95=quantiles.get("q95") or [],
                                    actual_ts=actual_ts, actual_y=actual_y,
                                    sigma_vol=sigma_vol, band_target=0.90
                                )
                                quantiles["adherence_metrics"] = metrics
                        except Exception:
                            metrics = {}
                except Exception:
                    metrics = {}

                # Ensure forecast is visible in current axes (extend limits if needed) - PyQtGraph version
                try:
                    last_x_num = x_numeric[-1] if len(x_numeric) > 0 else 0
                    y_last = float(q95_arr[-1]) if len(q95_arr) > 0 else 0

                    # Get current view range
                    view_range = self.ax.viewRange()
                    if view_range:
                        [[xmin, xmax], [ymin, ymax]] = view_range

                        # Extend X range if needed
                        if last_x_num > xmax:
                            span = (xmax - xmin) if (xmax > xmin) else 1.0
                            margin = 0.02 * span
                            self.ax.setXRange(xmin, last_x_num + margin, padding=0)

                        # Extend Y range if needed
                        pad = 0.02 * (ymax - ymin if ymax > ymin else 1.0)
                        if y_last > ymax:
                            self.ax.setYRange(ymin, y_last + pad, padding=0)
                        elif y_last < ymin:
                            self.ax.setYRange(y_last - pad, ymax, padding=0)
                except Exception as e:
                    logger.debug(f"Failed to extend view range: {e}")

                # Draw adherence badge (text annotation) - PyQtGraph version
                try:
                    adh = (metrics or {}).get("adherence", None)
                    # Robust numeric conversion: handles numpy scalars too; fallback to "0.00"
                    try:
                        val = float(adh)
                        if np.isnan(val):
                            val = None
                    except Exception:
                        val = None
                    txt = f"{val:.2f}" if val is not None else "0.00"

                    # Create text item for badge
                    from PySide6.QtGui import QColor
                    text_item = pg.TextItem(
                        text=txt,
                        color=QColor('white'),
                        anchor=(0, 0.5)
                    )
                    text_item.setPos(x_numeric[-1], float(q95_arr[-1]))
                    self.ax.addItem(text_item)
                    artists.append(text_item)

                    # Register globally for hover-hide
                    try:
                        if not hasattr(self, '_adh_badges'):
                            self._adh_badges = []
                        self._adh_badges.append(text_item)
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"Failed to add adherence badge: {e}")

            # aggiorna legenda unica (no duplicati per modello)
            try:
                self._refresh_legend_unique(loc='upper left')
            except Exception:
                pass

            fid = time.time()
            forecast = {
                "id": fid,
                "created_at": fid,
                "quantiles": quantiles,
                "future_ts": quantiles.get("future_ts"),
                "artists": artists,
                "source": quantiles.get("source", source)
            }
            self._forecasts.append(forecast)
            self._trim_forecasts()

            # Disable auto-range to prevent zoom changes when adding forecast
            self.ax.enableAutoRange(enable=False)

            # PyQtGraph handles legend and rendering automatically
            # Legend items are added via the 'name' parameter in plot() calls
            logger.debug(f"Forecast overlay added: {len(artists)} artists, model={model_name}")
        except Exception as e:
            logger.exception(f"Failed to plot forecast overlay: {e}")

    def _open_forecast_settings(self):
        from forex_diffusion.ui.prediction_settings_dialog import PredictionSettingsDialog
        dialog = PredictionSettingsDialog(self.view)
        # execute and then apply relevant runtime settings (max_forecasts, auto)
        dialog.exec()
        try:
            settings = PredictionSettingsDialog.get_settings_from_file()
            self.max_forecasts = int(settings.get("max_forecasts", self.max_forecasts))
            auto = bool(settings.get("auto_predict", False))
            interval = int(settings.get("auto_interval_seconds", self._auto_timer.interval() // 1000))
            self._auto_timer.setInterval(max(1, interval) * 1000)
            if auto:
                self.start_auto_forecast()
            else:
                self.stop_auto_forecast()
        except Exception:
            pass

    def _on_forecast_clicked(self):
        from forex_diffusion.ui.prediction_settings_dialog import PredictionSettingsDialog
        settings = PredictionSettingsDialog.get_settings_from_file()
        logger.info(f"Forecast clicked. Settings loaded: model_paths={settings.get('model_paths')}, horizons={settings.get('horizons')}")
        if not settings.get("model_paths") and not settings.get("model_path"):
            QMessageBox.warning(self.view, "Missing Model", "Please select a model file.")
            return

        payload = {"symbol": self.symbol, "timeframe": self.timeframe, **settings}
        logger.info(f"Forecast payload: {len(payload.get('model_paths', []))} models, horizons={payload.get('horizons')}")
        # add forecast granularity from UI
        try:
            if hasattr(self, "pred_step_combo") and self.pred_step_combo is not None:
                payload["forecast_step"] = self.pred_step_combo.currentText() or "auto"
        except Exception:
            pass
        self.forecastRequested.emit(payload)

    def _open_adv_forecast_settings(self):
        # Reuse same dialog which now contains advanced options
        from forex_diffusion.ui.prediction_settings_dialog import PredictionSettingsDialog
        dialog = PredictionSettingsDialog(self.view)
        dialog.exec()

    def _on_advanced_forecast_clicked(self):
        # advanced forecast: use same settings but tag source
        from forex_diffusion.ui.prediction_settings_dialog import PredictionSettingsDialog
        settings = PredictionSettingsDialog.get_settings_from_file()
        if not settings.get("model_paths") and not settings.get("model_path"):
            QMessageBox.warning(self.view, "Missing Model", "Please select a model file.")
            return
        payload = {"symbol": self.symbol, "timeframe": self.timeframe, "advanced": True, **settings}
        # add forecast granularity from UI
        try:
            if hasattr(self, "pred_step_combo") and self.pred_step_combo is not None:
                payload["forecast_step"] = self.pred_step_combo.currentText() or "auto"
        except Exception:
            pass
        self.forecastRequested.emit(payload)

    def on_forecast_ready(self, df: pd.DataFrame, quantiles: dict):
        """
        Slot to receive forecast results from controller/worker.
        Adds the forecast overlay without removing existing ones (trimming oldest if needed).
        """
        try:
            # Plot forecast overlay without redrawing the chart (which would clear previous forecasts)
            # The chart is already displayed, we just add the overlay on top
            source = quantiles.get("source", "basic") if isinstance(quantiles, dict) else "basic"
            self._plot_forecast_overlay(quantiles, source=source)
        except Exception as e:
            logger.exception(f"Error handling forecast result: {e}")

    def clear_all_forecasts(self):
        """Remove all forecast artists from axes and clear internal list."""
        try:
            for f in self._forecasts:
                for art in f.get("artists", []):
                    try:
                        art.remove()
                    except Exception:
                        pass
            self._forecasts = []

            # Clear draggable legend
            if hasattr(self, '_forecast_legend') and self._forecast_legend:
                self._forecast_legend.clear_all()

            # reset legenda/registro modelli
            try:
                self._legend_once = set()
                leg = self.ax.get_legend()
                if leg:
                    try:
                        leg.remove()
                    except Exception:
                        pass
            except Exception:
                pass
            # PyQtGraph updates automatically, only call draw for matplotlib
            try:
                if hasattr(self.canvas, 'draw'):
                    self.canvas.draw()
            except Exception:
                pass
        except Exception as e:
            logger.exception(f"Failed to clear forecasts: {e}")

    def _trim_forecasts(self):
        """Disabled: keep all forecast overlays visible."""
        try:
            return  # no trimming
        except Exception:
            return

    def start_auto_forecast(self):
        if not self._auto_timer.isActive():
            self._auto_timer.start()
            logger.info("Auto-forecast started")

    def stop_auto_forecast(self):
        if self._auto_timer.isActive():
            self._auto_timer.stop()
            logger.info("Auto-forecast stopped")

    def _auto_forecast_tick(self):
        """Called by timer: trigger both basic and advanced forecasts (emit signals)."""
        try:
            # Use saved settings to build payloads; emit two requests: basic and advanced
            from forex_diffusion.ui.prediction_settings_dialog import PredictionSettingsDialog
            settings = PredictionSettingsDialog.get_settings_from_file() or {}
            # Basic
            payload_basic = {"symbol": self.symbol, "timeframe": self.timeframe, **settings}
            self.forecastRequested.emit(payload_basic)
            # Advanced
            payload_adv = {"symbol": self.symbol, "timeframe": self.timeframe, "advanced": True, **settings}
            self.forecastRequested.emit(payload_adv)
        except Exception as e:
            logger.exception(f"Auto forecast tick failed: {e}")
