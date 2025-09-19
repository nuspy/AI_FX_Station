from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal
from PySide6 import QtUiTools
from PySide6.QtCore import QFile


class ChartTab(QWidget):
    forecastRequested = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._load_ui()
        self._wire()
        # Public state used elsewhere
        self.symbol = "EUR/USD"
        self.timeframe = "1m"
        self._last_df = None
        # optional controllers (attached by app after construction)
        self.controllers = {}
        # overlay registry per sorgente
        self._series = {}
        self._trades = {}
        self._trades_visible = False
        self._initial_view_applied = False
        self._range_restored = False
        self._auto_follow = True
        self._tz_mode = 'local'
        self._load_prefs()

    def _load_ui(self) -> None:
        loader = QtUiTools.QUiLoader()
        ui_file = QFile(self._ui_path())
        ui_file.open(QFile.ReadOnly)
        try:
            self._ui = loader.load(ui_file, self)
        finally:
            ui_file.close()
        # reparent contents into this QWidget
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._ui)

        # resolve handles
        self.symbolCombo = self._ui.findChild(type(self._ui), 'symbolCombo')
        self.timeframeCombo = self._ui.findChild(type(self._ui), 'timeframeCombo')
        self.btnForecast = self._ui.findChild(type(self._ui), 'btnForecast')
        self.btnSeries = self._ui.findChild(type(self._ui), 'btnSeries')
        self.colorProfileCombo = self._ui.findChild(type(self._ui), 'colorProfileCombo')
        self.btnPalette = self._ui.findChild(type(self._ui), 'btnPalette')
        self.plotContainer = self._ui.findChild(type(self._ui), 'plotContainer')
        self.logText = self._ui.findChild(type(self._ui), 'logText')
        # extra action buttons (may be absent in older .ui)
        self.btnIndicators = self._ui.findChild(type(self._ui), 'btnIndicators')
        self.btnBase = self._ui.findChild(type(self._ui), 'btnBase')
        self.btnRun = self._ui.findChild(type(self._ui), 'btnRun')
        self.btnAdv = self._ui.findChild(type(self._ui), 'btnAdv')
        self.btnRunAdv = self._ui.findChild(type(self._ui), 'btnRunAdv')
        self.btnTrades = self._ui.findChild(type(self._ui), 'btnTrades')

        # basic defaults
        try:
            self.symbolCombo.addItems(["EUR/USD"])  # TODO: load from DB
        except Exception:
            pass
        try:
            self.timeframeCombo.addItems(["1m","5m","15m","30m","1h","4h","1d"])
        except Exception:
            pass

        # add pyqtgraph widget
        try:
            import pyqtgraph as pg
            pg.setConfigOptions(antialias=True, useOpenGL=False)
            self.plot = pg.PlotWidget()
            self.plot.setBackground('w')
            self.plot.showGrid(x=True, y=True, alpha=0.3)
            lay_plot = QVBoxLayout(self.plotContainer)
            lay_plot.setContentsMargins(0, 0, 0, 0)
            lay_plot.addWidget(self.plot)
            try:
                self.legend = self.plot.addLegend()
            except Exception:
                self.legend = None
            # persist view range on changes
            try:
                vb = self.plot.getViewBox()
                if vb is not None:
                    vb.sigRangeChanged.connect(self._on_range_changed)
            except Exception:
                pass
            # persist view range on changes
            try:
                vb = self.plot.getViewBox()
                if vb is not None:
                    vb.sigRangeChanged.connect(self._on_range_changed)
            except Exception:
                pass
        except Exception:
            self.plot = None

    def _ui_path(self) -> str:
        import pathlib
        return str(pathlib.Path(__file__).with_name('chart_tab.ui'))

    def _wire(self) -> None:
        try:
            self.btnForecast.clicked.connect(self._emit_forecast)
            self.symbolCombo.currentTextChanged.connect(self._on_symbol)
            self.timeframeCombo.currentTextChanged.connect(self._on_timeframe)
            if self.btnSeries is not None:
                self.btnSeries.clicked.connect(self._toggle_series_menu)
            if self.colorProfileCombo is not None:
                self._init_color_profiles()
                self.colorProfileCombo.currentTextChanged.connect(self._on_color_profile)
            if self.btnPalette is not None:
                self.btnPalette.clicked.connect(self._open_palette_editor)
            # wire optional actions
            if self.btnIndicators is not None:
                self.btnIndicators.clicked.connect(self._open_indicators)
            if self.btnBase is not None:
                self.btnBase.clicked.connect(self._open_base_settings)
            if self.btnRun is not None:
                self.btnRun.clicked.connect(self._run_forecast_basic)
            if self.btnAdv is not None:
                self.btnAdv.clicked.connect(self._open_adv_settings)
            if self.btnRunAdv is not None:
                self.btnRunAdv.clicked.connect(self._run_forecast_advanced)
            if self.btnTrades is not None:
                self.btnTrades.clicked.connect(self._toggle_trades)
        except Exception:
            pass

    def _emit_forecast(self):
        payload = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "advanced": False,
        }
        self.forecastRequested.emit(payload)

    def _on_symbol(self, s: str):
        try:
            self.symbol = s
            # persist
            try:
                from ..utils.user_settings import set_setting, get_setting
            except Exception:
                try:
                    from ...utils.user_settings import set_setting, get_setting
                except Exception:
                    set_setting = None; get_setting = None
            if set_setting:
                cur = {}
                try:
                    cur = get_setting("chart_tab.settings", {}) or {}
                except Exception:
                    cur = {}
                cur["symbol"] = self.symbol
                set_setting("chart_tab.settings", cur)
        except Exception:
            pass

    # --- Optional actions (stubs invoking existing dialogs/flows) ---
    def _open_indicators(self):
        try:
            from ..controllers import UIController  # not used directly, but ensure consistency
            # Reuse menu handler if present
            mw = self.parent()
            if hasattr(mw, "controller") and hasattr(mw.controller, "handle_indicators_requested"):
                mw.controller.handle_indicators_requested()
        except Exception:
            pass

    def _open_base_settings(self):
        try:
            mw = self.parent()
            if hasattr(mw, "controller") and hasattr(mw.controller, "handle_prediction_settings_requested"):
                mw.controller.handle_prediction_settings_requested()
        except Exception:
            pass

    def _run_forecast_basic(self):
        try:
            self._emit_forecast()
        except Exception:
            pass

    def _open_adv_settings(self):
        try:
            # placeholder for a future advanced settings dialog
            self._open_base_settings()
        except Exception:
            pass

    def _run_forecast_advanced(self):
        try:
            payload = {"symbol": self.symbol, "timeframe": self.timeframe, "advanced": True}
            self.forecastRequested.emit(payload)
        except Exception:
            pass

    def _toggle_trades(self):
        try:
            # placeholder: will toggle simulated trades overlay once available
            pass
        except Exception:
            pass

    def _init_color_profiles(self):
        try:
            self._color_profiles = {
                "Default": [
                    (31,119,180), (255,127,14), (44,160,44), (214,39,40), (148,103,189),
                    (140,86,75), (227,119,194), (127,127,127), (188,189,34), (23,190,207)
                ],
                "Pastel": [
                    (102,194,165), (252,141,98), (141,160,203), (231,138,195), (166,216,84),
                    (255,217,47), (229,196,148), (179,179,179)
                ],
                "Dark": [
                    (57,106,177), (218,124,48), (62,150,81), (204,37,41), (107,76,154),
                    (146,36,40), (148,139,61), (0,0,0)
                ],
            }
            self.colorProfileCombo.clear()
            self.colorProfileCombo.addItems(list(self._color_profiles.keys()))
            # load persisted selection/palette if any
            self._active_palette = self._color_profiles["Default"]
            try:
                from ..utils.user_settings import get_setting
            except Exception:
                try:
                    from ...utils.user_settings import get_setting
                except Exception:
                    get_setting = None
            if get_setting:
                cfg = get_setting("chart_tab.palette", {}) or {}
                prof = cfg.get("profile")
                custom = cfg.get("custom")
                if isinstance(custom, list) and all(isinstance(t, (list, tuple)) and len(t)==3 for t in custom):
                    self._color_profiles["Custom"] = [tuple(map(int, t)) for t in custom]
                if prof and prof in self._color_profiles:
                    self._active_palette = self._color_profiles[prof]
                    self.colorProfileCombo.setCurrentText(prof)
        except Exception:
            self._active_palette = None

    def _on_color_profile(self, name: str):
        try:
            pal = self._color_profiles.get(name)
            if pal:
                self._active_palette = pal
                # redraw all series with new palette
                for label, curves in list(self._series.items()):
                    try:
                        x = curves.get("_x"); q50 = curves.get("_q50"); q05 = curves.get("_q05"); q95 = curves.get("_q95")
                        if x is not None:
                            self._update_series(label, x, q50, q05, q95)
                    except Exception:
                        continue
                # persist profile selection
                try:
                    from ..utils.user_settings import set_setting
                except Exception:
                    try:
                        from ...utils.user_settings import set_setting
                    except Exception:
                        set_setting = None
                if set_setting:
                    try:
                        set_setting("chart_tab.palette", {"profile": name, "custom": (self._color_profiles.get("Custom") or [])})
                    except Exception:
                        pass
        except Exception:
            pass

    def _toggle_series_menu(self):
        try:
            from PySide6.QtWidgets import QMenu, QAction
            menu = QMenu(self)
            for label in sorted(self._series.keys()):
                act = QAction(label, menu)
                act.setCheckable(True)
                # visible if any curve exists and is in plot
                visible = any(item is not None for item in self._series.get(label, {}).values())
                act.setChecked(visible)
                def _mk_toggler(lbl):
                    def _toggle(checked: bool):
                        self._set_series_visible(lbl, checked)
                        # persist visibility map
                        try:
                            from ..utils.user_settings import set_setting, get_setting
                        except Exception:
                            try:
                                from ...utils.user_settings import set_setting, get_setting
                            except Exception:
                                set_setting = None; get_setting = None
                        if set_setting:
                            try:
                                vis = get_setting("chart_tab.series_visibility", {}) or {}
                            except Exception:
                                vis = {}
                            vis[str(lbl)] = bool(checked)
                            try:
                                set_setting("chart_tab.series_visibility", vis)
                            except Exception:
                                pass
                    return _toggle
                act.toggled.connect(_mk_toggler(label))
                menu.addAction(act)
            menu.exec(self.btnSeries.mapToGlobal(self.btnSeries.rect().bottomLeft()))
        except Exception:
            pass

    def _open_palette_editor(self):
        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QColorDialog, QListWidget, QListWidgetItem
            dlg = QDialog(self)
            dlg.setWindowTitle("Palette Editor")
            vl = QVBoxLayout(dlg)
            lst = QListWidget(dlg)
            vl.addWidget(lst)
            pal = list(self._active_palette or [])
            for rgb in pal:
                item = QListWidgetItem(f"rgb{rgb}")
                item.setData(32, rgb)  # Qt.UserRole
                lst.addItem(item)
            btn_row = QHBoxLayout()
            btn_add = QPushButton("Aggiungi")
            btn_edit = QPushButton("Modifica")
            btn_del = QPushButton("Rimuovi")
            btn_ok = QPushButton("OK")
            btn_row.addWidget(btn_add); btn_row.addWidget(btn_edit); btn_row.addWidget(btn_del); btn_row.addStretch(1); btn_row.addWidget(btn_ok)
            vl.addLayout(btn_row)

            def _pick_color(initial=None):
                from PySide6.QtGui import QColor
                col = QColorDialog.getColor(QColor(*(initial or (0,0,0))), self, "Scegli colore")
                if col.isValid():
                    return (col.red(), col.green(), col.blue())
                return None

            def _refresh_list():
                lst.clear()
                for rgb in pal:
                    it = QListWidgetItem(f"rgb{rgb}")
                    it.setData(32, rgb)
                    lst.addItem(it)

            def on_add():
                c = _pick_color()
                if c:
                    pal.append(c)
                    _refresh_list()

            def on_edit():
                it = lst.currentItem()
                if not it:
                    return
                rgb = it.data(32)
                c = _pick_color(rgb)
                if c:
                    idx = lst.currentRow()
                    pal[idx] = c
                    _refresh_list()

            def on_del():
                idx = lst.currentRow()
                if idx >= 0:
                    del pal[idx]
                    _refresh_list()

            btn_add.clicked.connect(on_add)
            btn_edit.clicked.connect(on_edit)
            btn_del.clicked.connect(on_del)
            btn_ok.clicked.connect(dlg.accept)

            if dlg.exec():
                # save under a custom name and apply
                name = "Custom"
                try:
                    self._color_profiles[name] = list(pal)
                    self._active_palette = self._color_profiles[name]
                    # update combo selection
                    if self.colorProfileCombo is not None:
                        if self.colorProfileCombo.findText(name) < 0:
                            self.colorProfileCombo.addItem(name)
                        self.colorProfileCombo.setCurrentText(name)
                    # redraw
                    for label, curves in list(self._series.items()):
                        x = curves.get("_x"); q50 = curves.get("_q50"); q05 = curves.get("_q05"); q95 = curves.get("_q95")
                        if x is not None:
                            self._update_series(label, x, q50, q05, q95)
                    # persist custom palette
                    try:
                        from ..utils.user_settings import set_setting
                    except Exception:
                        try:
                            from ...utils.user_settings import set_setting
                        except Exception:
                            set_setting = None
                    if set_setting:
                        try:
                            set_setting("chart_tab.palette", {"profile": name, "custom": list(pal)})
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _set_series_visible(self, label: str, visible: bool):
        try:
            cur = self._series.get(label) or {}
            for k, item in cur.items():
                if k.startswith("_"):
                    continue
                try:
                    if item is not None:
                        item.setVisible(visible)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_timeframe(self, tf: str):
        try:
            self.timeframe = tf
            try:
                from ..utils.user_settings import set_setting, get_setting
            except Exception:
                try:
                    from ...utils.user_settings import set_setting, get_setting
                except Exception:
                    set_setting = None; get_setting = None
            if set_setting:
                cur = {}
                try:
                    cur = get_setting("chart_tab.settings", {}) or {}
                except Exception:
                    cur = {}
                cur["timeframe"] = self.timeframe
                set_setting("chart_tab.settings", cur)
        except Exception:
            pass

    # keep API used by the rest of app
    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        try:
            if self.symbolCombo is not None:
                idx = self.symbolCombo.findText(symbol)
                if idx >= 0:
                    self.symbolCombo.setCurrentIndex(idx)
            if self.timeframeCombo is not None:
                idx = self.timeframeCombo.findText(timeframe)
                if idx >= 0:
                    self.timeframeCombo.setCurrentIndex(idx)
        except Exception:
            pass

    def on_forecast_ready(self, df, quantiles):
        # Multi-sorgente con legenda: aggiorna/aggiunge la serie identificata da label
        try:
            self._last_df = df
            if self.plot is None:
                return
            import numpy as _np
            fut_ts = _np.asarray(quantiles.get("future_ts") or [], dtype=float)
            q50 = _np.asarray(quantiles.get("q50") or [], dtype=float)
            q05 = _np.asarray(quantiles.get("q05") or [], dtype=float)
            q95 = _np.asarray(quantiles.get("q95") or [], dtype=float)
            # always plot baseline price from df if available
            try:
                self._update_price_from_df(df)
            except Exception:
                pass
            # indicators overlay from df (EMA fast/slow as default)
            try:
                self._draw_indicators(df)
            except Exception:
                pass
            # track current source label for badge controller
            try:
                self.current_source_label = str(quantiles.get("label") or quantiles.get("source") or "forecast")
            except Exception:
                self.current_source_label = "forecast"
            # fallback X if timestamps missing
            x_vals = fut_ts
            if (x_vals.size == 0) and q50.size:
                x_vals = _np.arange(q50.size, dtype=float)
            self._update_series(self.current_source_label, x_vals, q50, q05, q95)
            # format X axis as datetime if timestamps
            try:
                self._ensure_time_axis(x_vals)
            except Exception:
                pass
            # draw adherence badge via controller if metrics provided
            try:
                metrics = (quantiles.get("adherence_metrics") or {})
                adh = metrics.get("adherence")
                if adh is not None:
                    ctrl = self.controllers.get('adherence')
                    if ctrl is not None:
                        ctrl.draw_badge(fut_ts, q50, q95, adh)
            except Exception:
                pass
            # initial/persisted view: restore last, else 2 settimane × 100 pips
            try:
                if not getattr(self, "_range_restored", False):
                    if self._restore_last_range():
                        self._range_restored = True
                        self._initial_view_applied = True
                if not getattr(self, "_initial_view_applied", False):
                    center_y = float(q50[-1]) if q50.size else (float(df["close"].iloc[-1]) if hasattr(df, "__getitem__") and "close" in getattr(df, 'columns', []) else None)
                    if center_y is not None:
                        self._apply_initial_view(x_vals, center_y)
                        self._initial_view_applied = True
                # auto-follow on new forecast if enabled
                try:
                    if getattr(self, '_auto_follow', True):
                        self._center_on_last()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

    def _ensure_time_axis(self, x_vals) -> None:
        # Format bottom axis as datetime when x looks like ms timestamps
        try:
            import numpy as _np
            if x_vals is None or getattr(x_vals, 'size', 0) == 0:
                return
            is_ms = bool(float(x_vals[-1]) > 1e6)
            if not is_ms:
                return
            ax = self.plot.getPlotItem().getAxis('bottom')
            if ax is None:
                return
            def _fmt_ticks(values, _scale, _spacing):
                out = []
                try:
                    import datetime as dt
                    local_tz = dt.datetime.now().astimezone().tzinfo
                    for v in values:
                        try:
                            ts_utc = dt.datetime.fromtimestamp(float(v)/1000.0, tz=dt.timezone.utc)
                            ts = ts_utc if getattr(self, '_tz_mode', 'local') == 'utc' else ts_utc.astimezone(local_tz)
                            out.append(ts.strftime('%Y-%m-%d\n%H:%M'))
                        except Exception:
                            out.append("")
                except Exception:
                    out = [str(v) for v in values]
                return out
            try:
                # monkey-patch tickStrings for AxisItem
                ax.tickStrings = _fmt_ticks
                self.plot.getPlotItem().update()
            except Exception:
                pass
        except Exception:
            pass

    def _update_price_from_df(self, df):
        try:
            import numpy as _np
            import pyqtgraph as pg
            if df is None or len(df) == 0:
                return
            cols = getattr(df, 'columns', [])
            if ("ts_utc" not in cols) or ("close" not in cols):
                return
            x = _np.asarray(df["ts_utc"], dtype=float)
            y = _np.asarray(df["close"], dtype=float)
            # drop NaNs
            mask = _np.isfinite(x) & _np.isfinite(y)
            x = x[mask]; y = y[mask]
            if x.size == 0 or y.size == 0:
                return
            # remove old price curve
            prev = self._series.get("price")
            if isinstance(prev, dict):
                for _, it in prev.items():
                    try:
                        if it is not None:
                            self.plot.removeItem(it)
                    except Exception:
                        pass
            pen = pg.mkPen(color=(60,60,60), width=1.5)
            price_item = self.plot.plot(x, y, pen=pen, name="price")
            self._series["price"] = {"line": price_item, "_x": x, "_y": y}
            # auto-follow: center last point if enabled
            try:
                if getattr(self, '_auto_follow', True):
                    self._center_on_last()
            except Exception:
                pass
        except Exception:
            pass

    def _center_on_last(self):
        try:
            vb = self.plot.getViewBox()
            if vb is None:
                return
            px = self._series.get("price", {}).get("_x")
            py = self._series.get("price", {}).get("_y")
            if px is None or py is None or len(px) == 0 or len(py) == 0:
                return
            x_last = float(px[-1]); y_last = float(py[-1])
            (xmin, xmax), (ymin, ymax) = vb.viewRange()
            if xmax <= xmin or ymax <= ymin:
                self._apply_initial_view(px, y_last)
                return
            half_w = 0.5 * (xmax - xmin)
            half_h = 0.5 * (ymax - ymin)
            vb.setRange(xRange=(x_last - half_w, x_last + half_w), yRange=(y_last - half_h, y_last + half_h), padding=0.0)
        except Exception:
            pass

    def _draw_indicators(self, df):
        try:
            import numpy as _np
            import pyqtgraph as pg
            if df is None or len(df) == 0:
                return
            if not ("ts_utc" in df.columns and "close" in df.columns):
                return
            x = _np.asarray(df["ts_utc"], dtype=float)
            y = _np.asarray(df["close"], dtype=float)
            # retrieve default EMA spans from controller settings when available
            ema_fast, ema_slow = 12, 26
            try:
                ctrl = getattr(self.parent(), 'controller', None)
                if ctrl is not None:
                    s = getattr(ctrl, 'indicators_settings', {}) or {}
                    ema_fast = int(s.get('ema_fast', ema_fast))
                    ema_slow = int(s.get('ema_slow', ema_slow))
            except Exception:
                pass
            def _ema(vals, span):
                try:
                    import pandas as _pd
                    return _pd.Series(vals).ewm(span=max(1,int(span)), adjust=False).mean().to_numpy()
                except Exception:
                    alpha = 2.0/(max(1,int(span))+1.0)
                    out = _np.empty_like(vals, dtype=float)
                    acc = 0.0; init=False
                    for i,v in enumerate(vals):
                        if not init:
                            acc = float(v); init=True
                        else:
                            acc = alpha*float(v) + (1-alpha)*acc
                        out[i] = acc
                    return out
            ema_f = _ema(y, ema_fast)
            ema_s = _ema(y, ema_slow)
            # Remove previous indicator curves
            for key in ["ema_fast","ema_slow"]:
                prev = self._series.get(key)
                if isinstance(prev, dict):
                    for _, it in prev.items():
                        try:
                            if it is not None:
                                self.plot.removeItem(it)
                        except Exception:
                            pass
            f_item = self.plot.plot(x, ema_f, pen=pg.mkPen(color=(0,120,255,180), width=1), name=f"EMA{ema_fast}")
            s_item = self.plot.plot(x, ema_s, pen=pg.mkPen(color=(200,80,0,180), width=1), name=f"EMA{ema_slow}")
            self._series["ema_fast"] = {"line": f_item}
            self._series["ema_slow"] = {"line": s_item}
        except Exception:
            pass

    def _apply_initial_view(self, x_vals, center_y: float):
        try:
            import numpy as _np
            vb = self.plot.getViewBox()
            if vb is None:
                return
            end_x = float(x_vals[-1]) if x_vals.size else 0.0
            # infer ms timestamps (very large values) vs index
            is_ms = (end_x > 1e6)
            if is_ms:
                two_weeks_ms = float(14 * 24 * 3600 * 1000)
                start_x = end_x - two_weeks_ms
            else:
                width_pts = float(max(10, min(2000, x_vals.size if x_vals.size else 500)))
                start_x = max(0.0, end_x - width_pts)
            pip = self._pip_size_for_symbol()
            half_span = 50.0 * pip
            y_min = center_y - half_span
            y_max = center_y + half_span
            vb.setRange(xRange=(start_x, end_x), yRange=(y_min, y_max), padding=0.0)
        except Exception:
            pass

    def _pip_size_for_symbol(self) -> float:
        try:
            s = str(getattr(self, 'symbol', '')).upper()
            if not s:
                return 0.0001
            if 'JPY' in s:
                return 0.01
            if 'XAU' in s or 'XAG' in s:
                return 0.1
            if 'BTC' in s or 'ETH' in s or 'CRYPTO' in s:
                return 1.0
            if 'SPX' in s or 'DAX' in s or 'NDX' in s or 'FTSE' in s:
                return 1.0
            return 0.0001
        except Exception:
            return 0.0001

    def _on_range_changed(self, *_args, **_kwargs):
        try:
            vb = self.plot.getViewBox()
            if vb is None:
                return
            (xmin, xmax), (ymin, ymax) = vb.viewRange()
            key = f"{getattr(self,'symbol','')}_{getattr(self,'timeframe','')}"
            data = {"key": key, "x": [float(xmin), float(xmax)], "y": [float(ymin), float(ymax)]}
            try:
                from ..utils.user_settings import set_setting
            except Exception:
                try:
                    from ...utils.user_settings import set_setting
                except Exception:
                    set_setting = None
            if set_setting:
                set_setting("chart_tab.range", data)
        except Exception:
            pass

    def _restore_last_range(self) -> bool:
        try:
            try:
                from ..utils.user_settings import get_setting
            except Exception:
                try:
                    from ...utils.user_settings import get_setting
                except Exception:
                    get_setting = None
            if not get_setting:
                return False
            data = get_setting("chart_tab.range", {}) or {}
            key = f"{getattr(self,'symbol','')}_{getattr(self,'timeframe','')}"
            if not data or str(data.get("key")) != key:
                return False
            xr = data.get("x"); yr = data.get("y")
            if not xr or not yr:
                return False
            vb = self.plot.getViewBox()
            if vb is None:
                return False
            vb.setRange(xRange=(float(xr[0]), float(xr[1])), yRange=(float(yr[0]), float(yr[1])), padding=0.0)
            return True
        except Exception:
            return False

    def _color_for_label(self, label: str) -> tuple:
        # deterministic color per label from a small palette
        palette = (self._active_palette or [
            (31,119,180), (255,127,14), (44,160,44), (214,39,40), (148,103,189),
            (140,86,75), (227,119,194), (127,127,127), (188,189,34), (23,190,207)
        ])
        idx = (abs(hash(label)) % len(palette))
        return palette[idx]

    def _update_series(self, label: str, x, q50, q05, q95):
        try:
            import pyqtgraph as pg
            # remove previous curves for this label
            prev = self._series.get(label)
            if isinstance(prev, dict):
                for k, item in prev.items():
                    try:
                        if item is not None:
                            self.plot.removeItem(item)
                    except Exception:
                        pass
            color = self._color_for_label(label)
            q50_item = None; q05_item = None; q95_item = None
            if x.size and q50.size:
                q50_item = self.plot.plot(x, q50, pen=pg.mkPen(color=color, width=2), name=f"{label} q50")
            if x.size and q05.size:
                q05_item = self.plot.plot(x, q05, pen=pg.mkPen(color=(color[0],color[1],color[2],160), width=1, style=pg.QtCore.Qt.DotLine), name=f"{label} q05")
            if x.size and q95.size:
                q95_item = self.plot.plot(x, q95, pen=pg.mkPen(color=(color[0],color[1],color[2],160), width=1, style=pg.QtCore.Qt.DotLine), name=f"{label} q95")
            # memo data for palette redraw
            self._series[label] = {"q50": q50_item, "q05": q05_item, "q95": q95_item, "_x": x, "_q50": q50, "_q05": q05, "_q95": q95}
            # apply persisted visibility if any
            try:
                from ..utils.user_settings import get_setting
            except Exception:
                try:
                    from ...utils.user_settings import get_setting
                except Exception:
                    get_setting = None
            if get_setting:
                try:
                    vis = get_setting("chart_tab.series_visibility", {}) or {}
                    visible = bool(vis.get(str(label), True))
                    for k, item in self._series[label].items():
                        if k.startswith("_") or item is None:
                            continue
                        item.setVisible(visible)
                except Exception:
                    pass
        except Exception:
            pass

    # compatibility for WS connector: accept tick updates and update price line
    def _handle_tick(self, tick: object):
        try:
            ts = None
            price = None
            try:
                if isinstance(tick, dict):
                    ts = tick.get('ts_utc') or tick.get('ts') or tick.get('time')
                    price = tick.get('price') or tick.get('close') or tick.get('last')
            except Exception:
                pass
            if ts is not None and price is not None:
                self._update_price_point(float(ts), float(price))
                try:
                    if getattr(self, '_auto_follow', True):
                        self._center_on_last()
                except Exception:
                    pass
            if self.logText is not None:
                self.logText.append(str(tick))
        except Exception:
            pass

    def _update_price_point(self, ts_utc: float, price: float):
        try:
            import numpy as _np
            import pyqtgraph as pg
            cur = self._series.get("price") or {}
            x = cur.get('_x'); y = cur.get('_y')
            line = cur.get('line')
            if x is None or y is None or line is None:
                # create fresh series
                if self.plot is None:
                    return
                pen = pg.mkPen(color=(60,60,60), width=1.5)
                line = self.plot.plot([ts_utc], [price], pen=pen, name="price")
                self._series["price"] = {"_x": _np.asarray([ts_utc], dtype=float), "_y": _np.asarray([price], dtype=float), "line": line}
                return
            x = _np.append(x, float(ts_utc))
            y = _np.append(y, float(price))
            # keep last N points to avoid memory bloat
            if x.size > 10000:
                x = x[-10000:]; y = y[-10000:]
            line.setData(x, y)
            self._series["price"].update({"_x": x, "_y": y, "line": line})
        except Exception:
            pass

    def _load_prefs(self) -> None:
        try:
            try:
                from ..utils.user_settings import get_setting
            except Exception:
                try:
                    from ...utils.user_settings import get_setting
                except Exception:
                    get_setting = None
            if not get_setting:
                return
            cfg = get_setting("chart_tab.settings", {}) or {}
            tz = str(cfg.get('tz_mode') or 'local')
            self._tz_mode = 'utc' if tz.lower() == 'utc' else 'local'
            af = cfg.get('auto_follow')
            if isinstance(af, bool):
                self._auto_follow = af
        except Exception:
            pass

    def set_timezone_mode(self, mode: str) -> None:
        try:
            mode = (mode or 'local').lower()
            if mode not in ('local','utc'):
                mode = 'local'
            self._tz_mode = mode
            # persist
            try:
                from ..utils.user_settings import set_setting, get_setting
            except Exception:
                try:
                    from ...utils.user_settings import set_setting, get_setting
                except Exception:
                    set_setting = None; get_setting = None
            if set_setting:
                cur = {}
                try:
                    cur = get_setting("chart_tab.settings", {}) or {}
                except Exception:
                    cur = {}
                cur['tz_mode'] = self._tz_mode
                set_setting("chart_tab.settings", cur)
            # refresh axis
            try:
                # reuse last x or current view for reformat
                px = self._series.get('price', {}).get('_x')
                if px is not None:
                    self._ensure_time_axis(px)
            except Exception:
                pass
        except Exception:
            pass

    def set_auto_follow(self, enabled: bool) -> None:
        try:
            self._auto_follow = bool(enabled)
            try:
                from ..utils.user_settings import set_setting, get_setting
            except Exception:
                try:
                    from ...utils.user_settings import set_setting, get_setting
                except Exception:
                    set_setting = None; get_setting = None
            if set_setting:
                cur = {}
                try:
                    cur = get_setting("chart_tab.settings", {}) or {}
                except Exception:
                    cur = {}
                cur['auto_follow'] = self._auto_follow
                set_setting("chart_tab.settings", cur)
        except Exception:
            pass


