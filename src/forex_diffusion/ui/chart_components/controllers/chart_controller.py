from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from forex_diffusion.ui.chart_tab import ChartTab

from ..services.plot_service import PlotService
from ..services.forecast_service import ForecastService
from ..services.interaction_service import InteractionService
from ..services.data_service import DataService
from ..services.action_service import ActionService


class ChartTabController:
    """Coordinator between ChartTab view and its domain services."""
    """Coordinator between ChartTab view and its domain services."""

    def __init__(self, view: "ChartTab", app_controller: Optional[Any] = None) -> None:
        self.view = view
        self.app_controller = app_controller
        self.plot_service = PlotService(view, self)
        self.forecast_service = ForecastService(view, self)
        self.interaction_service = InteractionService(view, self)
        self.data_service = DataService(view, self)
        self.action_service = ActionService(view, self)

    def update_plot(self, *args, **kwargs):
        return self.plot_service.update_plot(*args, **kwargs)

    def render_candles(self, *args, **kwargs):
        return self.plot_service._render_candles(*args, **kwargs)

    def ensure_osc_axis(self, *args, **kwargs):
        return self.plot_service._ensure_osc_axis(*args, **kwargs)

    def on_main_xlim_changed(self, *args, **kwargs):
        return self.plot_service._on_main_xlim_changed(*args, **kwargs)

    def plot_indicators(self, *args, **kwargs):
        return self.plot_service._plot_indicators(*args, **kwargs)

    def sma(self, *args, **kwargs):
        return self.plot_service._sma(*args, **kwargs)

    def ema(self, *args, **kwargs):
        return self.plot_service._ema(*args, **kwargs)

    def bollinger(self, *args, **kwargs):
        return self.plot_service._bollinger(*args, **kwargs)

    def donchian(self, *args, **kwargs):
        return self.plot_service._donchian(*args, **kwargs)

    def atr(self, *args, **kwargs):
        return self.plot_service._atr(*args, **kwargs)

    def keltner(self, *args, **kwargs):
        return self.plot_service._keltner(*args, **kwargs)

    def rsi(self, *args, **kwargs):
        return self.plot_service._rsi(*args, **kwargs)

    def macd(self, *args, **kwargs):
        return self.plot_service._macd(*args, **kwargs)

    def hurst_roll(self, *args, **kwargs):
        return self.plot_service._hurst_roll(*args, **kwargs)

    def on_mode_toggled(self, *args, **kwargs):
        return self.plot_service._on_mode_toggled(*args, **kwargs)

    def apply_theme(self, *args, **kwargs):
        return self.plot_service._apply_theme(*args, **kwargs)

    def get_color(self, *args, **kwargs):
        return self.plot_service._get_color(*args, **kwargs)

    def open_color_settings(self, *args, **kwargs):
        return self.plot_service._open_color_settings(*args, **kwargs)

    def toggle_drawbar(self, *args, **kwargs):
        return self.plot_service._toggle_drawbar(*args, **kwargs)

    def toggle_orders(self, *args, **kwargs):
        return self.plot_service._toggle_orders(*args, **kwargs)

    def get_indicator_settings(self, *args, **kwargs):
        return self.plot_service._get_indicator_settings(*args, **kwargs)

    def plot_forecast_overlay(self, *args, **kwargs):
        return self.forecast_service._plot_forecast_overlay(*args, **kwargs)

    def open_forecast_settings(self, *args, **kwargs):
        return self.forecast_service._open_forecast_settings(*args, **kwargs)

    def on_forecast_clicked(self, *args, **kwargs):
        return self.forecast_service._on_forecast_clicked(*args, **kwargs)

    def open_adv_forecast_settings(self, *args, **kwargs):
        return self.forecast_service._open_adv_forecast_settings(*args, **kwargs)

    def on_advanced_forecast_clicked(self, *args, **kwargs):
        return self.forecast_service._on_advanced_forecast_clicked(*args, **kwargs)

    def on_forecast_ready(self, *args, **kwargs):
        return self.forecast_service.on_forecast_ready(*args, **kwargs)

    def clear_all_forecasts(self, *args, **kwargs):
        return self.forecast_service.clear_all_forecasts(*args, **kwargs)

    def trim_forecasts(self, *args, **kwargs):
        return self.forecast_service._trim_forecasts(*args, **kwargs)

    def start_auto_forecast(self, *args, **kwargs):
        return self.forecast_service.start_auto_forecast(*args, **kwargs)

    def stop_auto_forecast(self, *args, **kwargs):
        return self.forecast_service.stop_auto_forecast(*args, **kwargs)

    def auto_forecast_tick(self, *args, **kwargs):
        return self.forecast_service._auto_forecast_tick(*args, **kwargs)

    def set_drawing_mode(self, *args, **kwargs):
        return self.interaction_service._set_drawing_mode(*args, **kwargs)

    def on_canvas_click(self, *args, **kwargs):
        return self.interaction_service._on_canvas_click(*args, **kwargs)

    def on_scroll_zoom(self, *args, **kwargs):
        return self.interaction_service._on_scroll_zoom(*args, **kwargs)

    def on_mouse_press(self, *args, **kwargs):
        return self.interaction_service._on_mouse_press(*args, **kwargs)

    def on_mouse_move(self, *args, **kwargs):
        return self.interaction_service._on_mouse_move(*args, **kwargs)

    def on_mouse_release(self, *args, **kwargs):
        return self.interaction_service._on_mouse_release(*args, **kwargs)

    def zoom_axis(self, *args, **kwargs):
        return self.interaction_service._zoom_axis(*args, **kwargs)

    def update_badge_visibility(self, *args, **kwargs):
        return self.interaction_service._update_badge_visibility(*args, **kwargs)

    def on_nav_home(self, *args, **kwargs):
        return self.interaction_service._on_nav_home(*args, **kwargs)

    def on_nav_pan(self, *args, **kwargs):
        return self.interaction_service._on_nav_pan(*args, **kwargs)

    def on_nav_zoom(self, *args, **kwargs):
        return self.interaction_service._on_nav_zoom(*args, **kwargs)

    def handle_tick(self, *args, **kwargs):
        return self.data_service._handle_tick(*args, **kwargs)

    def on_tick_main(self, *args, **kwargs):
        return self.data_service._on_tick_main(*args, **kwargs)

    def rt_flush(self, *args, **kwargs):
        return self.data_service._rt_flush(*args, **kwargs)

    def schedule_view_reload(self, *args, **kwargs):
        return self.data_service._schedule_view_reload(*args, **kwargs)

    def resolution_for_span(self, *args, **kwargs):
        return self.data_service._resolution_for_span(*args, **kwargs)

    def reload_view_window(self, *args, **kwargs):
        return self.data_service._reload_view_window(*args, **kwargs)

    def tf_to_timedelta(self, *args, **kwargs):
        return self.data_service._tf_to_timedelta(*args, **kwargs)

    def set_symbol_timeframe(self, *args, **kwargs):
        return self.data_service.set_symbol_timeframe(*args, **kwargs)

    def on_symbol_changed(self, *args, **kwargs):
        return self.data_service._on_symbol_changed(*args, **kwargs)

    def on_backfill_missing_clicked(self, *args, **kwargs):
        return self.data_service._on_backfill_missing_clicked(*args, **kwargs)

    def load_candles_from_db(self, *args, **kwargs):
        return self.data_service._load_candles_from_db(*args, **kwargs)

    def refresh_orders(self, *args, **kwargs):
        return self.data_service._refresh_orders(*args, **kwargs)

    def open_trade_dialog(self, *args, **kwargs):
        return self.action_service._open_trade_dialog(*args, **kwargs)

    def on_indicators_clicked(self, *args, **kwargs):
        return self.action_service._on_indicators_clicked(*args, **kwargs)

    def on_build_latents_clicked(self, *args, **kwargs):
        return self.action_service._on_build_latents_clicked(*args, **kwargs)

    def open_pattern_info(self, event_obj) -> None:
        """Mostra un dialog ricco con i dettagli del pattern, includendo immagine se disponibile."""
        from PySide6 import QtWidgets, QtCore, QtGui
        key = str(getattr(event_obj, "key", getattr(event_obj, "name", "Pattern")))
        # recupera metadati (se hai un registry usa quello)
        desc = getattr(event_obj, "description", "—")
        direction = getattr(event_obj, "direction", "neutral")
        tgt = getattr(event_obj, "target_price", None)
        tgt_txt = f"{tgt:.5f}" if isinstance(tgt, (float, int)) else "—"
        bench = getattr(event_obj, "benchmark", {})
        # immagine opzionale in assets (metti file in resources)
        img_path = getattr(event_obj, "image_path", None)

        dlg = QtWidgets.QDialog(self.view)
        dlg.setWindowTitle(key)
        lay = QtWidgets.QVBoxLayout(dlg)

        title = QtWidgets.QLabel(f"<h3>{key}</h3><i>{direction}</i>")
        title.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(title)

        if img_path:
            pm = QtGui.QPixmap(img_path)
            if not pm.isNull():
                lbl_img = QtWidgets.QLabel()
                lbl_img.setPixmap(pm.scaledToWidth(320, QtCore.Qt.TransformationMode.SmoothTransformation))
                lay.addWidget(lbl_img)

        info = QtWidgets.QLabel(f"<p>{desc}</p><p><b>Target:</b> {tgt_txt}</p>")
        info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        info.setWordWrap(True)
        lay.addWidget(info)

        if isinstance(bench, dict) and bench:
            tbl = QtWidgets.QTableWidget(dlg)
            tbl.setColumnCount(2)
            tbl.setHorizontalHeaderLabels(["Benchmark", "Valore"])
            tbl.horizontalHeader().setStretchLastSection(True)
            tbl.setRowCount(len(bench))
            for r, (k, v) in enumerate(bench.items()):
                tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(str(k)))
                tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(str(v)))
            lay.addWidget(tbl)

        btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btn.accepted.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.resize(480, 520)
        dlg.exec()


