"""
Event Handlers Mixin for ChartTab - handles all event-related methods.
"""
from __future__ import annotations

import time
from typing import Optional, Dict
import pandas as pd
# import matplotlib.dates as mdates  # matplotlib removed
from PySide6.QtWidgets import QDialog, QScrollArea, QVBoxLayout, QMessageBox
from PySide6.QtCore import Qt, QSignalBlocker
from loguru import logger

from ...utils.user_settings import get_setting, set_setting


class EventHandlersMixin:
    """Mixin containing all event handling methods for ChartTab."""

    def _setup_timers(self):
        """Setup all timers for the chart tab."""
        from PySide6.QtCore import QTimer

        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(int(get_setting("auto_interval_seconds", 60) * 1000))
        self._auto_timer.timeout.connect(self.chart_controller.auto_forecast_tick)

        self._orders_timer = QTimer(self)
        self._orders_timer.setInterval(1500)
        self._orders_timer.timeout.connect(self.chart_controller.refresh_orders)
        self._orders_timer.start()

        self._rt_dirty = False
        self._rt_timer = QTimer(self)
        self._rt_timer.setInterval(200)
        self._rt_timer.timeout.connect(self.chart_controller.rt_flush)
        self._rt_timer.start()

        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(250)
        self._reload_timer.timeout.connect(self.chart_controller.reload_view_window)

    def _init_control_defaults(self) -> None:
        """Initialize default values for all controls."""
        # Pattern toggles default states
        pattern_defaults = {
            "chart_patterns": get_setting("patterns.chart_enabled", False),
            "candle_patterns": get_setting("patterns.candle_enabled", False),
            "history_patterns": get_setting("patterns.history_enabled", False)
        }

        for pattern_type, default_state in pattern_defaults.items():
            if checkbox := getattr(self, f"cb_{pattern_type}", None):
                checkbox.setChecked(default_state)

        # Follow mode default
        self._follow_enabled = get_setting('chart.follow_enabled', True)
        self._follow_suspend_until = 0.0
        self._follow_suspend_seconds = float(get_setting('chart.follow_suspend_seconds', 30))

        if follow_cb := getattr(self, 'follow_checkbox', None):
            follow_cb.setChecked(self._follow_enabled)

        # Theme and other defaults
        if theme_combo := getattr(self, "theme_combo", None):
            theme_combo.setCurrentText(get_setting("chart.theme", "dark"))

        # Price mode
        self._price_mode = get_setting('chart.price_mode', 'candles')
        if mode_btn := getattr(self, 'mode_btn', None):
            mode_btn.setChecked(self._price_mode == 'candles')
            mode_btn.setText('Candles' if self._price_mode == 'candles' else 'Line')

    def _connect_ui_signals(self) -> None:
        """Connect all UI signals to their handlers."""
        signal_connections = {
            getattr(self, "symbol_combo", None): ("currentTextChanged", self._on_symbol_combo_changed),
            getattr(self, "tf_combo", None): ("currentTextChanged", self._on_timeframe_changed),
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
        }

        for widget, (signal_name, handler) in signal_connections.items():
            if widget:
                try:
                    signal = getattr(widget, signal_name)
                    signal.connect(handler)
                except (AttributeError, TypeError) as e:
                    logger.debug(f"Failed to connect {signal_name} signal: {e}")

        # Wire pattern checkboxes
        self._wire_pattern_checkboxes()

    def _connect_mouse_events(self):
        """Connect mouse events for chart interaction."""
        if hasattr(self, 'canvas'):
            self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
            self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
            self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)
            self.canvas.mpl_connect('pick_event', self._on_pick_pattern_artist)

    # Symbol and timeframe change handlers
    def _on_symbol_combo_changed(self, new_symbol: str) -> None:
        if not new_symbol:
            return
        # Reset pattern cache on symbol change
        try:
            self._clear_pattern_artists()
            self._patterns_cache = []
            self._patterns_cache_map = {}
        except Exception:
            pass
        set_setting('chart.symbol', new_symbol)
        self.symbol = new_symbol
        self.chart_controller.on_symbol_changed(new_symbol=new_symbol)

    def _on_timeframe_changed(self, value: str) -> None:
        set_setting('chart.timeframe', value)
        self.timeframe = value
        self._schedule_view_reload()

    def _on_pred_step_changed(self, value: str) -> None:
        set_setting('chart.pred_step', value)

    def _on_backfill_range_changed(self, _value: str) -> None:
        if (ys := getattr(self, 'years_combo', None)):
            set_setting('chart.backfill_years', ys.currentText())
        if (ms := getattr(self, 'months_combo', None)):
            set_setting('chart.backfill_months', ms.currentText())

    # Theme and UI changes
    def _on_theme_changed(self, theme: str) -> None:
        set_setting('chart.theme', theme)
        self._apply_theme(theme)
        # Re-apply grid styling after theme change
        try:
            self._apply_grid_style()
        except Exception as e:
            logger.warning(f"Failed to apply grid style after theme change: {e}")

    def _open_settings_dialog(self) -> None:
        from ..settings_dialog import SettingsDialog
        # Wrap settings dialog in a scrollable container to provide vertical scrollbar
        dialog = SettingsDialog(self)
        accepted = False
        try:
            wrapper = QDialog(self)
            wrapper.setWindowTitle(getattr(dialog, "windowTitle", lambda: "Settings")())
            area = QScrollArea(wrapper)
            area.setWidgetResizable(True)
            # Make inner dialog act as a widget
            dialog.setWindowFlags(Qt.WindowType.Widget)
            area.setWidget(dialog)
            lay = QVBoxLayout(wrapper)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(area)
            accepted = bool(wrapper.exec())
        except Exception as e:
            logger.debug(f"Scrollable settings wrapper failed, fallback: {e}")
            accepted = bool(dialog.exec())

        if accepted:
            if (tc := getattr(self, 'theme_combo', None)):
                self._apply_theme(tc.currentText())
            # Apply grid style again to reflect unified grid color setting
            try:
                self._apply_grid_style()
            except Exception as e:
                logger.warning(f"Failed to apply grid style after settings: {e}")
            self._follow_suspend_seconds = float(get_setting('chart.follow_suspend_seconds', self._follow_suspend_seconds))
            self._follow_enabled = bool(get_setting('chart.follow_enabled', self._follow_enabled))
            if self._follow_enabled:
                self._follow_suspend_until = 0.0
            if (fc := getattr(self, 'follow_checkbox', None)):
                with QSignalBlocker(fc):
                    fc.setChecked(self._follow_enabled)
            if getattr(self, '_last_df', None) is not None and not self._last_df.empty:
                prev_xlim, prev_ylim = self.ax.get_xlim(), self.ax.get_ylim()
                self.update_plot(self._last_df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)
            self._follow_center_if_needed()

    # Chart mode and follow behavior
    def _on_price_mode_toggled(self, checked: bool) -> None:
        self._price_mode = 'candles' if checked else 'line'
        if (mb := getattr(self, 'mode_btn', None)):
            mb.setText('Candles' if checked else 'Line')
        set_setting('chart.price_mode', self._price_mode)
        self.chart_controller.on_mode_toggled(checked=checked)

    def _on_follow_toggled(self, checked: bool) -> None:
        self._follow_enabled = bool(checked)
        set_setting('chart.follow_enabled', self._follow_enabled)
        if self._follow_enabled:
            self._follow_suspend_until = 0.0
            self._follow_center_if_needed()

    def _suspend_follow(self) -> None:
        if getattr(self, '_follow_enabled', False):
            duration = float(get_setting('chart.follow_suspend_seconds', getattr(self, '_follow_suspend_seconds', 30)))
            self._follow_suspend_until = time.time() + max(duration, 1.0)

    def _follow_center_if_needed(self) -> None:
        if not (getattr(self, '_follow_enabled', False) and
                time.time() >= getattr(self, '_follow_suspend_until', 0.0) and
                (ldf := getattr(self, '_last_df', None)) is not None and
                not ldf.empty and
                (ax := getattr(self, 'ax', None)) is not None):
            return

        y_col = 'close' if 'close' in ldf.columns else 'price'
        last_row = ldf.iloc[-1]
        last_ts, last_price = float(last_row['ts_utc']), float(last_row.get(y_col, last_row.get('price')))

        # Convert to naive datetime (local) safely, then to mdates number
        try:
            ts_obj = pd.to_datetime(last_ts, unit='ms', utc=True).tz_convert('UTC').tz_localize(None)
            last_dt = mdates.date2num(ts_obj)
        except Exception:
            last_dt = mdates.date2num(pd.to_datetime(last_ts, unit='ms'))

        # map to compressed X if compression is active
        try:
            comp = getattr(self.chart_controller.plot_service, "_compress_real_x", None)
            last_dt_comp = comp(last_dt) if callable(comp) else last_dt
        except Exception:
            last_dt_comp = last_dt

        # Center the view on the last data point
        x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
        ax.set_xlim(last_dt_comp - x_span * 0.8, last_dt_comp + x_span * 0.2)

        # Optional: also center Y around the current price
        y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.set_ylim(last_price - y_span * 0.5, last_price + y_span * 0.5)

        try:
            self.canvas.draw_idle()
        except Exception as e:
            logger.debug(f"Follow center draw failed: {e}")

    # Mouse event handlers
    def _on_mouse_press(self, event):
        if event and getattr(event, "button", None) in (1, 3):
            self._suspend_follow()
        return self.chart_controller.on_mouse_press(event=event)

    def _on_mouse_move(self, event):
        if event and (getattr(event, "button", None) in (1, 3) or
                     (ge := getattr(event, "guiEvent", None)) and
                     getattr(ge, "buttons", lambda: 0)()):
            self._suspend_follow()

        # Update overlays with cursor info unless dragging an overlay
        # This requires more context from the overlay manager
        # Suppress line update during overlay drag
        if getattr(self, "_overlay_dragging", False) or getattr(self, "_suppress_line_update", False):
            return None
        return self.chart_controller.on_mouse_move(event=event)

    def _on_mouse_release(self, event):
        if event and getattr(event, "button", None) in (1, 3):
            self._suspend_follow()
        return self.chart_controller.on_mouse_release(event=event)

    def _on_scroll_zoom(self, event):
        if event:
            self._suspend_follow()
        return self.chart_controller.on_scroll_zoom(event=event)

    # Pattern event handlers - these need to be implemented based on the patterns mixin
    def _on_pick_pattern_artist(self, event):
        """Handle pattern artist pick events."""
        # This would be implemented by the patterns mixin
        pass

    # Utility methods used by event handlers
    def _toggle_drawbar(self, visible: bool):
        """Toggle drawbar visibility."""
        return self.chart_controller.toggle_drawbar(visible=visible)

    def _wire_pattern_checkboxes(self) -> None:
        """Wire pattern checkbox signals."""
        # This would be implemented by the patterns mixin
        pass

    def _clear_pattern_artists(self):
        """Clear pattern artists from chart."""
        # This would be implemented by the patterns mixin
        pass

    def _apply_theme(self, theme: str):
        """Apply theme to chart."""
        return self.chart_controller.apply_theme(theme=theme)

    def _apply_grid_style(self):
        """Apply grid styling."""
        # This would be implemented by the UI builder or plot service
        pass

    def _schedule_view_reload(self):
        """Schedule a view reload."""
        return self.chart_controller.schedule_view_reload()