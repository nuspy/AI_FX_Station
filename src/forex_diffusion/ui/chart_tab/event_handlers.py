"""
Event Handlers Mixin for ChartTab - handles all event-related methods.
"""
from __future__ import annotations

import time
from typing import Optional, Dict
import numpy as np
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
            # Forecast settings buttons removed - now in Generative Forecast tab
            getattr(self, "forecast_btn", None): ("clicked", self.chart_controller.on_forecast_clicked),
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

        # Connect positions table signals (TASK 4)
        if hasattr(self, 'positions_table'):
            try:
                self.positions_table.position_selected.connect(self._on_position_selected)
                self.positions_table.close_position_requested.connect(self._on_close_position_requested)
                self.positions_table.modify_sl_requested.connect(self._on_modify_sl_requested)
                self.positions_table.modify_tp_requested.connect(self._on_modify_tp_requested)
                logger.info("Positions table signals connected")
            except Exception as e:
                logger.warning(f"Failed to connect positions table signals: {e}")

    def _connect_mouse_events(self):
        """Connect mouse events for chart interaction."""
        # Mouse events are handled directly by finplot/PyQtGraph
        # mpl_connect is matplotlib-specific and not available in finplot
        # Events will be connected through finplot's native event system
        pass

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
        logger.info(f"Timeframe changed to: {value}")
        set_setting('chart.timeframe', value)
        self.timeframe = value
        # Clear saved view range so chart resets to default zoom for new timeframe
        if hasattr(self.chart_controller, 'plot_service'):
            self.chart_controller.plot_service._saved_view_range = False  # False = don't save on next update
            logger.debug("Cleared saved view range for timeframe change")
        # Reload data from DB with new timeframe
        self.chart_controller.on_timeframe_changed(value)
        logger.debug("Reloaded data for timeframe change")

    def _on_pred_step_changed(self, value: str) -> None:
        set_setting('chart.pred_step', value)

    def _on_backfill_range_changed(self, _value: str) -> None:
        logger.info(f"Range changed to: {_value}")
        if (ys := getattr(self, 'years_combo', None)):
            set_setting('chart.backfill_years', ys.currentText())
        if (ms := getattr(self, 'months_combo', None)):
            set_setting('chart.backfill_months', ms.currentText())
            logger.debug(f"Set months to: {ms.currentText()}")
        # Clear saved view range so chart resets to default zoom for new range
        if hasattr(self.chart_controller, 'plot_service'):
            self.chart_controller.plot_service._saved_view_range = False  # False = don't save on next update
            logger.debug("Cleared saved view range for range change")
        # Reload data from DB with new range
        self.chart_controller.on_timeframe_changed(getattr(self, 'timeframe', '1m'))
        logger.debug("Reloaded data for range change")

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

        # PyQtGraph uses Unix timestamps in seconds
        try:
            last_dt = last_ts / 1000.0  # Convert milliseconds to seconds
        except Exception:
            last_dt = pd.to_datetime(last_ts, unit='ms').timestamp()

        # map to compressed X if compression is active
        try:
            comp = getattr(self.chart_controller.plot_service, "_compress_real_x", None)
            last_dt_comp = comp(last_dt) if callable(comp) else last_dt
        except Exception:
            last_dt_comp = last_dt

        # Center the view on the last data point using PyQtGraph API
        try:
            x_range = ax.viewRange()[0]  # Get current X range
            x_span = x_range[1] - x_range[0]

            # Validate x_span is valid
            if not (x_span > 0 and np.isfinite(x_span) and np.isfinite(last_dt_comp)):
                logger.debug(f"Invalid x_span ({x_span}) or last_dt_comp ({last_dt_comp}), skipping follow")
                return

            ax.setXRange(last_dt_comp - x_span * 0.8, last_dt_comp + x_span * 0.2, padding=0)

            # Optional: also center Y around the current price
            y_range = ax.viewRange()[1]  # Get current Y range
            y_span = y_range[1] - y_range[0]

            # Validate y_span is valid
            if y_span > 0 and np.isfinite(y_span) and np.isfinite(last_price):
                ax.setYRange(last_price - y_span * 0.5, last_price + y_span * 0.5, padding=0)
        except Exception as e:
            logger.debug(f"Follow center view update failed: {e}")

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

    # Position table event handlers (TASK 4)
    def _on_position_selected(self, position: dict):
        """
        Highlight position entry price on chart when selected from positions table.

        TASK 4: Position Handlers
        Centers the chart view on the position's entry price.
        """
        try:
            entry_price = position.get('entry_price')
            if entry_price is None or entry_price == 0:
                logger.warning("Position has no entry_price")
                return

            logger.info(f"Position selected: {position.get('symbol')} @ {entry_price}")

            # Center view on entry price
            if hasattr(self, 'ax') and self.ax:
                # Get current view range
                if hasattr(self.ax, 'viewRange'):
                    view_range = self.ax.viewRange()
                    y_range = view_range[1][1] - view_range[1][0]

                    # Center on entry price
                    new_ylim = (entry_price - y_range/2, entry_price + y_range/2)

                    # Apply new view
                    self.ax.setYRange(new_ylim[0], new_ylim[1], padding=0)

                    logger.debug(f"Centered view on entry price: {entry_price}")

        except Exception as e:
            logger.exception(f"Failed to handle position selection: {e}")

    def _on_close_position_requested(self, position_id: str):
        """
        Close position via trading engine.

        TASK 4: Position Handlers
        """
        try:
            logger.info(f"Close position requested: {position_id}")

            # Try to get trading engine from main window or controller
            trading_engine = None

            # Method 1: Check if trading_engine is available as attribute
            if hasattr(self, 'trading_engine'):
                trading_engine = self.trading_engine
            # Method 2: Try to get from parent window
            elif hasattr(self, 'parent') and self.parent():
                parent = self.parent()
                if hasattr(parent, 'trading_engine'):
                    trading_engine = parent.trading_engine
            # Method 3: Check controller
            elif hasattr(self, 'chart_controller') and hasattr(self.chart_controller, 'trading_engine'):
                trading_engine = self.chart_controller.trading_engine

            if trading_engine is None:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Trading Engine Not Available",
                    "Trading engine is not connected. Cannot close position."
                )
                logger.warning("Trading engine not available for closing position")
                return

            # Close the position
            success = trading_engine.close_position(position_id)

            if success:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Position Closed",
                    f"Position {position_id} closed successfully."
                )
                logger.info(f"Position {position_id} closed successfully")
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Close Failed",
                    f"Failed to close position {position_id}."
                )
                logger.warning(f"Failed to close position {position_id}")

        except Exception as e:
            logger.exception(f"Error closing position {position_id}: {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Error",
                f"Error closing position: {str(e)}"
            )

    def _on_modify_sl_requested(self, position_id: str, new_sl: float):
        """
        Modify stop loss for a position.

        TASK 4: Position Handlers
        """
        try:
            logger.info(f"Modify SL requested: {position_id} new_sl={new_sl}")

            # Try to get trading engine
            trading_engine = None
            if hasattr(self, 'trading_engine'):
                trading_engine = self.trading_engine
            elif hasattr(self, 'parent') and self.parent():
                parent = self.parent()
                if hasattr(parent, 'trading_engine'):
                    trading_engine = parent.trading_engine
            elif hasattr(self, 'chart_controller') and hasattr(self.chart_controller, 'trading_engine'):
                trading_engine = self.chart_controller.trading_engine

            if trading_engine is None:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Trading Engine Not Available",
                    "Trading engine is not connected. Cannot modify stop loss."
                )
                return

            # Modify stop loss
            success = trading_engine.modify_stop_loss(position_id, new_sl)

            if success:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Stop Loss Modified",
                    f"Stop loss for position {position_id} updated to {new_sl:.5f}"
                )
                logger.info(f"Stop loss modified: {position_id} -> {new_sl}")
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Modification Failed",
                    f"Failed to modify stop loss for position {position_id}."
                )

        except Exception as e:
            logger.exception(f"Error modifying stop loss for {position_id}: {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Error",
                f"Error modifying stop loss: {str(e)}"
            )

    def _on_modify_tp_requested(self, position_id: str, new_tp: float):
        """
        Modify take profit for a position.

        TASK 4: Position Handlers
        """
        try:
            logger.info(f"Modify TP requested: {position_id} new_tp={new_tp}")

            # Try to get trading engine
            trading_engine = None
            if hasattr(self, 'trading_engine'):
                trading_engine = self.trading_engine
            elif hasattr(self, 'parent') and self.parent():
                parent = self.parent()
                if hasattr(parent, 'trading_engine'):
                    trading_engine = parent.trading_engine
            elif hasattr(self, 'chart_controller') and hasattr(self.chart_controller, 'trading_engine'):
                trading_engine = self.chart_controller.trading_engine

            if trading_engine is None:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Trading Engine Not Available",
                    "Trading engine is not connected. Cannot modify take profit."
                )
                return

            # Modify take profit
            success = trading_engine.modify_take_profit(position_id, new_tp)

            if success:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Take Profit Modified",
                    f"Take profit for position {position_id} updated to {new_tp:.5f}"
                )
                logger.info(f"Take profit modified: {position_id} -> {new_tp}")
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Modification Failed",
                    f"Failed to modify take profit for position {position_id}."
                )

        except Exception as e:
            logger.exception(f"Error modifying take profit for {position_id}: {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Error",
                f"Error modifying take profit: {str(e)}"
            )

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

    def _restore_splitters(self):
        """
        Restore splitter positions from saved settings.

        TASK 6: Splitter Persistence
        """
        try:
            from ..utils.user_settings import get_setting

            # Restore main splitter (horizontal: market watch | chart+orders)
            if hasattr(self, 'main_splitter'):
                sizes = get_setting('chart.main_splitter_sizes', None)
                if sizes and isinstance(sizes, list) and len(sizes) == 2:
                    self.main_splitter.setSizes(sizes)
                    logger.debug(f"Restored main splitter sizes: {sizes}")

            # Restore right splitter (vertical: chart | orders)
            if hasattr(self, 'right_splitter'):
                sizes = get_setting('chart.right_splitter_sizes', None)
                if sizes and isinstance(sizes, list) and len(sizes) >= 2:
                    self.right_splitter.setSizes(sizes)
                    logger.debug(f"Restored right splitter sizes: {sizes}")

            # Restore chart area splitter (vertical: drawbar | chart)
            if hasattr(self, '_chart_area') and hasattr(self._chart_area, 'sizes'):
                sizes = get_setting('chart.chart_area_splitter_sizes', None)
                if sizes and isinstance(sizes, list) and len(sizes) >= 2:
                    self._chart_area.setSizes(sizes)
                    logger.debug(f"Restored chart area splitter sizes: {sizes}")

        except Exception as e:
            logger.exception(f"Failed to restore splitter positions: {e}")

    def _persist_splitter_positions(self):
        """
        Save splitter positions to settings.

        TASK 6: Splitter Persistence
        """
        try:
            from ..utils.user_settings import set_setting

            # Save main splitter
            if hasattr(self, 'main_splitter'):
                sizes = self.main_splitter.sizes()
                set_setting('chart.main_splitter_sizes', sizes)
                logger.debug(f"Saved main splitter sizes: {sizes}")

            # Save right splitter
            if hasattr(self, 'right_splitter'):
                sizes = self.right_splitter.sizes()
                set_setting('chart.right_splitter_sizes', sizes)
                logger.debug(f"Saved right splitter sizes: {sizes}")

            # Save chart area splitter
            if hasattr(self, '_chart_area') and hasattr(self._chart_area, 'sizes'):
                sizes = self._chart_area.sizes()
                set_setting('chart.chart_area_splitter_sizes', sizes)
                logger.debug(f"Saved chart area splitter sizes: {sizes}")

        except Exception as e:
            logger.exception(f"Failed to persist splitter positions: {e}")

    def _connect_splitter_signals(self):
        """
        Connect splitter moved signals to persistence handler.

        TASK 6: Splitter Persistence
        """
        try:
            # Connect all splitters to the persistence method
            if hasattr(self, 'main_splitter'):
                self.main_splitter.splitterMoved.connect(
                    lambda: self._persist_splitter_positions()
                )

            if hasattr(self, 'right_splitter'):
                self.right_splitter.splitterMoved.connect(
                    lambda: self._persist_splitter_positions()
                )

            if hasattr(self, '_chart_area') and hasattr(self._chart_area, 'splitterMoved'):
                self._chart_area.splitterMoved.connect(
                    lambda: self._persist_splitter_positions()
                )

            logger.info("Splitter persistence signals connected")

        except Exception as e:
            logger.exception(f"Failed to connect splitter signals: {e}")