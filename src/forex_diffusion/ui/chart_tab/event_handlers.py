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
        """Connect mouse events for chart interaction using PyQtGraph."""
        try:
            # Connect PyQtGraph scene mouse events
            if hasattr(self, 'main_plot') and self.main_plot:
                scene = self.main_plot.scene()
                if scene:
                    # Connect mouse press, move, release events
                    scene.sigMouseClicked.connect(self._on_pyqtgraph_mouse_click)
                    scene.sigMouseMoved.connect(self._on_pyqtgraph_mouse_move)
                    logger.debug("PyQtGraph mouse events connected")
        except Exception as e:
            logger.warning(f"Failed to connect PyQtGraph mouse events: {e}")

    def _on_pyqtgraph_mouse_click(self, event):
        """Handle PyQtGraph mouse click - convert to DrawingManager format."""
        try:
            from PySide6.QtCore import Qt

            # Check if we have a drawing tool active
            if not hasattr(self, 'drawing_manager') or not self.drawing_manager:
                return

            if not self.drawing_manager.current_tool:
                return

            # Get the mouse button
            button = event.button()
            if button != Qt.MouseButton.LeftButton:
                return

            # Get scene position
            scene_pos = event.scenePos()

            # Map to view (data) coordinates
            if hasattr(self, 'main_plot') and self.main_plot:
                viewbox = self.main_plot.getViewBox()
                view_pos = viewbox.mapSceneToView(scene_pos)

                # Extract x (timestamp) and y (price) in data coordinates
                x_data = view_pos.x()
                y_data = view_pos.y()

                # Check if this is the first click (start) or subsequent click
                if self.drawing_manager.current_drawing is None:
                    # Start new drawing
                    self.drawing_manager.start_drawing(x_data, y_data)
                    logger.debug(f"Started drawing at ({x_data}, {y_data})")

                    # For single-click tools (icon), finish immediately
                    if self.drawing_manager.current_tool in ['icon']:
                        self.drawing_manager.finish_drawing()
                        logger.debug("Single-click tool completed")
                else:
                    # Add point and check if we should finish
                    self.drawing_manager.add_point(x_data, y_data)
                    logger.debug(f"Added point at ({x_data}, {y_data})")

                    # For two-point tools, finish after second click
                    if self.drawing_manager.current_tool in ['line', 'arrow', 'rectangle', 'triangle', 'fib', 'gaussian']:
                        if len(self.drawing_manager.current_drawing.points) >= 2:
                            self.drawing_manager.finish_drawing()
                            logger.debug("Two-point tool completed")

                    # Freehand continues until explicit finish (double-click or Esc)

        except Exception as e:
            logger.warning(f"Failed to handle mouse click for drawing: {e}", exc_info=True)

    def _on_pyqtgraph_mouse_move(self, pos):
        """Handle PyQtGraph mouse move - update drawing preview."""
        try:
            # Check if we're currently drawing
            if not hasattr(self, 'drawing_manager') or not self.drawing_manager:
                return

            if not self.drawing_manager.current_drawing:
                return

            # Get scene position from tuple
            if isinstance(pos, tuple):
                scene_pos = pos[0]  # PyQtGraph sigMouseMoved sends (QPointF,)
            else:
                scene_pos = pos

            # Map to view (data) coordinates
            if hasattr(self, 'main_plot') and self.main_plot:
                viewbox = self.main_plot.getViewBox()
                view_pos = viewbox.mapSceneToView(scene_pos)

                # Extract coordinates
                x_data = view_pos.x()
                y_data = view_pos.y()

                # For freehand tool, continuously add points while dragging
                if self.drawing_manager.current_tool == 'freehand':
                    self.drawing_manager.add_point(x_data, y_data)

                # TODO: Add preview line/shape rendering here
                # This would show a rubber-band effect while drawing
                logger.debug(f"Mouse moved to ({x_data}, {y_data}) during drawing")

        except Exception as e:
            logger.debug(f"Failed to handle mouse move for drawing: {e}")

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
            # Update icon and tooltip based on mode
            if checked:
                # Currently in Candles mode, show line icon to switch to Line
                mb.setText('📈')  # Line chart icon
                mb.setToolTip('Usa Line Chart')
            else:
                # Currently in Line mode, show candlestick icon to switch to Candles
                mb.setText('📊')  # Candlestick icon
                mb.setToolTip('Usa Candlesticks')
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
        # Handle drawing tools first
        if hasattr(self, 'drawing_manager') and self.drawing_manager and self.drawing_manager.current_tool:
            if event and event.inaxes and getattr(event, "button", None) == 1:  # Left click
                try:
                    x, y = event.xdata, event.ydata
                    if x is not None and y is not None:
                        self.drawing_manager.start_drawing(x, y)
                        logger.debug(f"Drawing started at ({x:.2f}, {y:.2f})")
                        return  # Don't pass to controller when drawing
                except Exception as e:
                    logger.error(f"Error starting drawing: {e}")

        if event and getattr(event, "button", None) in (1, 3):
            self._suspend_follow()
        return self.chart_controller.on_mouse_press(event=event)

    def _on_mouse_move(self, event):
        # Handle drawing tools first (for freehand, add points during drag)
        if hasattr(self, 'drawing_manager') and self.drawing_manager:
            if self.drawing_manager.current_tool and self.drawing_manager.current_drawing:
                if event and event.inaxes and self.drawing_manager.current_tool == 'freehand':
                    try:
                        x, y = event.xdata, event.ydata
                        if x is not None and y is not None:
                            self.drawing_manager.add_point(x, y)
                            return  # Don't pass to controller when drawing
                    except Exception as e:
                        logger.error(f"Error adding drawing point: {e}")

        # Update hover legend with mouse position
        self._update_hover_info(event)

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
        # Handle drawing tools - finish drawing on release
        if hasattr(self, 'drawing_manager') and self.drawing_manager:
            if self.drawing_manager.current_tool and self.drawing_manager.current_drawing:
                if event and event.inaxes and getattr(event, "button", None) == 1:  # Left click
                    try:
                        x, y = event.xdata, event.ydata
                        if x is not None and y is not None:
                            # Add final point for non-freehand drawings
                            if self.drawing_manager.current_tool != 'freehand':
                                self.drawing_manager.add_point(x, y)
                            # Finish the drawing
                            self.drawing_manager.finish_drawing()
                            logger.debug(f"Drawing finished at ({x:.2f}, {y:.2f})")
                            return  # Don't pass to controller when drawing
                    except Exception as e:
                        logger.error(f"Error finishing drawing: {e}")

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
        """
        Wire pattern checkbox signals to enable/disable pattern detection.

        TASK 5: Pattern Checkboxes Integration
        """
        try:
            # Connect chart patterns checkbox
            if hasattr(self, 'chart_patterns_checkbox'):
                self.chart_patterns_checkbox.toggled.connect(
                    lambda enabled: self._toggle_chart_patterns(enabled)
                )
                logger.debug("Chart patterns checkbox connected")

            # Connect candle patterns checkbox
            if hasattr(self, 'candle_patterns_checkbox'):
                self.candle_patterns_checkbox.toggled.connect(
                    lambda enabled: self._toggle_candle_patterns(enabled)
                )
                logger.debug("Candle patterns checkbox connected")

            # Historical patterns checkbox removed - replaced by scan_historical_btn

            # Connect pattern action buttons
            if hasattr(self, 'scan_historical_btn'):
                self.scan_historical_btn.clicked.connect(self._scan_historical)
                logger.debug("Scan historical button connected")

            if hasattr(self, 'setting_patterns_btn'):
                self.setting_patterns_btn.clicked.connect(self._open_patterns_config)
                logger.debug("Setting patterns button connected")

        except Exception as e:
            logger.exception(f"Failed to wire pattern checkboxes: {e}")

    def _toggle_chart_patterns(self, enabled: bool):
        """Toggle chart pattern detection (head/shoulders, triangles, etc.)."""
        try:
            logger.info(f"Chart patterns {'enabled' if enabled else 'disabled'}")

            # Try to access patterns service through controller
            if hasattr(self, 'chart_controller') and hasattr(self.chart_controller, 'patterns_service'):
                patterns_service = self.chart_controller.patterns_service
                if hasattr(patterns_service, 'set_chart_enabled'):
                    patterns_service.set_chart_enabled(enabled)
                elif hasattr(patterns_service, 'chart_patterns_enabled'):
                    patterns_service.chart_patterns_enabled = enabled

            # Also update chart display if patterns are already detected
            if hasattr(self, '_redraw_patterns'):
                self._redraw_patterns()

        except Exception as e:
            logger.exception(f"Failed to toggle chart patterns: {e}")

    def _toggle_candle_patterns(self, enabled: bool):
        """Toggle candlestick pattern detection (doji, hammer, engulfing, etc.)."""
        try:
            logger.info(f"Candle patterns {'enabled' if enabled else 'disabled'}")

            if hasattr(self, 'chart_controller') and hasattr(self.chart_controller, 'patterns_service'):
                patterns_service = self.chart_controller.patterns_service
                if hasattr(patterns_service, 'set_candle_enabled'):
                    patterns_service.set_candle_enabled(enabled)
                elif hasattr(patterns_service, 'candle_patterns_enabled'):
                    patterns_service.candle_patterns_enabled = enabled

            if hasattr(self, '_redraw_patterns'):
                self._redraw_patterns()

        except Exception as e:
            logger.exception(f"Failed to toggle candle patterns: {e}")

    def _toggle_history_patterns(self, enabled: bool):
        """Toggle historical pattern display."""
        try:
            logger.info(f"Historical patterns {'enabled' if enabled else 'disabled'}")

            if hasattr(self, 'chart_controller') and hasattr(self.chart_controller, 'patterns_service'):
                patterns_service = self.chart_controller.patterns_service
                if hasattr(patterns_service, 'set_history_enabled'):
                    patterns_service.set_history_enabled(enabled)
                elif hasattr(patterns_service, 'history_patterns_enabled'):
                    patterns_service.history_patterns_enabled = enabled

            if hasattr(self, '_redraw_patterns'):
                self._redraw_patterns()

        except Exception as e:
            logger.exception(f"Failed to toggle historical patterns: {e}")

    def _clear_pattern_artists(self):
        """Clear pattern artists from chart."""
        # This would be implemented by the patterns mixin
        pass

    def _scan_historical(self):
        """Scan historical data for patterns - opens dialog to select time range."""
        logger.info("Historical pattern scan requested")

        try:
            # Get or create patterns service (create=True ensures it always exists)
            from ..chart_components.services.patterns_hook import get_patterns_service
            chart_controller = getattr(self, 'chart_controller', None)

            if not chart_controller:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Historical Scan",
                    "Chart controller not available."
                )
                return

            # Always create patterns service if it doesn't exist (ignore checkbox state)
            patterns_service = get_patterns_service(chart_controller, self, create=True)

            if not patterns_service or not hasattr(patterns_service, 'start_historical_scan_with_range'):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Historical Scan",
                    "Pattern service initialization failed."
                )
                return

            # Load default values from patterns.yaml
            historical_config = patterns_service._load_historical_config()
            default_start = historical_config.get('start_time', '30d')
            default_end = historical_config.get('end_time', '7d')

            # Open dialog to get time range
            from ..patterns_config_dialog import HistoricalScanDialog
            dialog = HistoricalScanDialog(self, default_start=default_start, default_end=default_end)

            if dialog.exec() == QDialog.DialogCode.Accepted:
                start_time, end_time = dialog.get_time_range()

                if not start_time or not end_time:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Invalid Range", "Please specify both start and end time.")
                    return

                # Update historical config temporarily for this scan
                patterns_service._historical_config['enabled'] = True  # Ensure enabled for this scan
                patterns_service._historical_config['start_time'] = start_time
                patterns_service._historical_config['end_time'] = end_time

                # Start historical scan with selected range
                patterns_service.start_historical_scan_with_range()
                logger.info(f"Historical pattern scan started: {start_time} to {end_time}")

        except Exception as e:
            logger.exception(f"Historical scan failed: {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Error",
                f"Historical scan failed: {str(e)}"
            )

    def _open_patterns_config(self):
        """Open pattern configuration dialog."""
        try:
            from ..patterns_config_dialog import PatternsConfigDialog

            chart_controller = getattr(self, 'chart_controller', None)
            patterns_service = getattr(chart_controller, 'patterns_service', None) if chart_controller else None

            dialog = PatternsConfigDialog(
                parent=self,
                yaml_path="configs/patterns.yaml",
                patterns_service=patterns_service
            )

            if dialog.exec():
                # Refresh patterns if config was changed
                logger.info("Pattern configuration updated")
                if hasattr(self, '_refresh_patterns'):
                    self._refresh_patterns()

        except Exception as e:
            logger.exception(f"Failed to open patterns config: {e}")

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
            from ...utils.user_settings import get_setting

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
            from ...utils.user_settings import set_setting

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

    def _update_hover_info(self, event):
        """Update hover legend with current mouse position and price info."""
        if not event or not event.inaxes:
            return

        try:
            if not hasattr(self, '_hover_legend') or self._hover_legend is None:
                return

            if not hasattr(self, '_hover_legend_text') or self._hover_legend_text is None:
                return

            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return

            # Format hover info
            info_text = f"Price: {y:.5f}"

            # Update legend text
            self._hover_legend_text.set_text(info_text)

            # Redraw canvas
            if hasattr(self, 'canvas'):
                self.canvas.draw_idle()

        except Exception as e:
            logger.debug(f"Failed to update hover info: {e}")