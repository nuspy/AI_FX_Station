"""
Overlay Manager Mixin for ChartTab - handles overlay and drawing functionality.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from loguru import logger


class OverlayManagerMixin:
    """Mixin containing all overlay and drawing functionality for ChartTab."""

    def _init_overlays(self):
        """Initialize overlay system."""
        try:
            # Create overlay containers
            self._overlays = []
            self._overlay_dragging = False
            self._suppress_line_update = False

            # Create cursor line and value display
            self._init_cursor_overlays()

            # Create legend overlay
            self._init_legend_overlay()

        except Exception as e:
            logger.debug(f"Failed to initialize overlays: {e}")

    def _init_cursor_overlays(self):
        """Initialize cursor line and value display overlays."""
        try:
            if not hasattr(self, 'chart_container'):
                return

            # Cursor crosshair lines (will be drawn on matplotlib)
            self._cursor_vline = None
            self._cursor_hline = None

            # Value display overlay
            self._cursor_overlay = QLabel(self.chart_container)
            self._cursor_overlay.setStyleSheet("""
                QLabel {
                    background: rgba(0, 0, 0, 180);
                    color: white;
                    border: 1px solid rgba(255, 255, 255, 100);
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-family: monospace;
                    font-size: 11px;
                }
            """)
            self._cursor_overlay.hide()
            self._cursor_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        except Exception as e:
            logger.debug(f"Failed to initialize cursor overlays: {e}")

    def _init_legend_overlay(self):
        """Initialize legend overlay."""
        try:
            if not hasattr(self, 'chart_container'):
                return

            # Create draggable legend overlay
            from .chart_tab_base import DraggableOverlay
            self._legend_overlay = DraggableOverlay("Legend", self.chart_container)
            self._legend_overlay.hide()

            # Connect drag signals
            self._legend_overlay.dragStarted.connect(self._on_legend_drag_start)
            self._legend_overlay.dragEnded.connect(self._on_legend_drag_end)

        except Exception as e:
            logger.debug(f"Failed to initialize legend overlay: {e}")

    def _on_legend_drag_start(self):
        """Handle start of legend dragging."""
        self._overlay_dragging = True
        self._suppress_line_update = True

    def _on_legend_drag_end(self):
        """Handle end of legend dragging."""
        self._overlay_dragging = False
        self._suppress_line_update = False

    def _update_cursor_overlay(self, event):
        """Update cursor overlay with current values."""
        try:
            if not hasattr(self, '_cursor_overlay') or not event:
                return

            if not event.inaxes or event.inaxes != self.ax:
                self._hide_cursor_overlay()
                return

            # Get data coordinates
            x_data, y_data = event.xdata, event.ydata
            if x_data is None or y_data is None:
                return

            # Convert x coordinate to timestamp
            try:
                # If compression is active, decompress X
                comp_service = getattr(self.chart_controller.plot_service, "_compress_real_x", None)
                if comp_service:
                    # Need decompression method - this is a placeholder
                    x_real = x_data  # Would need actual decompression
                else:
                    x_real = x_data

                dt = mdates.num2date(x_real)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = f"X: {x_data:.2f}"

            # Format price
            price_str = f"Price: {y_data:.5f}"

            # Get nearest data point if available
            nearest_info = self._get_nearest_data_point(x_data)
            if nearest_info:
                time_str = nearest_info.get('time', time_str)
                price_str = f"OHLC: {nearest_info.get('ohlc', 'N/A')}"

            # Update overlay text
            overlay_text = f"{time_str}\n{price_str}"
            self._cursor_overlay.setText(overlay_text)

            # Position overlay near cursor
            chart_pos = self.chart_container.mapFromGlobal(event.guiEvent.globalPos())
            overlay_x = min(chart_pos.x() + 10, self.chart_container.width() - self._cursor_overlay.width() - 10)
            overlay_y = max(10, chart_pos.y() - self._cursor_overlay.height() - 10)

            self._cursor_overlay.move(overlay_x, overlay_y)
            self._cursor_overlay.show()

        except Exception as e:
            logger.debug(f"Failed to update cursor overlay: {e}")

    def _hide_cursor_overlay(self):
        """Hide cursor overlay."""
        try:
            if hasattr(self, '_cursor_overlay'):
                self._cursor_overlay.hide()
        except Exception:
            pass

    def _get_nearest_data_point(self, x_coord: float) -> Optional[Dict[str, Any]]:
        """Get nearest data point to cursor position."""
        try:
            if not hasattr(self, '_x_cache_comp') or self._x_cache_comp is None:
                return None

            if not hasattr(self, '_last_df') or self._last_df.empty:
                return None

            # Find nearest point in compressed coordinates
            x_cache = np.array(self._x_cache_comp)
            distances = np.abs(x_cache - x_coord)
            nearest_idx = np.argmin(distances)

            if nearest_idx >= len(self._last_df):
                return None

            row = self._last_df.iloc[nearest_idx]

            # Format time
            try:
                ts = pd.to_datetime(row['ts_utc'], unit='ms', utc=True)
                time_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = str(row.get('ts_utc', 'N/A'))

            # Format OHLC if available
            ohlc_str = "N/A"
            if all(col in row for col in ['open', 'high', 'low', 'close']):
                o, h, l, c = row['open'], row['high'], row['low'], row['close']
                ohlc_str = f"O:{o:.5f} H:{h:.5f} L:{l:.5f} C:{c:.5f}"

            return {
                'time': time_str,
                'ohlc': ohlc_str,
                'row': row
            }

        except Exception as e:
            logger.debug(f"Failed to get nearest data point: {e}")
            return None

    def _rebuild_x_cache(self, df: pd.DataFrame):
        """Rebuild X coordinate cache for efficient overlay updates."""
        try:
            if df.empty or 'ts_utc' not in df.columns:
                self._x_cache_comp = None
                return

            # Convert timestamps to matplotlib dates
            x_ser = pd.to_datetime(df['ts_utc'], unit='ms', utc=True)
            # Convert to naive datetimes safely using Series.dt
            x_dt = x_ser.dt.tz_convert('UTC').dt.tz_localize(None)
            x_mpl = mdates.date2num(x_dt)

            # Apply compression if active
            comp = getattr(getattr(self.chart_controller, "plot_service", None), "_compress_real_x", None)
            if callable(comp):
                self._x_cache_comp = np.array([comp(float(v)) for v in x_mpl], dtype=float)
            else:
                self._x_cache_comp = np.array(x_mpl, dtype=float)

        except Exception as e:
            logger.debug(f"Failed to rebuild X cache: {e}")
            self._x_cache_comp = None

    def _update_legend_overlay(self, indicators: List[str]):
        """Update legend overlay with current indicators."""
        try:
            if not hasattr(self, '_legend_overlay'):
                return

            if not indicators:
                self._legend_overlay.hide()
                return

            # Create legend text
            legend_text = "Indicators:\n" + "\n".join(indicators)
            self._legend_overlay.setText(legend_text)

            # Position in top-right corner
            container_width = self.chart_container.width()
            self._legend_overlay.move(
                container_width - self._legend_overlay.width() - 10,
                10
            )
            self._legend_overlay.show()

        except Exception as e:
            logger.debug(f"Failed to update legend overlay: {e}")

    def _apply_grid_style(self):
        """Apply grid styling to the chart."""
        try:
            if not hasattr(self, 'ax'):
                return

            # Get theme-based grid settings
            theme = getattr(self, 'theme_combo', None)
            theme_name = theme.currentText() if theme else 'dark'

            grid_colors = {
                'dark': '#333333',
                'light': '#CCCCCC',
                'blue': '#2C3E50'
            }

            grid_color = grid_colors.get(theme_name, '#333333')

            # Apply grid
            self.ax.grid(True, alpha=0.3, color=grid_color, linewidth=0.5)
            self.ax.set_axisbelow(True)

            # Apply to oscillator axis if present
            if hasattr(self, '_osc_ax') and self._osc_ax:
                self._osc_ax.grid(True, alpha=0.3, color=grid_color, linewidth=0.5)
                self._osc_ax.set_axisbelow(True)

        except Exception as e:
            logger.debug(f"Failed to apply grid style: {e}")

    def _draw_cursor_lines(self, x_coord: float, y_coord: float):
        """Draw cursor crosshair lines."""
        try:
            if not hasattr(self, 'ax'):
                return

            # Remove existing lines
            self._clear_cursor_lines()

            # Draw new lines
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self._cursor_vline = self.ax.axvline(
                x_coord, color='white', alpha=0.5, linewidth=0.8, linestyle='--'
            )
            self._cursor_hline = self.ax.axhline(
                y_coord, color='white', alpha=0.5, linewidth=0.8, linestyle='--'
            )

        except Exception as e:
            logger.debug(f"Failed to draw cursor lines: {e}")

    def _clear_cursor_lines(self):
        """Clear cursor crosshair lines."""
        try:
            if hasattr(self, '_cursor_vline') and self._cursor_vline:
                self._cursor_vline.remove()
                self._cursor_vline = None

            if hasattr(self, '_cursor_hline') and self._cursor_hline:
                self._cursor_hline.remove()
                self._cursor_hline = None

        except Exception:
            pass

    def _ts_to_chart_x(self, ts_ms: int) -> float:
        """Convert timestamp to chart X coordinate."""
        try:
            import pandas as _pd
            import matplotlib.dates as _md
            dt = _pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert("UTC").tz_localize(None)
            x = _md.date2num(dt)
            comp = getattr(getattr(self.chart_controller, "plot_service", None), "_compress_real_x", None)
            return float(comp(x)) if callable(comp) else float(x)
        except Exception:
            return float(ts_ms)

    def _clear_all_overlays(self):
        """Clear all overlays from the chart."""
        try:
            # Clear cursor lines
            self._clear_cursor_lines()

            # Hide overlays
            if hasattr(self, '_cursor_overlay'):
                self._cursor_overlay.hide()

            if hasattr(self, '_legend_overlay'):
                self._legend_overlay.hide()

            # Clear any drawing overlays
            self._clear_drawing_overlays()

        except Exception as e:
            logger.debug(f"Failed to clear overlays: {e}")

    def _clear_drawing_overlays(self):
        """Clear drawing overlays (lines, shapes, etc.)."""
        try:
            # This would clear any user-drawn shapes, lines, etc.
            # Implementation would depend on the drawing system
            pass

        except Exception as e:
            logger.debug(f"Failed to clear drawing overlays: {e}")

    def _resize_overlays(self):
        """Resize overlays when chart container is resized."""
        try:
            if not hasattr(self, 'chart_container'):
                return

            # Update legend position if visible
            if (hasattr(self, '_legend_overlay') and
                self._legend_overlay.isVisible()):
                container_width = self.chart_container.width()
                self._legend_overlay.move(
                    container_width - self._legend_overlay.width() - 10,
                    10
                )

        except Exception as e:
            logger.debug(f"Failed to resize overlays: {e}")

    def resizeEvent(self, event):
        """Handle resize events to update overlays."""
        try:
            super().resizeEvent(event)
            self._resize_overlays()
        except Exception as e:
            logger.debug(f"Resize event handling failed: {e}")