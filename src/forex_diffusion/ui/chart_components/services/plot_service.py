from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Using finplot exclusively - matplotlib removed
import numpy as np
import pandas as pd
from loguru import logger
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication, QMessageBox
# split_range_avoid_weekend removed - no longer needed
from .patterns_hook import call_patterns_detection
from forex_diffusion.utils.user_settings import get_setting, set_setting
from .base import ChartServiceBase

# Import PyQtGraph for high-performance charting
import pyqtgraph as pg
from .pyqtgraph_candlestick import add_candlestick, DateAxisItem

# Enhanced indicators support
try:
    from forex_diffusion.features.indicators_talib import TALibIndicators as BTALibIndicators
    from forex_diffusion.features.indicator_ranges import indicator_range_classifier
    ENHANCED_INDICATORS_AVAILABLE = True
except ImportError:
    ENHANCED_INDICATORS_AVAILABLE = False
    BTALibIndicators = None
    indicator_range_classifier = None


class PlotService(ChartServiceBase):
    """Auto-generated service extracted from ChartTab."""

    def __init__(self, view, controller):
        super().__init__(view, controller)
        self._subplot_service = None
        self._subplot_enabled = False
        self._mouse_label = None
        self._crosshair_v = None
        self._crosshair_h = None
        self._mouse_proxy = None

    def enable_indicator_subplots(self):
        """Enable multi-subplot mode for indicators (PyQtGraph-based)"""
        # Subplots are managed dynamically in _update_plot_finplot
        # This method is kept for compatibility
        self._subplot_enabled = True
        logger.info("Indicator subplots enabled (PyQtGraph)")

    def disable_indicator_subplots(self):
        """Disable multi-subplot mode"""
        self._subplot_enabled = False
        logger.info("Indicator subplots disabled")

    def apply_theme_to_pyqtgraph(self):
        """Apply theme colors to PyQtGraph plots"""
        if not hasattr(self.view, 'use_finplot') or not self.view.use_finplot:
            return

        # Get colors from settings
        chart_bg = get_setting('chart_bg', '#0f1115')
        mini_chart_bg_1 = get_setting('mini_chart_bg_1', '#14181f')
        mini_chart_bg_2 = get_setting('mini_chart_bg_2', '#1a1e25')
        axes_color = get_setting('axes_color', '#cfd6e1')
        grid_color = get_setting('grid_color', '#3a4250')
        text_color = get_setting('text_color', '#e0e0e0')
        title_color = get_setting('title_bar_color', '#cfd6e1')

        try:
            import pyqtgraph as pg

            # Apply to all plots with specific backgrounds
            for i, plot_item in enumerate(self.view.finplot_axes):
                # Determine which background color to use
                if i == 0:
                    # Main price plot
                    bg_color = chart_bg
                elif hasattr(self.view, 'normalized_plot') and plot_item == self.view.normalized_plot:
                    # First subplot (normalized)
                    bg_color = mini_chart_bg_1
                elif hasattr(self.view, 'custom_plot') and plot_item == self.view.custom_plot:
                    # Second subplot (custom)
                    bg_color = mini_chart_bg_2
                else:
                    # Default to chart_bg
                    bg_color = chart_bg

                # Set background color using ViewBox
                plot_item.getViewBox().setBackgroundColor(bg_color)

                # Set axes colors and text
                for axis_name in ['left', 'right', 'top', 'bottom']:
                    axis = plot_item.getAxis(axis_name)
                    axis.setPen(axes_color)
                    axis.setTextPen(text_color)

                # Set grid color (show grid with default alpha)
                plot_item.showGrid(x=True, y=True, alpha=0.3)

                # Update title color if title exists (LabelItem uses setColor, not setDefaultTextColor)
                if hasattr(plot_item, 'titleLabel') and plot_item.titleLabel is not None:
                    try:
                        plot_item.setTitle(plot_item.titleLabel.text, color=title_color)
                    except Exception:
                        pass

            # Apply to graphics layout background (use chart_bg)
            if hasattr(self.view, 'graphics_layout'):
                self.view.graphics_layout.setBackground(chart_bg)

            # PlutoTouch logger.info("Theme applied to PyQtGraph plots")
        except Exception as e:
            logger.error(f"Failed to apply theme to PyQtGraph: {e}")

    def update_plot(self, df: pd.DataFrame, quantiles: Optional[dict] = None, restore_xlim=None, restore_ylim=None):
        # Check if using finplot
        if hasattr(self.view, 'use_finplot') and self.view.use_finplot:
            return self._update_plot_finplot(df, quantiles)

        if df is None or df.empty:
            self._clear_candles()
            if hasattr(self, '_price_line') and self._price_line is not None:
                try:
                    self._price_line.set_visible(False)
                except Exception:
                    pass
            try:
                self._refresh_legend_unique(loc='upper left')
            except Exception:
                pass
            return

        if not hasattr(self, '_extra_legend_items'):
            self._extra_legend_items = {}

        self._last_df = df.copy()

        prev_xlim = restore_xlim if restore_xlim is not None else None
        prev_ylim = restore_ylim if restore_ylim is not None else None

        try:
            df2 = df.copy()
            y_col = 'close' if 'close' in df2.columns else 'price'
            df2['ts_utc'] = pd.to_numeric(df2['ts_utc'], errors='coerce')
            df2[y_col] = pd.to_numeric(df2[y_col], errors='coerce')
            df2 = df2.dropna(subset=['ts_utc', y_col]).reset_index(drop=True)
            df2['ts_utc'] = df2['ts_utc'].astype('int64')
            df2 = df2.sort_values('ts_utc').reset_index(drop=True)

            x_dt = pd.to_datetime(df2['ts_utc'], unit='ms', utc=True)
            try:
                x_dt = x_dt.tz_localize(None)
            except Exception:
                pass
            y_vals = df2[y_col].astype(float).to_numpy()

            # Compress time axis by removing weekend periods
            x_dt_compressed, y_vals_compressed, weekend_markers, compressed_df = self._compress_weekend_periods(x_dt, y_vals, df2)
            x_dt = x_dt_compressed
            y_vals = y_vals_compressed
            # Use compressed dataframe for indicators
            if compressed_df is not None:
                df2 = compressed_df

            # --- patterns: pass robusto del DataFrame corrente ---
            try:
                from .patterns_hook import call_patterns_detection
                df_current = None

                # tenta nomi locali comuni
                for _name in ("df", "df2", "data", "df_plot"):
                    if _name in locals():
                        df_current = locals()[_name]
                        break

                # fallback: ultimo df usato dal plot_service
                if df_current is None:
                    df_current = getattr(self, "_last_df", None)

                # chiama comunque: il service ha altri fallback/controlli
                call_patterns_detection(self.controller, self.view, df_current)
            except Exception as e:
                logger.debug(f"patterns hook error: {e}")


        except Exception as e:
            logger.exception('Failed to normalize data for plotting: {}', e)
            return

        if len(x_dt) == 0 or len(y_vals) == 0:
            logger.info('Nothing to plot after cleaning ({} {}).', getattr(self, 'symbol', ''), getattr(self, 'timeframe', ''))
            self._clear_candles()
            return

        price_mode = str(getattr(self, '_price_mode', 'line')).lower()
        price_color = self._get_color_mpl('price_line_color', '#e0e0e0' if getattr(self, '_is_dark', True) else '#1a1e25')
        try:
            if not hasattr(self, '_price_line') or self._price_line is None:
                (self._price_line,) = self.ax.plot([], [], color=price_color, label='Price')
            self._price_line.set_data(x_dt, y_vals)
            self._price_line.set_color(price_color)
        except Exception:
            self._price_line = None

        if price_mode == 'candles' and {'open', 'high', 'low', 'close'}.issubset(df2.columns):
            if self._price_line is not None:
                try:
                    self._price_line.set_visible(False)
                except Exception:
                    pass
            self._render_candles(df2, x_dt)
        else:
            # break line across large gaps to avoid bridging closed-market periods
            if self._price_line is not None:
                try:
                    # compute expected seconds from timeframe; fallback to median spacing
                    tf = str(getattr(self, 'timeframe', 'auto') or 'auto').lower()
                    try:
                        if tf != "auto":
                            exp_td = self.controller.tf_to_timedelta(tf=tf)
                            exp_sec = float(getattr(exp_td, "total_seconds", lambda: 60.0)())
                        else:
                            diffs_days = np.diff(mdates.date2num(x_dt)) if len(x_dt) > 1 else np.array([])
                            exp_sec = float(np.median(diffs_days) * 86400.0) if len(diffs_days) else 60.0
                    except Exception:
                        exp_sec = 60.0
                    # build y with NaN at gaps
                    y_plot = y_vals.copy()
                    if len(x_dt) > 1:
                        tnum = mdates.date2num(x_dt)
                        dsec = np.diff(tnum) * 86400.0
                        gap_idx = np.where(dsec > exp_sec * 3.0)[0]  # break at i+1
                        for gi in gap_idx:
                            if 0 <= gi + 1 < len(y_plot):
                                y_plot[gi + 1] = np.nan
                    self._price_line.set_visible(True)
                    self._price_line.set_data(x_dt, y_plot)
                except Exception:
                    try:
                        self._price_line.set_visible(True)
                    except Exception:
                        pass
            self._clear_candles()

        # draw vertical cut lines for closed-market spans
        try:
            # cleanup previous markers
            for art in list(getattr(self, "_cut_lines", []) or []):
                try:
                    art.remove()
                except Exception:
                    pass
            self._cut_lines = []
            if len(x_dt) > 1:
                tnum = mdates.date2num(x_dt)
                dsec = np.diff(tnum) * 86400.0
                # expected spacing
                tf = str(getattr(self, 'timeframe', 'auto') or 'auto').lower()
                try:
                    if tf != "auto":
                        exp_td = self.controller.tf_to_timedelta(tf=tf)
                        exp_sec = float(getattr(exp_td, "total_seconds", lambda: 60.0)())
                    else:
                        exp_sec = float(np.median(dsec)) if len(dsec) else 60.0
                except Exception:
                    exp_sec = 60.0
                thr = exp_sec * 3.0
                cut_col = self._get_color_mpl("market_cut_color", "#7f8fa6")
                for i, gap in enumerate(dsec):
                    if gap <= thr:
                        continue
                    # classify gap as closed-market if it includes weekend
                    try:
                        dt_left = pd.to_datetime(df2["ts_utc"].iloc[i], unit="ms", utc=True).tz_convert(None)
                        dt_right = pd.to_datetime(df2["ts_utc"].iloc[i+1], unit="ms", utc=True).tz_convert(None)
                    except Exception:
                        continue
                    try:
                        # build daily range and check weekend presence
                        days = pd.date_range(dt_left.normalize(), dt_right.normalize(), freq="D")
                        has_weekend = any(d.weekday() >= 5 for d in days)
                    except Exception:
                        wd_l = getattr(dt_left, "weekday", lambda: 0)()
                        wd_r = getattr(dt_right, "weekday", lambda: 0)()
                        has_weekend = (wd_l >= 5) or (wd_r >= 5)
                    if not has_weekend:
                        continue  # not a closed-market span -> do not mark as cut
                    try:
                        line = self.ax.axvline(x_dt.iloc[i+1], color=cut_col, linestyle='--', linewidth=0.8, alpha=0.6, zorder=2)
                        self._cut_lines.append(line)
                    except Exception:
                        pass
        except Exception:
            pass

        self._plot_indicators(df2, x_dt)

        # Plot enhanced indicators from EnhancedIndicatorsDialog
        self._plot_enhanced_indicators(df2, x_dt)

        # Draw weekend markers (yellow dashed lines)
        self._draw_weekend_markers()

        if quantiles:
            self._plot_forecast_overlay(quantiles)

        title_color = self._get_color_mpl('title_bar_color', '#cfd6e1' if getattr(self, '_is_dark', True) else '#1a1e25')
        self.ax.set_title(f"{getattr(self, 'symbol', '')} - {getattr(self, 'timeframe', '')}", pad=2, color=title_color)
        axes_color_qt = self._get_qcolor('axes_color', '#cfd6e1')
        axes_col = self._color_to_mpl(axes_color_qt)
        axes_col_hex = axes_color_qt.name(QColor.HexRgb)
        try:
            self.ax.tick_params(colors=axes_col_hex)
            self.ax.xaxis.label.set_color(axes_col_hex)
            self.ax.yaxis.label.set_color(axes_col_hex)
            for spine in self.ax.spines.values():
                spine.set_color(axes_col_hex)
            self.ax.set_xlabel('')
            try:
                self.ax.xaxis.get_offset_text().set_visible(False)
            except Exception:
                pass
        except Exception:
            pass

        try:
            locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax.xaxis.set_major_locator(locator)
            self.ax.xaxis.set_major_formatter(formatter)
            # Smart minor ticks between majors
            try:
                self.ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
            except Exception:
                pass
            # Grid styling from settings (major stronger, minor lighter)
            grid_col = self._get_color_mpl('grid_color', '#3a4250' if getattr(self, '_is_dark', True) else '#d8dee9')
            try:
                self.ax.grid(True, which='major', axis='x', color=grid_col, alpha=0.35, linewidth=0.8)
                self.ax.grid(True, which='minor', axis='x', color=grid_col, alpha=0.15, linewidth=0.5)
            except Exception:
                pass
        except Exception:
            pass

        try:
            self.ax.margins(x=0.001, y=0.05)
            self.canvas.figure.subplots_adjust(left=0.04, right=0.995, top=0.97, bottom=0.08)
        except Exception:
            pass

        try:
            if prev_xlim is not None:
                self.ax.set_xlim(prev_xlim)
            if prev_ylim is not None:
                self.ax.set_ylim(prev_ylim)
            if prev_xlim is None and prev_ylim is None:
                self.ax.relim()
                self.ax.autoscale_view()
        except Exception:
            pass

        try:
            if self._osc_ax is not None:
                self._osc_ax.set_xlim(self.ax.get_xlim())
        except Exception:
            pass

        try:
            self._refresh_legend_unique(loc='upper left')
        except Exception as exc:
            logger.debug('Legend refresh skipped: {}', exc)

        # --- patterns: comunica l'asse prezzo al renderer ---
        try:
            from .patterns_hook import get_patterns_service
            ps = get_patterns_service(self.controller, self.view, create=False)
            if ps is not None and hasattr(ps, "renderer"):
                ax_price = (locals().get("ax_price")
                            or getattr(self, "ax_price", None)
                            or getattr(self, "axes_price", None))
                if ax_price is None:
                    fig = getattr(getattr(self.view, "canvas", None), "figure", None)
                    if fig and getattr(fig, "axes", None):
                        ax_price = fig.axes[0]
                if ax_price is not None:
                    ps.renderer.set_axes(ax_price)
        except Exception as e:
            logger.debug(f"patterns set_axes failed: {e}")

        try:
            self.canvas.draw_idle()
        except Exception:
            self.canvas.draw()

        # --- hook pattern detection ---
        try:
            call_patterns_detection(self.controller, self.view, df if 'df' in locals() else df2)
        except Exception as e:
            logger.debug(f'patterns hook error: {e}')

    def _update_plot_finplot(self, df: pd.DataFrame, quantiles: Optional[dict] = None):
        """Update plot using PyQtGraph (high-performance financial charting)"""
        try:
            if df is None or df.empty:
                return

            # Save current view range before update (unless explicitly cleared by user action)
            saved_range = getattr(self, '_saved_view_range', None)
            if hasattr(self, 'ax') and self.ax is not None and saved_range is not False:
                try:
                    view_range = self.ax.viewRange()
                    self._saved_view_range = view_range
                    # PlutoTouch logger.debug(f"Saved view range before update: {view_range}")
                except Exception as e:
                    logger.debug(f"Could not save view range: {e}")

            self._last_df = df.copy()

            # Prepare data
            df2 = df.copy()
            y_col = 'close' if 'close' in df2.columns else 'price'
            df2['ts_utc'] = pd.to_numeric(df2['ts_utc'], errors='coerce')
            df2[y_col] = pd.to_numeric(df2[y_col], errors='coerce')
            df2 = df2.dropna(subset=['ts_utc', y_col]).reset_index(drop=True)

            # Detect and convert timestamp unit (nanoseconds vs milliseconds)
            if not df2.empty:
                sample_ts = df2['ts_utc'].iloc[0]
                # Nanoseconds: > 1e15 (year 2001 in nanoseconds)
                # Milliseconds: > 1e12 (year 2001 in milliseconds)
                # Seconds: > 1e9 (year 2001 in seconds)
                if sample_ts > 1e15:
                    logger.debug(f"Converting timestamps from nanoseconds to milliseconds (sample: {sample_ts})")
                    df2['ts_utc'] = df2['ts_utc'] / 1e6  # nanoseconds to milliseconds
                elif sample_ts < 1e10:
                    logger.debug(f"Converting timestamps from seconds to milliseconds (sample: {sample_ts})")
                    df2['ts_utc'] = df2['ts_utc'] * 1000  # seconds to milliseconds

            # Validate timestamp range (reject corrupted timestamps)
            # Valid range: 1970-2100 (in milliseconds: ~0 to ~4e12)
            min_valid_ts = 0
            max_valid_ts = 4e12  # Year 2096
            ts_valid_mask = (df2['ts_utc'] >= min_valid_ts) & (df2['ts_utc'] <= max_valid_ts)
            invalid_count = (~ts_valid_mask).sum()
            if invalid_count > 0:
                logger.warning(f"Dropping {invalid_count} rows with invalid timestamps (out of range {min_valid_ts} - {max_valid_ts})")
                logger.debug(f"Sample invalid timestamps: {df2.loc[~ts_valid_mask, 'ts_utc'].head().tolist()}")
                df2 = df2[ts_valid_mask].reset_index(drop=True)

            if df2.empty:
                logger.error("All timestamps were invalid - cannot plot")
                return

            # Convert timestamp to datetime
            df2['time'] = pd.to_datetime(df2['ts_utc'], unit='ms', utc=True, errors='coerce')
            # Drop any rows where datetime conversion failed
            df2 = df2.dropna(subset=['time']).reset_index(drop=True)
            df2 = df2.set_index('time')

            # Get enabled indicators to check if we need 2 subplots
            enabled_indicator_names = get_setting("indicators.enabled_list", [])
            indicator_colors = get_setting("indicators.colors", {})

            # Filter enabled indicators to only include those that exist in TA-Lib
            if ENHANCED_INDICATORS_AVAILABLE and enabled_indicator_names:
                indicators_system_temp = BTALibIndicators()
                available_names = set(indicators_system_temp.enabled_indicators.keys())
                enabled_indicator_names = [name for name in enabled_indicator_names if name in available_names]

            # Check what subplots we need
            has_normalized = False
            has_custom = False
            if ENHANCED_INDICATORS_AVAILABLE and enabled_indicator_names:
                for name in enabled_indicator_names:
                    range_info = indicator_range_classifier.get_range_info(name)
                    if range_info:
                        if range_info.subplot_recommendation == 'normalized_subplot':
                            has_normalized = True
                        elif range_info.subplot_recommendation == 'custom_subplot':
                            has_custom = True

            # Manage subplots in PyQtGraph GraphicsLayoutWidget
            graphics_layout = self.view.graphics_layout
            current_rows = len(self.view.finplot_axes) if hasattr(self.view, 'finplot_axes') else 1
            # Count needed rows: 1 (price) + normalized + custom
            needed_rows = 1 + (1 if has_normalized else 0) + (1 if has_custom else 0)

            # Recreate plots if row count changed
            if current_rows != needed_rows:
                # Save forecast items before clearing
                forecast_items = []
                try:
                    from .forecast_service import get_forecast_service
                    forecast_svc = get_forecast_service(self.controller, self.view, create=False)
                    if forecast_svc and hasattr(forecast_svc, '_forecasts'):
                        for f in forecast_svc._forecasts:
                            for art in f.get("artists", []):
                                forecast_items.append(art)
                except Exception:
                    pass

                # Unlink volume plot before clearing to prevent ViewBox deletion errors
                if hasattr(self.view, 'volume_plot') and self.view.volume_plot is not None:
                    try:
                        self.view.volume_plot.setXLink(None)  # Unlink from main plot
                    except Exception:
                        pass

                graphics_layout.clear()
                self.view.finplot_axes = []
                # Reset click filter flag so it will be reinstalled
                self._click_filter_installed = False

                # Set minimal spacing between plots (using central item's layout)
                graphics_layout.ci.layout.setSpacing(0)
                graphics_layout.ci.layout.setContentsMargins(0, 0, 0, 0)

                # Create main price plot WITHOUT bottom axis (will be on volume plot)
                main_plot = graphics_layout.addPlot(row=0, col=0)
                main_plot.hideAxis('bottom')  # Hide x-axis labels on main plot
                main_plot.showGrid(x=True, y=True, alpha=0.3)
                main_plot.setMinimumHeight(300)
                # Minimize margins
                main_plot.setContentsMargins(0, 0, 0, 0)
                main_plot.getViewBox().setContentsMargins(0, 0, 0, 0)
                # Add legend (movable by default in PyQtGraph)
                main_plot.addLegend(offset=(10, 10))
                self.view.finplot_axes.append(main_plot)
                self.view.main_plot = main_plot

                row_idx = 1
                if has_normalized:
                    # Create normalized subplot with DateAxisItem
                    from ...chart_tab.ui_builder import DateAxisItem
                    normalized_date_axis = DateAxisItem(orientation='bottom')
                    normalized_date_axis.set_date_format(get_setting("chart.date_format", "YYYY-MM-DD"))
                    
                    normalized_plot = graphics_layout.addPlot(row=row_idx, col=0, axisItems={'bottom': normalized_date_axis})
                    normalized_plot.showGrid(x=True, y=True, alpha=0.3)
                    normalized_plot.setYRange(0, 100)  # Normalized range 0-100
                    normalized_plot.setMaximumHeight(63)
                    # Minimize margins
                    normalized_plot.setContentsMargins(0, 0, 0, 0)
                    normalized_plot.getViewBox().setContentsMargins(0, 0, 0, 0)
                    # Hide Y-axis labels
                    normalized_plot.showAxis('left', False)
                    # Add legend (movable by default in PyQtGraph)
                    normalized_plot.addLegend(offset=(10, 10))
                    # Link X axis for synchronized zoom/pan but NOT Y axis
                    normalized_plot.setXLink(main_plot)
                    # Disable Y-axis zoom and pan for this subplot
                    normalized_plot.setMouseEnabled(y=False)
                    normalized_plot.getViewBox().setMouseEnabled(y=False)
                    self.view.finplot_axes.append(normalized_plot)
                    self.view.normalized_plot = normalized_plot
                    row_idx += 1
                else:
                    self.view.normalized_plot = None

                if has_custom:
                    # Create custom subplot with DateAxisItem
                    from ...chart_tab.ui_builder import DateAxisItem
                    custom_date_axis = DateAxisItem(orientation='bottom')
                    custom_date_axis.set_date_format(get_setting("chart.date_format", "YYYY-MM-DD"))
                    
                    custom_plot = graphics_layout.addPlot(row=row_idx, col=0, axisItems={'bottom': custom_date_axis})
                    custom_plot.showGrid(x=True, y=True, alpha=0.3)
                    custom_plot.setMaximumHeight(63)
                    # Minimize margins
                    custom_plot.setContentsMargins(0, 0, 0, 0)
                    custom_plot.getViewBox().setContentsMargins(0, 0, 0, 0)
                    # Hide Y-axis labels
                    custom_plot.showAxis('left', False)
                    # Add legend (movable by default in PyQtGraph)
                    custom_plot.addLegend(offset=(10, 10))
                    # Link X axis for synchronized zoom/pan but NOT Y axis
                    custom_plot.setXLink(main_plot)
                    # Disable Y-axis zoom and pan for this subplot
                    custom_plot.setMouseEnabled(y=False)
                    custom_plot.getViewBox().setMouseEnabled(y=False)
                    self.view.finplot_axes.append(custom_plot)
                    self.view.custom_plot = custom_plot
                    row_idx += 1
                else:
                    self.view.custom_plot = None

                # Recreate volume subplot with better height and auto-range
                # Always add volume plot at the bottom with the date axis
                date_axis = DateAxisItem(orientation='bottom')
                volume_plot = graphics_layout.addPlot(row=row_idx, col=0, axisItems={'bottom': date_axis})
                volume_plot.setLabel('left', 'Volume')
                volume_plot.setMaximumHeight(120)  # Increased from 80 to 120 for better visibility
                volume_plot.showGrid(x=True, y=True, alpha=0.3)
                volume_plot.setContentsMargins(0, 0, 0, 0)
                volume_plot.getViewBox().setContentsMargins(0, 0, 0, 0)
                # Enable auto-range for Y axis
                volume_plot.enableAutoRange(axis='y', enable=True)
                # Hide Y-axis labels (volume values not important)
                volume_plot.showAxis('left', False)
                # Link x-axis to main plot for synchronized zoom/pan
                volume_plot.setXLink(main_plot)
                self.view.finplot_axes.append(volume_plot)
                self.view.volume_plot = volume_plot
                self.view.volume_bars = None  # Reset volume bars reference

                # Re-plot forecasts from saved data (items are invalidated after graphics_layout.clear())
                try:
                    from .forecast_service import get_forecast_service
                    forecast_svc = get_forecast_service(self.controller, self.view, create=False)
                    if forecast_svc and hasattr(forecast_svc, '_forecasts') and forecast_svc._forecasts:
                        logger.debug(f"Re-plotting {len(forecast_svc._forecasts)} forecasts after subplot change")
                        # Save forecasts data before clearing
                        saved_forecasts = []
                        for f in forecast_svc._forecasts:
                            saved_forecasts.append(f.get("quantiles"))
                        # Clear artists (they're invalid now)
                        forecast_svc._forecasts = []
                        # Re-plot each forecast
                        for quantiles in saved_forecasts:
                            if quantiles:
                                source = quantiles.get("source", "basic")
                                forecast_svc._plot_forecast_overlay(quantiles, source=source)
                        logger.debug(f"Successfully re-plotted {len(saved_forecasts)} forecasts")
                except Exception as e:
                    logger.warning(f"Failed to restore forecasts after subplot change: {e}")
            else:
                # Remove only candlestick/indicator items from previous render, preserve forecast overlays
                # Check if we have saved chart items from previous render
                if hasattr(self, '_chart_items'):
                    for plot, items in self._chart_items.items():
                        for item in items:
                            try:
                                plot.removeItem(item)
                            except Exception:
                                pass
                    # PlutoTouch logger.debug(f"Removed {sum(len(items) for items in self._chart_items.values())} chart items, preserved forecasts")

                # Initialize dict to save new chart items (candlestick + indicators)
                self._chart_items = {}

                # Ensure references are set correctly
                if len(self.view.finplot_axes) >= 1:
                    self.view.main_plot = self.view.finplot_axes[0]

                # Set normalized and custom plot references based on what exists
                row_idx = 1
                if has_normalized and len(self.view.finplot_axes) > row_idx:
                    self.view.normalized_plot = self.view.finplot_axes[row_idx]
                    row_idx += 1
                else:
                    self.view.normalized_plot = None

                if has_custom and len(self.view.finplot_axes) > row_idx:
                    self.view.custom_plot = self.view.finplot_axes[row_idx]
                else:
                    self.view.custom_plot = None

            # Get plot references
            ax_price = self.view.main_plot
            ax_normalized = getattr(self.view, 'normalized_plot', None)
            ax_custom = getattr(self.view, 'custom_plot', None)

            # Check if data needs pip format conversion for display
            sample_close = df2['close'].iloc[0] if 'close' in df2.columns else df2[y_col].iloc[0]
            is_pip_format = sample_close > 100
            pip_divisor = 10000.0 if is_pip_format else 1.0

            # Determine chart mode (candles vs line)
            chart_mode = get_setting('chart.price_mode', 'candles')

            # Plot price data (convert from pip format if needed)
            if chart_mode == 'candles' and {'open', 'high', 'low', 'close'}.issubset(df2.columns):
                # Convert candlestick data from pip format
                candle_df = df2[['open', 'high', 'low', 'close']].copy()
                if is_pip_format:
                    candle_df = candle_df / pip_divisor
                # Get candle colors from settings
                up_color = get_setting('candle_up_color', '#2ecc71')
                down_color = get_setting('candle_down_color', '#e74c3c')
                candle_item = add_candlestick(ax_price, candle_df, up_color=up_color, down_color=down_color)
                # Save reference to candlestick item for later removal
                if not hasattr(self, '_chart_items'):
                    self._chart_items = {}
                if ax_price not in self._chart_items:
                    self._chart_items[ax_price] = []
                self._chart_items[ax_price].append(candle_item)
            else:
                # Plot line chart (convert from pip format) using timestamps
                x_data = df2.index.astype(np.int64) / 10**9  # Convert datetime to timestamp (seconds)
                y_data = df2[y_col].values / pip_divisor
                symbol = getattr(self.view, 'symbol', 'Price')
                # Get price line color from settings
                price_line_color = get_setting('price_line_color', '#2196F3')
                line_item = ax_price.plot(x_data, y_data, pen=pg.mkPen(price_line_color, width=1.5), name=symbol)
                # Save reference to line item for later removal
                if not hasattr(self, '_chart_items'):
                    self._chart_items = {}
                if ax_price not in self._chart_items:
                    self._chart_items[ax_price] = []
                self._chart_items[ax_price].append(line_item)

            # Plot volume bars on volume subplot (1/10 height, 66% opacity)
            if hasattr(self.view, 'volume_plot') and 'volume' in df2.columns:
                ax_volume = self.view.volume_plot

                # Clear existing volume bars
                if hasattr(self.view, 'volume_bars') and self.view.volume_bars is not None:
                    ax_volume.removeItem(self.view.volume_bars)
                    self.view.volume_bars = None

                # Prepare volume data
                x_data = df2.index.astype(np.int64) / 10**9  # Convert datetime to timestamp (seconds)
                volume_data = df2['volume'].values

                # Determine bar colors based on price movement (green=up, red=down)
                # Compare close vs open (or close vs previous close if open not available)
                if 'close' in df2.columns and 'open' in df2.columns:
                    colors = np.where(df2['close'] >= df2['open'],
                                     '#2ecc71',  # Green for up
                                     '#e74c3c')  # Red for down
                elif 'close' in df2.columns:
                    # Use close vs previous close
                    close_diff = df2['close'].diff()
                    colors = np.where(close_diff >= 0,
                                     '#2ecc71',  # Green for up
                                     '#e74c3c')  # Red for down
                else:
                    colors = ['#2196F3'] * len(x_data)  # Default blue

                # Create volume bars with 66% opacity
                # Calculate bar width (spacing between timestamps)
                if len(x_data) > 1:
                    bar_width = (x_data[-1] - x_data[-2]) * 0.8  # 80% of spacing
                else:
                    bar_width = 60  # Default 60 seconds

                # Create individual bars with colors and 66% opacity
                brushes = []
                for color in colors:
                    qcolor = QColor(color)
                    qcolor.setAlphaF(0.66)  # 66% opacity
                    brushes.append(qcolor)

                # Use BarGraphItem for volume bars
                volume_bars = pg.BarGraphItem(
                    x=x_data,
                    height=volume_data,
                    width=bar_width,
                    brushes=brushes,
                    pen=None  # No border for cleaner look
                )
                ax_volume.addItem(volume_bars)
                self.view.volume_bars = volume_bars

                # Auto-scale volume axis
                ax_volume.enableAutoRange(axis='y')

            # Plot indicators
            if ENHANCED_INDICATORS_AVAILABLE and enabled_indicator_names:
                indicators_system = BTALibIndicators()

                # Prepare OHLCV DataFrame
                # Check if data is in pip format (values > 100 indicate pip format for forex)
                sample_close = df2['close'].iloc[0] if 'close' in df2.columns else df2[y_col].iloc[0]
                is_pip_format = sample_close > 100  # Forex prices are typically 0.5-2.0, so >100 means pip format

                if is_pip_format:
                    # Convert from pip format (11700) to actual price (1.1700)
                    pip_divisor = 10000.0
                    logger.debug(f"Converting price data from pip format (divisor={pip_divisor})")
                    indicator_df = pd.DataFrame({
                        'open': df2.get('open', df2[y_col]) / pip_divisor,
                        'high': df2.get('high', df2[y_col]) / pip_divisor,
                        'low': df2.get('low', df2[y_col]) / pip_divisor,
                        'close': (df2['close'] if 'close' in df2.columns else df2[y_col]) / pip_divisor,
                        'volume': df2.get('volume', pd.Series([1.0] * len(df2), index=df2.index))
                    })
                else:
                    indicator_df = pd.DataFrame({
                        'open': df2.get('open', df2[y_col]),
                        'high': df2.get('high', df2[y_col]),
                        'low': df2.get('low', df2[y_col]),
                        'close': df2['close'] if 'close' in df2.columns else df2[y_col],
                        'volume': df2.get('volume', pd.Series([1.0] * len(df2), index=df2.index))
                    })

                # Debug: log price data range
                # PlutoTouch logger.debug(f"Price data range - close: min={indicator_df['close'].min():.6f}, max={indicator_df['close'].max():.6f}")

                for indicator_name in enabled_indicator_names:
                    try:
                        # Calculate indicator using correct API
                        result_dict = indicators_system.calculate_indicator(indicator_df, indicator_name)

                        if not result_dict:
                            continue

                        # Extract first result or all results for multi-series
                        if len(result_dict) == 1:
                            result = next(iter(result_dict.values()))
                        else:
                            # Multi-series indicator - plot all
                            result = result_dict

                        # Get subplot recommendation
                        range_info = indicator_range_classifier.get_range_info(indicator_name)
                        subplot_type = range_info.subplot_recommendation if range_info else 'main_chart'

                        # Get color
                        color = indicator_colors.get(indicator_name, None)

                        # Plot based on subplot
                        if subplot_type == 'normalized_subplot' and ax_normalized:
                            target_ax = ax_normalized
                        elif subplot_type == 'custom_subplot' and ax_custom:
                            target_ax = ax_custom
                        else:
                            target_ax = ax_price

                        # Prepare x-axis data (timestamps in seconds)
                        x_data = df2.index.astype(np.int64) / 10**9  # Convert datetime to timestamp (seconds)

                        # Convert color to PyQtGraph format
                        pg_pen = pg.mkPen(color if color else '#FFA726', width=1.5)

                        if isinstance(result, pd.Series):
                            # Single series - plot with PyQtGraph
                            y_data = result.values
                            # Debug: log indicator value range
                            valid_data = y_data[~np.isnan(y_data)]
                            # PlutoTouch if len(valid_data) > 0:
                                # PlutoTouch logger.debug(f"Indicator {indicator_name} ({subplot_type}) - min={valid_data.min():.2f}, max={valid_data.max():.2f}, mean={valid_data.mean():.2f}")
                            if len(y_data) == len(x_data):
                                indicator_item = target_ax.plot(x_data, y_data, pen=pg_pen, name=indicator_name)
                                # Save reference to indicator item for later removal
                                if not hasattr(self, '_chart_items'):
                                    self._chart_items = {}
                                if target_ax not in self._chart_items:
                                    self._chart_items[target_ax] = []
                                self._chart_items[target_ax].append(indicator_item)
                        elif isinstance(result, dict):
                            # Multi-series (MACD, Bollinger Bands, etc.)
                            for i, (key, series) in enumerate(result.items()):
                                if isinstance(series, pd.Series):
                                    y_data = series.values
                                    if len(y_data) == len(x_data):
                                        # Use different colors for multi-series
                                        colors = ['#FFA726', '#66BB6A', '#EF5350']
                                        pg_pen = pg.mkPen(colors[i % len(colors)], width=1.5)
                                        indicator_item = target_ax.plot(x_data, y_data, pen=pg_pen, name=key)
                                        # Save reference to indicator item for later removal
                                        if not hasattr(self, '_chart_items'):
                                            self._chart_items = {}
                                        if target_ax not in self._chart_items:
                                            self._chart_items[target_ax] = []
                                        self._chart_items[target_ax].append(indicator_item)

                    except Exception as e:
                        logger.error(f"Failed to plot indicator {indicator_name}: {e}")

            # Store axis reference
            self.ax = ax_price

            # Check if we have a saved view range to restore
            saved_view_range = getattr(self, '_saved_view_range', None)

            if saved_view_range is not None and saved_view_range is not False:
                # Restore previous view range
                try:
                    x_range, y_range = saved_view_range
                    ax_price.setXRange(x_range[0], x_range[1], padding=0)
                    ax_price.setYRange(y_range[0], y_range[1], padding=0)
                    ax_price.enableAutoRange(enable=False)
                    # PlutoTouch logger.debug(f"Restored view range: X={x_range}, Y={y_range}")
                except Exception as e:
                    logger.warning(f"Failed to restore view range: {e}")
                    saved_view_range = None
            elif saved_view_range is False:
                # False means user intentionally changed timeframe/range, reset to None for future saves
                self._saved_view_range = None
                saved_view_range = None
                logger.debug("Skipped view range restore due to user action")

            # Set initial zoom only if no saved view range
            if saved_view_range is None:
                # Show last N bars based on timeframe
                n_bars_to_show = 100  # Default for 1m
                if hasattr(self, 'timeframe'):
                    tf = self.timeframe.lower()
                    if '1m' in tf:
                        n_bars_to_show = 100
                    elif '5m' in tf:
                        n_bars_to_show = 80
                    elif '15m' in tf:
                        n_bars_to_show = 60
                    elif '1h' in tf or '60m' in tf:
                        n_bars_to_show = 40
                    elif '4h' in tf:
                        n_bars_to_show = 30
                    elif '1d' in tf:
                        n_bars_to_show = 20

                # Set X range to show last n_bars_to_show
                if len(x_data) > n_bars_to_show:
                    x_min = x_data[-n_bars_to_show]
                    x_max = x_data[-1]
                    ax_price.setXRange(x_min, x_max, padding=0.02)

                    # Set Y range based on visible data
                    visible_y = y_data[-n_bars_to_show:]
                    y_min = np.nanmin(visible_y)
                    y_max = np.nanmax(visible_y)
                    y_padding = (y_max - y_min) * 0.1
                    ax_price.setYRange(y_min - y_padding, y_max + y_padding, padding=0)

                    # Disable auto-range after setting initial range
                    ax_price.enableAutoRange(enable=False)

            # Setup mouse tracking for coordinates display
            self._setup_mouse_tracking(ax_price, df2)

            # Apply theme colors to PyQtGraph elements
            self.apply_theme_to_pyqtgraph()

            # PlutoTouch logger.info(f"Chart updated: {len(df2)} candles, {len(enabled_indicator_names)} indicators, {needed_rows} subplots")

        except Exception as e:
            logger.error(f"Error in _update_plot_finplot: {e}")
            import traceback
            traceback.print_exc()

    def _setup_mouse_tracking(self, plot_item, df):
        """Setup mouse tracking to display coordinates and price"""
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QDialog, QFrame
        from PySide6.QtCore import Qt as QtCore_Qt, QPoint
        from PySide6.QtGui import QFont
        from datetime import datetime
        from ..services.draggable_legend import DraggableLegend

        # Create or update mouse position label as draggable QFrame widget
        if self._mouse_label is None:
            # Get the plot widget as parent
            parent = getattr(self.view, 'finplot_widget', self.view)

            # Create draggable frame
            mouse_frame = QFrame(parent)
            mouse_frame.setFrameStyle(QFrame.Box | QFrame.Plain)
            mouse_frame.setLineWidth(1)
            mouse_frame.setStyleSheet("""
                QFrame {
                    background-color: rgba(40, 40, 40, 220);
                    border: 1px solid #666;
                    border-radius: 4px;
                    padding: 5px;
                }
            """)

            # Add label inside frame
            layout = QVBoxLayout(mouse_frame)
            layout.setContentsMargins(6, 4, 6, 4)
            label = QLabel("X: --\nY: --\nPrice: --")
            label_font = QFont()
            label_font.setPointSize(8)
            label.setFont(label_font)
            label.setStyleSheet("color: #ffffff; background: transparent; border: none;")
            layout.addWidget(label)

            # Make it draggable
            mouse_frame._dragging = False
            mouse_frame._drag_offset = QPoint()
            mouse_frame._label = label

            def mousePressEvent(event):
                if event.button() == QtCore_Qt.MouseButton.LeftButton:
                    mouse_frame._dragging = True
                    mouse_frame._drag_offset = event.pos()
                    mouse_frame.setCursor(QtCore_Qt.CursorShape.ClosedHandCursor)

            def mouseMoveEvent(event):
                if mouse_frame._dragging:
                    new_pos = mouse_frame.mapToParent(event.pos() - mouse_frame._drag_offset)
                    mouse_frame.move(new_pos)

            def mouseReleaseEvent(event):
                if event.button() == QtCore_Qt.MouseButton.LeftButton and mouse_frame._dragging:
                    mouse_frame._dragging = False
                    mouse_frame.setCursor(QtCore_Qt.CursorShape.ArrowCursor)
                    # Save position
                    from forex_diffusion.utils.user_settings import set_setting
                    set_setting('mouse_info_position', {'x': mouse_frame.x(), 'y': mouse_frame.y()})

            mouse_frame.mousePressEvent = mousePressEvent
            mouse_frame.mouseMoveEvent = mouseMoveEvent
            mouse_frame.mouseReleaseEvent = mouseReleaseEvent

            # Load saved position or default to top-right
            from forex_diffusion.utils.user_settings import get_setting
            saved_pos = get_setting('mouse_info_position')
            if saved_pos and isinstance(saved_pos, dict):
                mouse_frame.move(saved_pos.get('x', 10), saved_pos.get('y', 10))
            else:
                # Default: top-right with margin
                mouse_frame.move(parent.width() - 200, 10)

            mouse_frame.show()
            mouse_frame.raise_()
            self._mouse_label = mouse_frame

        # Remove old proxy if exists
        if self._mouse_proxy is not None:
            try:
                plot_item.scene().sigMouseMoved.disconnect(self._mouse_proxy)
            except:
                pass

        # Create mouse move handler
        def mouse_moved(pos):
            try:
                if plot_item.sceneBoundingRect().contains(pos):
                    mouse_point = plot_item.vb.mapSceneToView(pos)
                    x_timestamp = mouse_point.x()
                    y_price = mouse_point.y()

                    # Convert timestamp to datetime
                    try:
                        dt = datetime.fromtimestamp(x_timestamp)
                        date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        date_str = f"{x_timestamp:.0f}"

                    # Update QLabel text inside frame
                    label_text = f"X: {date_str}\nY: {y_price:.5f}\nPrice: {y_price:.5f}"
                    self._mouse_label._label.setText(label_text)
                    self._mouse_label.adjustSize()  # Resize frame to fit content
            except Exception as e:
                pass

        # Connect signal
        self._mouse_proxy = mouse_moved
        plot_item.scene().sigMouseMoved.connect(mouse_moved)

        # Set up mouse click handler for forecast (Alt+Click)
        def mouse_clicked(event):
            from PySide6.QtCore import Qt as QtCore_Qt

            # PyQtGraph MouseClickEvent doesn't have type() method - it's already a click event
            # Get modifiers from the event
            try:
                modifiers = event.modifiers()
                button = event.button()
                logger.debug(f"Mouse clicked: button={button}, modifiers={modifiers}, Alt={bool(modifiers & QtCore_Qt.KeyboardModifier.AltModifier)}")
            except Exception as e:
                logger.debug(f"Failed to get event modifiers: {e}")
                return

            # Check if click is on any scatter plot items (e.g., pattern badges)
            # PyQtGraph scatter items handle their own events via sigClicked
            # We need to check if there are any clickable scatter items and give them priority
            # by not consuming the event if click might be on a scatter point
            try:
                # Get all items in the plot
                all_items = plot_item.listDataItems()
                # Check if any ScatterPlotItem exists with connected signals
                has_clickable_scatters = any(
                    hasattr(item, 'sigClicked') and
                    hasattr(item, 'scatter') and  # ScatterPlotItem has scatter attribute
                    item.scatter is not None
                    for item in all_items
                )

                if has_clickable_scatters:
                    # There are clickable scatter items - check if click is near any point
                    scene_pos = event.scenePos()
                    view_pos = plot_item.vb.mapSceneToView(scene_pos)

                    for item in all_items:
                        if hasattr(item, 'scatter') and item.scatter is not None:
                            # Get scatter points
                            try:
                                points = item.scatter.points()
                                for point in points:
                                    point_pos = point.pos()
                                    # Calculate distance in view coordinates
                                    dx = abs(point_pos.x() - view_pos.x())
                                    dy = abs(point_pos.y() - view_pos.y())

                                    # Get view range to calculate pixel distance
                                    view_range = plot_item.vb.viewRange()
                                    view_width = view_range[0][1] - view_range[0][0]
                                    view_height = view_range[1][1] - view_range[1][0]

                                    # Widget size
                                    widget_rect = plot_item.vb.rect()
                                    px_per_unit_x = widget_rect.width() / view_width if view_width > 0 else 1
                                    px_per_unit_y = widget_rect.height() / view_height if view_height > 0 else 1

                                    # Distance in pixels
                                    dist_px = ((dx * px_per_unit_x) ** 2 + (dy * px_per_unit_y) ** 2) ** 0.5

                                    # If within 20 pixels of a scatter point, let it handle the event
                                    if dist_px < 20:
                                        logger.debug(f"Click near scatter point (dist={dist_px:.1f}px), letting item handle it")
                                        return
                            except Exception as e:
                                logger.debug(f"Error checking scatter points: {e}")
                                continue
            except Exception as e:
                logger.debug(f"Error checking for clickable items: {e}")

            # Check for left button + Alt modifier (forecast trigger)
            # button is MouseButton enum, check for LeftButton
            if button == QtCore_Qt.MouseButton.LeftButton and (modifiers & QtCore_Qt.KeyboardModifier.AltModifier):
                logger.info("Alt+Click detected, triggering forecast")
                # Get mouse position in view coordinates
                scene_pos = event.scenePos()
                if plot_item.sceneBoundingRect().contains(scene_pos):
                    view_pos = plot_item.vb.mapSceneToView(scene_pos)

                    # Create a mock event object compatible with _on_canvas_click
                    class MockGUI:
                        def __init__(self, mods):
                            self._mods = mods
                        def modifiers(self):
                            return self._mods

                    class MockEvent:
                        def __init__(self, x_timestamp, y_price, modifiers):
                            self.xdata = x_timestamp
                            self.ydata = y_price
                            self.button = 1
                            self._gui_event = MockGUI(modifiers)

                        @property
                        def guiEvent(self):
                            return self._gui_event

                    mock_event = MockEvent(view_pos.x(), view_pos.y(), modifiers)

                    # Call forecast handler from interaction service
                    try:
                        from ..services.interaction_service import InteractionService
                        if hasattr(self, 'controller') and hasattr(self.controller, 'interaction_service'):
                            self.controller.interaction_service._on_canvas_click(mock_event)
                    except Exception as e:
                        logger.error(f"Failed to trigger forecast: {e}")

        # Install event filter on the scene
        if not hasattr(self, '_click_filter_installed') or not self._click_filter_installed:
            plot_item.scene().sigMouseClicked.connect(mouse_clicked)
            self._click_filter_installed = True
            logger.info("Mouse click handler installed for forecast functionality")

    def _render_candles(self, df2: pd.DataFrame, x_dt: Optional[pd.Series] = None):
        """Render OHLC candles on the main axis."""
        try:
            required = {'open', 'high', 'low', 'close'}
            if not required.issubset(df2.columns):
                self._clear_candles()
                return

            self._clear_candles()

            # build xs (mdates numbers) either from provided compressed x or from timestamps
            if x_dt is None:
                ts = pd.to_numeric(df2['ts_utc'], errors='coerce').dropna()
                if ts.empty:
                    self._clear_candles()
                    return
                xs = mdates.date2num(pd.to_datetime(ts.astype('int64'), unit='ms', utc=True).tz_convert(None))
            else:
                # accept compressed numeric x (mdates numbers) or datetime-like
                try:
                    arr = np.asarray(x_dt)
                    if np.issubdtype(arr.dtype, np.number):
                        xs = arr.astype(float)
                    else:
                        xs = mdates.date2num(pd.to_datetime(x_dt))
                except Exception:
                    xs = mdates.date2num(pd.to_datetime(x_dt))

            diffs = xs[1:] - xs[:-1] if len(xs) > 1 else []
            positive = [d for d in diffs if d > 0]
            width = (float(np.median(positive)) * 0.7) if positive else self._default_candle_width()
            if width <= 0:
                width = self._default_candle_width()
            price_span = float(df2['high'].max() - df2['low'].min() or 1.0)
            epsilon = max(price_span * 0.0005, 1e-6)

            bull_color = self._get_color_mpl('candle_up_color', '#2ecc71')
            bear_color = self._get_color_mpl('candle_down_color', '#e74c3c')

            candle_artists: List = []
            for idx, row in df2.iterrows():
                try:
                    o = float(row['open']); c = float(row['close'])
                    h = float(row['high']); l = float(row['low'])
                except Exception:
                    continue
                if any(pd.isna(v) for v in (o, c, h, l)):
                    continue
                color = bull_color if c >= o else bear_color
                body_height = abs(c - o)
                if body_height == 0:
                    body_height = epsilon
                    lower = o - body_height / 2.0
                else:
                    lower = min(o, c)
                rect = Rectangle((xs[idx] - width / 2.0, lower), width, body_height,
                                 facecolor=color, edgecolor=color, linewidth=0.6,
                                 alpha=0.9, zorder=3)
                self.ax.add_patch(rect)
                wick_line, = self.ax.plot([xs[idx], xs[idx]], [l, h], color=color, linewidth=0.7,
                                           zorder=2.5, solid_capstyle='round')
                candle_artists.extend([rect, wick_line])

            self._candle_artists = candle_artists
            extra = dict(getattr(self, '_extra_legend_items', {}))
            bull_patch = Patch(facecolor=bull_color, edgecolor='none', label='Bull Candle')
            bear_patch = Patch(facecolor=bear_color, edgecolor='none', label='Bear Candle')
            extra['candles_bull'] = (bull_patch, bull_patch.get_label())
            extra['candles_bear'] = (bear_patch, bear_patch.get_label())
            self._extra_legend_items = extra
        except Exception as exc:
            logger.exception('Failed to render candles: {}', exc)
            self._clear_candles()

    def _clear_candles(self) -> None:
        """Remove candle artists and legend patches."""
        artists = list(getattr(self, '_candle_artists', []))
        for artist in artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._candle_artists = []
        extra = dict(getattr(self, '_extra_legend_items', {}))
        for key in ['candles_bull', 'candles_bear']:
            extra.pop(key, None)
        self._extra_legend_items = extra

    def _default_candle_width(self) -> float:
        """Fallback candle width expressed in Matplotlib date units."""
        tf = str(getattr(self, 'timeframe', '1m')).lower()
        try:
            if tf.endswith('m'):
                minutes = max(1, int(tf[:-1] or '1'))
                return minutes / (24 * 60) * 0.7
            if tf.endswith('h'):
                hours = max(1, int(tf[:-1] or '1'))
                return hours / 24 * 0.6
            if tf.endswith('d'):
                days = max(1, int(tf[:-1] or '1'))
                return days * 0.5
        except Exception:
            pass
        return 1 / (24 * 60) * 0.7

    def _ensure_osc_axis(self, need: bool):
        """Crea/mostra o nasconde l’asse “oscillatori” (inset sotto il main)."""
        mini_color = getattr(self, '_mini_chart_color', self._get_qcolor('mini_chart_bg', '#14181f' if getattr(self, '_is_dark', True) else '#f5f7fb'))
        if need and self._osc_ax is None:
            try:
                self._osc_ax = inset_axes(self.ax, width="100%", height="28%",
                                          loc="lower left", borderpad=1.2)
                self._osc_ax.set_facecolor(self._color_to_mpl(mini_color))
                self._osc_ax.grid(True, alpha=0.12)
            except Exception:
                self._osc_ax = None
        if not need and self._osc_ax is not None:
            try:
                self._osc_ax.cla()
                self._osc_ax.set_visible(False)
                self.canvas.draw_idle()
            except Exception:
                pass
        elif need and self._osc_ax is not None:
            try:
                self._osc_ax.set_facecolor(self._color_to_mpl(mini_color))
            except Exception:
                pass
    def _on_main_xlim_changed(self, ax):
        """Sync oscillator inset x-limits to main axis."""
        try:
            if self._osc_ax is not None:
                self._osc_ax.set_xlim(self.ax.get_xlim())
        except Exception:
            pass



    def _plot_indicators(self, df2: pd.DataFrame, x_dt: pd.Series):
        cfg = self._get_indicator_settings()
        if not isinstance(cfg, dict) or not cfg:
            self._ensure_osc_axis(False)
            return

        close = df2['close'] if 'close' in df2.columns else df2['price']
        high = df2.get('high', close)
        low = df2.get('low', close)

        # reuse artists se presenti
        def _get_art(key: str, color: str, style: str = '-'):
            arr = self._ind_artists.get(key)
            if arr:
                line = arr[0]
            else:
                (line,) = self.ax.plot([], [], linestyle=style, color=color, linewidth=1.2, alpha=0.95, label=key)
                self._ind_artists[key] = [line]
            return self._ind_artists[key][0]

        # --- Overlays sull’asse prezzo ---
        # SMA
        if cfg.get('use_sma', False):
            n = int(cfg.get('sma_n', 20))
            c = cfg.get('color_sma', '#7f7f7f')
            sma = self._sma(close, n)
            line = _get_art('SMA', c, '-')
            line.set_data(x_dt, sma.values)
            line.set_visible(True)
        else:
            if 'SMA' in self._ind_artists: self._ind_artists['SMA'][0].set_visible(False)

        # EMA fast/slow
        if cfg.get('use_ema', False):
            nf = int(cfg.get('ema_fast', 12));
            ns = int(cfg.get('ema_slow', 26))
            cf = cfg.get('color_ema', '#bcbd22')
            ema_f = self._ema(close, nf);
            ema_s = self._ema(close, ns)
            linef = _get_art('EMA_fast', cf, '-');
            lines = _get_art('EMA_slow', cf, '--')
            linef.set_data(x_dt, ema_f.values);
            linef.set_visible(True)
            lines.set_data(x_dt, ema_s.values);
            lines.set_visible(True)
        else:
            for k in ['EMA_fast', 'EMA_slow']:
                if k in self._ind_artists: self._ind_artists[k][0].set_visible(False)

        # Bollinger (upper/lower)
        if cfg.get('use_bollinger', False):
            n = int(cfg.get('bb_n', 20));
            k = float(cfg.get('bb_k', 2.0))
            c = cfg.get('color_bollinger', '#2ca02c')
            mid, up, lo = self._bollinger(close, n, k)
            upline = _get_art('BB_upper', c, ':');
            loline = _get_art('BB_lower', c, ':')
            upline.set_data(x_dt, up.values);
            upline.set_visible(True)
            loline.set_data(x_dt, lo.values);
            loline.set_visible(True)
        else:
            for k in ['BB_upper', 'BB_lower']:
                if k in self._ind_artists: self._ind_artists[k][0].set_visible(False)

        # Donchian (upper/lower)
        if cfg.get('use_don', False):
            n = int(cfg.get('don_n', 20));
            c = cfg.get('color_don', '#8c564b')
            up, lo = self._donchian(high, low, n)
            upline = _get_art('DON_upper', c, '-.');
            loline = _get_art('DON_lower', c, '-.')
            upline.set_data(x_dt, up.values);
            upline.set_visible(True)
            loline.set_data(x_dt, lo.values);
            loline.set_visible(True)
        else:
            for k in ['DON_upper', 'DON_lower']:
                if k in self._ind_artists: self._ind_artists[k][0].set_visible(False)

        # Keltner (upper/lower) – usa bb_n come finestra base
        if cfg.get('use_keltner', False):
            n = int(cfg.get('bb_n', 20));
            k = float(cfg.get('keltner_k', 1.5))
            c = cfg.get('color_keltner', '#17becf')
            up, lo = self._keltner(high, low, close, n, k)
            upline = _get_art('KELT_upper', c, '--');
            loline = _get_art('KELT_lower', c, '--')
            upline.set_data(x_dt, up.values);
            upline.set_visible(True)
            loline.set_data(x_dt, lo.values);
            loline.set_visible(True)
        else:
            for k in ['KELT_upper', 'KELT_lower']:
                if k in self._ind_artists: self._ind_artists[k][0].set_visible(False)

        # Wollinger-Keltner shaded area (overlap between Bollinger and Keltner), controllata da 'fill_wk'
        try:
            draw_wk = bool(cfg.get('fill_wk', get_setting("indicators.fill_wk", True)))
        except Exception:
            draw_wk = True
        if draw_wk and cfg.get('use_bollinger', False) and cfg.get('use_keltner', False):
            try:
                n_bb = int(cfg.get('bb_n', 20)); k_bb = float(cfg.get('bb_k', 2.0))
                _, bb_up, bb_lo = self._bollinger(close, n_bb, k_bb)
                n_k = int(cfg.get('bb_n', 20)); k_k = float(cfg.get('keltner_k', 1.5))
                k_up, k_lo = self._keltner(high, low, close, n_k, k_k)
                import numpy as _np
                minlen = min(len(x_dt), len(bb_up), len(bb_lo), len(k_up), len(k_lo))
                if minlen > 0:
                    upper = _np.minimum(bb_up.values[:minlen], k_up.values[:minlen])
                    lower = _np.maximum(bb_lo.values[:minlen], k_lo.values[:minlen])
                    valid = upper > lower
                    # rimuovi precedente fill se presente
                    if 'WK_fill' in self._ind_artists:
                        try:
                            self._ind_artists['WK_fill'][0].remove()
                        except Exception:
                            pass
                    fill_color = cfg.get('color_wk_fill', cfg.get('color_keltner', '#17becf'))
                    try:
                        poly = self.ax.fill_between(
                            x_dt[:minlen], lower, upper,
                            where=valid, interpolate=True,
                            color=fill_color, alpha=float(cfg.get('alpha_wk_fill', 0.12)), label=None
                        )
                    except Exception:
                        poly = self.ax.fill_between(
                            x_dt[:minlen], lower, upper,
                            color=fill_color, alpha=float(cfg.get('alpha_wk_fill', 0.12)), label=None
                        )
                    self._ind_artists['WK_fill'] = [poly]
            except Exception:
                if 'WK_fill' in self._ind_artists:
                    try:
                        self._ind_artists['WK_fill'][0].remove()
                    except Exception:
                        pass
                    self._ind_artists.pop('WK_fill', None)
        else:
            if 'WK_fill' in self._ind_artists:
                try:
                    self._ind_artists['WK_fill'][0].remove()
                except Exception:
                    pass
                self._ind_artists.pop('WK_fill', None)

        # --- Pannello oscillatori (RSI/MACD/ATR/Hurst) ---
        need_osc = any([cfg.get('use_rsi', False), cfg.get('use_macd', False),
                        cfg.get('use_atr', False), cfg.get('use_hurst', False)])
        self._ensure_osc_axis(need_osc)
        if self._osc_ax and need_osc:
            axo = self._osc_ax
            axo.set_visible(True)
            # align x-range with main axis
            try:
                axo.set_xlim(self.ax.get_xlim())
            except Exception:
                pass
            try:
                axo.cla()
            except Exception:
                pass
            axo.grid(True, alpha=0.2)

            # RSI
            if cfg.get('use_rsi', False):
                n = int(cfg.get('rsi_n', 14));
                c = cfg.get('color_rsi', '#1f77b4')
                rsi = self._rsi(close, n)
                axo.plot(x_dt, rsi.values, color=c, linewidth=1.0, label='RSI')
                axo.axhline(70, color=c, linestyle=':', linewidth=0.8, alpha=0.6)
                axo.axhline(30, color=c, linestyle=':', linewidth=0.8, alpha=0.6)
                axo.set_ylim(0, 100)

            # MACD
            if cfg.get('use_macd', False):
                f = int(cfg.get('macd_fast', 12));
                s = int(cfg.get('macd_slow', 26));
                sig = int(cfg.get('macd_signal', 9))
                c = cfg.get('color_macd', '#ff7f0e')
                macd, signal, hist = self._macd(close, f, s, sig)
                axo.plot(x_dt, macd.values, color=c, linewidth=1.0, label='MACD')
                axo.plot(x_dt, signal.values, color=c, linewidth=0.9, linestyle='--', alpha=0.8, label='Signal')
                try:
                    axo.fill_between(x_dt, 0, hist.values, alpha=0.12, color=c, step='pre')
                except Exception:
                    pass

            # ATR
            if cfg.get('use_atr', False):
                n = int(cfg.get('atr_n', 14));
                c = cfg.get('color_atr', '#d62728')
                atr = self._atr(high, low, close, n)
                axo.plot(x_dt, atr.values, color=c, linewidth=0.9, label='ATR', alpha=0.9)

            # Hurst
            if cfg.get('use_hurst', False):
                win = int(cfg.get('hurst_window', 64));
                c = cfg.get('color_hurst', '#9467bd')
                h = self._hurst_roll(close, win)
                axo.plot(x_dt, h.values, color=c, linewidth=0.9, label='H', alpha=0.9)
                axo.axhline(0.5, color=c, linestyle=':', linewidth=0.8, alpha=0.6)

            # cosmetica
            axes_color_qt = self._get_qcolor("axes_color", "#cfd6e1")
            axes_col = axes_color_qt.name(QColor.HexRgb)
            try:
                axo.tick_params(colors=axes_col, labelsize=8)
                for spine in axo.spines.values():
                    spine.set_color(axes_col)
            except Exception:
                pass

    def _plot_enhanced_indicators(self, df2: pd.DataFrame, x_dt: pd.Series):
        """
        Plot enhanced indicators from EnhancedIndicatorsDialog using MatplotlibSubplotService
        """
        if not ENHANCED_INDICATORS_AVAILABLE:
            return

        try:
            # Get enabled indicators from settings
            enabled_indicator_names = get_setting("indicators.enabled_list", [])
            if not enabled_indicator_names or not isinstance(enabled_indicator_names, list):
                return

            # Get indicator colors from settings
            indicator_colors = get_setting("indicators.colors", {})

            logger.debug(f"Plotting {len(enabled_indicator_names)} enhanced indicators")

            # Initialize BTALibIndicators system
            indicators_system = BTALibIndicators()

            # Prepare OHLCV data for indicator calculation
            close = df2['close'] if 'close' in df2.columns else df2['price']
            high = df2.get('high', close)
            low = df2.get('low', close)
            open_price = df2.get('open', close)
            volume = df2.get('volume', pd.Series([1.0] * len(close), index=close.index))

            # Check if we have normalized indicators
            has_normalized = any(
                indicator_range_classifier.get_range_info(name).subplot_recommendation == 'normalized_subplot'
                for name in enabled_indicator_names
                if indicator_range_classifier.get_range_info(name)
            )

            # Enable subplot service if we have normalized indicators
            if has_normalized:
                if not self._subplot_enabled:
                    self.enable_indicator_subplots()

                if self._subplot_service:
                    # Create subplots: main chart + normalized subplot
                    self._subplot_service.create_subplots(has_normalized=True)

            # Prepare DataFrame for indicator calculation
            indicator_df = pd.DataFrame({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

            # Calculate and plot each indicator
            for indicator_name in enabled_indicator_names:
                try:
                    # Get indicator config from enabled_indicators dict
                    indicator_config = indicators_system.enabled_indicators.get(indicator_name)
                    if not indicator_config:
                        logger.warning(f"Indicator {indicator_name} not found in system")
                        continue

                    # Calculate indicator using correct API: calculate_indicator(data, indicator_name, custom_params)
                    result_dict = indicators_system.calculate_indicator(indicator_df, indicator_name)

                    if not result_dict:
                        continue

                    # Get subplot recommendation
                    range_info = indicator_range_classifier.get_range_info(indicator_name)
                    subplot_type = range_info.subplot_recommendation if range_info else 'main_chart'

                    # Get indicator color
                    color = indicator_colors.get(indicator_name, None)

                    # Determine if multi-series or single series
                    is_multi_series = len(result_dict) > 1

                    # Plot based on subplot type
                    if subplot_type == 'main_chart' and not self._subplot_enabled:
                        # Overlay on main price chart
                        if is_multi_series:
                            # Handle multi-series indicators (like Bollinger Bands, MACD)
                            for key, series in result_dict.items():
                                if isinstance(series, pd.Series):
                                    self.ax.plot(x_dt, series.values, alpha=0.6, linewidth=1.0,
                                               label=key, linestyle='--')
                        else:
                            # Single series
                            result = next(iter(result_dict.values()))
                            if isinstance(result, pd.Series):
                                plot_kwargs = {'alpha': 0.7, 'linewidth': 1.2, 'label': indicator_name}
                                if color:
                                    plot_kwargs['color'] = color
                                self.ax.plot(x_dt, result.values, **plot_kwargs)

                    elif self._subplot_enabled and self._subplot_service:
                        # Use subplot service
                        if is_multi_series:
                            # Handle multi-series (e.g., bands)
                            # Check for common band patterns
                            keys_lower = [k.lower() for k in result_dict.keys()]
                            has_bands = any(k in keys_lower for k in ['upper', 'top', 'middle', 'lower', 'bottom'])

                            if has_bands and len(result_dict) >= 3:
                                # Try to identify upper, middle, lower
                                series_list = list(result_dict.values())
                                if len(series_list) == 3:
                                    self._subplot_service.plot_bands(
                                        indicator_name,
                                        series_list[0],  # upper/top
                                        series_list[1],  # middle
                                        series_list[2]   # lower/bottom
                                    )
                            else:
                                # Plot each series separately
                                for key, series in result_dict.items():
                                    if isinstance(series, pd.Series):
                                        self._subplot_service.plot_indicator(key, series)
                        else:
                            # Single series
                            result = next(iter(result_dict.values()))
                            if isinstance(result, pd.Series):
                                plot_kwargs = {}
                                if color:
                                    plot_kwargs['color'] = color
                                self._subplot_service.plot_indicator(indicator_name, result, **plot_kwargs)

                except Exception as e:
                    logger.error(f"Failed to plot indicator {indicator_name}: {e}")
                    continue

            # Update legend if we added indicators to main axis
            if not self._subplot_enabled:
                try:
                    self._refresh_legend_unique(loc='upper left')
                except Exception as e:
                    logger.error(f"Failed to refresh legend: {e}")

        except Exception as e:
            logger.error(f"Error in _plot_enhanced_indicators: {e}")

    def _sma(self, x: pd.Series, n: int) -> pd.Series:
        return x.rolling(n, min_periods=max(1, n // 2)).mean()

    def _ema(self, x: pd.Series, n: int) -> pd.Series:
        return x.ewm(span=max(1, n), adjust=False).mean()

    def _bollinger(self, x: pd.Series, n: int, k: float):
        ma = self._sma(x, n)
        sd = x.rolling(n, min_periods=max(1, n // 2)).std()
        return ma, ma + k * sd, ma - k * sd

    def _donchian(self, high: pd.Series, low: pd.Series, n: int):
        upper = high.rolling(n, min_periods=max(1, n // 2)).max()
        lower = low.rolling(n, min_periods=max(1, n // 2)).min()
        return upper, lower

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int):
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=max(1, n // 2)).mean()

    def _keltner(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int, k: float):
        ema_mid = self._ema(close, n)
        atr = self._atr(high, low, close, n)
        return ema_mid + k * atr, ema_mid - k * atr

    def _rsi(self, x: pd.Series, n: int):
        delta = x.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(n).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(n).mean()
        rs = np.where(loss == 0, np.nan, gain / loss)
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=x.index)

    def _macd(self, x: pd.Series, fast: int, slow: int, signal: int):
        ema_fast = self._ema(x, fast)
        ema_slow = self._ema(x, slow)
        macd = ema_fast - ema_slow
        sig = self._ema(macd, signal)
        hist = macd - sig
        return macd, sig, hist

    def _hurst_roll(self, x: pd.Series, window: int):
        # stima leggera Hurst per finestra mobile (attenzione: può essere costosa su dataset enormi)
        res = pd.Series(index=x.index, dtype=float)
        if window < 8 or len(x) < window:
            return res
        for i in range(window, len(x) + 1):
            seg = x.iloc[i - window:i].values
            if np.all(seg == seg[0]):
                res.iloc[i - 1] = np.nan
                continue
            Y = seg - seg.mean()
            Z = np.cumsum(Y)
            R = Z.max() - Z.min()
            S = Y.std()
            if not S or np.isnan(S):
                h = np.nan
            else:
                h = np.log(R / S) / np.log(window) if window > 1 else np.nan
            res.iloc[i - 1] = h
        return res

    def _get_qcolor(self, key: str, default: str) -> QColor:
        """Return a QColor. If theme preset != 'custom', ignore stored settings and use provided default."""
        # Normalize current preset
        preset = getattr(self, "_theme_current", None)
        if isinstance(preset, str):
            preset = preset.lower().strip()
        else:
            preset = ""
        # Read from settings only for custom
        if preset == "custom":
            raw = get_setting(key, default)
        else:
            raw = default
        text = str(raw).strip() if raw is not None else str(default)
        color = QColor(text)
        # Handle #RRGGBBAA (alpha at the end) by reordering
        if not color.isValid() and isinstance(text, str) and text.startswith('#') and len(text) == 9:
            alpha_last = text[-2:]
            rgb = text[1:-2]
            reordered = f"#{alpha_last}{rgb}"
            color = QColor(reordered)
        if not color.isValid():
            color = QColor(default)
        if not color.isValid():
            color = QColor('#000000')
        return color

    def _color_to_css(self, color: QColor) -> str:
        alpha = round(color.alphaF(), 3)
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha:.3f})"

    def _color_to_mpl(self, color: QColor):
        r, g, b, a = color.getRgbF()
        if a >= 0.999:
            return '#{0:02X}{1:02X}{2:02X}'.format(color.red(), color.green(), color.blue())
        return (r, g, b, a)

    def _hover_color(self, color: QColor) -> QColor:
        return color.lighter(120) if self._is_dark else color.darker(110)

    def _on_mode_toggled(self, checked: bool):
        """Toggle between candle and line rendering."""
        try:
            self._price_mode = 'candles' if checked else 'line'
            if hasattr(self, 'mode_btn') and self.mode_btn is not None:
                self.mode_btn.setText('Candles' if checked else 'Line')
            if self._last_df is not None and not self._last_df.empty:
                prev_xlim = self.ax.get_xlim() if hasattr(self.ax, 'get_xlim') else None
                prev_ylim = self.ax.get_ylim() if hasattr(self.ax, 'get_ylim') else None
                self.update_plot(self._last_df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)
        except Exception as exc:
            logger.debug('Price mode toggle failed: {}', exc)

    def _apply_theme(self, theme: str):
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication

        # Normalize preset
        t = (theme or "Dark").strip().lower()
        app = QApplication.instance()

        # Resolve is_dark based on preset
        def _is_system_dark() -> bool:
            try:
                pal = app.palette() if app else QPalette()
                col = pal.color(QPalette.Window)
                # lightnessF in [0..1]; dark if < ~0.5
                return float(getattr(col, "lightnessF", lambda: col.lightness() / 255.0)()) < 0.5
            except Exception:
                return True

        if t == "system":
            self._is_dark = _is_system_dark()
        elif t == "light":
            self._is_dark = False
        else:
            # include 'dark' and fallback (default to dark)
            self._is_dark = (t != "light")

        # Persist current preset string for color resolution logic
        self._theme_current = t  # used by _get_qcolor

        # Base colors: when preset != 'custom' we ignore stored settings and use defaults here
        window_bg_q = self._get_qcolor("window_bg", "#0f1115" if self._is_dark else "#f3f5f8")
        panel_bg_q  = self._get_qcolor("panel_bg",  "#12151b" if self._is_dark else "#ffffff")
        text_q      = self._get_qcolor("text_color", "#e0e0e0" if self._is_dark else "#1a1e25")
        chart_bg_q  = self._get_qcolor("chart_bg",  "#0f1115" if self._is_dark else "#ffffff")
        mini_bg_q   = self._get_qcolor("mini_chart_bg", "#14181f" if self._is_dark else "#f5f7fb")
        border_q    = self._get_qcolor("border_color", "#2a2f3a" if self._is_dark else "#c7cfdb")
        splitter_q  = self._get_qcolor("splitter_handle_color", "#2f3541" if self._is_dark else "#d0d7e6")
        bidask_q    = self._get_qcolor("bidask_color", "#ffd479" if self._is_dark else "#2d384f")
        price_line_q= self._get_qcolor("price_line_color", "#e0e0e0" if self._is_dark else "#1a1e25")
        legend_q    = self._get_qcolor("legend_text_color", "#cfd6e1" if self._is_dark else "#1a1e25")
        grid_q      = self._get_qcolor("grid_color", "#3a4250" if self._is_dark else "#d8dee9")
        tab_bg_q    = self._get_qcolor("tab_bg", panel_bg_q.name(QColor.HexRgb) if hasattr(panel_bg_q, 'name') else "#12151b")
        tab_text_q  = self._get_qcolor("tab_text_color", text_q.name(QColor.HexRgb) if hasattr(text_q, 'name') else "#e0e0e0")

        # Update CSS / Palette
        hover_css = self._color_to_css(self._hover_color(panel_bg_q))
        base_css = f"""
        QWidget {{ background-color: {self._color_to_css(window_bg_q)}; color: {self._color_to_css(text_q)}; }}
        QPushButton, QComboBox, QToolButton {{ background-color: {self._color_to_css(panel_bg_q)}; color: {self._color_to_css(text_q)}; border: 1px solid {self._color_to_css(border_q)}; padding: 4px 8px; border-radius: 4px; }}
        QPushButton:hover, QToolButton:hover {{ background-color: {hover_css}; }}
        QTableWidget, QListWidget {{ background-color: {self._color_to_css(panel_bg_q)}; color: {self._color_to_css(text_q)}; gridline-color: {self._color_to_css(border_q)}; }}
        QHeaderView::section {{ background-color: {self._color_to_css(panel_bg_q)}; color: {self._color_to_css(text_q)}; border: 0px; }}
        QSplitter::handle {{ background-color: {self._color_to_css(splitter_q)}; }}
        QWidget#chart_container {{ border: 1px solid {self._color_to_css(border_q)}; border-radius: 4px; }}
        QWidget#drawbar_container {{ border-bottom: 1px solid {self._color_to_css(border_q)}; }}
        QWidget#chartTab {{ border: 1px solid {self._color_to_css(border_q)}; border-radius: 4px; }}
        QTabWidget::pane {{ border: 1px solid {self._color_to_css(border_q)}; }}
        QTabBar::tab {{ background: {self._color_to_css(tab_bg_q)}; color: {self._color_to_css(tab_text_q)}; padding: 4px 10px; border: 1px solid {self._color_to_css(border_q)}; border-bottom-color: {self._color_to_css(border_q)}; }}
        QTabBar::tab:selected {{ background: {hover_css}; color: {self._color_to_css(tab_text_q)}; }}
        """
        custom_qss = get_setting("custom_qss", "")
        if app:
            app.setStyleSheet(base_css + "\n" + (custom_qss or ""))
            pal = QPalette()
            pal.setColor(QPalette.Window, window_bg_q)
            pal.setColor(QPalette.WindowText, text_q)
            pal.setColor(QPalette.Base, panel_bg_q)
            pal.setColor(QPalette.AlternateBase, panel_bg_q)
            pal.setColor(QPalette.Text, text_q)
            pal.setColor(QPalette.Button, panel_bg_q)
            pal.setColor(QPalette.ButtonText, text_q)
            app.setPalette(pal)

        # Update chart visuals
        try:
            mpl_chart_bg = self._color_to_mpl(chart_bg_q)
            self.canvas.figure.set_facecolor(mpl_chart_bg)
            self.ax.set_facecolor(mpl_chart_bg)
            if self._osc_ax is not None:
                self._osc_ax.set_facecolor(self._color_to_mpl(mini_bg_q))
        except Exception:
            pass

        # Store for other methods
        self._mini_chart_color = mini_bg_q
        if hasattr(self, '_price_line') and self._price_line is not None:
            try:
                self._price_line.set_color(self._color_to_mpl(price_line_q))
            except Exception:
                pass

        if hasattr(self.view, 'bidask_label') and self.view.bidask_label is not None:
            self.view.bidask_label.setStyleSheet(f"font-weight: bold; color: {self._color_to_css(bidask_q)};")

        # Persist ui_theme exactly as chosen preset (System/Light/Dark/Custom)
        try:
            from forex_diffusion.utils.user_settings import set_setting
            if t in ("system", "light", "dark", "custom"):
                set_setting("ui_theme", {"system": "System", "light": "Light", "dark": "Dark", "custom": "Custom"}[t])
            else:
                set_setting("ui_theme", "Dark" if self._is_dark else "Light")
        except Exception:
            pass

        # Try to recolor existing legend (if present) to new legend color
        try:
            existing = getattr(self.ax, 'legend_', None)
            if existing is not None:
                for txt in existing.get_texts() or []:
                    txt.set_color(self._color_to_mpl(legend_q))
                ttl = existing.get_title()
                if ttl:
                    ttl.set_color(self._color_to_mpl(legend_q))
        except Exception:
            pass

        try:
            self.canvas.draw()
        except Exception:
            pass

    def _get_color(self, key: str, default: str) -> str:
        return self._color_to_css(self._get_qcolor(key, default))

    def _get_color_mpl(self, key: str, default: str):
        return self._color_to_mpl(self._get_qcolor(key, default))

    def _open_color_settings(self):
        try:
            from forex_diffusion.ui.color_settings_dialog import ColorSettingsDialog
            dlg = ColorSettingsDialog(self.view)

            # Connect the theme changed signal to refresh the chart
            dlg.themeChanged.connect(lambda: self._refresh_theme_colors())

            if dlg.exec():
                # Theme has already been applied via signal if colors were saved
                pass
        except Exception as e:
            QMessageBox.warning(self.view, "Colors", str(e))

    def _refresh_theme_colors(self):
        """Refresh chart with new theme colors"""
        try:
            # Apply PyQtGraph theme
            self.apply_theme_to_pyqtgraph()

            # Re-draw plot to apply new colors to data
            if self._last_df is not None and not self._last_df.empty:
                self.update_plot(self._last_df)

            # Apply matplotlib theme if applicable
            if hasattr(self.view, 'theme_combo') and self.view.theme_combo:
                self._apply_theme(self.view.theme_combo.currentText())

            logger.info("Theme colors refreshed")
        except Exception as e:
            logger.error(f"Failed to refresh theme colors: {e}")

    def _toggle_drawbar(self, visible: bool):
        try:
            if hasattr(self.view, "_drawbar") and self.view._drawbar is not None:
                self.view._drawbar.setVisible(bool(visible))
                logger.debug(f"Drawbar visibility set to {visible}")
        except Exception as e:
            logger.debug(f"Failed to toggle drawbar: {e}")

    def _refresh_legend_unique(self, loc: str = "upper left"):
        """Refresh legend showing only visible, unique labels (main + oscillator axes)."""
        try:
            handles_labels: List[Tuple[object, str]] = []
            for axis in (self.ax, getattr(self, '_osc_ax', None)):
                if axis is None:
                    continue
                handles, labels = axis.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if not label or label.startswith('_'):
                        continue
                    visible = getattr(handle, 'get_visible', lambda: True)()
                    if not visible:
                        continue
                    handles_labels.append((handle, label))

            for handle, label in getattr(self, '_extra_legend_items', {}).values():
                handles_labels.append((handle, label))

            unique: List[Tuple[object, str]] = []
            seen = set()
            for handle, label in handles_labels:
                if label in seen:
                    continue
                seen.add(label)
                unique.append((handle, label))

            existing = getattr(self.ax, 'legend_', None)
            if existing is not None:
                existing.remove()

            if not unique:
                return

            legend = self.ax.legend([h for h, _ in unique], [l for _, l in unique], loc=loc, fontsize=8, frameon=False)
            try:
                legend.set_draggable(True)
            except Exception:
                pass
            # Apply unified legend/cursor text color
            try:
                txt_col = self._get_color_mpl("legend_text_color", "#cfd6e1" if getattr(self, "_is_dark", True) else "#1a1e25")
                for txt in legend.get_texts() or []:
                    try:
                        txt.set_color(txt_col)
                    except Exception:
                        pass
                ttl = legend.get_title()
                if ttl:
                    try:
                        ttl.set_color(txt_col)
                    except Exception:
                        pass
            except Exception:
                pass
            self._legend_artist = legend
        except Exception as exc:
            logger.debug('Legend refresh failed: {}', exc)

    def _toggle_orders(self, visible: bool):
        try:
            self._orders_visible = bool(visible)
            if hasattr(self, "orders_table") and self.orders_table is not None:
                self.orders_table.setVisible(self._orders_visible)
        except Exception:
            pass

    def _get_indicator_settings(self) -> dict:
        """Prende la config indicatori dal controller (se c’è) o dalla sessione salvata."""
        try:
            ctrl = self.app_controller or getattr(self._main_window, "controller", None)
            s = getattr(ctrl, "indicators_settings", None)
            if isinstance(s, dict):
                return s
        except Exception:
            pass
        # fallback: user_settings
        try:
            from forex_diffusion.utils.user_settings import get_setting
            sessions = get_setting("indicators.sessions", {}) or {}
            def_name = get_setting("indicators.default_session", "default")
            return sessions.get(def_name, {}) or {}
        except Exception:
            return {}

    # ---- compressed X mapping helpers ----
    def _expand_compressed_x(self, x_c: float) -> float:
        """Map compressed x (weekend removed) back to real matplotlib date float."""
        try:
            segs = getattr(self, "_x_segments_compressed", None)
            if not segs:
                return float(x_c)
            # segs: (start_comp, end_comp, closed_before)
            for s_c, e_c, closed_before in segs:
                if float(s_c) <= float(x_c) <= float(e_c):
                    return float(x_c) + float(closed_before)
            # outside known segments
            if float(x_c) < float(segs[0][0]):
                return float(x_c) + float(segs[0][2])
            return float(x_c) + float(segs[-1][2])
        except Exception:
            return float(x_c)

    def _compress_real_x(self, x_r: float) -> float:
        """Map real matplotlib date float to compressed coordinate removing weekend gaps."""
        try:
            segs = getattr(self, "_x_segments_real", None)
            if not segs:
                return float(x_r)
            # segs: (start_real, end_real, closed_before)
            for s_r, e_r, closed_before in segs:
                if float(s_r) <= float(x_r) <= float(e_r):
                    return float(x_r) - float(closed_before)
            if float(x_r) < float(segs[0][0]):
                return float(x_r) - float(segs[0][2])
            return float(x_r) - float(segs[-1][2])
        except Exception:
            return float(x_r)

    def _compress_weekend_periods(self, x_dt: pd.Series, y_vals: np.ndarray, df: pd.DataFrame = None) -> tuple:
        """
        Compress time axis by removing weekend periods (Friday 22:00 - Sunday 22:00).
        Returns compressed time series, values, weekend boundary markers, and compressed DataFrame.
        """
        try:
            from forex_diffusion.utils.time_utils import is_in_weekend_range, WEEKEND_START_HOUR, WEEKEND_END_HOUR

            # Convert to timezone-aware UTC if not already
            if x_dt.dt.tz is None:
                x_dt_utc = x_dt.dt.tz_localize('UTC')
            else:
                x_dt_utc = x_dt.dt.tz_convert('UTC')

            # Find indices where weekend periods start and end
            weekend_starts = []
            weekend_ends = []
            compressed_indices = []

            prev_was_weekend = False

            for i, dt in enumerate(x_dt_utc):
                is_weekend = is_in_weekend_range(dt.to_pydatetime())

                if not prev_was_weekend and is_weekend:
                    # Entering weekend - mark end of trading week
                    weekend_starts.append(i)
                elif prev_was_weekend and not is_weekend:
                    # Exiting weekend - mark start of new trading week
                    weekend_ends.append(i)

                if not is_weekend:
                    compressed_indices.append(i)

                prev_was_weekend = is_weekend

            # Create compressed arrays
            if compressed_indices:
                x_compressed = x_dt.iloc[compressed_indices].reset_index(drop=True)
                y_compressed = y_vals[compressed_indices]

                # Compress DataFrame if provided
                compressed_df = None
                if df is not None:
                    compressed_df = df.iloc[compressed_indices].reset_index(drop=True)

                # Calculate weekend markers in compressed coordinates
                weekend_markers = []
                compressed_pos = 0

                for orig_idx in compressed_indices:
                    # Check if this is the first point after a weekend
                    if orig_idx in weekend_ends:
                        weekend_markers.append(compressed_pos)
                    compressed_pos += 1

                # Store weekend markers for drawing yellow lines
                self._weekend_markers = weekend_markers
                self._compressed_x_dt = x_compressed

                return x_compressed, y_compressed, weekend_markers, compressed_df
            else:
                return x_dt, y_vals, [], df

        except Exception as e:
            logger.debug(f"Error compressing weekend periods: {e}")
            return x_dt, y_vals, [], df

    def _draw_weekend_markers(self):
        """Draw yellow dashed lines at weekend boundaries (PyQtGraph version)"""
        try:
            if not hasattr(self, '_weekend_markers') or not hasattr(self, '_compressed_x_dt'):
                return

            if not hasattr(self, '_weekend_lines'):
                self._weekend_lines = []

            # Clear existing weekend lines
            for line in self._weekend_lines:
                try:
                    if hasattr(self.ax, 'removeItem'):
                        # PyQtGraph
                        self.ax.removeItem(line)
                    else:
                        # matplotlib fallback
                        line.remove()
                except Exception:
                    pass
            self._weekend_lines = []

            # Draw new weekend markers
            for marker_pos in self._weekend_markers:
                if marker_pos < len(self._compressed_x_dt):
                    x_dt = self._compressed_x_dt.iloc[marker_pos]

                    # Convert datetime to timestamp in seconds for PyQtGraph
                    try:
                        # PyQtGraph uses timestamp in seconds (or milliseconds depending on axis)
                        x_timestamp = x_dt.timestamp()

                        # Try PyQtGraph first (InfiniteLine)
                        try:
                            from PySide6.QtCore import Qt
                            import pyqtgraph as pg

                            line = pg.InfiniteLine(
                                pos=x_timestamp,
                                angle=90,  # Vertical line
                                pen=pg.mkPen(color='gold', style=Qt.DashLine, width=1),
                                movable=False
                            )
                            line.setZValue(5)  # Above candles but below overlays
                            self.ax.addItem(line)
                            self._weekend_lines.append(line)
                            logger.debug(f"Drew weekend marker at timestamp {x_timestamp}")
                        except Exception as e:
                            # Fallback to matplotlib if PyQtGraph fails
                            logger.debug(f"PyQtGraph weekend marker failed, trying matplotlib: {e}")
                            line = self.ax.axvline(x=x_dt, color='gold', linestyle='--',
                                                 linewidth=1.0, alpha=0.7, zorder=5)
                            self._weekend_lines.append(line)

                    except Exception as e:
                        logger.debug(f"Error converting datetime to timestamp: {e}")

        except Exception as e:
            logger.debug(f"Error in _draw_weekend_markers: {e}")

# TODO: Inserire manualmente:
# from .patterns_hook import call_patterns_detection
# call_patterns_detection(self.controller, self.view, df2)
