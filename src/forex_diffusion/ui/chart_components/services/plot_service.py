from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Using finplot exclusively - matplotlib removed
import numpy as np
import pandas as pd
from loguru import logger
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication, QMessageBox
from forex_diffusion.utils.time_utils import split_range_avoid_weekend
from .patterns_hook import call_patterns_detection
from forex_diffusion.utils.user_settings import get_setting, set_setting
from .base import ChartServiceBase

# Import finplot
import finplot as fplt

# Enhanced indicators support
try:
    from forex_diffusion.features.indicators_btalib import BTALibIndicators
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

    def enable_indicator_subplots(self):
        """Enable multi-subplot mode for indicators"""
        try:
            from .matplotlib_subplot_service import MatplotlibSubplotService
            if hasattr(self.view, 'canvas') and hasattr(self.view.canvas, 'figure'):
                self._subplot_service = MatplotlibSubplotService(self.view.canvas.figure)
                self._subplot_enabled = True
                logger.info("Indicator subplots enabled")
        except Exception as e:
            logger.error(f"Failed to enable indicator subplots: {e}")
            self._subplot_enabled = False

    def disable_indicator_subplots(self):
        """Disable multi-subplot mode"""
        self._subplot_service = None
        self._subplot_enabled = False

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
        """Update plot using finplot (high-performance financial charting)"""
        try:
            import finplot as fplt

            if df is None or df.empty:
                return

            self._last_df = df.copy()

            # Prepare data
            df2 = df.copy()
            y_col = 'close' if 'close' in df2.columns else 'price'
            df2['ts_utc'] = pd.to_numeric(df2['ts_utc'], errors='coerce')
            df2[y_col] = pd.to_numeric(df2[y_col], errors='coerce')
            df2 = df2.dropna(subset=['ts_utc', y_col]).reset_index(drop=True)

            # Convert timestamp to datetime
            df2['time'] = pd.to_datetime(df2['ts_utc'], unit='ms', utc=True)
            df2 = df2.set_index('time')

            # Clear previous plots
            fplt.close()

            # Get enabled indicators
            enabled_indicator_names = get_setting("indicators.enabled_list", [])
            indicator_colors = get_setting("indicators.colors", {})

            # Check what subplots we need
            has_normalized = False
            if ENHANCED_INDICATORS_AVAILABLE and enabled_indicator_names:
                has_normalized = any(
                    indicator_range_classifier.get_range_info(name).subplot_recommendation == 'normalized_subplot'
                    for name in enabled_indicator_names
                    if indicator_range_classifier.get_range_info(name)
                )

            # Create subplots
            if has_normalized:
                # Main chart + normalized subplot
                axs = fplt.create_plot(rows=2, init_zoom_periods=100)
                ax_price = axs[0]
                ax_normalized = axs[1]
            else:
                # Only main chart
                axs = fplt.create_plot(rows=1, init_zoom_periods=100)
                ax_price = axs[0] if isinstance(axs, list) else axs
                ax_normalized = None

            # Plot candlesticks if OHLC data available
            if {'open', 'high', 'low', 'close'}.issubset(df2.columns):
                fplt.candlestick_ochl(df2[['open', 'close', 'high', 'low']], ax=ax_price)
            else:
                fplt.plot(df2.index, df2[y_col], ax=ax_price, legend='Price')

            # Plot indicators
            if ENHANCED_INDICATORS_AVAILABLE and enabled_indicator_names:
                indicators_system = BTALibIndicators()

                # Prepare OHLCV DataFrame
                indicator_df = pd.DataFrame({
                    'open': df2.get('open', df2[y_col]),
                    'high': df2.get('high', df2[y_col]),
                    'low': df2.get('low', df2[y_col]),
                    'close': df2['close'] if 'close' in df2.columns else df2[y_col],
                    'volume': df2.get('volume', pd.Series([1.0] * len(df2), index=df2.index))
                })

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
                        target_ax = ax_normalized if (subplot_type == 'normalized_subplot' and ax_normalized) else ax_price

                        if isinstance(result, pd.Series):
                            # Single series
                            fplt.plot(result.index, result.values, ax=target_ax,
                                     legend=indicator_name, color=color)
                        elif isinstance(result, dict):
                            # Multi-series (MACD, Bollinger Bands, etc.)
                            for key, series in result.items():
                                if isinstance(series, pd.Series):
                                    fplt.plot(series.index, series.values, ax=target_ax,
                                             legend=key, color=color)

                    except Exception as e:
                        logger.error(f"Failed to plot indicator {indicator_name} with finplot: {e}")

            # Show the plot
            fplt.show()

            # Store axis reference
            self.ax = ax_price
            self.view.finplot_axes = [ax_price]
            if ax_normalized:
                self.view.finplot_axes.append(ax_normalized)

        except Exception as e:
            logger.error(f"Error in _update_plot_finplot: {e}")
            import traceback
            traceback.print_exc()

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
            dlg = ColorSettingsDialog(self)
            if dlg.exec():
                # re-draw to apply new colors
                if self._last_df is not None and not self._last_df.empty:
                    self.update_plot(self._last_df)
                # ri-applica il tema per aggiornare QSS/palette
                self._apply_theme(self.theme_combo.currentText())
        except Exception as e:
            QMessageBox.warning(self.view, "Colors", str(e))

    def _toggle_drawbar(self, visible: bool):
        try:
            if hasattr(self, "_drawbar") and self._drawbar is not None:
                self._drawbar.setVisible(bool(visible))
        except Exception:
            pass

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
        """Draw yellow dashed lines at weekend boundaries"""
        try:
            if not hasattr(self, '_weekend_markers') or not hasattr(self, '_compressed_x_dt'):
                return

            if not hasattr(self, '_weekend_lines'):
                self._weekend_lines = []

            # Clear existing weekend lines
            for line in self._weekend_lines:
                try:
                    line.remove()
                except Exception:
                    pass
            self._weekend_lines = []

            # Draw new weekend markers
            for marker_pos in self._weekend_markers:
                if marker_pos < len(self._compressed_x_dt):
                    x_pos = self._compressed_x_dt.iloc[marker_pos]
                    try:
                        line = self.ax.axvline(x=x_pos, color='gold', linestyle='--',
                                             linewidth=1.0, alpha=0.7, zorder=5)
                        self._weekend_lines.append(line)
                    except Exception as e:
                        logger.debug(f"Error drawing weekend marker: {e}")

        except Exception as e:
            logger.debug(f"Error in _draw_weekend_markers: {e}")

# TODO: Inserire manualmente:
# from .patterns_hook import call_patterns_detection
# call_patterns_detection(self.controller, self.view, df2)
