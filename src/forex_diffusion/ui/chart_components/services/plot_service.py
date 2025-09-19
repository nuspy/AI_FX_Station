from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import time

import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd
from loguru import logger
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication, QMessageBox

from forex_diffusion.utils.user_settings import get_setting, set_setting

from .base import ChartServiceBase


class PlotService(ChartServiceBase):
    """Auto-generated service extracted from ChartTab."""

    def update_plot(self, df: pd.DataFrame, quantiles: Optional[dict] = None, restore_xlim=None, restore_ylim=None):
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
        except Exception as e:
            logger.exception('Failed to normalize data for plotting: {}', e)
            return

        if len(x_dt) == 0 or len(y_vals) == 0:
            logger.info('Nothing to plot after cleaning ({} {}).', getattr(self, 'symbol', ''), getattr(self, 'timeframe', ''))
            self._clear_candles()
            return

        price_mode = str(getattr(self, '_price_mode', 'line')).lower()
        price_color = self._get_color('price_color', '#e0e0e0' if getattr(self, '_is_dark', True) else '#000000')
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
            if self._price_line is not None:
                try:
                    self._price_line.set_visible(True)
                except Exception:
                    pass
            self._clear_candles()

        self._plot_indicators(df2, x_dt)

        if quantiles:
            self._plot_forecast_overlay(quantiles)

        self.ax.set_title(f"{getattr(self, 'symbol', '')} - {getattr(self, 'timeframe', '')}", pad=2)
        axes_col = self._get_color('axes_color', '#cfd6e1')
        try:
            self.ax.tick_params(colors=axes_col)
            self.ax.xaxis.label.set_color(axes_col)
            self.ax.yaxis.label.set_color(axes_col)
            for spine in self.ax.spines.values():
                spine.set_color(axes_col)
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

        try:
            self.canvas.draw_idle()
        except Exception:
            self.canvas.draw()

    def _render_candles(self, df2: pd.DataFrame, x_dt: Optional[pd.Series] = None):
        """Render OHLC candles on the main axis."""
        try:
            required = {'open', 'high', 'low', 'close'}
            if not required.issubset(df2.columns):
                self._clear_candles()
                return

            self._clear_candles()

            if x_dt is None:
                ts = pd.to_numeric(df2['ts_utc'], errors='coerce').dropna()
                if ts.empty:
                    self._clear_candles()
                    return
                x_dt = pd.to_datetime(ts.astype('int64'), unit='ms', utc=True).tz_convert(None)

            xs = mdates.date2num(pd.to_datetime(x_dt))
            diffs = xs[1:] - xs[:-1] if len(xs) > 1 else []
            positive = [d for d in diffs if d > 0]
            width = (float(np.median(positive)) * 0.7) if positive else self._default_candle_width()
            if width <= 0:
                width = self._default_candle_width()
            price_span = float(df2['high'].max() - df2['low'].min() or 1.0)
            epsilon = max(price_span * 0.0005, 1e-6)

            bull_color = self._get_color('candle_up_color', '#2ecc71')
            bear_color = self._get_color('candle_down_color', '#e74c3c')

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
        if need and self._osc_ax is None:
            try:
                self._osc_ax = inset_axes(self.ax, width="100%", height="28%",
                                          loc="lower left", borderpad=1.2)
                self._osc_ax.set_facecolor("#111418" if getattr(self, "_is_dark", True) else "#f5f7fb")
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
            axes_col = self._get_color("axes_color", "#cfd6e1")
            try:
                axo.tick_params(colors=axes_col, labelsize=8)
                for spine in axo.spines.values():
                    spine.set_color(axes_col)
            except Exception:
                pass

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

    def _on_mode_toggled(self, checked: bool):
        """Toggle between candle and line rendering."""
        try:
            self._price_mode = 'candles' if checked else 'line'
            if hasattr(self, 'mode_btn') and self.mode_btn is not None:
                self.mode_btn.setText('Linea' if checked else 'Candles')
            if self._last_df is not None and not self._last_df.empty:
                prev_xlim = self.ax.get_xlim() if hasattr(self.ax, 'get_xlim') else None
                prev_ylim = self.ax.get_ylim() if hasattr(self.ax, 'get_ylim') else None
                self.update_plot(self._last_df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)
        except Exception as exc:
            logger.debug('Price mode toggle failed: {}', exc)

    def _apply_theme(self, theme: str):
        from PySide6.QtGui import QPalette, QColor
        from PySide6.QtWidgets import QApplication
        t = (theme or "Dark").lower()
        self._is_dark = (t == "dark")
        app = QApplication.instance()
        # Colori da settings (con fallback per dark/light)
        window_bg = self._get_color("window_bg", "#0f1115" if self._is_dark else "#f3f5f8")
        panel_bg = self._get_color("panel_bg", "#12151b" if self._is_dark else "#ffffff")
        text_color = self._get_color("text_color", "#e0e0e0" if self._is_dark else "#1a1e25")
        chart_bg = self._get_color("chart_bg", "#0f1115" if self._is_dark else "#ffffff")
        base_css = f"""
        QWidget {{ background-color: {window_bg}; color: {text_color}; }}
        QPushButton, QComboBox, QToolButton {{ background-color: {('#1c1f26' if self._is_dark else '#ffffff')}; color: {text_color}; border: 1px solid {('#2a2f3a' if self._is_dark else '#cfd6e1')}; padding: 4px 8px; border-radius: 4px; }}
        QPushButton:hover, QToolButton:hover {{ background-color: {('#242a35' if self._is_dark else '#eaeef4')}; }}
        QTableWidget, QListWidget {{ background-color: {panel_bg}; color: {text_color}; gridline-color: {('#2a2f3a' if self._is_dark else '#cfd6e1')}; }}
        QHeaderView::section {{ background-color: {('#1a1e25' if self._is_dark else '#e8edf4')}; color: {text_color}; border: 0px; }}
        """
        custom_qss = get_setting("custom_qss", "")
        if app:
            app.setStyleSheet(base_css + "\n" + (custom_qss or ""))
            pal = QPalette()
            pal.setColor(QPalette.Window, QColor(window_bg))
            pal.setColor(QPalette.WindowText, QColor(text_color))
            pal.setColor(QPalette.Base, QColor(panel_bg))
            pal.setColor(QPalette.AlternateBase, QColor(panel_bg))
            pal.setColor(QPalette.Text, QColor(text_color))
            pal.setColor(QPalette.Button, QColor(panel_bg))
            pal.setColor(QPalette.ButtonText, QColor(text_color))
            app.setPalette(pal)
        # colori figure
        try:
            self.canvas.figure.set_facecolor(chart_bg)
            self.ax.set_facecolor(chart_bg)
        except Exception:
            pass
        set_setting("ui_theme", "Dark" if self._is_dark else "Light")
        try: self.canvas.draw()
        except Exception: pass

    def _get_color(self, key: str, default: str) -> str:
        try:
            return str(get_setting(key, default))
        except Exception:
            return default

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
