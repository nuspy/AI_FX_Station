"""
Controller Proxy Mixin for ChartTab - handles all controller passthrough methods.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import pandas as pd
from PySide6.QtCore import QSignalBlocker
from loguru import logger

from ...utils.user_settings import set_setting


class ControllerProxyMixin:
    """Mixin containing all controller passthrough methods for ChartTab."""

    # --- Data Handling Methods ---
    def _handle_tick(self, payload: dict):
        """Handle incoming tick data."""
        return self.chart_controller.handle_tick(payload=payload)

    def _on_tick_main(self, payload: dict):
        """Main tick handler."""
        return self.chart_controller.on_tick_main(payload=payload)

    def _rt_flush(self):
        """Flush real-time data."""
        self.chart_controller.rt_flush()
        self._follow_center_if_needed()

    def _load_candles_from_db(self, symbol: str, timeframe: str, limit: int = 5000,
                             start_ms: Optional[int] = None, end_ms: Optional[int] = None):
        """Load candles from database."""
        return self.chart_controller.load_candles_from_db(
            symbol=symbol, timeframe=timeframe, limit=limit,
            start_ms=start_ms, end_ms=end_ms
        )

    def _schedule_view_reload(self):
        """Schedule a view reload."""
        return self.chart_controller.schedule_view_reload()

    def _reload_view_window(self):
        """Reload the view window."""
        return self.chart_controller.reload_view_window()

    # --- Symbol and Timeframe Management ---
    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        """Set symbol and timeframe."""
        for combo, text in [(getattr(self, "symbol_combo", None), symbol),
                           (getattr(self, "tf_combo", None), timeframe)]:
            if combo:
                with QSignalBlocker(combo):
                    combo.setCurrentText(text)
        self.symbol, self.timeframe = symbol, timeframe
        set_setting('chart.symbol', symbol)
        set_setting('chart.timeframe', timeframe)
        return self.chart_controller.set_symbol_timeframe(
            db_service=db_service, symbol=symbol, timeframe=timeframe
        )

    def _on_symbol_changed(self, new_symbol: str):
        """Handle symbol change."""
        return self.chart_controller.on_symbol_changed(new_symbol=new_symbol)

    # --- Chart Interaction Methods ---
    def _set_drawing_mode(self, mode: Optional[str]):
        """Set drawing mode."""
        return self.chart_controller.set_drawing_mode(mode=mode)

    def _on_canvas_click(self, event):
        """Handle canvas click."""
        return self.chart_controller.on_canvas_click(event=event)

    def _on_scroll_zoom(self, event):
        """Handle scroll zoom."""
        if event:
            self._suspend_follow()
        return self.chart_controller.on_scroll_zoom(event=event)

    def _on_mouse_press(self, event):
        """Handle mouse press."""
        if event and getattr(event, "button", None) in (1, 3):
            self._suspend_follow()
        return self.chart_controller.on_mouse_press(event=event)

    def _on_mouse_move(self, event):
        """Handle mouse move."""
        if event and (getattr(event, "button", None) in (1, 3) or
                     (ge := getattr(event, "guiEvent", None)) and
                     getattr(ge, "buttons", lambda: 0)()):
            self._suspend_follow()

        # Suppress line update during overlay drag
        if getattr(self, "_overlay_dragging", False) or getattr(self, "_suppress_line_update", False):
            return None
        return self.chart_controller.on_mouse_move(event=event)

    def _on_mouse_release(self, event):
        """Handle mouse release."""
        if event and getattr(event, "button", None) in (1, 3):
            self._suspend_follow()
        return self.chart_controller.on_mouse_release(event=event)

    # --- Plot and Rendering Methods ---
    def update_plot(self, df: pd.DataFrame, quantiles: Optional[dict] = None,
                   restore_xlim=None, restore_ylim=None):
        """Update the main plot."""
        res = self.chart_controller.update_plot(
            df=df, quantiles=quantiles,
            restore_xlim=restore_xlim, restore_ylim=restore_ylim
        )
        # Keep reference for overlays and rebuild X cache
        try:
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                self._last_df = df.copy()
                self._rebuild_x_cache(df)
            self._redraw_cached_patterns()
            # Ask PatternsService to repaint overlays from its cache (no detection)
            try:
                from ..chart_components.services.patterns_hook import get_patterns_service
                ps = get_patterns_service(self.chart_controller, self, create=False)
                if ps:
                    ps._repaint()
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"post-update_plot hooks failed: {e}")
        return res

    def _render_candles(self, df2: pd.DataFrame):
        """Render candles on the chart."""
        return self.chart_controller.render_candles(df2=df2)

    def _on_mode_toggled(self, checked: bool):
        """Handle mode toggle (candles/line)."""
        return self.chart_controller.on_mode_toggled(checked=checked)

    # --- Indicators and Technical Analysis ---
    def _on_indicators_clicked(self):
        """Handle indicators button click."""
        return self.chart_controller.on_indicators_clicked()

    def _get_indicator_settings(self) -> dict:
        """Get indicator settings."""
        return self.chart_controller.get_indicator_settings()

    def _ensure_osc_axis(self, need: bool):
        """Ensure oscillator axis exists."""
        return self.chart_controller.ensure_osc_axis(need=need)

    def _on_main_xlim_changed(self, ax):
        """Handle main axis xlim change."""
        return self.chart_controller.on_main_xlim_changed(ax=ax)

    def _plot_indicators(self, df2: pd.DataFrame, x_dt: pd.Series):
        """Plot indicators on chart."""
        return self.chart_controller.plot_indicators(df2=df2, x_dt=x_dt)

    # --- Technical Indicator Calculations ---
    def _sma(self, x: pd.Series, n: int) -> pd.Series:
        """Simple moving average."""
        return self.chart_controller.sma(x=x, n=n)

    def _ema(self, x: pd.Series, n: int) -> pd.Series:
        """Exponential moving average."""
        return self.chart_controller.ema(x=x, n=n)

    def _bollinger(self, x: pd.Series, n: int, k: float):
        """Bollinger bands."""
        return self.chart_controller.bollinger(x=x, n=n, k=k)

    def _donchian(self, high: pd.Series, low: pd.Series, n: int):
        """Donchian channels."""
        return self.chart_controller.donchian(high=high, low=low, n=n)

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int):
        """Average True Range."""
        return self.chart_controller.atr(high=high, low=low, close=close, n=n)

    def _keltner(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int, k: float):
        """Keltner channels."""
        return self.chart_controller.keltner(high=high, low=low, close=close, n=n, k=k)

    def _rsi(self, x: pd.Series, n: int):
        """Relative Strength Index."""
        return self.chart_controller.rsi(x=x, n=n)

    def _macd(self, x: pd.Series, fast: int, slow: int, signal: int):
        """MACD indicator."""
        return self.chart_controller.macd(x=x, fast=fast, slow=slow, signal=signal)

    def _hurst_roll(self, x: pd.Series, window: int):
        """Rolling Hurst exponent."""
        return self.chart_controller.hurst_roll(x=x, window=window)

    # --- Forecasting Methods ---
    def _on_forecast_clicked(self):
        """Handle forecast button click."""
        return self.chart_controller.on_forecast_clicked()

    def _open_forecast_settings(self):
        """Open forecast settings dialog."""
        return self.chart_controller.open_forecast_settings()

    def _open_adv_forecast_settings(self):
        """Open advanced forecast settings dialog."""
        return self.chart_controller.open_adv_forecast_settings()

    def _on_advanced_forecast_clicked(self):
        """Handle advanced forecast button click."""
        return self.chart_controller.on_advanced_forecast_clicked()

    def on_forecast_ready(self, df: pd.DataFrame, quantiles: dict):
        """Handle forecast ready signal."""
        return self.chart_controller.on_forecast_ready(df=df, quantiles=quantiles)

    def clear_all_forecasts(self):
        """Clear all forecasts."""
        return self.chart_controller.clear_all_forecasts()

    def start_auto_forecast(self):
        """Start automatic forecasting."""
        return self.chart_controller.start_auto_forecast()

    def stop_auto_forecast(self):
        """Stop automatic forecasting."""
        return self.chart_controller.stop_auto_forecast()

    def _auto_forecast_tick(self):
        """Auto forecast tick handler."""
        return self.chart_controller.auto_forecast_tick()

    def _plot_forecast_overlay(self, quantiles: dict, source: str = "basic"):
        """Plot forecast overlay."""
        return self.chart_controller.plot_forecast_overlay(quantiles=quantiles, source=source)

    def _trim_forecasts(self):
        """Trim old forecasts."""
        return self.chart_controller.trim_forecasts()

    # --- Theme and Visual Methods ---
    def _apply_theme(self, theme: str):
        """Apply theme to chart."""
        return self.chart_controller.apply_theme(theme=theme)

    def _get_color(self, key: str, default: str) -> str:
        """Get color from theme."""
        return self.chart_controller.get_color(key=key, default=default)

    def _open_color_settings(self):
        """Open color settings dialog."""
        return self.chart_controller.open_color_settings()

    def _toggle_drawbar(self, visible: bool):
        """Toggle drawbar visibility."""
        return self.chart_controller.toggle_drawbar(visible=visible)

    def _toggle_orders(self, visible: bool):
        """Toggle orders table visibility."""
        return self.chart_controller.toggle_orders(visible=visible)

    def _update_badge_visibility(self, event):
        """Update badge visibility."""
        return self.chart_controller.update_badge_visibility(event=event)

    # --- Data and Backfill Methods ---
    def _on_backfill_missing_clicked(self):
        """Handle backfill missing data button click."""
        return self.chart_controller.on_backfill_missing_clicked()

    def _on_build_latents_clicked(self):
        """Handle build latents button click."""
        return self.chart_controller.on_build_latents_clicked()

    def _refresh_orders(self):
        """Refresh orders table."""
        return self.chart_controller.refresh_orders()

    # --- Trading Methods ---
    def _open_trade_dialog(self):
        """Open trade dialog."""
        return self.chart_controller.open_trade_dialog()

    # --- Utility Methods ---
    def _tf_to_timedelta(self, tf: str):
        """Convert timeframe to timedelta."""
        return self.chart_controller.tf_to_timedelta(tf=tf)

    def _zoom_axis(self, axis: str, center: float, factor: float):
        """Zoom axis around center point."""
        return self.chart_controller.zoom_axis(axis=axis, center=center, factor=factor)

    def _resolution_for_span(self, ms_span: int) -> str:
        """Get appropriate resolution for time span."""
        return self.chart_controller.resolution_for_span(ms_span=ms_span)

    # --- Helper Methods ---
    def _rebuild_x_cache(self, df: pd.DataFrame):
        """Rebuild X coordinate cache for overlays."""
        # This method would be implemented in the overlay manager
        pass

    def _redraw_cached_patterns(self):
        """Redraw cached patterns."""
        # This method would be implemented in the patterns mixin
        pass