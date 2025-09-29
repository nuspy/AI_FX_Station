"""
Patterns Integration Mixin for ChartTab - handles pattern detection and display.
"""
from __future__ import annotations

import time
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from ..chart_components.services.patterns_hook import (
        get_patterns_service, set_patterns_toggle, call_patterns_detection
    )
except ImportError:
    # Fallback for development
    def get_patterns_service(*args, **kwargs):
        return None
    def set_patterns_toggle(*args, **kwargs):
        pass
    def call_patterns_detection(*args, **kwargs):
        pass


class PatternsMixin:
    """Mixin containing all pattern-related functionality for ChartTab."""

    def _wire_pattern_checkboxes(self) -> None:
        """Wire pattern checkbox signals to the patterns service."""
        # Se non hai tutti i checkbox, esci silenziosamente
        checkboxes = [
            getattr(self, "cb_chart_patterns", None),
            getattr(self, "cb_candle_patterns", None),
            getattr(self, "cb_history_patterns", None)
        ]

        if not any(checkboxes):
            logger.debug("Patterns wiring: nessun checkbox trovato in UI (chart/candle/history).")
            return

        ctrl = self.chart_controller
        # Istanzia lazy il service (registry interno, no setattr sul controller)
        get_patterns_service(ctrl, self, create=True)

        # Prova ad agganciare un callback/signal per ricevere i pattern rilevati
        try:
            ps = get_patterns_service(ctrl, self, create=False)
            if ps:
                # Connetti i checkbox ai toggle del service
                if self.cb_chart_patterns:
                    self.cb_chart_patterns.toggled.connect(
                        lambda checked: self._on_pattern_toggle('chart', checked)
                    )
                if self.cb_candle_patterns:
                    self.cb_candle_patterns.toggled.connect(
                        lambda checked: self._on_pattern_toggle('candle', checked)
                    )
                if self.cb_history_patterns:
                    self.cb_history_patterns.toggled.connect(
                        lambda checked: self._on_pattern_toggle('history', checked)
                    )

                # Setup manual scan button if available
                if hasattr(self, 'btn_scan_historical'):
                    self.btn_scan_historical.clicked.connect(self._scan_historical)

                # Setup config button if available
                if hasattr(self, 'btn_config_patterns'):
                    self.btn_config_patterns.clicked.connect(self._open_patterns_config)

        except Exception as e:
            logger.debug(f"Patterns wiring failed: {e}")

    def _on_pattern_toggle(self, pattern_type: str, checked: bool):
        """Handle pattern type toggle."""
        try:
            # Use the patterns service toggle API
            kwargs = {f"{pattern_type}": checked}
            set_patterns_toggle(self.chart_controller, self, **kwargs)

            # Update settings
            from ...utils.user_settings import set_setting
            set_setting(f"patterns.{pattern_type}_enabled", checked)

            logger.debug(f"Pattern {pattern_type} {'enabled' if checked else 'disabled'}")

        except Exception as e:
            logger.error(f"Failed to toggle {pattern_type} patterns: {e}")

    def _scan_historical(self):
        """Trigger historical pattern scanning."""
        try:
            ps = get_patterns_service(self.chart_controller, self, create=False)
            if ps is None:
                return

            symbol = getattr(self, "symbol", None) or (
                self.symbol_combo.currentText() if hasattr(self, "symbol_combo") else None
            )
            if not symbol:
                return

            view_df = getattr(self.chart_controller.plot_service, "_last_df", None)
            if view_df is None or view_df.empty:
                return

            # Scan on all timeframes using current df snapshot
            tfs = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            for tf in tfs:
                try:
                    self._patterns_scan_tf_hint = tf
                    ps.start_historical_scan(view_df)
                except Exception:
                    continue

        except Exception as e:
            logger.exception("Historical scan failed: {}", e)

    def _open_patterns_config(self):
        """Open pattern configuration dialog."""
        # This would open a dialog to configure pattern detection parameters
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Pattern Config", "Pattern configuration dialog not yet implemented.")

    def _clear_pattern_artists(self):
        """Clear all pattern artists from the chart."""
        try:
            # Remove artists from matplotlib
            for artist in getattr(self, '_pattern_artists', []):
                try:
                    artist.remove()
                except Exception:
                    pass

            # Clear the list
            self._pattern_artists = []

            # Clear pattern cache
            self._pattern_cache = {}
            if hasattr(self, '_patterns_cache'):
                self._patterns_cache = []
            if hasattr(self, '_patterns_cache_map'):
                self._patterns_cache_map = {}

            # Redraw canvas
            if hasattr(self, 'canvas'):
                self.canvas.draw_idle()

        except Exception as e:
            logger.debug(f"Failed to clear pattern artists: {e}")

    def _redraw_cached_patterns(self):
        """Redraw patterns from cache after plot update."""
        try:
            # Ask PatternsService to repaint from its internal cache
            ps = get_patterns_service(self.chart_controller, self, create=False)
            if ps and hasattr(ps, '_repaint'):
                ps._repaint()

        except Exception as e:
            logger.debug(f"Failed to redraw cached patterns: {e}")

    def _on_pick_pattern_artist(self, event):
        """Handle clicking on pattern artists for info display."""
        try:
            if not hasattr(event, 'artist') or not event.artist:
                return

            artist = event.artist

            # Check if this is a pattern artist
            pattern_info = getattr(artist, '_pattern_info', None)
            if not pattern_info:
                return

            # Show pattern details
            self._show_pattern_info(pattern_info, event)

        except Exception as e:
            logger.debug(f"Pattern pick event failed: {e}")

    def _show_pattern_info(self, pattern_info: Dict[str, Any], event=None):
        """Show detailed pattern information."""
        try:
            from PySide6.QtWidgets import QMessageBox

            name = pattern_info.get('name', 'Unknown Pattern')
            pattern_type = pattern_info.get('type', 'Unknown')
            direction = pattern_info.get('direction', 'Neutral')
            confidence = pattern_info.get('confidence', 0.0)
            target_price = pattern_info.get('target_price', None)

            info_text = f"""
Pattern: {name}
Type: {pattern_type}
Direction: {direction}
Confidence: {confidence:.2%}
"""

            if target_price:
                info_text += f"Target Price: {target_price:.5f}\n"

            # Add additional details if available
            if 'description' in pattern_info:
                info_text += f"\nDescription:\n{pattern_info['description']}"

            QMessageBox.information(self, f"Pattern Info - {name}", info_text)

        except Exception as e:
            logger.error(f"Failed to show pattern info: {e}")

    def _update_pattern_overlays(self):
        """Update pattern overlays based on current settings."""
        try:
            # Check if any pattern types are enabled
            chart_enabled = getattr(self.cb_chart_patterns, 'isChecked', lambda: False)()
            candle_enabled = getattr(self.cb_candle_patterns, 'isChecked', lambda: False)()
            history_enabled = getattr(self.cb_history_patterns, 'isChecked', lambda: False)()

            if not (chart_enabled or candle_enabled or history_enabled):
                # Clear patterns if none are enabled
                self._clear_pattern_artists()
                return

            # Trigger pattern detection if any are enabled
            ps = get_patterns_service(self.chart_controller, self, create=False)
            if ps:
                df = getattr(self.chart_controller.plot_service, "_last_df", None)
                if df is not None and not df.empty:
                    call_patterns_detection(self.chart_controller, self, df=df)

        except Exception as e:
            logger.debug(f"Failed to update pattern overlays: {e}")

    def _get_pattern_cache_key(self, symbol: str, timeframe: str, data_hash: str) -> str:
        """Generate cache key for pattern data."""
        return f"{symbol}_{timeframe}_{data_hash}"

    def _should_update_patterns(self, symbol: str, timeframe: str) -> bool:
        """Check if patterns should be updated based on cache and timing."""
        try:
            current_time = time.time()
            last_scan = getattr(self, '_last_patterns_scan', 0)

            # Don't scan too frequently (minimum 5 seconds between scans)
            if current_time - last_scan < 5:
                return False

            # Update timestamp
            self._last_patterns_scan = current_time
            return True

        except Exception:
            return True

    def _cache_pattern_results(self, symbol: str, timeframe: str, patterns: List[Dict[str, Any]]):
        """Cache pattern detection results."""
        try:
            if not hasattr(self, '_pattern_cache'):
                self._pattern_cache = {}

            cache_key = f"{symbol}_{timeframe}"
            self._pattern_cache[cache_key] = {
                'patterns': patterns,
                'timestamp': time.time()
            }

            # Limit cache size
            if len(self._pattern_cache) > 50:
                # Remove oldest entries
                sorted_items = sorted(
                    self._pattern_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                for key, _ in sorted_items[:-40]:  # Keep only 40 most recent
                    del self._pattern_cache[key]

        except Exception as e:
            logger.debug(f"Failed to cache pattern results: {e}")

    def _get_cached_patterns(self, symbol: str, timeframe: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached pattern results if available and recent."""
        try:
            if not hasattr(self, '_pattern_cache'):
                return None

            cache_key = f"{symbol}_{timeframe}"
            cached = self._pattern_cache.get(cache_key)

            if not cached:
                return None

            # Check if cache is still valid (within 5 minutes)
            if time.time() - cached['timestamp'] > 300:
                del self._pattern_cache[cache_key]
                return None

            return cached['patterns']

        except Exception as e:
            logger.debug(f"Failed to get cached patterns: {e}")
            return None