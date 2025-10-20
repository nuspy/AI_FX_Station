"""
Enhanced Chart Service with Dual Chart System Support
Integrates matplotlib-based charts with advanced finplot subplot system
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger

from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QMessageBox

from forex_diffusion.utils.user_settings import get_setting, set_setting
from .base import ChartServiceBase

# Try to import finplot service
try:
    from .enhanced_finplot_subplot_service import EnhancedFinplotSubplotService
    FINPLOT_SERVICE_AVAILABLE = True
except ImportError:
    FINPLOT_SERVICE_AVAILABLE = False
    EnhancedFinplotSubplotService = None
    logger.warning("Enhanced finplot subplot service not available")

# Try to import indicators system
try:
    from ....features.indicators_btalib import BTALibIndicators
    from ....features.indicator_ranges import indicator_range_classifier
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False
    BTALibIndicators = None
    indicator_range_classifier = None
    logger.warning("Enhanced indicators system not available")


class ChartSystemSelector(QDialog):
    """Dialog to select between chart systems"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chart System Selection")
        self.setModal(True)
        self.resize(500, 300)

        self.selected_system = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Select Chart System")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(title)

        # System options
        self.system_combo = QComboBox()
        systems = ["matplotlib (current)", "finplot (advanced)"]
        if FINPLOT_SERVICE_AVAILABLE:
            systems.append("finplot (enhanced subplots)")
        self.system_combo.addItems(systems)

        # Set default from settings
        current_system = get_setting("chart.system", "matplotlib")
        if current_system == "finplot_enhanced" and FINPLOT_SERVICE_AVAILABLE:
            self.system_combo.setCurrentText("finplot (enhanced subplots)")
        elif current_system == "finplot":
            self.system_combo.setCurrentText("finplot (advanced)")
        else:
            self.system_combo.setCurrentText("matplotlib (current)")

        layout.addWidget(self.system_combo)

        # Info about systems
        info_text = """
<b>matplotlib (current):</b> Traditional charting with single axis and basic indicators<br><br>
<b>finplot (advanced):</b> High-performance financial charting with better performance<br><br>
"""
        if FINPLOT_SERVICE_AVAILABLE:
            info_text += "<b>finplot (enhanced subplots):</b> Professional multi-subplot system with intelligent indicator organization:<br>" \
                        "• Main chart for price overlay indicators (Moving Averages, Bollinger Bands)<br>" \
                        "• Normalized subplot for 0-100 range indicators (RSI, Stochastic)<br>" \
                        "• Volume subplot for volume-based indicators<br>" \
                        "• Custom subplot for other indicators (MACD, CCI)<br>" \
                        "• 10-100x performance improvement over matplotlib"

        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info_label)

        # Buttons
        buttons_layout = QHBoxLayout()
        self.ok_button = QPushButton("Select")
        self.cancel_button = QPushButton("Cancel")

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        buttons_layout.addStretch()
        buttons_layout.addWidget(self.ok_button)
        buttons_layout.addWidget(self.cancel_button)

        layout.addLayout(buttons_layout)

    def accept(self):
        system_text = self.system_combo.currentText()
        if "enhanced subplots" in system_text:
            self.selected_system = "finplot_enhanced"
        elif "finplot" in system_text:
            self.selected_system = "finplot"
        else:
            self.selected_system = "matplotlib"

        # Save selection
        set_setting("chart.system", self.selected_system)
        super().accept()


class EnhancedChartService(ChartServiceBase):
    """
    Enhanced chart service that can use either matplotlib or finplot with intelligent subplot management
    """

    def __init__(self, view, controller):
        super().__init__(view, controller)

        self.chart_system = get_setting("chart.system", "matplotlib")
        self.finplot_service = None
        self.indicators_system = None

        # Initialize indicators system if available
        if INDICATORS_AVAILABLE:
            available_data = ['open', 'high', 'low', 'close', 'volume']
            self.indicators_system = BTALibIndicators(available_data)

        # Initialize finplot service if requested and available
        if self.chart_system == "finplot_enhanced" and FINPLOT_SERVICE_AVAILABLE:
            self._initialize_finplot_service()

    def _initialize_finplot_service(self):
        """Initialize the enhanced finplot service"""
        try:
            available_data = ['open', 'high', 'low', 'close', 'volume']
            self.finplot_service = EnhancedFinplotSubplotService(
                available_data=available_data,
                theme="professional",
                real_time=True
            )
            logger.info("Enhanced finplot service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize finplot service: {e}")
            self.finplot_service = None
            # Fall back to matplotlib
            self.chart_system = "matplotlib"
            set_setting("chart.system", "matplotlib")

    def show_system_selector(self):
        """Show dialog to select chart system"""
        dialog = ChartSystemSelector(self.view)
        if dialog.exec() == QDialog.Accepted:
            new_system = dialog.selected_system
            if new_system != self.chart_system:
                self.switch_chart_system(new_system)

    def switch_chart_system(self, new_system: str):
        """Switch between chart systems"""
        old_system = self.chart_system
        self.chart_system = new_system

        if new_system == "finplot_enhanced" and FINPLOT_SERVICE_AVAILABLE:
            if self.finplot_service is None:
                self._initialize_finplot_service()

            if self.finplot_service:
                logger.info("Switched to enhanced finplot system")
                QMessageBox.information(self.view, "Chart System",
                                      "Switched to enhanced finplot system with multi-subplot support!")
            else:
                logger.error("Failed to initialize finplot service, staying with matplotlib")
                self.chart_system = "matplotlib"
                QMessageBox.warning(self.view, "Chart System",
                                  "Failed to initialize finplot service. Staying with matplotlib.")

        elif new_system == "matplotlib":
            logger.info("Switched to matplotlib system")
            QMessageBox.information(self.view, "Chart System",
                                  "Switched to traditional matplotlib charting system.")

        # Force chart update if we have data
        if hasattr(self, '_last_df') and self._last_df is not None and not self._last_df.empty:
            self.update_plot(self._last_df)

    def update_plot(self, df: pd.DataFrame, quantiles: Optional[dict] = None, restore_xlim=None, restore_ylim=None):
        """Update plot using the selected chart system"""
        if df is None or df.empty:
            return

        if self.chart_system == "finplot_enhanced" and self.finplot_service:
            self._update_plot_finplot_enhanced(df, quantiles, restore_xlim, restore_ylim)
        else:
            # Fall back to matplotlib or handle other systems
            self._update_plot_matplotlib(df, quantiles, restore_xlim, restore_ylim)

    def _update_plot_finplot_enhanced(self, df: pd.DataFrame, quantiles: Optional[dict] = None,
                                    restore_xlim=None, restore_ylim=None):
        """Update plot using enhanced finplot system"""
        try:
            # Create chart layout if not exists
            if not hasattr(self, '_finplot_layout_created'):
                self.finplot_service.create_chart_layout(
                    show_volume=True,
                    show_normalized=True,
                    show_custom=True
                )
                self._finplot_layout_created = True

            # Plot OHLC data
            symbol = getattr(self, 'symbol', 'Unknown')
            self.finplot_service.plot_ohlc_data(df, symbol)

            # Plot enabled indicators
            self._plot_indicators_finplot(df)

            logger.info(f"Updated finplot chart with {len(df)} data points")

        except Exception as e:
            logger.error(f"Error updating finplot chart: {e}")
            # Fall back to matplotlib
            self.chart_system = "matplotlib"
            self._update_plot_matplotlib(df, quantiles, restore_xlim, restore_ylim)

    def _update_plot_matplotlib(self, df: pd.DataFrame, quantiles: Optional[dict] = None,
                              restore_xlim=None, restore_ylim=None):
        """Update plot using matplotlib system (fallback to existing implementation)"""
        try:
            # This would delegate to the existing plot service
            # For now, we'll implement a basic version
            if hasattr(self, 'ax') and self.ax:
                self.ax.clear()

                # Basic candlestick plot with matplotlib
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    # Simple line plot as placeholder
                    dates = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
                    self.ax.plot(dates, df['close'], label='Close', linewidth=1)

                # Plot basic indicators if enabled
                self._plot_indicators_matplotlib(df)

                self.ax.legend()
                self.ax.grid(True, alpha=0.3)

                if hasattr(self, 'canvas'):
                    self.canvas.draw()

                logger.info(f"Updated matplotlib chart with {len(df)} data points")

        except Exception as e:
            logger.error(f"Error updating matplotlib chart: {e}")

    def _plot_indicators_finplot(self, df: pd.DataFrame):
        """Plot indicators using finplot system"""
        if not self.indicators_system or not self.finplot_service:
            return

        try:
            # Get enabled indicators from settings
            enabled_indicators = get_setting("indicators.enabled_list", [])

            for indicator_name in enabled_indicators:
                try:
                    # Calculate indicator
                    indicator_data = self.indicators_system.calculate_indicator(indicator_name, df)

                    if indicator_data is not None and not indicator_data.empty:
                        if isinstance(indicator_data, pd.DataFrame):
                            # Multi-column indicator (like Bollinger Bands)
                            self.finplot_service.add_price_overlay_indicator(
                                indicator_name, indicator_data
                            )
                        else:
                            # Single-column indicator
                            range_info = indicator_range_classifier.get_range_info(indicator_name)
                            if range_info and range_info.range_type.value == "normalized_0_1":
                                # Normalized indicator
                                self.finplot_service.add_normalized_indicator(
                                    indicator_name, indicator_data
                                )
                            else:
                                # Regular indicator
                                self.finplot_service.add_indicator(indicator_name, indicator_data)

                        logger.debug(f"Added indicator {indicator_name} to finplot chart")

                except Exception as e:
                    logger.warning(f"Failed to plot indicator {indicator_name}: {e}")

        except Exception as e:
            logger.error(f"Error plotting indicators in finplot: {e}")

    def _plot_indicators_matplotlib(self, df: pd.DataFrame):
        """Plot indicators using matplotlib system"""
        if not self.indicators_system:
            return

        try:
            # Get enabled indicators from settings
            enabled_indicators = get_setting("indicators.enabled_list", [])

            for indicator_name in enabled_indicators[:5]:  # Limit to 5 for matplotlib
                try:
                    # Calculate indicator
                    indicator_data = self.indicators_system.calculate_indicator(indicator_name, df)

                    if indicator_data is not None and not indicator_data.empty:
                        dates = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index

                        if isinstance(indicator_data, pd.DataFrame):
                            # Multi-column indicator
                            for col in indicator_data.columns:
                                self.ax.plot(dates, indicator_data[col],
                                           label=f"{indicator_name}_{col}", alpha=0.7)
                        else:
                            # Single-column indicator
                            self.ax.plot(dates, indicator_data,
                                       label=indicator_name, alpha=0.7)

                        logger.debug(f"Added indicator {indicator_name} to matplotlib chart")

                except Exception as e:
                    logger.warning(f"Failed to plot indicator {indicator_name}: {e}")

        except Exception as e:
            logger.error(f"Error plotting indicators in matplotlib: {e}")

    def get_subplot_info(self) -> Dict[str, Any]:
        """Get information about current subplot configuration"""
        if self.chart_system == "finplot_enhanced" and self.finplot_service:
            return self.finplot_service.get_subplot_info()
        else:
            return {
                "system": self.chart_system,
                "subplots": {"main": {"active": True, "indicator_count": 0}}
            }

    def clear_indicators(self, subplot: str = None):
        """Clear indicators from chart"""
        if self.chart_system == "finplot_enhanced" and self.finplot_service:
            self.finplot_service.clear_indicators(subplot)
        else:
            # Clear from matplotlib
            if hasattr(self, 'ax') and self.ax:
                self.ax.clear()
                if hasattr(self, 'canvas'):
                    self.canvas.draw()

    def export_chart(self, filepath: str) -> bool:
        """Export chart as image"""
        if self.chart_system == "finplot_enhanced" and self.finplot_service:
            return self.finplot_service.export_chart_image(filepath)
        else:
            # Export matplotlib chart
            try:
                if hasattr(self, 'canvas'):
                    self.canvas.figure.savefig(filepath, dpi=300, bbox_inches='tight')
                    return True
            except Exception as e:
                logger.error(f"Failed to export matplotlib chart: {e}")
            return False

    def close(self):
        """Cleanup chart resources"""
        if self.finplot_service:
            self.finplot_service.close()