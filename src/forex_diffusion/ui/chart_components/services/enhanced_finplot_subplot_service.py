"""
Enhanced FinplotChartService with Multiple Subplot Support
Implements intelligent subplot organization for different indicator ranges
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass

# Import finplot
try:
    import finplot as fplt
    FINPLOT_AVAILABLE = True
except ImportError:
    FINPLOT_AVAILABLE = False
    fplt = None

# Import our indicators system and range classifier
try:
    from ....features.indicators_btalib import BTALibIndicators, IndicatorCategories
    from ....features.indicator_ranges import IndicatorRangeClassifier, IndicatorRange, indicator_range_classifier
    BTALIB_AVAILABLE = True
except ImportError:
    BTALIB_AVAILABLE = False
    BTALibIndicators = None
    indicator_range_classifier = None

logger = logging.getLogger(__name__)


@dataclass
class SubplotConfig:
    """Configuration for a subplot"""
    name: str
    title: str
    height_ratio: float
    y_range: Optional[Tuple[float, float]] = None
    y_label: str = ""
    show_x_axis: bool = False
    indicator_types: List[str] = None


class EnhancedFinplotSubplotService:
    """
    Enhanced finplot chart service with intelligent subplot management
    Automatically organizes indicators into appropriate subplots based on value ranges
    """

    def __init__(self,
                 available_data: List[str] = None,
                 theme: str = "professional",
                 real_time: bool = True):
        """
        Initialize enhanced finplot chart service with subplot support

        Args:
            available_data: List of available data columns
            theme: Chart theme ('professional', 'dark', 'light')
            real_time: Enable real-time streaming capabilities
        """
        if not FINPLOT_AVAILABLE:
            raise ImportError("finplot library is required but not available")

        self.available_data = available_data or ['open', 'high', 'low', 'close']
        self.theme = theme
        self.real_time = real_time

        # Initialize indicators system
        if BTALIB_AVAILABLE:
            self.indicators_system = BTALibIndicators(available_data)
            self.range_classifier = indicator_range_classifier
        else:
            logger.warning("bta-lib not available, limited indicator support")
            self.indicators_system = None
            self.range_classifier = None

        # Chart state
        self.current_data = None
        self.current_indicators = {}
        self.subplots = {}
        self.main_chart_ax = None

        # Subplot configuration
        self.subplot_configs = self._initialize_subplot_configs()

        # Initialize finplot
        self._setup_finplot()

    def _initialize_subplot_configs(self) -> Dict[str, SubplotConfig]:
        """Initialize subplot configurations"""
        return {
            'main_chart': SubplotConfig(
                name='main_chart',
                title='Price Chart',
                height_ratio=0.6,
                y_label='Price',
                show_x_axis=False,
                indicator_types=['price_overlay']
            ),
            'normalized_subplot': SubplotConfig(
                name='normalized_subplot',
                title='Normalized Indicators (0-100)',
                height_ratio=0.15,
                y_range=(0, 100),
                y_label='%',
                show_x_axis=False,
                indicator_types=['normalized_0_1', 'normalized_-1_+1']
            ),
            'volume_subplot': SubplotConfig(
                name='volume_subplot',
                title='Volume',
                height_ratio=0.1,
                y_label='Volume',
                show_x_axis=False,
                indicator_types=['volume_based']
            ),
            'custom_subplot': SubplotConfig(
                name='custom_subplot',
                title='Custom Range Indicators',
                height_ratio=0.15,
                y_label='Value',
                show_x_axis=True,
                indicator_types=['custom_range', 'unbounded']
            )
        }

    def _setup_finplot(self):
        """Setup finplot with professional theme"""
        if not FINPLOT_AVAILABLE:
            return

        # Set theme
        if self.theme == "professional":
            fplt.foreground = '#ffffff'
            fplt.background = '#0a0a0a'
            fplt.candle_bull_color = '#26a69a'
            fplt.candle_bear_color = '#ef5350'
            fplt.volume_bull_color = '#26a69a'
            fplt.volume_bear_color = '#ef5350'
        elif self.theme == "dark":
            fplt.foreground = '#cccccc'
            fplt.background = '#1e1e1e'
        elif self.theme == "light":
            fplt.foreground = '#000000'
            fplt.background = '#ffffff'

        # Performance optimizations
        fplt.max_zoom_points = 10000  # Limit for performance
        fplt.autoviewrestore()

    def create_chart_layout(self, show_volume: bool = True,
                          show_normalized: bool = True,
                          show_custom: bool = True) -> Dict[str, Any]:
        """
        Create multi-subplot chart layout

        Args:
            show_volume: Show volume subplot
            show_normalized: Show normalized indicators subplot
            show_custom: Show custom indicators subplot

        Returns:
            Dictionary mapping subplot names to axes
        """
        if not FINPLOT_AVAILABLE:
            logger.error("finplot not available")
            return {}

        # Clear previous charts
        fplt.close()

        # Calculate subplot configuration
        active_subplots = ['main_chart']
        if show_volume:
            active_subplots.append('volume_subplot')
        if show_normalized:
            active_subplots.append('normalized_subplot')
        if show_custom:
            active_subplots.append('custom_subplot')

        # Create subplots with proper height ratios
        subplot_count = len(active_subplots)
        height_ratios = []

        for subplot_name in active_subplots:
            config = self.subplot_configs[subplot_name]
            height_ratios.append(config.height_ratio)

        # Normalize height ratios
        total_ratio = sum(height_ratios)
        normalized_ratios = [r / total_ratio for r in height_ratios]

        # Create axes
        axes = {}
        for i, subplot_name in enumerate(active_subplots):
            config = self.subplot_configs[subplot_name]

            if i == 0:
                # Main chart
                ax = fplt.create_plot(title=config.title, maximize=True)
                self.main_chart_ax = ax
            else:
                # Secondary subplots
                ax = fplt.create_plot(title=config.title, rows=subplot_count, init_zoom_periods=100)

            # Configure axis
            if config.y_range:
                ax.setYRange(*config.y_range, padding=0.02)

            if not config.show_x_axis and i < len(active_subplots) - 1:
                ax.axes['bottom']['item'].hide()

            axes[subplot_name] = ax
            self.subplots[subplot_name] = ax

        logger.info(f"Created chart layout with {len(axes)} subplots: {list(axes.keys())}")
        return axes

    def plot_ohlc_data(self, data: pd.DataFrame, symbol: str = ""):
        """Plot OHLC candlestick data on main chart"""
        if not FINPLOT_AVAILABLE or self.main_chart_ax is None:
            return

        # Ensure data has required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Data missing required OHLC columns: {required_cols}")
            return

        # Plot candlesticks
        fplt.candlestick_ochl(data[['open', 'close', 'high', 'low']], ax=self.main_chart_ax)

        # Plot volume if available
        if 'volume' in data.columns and 'volume_subplot' in self.subplots:
            volume_ax = self.subplots['volume_subplot']
            fplt.volume_ocv(data[['open', 'close', 'volume']], ax=volume_ax)

        self.current_data = data
        logger.info(f"Plotted OHLC data for {symbol}: {len(data)} bars")

    def add_indicator(self, indicator_name: str, indicator_data: pd.Series,
                     color: str = None, style: str = '-', width: int = 1):
        """
        Add indicator to appropriate subplot based on range classification

        Args:
            indicator_name: Name of the indicator
            indicator_data: Pandas Series with indicator values
            color: Plot color (auto-assigned if None)
            style: Line style ('-', '--', ':', '-.')
            width: Line width
        """
        if not FINPLOT_AVAILABLE:
            logger.error("finplot not available")
            return

        # Get subplot recommendation
        if self.range_classifier:
            subplot_rec = self.range_classifier.get_subplot_recommendation(indicator_name)
        else:
            subplot_rec = "custom_subplot"  # Fallback

        # Get target subplot
        target_ax = self.subplots.get(subplot_rec, self.main_chart_ax)
        if target_ax is None:
            logger.warning(f"Subplot {subplot_rec} not available, using main chart")
            target_ax = self.main_chart_ax

        # Auto-assign color if not provided
        if color is None:
            color = self._get_next_color(subplot_rec)

        # Plot indicator
        try:
            fplt.plot(indicator_data, ax=target_ax, color=color, style=style,
                     width=width, legend=indicator_name)

            self.current_indicators[indicator_name] = {
                'data': indicator_data,
                'subplot': subplot_rec,
                'color': color,
                'style': style,
                'width': width
            }

            logger.info(f"Added indicator {indicator_name} to {subplot_rec}")

        except Exception as e:
            logger.error(f"Failed to plot indicator {indicator_name}: {e}")

    def add_price_overlay_indicator(self, indicator_name: str, indicator_data: pd.DataFrame,
                                  colors: Dict[str, str] = None):
        """
        Add price overlay indicator (like Bollinger Bands) to main chart

        Args:
            indicator_name: Name of the indicator
            indicator_data: DataFrame with indicator columns
            colors: Dictionary mapping column names to colors
        """
        if not FINPLOT_AVAILABLE or self.main_chart_ax is None:
            return

        colors = colors or {}

        try:
            for col in indicator_data.columns:
                col_name = f"{indicator_name}_{col}"
                color = colors.get(col, self._get_next_color('main_chart'))

                fplt.plot(indicator_data[col], ax=self.main_chart_ax, color=color,
                         legend=col_name, width=1)

                self.current_indicators[col_name] = {
                    'data': indicator_data[col],
                    'subplot': 'main_chart',
                    'color': color
                }

            logger.info(f"Added price overlay indicator {indicator_name} with {len(indicator_data.columns)} bands")

        except Exception as e:
            logger.error(f"Failed to plot price overlay {indicator_name}: {e}")

    def add_normalized_indicator(self, indicator_name: str, indicator_data: pd.Series,
                               overbought: float = 70, oversold: float = 30):
        """
        Add normalized indicator with overbought/oversold levels

        Args:
            indicator_name: Name of the indicator
            indicator_data: Pandas Series with indicator values
            overbought: Overbought level
            oversold: Oversold level
        """
        if 'normalized_subplot' not in self.subplots:
            logger.warning("Normalized subplot not available")
            return

        ax = self.subplots['normalized_subplot']
        color = self._get_next_color('normalized_subplot')

        try:
            # Plot indicator
            fplt.plot(indicator_data, ax=ax, color=color, legend=indicator_name)

            # Add overbought/oversold lines
            if overbought and overbought <= 100:
                fplt.add_line((indicator_data.index[0], overbought),
                             (indicator_data.index[-1], overbought),
                             color='red', style='--', width=1, ax=ax)

            if oversold and oversold >= 0:
                fplt.add_line((indicator_data.index[0], oversold),
                             (indicator_data.index[-1], oversold),
                             color='green', style='--', width=1, ax=ax)

            self.current_indicators[indicator_name] = {
                'data': indicator_data,
                'subplot': 'normalized_subplot',
                'color': color,
                'overbought': overbought,
                'oversold': oversold
            }

            logger.info(f"Added normalized indicator {indicator_name}")

        except Exception as e:
            logger.error(f"Failed to plot normalized indicator {indicator_name}: {e}")

    def clear_indicators(self, subplot: str = None):
        """Clear indicators from specific subplot or all subplots"""
        if subplot:
            # Clear specific subplot
            indicators_to_remove = [name for name, info in self.current_indicators.items()
                                  if info['subplot'] == subplot]
            for name in indicators_to_remove:
                del self.current_indicators[name]
        else:
            # Clear all indicators
            self.current_indicators.clear()

        # Refresh chart
        self.refresh_chart()

    def refresh_chart(self):
        """Refresh the chart with current data and indicators"""
        if not FINPLOT_AVAILABLE or self.current_data is None:
            return

        # Clear and recreate
        for ax in self.subplots.values():
            ax.clear()

        # Re-plot OHLC data
        self.plot_ohlc_data(self.current_data)

        # Re-plot indicators
        for name, info in self.current_indicators.items():
            self.add_indicator(name, info['data'], info['color'],
                             info.get('style', '-'), info.get('width', 1))

    def _get_next_color(self, subplot: str) -> str:
        """Get next available color for subplot"""
        color_schemes = {
            'main_chart': ['#2E86C1', '#F39C12', '#27AE60', '#8E44AD', '#E74C3C'],
            'normalized_subplot': ['#3498DB', '#E67E22', '#2ECC71', '#9B59B6', '#E74C3C'],
            'volume_subplot': ['#34495E', '#F39C12'],
            'custom_subplot': ['#16A085', '#D35400', '#C0392B', '#8E44AD', '#F39C12']
        }

        colors = color_schemes.get(subplot, color_schemes['main_chart'])
        current_count = len([i for i in self.current_indicators.values()
                           if i['subplot'] == subplot])

        return colors[current_count % len(colors)]

    def get_subplot_info(self) -> Dict[str, Dict]:
        """Get information about current subplots"""
        return {
            name: {
                'config': config.__dict__,
                'indicator_count': len([i for i in self.current_indicators.values()
                                      if i['subplot'] == name]),
                'active': name in self.subplots
            }
            for name, config in self.subplot_configs.items()
        }

    def export_chart_image(self, filepath: str, width: int = 1920, height: int = 1080):
        """Export chart as image"""
        if not FINPLOT_AVAILABLE:
            logger.error("finplot not available")
            return False

        try:
            fplt.screenshot(open(filepath, 'wb'))
            logger.info(f"Chart exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export chart: {e}")
            return False

    def close(self):
        """Close chart and cleanup"""
        if FINPLOT_AVAILABLE:
            fplt.close()
        self.subplots.clear()
        self.current_indicators.clear()
        self.current_data = None