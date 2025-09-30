# src/forex_diffusion/ui/chart_components/services/matplotlib_subplot_service.py
"""
Matplotlib Subplot Service for Multiple Indicator Ranges
Manages two separate subplots: one for normalized (0-1) indicators, one for price-range indicators
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.dates as mdates
from loguru import logger


class MatplotlibSubplotService:
    """
    Service to manage multiple subplots for different indicator value ranges.

    Subplots:
    1. Main price chart (largest)
    2. Normalized indicators (0-1 range): RSI, Stoch, etc.
    3. Price-range indicators: Moving averages, Bollinger Bands, etc.
    """

    def __init__(self, figure: Figure):
        """
        Initialize subplot service

        Args:
            figure: Matplotlib Figure object to create subplots in
        """
        self.figure = figure
        self.axes: Dict[str, Axes] = {}

        # Indicator classification
        self.normalized_indicators = {
            'rsi', 'stoch', 'stochf', 'stochrsi', 'willr', 'cci', 'mfi',
            'adosc', 'bop', 'cmo', 'dx', 'aroon', 'aroonosc', 'mom',
            'roc', 'rocp', 'rocr', 'trix', 'ultosc', 'adx', 'adxr',
        }

        self.price_range_indicators = {
            'sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama', 't3',
            'ma', 'mama', 'fama', 'bbands', 'midpoint', 'midprice',
            'sar', 'sarext', 'atr', 'natr', 'trange', 'avgprice',
            'medprice', 'typprice', 'wclprice', 'ht_trendline',
            'keltner', 'donchian', 'vwap',
        }

        # Track active indicators
        self.active_indicators: Dict[str, List[str]] = {
            'price': [],       # Overlaid on price chart
            'normalized': [],  # 0-1 subplot
            'volume': []       # Volume subplot (if needed)
        }

    def create_subplots(self, has_normalized: bool = False, has_volume: bool = False) -> Dict[str, Axes]:
        """
        Create subplot layout based on what indicators are active

        Args:
            has_normalized: Whether normalized indicators are active
            has_volume: Whether volume indicator is active

        Returns:
            Dictionary of axes by name
        """
        self.figure.clear()
        self.axes = {}

        # Calculate grid layout
        n_subplots = 1  # Always have price chart
        if has_normalized:
            n_subplots += 1
        if has_volume:
            n_subplots += 1

        # Height ratios: price gets most space
        if n_subplots == 1:
            height_ratios = [1]
        elif n_subplots == 2:
            height_ratios = [3, 1]  # price:indicators = 3:1
        else:
            height_ratios = [3, 1, 1]  # price:normalized:volume = 3:1:1

        # Create subplots
        axes_list = self.figure.subplots(
            n_subplots, 1,
            gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.05},
            sharex=True
        )

        # Handle single vs multiple axes
        if n_subplots == 1:
            axes_list = [axes_list]

        # Assign axes
        idx = 0
        self.axes['price'] = axes_list[idx]
        self.axes['price'].set_ylabel('Price', fontsize=9)
        self.axes['price'].grid(True, alpha=0.3)
        idx += 1

        if has_normalized:
            self.axes['normalized'] = axes_list[idx]
            self.axes['normalized'].set_ylabel('Normalized (0-1)', fontsize=9)
            self.axes['normalized'].set_ylim(0, 100)  # 0-100 for percentage-based
            self.axes['normalized'].grid(True, alpha=0.3)
            self.axes['normalized'].axhline(y=30, color='g', linestyle='--', alpha=0.3, linewidth=0.8)
            self.axes['normalized'].axhline(y=70, color='r', linestyle='--', alpha=0.3, linewidth=0.8)
            self.axes['normalized'].axhline(y=50, color='gray', linestyle='-', alpha=0.2, linewidth=0.5)
            idx += 1

        if has_volume:
            self.axes['volume'] = axes_list[idx]
            self.axes['volume'].set_ylabel('Volume', fontsize=9)
            self.axes['volume'].grid(True, alpha=0.3)
            idx += 1

        # Only show x-axis labels on bottom subplot
        for ax in axes_list[:-1]:
            ax.tick_params(axis='x', labelbottom=False)

        axes_list[-1].set_xlabel('Date', fontsize=9)

        # Format x-axis for dates
        for ax in axes_list:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        return self.axes

    def classify_indicator(self, indicator_name: str) -> str:
        """
        Classify indicator into subplot category

        Args:
            indicator_name: Name of the indicator

        Returns:
            Subplot name: 'price', 'normalized', or 'volume'
        """
        indicator_lower = indicator_name.lower()

        if 'volume' in indicator_lower or 'obv' in indicator_lower:
            return 'volume'
        elif any(norm in indicator_lower for norm in self.normalized_indicators):
            return 'normalized'
        elif any(price in indicator_lower for price in self.price_range_indicators):
            return 'price'
        else:
            # Default: if looks like percentage, put in normalized
            return 'normalized'

    def plot_indicator(
        self,
        indicator_name: str,
        data: pd.Series,
        subplot: Optional[str] = None,
        **plot_kwargs
    ) -> Optional[Any]:
        """
        Plot indicator in appropriate subplot

        Args:
            indicator_name: Name of indicator
            data: Indicator data (Series with datetime index)
            subplot: Force specific subplot (optional)
            **plot_kwargs: Additional matplotlib plot arguments

        Returns:
            Line object or None
        """
        try:
            # Determine subplot
            if subplot is None:
                subplot = self.classify_indicator(indicator_name)

            # Check if subplot exists
            if subplot not in self.axes:
                logger.warning(f"Subplot '{subplot}' not created yet. Call create_subplots() first.")
                return None

            ax = self.axes[subplot]

            # Default plot style
            if 'label' not in plot_kwargs:
                plot_kwargs['label'] = indicator_name
            if 'alpha' not in plot_kwargs:
                plot_kwargs['alpha'] = 0.7
            if 'linewidth' not in plot_kwargs:
                plot_kwargs['linewidth'] = 1.2

            # Plot
            line = ax.plot(data.index, data.values, **plot_kwargs)

            # Update legend
            if ax.get_legend() is None:
                ax.legend(loc='upper left', fontsize=8, framealpha=0.7)
            else:
                # Update existing legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc='upper left', fontsize=8, framealpha=0.7)

            # Track active indicator
            if subplot not in self.active_indicators:
                self.active_indicators[subplot] = []
            if indicator_name not in self.active_indicators[subplot]:
                self.active_indicators[subplot].append(indicator_name)

            return line[0] if line else None

        except Exception as e:
            logger.error(f"Failed to plot indicator {indicator_name}: {e}")
            return None

    def plot_price_overlay(
        self,
        indicator_name: str,
        data: pd.Series,
        **plot_kwargs
    ) -> Optional[Any]:
        """
        Plot indicator overlaid on price chart

        Args:
            indicator_name: Name of indicator
            data: Indicator data
            **plot_kwargs: Plot arguments

        Returns:
            Line object or None
        """
        return self.plot_indicator(indicator_name, data, subplot='price', **plot_kwargs)

    def plot_bands(
        self,
        name: str,
        upper: pd.Series,
        middle: pd.Series,
        lower: pd.Series,
        subplot: str = 'price',
        **kwargs
    ) -> Optional[Tuple[Any, Any, Any]]:
        """
        Plot indicator bands (e.g., Bollinger Bands)

        Args:
            name: Band indicator name
            upper: Upper band data
            middle: Middle line data
            lower: Lower band data
            subplot: Target subplot
            **kwargs: Additional plot arguments

        Returns:
            Tuple of line objects or None
        """
        try:
            if subplot not in self.axes:
                logger.warning(f"Subplot '{subplot}' not available")
                return None

            ax = self.axes[subplot]

            # Plot bands
            line_upper = ax.plot(upper.index, upper.values, linestyle='--', alpha=0.5,
                               label=f'{name} Upper', **kwargs)[0]
            line_middle = ax.plot(middle.index, middle.values, linestyle='-', alpha=0.7,
                                label=f'{name}', **kwargs)[0]
            line_lower = ax.plot(lower.index, lower.values, linestyle='--', alpha=0.5,
                               label=f'{name} Lower', **kwargs)[0]

            # Fill between
            ax.fill_between(
                upper.index, upper.values, lower.values,
                alpha=0.1, color=line_middle.get_color()
            )

            # Update legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper left', fontsize=8, framealpha=0.7)

            return line_upper, line_middle, line_lower

        except Exception as e:
            logger.error(f"Failed to plot bands for {name}: {e}")
            return None

    def clear_subplot(self, subplot: str):
        """Clear specific subplot"""
        if subplot in self.axes:
            self.axes[subplot].clear()
            if subplot in self.active_indicators:
                self.active_indicators[subplot] = []

    def clear_all(self):
        """Clear all subplots"""
        for ax in self.axes.values():
            ax.clear()
        self.active_indicators = {
            'price': [],
            'normalized': [],
            'volume': []
        }

    def refresh_layout(self):
        """Refresh subplot layout based on active indicators"""
        has_normalized = len(self.active_indicators.get('normalized', [])) > 0
        has_volume = len(self.active_indicators.get('volume', [])) > 0

        # Recreate subplots if needed
        self.create_subplots(has_normalized=has_normalized, has_volume=has_volume)

    def get_subplot_for_indicator(self, indicator_name: str) -> Optional[str]:
        """Get the subplot where indicator should be plotted"""
        return self.classify_indicator(indicator_name)

    def set_date_limits(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Set x-axis limits for all subplots"""
        for ax in self.axes.values():
            ax.set_xlim(start_date, end_date)

    def autoscale_y(self, subplot: Optional[str] = None):
        """Autoscale y-axis for subplot(s)"""
        if subplot:
            if subplot in self.axes:
                self.axes[subplot].relim()
                self.axes[subplot].autoscale_view(scalex=False, scaley=True)
        else:
            for ax in self.axes.values():
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)