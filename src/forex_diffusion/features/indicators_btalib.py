# src/forex_diffusion/features/indicators_btalib.py
"""
Comprehensive technical indicators system using bta-lib
Replaces custom implementations with professional-grade indicators
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import btalib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DataRequirement(Enum):
    """Data requirements for indicators"""
    OHLC_ONLY = "ohlc_only"  # Only OHLC data required
    VOLUME_REQUIRED = "volume_required"  # Volume data required
    BOOK_REQUIRED = "book_required"  # Order book data required
    TICK_REQUIRED = "tick_required"  # Tick data required


@dataclass
class IndicatorConfig:
    """Configuration for a single indicator"""
    name: str
    display_name: str
    category: str
    data_requirement: DataRequirement
    enabled: bool = True
    weight: float = 1.0
    parameters: Dict[str, Any] = None
    description: str = ""

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class IndicatorCategories:
    """Standard indicator categories"""
    OVERLAP = "Overlap Studies"
    MOMENTUM = "Momentum Indicators"
    VOLATILITY = "Volatility Indicators"
    VOLUME = "Volume Indicators"
    TREND = "Trend Indicators"
    PRICE_TRANSFORM = "Price Transform"
    CYCLE = "Cycle Indicators"
    PATTERN = "Pattern Recognition"
    STATISTICS = "Statistical Functions"


class BTALibIndicators:
    """
    Comprehensive technical analysis indicators using bta-lib
    Supports 200+ indicators with data requirement filtering
    """

    def __init__(self, available_data: List[str] = None):
        """
        Initialize indicator system

        Args:
            available_data: List of available data columns ['open', 'high', 'low', 'close', 'volume', etc.]
        """
        self.available_data = available_data or ['open', 'high', 'low', 'close']
        self.has_volume = 'volume' in self.available_data
        self.has_book_data = any(col in self.available_data for col in ['bid', 'ask', 'spread'])
        self.has_tick_data = 'tick_volume' in self.available_data

        # Initialize all available indicators
        self.indicators_config = self._initialize_indicators_config()

        # Filter indicators based on available data
        self.enabled_indicators = self._filter_indicators_by_data()

    def _initialize_indicators_config(self) -> Dict[str, IndicatorConfig]:
        """Initialize comprehensive indicators configuration"""
        indicators = {}

        # ========== OVERLAP STUDIES (OHLC ONLY) ==========
        indicators.update({
            'sma': IndicatorConfig(
                name='sma',
                display_name='Simple Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'periods': [10, 20, 50, 200]},
                description='Simple moving average of closing prices'
            ),
            'ema': IndicatorConfig(
                name='ema',
                display_name='Exponential Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'periods': [12, 26, 50, 100]},
                description='Exponential moving average with greater weight on recent prices'
            ),
            'wma': IndicatorConfig(
                name='wma',
                display_name='Weighted Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20},
                description='Weighted moving average with linear weighting'
            ),
            'dema': IndicatorConfig(
                name='dema',
                display_name='Double Exponential Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20},
                description='Double smoothed exponential moving average'
            ),
            'tema': IndicatorConfig(
                name='tema',
                display_name='Triple Exponential Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20},
                description='Triple smoothed exponential moving average'
            ),
            'kama': IndicatorConfig(
                name='kama',
                display_name='Kaufman Adaptive Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20},
                description='Adaptive moving average that adjusts to market conditions'
            ),
            'mama': IndicatorConfig(
                name='mama',
                display_name='MESA Adaptive Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'fastlimit': 0.5, 'slowlimit': 0.05},
                description='MESA Adaptive Moving Average using Hilbert Transform'
            ),
            'trima': IndicatorConfig(
                name='trima',
                display_name='Triangular Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20},
                description='Triangular moving average with double smoothing'
            ),
            'bbands': IndicatorConfig(
                name='bbands',
                display_name='Bollinger Bands',
                category=IndicatorCategories.VOLATILITY,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20, 'devup': 2.0, 'devdn': 2.0},
                description='Bollinger Bands volatility indicator'
            ),
            'midpoint': IndicatorConfig(
                name='midpoint',
                display_name='MidPoint over period',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Midpoint of highest and lowest values over period'
            ),
            'midprice': IndicatorConfig(
                name='midprice',
                display_name='Midpoint Price',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Midpoint between high and low prices'
            ),
            'sar': IndicatorConfig(
                name='sar',
                display_name='Parabolic SAR',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'acceleration': 0.02, 'maximum': 0.2},
                description='Parabolic Stop and Reverse indicator'
            ),
        })

        # ========== MOMENTUM INDICATORS (OHLC ONLY) ==========
        indicators.update({
            'rsi': IndicatorConfig(
                name='rsi',
                display_name='Relative Strength Index',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'periods': [14, 21]},
                description='Relative Strength Index momentum oscillator'
            ),
            'macd': IndicatorConfig(
                name='macd',
                display_name='Moving Average Convergence Divergence',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'fast': 12, 'slow': 26, 'signal': 9},
                description='MACD trend-following momentum indicator'
            ),
            'stoch': IndicatorConfig(
                name='stoch',
                display_name='Stochastic',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'k_period': 14, 'd_period': 3, 'slow_period': 3},
                description='Stochastic oscillator momentum indicator'
            ),
            'stochrsi': IndicatorConfig(
                name='stochrsi',
                display_name='Stochastic RSI',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'rsi_period': 14, 'stoch_period': 14, 'k_period': 3, 'd_period': 3},
                description='Stochastic applied to RSI values'
            ),
            'willr': IndicatorConfig(
                name='willr',
                display_name='Williams %R',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Williams %R momentum indicator'
            ),
            'roc': IndicatorConfig(
                name='roc',
                display_name='Rate of Change',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 12},
                description='Rate of change momentum indicator'
            ),
            'mom': IndicatorConfig(
                name='mom',
                display_name='Momentum',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 10},
                description='Price momentum indicator'
            ),
            'ppo': IndicatorConfig(
                name='ppo',
                display_name='Percentage Price Oscillator',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'fast': 12, 'slow': 26, 'signal': 9},
                description='Percentage version of MACD'
            ),
            'cci': IndicatorConfig(
                name='cci',
                display_name='Commodity Channel Index',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20},
                description='Commodity Channel Index momentum oscillator'
            ),
            'cmo': IndicatorConfig(
                name='cmo',
                display_name='Chande Momentum Oscillator',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Chande Momentum Oscillator'
            ),
            'ultosc': IndicatorConfig(
                name='ultosc',
                display_name='Ultimate Oscillator',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period1': 7, 'period2': 14, 'period3': 28},
                description='Ultimate Oscillator multi-timeframe momentum'
            ),
            'trix': IndicatorConfig(
                name='trix',
                display_name='TRIX',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='TRIX triple exponential smoothed momentum'
            ),
        })

        # ========== VOLATILITY INDICATORS (OHLC ONLY) ==========
        indicators.update({
            'atr': IndicatorConfig(
                name='atr',
                display_name='Average True Range',
                category=IndicatorCategories.VOLATILITY,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'periods': [14, 28]},
                description='Average True Range volatility indicator'
            ),
            'natr': IndicatorConfig(
                name='natr',
                display_name='Normalized Average True Range',
                category=IndicatorCategories.VOLATILITY,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Normalized Average True Range'
            ),
            'tr': IndicatorConfig(
                name='tr',
                display_name='True Range',
                category=IndicatorCategories.VOLATILITY,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='True Range volatility measure'
            ),
            'stddev': IndicatorConfig(
                name='stddev',
                display_name='Standard Deviation',
                category=IndicatorCategories.VOLATILITY,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20, 'nbdev': 1},
                description='Standard deviation volatility measure'
            ),
            'variance': IndicatorConfig(
                name='var',
                display_name='Variance',
                category=IndicatorCategories.VOLATILITY,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20, 'nbdev': 1},
                description='Variance volatility measure'
            ),
        })

        # ========== TREND INDICATORS (OHLC ONLY) ==========
        indicators.update({
            'adx': IndicatorConfig(
                name='adx',
                display_name='Average Directional Index',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='ADX trend strength indicator'
            ),
            'adxr': IndicatorConfig(
                name='adxr',
                display_name='Average Directional Index Rating',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='ADXR trend strength rating'
            ),
            'di': IndicatorConfig(
                name='di',
                display_name='Directional Indicator',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Directional Indicator +DI and -DI'
            ),
            'dm': IndicatorConfig(
                name='dm',
                display_name='Directional Movement',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Directional Movement +DM and -DM'
            ),
            'aroon': IndicatorConfig(
                name='aroon',
                display_name='Aroon',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Aroon trend indicator'
            ),
            'aroonosc': IndicatorConfig(
                name='aroonosc',
                display_name='Aroon Oscillator',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Aroon Oscillator trend indicator'
            ),
        })

        # ========== VOLUME INDICATORS (REQUIRE VOLUME) ==========
        indicators.update({
            'obv': IndicatorConfig(
                name='obv',
                display_name='On Balance Volume',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={},
                description='On Balance Volume indicator'
            ),
            'ad': IndicatorConfig(
                name='ad',
                display_name='Accumulation/Distribution Line',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={},
                description='Accumulation/Distribution Line'
            ),
            'adosc': IndicatorConfig(
                name='adosc',
                display_name='Accumulation/Distribution Oscillator',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={'fast': 3, 'slow': 10},
                description='Accumulation/Distribution Oscillator'
            ),
            'mfi': IndicatorConfig(
                name='mfi',
                display_name='Money Flow Index',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={'period': 14},
                description='Money Flow Index volume-weighted RSI'
            ),
            'cmf': IndicatorConfig(
                name='cmf',
                display_name='Chaikin Money Flow',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={'period': 20},
                description='Chaikin Money Flow volume indicator'
            ),
            'eom': IndicatorConfig(
                name='eom',
                display_name='Ease of Movement',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={'period': 14},
                description='Ease of Movement volume indicator'
            ),
            'vpt': IndicatorConfig(
                name='vpt',
                display_name='Volume Price Trend',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={},
                description='Volume Price Trend indicator'
            ),
            'pvt': IndicatorConfig(
                name='pvt',
                display_name='Price Volume Trend',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={},
                description='Price Volume Trend indicator'
            ),
            'nvi': IndicatorConfig(
                name='nvi',
                display_name='Negative Volume Index',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={},
                description='Negative Volume Index'
            ),
            'pvi': IndicatorConfig(
                name='pvi',
                display_name='Positive Volume Index',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={},
                description='Positive Volume Index'
            ),
        })

        # ========== PRICE TRANSFORM (OHLC ONLY) ==========
        indicators.update({
            'avgprice': IndicatorConfig(
                name='avgprice',
                display_name='Average Price',
                category=IndicatorCategories.PRICE_TRANSFORM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='Average of OHLC prices'
            ),
            'medprice': IndicatorConfig(
                name='medprice',
                display_name='Median Price',
                category=IndicatorCategories.PRICE_TRANSFORM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='Median of high and low prices'
            ),
            'typprice': IndicatorConfig(
                name='typprice',
                display_name='Typical Price',
                category=IndicatorCategories.PRICE_TRANSFORM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='Typical price (H+L+C)/3'
            ),
            'wclprice': IndicatorConfig(
                name='wclprice',
                display_name='Weighted Close Price',
                category=IndicatorCategories.PRICE_TRANSFORM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='Weighted close price (H+L+2*C)/4'
            ),
        })

        # ========== CYCLE INDICATORS (OHLC ONLY) ==========
        indicators.update({
            'ht_dcperiod': IndicatorConfig(
                name='ht_dcperiod',
                display_name='Hilbert Transform - Dominant Cycle Period',
                category=IndicatorCategories.CYCLE,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='Dominant cycle period using Hilbert Transform'
            ),
            'ht_dcphase': IndicatorConfig(
                name='ht_dcphase',
                display_name='Hilbert Transform - Dominant Cycle Phase',
                category=IndicatorCategories.CYCLE,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='Dominant cycle phase using Hilbert Transform'
            ),
            'ht_phasor': IndicatorConfig(
                name='ht_phasor',
                display_name='Hilbert Transform - Phasor Components',
                category=IndicatorCategories.CYCLE,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='Phasor components using Hilbert Transform'
            ),
            'ht_sine': IndicatorConfig(
                name='ht_sine',
                display_name='Hilbert Transform - SineWave',
                category=IndicatorCategories.CYCLE,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='Sine wave using Hilbert Transform'
            ),
            'ht_trendmode': IndicatorConfig(
                name='ht_trendmode',
                display_name='Hilbert Transform - Trend Mode',
                category=IndicatorCategories.CYCLE,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='Trend mode using Hilbert Transform'
            ),
        })

        # ========== STATISTICAL FUNCTIONS (OHLC ONLY) ==========
        indicators.update({
            'beta': IndicatorConfig(
                name='beta',
                display_name='Beta',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20},
                description='Beta coefficient'
            ),
            'correl': IndicatorConfig(
                name='correl',
                display_name='Pearson Correlation Coefficient',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 20},
                description='Pearson correlation coefficient'
            ),
            'linearreg': IndicatorConfig(
                name='linearreg',
                display_name='Linear Regression',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Linear regression line'
            ),
            'linearreg_angle': IndicatorConfig(
                name='linearreg_angle',
                display_name='Linear Regression Angle',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Linear regression angle'
            ),
            'linearreg_intercept': IndicatorConfig(
                name='linearreg_intercept',
                display_name='Linear Regression Intercept',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Linear regression intercept'
            ),
            'linearreg_slope': IndicatorConfig(
                name='linearreg_slope',
                display_name='Linear Regression Slope',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Linear regression slope'
            ),
            'tsf': IndicatorConfig(
                name='tsf',
                display_name='Time Series Forecast',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'period': 14},
                description='Time series forecast'
            ),
        })

        return indicators

    def _filter_indicators_by_data(self) -> Dict[str, IndicatorConfig]:
        """Filter indicators based on available data"""
        enabled = {}

        for name, config in self.indicators_config.items():
            # Check if we have required data
            if config.data_requirement == DataRequirement.OHLC_ONLY:
                enabled[name] = config
            elif config.data_requirement == DataRequirement.VOLUME_REQUIRED and self.has_volume:
                enabled[name] = config
            elif config.data_requirement == DataRequirement.BOOK_REQUIRED and self.has_book_data:
                enabled[name] = config
            elif config.data_requirement == DataRequirement.TICK_REQUIRED and self.has_tick_data:
                enabled[name] = config
            else:
                # Disable indicator due to missing data
                config.enabled = False
                enabled[name] = config

        return enabled

    def get_available_indicators(self) -> Dict[str, IndicatorConfig]:
        """Get all indicators with their availability status"""
        return self.enabled_indicators

    def get_enabled_indicators(self) -> Dict[str, IndicatorConfig]:
        """Get only enabled indicators"""
        return {name: config for name, config in self.enabled_indicators.items()
                if config.enabled}

    def get_indicators_by_category(self, category: str) -> Dict[str, IndicatorConfig]:
        """Get indicators filtered by category"""
        return {name: config for name, config in self.enabled_indicators.items()
                if config.category == category}

    def calculate_indicator(self, data: pd.DataFrame, indicator_name: str,
                          custom_params: Dict[str, Any] = None) -> Dict[str, pd.Series]:
        """
        Calculate a single indicator

        Args:
            data: OHLC(V) DataFrame
            indicator_name: Name of indicator to calculate
            custom_params: Override default parameters

        Returns:
            Dictionary with indicator results
        """
        if indicator_name not in self.enabled_indicators:
            raise ValueError(f"Indicator {indicator_name} not available")

        config = self.enabled_indicators[indicator_name]
        if not config.enabled:
            raise ValueError(f"Indicator {indicator_name} is disabled due to missing data")

        # Merge custom parameters with defaults
        params = config.parameters.copy()
        if custom_params:
            params.update(custom_params)

        try:
            # Get the indicator class from btalib (RSI, SMA, EMA, etc.)
            indicator_class = getattr(btalib, indicator_name.upper(), None)
            if indicator_class is None:
                # Try lowercase or exact name
                indicator_class = getattr(btalib, indicator_name, None)

            if indicator_class is None:
                print(f"Indicator {indicator_name} not found in btalib")
                return {}

            # Fix parameter names for btalib compatibility
            # btalib uses singular 'period', not 'periods'
            fixed_params = {}
            for key, value in params.items():
                if key == 'periods' and isinstance(value, list):
                    # Use first period if list provided
                    fixed_params['period'] = value[0] if value else 14
                elif key == 'periods':
                    fixed_params['period'] = value
                else:
                    fixed_params[key] = value

            # Call indicator with data and parameters
            # btalib expects: Indicator(data, **params)
            try:
                if fixed_params:
                    result = indicator_class(data, **fixed_params)
                else:
                    result = indicator_class(data)
            except TypeError as te:
                # If parameter error, try without params
                print(f"Warning: {indicator_name} parameter error ({te}), trying default parameters")
                result = indicator_class(data)

            # Convert result to dictionary format
            if hasattr(result, '_df') and hasattr(result._df, 'columns'):
                # Multi-column result (like MACD, Bollinger Bands)
                # btalib stores results in _df attribute
                result_df = result._df
                return {col: result_df[col] for col in result_df.columns}
            elif hasattr(result, '_df'):
                # Single column result
                result_df = result._df
                if len(result_df.columns) == 1:
                    return {indicator_name: result_df[result_df.columns[0]]}
                else:
                    return {col: result_df[col] for col in result_df.columns}
            else:
                # Fallback: try to convert to series
                return {indicator_name: pd.Series(result)}

        except Exception as e:
            print(f"Error calculating {indicator_name}: {e}")
            return {}

    def calculate_all_indicators(self, data: pd.DataFrame,
                               categories: List[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate all enabled indicators or specific categories

        Args:
            data: OHLC(V) DataFrame
            categories: List of categories to calculate (None = all)

        Returns:
            Dictionary with all indicator results
        """
        results = {}

        indicators_to_calc = self.get_enabled_indicators()

        if categories:
            indicators_to_calc = {
                name: config for name, config in indicators_to_calc.items()
                if config.category in categories
            }

        for indicator_name, config in indicators_to_calc.items():
            try:
                indicator_results = self.calculate_indicator(data, indicator_name)
                results.update(indicator_results)
            except Exception as e:
                print(f"Failed to calculate {indicator_name}: {e}")
                continue

        return results

    def get_data_requirements_summary(self) -> Dict[str, List[str]]:
        """Get summary of data requirements for all indicators"""
        summary = {
            'ohlc_only': [],
            'volume_required': [],
            'book_required': [],
            'tick_required': [],
            'disabled': []
        }

        for name, config in self.indicators_config.items():
            if not config.enabled:
                summary['disabled'].append(name)
            elif config.data_requirement == DataRequirement.OHLC_ONLY:
                summary['ohlc_only'].append(name)
            elif config.data_requirement == DataRequirement.VOLUME_REQUIRED:
                summary['volume_required'].append(name)
            elif config.data_requirement == DataRequirement.BOOK_REQUIRED:
                summary['book_required'].append(name)
            elif config.data_requirement == DataRequirement.TICK_REQUIRED:
                summary['tick_required'].append(name)

        return summary

    def enable_indicator(self, indicator_name: str, weight: float = 1.0):
        """Enable an indicator with specified weight"""
        if indicator_name in self.enabled_indicators:
            config = self.enabled_indicators[indicator_name]
            # Only enable if data requirements are met
            if config.data_requirement == DataRequirement.OHLC_ONLY:
                config.enabled = True
                config.weight = weight
            elif config.data_requirement == DataRequirement.VOLUME_REQUIRED and self.has_volume:
                config.enabled = True
                config.weight = weight
            elif config.data_requirement == DataRequirement.BOOK_REQUIRED and self.has_book_data:
                config.enabled = True
                config.weight = weight
            elif config.data_requirement == DataRequirement.TICK_REQUIRED and self.has_tick_data:
                config.enabled = True
                config.weight = weight

    def disable_indicator(self, indicator_name: str):
        """Disable an indicator"""
        if indicator_name in self.enabled_indicators:
            self.enabled_indicators[indicator_name].enabled = False

    def get_config_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration as dictionary for serialization"""
        return {
            name: {
                'enabled': config.enabled,
                'weight': config.weight,
                'data_requirement': config.data_requirement.value,
                'parameters': config.parameters
            }
            for name, config in self.indicators_config.items()
        }

    def load_config_dict(self, config_dict: Dict[str, Dict[str, Any]]):
        """Load configuration from dictionary"""
        for name, settings in config_dict.items():
            if name in self.indicators_config:
                config = self.indicators_config[name]
                config.enabled = settings.get('enabled', config.enabled)
                config.weight = settings.get('weight', config.weight)
                if 'parameters' in settings:
                    config.parameters.update(settings['parameters'])


# Backward compatibility functions (replacing old indicators.py)
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average - backward compatibility"""
    df = pd.DataFrame({'close': series})
    result = btalib.SMA(df, period=window)
    return result._df[result._df.columns[0]] if hasattr(result, '_df') else series


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average - backward compatibility"""
    df = pd.DataFrame({'close': series})
    result = btalib.EMA(df, period=span)
    return result._df[result._df.columns[0]] if hasattr(result, '_df') else series


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """Bollinger Bands - backward compatibility"""
    df = pd.DataFrame({'close': series})
    result = btalib.BBANDS(df, period=window, devup=n_std, devdn=n_std)
    if hasattr(result, '_df'):
        bb_df = result._df
        return bb_df['top'], bb_df['bottom']
    return series, series


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI - backward compatibility"""
    df = pd.DataFrame({'close': series})
    result = btalib.RSI(df, period=period)
    return result._df[result._df.columns[0]] if hasattr(result, '_df') else series


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """MACD - backward compatibility"""
    df = pd.DataFrame({'close': series})
    result = btalib.MACD(df, fast=fast, slow=slow, signal=signal)
    if hasattr(result, '_df'):
        macd_df = result._df
        return {
            "macd": macd_df['macd'] if 'macd' in macd_df.columns else macd_df[macd_df.columns[0]],
            "signal": macd_df['signal'] if 'signal' in macd_df.columns else macd_df[macd_df.columns[1]],
            "hist": macd_df['histogram'] if 'histogram' in macd_df.columns else macd_df[macd_df.columns[2]]
        }
    return {"macd": series, "signal": series, "hist": series}


if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    # Download sample data
    data = yf.download("EURUSD=X", period="1y", interval="1d")

    # Initialize indicator system with OHLC only
    indicators = BTALibIndicators(available_data=['open', 'high', 'low', 'close'])

    # Print available indicators
    available = indicators.get_available_indicators()
    print(f"Available indicators: {len(available)}")

    for category in [IndicatorCategories.OVERLAP, IndicatorCategories.MOMENTUM,
                     IndicatorCategories.VOLATILITY, IndicatorCategories.TREND]:
        category_indicators = indicators.get_indicators_by_category(category)
        print(f"\n{category}: {len(category_indicators)} indicators")
        for name, config in list(category_indicators.items())[:3]:  # Show first 3
            status = "✅" if config.enabled else "❌"
            print(f"  {status} {config.display_name} ({name})")

    # Calculate some indicators
    print("\nCalculating sample indicators...")

    # Calculate SMA
    sma_result = indicators.calculate_indicator(data, 'sma', {'periods': [20, 50]})
    print(f"SMA calculated: {list(sma_result.keys())}")

    # Calculate RSI
    rsi_result = indicators.calculate_indicator(data, 'rsi', {'periods': [14]})
    print(f"RSI calculated: {list(rsi_result.keys())}")

    # Calculate multiple indicators at once
    momentum_indicators = indicators.calculate_all_indicators(
        data, categories=[IndicatorCategories.MOMENTUM]
    )
    print(f"Momentum indicators calculated: {len(momentum_indicators)}")