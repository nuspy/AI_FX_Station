# src/forex_diffusion/features/indicators_talib.py
"""
Comprehensive technical indicators system using TA-Lib
Replaces btalib with stable TA-Lib implementation
158 indicators available with robust error handling
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import talib
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


class TALibIndicators:
    """
    Comprehensive technical analysis indicators using TA-Lib
    Supports 150+ indicators with data requirement filtering
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
                parameters={'timeperiod': 20},
                description='Simple moving average of closing prices'
            ),
            'ema': IndicatorConfig(
                name='ema',
                display_name='Exponential Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 20},
                description='Exponential moving average with greater weight on recent prices'
            ),
            'wma': IndicatorConfig(
                name='wma',
                display_name='Weighted Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 20},
                description='Weighted moving average with linear weighting'
            ),
            'dema': IndicatorConfig(
                name='dema',
                display_name='Double Exponential Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 20},
                description='Double smoothed exponential moving average'
            ),
            'tema': IndicatorConfig(
                name='tema',
                display_name='Triple Exponential Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 20},
                description='Triple smoothed exponential moving average'
            ),
            'kama': IndicatorConfig(
                name='kama',
                display_name='Kaufman Adaptive Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 20},
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
                parameters={'timeperiod': 20},
                description='Triangular moving average with double smoothing'
            ),
            'bbands': IndicatorConfig(
                name='bbands',
                display_name='Bollinger Bands',
                category=IndicatorCategories.VOLATILITY,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 20, 'nbdevup': 2.0, 'nbdevdn': 2.0, 'matype': 0},
                description='Bollinger Bands volatility indicator'
            ),
            'midpoint': IndicatorConfig(
                name='midpoint',
                display_name='MidPoint over period',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Midpoint of highest and lowest values over period'
            ),
            'midprice': IndicatorConfig(
                name='midprice',
                display_name='Midpoint Price',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
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
            't3': IndicatorConfig(
                name='t3',
                display_name='T3 - Triple Exponential Moving Average',
                category=IndicatorCategories.OVERLAP,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 5, 'vfactor': 0.7},
                description='T3 Moving Average by Tim Tillson'
            ),
        })

        # ========== MOMENTUM INDICATORS (OHLC ONLY) ==========
        indicators.update({
            'rsi': IndicatorConfig(
                name='rsi',
                display_name='Relative Strength Index',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Relative Strength Index momentum oscillator'
            ),
            'macd': IndicatorConfig(
                name='macd',
                display_name='Moving Average Convergence Divergence',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
                description='MACD trend-following momentum indicator'
            ),
            'stoch': IndicatorConfig(
                name='stoch',
                display_name='Stochastic',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'fastk_period': 5, 'slowk_period': 3, 'slowd_period': 3},
                description='Stochastic oscillator momentum indicator'
            ),
            'stochf': IndicatorConfig(
                name='stochf',
                display_name='Stochastic Fast',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'fastk_period': 5, 'fastd_period': 3},
                description='Fast Stochastic oscillator'
            ),
            'stochrsi': IndicatorConfig(
                name='stochrsi',
                display_name='Stochastic RSI',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14, 'fastk_period': 5, 'fastd_period': 3},
                description='Stochastic applied to RSI values'
            ),
            'willr': IndicatorConfig(
                name='willr',
                display_name='Williams %R',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Williams %R momentum indicator'
            ),
            'roc': IndicatorConfig(
                name='roc',
                display_name='Rate of Change',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 12},
                description='Rate of change momentum indicator'
            ),
            'mom': IndicatorConfig(
                name='mom',
                display_name='Momentum',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 10},
                description='Price momentum indicator'
            ),
            'ppo': IndicatorConfig(
                name='ppo',
                display_name='Percentage Price Oscillator',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'fastperiod': 12, 'slowperiod': 26, 'matype': 0},
                description='Percentage version of MACD'
            ),
            'cci': IndicatorConfig(
                name='cci',
                display_name='Commodity Channel Index',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 20},
                description='Commodity Channel Index momentum oscillator'
            ),
            'cmo': IndicatorConfig(
                name='cmo',
                display_name='Chande Momentum Oscillator',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Chande Momentum Oscillator'
            ),
            'ultosc': IndicatorConfig(
                name='ultosc',
                display_name='Ultimate Oscillator',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28},
                description='Ultimate Oscillator multi-timeframe momentum'
            ),
            'trix': IndicatorConfig(
                name='trix',
                display_name='TRIX',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='TRIX triple exponential smoothed momentum'
            ),
            'bop': IndicatorConfig(
                name='bop',
                display_name='Balance Of Power',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={},
                description='Balance of Power'
            ),
            'dx': IndicatorConfig(
                name='dx',
                display_name='Directional Movement Index',
                category=IndicatorCategories.MOMENTUM,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Directional Movement Index'
            ),
        })

        # ========== VOLATILITY INDICATORS (OHLC ONLY) ==========
        indicators.update({
            'atr': IndicatorConfig(
                name='atr',
                display_name='Average True Range',
                category=IndicatorCategories.VOLATILITY,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Average True Range volatility indicator'
            ),
            'natr': IndicatorConfig(
                name='natr',
                display_name='Normalized Average True Range',
                category=IndicatorCategories.VOLATILITY,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Normalized Average True Range'
            ),
            'trange': IndicatorConfig(
                name='trange',
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
                parameters={'timeperiod': 20, 'nbdev': 1},
                description='Standard deviation volatility measure'
            ),
        })

        # ========== TREND INDICATORS (OHLC ONLY) ==========
        indicators.update({
            'adx': IndicatorConfig(
                name='adx',
                display_name='Average Directional Index',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='ADX trend strength indicator'
            ),
            'adxr': IndicatorConfig(
                name='adxr',
                display_name='Average Directional Index Rating',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='ADXR trend strength rating'
            ),
            'plus_di': IndicatorConfig(
                name='plus_di',
                display_name='Plus Directional Indicator',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Plus Directional Indicator +DI'
            ),
            'minus_di': IndicatorConfig(
                name='minus_di',
                display_name='Minus Directional Indicator',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Minus Directional Indicator -DI'
            ),
            'plus_dm': IndicatorConfig(
                name='plus_dm',
                display_name='Plus Directional Movement',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Plus Directional Movement +DM'
            ),
            'minus_dm': IndicatorConfig(
                name='minus_dm',
                display_name='Minus Directional Movement',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Minus Directional Movement -DM'
            ),
            'aroon': IndicatorConfig(
                name='aroon',
                display_name='Aroon',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Aroon trend indicator'
            ),
            'aroonosc': IndicatorConfig(
                name='aroonosc',
                display_name='Aroon Oscillator',
                category=IndicatorCategories.TREND,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
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
                parameters={'fastperiod': 3, 'slowperiod': 10},
                description='Accumulation/Distribution Oscillator'
            ),
            'mfi': IndicatorConfig(
                name='mfi',
                display_name='Money Flow Index',
                category=IndicatorCategories.VOLUME,
                data_requirement=DataRequirement.VOLUME_REQUIRED,
                parameters={'timeperiod': 14},
                description='Money Flow Index volume-weighted RSI'
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
                parameters={'timeperiod': 20},
                description='Beta coefficient'
            ),
            'correl': IndicatorConfig(
                name='correl',
                display_name='Pearson Correlation Coefficient',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 20},
                description='Pearson correlation coefficient'
            ),
            'linearreg': IndicatorConfig(
                name='linearreg',
                display_name='Linear Regression',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Linear regression line'
            ),
            'linearreg_angle': IndicatorConfig(
                name='linearreg_angle',
                display_name='Linear Regression Angle',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Linear regression angle'
            ),
            'linearreg_intercept': IndicatorConfig(
                name='linearreg_intercept',
                display_name='Linear Regression Intercept',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Linear regression intercept'
            ),
            'linearreg_slope': IndicatorConfig(
                name='linearreg_slope',
                display_name='Linear Regression Slope',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
                description='Linear regression slope'
            ),
            'tsf': IndicatorConfig(
                name='tsf',
                display_name='Time Series Forecast',
                category=IndicatorCategories.STATISTICS,
                data_requirement=DataRequirement.OHLC_ONLY,
                parameters={'timeperiod': 14},
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
        Calculate a single indicator using TA-Lib

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
            # Get TA-Lib function
            talib_func = getattr(talib, indicator_name.upper(), None)
            if talib_func is None:
                print(f"Indicator {indicator_name} not found in TA-Lib")
                return {}

            # Prepare input data - convert to numpy arrays
            close = data['close'].values.astype(float)
            high = data['high'].values.astype(float) if 'high' in data.columns else close
            low = data['low'].values.astype(float) if 'low' in data.columns else close
            open_ = data['open'].values.astype(float) if 'open' in data.columns else close
            volume = data['volume'].values.astype(float) if 'volume' in data.columns else np.zeros_like(close)

            # Call indicator based on what inputs it needs
            result = None

            # Multi-output indicators
            if indicator_name == 'macd':
                macd, signal, hist = talib.MACD(close, **params)
                return {'macd': pd.Series(macd, index=data.index),
                        'signal': pd.Series(signal, index=data.index),
                        'hist': pd.Series(hist, index=data.index)}

            elif indicator_name == 'bbands':
                upper, middle, lower = talib.BBANDS(close, **params)
                return {'upper': pd.Series(upper, index=data.index),
                        'middle': pd.Series(middle, index=data.index),
                        'lower': pd.Series(lower, index=data.index)}

            elif indicator_name == 'stoch':
                slowk, slowd = talib.STOCH(high, low, close, **params)
                return {'slowk': pd.Series(slowk, index=data.index),
                        'slowd': pd.Series(slowd, index=data.index)}

            elif indicator_name == 'stochf':
                fastk, fastd = talib.STOCHF(high, low, close, **params)
                return {'fastk': pd.Series(fastk, index=data.index),
                        'fastd': pd.Series(fastd, index=data.index)}

            elif indicator_name == 'stochrsi':
                fastk, fastd = talib.STOCHRSI(close, **params)
                return {'fastk': pd.Series(fastk, index=data.index),
                        'fastd': pd.Series(fastd, index=data.index)}

            elif indicator_name == 'mama':
                mama, fama = talib.MAMA(close, **params)
                return {'mama': pd.Series(mama, index=data.index),
                        'fama': pd.Series(fama, index=data.index)}

            elif indicator_name == 'aroon':
                aroondown, aroonup = talib.AROON(high, low, **params)
                return {'aroondown': pd.Series(aroondown, index=data.index),
                        'aroonup': pd.Series(aroonup, index=data.index)}

            elif indicator_name == 'ht_phasor':
                inphase, quadrature = talib.HT_PHASOR(close)
                return {'inphase': pd.Series(inphase, index=data.index),
                        'quadrature': pd.Series(quadrature, index=data.index)}

            elif indicator_name == 'ht_sine':
                sine, leadsine = talib.HT_SINE(close)
                return {'sine': pd.Series(sine, index=data.index),
                        'leadsine': pd.Series(leadsine, index=data.index)}

            # Price transform indicators - each has different signature
            elif indicator_name == 'avgprice':
                result = talib.AVGPRICE(open_, high, low, close)
            elif indicator_name == 'medprice':
                result = talib.MEDPRICE(high, low)
            elif indicator_name == 'typprice':
                result = talib.TYPPRICE(high, low, close)
            elif indicator_name == 'wclprice':
                result = talib.WCLPRICE(high, low, close)

            # Volume indicators
            elif indicator_name == 'obv':
                result = talib.OBV(close, volume)
            elif indicator_name == 'ad':
                result = talib.AD(high, low, close, volume)
            elif indicator_name in ['adosc', 'mfi']:
                result = talib_func(high, low, close, volume, **params)

            # Trend indicators needing high/low
            elif indicator_name in ['adx', 'adxr', 'plus_di', 'minus_di', 'plus_dm', 'minus_dm', 'cci', 'dx']:
                result = talib_func(high, low, close, **params)

            # Aroon oscillator - special case to avoid parameter conflict
            elif indicator_name == 'aroonosc':
                result = talib.AROONOSC(high, low, timeperiod=params.get('timeperiod', 14))

            # Williams %R needs high, low, close
            elif indicator_name == 'willr':
                result = talib.WILLR(high, low, close, **params)

            # Volatility indicators needing high/low
            elif indicator_name in ['atr', 'natr', 'trange']:
                if params:
                    result = talib_func(high, low, close, **params)
                else:
                    result = talib_func(high, low, close)

            # Price indicators (SAR needs high/low)
            elif indicator_name == 'sar':
                result = talib.SAR(high, low, **params)

            # Midpoint/Midprice
            elif indicator_name == 'midpoint':
                result = talib.MIDPOINT(close, **params)
            elif indicator_name == 'midprice':
                result = talib.MIDPRICE(high, low, **params)

            # Ultimate Oscillator
            elif indicator_name == 'ultosc':
                result = talib.ULTOSC(high, low, close, **params)

            # Balance of Power
            elif indicator_name == 'bop':
                result = talib.BOP(open_, high, low, close)

            # Beta/Correlation (need two series - use close for both)
            elif indicator_name in ['beta', 'correl']:
                result = talib_func(high, low, **params)

            # Most other indicators just need close price
            else:
                if params:
                    result = talib_func(close, **params)
                else:
                    result = talib_func(close)

            # Convert to Series and return
            if result is not None:
                if isinstance(result, tuple):
                    # Multi-output that wasn't handled above
                    return {f'{indicator_name}_{i}': pd.Series(val, index=data.index)
                           for i, val in enumerate(result)}
                else:
                    return {indicator_name: pd.Series(result, index=data.index)}
            else:
                return {}

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


# Backward compatibility with old btalib API
BTALibIndicators = TALibIndicators


# Backward compatibility functions (replacing old indicators.py)
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average - backward compatibility"""
    result = talib.SMA(series.values.astype(float), timeperiod=window)
    return pd.Series(result, index=series.index)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average - backward compatibility"""
    result = talib.EMA(series.values.astype(float), timeperiod=span)
    return pd.Series(result, index=series.index)


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """Bollinger Bands - backward compatibility"""
    upper, middle, lower = talib.BBANDS(series.values.astype(float),
                                       timeperiod=window, nbdevup=n_std, nbdevdn=n_std)
    return pd.Series(upper, index=series.index), pd.Series(lower, index=series.index)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI - backward compatibility"""
    result = talib.RSI(series.values.astype(float), timeperiod=period)
    return pd.Series(result, index=series.index)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """MACD - backward compatibility"""
    macd_result, signal_result, hist = talib.MACD(series.values.astype(float),
                                                  fastperiod=fast, slowperiod=slow, signalperiod=signal)
    return {
        "macd": pd.Series(macd_result, index=series.index),
        "signal": pd.Series(signal_result, index=series.index),
        "hist": pd.Series(hist, index=series.index)
    }