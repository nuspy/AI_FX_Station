"""
Indicator Range Classification System
Categorizes indicators based on their value ranges for optimal chart subplot organization
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass


class IndicatorRange(Enum):
    """Classification of indicators based on their value ranges"""
    NORMALIZED_0_1 = "normalized_0_1"           # RSI, Stochastic, Williams %R (0-1 range)
    NORMALIZED_NEG1_POS1 = "normalized_-1_+1"   # CCI, CMO (-1 to +1 range)
    PRICE_OVERLAY = "price_overlay"             # Moving averages, Bollinger Bands (price range)
    VOLUME_BASED = "volume_based"               # Volume indicators (volume range)
    CUSTOM_RANGE = "custom_range"               # Custom range indicators (MACD, etc.)
    UNBOUNDED = "unbounded"                     # No specific bounds (momentum, etc.)


@dataclass
class IndicatorRangeConfig:
    """Configuration for indicator range classification"""
    name: str
    display_name: str
    range_type: IndicatorRange
    min_value: float = None
    max_value: float = None
    typical_range: str = ""
    subplot_recommendation: str = ""
    description: str = ""


class IndicatorRangeClassifier:
    """
    Classifier to categorize indicators by their value ranges
    Used to determine optimal subplot placement in charts
    """

    def __init__(self):
        self.range_configs = self._initialize_range_classifications()

    def _initialize_range_classifications(self) -> Dict[str, IndicatorRangeConfig]:
        """Initialize comprehensive range classifications for all indicators"""
        configs = {}

        # ========== NORMALIZED 0-1 INDICATORS ==========
        normalized_0_1_indicators = {
            'rsi': IndicatorRangeConfig(
                name='rsi',
                display_name='Relative Strength Index',
                range_type=IndicatorRange.NORMALIZED_0_1,
                min_value=0,
                max_value=100,
                typical_range="0-100",
                subplot_recommendation="normalized_subplot",
                description="Momentum oscillator measuring speed and magnitude of price changes"
            ),
            'stoch': IndicatorRangeConfig(
                name='stoch',
                display_name='Stochastic Oscillator',
                range_type=IndicatorRange.NORMALIZED_0_1,
                min_value=0,
                max_value=100,
                typical_range="0-100",
                subplot_recommendation="normalized_subplot",
                description="Momentum indicator comparing closing price to price range"
            ),
            'stochf': IndicatorRangeConfig(
                name='stochf',
                display_name='Stochastic Fast',
                range_type=IndicatorRange.NORMALIZED_0_1,
                min_value=0,
                max_value=100,
                typical_range="0-100",
                subplot_recommendation="normalized_subplot",
                description="Fast version of stochastic oscillator"
            ),
            'willr': IndicatorRangeConfig(
                name='willr',
                display_name='Williams %R',
                range_type=IndicatorRange.NORMALIZED_0_1,
                min_value=-100,
                max_value=0,
                typical_range="-100 to 0",
                subplot_recommendation="normalized_subplot",
                description="Momentum indicator measuring overbought/oversold levels"
            ),
            'mfi': IndicatorRangeConfig(
                name='mfi',
                display_name='Money Flow Index',
                range_type=IndicatorRange.NORMALIZED_0_1,
                min_value=0,
                max_value=100,
                typical_range="0-100",
                subplot_recommendation="normalized_subplot",
                description="Volume-weighted version of RSI"
            ),
            'ult': IndicatorRangeConfig(
                name='ult',
                display_name='Ultimate Oscillator',
                range_type=IndicatorRange.NORMALIZED_0_1,
                min_value=0,
                max_value=100,
                typical_range="0-100",
                subplot_recommendation="normalized_subplot",
                description="Momentum oscillator using multiple timeframes"
            ),
            'aroonosc': IndicatorRangeConfig(
                name='aroonosc',
                display_name='Aroon Oscillator',
                range_type=IndicatorRange.NORMALIZED_NEG1_POS1,
                min_value=-100,
                max_value=100,
                typical_range="-100 to +100",
                subplot_recommendation="normalized_subplot",
                description="Trend-following indicator showing relationship between highs and lows"
            )
        }
        configs.update(normalized_0_1_indicators)

        # ========== PRICE OVERLAY INDICATORS ==========
        price_overlay_indicators = {
            'sma': IndicatorRangeConfig(
                name='sma',
                display_name='Simple Moving Average',
                range_type=IndicatorRange.PRICE_OVERLAY,
                typical_range="Price range",
                subplot_recommendation="main_chart",
                description="Average of closing prices over specified period"
            ),
            'ema': IndicatorRangeConfig(
                name='ema',
                display_name='Exponential Moving Average',
                range_type=IndicatorRange.PRICE_OVERLAY,
                typical_range="Price range",
                subplot_recommendation="main_chart",
                description="Weighted average giving more importance to recent prices"
            ),
            'wma': IndicatorRangeConfig(
                name='wma',
                display_name='Weighted Moving Average',
                range_type=IndicatorRange.PRICE_OVERLAY,
                typical_range="Price range",
                subplot_recommendation="main_chart",
                description="Moving average with linear weighting"
            ),
            'tema': IndicatorRangeConfig(
                name='tema',
                display_name='Triple Exponential Moving Average',
                range_type=IndicatorRange.PRICE_OVERLAY,
                typical_range="Price range",
                subplot_recommendation="main_chart",
                description="Triple smoothed exponential moving average"
            ),
            'bbands': IndicatorRangeConfig(
                name='bbands',
                display_name='Bollinger Bands',
                range_type=IndicatorRange.PRICE_OVERLAY,
                typical_range="Price range ± 2σ",
                subplot_recommendation="main_chart",
                description="Volatility bands around moving average"
            ),
            'sar': IndicatorRangeConfig(
                name='sar',
                display_name='Parabolic SAR',
                range_type=IndicatorRange.PRICE_OVERLAY,
                typical_range="Price range",
                subplot_recommendation="main_chart",
                description="Stop and reverse indicator for trend following"
            ),
            'kama': IndicatorRangeConfig(
                name='kama',
                display_name='Kaufman Adaptive Moving Average',
                range_type=IndicatorRange.PRICE_OVERLAY,
                typical_range="Price range",
                subplot_recommendation="main_chart",
                description="Adaptive moving average adjusting to market volatility"
            ),
            'mama': IndicatorRangeConfig(
                name='mama',
                display_name='MESA Adaptive Moving Average',
                range_type=IndicatorRange.PRICE_OVERLAY,
                typical_range="Price range",
                subplot_recommendation="main_chart",
                description="Adaptive moving average based on Hilbert Transform"
            )
        }
        configs.update(price_overlay_indicators)

        # ========== VOLUME INDICATORS ==========
        volume_indicators = {
            'ad': IndicatorRangeConfig(
                name='ad',
                display_name='Chaikin A/D Line',
                range_type=IndicatorRange.VOLUME_BASED,
                typical_range="Cumulative volume",
                subplot_recommendation="volume_subplot",
                description="Accumulation/Distribution Line"
            ),
            'adosc': IndicatorRangeConfig(
                name='adosc',
                display_name='Chaikin A/D Oscillator',
                range_type=IndicatorRange.CUSTOM_RANGE,
                typical_range="Oscillating around zero",
                subplot_recommendation="custom_subplot",
                description="Oscillator based on A/D Line"
            ),
            'obv': IndicatorRangeConfig(
                name='obv',
                display_name='On Balance Volume',
                range_type=IndicatorRange.VOLUME_BASED,
                typical_range="Cumulative volume",
                subplot_recommendation="volume_subplot",
                description="Volume momentum indicator"
            )
        }
        configs.update(volume_indicators)

        # ========== CUSTOM RANGE INDICATORS ==========
        custom_range_indicators = {
            'macd': IndicatorRangeConfig(
                name='macd',
                display_name='MACD',
                range_type=IndicatorRange.CUSTOM_RANGE,
                typical_range="Oscillating around zero",
                subplot_recommendation="custom_subplot",
                description="Moving Average Convergence Divergence"
            ),
            'macdext': IndicatorRangeConfig(
                name='macdext',
                display_name='MACD Extended',
                range_type=IndicatorRange.CUSTOM_RANGE,
                typical_range="Oscillating around zero",
                subplot_recommendation="custom_subplot",
                description="MACD with configurable MA types"
            ),
            'macdfix': IndicatorRangeConfig(
                name='macdfix',
                display_name='MACD Fixed',
                range_type=IndicatorRange.CUSTOM_RANGE,
                typical_range="Oscillating around zero",
                subplot_recommendation="custom_subplot",
                description="MACD with fixed 12/26 periods"
            ),
            'ppo': IndicatorRangeConfig(
                name='ppo',
                display_name='Percentage Price Oscillator',
                range_type=IndicatorRange.CUSTOM_RANGE,
                typical_range="Percentage oscillation",
                subplot_recommendation="custom_subplot",
                description="Percentage version of MACD"
            ),
            'apo': IndicatorRangeConfig(
                name='apo',
                display_name='Absolute Price Oscillator',
                range_type=IndicatorRange.CUSTOM_RANGE,
                typical_range="Price difference",
                subplot_recommendation="custom_subplot",
                description="Absolute difference version of MACD"
            ),
            'cci': IndicatorRangeConfig(
                name='cci',
                display_name='Commodity Channel Index',
                range_type=IndicatorRange.UNBOUNDED,
                typical_range="Typically ±100",
                subplot_recommendation="custom_subplot",
                description="Momentum oscillator identifying cyclical trends"
            ),
            'cmo': IndicatorRangeConfig(
                name='cmo',
                display_name='Chande Momentum Oscillator',
                range_type=IndicatorRange.NORMALIZED_NEG1_POS1,
                min_value=-100,
                max_value=100,
                typical_range="-100 to +100",
                subplot_recommendation="normalized_subplot",
                description="Momentum oscillator with fixed range"
            ),
            'roc': IndicatorRangeConfig(
                name='roc',
                display_name='Rate of Change',
                range_type=IndicatorRange.CUSTOM_RANGE,
                typical_range="Percentage change",
                subplot_recommendation="custom_subplot",
                description="Momentum indicator measuring rate of change"
            ),
            'rocp': IndicatorRangeConfig(
                name='rocp',
                display_name='Rate of Change Percentage',
                range_type=IndicatorRange.CUSTOM_RANGE,
                typical_range="Percentage change",
                subplot_recommendation="custom_subplot",
                description="Rate of change as percentage"
            ),
            'rocr': IndicatorRangeConfig(
                name='rocr',
                display_name='Rate of Change Ratio',
                range_type=IndicatorRange.CUSTOM_RANGE,
                typical_range="Ratio around 1.0",
                subplot_recommendation="custom_subplot",
                description="Rate of change as ratio"
            ),
            'mom': IndicatorRangeConfig(
                name='mom',
                display_name='Momentum',
                range_type=IndicatorRange.CUSTOM_RANGE,
                typical_range="Price difference",
                subplot_recommendation="custom_subplot",
                description="Simple momentum indicator"
            )
        }
        configs.update(custom_range_indicators)

        return configs

    def get_subplot_recommendation(self, indicator_name: str) -> str:
        """Get subplot recommendation for an indicator"""
        config = self.range_configs.get(indicator_name.lower())
        if config:
            return config.subplot_recommendation
        return "custom_subplot"  # Default fallback

    def get_indicators_by_subplot(self, subplot_type: str) -> List[str]:
        """Get all indicators recommended for a specific subplot"""
        return [
            name for name, config in self.range_configs.items()
            if config.subplot_recommendation == subplot_type
        ]

    def get_range_info(self, indicator_name: str) -> IndicatorRangeConfig:
        """Get complete range information for an indicator"""
        return self.range_configs.get(indicator_name.lower())

    def get_all_subplot_types(self) -> Set[str]:
        """Get all available subplot types"""
        return {config.subplot_recommendation for config in self.range_configs.values()}


# Singleton instance for easy access
indicator_range_classifier = IndicatorRangeClassifier()