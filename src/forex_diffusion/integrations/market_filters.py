"""Market Filters - VIX, Sentiment, Volume"""
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

class VixFilter:
    """VIX-based volatility filter"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.enabled = self.config.get('filter_vix_enabled', True)
        self.high_threshold = self.config.get('filter_vix_high_threshold', 30.0)
        self.extreme_threshold = self.config.get('filter_vix_extreme_threshold', 50.0)
        self.high_reduction = self.config.get('filter_vix_high_reduction_pct', 0.5)
        self.extreme_reduction = self.config.get('filter_vix_extreme_reduction_pct', 0.7)
    
    def get_adjustment_factor(self, vix_level: float) -> float:
        """Calculate position size adjustment based on VIX"""
        if not self.enabled:
            return 1.0
        
        if vix_level >= self.extreme_threshold:
            return 1.0 - self.extreme_reduction  # e.g., 0.3 (reduce by 70%)
        elif vix_level >= self.high_threshold:
            return 1.0 - self.high_reduction  # e.g., 0.5 (reduce by 50%)
        else:
            return 1.0  # No adjustment
    
    def is_extreme_volatility(self, vix_level: float) -> bool:
        """Check if VIX indicates extreme volatility"""
        return vix_level >= self.extreme_threshold


class SentimentFilter:
    """Sentiment-based contrarian filter"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.enabled = self.config.get('filter_sentiment_enabled', True)
        self.contrarian_threshold = self.config.get('filter_sentiment_contrarian_threshold', 0.75)
        self.confidence_threshold = self.config.get('filter_sentiment_confidence_threshold', 0.6)
        self.fade_strength = self.config.get('filter_sentiment_fade_strength', 1.0)
    
    def should_fade_crowd(self, sentiment_metrics: Dict) -> bool:
        """Check if sentiment is extreme (contrarian opportunity)"""
        if not self.enabled or not sentiment_metrics:
            return False
        
        sentiment_level = sentiment_metrics.get('sentiment', 0.5)
        confidence = sentiment_metrics.get('confidence', 0.0)
        
        # High confidence + extreme sentiment = fade
        if confidence >= self.confidence_threshold:
            if sentiment_level >= self.contrarian_threshold or sentiment_level <= (1 - self.contrarian_threshold):
                return True
        
        return False
    
    def get_contrarian_multiplier(self, sentiment_metrics: Dict) -> float:
        """Get multiplier for contrarian strategy"""
        if not self.should_fade_crowd(sentiment_metrics):
            return 1.0
        
        return self.fade_strength
    
    def apply_sentiment_filter(self, signal: int, sentiment_metrics: Dict) -> int:
        """Apply sentiment filter (potentially invert signal)"""
        if not self.enabled or not sentiment_metrics:
            return signal
        
        if self.should_fade_crowd(sentiment_metrics):
            sentiment_level = sentiment_metrics.get('sentiment', 0.5)
            
            # If crowd is very bullish and our signal is bullish, consider inverting
            if sentiment_level > self.contrarian_threshold and signal > 0:
                logger.info("Sentiment conflict: Crowd bullish, fading signal")
                return -signal  # Fade the crowd
            elif sentiment_level < (1 - self.contrarian_threshold) and signal < 0:
                logger.info("Sentiment conflict: Crowd bearish, fading signal")
                return -signal  # Fade the crowd
        
        return signal


class VolumeFilter:
    """Volume-based liquidity filter"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.enabled = self.config.get('filter_volume_enabled', True)
        self.obv_period = self.config.get('filter_volume_obv_period', 20)
        self.vwap_period = self.config.get('filter_volume_vwap_period', 50)
        self.spike_threshold = self.config.get('filter_volume_spike_threshold', 2.0)
        self.min_liquidity = self.config.get('filter_volume_min_liquidity_pct', 0.7)
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)
        
        obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        return obv
    
    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume-Weighted Average Price"""
        if 'volume' not in data.columns:
            return data['close']
        
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(self.vwap_period).sum() / data['volume'].rolling(self.vwap_period).sum()
        
        return vwap.fillna(data['close'])
    
    def detect_volume_spike(self, data: pd.DataFrame) -> bool:
        """Detect if current volume is a spike"""
        if 'volume' not in data.columns or len(data) < self.obv_period:
            return False
        
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].tail(self.obv_period).mean()
        
        return current_volume >= (avg_volume * self.spike_threshold)
    
    def is_sufficient_liquidity(self, current_volume: float, avg_volume: float) -> bool:
        """Check if liquidity is sufficient"""
        if not self.enabled:
            return True
        
        return current_volume >= (avg_volume * self.min_liquidity)
