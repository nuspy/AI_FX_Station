"""
Consolidated Technical Indicators Module

HIGH-001: Single source of truth for all technical indicator calculations.
Replaces duplicate implementations across:
- features/indicators.py
- features/indicators_talib.py  
- features/indicators_btalib.py
- features/pipeline.py
- patterns/primitives.py

This module provides a unified API with optional acceleration via TA-Lib or BTAlib.
Falls back to pure NumPy/Pandas implementations if external libraries unavailable.

Usage:
    >>> from forex_diffusion.features.consolidated_indicators import calculate_indicators
    >>> df = pd.DataFrame(...)  # OHLCV data
    >>> indicators = calculate_indicators(df, ['rsi', 'atr', 'macd'])
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from loguru import logger

# Try importing acceleration libraries
try:
    import talib
    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False

try:
    import ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False

try:
    import btalib
    _HAS_BTALIB = True
except ImportError:
    _HAS_BTALIB = False


# Configuration: which library to prefer (talib > ta > btalib > numpy)
PREFERRED_BACKEND = 'auto'  # auto, talib, ta, btalib, numpy


def _get_backend() -> str:
    """Determine which backend to use based on availability and preference."""
    if PREFERRED_BACKEND != 'auto':
        return PREFERRED_BACKEND
    
    if _HAS_TALIB:
        return 'talib'
    elif _HAS_TA:
        return 'ta'
    elif _HAS_BTALIB:
        return 'btalib'
    else:
        return 'numpy'


# ==============================================================================
# CORE INDICATOR FUNCTIONS (NumPy/Pandas implementations)
# ==============================================================================

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)
    
    Args:
        series: Price series (typically close)
        period: RSI period (default 14)
        
    Returns:
        RSI values (0-100)
    """
    backend = _get_backend()
    
    if backend == 'talib' and _HAS_TALIB:
        try:
            return pd.Series(talib.RSI(series.values, timeperiod=period), index=series.index)
        except Exception as e:
            logger.debug(f"TA-Lib RSI failed: {e}, falling back to numpy")
    
    elif backend == 'ta' and _HAS_TA:
        try:
            return ta.momentum.RSIIndicator(close=series, window=period).rsi()
        except Exception as e:
            logger.debug(f"TA RSI failed: {e}, falling back to numpy")
    
    elif backend == 'btalib' and _HAS_BTALIB:
        try:
            result = btalib.rsi(series, period=period)
            return pd.Series(result.df.iloc[:, 0].values, index=series.index)
        except Exception as e:
            logger.debug(f"BTAlib RSI failed: {e}, falling back to numpy")
    
    # NumPy fallback (Wilder's smoothing)
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1.0 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1.0/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values.fillna(50.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Average True Range (ATR)
    
    Consolidated from multiple implementations - this is the canonical version.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        n: Period for ATR calculation (default 14)
        
    Returns:
        Series of ATR values
    """
    backend = _get_backend()
    
    if backend == 'talib' and _HAS_TALIB:
        try:
            return pd.Series(
                talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=n),
                index=df.index
            )
        except Exception as e:
            logger.debug(f"TA-Lib ATR failed: {e}, falling back to numpy")
    
    elif backend == 'ta' and _HAS_TA:
        try:
            return ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=n
            ).average_true_range()
        except Exception as e:
            logger.debug(f"TA ATR failed: {e}, falling back to numpy")
    
    # NumPy fallback
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    prev_close = np.roll(close, 1)

    # True Range: max of (high-low, |high-prev_close|, |low-prev_close|)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )
    )
    tr[0] = high[0] - low[0]  # First TR is just high-low

    # ATR is EMA of TR
    atr_series = pd.Series(tr, index=df.index).ewm(span=n, adjust=False).mean()
    return atr_series


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD)
    
    Args:
        series: Price series (typically close)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        
    Returns:
        Dictionary with 'macd', 'signal', 'hist' keys
    """
    backend = _get_backend()
    
    if backend == 'talib' and _HAS_TALIB:
        try:
            macd_line, signal_line, hist = talib.MACD(
                series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            return {
                "macd": pd.Series(macd_line, index=series.index),
                "signal": pd.Series(signal_line, index=series.index),
                "hist": pd.Series(hist, index=series.index)
            }
        except Exception as e:
            logger.debug(f"TA-Lib MACD failed: {e}, falling back to numpy")
    
    elif backend == 'ta' and _HAS_TA:
        try:
            macd_indicator = ta.trend.MACD(
                close=series, window_fast=fast, window_slow=slow, window_sign=signal
            )
            return {
                "macd": macd_indicator.macd(),
                "signal": macd_indicator.macd_signal(),
                "hist": macd_indicator.macd_diff()
            }
        except Exception as e:
            logger.debug(f"TA MACD failed: {e}, falling back to numpy")
    
    # NumPy fallback
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands
    
    Args:
        series: Price series (typically close)
        window: Rolling window for SMA and std (default 20)
        n_std: Number of standard deviations for bands (default 2.0)
        
    Returns:
        Tuple of (middle, upper, lower) bands
    """
    backend = _get_backend()
    
    if backend == 'talib' and _HAS_TALIB:
        try:
            upper, middle, lower = talib.BBANDS(
                series.values, timeperiod=window, nbdevup=n_std, nbdevdn=n_std
            )
            return (
                pd.Series(middle, index=series.index),
                pd.Series(upper, index=series.index),
                pd.Series(lower, index=series.index)
            )
        except Exception as e:
            logger.debug(f"TA-Lib Bollinger failed: {e}, falling back to numpy")
    
    elif backend == 'ta' and _HAS_TA:
        try:
            bb = ta.volatility.BollingerBands(close=series, window=window, window_dev=n_std)
            return (
                bb.bollinger_mavg(),
                bb.bollinger_hband(),
                bb.bollinger_lband()
            )
        except Exception as e:
            logger.debug(f"TA Bollinger failed: {e}, falling back to numpy")
    
    # NumPy fallback
    sma_series = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std().fillna(0.0)
    upper = sma_series + n_std * std
    lower = sma_series - n_std * std
    return sma_series, upper, lower


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        k_period: %K period (default 14)
        d_period: %D smoothing period (default 3)
        
    Returns:
        Tuple of (%K, %D) series
    """
    backend = _get_backend()
    
    if backend == 'talib' and _HAS_TALIB:
        try:
            k, d = talib.STOCH(
                df['high'].values, df['low'].values, df['close'].values,
                fastk_period=k_period, slowk_period=d_period, slowd_period=d_period
            )
            return pd.Series(k, index=df.index), pd.Series(d, index=df.index)
        except Exception as e:
            logger.debug(f"TA-Lib Stochastic failed: {e}, falling back to numpy")
    
    # NumPy fallback
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(window=d_period).mean()
    return k.fillna(50.0), d.fillna(50.0)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX)
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ADX period (default 14)
        
    Returns:
        ADX values (0-100)
    """
    backend = _get_backend()
    
    if backend == 'talib' and _HAS_TALIB:
        try:
            return pd.Series(
                talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=period),
                index=df.index
            )
        except Exception as e:
            logger.debug(f"TA-Lib ADX failed: {e}, falling back to numpy")
    
    elif backend == 'ta' and _HAS_TA:
        try:
            return ta.trend.ADXIndicator(
                high=df['high'], low=df['low'], close=df['close'], window=period
            ).adx()
        except Exception as e:
            logger.debug(f"TA ADX failed: {e}, falling back to numpy")
    
    # NumPy fallback (simplified - full DMI calculation is complex)
    tr = atr(df, period)
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / tr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_values = dx.ewm(span=period, adjust=False).mean()
    
    return adx_values.fillna(0.0)


# ==============================================================================
# HIGH-LEVEL API
# ==============================================================================

def calculate_indicators(
    df: pd.DataFrame,
    indicators: Union[List[str], Dict[str, Dict[str, Any]]],
    prefix: str = ""
) -> pd.DataFrame:
    """
    Calculate multiple indicators at once.
    
    HIGH-001: Consolidated indicator calculation with unified API.
    
    Args:
        df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        indicators: Either:
            - List of indicator names: ['rsi', 'atr', 'macd']
            - Dict with parameters: {'rsi': {'period': 14}, 'atr': {'n': 14}}
        prefix: Optional prefix for column names (e.g., "5m_")
        
    Returns:
        DataFrame with original data plus indicator columns
        
    Example:
        >>> df = pd.DataFrame(...)  # OHLCV data
        >>> result = calculate_indicators(df, ['rsi', 'atr', 'macd'])
        >>> print(result.columns)
        # ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'atr_14', 'macd', 'macd_signal', 'macd_hist']
    """
    result = df.copy()
    
    # Convert list to dict with default params
    if isinstance(indicators, list):
        indicators = {name: {} for name in indicators}
    
    for name, params in indicators.items():
        name_lower = name.lower()
        
        try:
            if name_lower == 'rsi':
                period = params.get('period', params.get('n', 14))
                result[f'{prefix}rsi_{period}'] = rsi(df['close'], period=period)
                
            elif name_lower == 'atr':
                n = params.get('n', params.get('period', 14))
                result[f'{prefix}atr_{n}'] = atr(df, n=n)
                
            elif name_lower == 'macd':
                fast = params.get('fast', 12)
                slow = params.get('slow', 26)
                signal_period = params.get('signal', 9)
                macd_result = macd(df['close'], fast=fast, slow=slow, signal=signal_period)
                result[f'{prefix}macd'] = macd_result['macd']
                result[f'{prefix}macd_signal'] = macd_result['signal']
                result[f'{prefix}macd_hist'] = macd_result['hist']
                
            elif name_lower == 'bollinger':
                window = params.get('window', params.get('n', 20))
                n_std = params.get('n_std', params.get('dev', 2.0))
                middle, upper, lower = bollinger(df['close'], window=window, n_std=n_std)
                result[f'{prefix}bb_middle'] = middle
                result[f'{prefix}bb_upper'] = upper
                result[f'{prefix}bb_lower'] = lower
                
            elif name_lower == 'sma':
                window = params.get('window', params.get('period', 20))
                result[f'{prefix}sma_{window}'] = sma(df['close'], window=window)
                
            elif name_lower == 'ema':
                span = params.get('span', params.get('period', 20))
                result[f'{prefix}ema_{span}'] = ema(df['close'], span=span)
                
            elif name_lower == 'stochastic':
                k_period = params.get('k_period', 14)
                d_period = params.get('d_period', 3)
                k, d = stochastic(df, k_period=k_period, d_period=d_period)
                result[f'{prefix}stoch_k'] = k
                result[f'{prefix}stoch_d'] = d
                
            elif name_lower == 'adx':
                period = params.get('period', 14)
                result[f'{prefix}adx_{period}'] = adx(df, period=period)
                
            else:
                logger.warning(f"Unknown indicator: {name}")
                
        except Exception as e:
            logger.error(f"Failed to calculate {name}: {e}")
    
    return result


def get_available_indicators() -> List[str]:
    """Get list of available indicator names."""
    return ['rsi', 'atr', 'macd', 'bollinger', 'sma', 'ema', 'stochastic', 'adx']


def get_backend_info() -> Dict[str, Any]:
    """Get information about available backends."""
    return {
        'active_backend': _get_backend(),
        'available_backends': {
            'talib': _HAS_TALIB,
            'ta': _HAS_TA,
            'btalib': _HAS_BTALIB,
            'numpy': True  # Always available
        },
        'preferred_backend': PREFERRED_BACKEND
    }
