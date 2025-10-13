"""
Centralized data loading utilities for training and inference.

Consolidates duplicate data loading functions from training modules.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd
from loguru import logger


def fetch_candles_from_db(
    symbol: str,
    timeframe: str,
    days_history: int = 90,
    engine_url: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from database for a given symbol and timeframe.

    This is the centralized version consolidating duplicate implementations from:
    - training/train_sklearn.py
    - training/train_sklearn_btalib.py
    - And other modules

    Args:
        symbol: Trading pair symbol (e.g., "EUR/USD")
        timeframe: Timeframe string (e.g., "1m", "5m", "1h")
        days_history: Number of days of historical data to fetch
        engine_url: Optional database URL override

    Returns:
        DataFrame with OHLCV + timestamp data

    Raises:
        RuntimeError: If MarketDataService unavailable or data fetch fails
        ImportError: If MarketDataService module not found
    """
    try:
        from ..services.marketdata import MarketDataService
    except ImportError as e:
        raise RuntimeError(f"MarketDataService not available: {e}") from e

    try:
        if engine_url:
            ms = MarketDataService(database_url=engine_url)
        else:
            ms = MarketDataService()
    except ConnectionError as e:
        raise RuntimeError(f"Database connection failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate MarketDataService: {e}") from e

    try:
        df = ms.get_candles(symbol, timeframe, days_history=days_history)

        if df is None or df.empty:
            raise RuntimeError(
                f"No data returned for {symbol} {timeframe} "
                f"(requested {days_history} days history)"
            )

        logger.debug(
            f"Loaded {len(df)} candles for {symbol} {timeframe} "
            f"({days_history} days)"
        )

        return df

    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch candles for {symbol} {timeframe}: {e}"
        ) from e


def fetch_candles_from_db_recent(
    symbol: str,
    timeframe: str,
    n_bars: int = 200,
    engine_url: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch only recent N bars for inference (optimization).

    More efficient than loading full history when only recent data needed
    for indicator computation in inference.

    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe string
        n_bars: Number of recent bars to fetch
        engine_url: Optional database URL override

    Returns:
        DataFrame with recent N candles
    """
    try:
        from ..services.marketdata import MarketDataService
    except ImportError as e:
        raise RuntimeError(f"MarketDataService not available: {e}") from e

    try:
        if engine_url:
            ms = MarketDataService(database_url=engine_url)
        else:
            ms = MarketDataService()

        # Fetch with limit parameter if supported, otherwise fetch and slice
        df = ms.get_candles(symbol, timeframe, days_history=90)  # Fallback

        if df is None or df.empty:
            raise RuntimeError(f"No data for {symbol} {timeframe}")

        # Return only last n_bars
        df_recent = df.tail(n_bars).copy()

        logger.debug(f"Loaded {len(df_recent)} recent candles for {symbol} {timeframe}")

        return df_recent

    except Exception as e:
        raise RuntimeError(f"Failed to fetch recent candles: {e}") from e
