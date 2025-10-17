"""
Centralized data loading utilities for training and inference.

Consolidates duplicate data loading functions from training modules.
"""
from __future__ import annotations

import time
from typing import Optional, Callable, TypeVar
from functools import wraps
import pandas as pd
from loguru import logger

T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}: {e}"
                        )

            # If we get here, all retries failed
            raise last_exception

        return wrapper
    return decorator


@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    exceptions=(ConnectionError, TimeoutError, RuntimeError)
)
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

    BUG-002: Enhanced with retry logic and exponential backoff for transient failures.
    Retries up to 3 times with delays of 1s, 2s, 4s on ConnectionError, TimeoutError, or RuntimeError.

    Args:
        symbol: Trading pair symbol (e.g., "EUR/USD")
        timeframe: Timeframe string (e.g., "1m", "5m", "1h")
        days_history: Number of days of historical data to fetch
        engine_url: Optional database URL override

    Returns:
        DataFrame with OHLCV + timestamp data

    Raises:
        RuntimeError: If MarketDataService unavailable or data fetch fails after retries
        ImportError: If MarketDataService module not found (no retry)
    """
    try:
        from ..services.marketdata import MarketDataService
    except ImportError as e:
        # Don't retry import errors - these are permanent failures
        raise RuntimeError(f"MarketDataService not available: {e}") from e

    try:
        if engine_url:
            ms = MarketDataService(database_url=engine_url)
        else:
            ms = MarketDataService()
    except ConnectionError as e:
        logger.error(f"Database connection failed: {e}")
        raise  # Will be caught by retry decorator
    except Exception as e:
        logger.error(f"Failed to instantiate MarketDataService: {e}")
        raise RuntimeError(f"Failed to instantiate MarketDataService: {e}") from e

    try:
        logger.info(f"Fetching {days_history} days of {symbol} {timeframe} data from database...")
        df = ms.get_candles(symbol, timeframe, days_history=days_history)

        if df is None or df.empty:
            raise RuntimeError(
                f"No data returned for {symbol} {timeframe} "
                f"(requested {days_history} days history)"
            )

        logger.info(
            f"✓ Loaded {len(df)} candles for {symbol} {timeframe} "
            f"({days_history} days)"
        )

        return df

    except Exception as e:
        logger.error(f"Failed to fetch candles for {symbol} {timeframe}: {e}")
        raise RuntimeError(
            f"Failed to fetch candles for {symbol} {timeframe}: {e}"
        ) from e


@retry_with_backoff(
    max_retries=3,
    initial_delay=0.5,
    backoff_factor=2.0,
    exceptions=(ConnectionError, TimeoutError, RuntimeError)
)
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

    BUG-002: Enhanced with retry logic (shorter delays for inference use case).

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
        logger.debug(f"Fetching recent {n_bars} bars for {symbol} {timeframe}...")
        
        # Query DB directly
        from sqlalchemy import select
        from ..data.schema import MarketDataCandle
        from sqlalchemy.orm import Session
        
        with Session(ms.engine) as session:
            query = (
                select(MarketDataCandle)
                .where(
                    MarketDataCandle.symbol == symbol,
                    MarketDataCandle.timeframe == timeframe
                )
                .order_by(MarketDataCandle.timestamp_ms.desc())
                .limit(n_bars)
            )
            candles = session.execute(query).scalars().all()
        
        if not candles:
            raise RuntimeError(f"No data for {symbol} {timeframe}")
        
        # Convert to DataFrame
        import pandas as pd
        df_recent = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(c.timestamp_ms, unit='ms', utc=True),
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            }
            for c in reversed(candles)
        ])
        df_recent = df_recent.set_index('timestamp')

        logger.debug(f"✓ Loaded {len(df_recent)} recent candles for {symbol} {timeframe}")

        return df_recent

    except Exception as e:
        logger.error(f"Failed to fetch recent candles: {e}")
        raise RuntimeError(f"Failed to fetch recent candles: {e}") from e
