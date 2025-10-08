"""
Pytest Configuration and Fixtures for ForexGPT E2E Tests.

This module provides shared fixtures and configuration for the E2E test suite.
"""

from __future__ import annotations

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Generator, Dict, Any
from unittest.mock import Mock, MagicMock
import sqlite3

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.e2e_utils import (
    E2ETestConfig,
    DatabaseValidator,
    PerformanceMonitor,
    setup_test_logger
)


def make_trending_series(n=300, slope=0.01, noise=0.05, seed=123):
    """Original helper for generating test data."""
    rng = np.random.default_rng(seed)
    base = np.arange(n)*slope + rng.normal(0, noise, size=n)
    high = base + rng.uniform(0.05,0.1,size=n)
    low  = base - rng.uniform(0.05,0.1,size=n)
    open_ = base + rng.normal(0, noise/2, size=n)
    close = base + rng.normal(0, noise/2, size=n)
    time = pd.date_range("2020-01-01", periods=n, freq="H")
    return pd.DataFrame({"time":time,"open":open_,"high":high,"low":low,"close":close,"volume":1.0})


# ==================== E2E Test Fixtures ====================

@pytest.fixture(scope="session")
def e2e_config() -> E2ETestConfig:
    """Provide E2E test configuration."""
    config = E2ETestConfig()
    setup_test_logger(config)
    return config


@pytest.fixture(scope="session")
def db_validator(e2e_config: E2ETestConfig) -> Generator[DatabaseValidator, None, None]:
    """Provide database validator."""
    validator = DatabaseValidator(e2e_config.db_path)
    validator.connect()
    yield validator
    validator.close()


@pytest.fixture(scope="session")
def performance_monitor() -> PerformanceMonitor:
    """Provide performance monitor."""
    return PerformanceMonitor()


@pytest.fixture(scope="function")
def clean_database(db_validator: DatabaseValidator) -> Generator[None, None, None]:
    """Provide clean database for tests."""
    # Clear all tables before test
    results = db_validator.clear_all_tables()
    yield
    # Optionally clean up after test
    # db_validator.clear_all_tables()


@pytest.fixture(scope="session")
def sample_ohlc_data() -> pd.DataFrame:
    """Provide sample OHLC data for testing."""
    n = 1000
    ts_start = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
    ts_interval = 60_000  # 1 minute

    data = {
        'ts_utc': [ts_start + i * ts_interval for i in range(n)],
        'open': np.random.uniform(1.0800, 1.0900, n),
        'high': np.random.uniform(1.0900, 1.1000, n),
        'low': np.random.uniform(1.0700, 1.0800, n),
        'close': np.random.uniform(1.0800, 1.0900, n),
        'volume': np.random.uniform(100, 1000, n)
    }

    df = pd.DataFrame(data)

    # Ensure OHLC consistency
    for i in range(len(df)):
        df.loc[i, 'high'] = max(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'high'])
        df.loc[i, 'low'] = min(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'low'])

    return df


@pytest.fixture(scope="session")
def mock_broker_service() -> Mock:
    """Provide mock broker service for testing without real API calls."""
    mock = Mock()

    # Mock connection
    mock.connect = Mock(return_value=True)
    mock.is_connected = Mock(return_value=True)
    mock.disconnect = Mock(return_value=True)

    # Mock market data
    mock.get_current_price = Mock(return_value={'bid': 1.0850, 'ask': 1.0852})
    mock.get_historical_data = Mock(return_value=pd.DataFrame({
        'ts_utc': [1000000, 1001000, 1002000],
        'open': [1.08, 1.081, 1.082],
        'high': [1.082, 1.083, 1.084],
        'low': [1.079, 1.080, 1.081],
        'close': [1.081, 1.082, 1.083],
        'volume': [100, 150, 120]
    }))

    # Mock order placement
    mock.place_order = Mock(return_value={
        'order_id': 'TEST-001',
        'status': 'filled',
        'filled_price': 1.0851
    })
    mock.get_open_positions = Mock(return_value=[])
    mock.close_position = Mock(return_value={'status': 'closed'})

    return mock


@pytest.fixture(scope="session")
def mock_ctrader_provider() -> Mock:
    """Provide mock cTrader provider for testing."""
    mock = Mock()

    # Connection methods
    mock.test_connection = Mock(return_value=True)
    mock.connect = Mock(return_value=True)
    mock.is_connected = Mock(return_value=True)

    # API quota
    mock.get_api_quota = Mock(return_value={
        'remaining_calls': 5000,
        'reset_time': int((datetime.now() + timedelta(hours=1)).timestamp())
    })

    # Historical data download
    def mock_download(symbol, timeframe, start_ts, end_ts):
        n_candles = min(1000, int((end_ts - start_ts) / 60000))  # 1min candles
        ts_values = np.linspace(start_ts, end_ts, n_candles, dtype=int)

        return pd.DataFrame({
            'ts_utc': ts_values,
            'open': np.random.uniform(1.08, 1.09, n_candles),
            'high': np.random.uniform(1.09, 1.10, n_candles),
            'low': np.random.uniform(1.07, 1.08, n_candles),
            'close': np.random.uniform(1.08, 1.09, n_candles),
            'volume': np.random.uniform(50, 500, n_candles),
            'spread': np.random.uniform(0.0001, 0.0003, n_candles)
        })

    mock.download_historical = Mock(side_effect=mock_download)

    # Real-time data
    mock.subscribe_ticks = Mock(return_value=True)
    mock.unsubscribe_ticks = Mock(return_value=True)

    return mock


@pytest.fixture(scope="function")
def mock_trading_engine() -> Mock:
    """Provide mock trading engine for integrated testing."""
    mock = Mock()

    # Position management
    mock.open_position = Mock(return_value={
        'position_id': 'POS-001',
        'status': 'open',
        'entry_price': 1.0851,
        'sl': 1.0831,
        'tp': 1.0901
    })
    mock.close_position = Mock(return_value={'status': 'closed', 'pnl': 50.0})
    mock.get_positions = Mock(return_value=[])

    # Risk management
    mock.calculate_position_size = Mock(return_value=0.1)  # 0.1 lots
    mock.validate_risk = Mock(return_value=True)
    mock.get_portfolio_risk = Mock(return_value={'total_risk_pct': 2.5})

    # Signal generation
    mock.generate_signal = Mock(return_value={
        'direction': 'BUY',
        'strength': 0.75,
        'entry_price': 1.0851,
        'sl': 1.0831,
        'tp': 1.0901
    })

    return mock


@pytest.fixture(scope="function")
def test_database_with_tables(db_validator: DatabaseValidator, tmp_path: Path) -> Generator[str, None, None]:
    """
    Provide a temporary test database with all required tables created.
    """
    # Create a temporary database
    test_db_path = str(tmp_path / "test_market.db")

    # Connect and create schema
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()

    # Create candles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(64) NOT NULL,
            timeframe VARCHAR(32) NOT NULL,
            ts_utc BIGINT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL,
            spread REAL,
            tick_volume INTEGER,
            volatility REAL,
            UNIQUE(symbol, timeframe, ts_utc)
        )
    """)

    # Create market_data_ticks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_data_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(64) NOT NULL,
            timeframe VARCHAR(32) NOT NULL,
            ts_utc BIGINT NOT NULL,
            price REAL,
            bid REAL,
            ask REAL,
            volume REAL,
            ts_created_ms BIGINT,
            UNIQUE(symbol, timeframe, ts_utc)
        )
    """)

    conn.commit()
    conn.close()

    yield test_db_path

    # Cleanup is automatic with tmp_path


@pytest.fixture(scope="session")
def test_symbols() -> list[str]:
    """Provide list of test symbols."""
    return ["EURUSD", "GBPUSD", "USDJPY"]


@pytest.fixture(scope="session")
def test_timeframes() -> list[str]:
    """Provide list of test timeframes."""
    return ["1m", "5m", "15m", "1h", "4h", "1d"]


@pytest.fixture(scope="function")
def phase_timer() -> Generator[Dict[str, Any], None, None]:
    """Track phase execution times."""
    timer = {'start': datetime.now(), 'phases': {}}

    def record_phase(phase_name: str):
        timer['phases'][phase_name] = {
            'start': datetime.now(),
            'duration': None
        }

    timer['record'] = record_phase
    yield timer


def pytest_addoption(parser):
    """Add custom pytest command line options."""
    parser.addoption(
        "--enable-concurrency",
        action="store_true",
        default=False,
        help="Enable concurrency testing mode"
    )


@pytest.fixture(scope="session")
def concurrency_enabled(request) -> bool:
    """Check if concurrency mode is enabled."""
    return request.config.getoption("--enable-concurrency")
