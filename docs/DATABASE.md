# Database Schema Documentation

## Overview

ForexGPT uses SQLite with Alembic migrations for schema versioning. The database supports multi-provider data storage with full backward compatibility.

## Schema Version: 0007 (Multi-Provider Support)

### Migration History

| Version | Description | Date |
|---------|-------------|------|
| 0001 | Initial anchor migration | 2025-09-17 |
| 0002 | Add market_data_ticks | 2025-09-17 |
| 0003 | Fix ticks and constraints | 2025-09-17 |
| 0004 | Add backtesting tables | 2025-09-17 |
| 0005 | Add pattern tables | 2025-09-17 |
| 0006 | Add optimization system | 2025-09-28 |
| 0007 | **Add cTrader multi-provider** | **2025-01-05** |

## Tables

### market_data_candles (Extended in 0007)

OHLCV candlestick data with multi-provider support.

```sql
CREATE TABLE market_data_candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(64) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    ts_utc INTEGER NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT,
    resampled BOOLEAN DEFAULT 0,

    -- Added in 0007
    tick_volume FLOAT,           -- Tick count volume
    real_volume FLOAT,           -- Actual traded volume
    provider_source VARCHAR(32) DEFAULT 'tiingo'
);

-- Indexes
CREATE UNIQUE INDEX ux_market_symbol_tf_ts
    ON market_data_candles(symbol, timeframe, ts_utc);

CREATE INDEX ix_candles_provider
    ON market_data_candles(provider_source);

CREATE INDEX ix_candles_symbol_tf_ts_provider
    ON market_data_candles(symbol, timeframe, ts_utc, provider_source);
```

**Columns**:
- `symbol`: Trading pair (e.g., "EUR/USD")
- `timeframe`: Candle period ("1m", "5m", "1h", "1d")
- `ts_utc`: Timestamp in milliseconds (UTC)
- `open/high/low/close`: OHLC prices
- `volume`: Generic volume (legacy)
- `tick_volume`: Number of ticks in period (cTrader)
- `real_volume`: Actual traded volume in currency (cTrader)
- `provider_source`: Source provider name
- `resampled`: Flag for resampled data

**Usage**:
```python
# Query by provider
df = pd.read_sql(
    "SELECT * FROM market_data_candles WHERE provider_source = 'ctrader'",
    engine
)

# Upsert with provider
data_io.upsert_candles(
    engine, df, symbol="EUR/USD", timeframe="1h"
)
```

### market_depth (New in 0007)

Depth of Market (DOM) / Level 2 data.

```sql
CREATE TABLE market_depth (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(64) NOT NULL,
    ts_utc INTEGER NOT NULL,
    provider VARCHAR(32) NOT NULL,

    -- DOM data (JSON arrays)
    bids JSON NOT NULL,  -- [(price, volume), ...]
    asks JSON NOT NULL,  -- [(price, volume), ...]

    -- Derived metrics
    mid_price FLOAT,
    spread FLOAT,
    imbalance FLOAT,     -- bid_volume / ask_volume ratio

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_depth_symbol_ts_provider
        UNIQUE (symbol, ts_utc, provider)
);

-- Indexes
CREATE INDEX ix_depth_symbol_ts ON market_depth(symbol, ts_utc);
CREATE INDEX ix_depth_provider ON market_depth(provider);
```

**Columns**:
- `bids`: JSON array of [price, volume] tuples (descending price)
- `asks`: JSON array of [price, volume] tuples (ascending price)
- `mid_price`: (best_bid + best_ask) / 2
- `spread`: best_ask - best_bid
- `imbalance`: sum(bid_volumes) / sum(ask_volumes)

**JSON Format**:
```json
{
  "bids": [
    [1.0850, 1000000],  // [price, volume in base currency]
    [1.0849, 500000],
    [1.0848, 750000]
  ],
  "asks": [
    [1.0851, 800000],
    [1.0852, 600000],
    [1.0853, 900000]
  ]
}
```

**Usage**:
```python
# Insert DOM snapshot
depth_data = {
    "symbol": "EUR/USD",
    "ts_utc": int(time.time() * 1000),
    "provider": "ctrader",
    "bids": [[1.0850, 1000000], [1.0849, 500000]],
    "asks": [[1.0851, 800000], [1.0852, 600000]],
    "mid_price": 1.08505,
    "spread": 0.0001,
    "imbalance": 1.875  # (1M + 0.5M) / (0.8M + 0.6M)
}

with engine.begin() as conn:
    conn.execute(
        market_depth_table.insert().values(depth_data)
    )

# Query DOM by time range
dom_df = pd.read_sql(
    """
    SELECT * FROM market_depth
    WHERE symbol = 'EUR/USD'
      AND ts_utc BETWEEN :start AND :end
    ORDER BY ts_utc
    """,
    engine,
    params={"start": start_ms, "end": end_ms}
)
```

### sentiment_data (New in 0007)

Trader sentiment indicators.

```sql
CREATE TABLE sentiment_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(64) NOT NULL,
    ts_utc INTEGER NOT NULL,
    provider VARCHAR(32) NOT NULL,

    long_pct FLOAT NOT NULL,     -- % of traders long
    short_pct FLOAT NOT NULL,    -- % of traders short
    total_traders INTEGER,       -- Total trader count

    confidence FLOAT,            -- Confidence level (0-1)

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_sentiment_symbol_ts_provider
        UNIQUE (symbol, ts_utc, provider)
);

-- Indexes
CREATE INDEX ix_sentiment_symbol_ts ON sentiment_data(symbol, ts_utc);
CREATE INDEX ix_sentiment_provider ON sentiment_data(provider);
```

**Usage**:
```python
# Insert sentiment
sentiment = {
    "symbol": "EUR/USD",
    "ts_utc": int(time.time() * 1000),
    "provider": "ctrader",
    "long_pct": 62.5,
    "short_pct": 37.5,
    "total_traders": 10000,
    "confidence": 0.85
}

# Query recent sentiment
sent_df = pd.read_sql(
    """
    SELECT * FROM sentiment_data
    WHERE symbol = 'EUR/USD'
      AND ts_utc > :since
    ORDER BY ts_utc DESC
    LIMIT 100
    """,
    engine,
    params={"since": since_ms}
)
```

### news_events (New in 0007)

News feed with categorization.

```sql
CREATE TABLE news_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc INTEGER NOT NULL,
    provider VARCHAR(32) NOT NULL,

    title VARCHAR(512) NOT NULL,
    content TEXT,
    url VARCHAR(1024),

    currency VARCHAR(16),        -- e.g., "USD", "EUR"
    impact VARCHAR(16),          -- "high", "medium", "low"
    category VARCHAR(64),        -- e.g., "monetary_policy", "employment"

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_news_title_ts_provider
        UNIQUE (title, ts_utc, provider)
);

-- Indexes
CREATE INDEX ix_news_ts ON news_events(ts_utc);
CREATE INDEX ix_news_currency ON news_events(currency);
CREATE INDEX ix_news_impact ON news_events(impact);
CREATE INDEX ix_news_provider ON news_events(provider);
```

**Usage**:
```python
# Insert news
news = {
    "ts_utc": int(time.time() * 1000),
    "provider": "ctrader",
    "title": "ECB Raises Interest Rates by 50bps",
    "content": "The European Central Bank...",
    "url": "https://example.com/news/123",
    "currency": "EUR",
    "impact": "high",
    "category": "monetary_policy"
}

# Query high-impact EUR news
news_df = pd.read_sql(
    """
    SELECT * FROM news_events
    WHERE currency = 'EUR'
      AND impact = 'high'
      AND ts_utc > :since
    ORDER BY ts_utc DESC
    """,
    engine,
    params={"since": since_ms}
)
```

### economic_calendar (New in 0007)

Economic event calendar.

```sql
CREATE TABLE economic_calendar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id VARCHAR(128) UNIQUE NOT NULL,  -- Provider's event ID
    ts_utc INTEGER NOT NULL,
    provider VARCHAR(32) NOT NULL,

    event_name VARCHAR(256) NOT NULL,
    currency VARCHAR(16) NOT NULL,
    country VARCHAR(64),

    forecast VARCHAR(64),        -- Expected value
    previous VARCHAR(64),        -- Previous value
    actual VARCHAR(64),          -- Actual value (populated after release)

    impact VARCHAR(16),          -- "high", "medium", "low"

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX ix_calendar_ts ON economic_calendar(ts_utc);
CREATE INDEX ix_calendar_currency ON economic_calendar(currency);
CREATE INDEX ix_calendar_impact ON economic_calendar(impact);
CREATE INDEX ix_calendar_provider ON economic_calendar(provider);
```

**Usage**:
```python
# Insert calendar event
event = {
    "event_id": "nfp_2025_01_05",
    "ts_utc": 1735905600000,  # 2025-01-05 08:30 UTC
    "provider": "ctrader",
    "event_name": "Non-Farm Payrolls",
    "currency": "USD",
    "country": "United States",
    "forecast": "200K",
    "previous": "185K",
    "actual": None,  # Not released yet
    "impact": "high"
}

# Update with actual value
with engine.begin() as conn:
    conn.execute(
        """
        UPDATE economic_calendar
        SET actual = :actual, updated_at = CURRENT_TIMESTAMP
        WHERE event_id = :event_id
        """,
        {"actual": "210K", "event_id": "nfp_2025_01_05"}
    )

# Query upcoming high-impact events
cal_df = pd.read_sql(
    """
    SELECT * FROM economic_calendar
    WHERE ts_utc > :now
      AND impact = 'high'
    ORDER BY ts_utc
    LIMIT 10
    """,
    engine,
    params={"now": int(time.time() * 1000)}
)
```

## Alembic Migrations

### Running Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific version
alembic upgrade 0007_add_ctrader_features

# Downgrade one version
alembic downgrade -1

# Downgrade to specific version
alembic downgrade 0006_add_optimization_system
```

### Creating New Migrations

```bash
# Auto-generate migration
alembic revision --autogenerate -m "add_new_feature"

# Manual migration
alembic revision -m "add_new_feature"
```

### Migration 0007 Details

**Upgrade (adds cTrader support)**:
1. Adds columns to `market_data_candles`:
   - `tick_volume` (Float, nullable)
   - `real_volume` (Float, nullable)
   - `provider_source` (String, default='tiingo')
2. Creates new tables:
   - `market_depth`
   - `sentiment_data`
   - `news_events`
   - `economic_calendar`
3. Creates composite indexes for performance
4. Backfills `provider_source='tiingo'` for existing data

**Downgrade (removes cTrader support)**:
1. Drops all new tables
2. Drops new indexes
3. Drops new columns from `market_data_candles`

**Note**: Downgrade is DESTRUCTIVE - all cTrader data will be lost.

## Query Examples

### Multi-Provider Aggregation

```sql
-- Compare prices from different providers
SELECT
    symbol,
    timeframe,
    ts_utc,
    provider_source,
    close,
    ABS(close - LAG(close) OVER (
        PARTITION BY symbol, timeframe, ts_utc
        ORDER BY provider_source
    )) AS price_diff
FROM market_data_candles
WHERE symbol = 'EUR/USD'
  AND timeframe = '1h'
  AND ts_utc BETWEEN :start AND :end
ORDER BY ts_utc, provider_source;
```

### Volume Analysis

```sql
-- Compare tick volume vs real volume
SELECT
    symbol,
    timeframe,
    ts_utc,
    tick_volume,
    real_volume,
    CASE
        WHEN tick_volume > 0
        THEN real_volume / tick_volume
        ELSE NULL
    END AS avg_trade_size
FROM market_data_candles
WHERE provider_source = 'ctrader'
  AND tick_volume IS NOT NULL
  AND real_volume IS NOT NULL
ORDER BY ts_utc DESC;
```

### DOM Analysis

```sql
-- Market depth imbalance
SELECT
    symbol,
    ts_utc,
    mid_price,
    spread,
    imbalance,
    CASE
        WHEN imbalance > 1.5 THEN 'bullish'
        WHEN imbalance < 0.67 THEN 'bearish'
        ELSE 'neutral'
    END AS bias
FROM market_depth
WHERE symbol = 'EUR/USD'
ORDER BY ts_utc DESC
LIMIT 100;
```

### Sentiment Correlation

```sql
-- Sentiment vs price movement
SELECT
    s.symbol,
    s.ts_utc,
    s.long_pct,
    c.close AS price,
    LAG(c.close, 1) OVER (ORDER BY s.ts_utc) AS prev_price,
    (c.close - LAG(c.close, 1) OVER (ORDER BY s.ts_utc)) /
        LAG(c.close, 1) OVER (ORDER BY s.ts_utc) * 100 AS price_change_pct
FROM sentiment_data s
JOIN market_data_candles c
    ON s.symbol = c.symbol
    AND ABS(s.ts_utc - c.ts_utc) < 60000  -- Within 1 minute
WHERE s.symbol = 'EUR/USD'
ORDER BY s.ts_utc;
```

### News Impact Analysis

```sql
-- Price volatility around news events
WITH news_times AS (
    SELECT ts_utc, title, impact
    FROM news_events
    WHERE currency = 'EUR' AND impact = 'high'
),
price_volatility AS (
    SELECT
        ts_utc,
        high - low AS range,
        (high - low) / close * 100 AS range_pct
    FROM market_data_candles
    WHERE symbol = 'EUR/USD' AND timeframe = '1m'
)
SELECT
    n.title,
    n.ts_utc AS news_time,
    AVG(CASE
        WHEN ABS(p.ts_utc - n.ts_utc) < 300000  -- 5 min before/after
        THEN p.range_pct
    END) AS avg_volatility_around_news,
    AVG(CASE
        WHEN ABS(p.ts_utc - n.ts_utc) > 1800000  -- 30 min+ away
        THEN p.range_pct
    END) AS avg_volatility_baseline
FROM news_times n
CROSS JOIN price_volatility p
GROUP BY n.title, n.ts_utc;
```

## Maintenance

### Database Vacuum

```bash
# Manual vacuum
python -m app db vacuum

# Scheduled vacuum (every 7 days)
# Configured in configs/default.yaml:
# refresh_rates.database.vacuum_interval_days: 7
```

### Data Archival

```sql
-- Archive old data (> 1 year)
INSERT INTO market_data_candles_archive
SELECT * FROM market_data_candles
WHERE ts_utc < :cutoff_ts;

DELETE FROM market_data_candles
WHERE ts_utc < :cutoff_ts;
```

### Backup & Restore

```bash
# Backup
sqlite3 data/forex_diffusion.db ".backup data/forex_diffusion_backup.db"

# Restore
sqlite3 data/forex_diffusion.db ".restore data/forex_diffusion_backup.db"
```

## Performance Optimization

### Indexes

All critical query paths are indexed:
- Composite index on (symbol, timeframe, ts_utc, provider_source)
- Individual indexes on provider columns
- Timestamp indexes for range queries

### Query Tips

1. **Always filter by timestamp range**:
   ```sql
   WHERE ts_utc BETWEEN :start AND :end
   ```

2. **Use provider filtering**:
   ```sql
   WHERE provider_source = 'ctrader'
   ```

3. **Limit result sets**:
   ```sql
   ORDER BY ts_utc DESC LIMIT 1000
   ```

4. **Use EXPLAIN QUERY PLAN**:
   ```sql
   EXPLAIN QUERY PLAN
   SELECT * FROM market_data_candles
   WHERE symbol = 'EUR/USD';
   ```

---

**Last Updated**: 2025-01-05
**Schema Version**: 0007
