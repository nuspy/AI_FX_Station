# ForexGPT Multi-Provider Architecture

## Overview

ForexGPT implements a flexible multi-provider architecture for market data acquisition, supporting multiple data sources (Tiingo, cTrader, AlphaVantage) with unified interfaces and automatic failover.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         GUI Layer                            │
│  (FinPlot Charts, News/Calendar Tabs, Sentiment Widget)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Provider Manager                            │
│  - Factory pattern for provider instantiation                │
│  - Health monitoring & automatic failover                    │
│  - Provider capability routing                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬─────────────┐
        │              │               │             │
┌───────▼──────┐ ┌────▼─────┐ ┌───────▼─────┐ ┌────▼──────┐
│   Tiingo     │ │ cTrader  │ │AlphaVantage │ │  Future   │
│  Provider    │ │ Provider │ │  Provider   │ │ Providers │
└───────┬──────┘ └────┬─────┘ └───────┬─────┘ └───────────┘
        │              │               │
┌───────▼──────────────▼───────────────▼──────────────────────┐
│                 BaseProvider Interface                       │
│  - Capability declarations (QUOTES, BARS, TICKS, DOM, etc.) │
│  - Async methods for data retrieval                         │
│  - Health monitoring                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Data Pipeline                              │
│  - WebSocket streams (Twisted→asyncio bridge)               │
│  - REST API polling                                         │
│  - Aggregators (Candles, DOM, Sentiment)                   │
│  - Worker threads                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                Storage Layer                                 │
│  - SQLite database (Alembic migrations)                     │
│  - LRU cache (RAM) for real-time data                      │
│  - Credential storage (OS keyring + Fernet encryption)      │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Provider System

#### BaseProvider (Abstract)
- **Location**: `src/forex_diffusion/providers/base.py`
- **Purpose**: Abstract interface for all market data providers
- **Key Methods**:
  - `connect()` / `disconnect()`: Connection lifecycle
  - `get_current_price()`: Current quotes
  - `get_historical_bars()`: OHLCV data
  - `get_market_depth()`: DOM/Level 2
  - `stream_quotes()`: Real-time WebSocket streaming
  - `get_sentiment()`, `get_news()`, `get_economic_calendar()`: Supplementary data

#### ProviderCapability (Enum)
Defines supported features:
- `QUOTES`, `BARS`, `TICKS`, `VOLUMES`
- `DOM` (Depth of Market)
- `SENTIMENT`, `NEWS`, `CALENDAR`
- `WEBSOCKET`
- `HISTORICAL_BARS`, `HISTORICAL_TICKS`

#### TiingoProvider
- **Capabilities**: QUOTES, BARS, HISTORICAL_BARS, WEBSOCKET
- **Implementation**: Wraps existing `TiingoClient` and `TiingoWSConnector`
- **Data Flow**: WebSocket → asyncio.Queue → Stream

#### CTraderProvider
- **Capabilities**: Full support (QUOTES, BARS, TICKS, VOLUMES, DOM, SENTIMENT, NEWS, CALENDAR, WEBSOCKET, HISTORICAL_BARS, HISTORICAL_TICKS)
- **Implementation**: Twisted → asyncio bridge with Protobuf messages
- **Rate Limiting**: 5 req/sec for historical data

### 2. Credentials Management

#### CredentialsManager
- **Location**: `src/forex_diffusion/credentials/manager.py`
- **Storage**: OS-level keyring (Windows Credential Manager, macOS Keychain, Linux Secret Service)
- **Encryption**: Fernet symmetric encryption (cryptography library)
- **Key Methods**:
  - `save(credentials)`: Store encrypted credentials
  - `load(provider_name)`: Retrieve and decrypt
  - `delete(provider_name)`: Remove credentials
  - `update_token()`: Update OAuth tokens

#### OAuth2Flow
- **Location**: `src/forex_diffusion/credentials/oauth.py`
- **Flow**: Authorization Code Flow with PKCE
- **Implementation**:
  1. Generate authorization URL
  2. Start localhost HTTP server (port 5000)
  3. Open browser for user authorization
  4. Capture authorization code from callback
  5. Exchange code for access/refresh tokens
- **Security**: CSRF protection with state parameter

### 3. Configuration

#### YAML Configuration
- **Location**: `configs/default.yaml`
- **Structure**:
  ```yaml
  providers:
    default: "tiingo"
    secondary: "ctrader"  # Failover

    tiingo:
      enabled: true
      key: "${TIINGO_API_KEY}"

    ctrader:
      enabled: false
      environment: "demo"  # or "live"
      client_id: "${CTRADER_CLIENT_ID}"
      client_secret: "${CTRADER_CLIENT_SECRET}"

  refresh_rates:
    rest:
      news_feed: 300       # 5 minutes
      economic_calendar: 21600  # 6 hours
      sentiment: 30        # 30 seconds

    websocket:
      reconnect_backoff: [1, 2, 4, 8, 16]
      heartbeat_interval: 30

  data_sources:
    quotes:
      primary: "ctrader"
      fallback: "tiingo"
  ```

### 4. Database Schema

#### Extended Tables (Migration 0007)

**market_data_candles** (extended):
- `tick_volume` (Float): Tick count volume
- `real_volume` (Float): Actual traded volume
- `provider_source` (String): Source provider name

**market_depth** (new):
- DOM/Level 2 data with bids/asks
- Derived metrics: mid_price, spread, imbalance

**sentiment_data** (new):
- Trader sentiment: long_pct, short_pct, total_traders

**news_events** (new):
- News feed with title, content, currency, impact

**economic_calendar** (new):
- Economic events with forecast, actual, previous values

### 5. Data Flow

#### Real-Time Flow (WebSocket)
```
Provider WebSocket
  │
  ▼
Twisted Callback (sync)
  │
  ▼
asyncio.Queue
  │
  ▼
Stream Consumer (async)
  │
  ▼
Aggregator
  │
  ▼
Database (bulk insert)
  │
  ▼
GUI Update (via signals)
```

#### Historical Flow (REST)
```
MarketDataService.backfill()
  │
  ▼
ProviderManager.get_primary_provider()
  │
  ▼
Provider.get_historical_bars()
  │
  ▼
Rate Limiter
  │
  ▼
Validation & QA
  │
  ▼
Database Upsert
  │
  ▼
Feature Computation (optional)
```

#### Failover Flow
```
Primary Provider Error
  │
  ▼
Health Check Fails
  │
  ▼
ProviderManager.failover_to_secondary()
  │
  ▼
Connect to Secondary
  │
  ▼
Swap Primary ↔ Secondary
  │
  ▼
Continue Operations
```

## Key Design Patterns

### 1. Strategy Pattern
- `BaseProvider` defines interface
- Each provider implements strategy
- Runtime provider selection via `ProviderManager`

### 2. Factory Pattern
- `ProviderManager.create_provider(name, config)`
- Centralized instantiation
- Configuration injection

### 3. Observer Pattern
- Providers emit health events
- GUI observes provider health
- Automatic UI updates on status changes

### 4. Bridge Pattern
- Twisted (sync callbacks) → asyncio (async/await)
- WebSocket messages → asyncio.Queue → AsyncIterator

## Performance Optimizations

### 1. Caching Strategy
- **RAM Cache (LRU)**:
  - News feed (last 100 items, 24h TTL)
  - DOM updates (circular buffer, 1000 items)
  - Sentiment ticks (buffer 500 items)
  - WebSocket messages (debug, 1000 items)

- **Database**:
  - OHLCV bars (persistent)
  - Tick data (configurable)
  - Market depth snapshots (every 10 seconds)
  - Sentiment snapshots (every minute)

### 2. Worker Threads
- **WebSocket Thread**: Receives streams, pushes to queue
- **Aggregator Thread**: Consumes queue, creates candles, saves DB
- **Backfill Thread**: Detects gaps, requests historical data
- **Housekeeping Thread**: Cleans cache, archives old data

### 3. Rate Limiting
- cTrader: 5 req/sec (historical data)
- Exponential backoff on errors: [1, 2, 4, 8, 16] seconds
- Queue-based request throttling

## Security

### 1. Credential Storage
- **Never** store plaintext credentials
- OS-level keyring (platform-specific secure storage)
- Fernet encryption (AES-128-CBC with HMAC)
- Auto-generated encryption key per installation

### 2. OAuth Security
- State parameter for CSRF protection
- Localhost callback (no external redirect)
- Token refresh before expiration
- Secure token storage in keyring

### 3. API Key Protection
- Environment variable override support
- No API keys in version control
- Masked logging (show only last 4 chars)

## Error Handling

### 1. Network Errors
- Retry with exponential backoff
- Max 5 attempts per request
- Automatic failover to secondary provider
- Health monitoring with error rate tracking

### 2. Data Quality
- Validation pipeline for all incoming data
- Outlier detection (Z-score > 8)
- Gap detection and auto-backfill
- Duplicate removal

### 3. Provider Failures
- Health checks every 30 seconds
- Automatic reconnection
- Graceful degradation (disable unavailable features)
- User notification via GUI

## Extension Points

### Adding New Providers

1. **Create Provider Class**:
   ```python
   class MyProvider(BaseProvider):
       @property
       def capabilities(self):
           return [ProviderCapability.QUOTES, ...]

       async def _get_historical_bars_impl(self, ...):
           # Implementation
   ```

2. **Register in ProviderManager**:
   ```python
   # In src/forex_diffusion/providers/manager.py
   self._provider_classes = {
       "tiingo": TiingoProvider,
       "ctrader": CTraderProvider,
       "myprovider": MyProvider,  # Add here
   }
   ```

3. **Add Configuration**:
   ```yaml
   # In configs/default.yaml
   providers:
     myprovider:
       enabled: false
       api_key: "${MYPROVIDER_KEY}"
   ```

4. **Update Credentials** (if needed):
   ```python
   # Store credentials
   creds = ProviderCredentials(
       provider_name="myprovider",
       api_key="your_key",
   )
   credentials_manager.save(creds)
   ```

## Testing Strategy

### 1. Unit Tests
- Mock provider responses
- Test capability routing
- Validate data normalization
- Test error handling

### 2. Integration Tests
- Full pipeline: WebSocket → Aggregator → DB → GUI
- Provider failover scenarios
- Backfill with existing data (no duplicates)
- OAuth flow with mock server

### 3. Performance Tests
- Throughput: ticks/sec handling
- Memory leak detection (24h runs)
- Database growth rate estimation
- GUI responsiveness under load (10k msg/sec)

## Deployment

### Dependencies
```bash
pip install ctrader-open-api twisted protobuf keyring cryptography httpx
```

### Database Migration
```bash
alembic upgrade head  # Apply migration 0007
```

### Configuration
1. Copy `configs/default.yaml` to `configs/production.yaml`
2. Set environment variables or update YAML
3. Run setup wizard: `python -m app providers add ctrader`

## Future Enhancements

1. **Additional Providers**: Interactive Brokers, MetaTrader 5
2. **Advanced Features**: Order execution, portfolio management
3. **Machine Learning**: Anomaly detection, sentiment analysis
4. **Distributed Architecture**: Redis pub/sub for multi-instance deployment
5. **Cloud Deployment**: AWS Lambda for serverless data ingestion

---

**Last Updated**: 2025-01-05
**Version**: 2.0.0 (Multi-Provider Architecture)
