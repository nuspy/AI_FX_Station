# Provider Guide

## Overview

ForexGPT supports multiple market data providers through a unified interface. This document explains how to configure, use, and extend providers.

## Supported Providers

### 1. Tiingo (Default)
- **Type**: REST + WebSocket
- **Capabilities**: Quotes, Historical Bars, Real-time Streaming
- **Setup**:
  ```bash
  export TIINGO_API_KEY="your_api_key"
  ```
- **Free Tier**: Yes (limited)
- **Rate Limits**: ~50 req/min

### 2. cTrader Open API
- **Type**: WebSocket + REST
- **Capabilities**: Full (Quotes, Bars, Ticks, DOM, Sentiment, News, Calendar)
- **Setup**: OAuth 2.0 (see below)
- **Free Tier**: Demo account
- **Rate Limits**: 5 req/sec (historical data)

### 3. AlphaVantage
- **Type**: REST only
- **Capabilities**: Historical Bars
- **Setup**:
  ```bash
  export ALPHAVANTAGE_KEY="your_api_key"
  ```
- **Free Tier**: Yes (5 req/min)
- **Rate Limits**: Very restrictive

## Configuration

### YAML Configuration

Edit `configs/default.yaml`:

```yaml
providers:
  default: "tiingo"      # Primary provider
  secondary: "ctrader"   # Failover provider

  tiingo:
    enabled: true
    key: "${TIINGO_API_KEY}"
    ws_uri: "wss://api.tiingo.com/fx"

  ctrader:
    enabled: false
    environment: "demo"  # or "live"
    client_id: "${CTRADER_CLIENT_ID}"
    client_secret: "${CTRADER_CLIENT_SECRET}"
```

### Environment Variables

```bash
# Tiingo
export TIINGO_API_KEY="your_key"

# cTrader
export CTRADER_CLIENT_ID="your_client_id"
export CTRADER_CLIENT_SECRET="your_client_secret"

# AlphaVantage
export ALPHAVANTAGE_KEY="your_key"
```

## cTrader Setup (OAuth 2.0)

### Step 1: Register Application

1. Go to [cTrader Open API Portal](https://openapi.ctrader.com/)
2. Create new application
3. Note `Client ID` and `Client Secret`
4. Set redirect URI: `http://localhost:5000/callback`

### Step 2: Add Credentials

#### Option A: CLI
```bash
python -m app providers add ctrader
```

#### Option B: Python API
```python
from forex_diffusion.credentials import CredentialsManager, ProviderCredentials, OAuth2Flow

# OAuth Flow
oauth = OAuth2Flow(
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# Authorize (opens browser)
token_data = await oauth.authorize()

# Save credentials
creds = ProviderCredentials(
    provider_name="ctrader",
    client_id="your_client_id",
    client_secret="your_client_secret",
    access_token=token_data["access_token"],
    refresh_token=token_data["refresh_token"],
    environment="demo",  # or "live"
)

manager = CredentialsManager()
manager.save(creds)
```

### Step 3: Verify Connection

```bash
python -m app providers test ctrader
```

## Using Providers

### Basic Usage

```python
from forex_diffusion.providers import get_provider_manager

# Get manager
manager = get_provider_manager()

# Create provider
tiingo = manager.create_provider("tiingo", config={
    "api_key": "your_key",
    "tickers": ["eurusd", "gbpusd"]
})

# Connect
await tiingo.connect()

# Get data
price = await tiingo.get_current_price("EUR/USD")
bars = await tiingo.get_historical_bars(
    symbol="EUR/USD",
    timeframe="1h",
    start_ts_ms=1600000000000,
    end_ts_ms=1600086400000
)

# Stream quotes
async for quote in await tiingo.stream_quotes(["EUR/USD"]):
    print(quote)
```

### Advanced: Failover

```python
# Set primary and secondary
manager.set_primary_provider("ctrader")
manager.set_secondary_provider("tiingo")

# Automatic failover on primary failure
await manager.failover_to_secondary()
```

### Health Monitoring

```python
# Check health
health = tiingo.get_health()
print(health.to_dict())
# {
#   "is_connected": True,
#   "latency_ms": 45.2,
#   "data_rate_msg_per_sec": 120.5,
#   "error_rate_pct": 0.1,
#   "uptime_seconds": 3600.0,
#   "recent_errors": []
# }

# Check if healthy
if tiingo.is_healthy():
    print("Provider is healthy")
```

## Provider Capabilities

### Checking Capabilities

```python
from forex_diffusion.providers.base import ProviderCapability

provider = manager.get_provider("ctrader")

# Check specific capability
if provider.supports(ProviderCapability.DOM):
    depth = await provider.get_market_depth("EUR/USD", levels=10)

# Get all capabilities
print(provider.capabilities)
# [ProviderCapability.QUOTES, ProviderCapability.BARS, ...]
```

### Capability Matrix

| Capability | Tiingo | cTrader | AlphaVantage |
|------------|--------|---------|--------------|
| QUOTES | ✓ | ✓ | ✗ |
| BARS | ✓ | ✓ | ✓ |
| TICKS | ✗ | ✓ | ✗ |
| VOLUMES | ✗ | ✓ | ✗ |
| DOM | ✗ | ✓ | ✗ |
| SENTIMENT | ✗ | ✓ | ✗ |
| NEWS | ✗ | ✓ | ✗ |
| CALENDAR | ✗ | ✓ | ✗ |
| WEBSOCKET | ✓ | ✓ | ✗ |
| HISTORICAL_BARS | ✓ | ✓ | ✓ |
| HISTORICAL_TICKS | ✗ | ✓ | ✗ |

## Data Source Priority

Configure which provider to use for each data type in `configs/default.yaml`:

```yaml
data_sources:
  quotes:
    primary: "ctrader"
    fallback: "tiingo"

  historical_bars:
    primary: "tiingo"      # Tiingo has more history
    fallback: "ctrader"

  ticks:
    primary: "ctrader"     # Only cTrader supports ticks
    fallback: null

  dom:
    primary: "ctrader"
    fallback: null         # No fallback for DOM
```

## Refresh Rates

Configure polling intervals in `configs/default.yaml`:

```yaml
refresh_rates:
  rest:
    news_feed: 300           # Poll news every 5 minutes
    economic_calendar: 21600 # Poll calendar every 6 hours
    sentiment: 30            # Poll sentiment every 30 seconds

  websocket:
    reconnect_backoff: [1, 2, 4, 8, 16]  # Exponential backoff
    heartbeat_interval: 30     # Heartbeat every 30 sec
```

## Adding a New Provider

### Step 1: Implement BaseProvider

```python
# src/forex_diffusion/providers/myprovider.py

from .base import BaseProvider, ProviderCapability
import pandas as pd

class MyProvider(BaseProvider):
    def __init__(self, config=None):
        super().__init__(name="myprovider", config=config)
        self.api_key = config.get("api_key") if config else None

    @property
    def capabilities(self):
        return [
            ProviderCapability.QUOTES,
            ProviderCapability.HISTORICAL_BARS,
        ]

    async def connect(self):
        # Implement connection logic
        self.health.is_connected = True
        return True

    async def disconnect(self):
        # Cleanup
        self.health.is_connected = False

    async def _get_historical_bars_impl(
        self, symbol, timeframe, start_ts_ms, end_ts_ms
    ):
        # Implement data retrieval
        # Return pd.DataFrame with columns: ts_utc, open, high, low, close, volume
        pass
```

### Step 2: Register Provider

```python
# src/forex_diffusion/providers/manager.py

from .myprovider import MyProvider

class ProviderManager:
    def __init__(self):
        self._provider_classes = {
            "tiingo": TiingoProvider,
            "ctrader": CTraderProvider,
            "myprovider": MyProvider,  # Add here
        }
```

### Step 3: Add Configuration

```yaml
# configs/default.yaml

providers:
  myprovider:
    enabled: false
    api_key: "${MYPROVIDER_KEY}"
    base_url: "https://api.myprovider.com"
```

### Step 4: Update __init__.py

```python
# src/forex_diffusion/providers/__init__.py

from .myprovider import MyProvider

__all__ = [
    "BaseProvider",
    "ProviderCapability",
    "ProviderManager",
    "TiingoProvider",
    "CTraderProvider",
    "MyProvider",  # Add here
]
```

## Troubleshooting

### Provider Won't Connect

1. **Check credentials**:
   ```python
   from forex_diffusion.credentials import get_credentials_manager
   creds = get_credentials_manager().load("ctrader")
   print(creds)
   ```

2. **Test connection**:
   ```bash
   python -m app providers test ctrader
   ```

3. **Check logs**:
   ```bash
   tail -f logs/forex_diffusion.log
   ```

### OAuth Fails

1. **Verify redirect URI**: Must be `http://localhost:5000/callback`
2. **Check firewall**: Port 5000 must be accessible
3. **Clear old credentials**:
   ```python
   from forex_diffusion.credentials import get_credentials_manager
   get_credentials_manager().delete("ctrader")
   ```

### Rate Limiting

1. **Check rate limits** in provider docs
2. **Configure backoff** in YAML:
   ```yaml
   providers:
     myprovider:
       retry:
         attempts: 8
         backoff_base_seconds: 1.0
         max_backoff_seconds: 60.0
   ```

### Data Quality Issues

1. **Enable QA logging**:
   ```yaml
   qa:
     outlier_zscore_threshold: 8.0
     gap_flag_multiplier: 1.1
     log_qa: true
   ```

2. **Check validation reports** in `artifacts/reports/`

## Best Practices

### 1. Use Failover

Always configure a secondary provider for critical operations:

```python
manager.set_primary_provider("ctrader")
manager.set_secondary_provider("tiingo")
```

### 2. Monitor Health

Implement health checks in your application:

```python
async def monitor_providers():
    while True:
        health = manager.get_health_summary()
        if health["ctrader"]["error_rate_pct"] > 10:
            await manager.failover_to_secondary()
        await asyncio.sleep(30)
```

### 3. Handle Errors Gracefully

```python
try:
    bars = await provider.get_historical_bars(...)
except Exception as e:
    logger.error(f"Failed to get bars: {e}")
    # Try fallback provider
    fallback = manager.get_secondary_provider()
    if fallback:
        bars = await fallback.get_historical_bars(...)
```

### 4. Cache Aggressively

Use the built-in caching for expensive operations:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_bars(symbol, timeframe, start, end):
    return provider.get_historical_bars(...)
```

### 5. Respect Rate Limits

Implement request throttling:

```python
import asyncio
from collections import deque

class RateLimiter:
    def __init__(self, max_per_second=5):
        self.max_per_second = max_per_second
        self.requests = deque(maxlen=max_per_second)

    async def wait(self):
        now = time.time()
        while self.requests and (now - self.requests[0]) > 1.0:
            self.requests.popleft()

        if len(self.requests) >= self.max_per_second:
            wait_time = 1.0 - (now - self.requests[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(time.time())
```

---

**Last Updated**: 2025-01-05
**Version**: 2.0.0
