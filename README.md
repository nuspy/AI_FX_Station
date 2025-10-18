# ForexGPT - Multi-Provider Trading Platform

## Overview

ForexGPT is an advanced trading platform with AI-powered forecasting, pattern recognition, and multi-provider market data integration. The system supports multiple data sources (Tiingo, cTrader, AlphaVantage) with unified interfaces and automatic failover.

## Features

### Core Capabilities
- **Multi-Provider Architecture**: Unified interface for Tiingo, cTrader, AlphaVantage
- **Real-Time Data**: WebSocket streaming with asyncio integration
- **Historical Backfill**: Intelligent gap detection and filling
- **AI Forecasting**: LDM4TS (vision-enhanced diffusion), SSSD, traditional ML
- **Memory Optimization**: SageAttention/FlashAttention for VRAM reduction (35-55%)
- **Pattern Recognition**: Advanced pattern detection with optimization
- **Sentiment Analysis**: Trader sentiment indicators
- **News & Calendar**: Economic events and news feed integration
- **Market Depth**: DOM/Level 2 order book data

### Technical Stack
- **ML Frameworks**: PyTorch Lightning, TensorFlow, scikit-learn
- **GUI**: FinPlot for financial charting
- **Database**: SQLite with Alembic migrations
- **Async**: asyncio with Twisted bridge for cTrader
- **Security**: OS keyring + Fernet encryption for credentials

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ForexGPT.git
cd ForexGPT

# Install dependencies
pip install -e .

# Install PyTorch with CUDA support (optional, for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install VectorBT Pro (if available)
pip install ./VectorBt_PRO/vectorbtpro-2025.7.27-py3-none-any.whl

# Optional: Memory optimization for LDM4TS training (VRAM < 12 GB)
# Install SageAttention 2 (~35% VRAM reduction)
python install_memory_optimization.py
# Or install with FlashAttention 2 (~45% VRAM reduction, requires RTX 30/40)
python install_memory_optimization.py --flash

# Run database migrations
alembic upgrade head
```

### Configuration

1. **Set API Keys** (environment variables or YAML):
```bash
export TIINGO_API_KEY="your_tiingo_key"
export ALPHAVANTAGE_KEY="your_alphavantage_key"
export CTRADER_CLIENT_ID="your_client_id"
export CTRADER_CLIENT_SECRET="your_client_secret"
```

2. **Configure Providers** in `configs/default.yaml`:
```yaml
providers:
  default: "tiingo"
  secondary: "ctrader"

  tiingo:
    enabled: true
    key: "${TIINGO_API_KEY}"

  ctrader:
    enabled: false
    environment: "demo"
```

3. **Setup cTrader OAuth** (optional):
```bash
python -m app providers add ctrader
```

### Running the Application

```bash
# Start GUI
python -m forex_diffusion.ui.main

# Run backfill
python -m app data backfill --symbol "EUR/USD" --timeframe 1h --days 30

# Train model
python -m forex_diffusion.training.train --symbol EUR/USD --timeframe 1h
```

## Architecture

### System Components

```
┌─────────────────┐
│   GUI Layer     │  ← FinPlot Charts, Widgets
└────────┬────────┘
         │
┌────────▼────────┐
│ Provider Manager│  ← Factory, Failover, Health
└────────┬────────┘
         │
    ┌────┴────┬────────┬─────────┐
    │         │        │         │
┌───▼──┐  ┌──▼──┐  ┌──▼───┐  ┌──▼────┐
│Tiingo│  │cTrad│  │Alpha │  │Future │
└───┬──┘  └──┬──┘  └──┬───┘  └───────┘
    └────────┴────────┴─────────┐
                                 │
                    ┌────────────▼────────┐
                    │  Data Pipeline      │
                    │  (Aggregators, etc.)│
                    └────────────┬────────┘
                                 │
                    ┌────────────▼────────┐
                    │  Storage Layer      │
                    │  (SQLite + Cache)   │
                    └─────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture.

### Database Schema

**Core Tables**:
- `market_data_candles`: OHLCV with tick/real volume
- `market_depth`: DOM/Level 2 data
- `sentiment_data`: Trader sentiment
- `news_events`: News feed
- `economic_calendar`: Economic events

See [docs/DATABASE.md](docs/DATABASE.md) for schema details.

## Provider Setup

### Tiingo (Default)

Free tier available with rate limits.

```bash
export TIINGO_API_KEY="your_key"
```

### cTrader Open API

Requires OAuth 2.0 authorization:

1. Register app at [cTrader Open API Portal](https://openapi.ctrader.com/)
2. Run setup wizard:
```bash
python -m app providers add ctrader
```
3. Authorize in browser (opens automatically)

See [docs/PROVIDERS.md](docs/PROVIDERS.md) for detailed setup.

### AlphaVantage

Free tier with very restrictive rate limits.

```bash
export ALPHAVANTAGE_KEY="your_key"
```

## Usage Examples

### Basic Data Retrieval

```python
from forex_diffusion.providers import get_provider_manager

# Get provider
manager = get_provider_manager()
provider = manager.create_provider("tiingo", config={
    "api_key": "your_key"
})

# Connect
await provider.connect()

# Get historical data
bars = await provider.get_historical_bars(
    symbol="EUR/USD",
    timeframe="1h",
    start_ts_ms=1600000000000,
    end_ts_ms=1600086400000
)

# Stream real-time quotes
async for quote in await provider.stream_quotes(["EUR/USD"]):
    print(f"{quote['symbol']}: {quote['price']}")
```

### Failover Configuration

```python
# Set primary and secondary providers
manager.set_primary_provider("ctrader")
manager.set_secondary_provider("tiingo")

# Automatic failover on primary failure
if not primary.is_healthy():
    await manager.failover_to_secondary()
```

### Advanced Features

```python
# Get market depth (DOM)
depth = await provider.get_market_depth("EUR/USD", levels=10)
print(f"Bids: {depth['bids']}, Asks: {depth['asks']}")

# Get sentiment
sentiment = await provider.get_sentiment("EUR/USD")
print(f"Long: {sentiment['long_pct']}%, Short: {sentiment['short_pct']}%")

# Get news
news = await provider.get_news(currency="EUR", limit=10)
for item in news:
    print(f"{item['title']} - Impact: {item['impact']}")
```

## Training Models

### Traditional ML (sklearn)

```bash
python -m forex_diffusion.training.train_sklearn \
  --symbol "EUR/USD" \
  --timeframe 1h \
  --indicators rsi,macd,bollinger \
  --model ridge
```

### Diffusion Models (PyTorch Lightning)

```bash
python -m forex_diffusion.training.train \
  --symbol "EUR/USD" \
  --timeframe 1m \
  --horizon 100 \
  --max_epochs 50 \
  --batch_size 64
```

### Pattern Optimization

```bash
python -m forex_diffusion.patterns.optimize \
  --pattern head_and_shoulders \
  --symbol "EUR/USD" \
  --timeframe 1h
```

## CLI Commands

```bash
# Provider management
python -m app providers list
python -m app providers add ctrader
python -m app providers test tiingo

# Data operations
python -m app data backfill --provider tiingo --symbol EURUSD --days 30
python -m app db migrate
python -m app db vacuum
```

## Configuration

### Refresh Rates

Configure in `configs/default.yaml`:

```yaml
refresh_rates:
  rest:
    news_feed: 300        # 5 minutes
    economic_calendar: 21600  # 6 hours
    sentiment: 30         # 30 seconds

  websocket:
    reconnect_backoff: [1, 2, 4, 8, 16]
    heartbeat_interval: 30

  database:
    commit_interval: 60
    commit_batch_size: 1000
```

### Data Source Priority

```yaml
data_sources:
  quotes:
    primary: "ctrader"
    fallback: "tiingo"
  historical_bars:
    primary: "tiingo"
    fallback: "ctrader"
```

## Documentation

- [**Architecture Guide**](docs/ARCHITECTURE.md) - System design and components
- [**Provider Guide**](docs/PROVIDERS.md) - Provider setup and usage
- [**Database Schema**](docs/DATABASE.md) - Schema and query examples
- [**Architectural Decisions**](docs/DECISIONS.md) - ADRs and design choices

## Development

### Adding a New Provider

1. Create provider class:
```python
# src/forex_diffusion/providers/myprovider.py
from .base import BaseProvider, ProviderCapability

class MyProvider(BaseProvider):
    @property
    def capabilities(self):
        return [ProviderCapability.QUOTES, ProviderCapability.HISTORICAL_BARS]

    async def _get_historical_bars_impl(self, ...):
        # Implementation
        pass
```

2. Register in ProviderManager
3. Add configuration to YAML
4. Write tests

See [docs/PROVIDERS.md#adding-a-new-provider](docs/PROVIDERS.md#adding-a-new-provider) for details.

### Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Provider tests
pytest tests/providers/
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "add_new_feature"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Project Structure

```
ForexGPT/
├── src/forex_diffusion/
│   ├── providers/          # Multi-provider system
│   │   ├── base.py        # BaseProvider interface
│   │   ├── tiingo_provider.py
│   │   ├── ctrader_provider.py
│   │   └── manager.py     # ProviderManager
│   ├── credentials/       # Secure credential storage
│   │   ├── manager.py     # CredentialsManager
│   │   └── oauth.py       # OAuth2Flow
│   ├── training/          # ML training
│   ├── inference/         # Model inference
│   ├── patterns/          # Pattern detection
│   ├── ui/               # FinPlot GUI
│   └── services/         # Core services
├── configs/              # YAML configuration
├── migrations/           # Alembic migrations
├── docs/                 # Documentation
├── tests/               # Test suite
└── scripts/             # Utility scripts
```

## Troubleshooting

### Provider Connection Issues

```bash
# Test provider
python -m app providers test ctrader

# Check logs
tail -f logs/forex_diffusion.log

# Verify credentials
python -c "from forex_diffusion.credentials import get_credentials_manager; print(get_credentials_manager().load('ctrader'))"
```

### OAuth Failures

1. Verify redirect URI: `http://localhost:5000/callback`
2. Check firewall (port 5000 must be open)
3. Clear old credentials: `python -m app providers delete ctrader`

### Database Issues

```bash
# Vacuum database
python -m app db vacuum

# Check migration status
alembic current

# Reset database (DESTRUCTIVE)
rm data/forex_diffusion.db
alembic upgrade head
```

## Performance Tips

1. **Use Caching**: News and DOM data cached in RAM
2. **Configure Refresh Rates**: Tune polling intervals for your use case
3. **Enable GPU**: Use CUDA for faster training
4. **Parallel Backfill**: Set `parallel_backfill_workers: 4` in config
5. **Index Usage**: Database queries optimized with composite indexes

## Security

- **Never commit API keys** - Use environment variables
- **Credentials encrypted** - OS keyring + Fernet encryption
- **OAuth CSRF protection** - State parameter validation
- **Secure token storage** - Automatic token refresh
- **Audit logging** - All provider changes logged

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Update documentation
5. Submit pull request

## License

Proprietary - ForexMagic Team

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ForexGPT/issues)
- **Docs**: [Documentation](docs/)
- **Email**: dev@forexmagic.local

---

**Version**: 2.0.0 (Multi-Provider Architecture)
**Last Updated**: 2025-01-05
