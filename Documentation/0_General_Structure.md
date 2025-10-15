# ForexGPT - General Structure & Architecture

**Version**: 2.0.0  
**Last Updated**: 2025-10-13  
**Status**: Production

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Module Organization](#module-organization)
5. [Technology Stack](#technology-stack)
6. [Data Flow](#data-flow)
7. [Key Workflows](#key-workflows)
8. [Configuration Management](#configuration-management)
9. [Development Guidelines](#development-guidelines)

---

## Executive Summary

**ForexGPT** is an advanced AI-powered Forex trading platform featuring:

- **Multi-Provider Data Integration**: Unified interface for Tiingo, cTrader Open API, AlphaVantage, Dukascopy
- **AI/ML Models**: Diffusion models (SSSD, VAE), Transformers, traditional ML (scikit-learn)
- **Pattern Recognition**: 30+ chart patterns with confidence calibration and genetic optimization
- **Automated Trading**: Signal generation, risk management, smart execution
- **Real-Time Visualization**: PySide6 GUI with FinPlot financial charting
- **Production-Ready**: Comprehensive testing, monitoring, and deployment infrastructure

**Architecture Type**: Modular layered architecture with event-driven components

---

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         GUI Layer (PySide6)                       │
│  FinPlot Charts | Tabs | Dialogs | Workers | Controllers         │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                    Application Services                           │
│  MarketData | DBService | Aggregators | ModelService             │
└───────────────────────────┬──────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┬──────────────────┐
        │                   │                   │                  │
┌───────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐ ┌──────▼──────┐
│  Data Layer    │ │  Model Layer    │ │ Trading Layer   │ │  API Layer  │
│                │ │                 │ │                 │ │             │
│ Providers      │ │ Training        │ │ Execution       │ │ FastAPI     │
│ Database       │ │ Inference       │ │ Risk Management │ │ REST/WS     │
│ Aggregators    │ │ Patterns        │ │ Backtesting     │ │             │
└────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────┘
```

### Architecture Principles

1. **Separation of Concerns**: Clear boundaries between data, business logic, and presentation
2. **Provider Abstraction**: Unified interface for multiple data sources
3. **Async-First**: asyncio for I/O operations, threading for CPU-bound tasks
4. **Event-Driven**: Event bus for component communication
5. **Configuration Over Code**: YAML-based configuration with environment overrides
6. **Testability**: Dependency injection and mockable interfaces

---

## Core Components

### Layer 1: Data Acquisition

**Location**: `src/forex_diffusion/providers/`, `src/forex_diffusion/services/`

#### Multi-Provider System

**BaseProvider** (`providers/base.py`):
- Abstract interface defining data provider contract
- Capability enumeration (QUOTES, BARS, TICKS, DOM, SENTIMENT, NEWS, CALENDAR)
- Health monitoring (latency, error rate, uptime)
- Connection lifecycle management

**Concrete Providers**:
- **TiingoProvider**: REST API + WebSocket streaming
- **CTraderProvider**: Twisted→asyncio bridge, Protobuf messages, OAuth 2.0
- **AlphaVantageProvider**: REST API with rate limiting

**ProviderManager** (`providers/manager.py`):
- Factory pattern for provider instantiation
- Capability-based routing
- Primary/secondary failover
- Health-based automatic failover

#### Services

**MarketDataService** (`services/marketdata.py`):
- Historical backfill with gap detection
- Timeframe resampling (1m → 5m, 15m, 1h, etc.)
- Multi-symbol/timeframe management
- Progress tracking for GUI

**AggregatorService** (`services/aggregator.py`):
- Real-time candle aggregation from tick/quote streams
- Derives higher timeframes from 1m base
- Volume aggregation (tick_volume, real_volume)
- Background worker thread

**DOMAggregatorService** (`services/dom_aggregator.py`):
- Order book aggregation
- Spread, mid-price, imbalance calculation
- LRU cache for real-time access

**SentimentAggregatorService** (`services/sentiment_aggregator.py`):
- Sentiment metrics with moving averages (5m, 15m, 1h)
- Contrarian signal generation
- Change rate analysis

---

### Layer 2: Data Storage

**Location**: `migrations/`, `src/forex_diffusion/db_adapter.py`

#### Database Schema (SQLite + Alembic)

**Core Tables**:
1. **market_data_candles**
   - OHLCV data with timestamp, symbol, timeframe
   - tick_volume, real_volume, provider_source
   - Composite index: (symbol, timeframe, ts_utc)

2. **market_depth**
   - DOM snapshots (bids/asks as JSON)
   - Derived metrics: mid_price, spread, imbalance

3. **sentiment_data**
   - Trader sentiment: long_pct, short_pct, total_traders, confidence

4. **news_events**
   - News feed: title, content, currency, impact, category

5. **economic_calendar**
   - Economic events: event_name, forecast, actual, previous, impact

6. **optimization_results**
   - Pattern parameters and backtest metrics

7. **training_queue**
   - Training job queue with status tracking

#### Services

**DBService** (`services/db_service.py`):
- SQLAlchemy ORM wrapper
- Connection pooling
- Migration management (Alembic)

**DBWriter** (`services/db_writer.py`):
- Async batch writer with queue
- Commit every 60s or 500 records (configurable)
- Thread-safe operation

---

### Layer 3: Feature Engineering

**Location**: `src/forex_diffusion/features/`

#### Feature Pipeline

**Causal Features**:
- Multi-timeframe technical indicators (ATR, RSI, Bollinger, MACD, Donchian, Keltner, Hurst)
- BTA-Lib integration for indicator calculation
- Cyclic time features (hour_sin, hour_cos)
- Volume features (tick + real)
- DOM features (spread, imbalance, order flow)

**Horizon Management** (`utils/horizon_converter.py`):
- Time-based horizons: 1m to 5d
- Multi-horizon scaling with regime detection
- Quantile predictions (5%, 50%, 95%)

**Key Principles**:
- **No Look-Ahead Bias**: All features calculated causally
- **Standardization**: Mean/std saved with model for inference consistency
- **Warmup Period**: Initial bars discarded to stabilize indicators

---

### Layer 4: Model System

**Location**: `src/forex_diffusion/models/`, `src/forex_diffusion/training/`

#### Model Types

**A. Traditional ML** (scikit-learn):
- **Ridge/Lasso/ElasticNet**: L1/L2 regularization
- **Random Forest**: Non-linear ensemble
- **Encoders**: None, PCA, Latents

**B. Deep Learning** (PyTorch):
- **Diffusion Models**: Cosine schedule, v-prediction parametrization
- **VAE**: Patch-based encoder/decoder (patch_len=64, z_dim=128)
- **SSSD**: Structured State Space Diffusion with S4 layers
- **Transformers**: Attention-based sequence models

**C. Specialized Models**:
- **Pattern Autoencoder**: Learned pattern embeddings
- **Multi-Timeframe Ensemble**: Combines predictions across timeframes
- **ML Stacked Ensemble**: Meta-learning over base models

#### Training Infrastructure

**TrainingOrchestrator** (`training_pipeline/training_orchestrator.py`):
- Async training job management
- Queue-based execution
- Checkpoint management
- Progress tracking

**Optimization** (`training/optimization/`):
- **Single-Objective GA**: Maximize R² or Sharpe
- **NSGA-II**: Multi-objective (minimize -R², MAE)
- **Optuna**: Bayesian hyperparameter optimization
- **Regime-Aware Optimization**: Parameter adaptation per market regime

**NVIDIA Stack Support**:
- PyTorch Lightning with DDP (multi-GPU)
- APEX fused optimizers
- xFormers memory-efficient attention
- Flash Attention 2 (Ampere+ GPUs)
- NVIDIA DALI data loading (Linux/WSL)

---

### Layer 5: Inference & Prediction

**Location**: `src/forex_diffusion/inference/`

#### Inference Modes

**InferenceService** (`inference/service.py`):
- Single-horizon prediction
- Multi-horizon (1m to 5d simultaneously)
- Ensemble predictions (mean/median/voting)
- Quantile predictions (probabilistic)

**Samplers**:
- **DDIM**: 20 steps (default, fast)
- **DPM++**: 20 steps (higher quality)

**Parallel Inference** (`inference/parallel_inference.py`):
- Thread pool for multi-symbol/timeframe batching
- GPU memory management
- Batch size optimization

---

### Layer 6: Pattern Recognition

**Location**: `src/forex_diffusion/patterns/`

#### Pattern Categories (30+ patterns)

1. **Chart Patterns**:
   - Head & Shoulders, Inverse H&S
   - Triangles (Ascending, Descending, Symmetrical)
   - Wedges (Rising, Falling)
   - Flags, Pennants, Channels
   - Double/Triple Top/Bottom
   - Cup & Handle, Rounding Bottom

2. **Candlestick Patterns** (20+ patterns):
   - Engulfing (Bullish/Bearish)
   - Hammer, Hanging Man
   - Doji, Spinning Top
   - Morning/Evening Star
   - Three White Soldiers/Black Crows

3. **Harmonic Patterns**:
   - Gartley, Butterfly, Bat, Crab
   - ABCD patterns
   - Fibonacci-based validation

4. **Elliott Wave**:
   - Wave counting and labeling
   - Impulse and corrective waves

5. **DOM Patterns**:
   - Order flow imbalance
   - Large order detection
   - Iceberg order identification

#### Pattern System Architecture

**ProgressiveFormationScanner** (`patterns/progressive_formation.py`):
- Real-time pattern formation detection
- Incremental pattern validation
- Confidence scoring

**PatternOptimizer** (NSGA-II genetic algorithm):
- Multi-objective optimization (accuracy vs. profit)
- Parameter space exploration
- Backtest-based fitness evaluation

**ConfidenceCalibrator** (`patterns/confidence_calibrator.py`):
- Isotonic regression calibration
- Historical pattern success rate
- Regime-specific confidence adjustment

---

### Layer 7: Regime Detection

**Location**: `src/forex_diffusion/regime/`

#### Regime Classification

**RegimeClassifier**:
- Market states: Trending, Ranging, Volatile, Calm
- Hidden Markov Models (HMM)
- Hurst exponent analysis (persistence)
- Volatility clustering (ARCH/GARCH)

**Regime-Aware Systems**:
- **Position Sizing**: Kelly criterion adjustment per regime
- **Pattern Parameters**: Adaptive thresholds
- **Risk Management**: Dynamic stop-loss/take-profit

---

### Layer 8: Risk Management

**Location**: `src/forex_diffusion/risk/`, `src/forex_diffusion/backtesting/`

#### Risk Components

**Multi-Level Stop Loss** (`risk/multi_level_stop_loss.py`):
- Initial stop loss
- Trailing stop (profit protection)
- Time-based exit (theta decay)

**Position Sizing** (`risk/regime_position_sizer.py`):
- Kelly criterion
- Regime-adjusted sizing
- Maximum drawdown constraints

**Pre-Trade Validation** (`ui/pre_trade_validation_dialog.py`):
- Spread checks (max allowed spread)
- DOM confirmation (imbalance > threshold)
- Sentiment filters (contrarian/momentum)
- Economic calendar risk (avoid high-impact events)

**Risk Profiles**:
- Conservative: Lower leverage, tighter stops
- Moderate: Balanced risk/reward
- Aggressive: Higher leverage, wider stops

#### Backtesting

**AdvancedBacktestEngine** (`backtesting/advanced_backtest_engine.py`):
- VectorBT Pro integration
- Realistic slippage modeling
- Commission structure
- Multi-asset portfolio backtesting

**Probabilistic Metrics** (`backtesting/probabilistic_metrics.py`):
- CRPS (Continuous Ranked Probability Score)
- PIT-KS (Probability Integral Transform - Kolmogorov-Smirnov)
- Calibration curves

**Walk-Forward Analysis** (`validation/walk_forward.py`):
- Rolling window validation
- Out-of-sample testing
- Combinatorial purged cross-validation

---

### Layer 9: Trading Execution

**Location**: `src/forex_diffusion/execution/`, `src/forex_diffusion/trading/`

#### Execution System

**SmartExecution** (`execution/smart_execution.py`):
- **TWAP** (Time-Weighted Average Price)
- **VWAP** (Volume-Weighted Average Price)
- **Iceberg Orders**: Hide large orders
- **Slippage Modeling**: Expected vs. actual fill

**AutomatedTradingEngine** (`trading/automated_trading_engine.py`):
- Signal generation from patterns/forecasts
- Pre-trade validation
- Order execution
- Position monitoring
- Auto-exit on conditions (stop loss, take profit, time)

**Order Flow Analysis** (`analysis/order_flow_analyzer.py`):
- Real-time imbalance detection
- Large order tracking (95th percentile)
- Z-score-based alerts

#### Broker Integration

**Broker Adapters** (`broker/`):
- cTrader Open API (live/demo)
- FxPro (placeholder)
- Extensible adapter pattern

---

### Layer 10: Portfolio Management

**Location**: `src/forex_diffusion/portfolio/`

#### Portfolio Optimization

**Riskfolio-Lib Integration**:
- Mean-variance optimization
- CVaR (Conditional Value at Risk)
- Maximum Sharpe ratio
- Risk parity

**Multi-Currency Portfolio**:
- Cointegration analysis (statsmodels)
- Currency correlation matrix
- Diversification metrics

**Rebalancing**:
- Periodic rebalancing (daily/weekly)
- Threshold-based rebalancing
- Transaction cost optimization

---

### Layer 11: GUI Layer

**Location**: `src/forex_diffusion/ui/`

#### Main Application

**App Entry Point** (`ui/app.py`):
- PySide6 Qt application
- Service initialization (DB, MarketData, Aggregators)
- Tab widget management
- Menu bar and shortcuts

#### Key Tabs

1. **Chart Tab** (`ui/chart_tab/`):
   - FinPlot candlestick charts
   - Drawing tools (trendlines, rectangles, Fibonacci)
   - Indicator overlays
   - Pattern overlays with confidence bands
   - Volume subplot

2. **Training Tab** (`ui/training_tab.py`):
   - Model training configuration
   - Indicator × Timeframe selection grid
   - Optimization settings (GA/NSGA-II)
   - Training queue management
   - Progress tracking

3. **Forecast Tab** (`ui/forecast_settings_tab.py`):
   - Prediction settings (single/multi-horizon)
   - Ensemble configuration
   - Quantile selection
   - Visualization on chart

4. **Patterns Tab** (`ui/patterns_tab.py`):
   - Pattern scanning controls
   - Parameter configuration
   - Historical scan results
   - Confidence calibration

5. **Live Trading Tab** (`ui/live_trading_tab.py`):
   - Real-time trading interface
   - Position monitoring
   - Order management
   - P&L tracking

6. **Backtesting Tab** (`ui/backtesting_tab.py`):
   - Strategy configuration
   - Backtest execution
   - Performance metrics
   - Equity curve visualization

7. **Portfolio Tab** (`ui/portfolio_tab.py`):
   - Multi-currency portfolio view
   - Optimization controls
   - Correlation matrix
   - Allocation charts

8. **Regime Analysis Tab** (`ui/regime_analysis_tab.py`):
   - Current regime indicator
   - Regime history
   - Regime definition editor

9. **Data Sources Tab** (`ui/data_sources_tab.py`):
   - Provider configuration
   - Connection status
   - Data quality metrics

10. **News/Calendar Tab** (`ui/news_calendar_tab.py`):
    - Economic calendar with filtering
    - News feed
    - Impact level color coding

11. **Reports 3D Tab** (`ui/reports_3d_tab.py`):
    - 3D visualizations (plotly)
    - Multi-dimensional analysis

#### UI Controllers

**UIController** (`ui/controllers/ui_controller.py`):
- Main GUI orchestration
- Menu action handlers
- Service coordination

**TrainingController** (`ui/controllers/training_controller.py`):
- Async training job management
- Progress signals
- Queue management

**Workers** (`ui/workers/`):
- **ForecastWorker**: Background prediction execution
- **ScanWorker**: Pattern scanning
- **DetectionWorker**: Real-time pattern detection

---

### Layer 12: API Layer

**Location**: `src/forex_diffusion/api/`

#### FastAPI REST API

**Main Application** (`api/main.py`):
- FastAPI with async handlers
- CORS middleware
- Health monitoring
- Request/response logging

**Endpoints**:
- `GET /api/sentiment/{symbol}`: Latest sentiment metrics
- `GET /api/sentiment/{symbol}/history`: Historical sentiment
- `GET /api/sentiment/{symbol}/signal`: Trading signal from sentiment
- `GET /api/calendar/upcoming`: Upcoming economic events
- `GET /api/calendar/risk/{symbol}`: Risk score for symbol
- `POST /api/predict`: Model predictions (placeholder)

**Features**:
- Redis caching (optional)
- Rate limiting (TODO for production)
- Authentication (TODO for production)

**Deployment**:
```bash
python run_api.py --production --workers 4
```

---

### Layer 13: CLI Layer

**Location**: `src/forex_diffusion/cli/`

#### CLI Commands

**Provider Management** (`cli/providers.py`):
```bash
python -m forex_diffusion.cli.providers list
python -m forex_diffusion.cli.providers add ctrader
python -m forex_diffusion.cli.providers test tiingo
python -m forex_diffusion.cli.providers delete ctrader
python -m forex_diffusion.cli.providers capabilities ctrader
```

**Data Operations** (`cli/data.py`):
```bash
python -m forex_diffusion.cli.data backfill --provider tiingo --symbol EURUSD --days 30
python -m forex_diffusion.cli.data vacuum
python -m forex_diffusion.cli.data stats
```

---

## Module Organization

### Source Tree Structure

```
src/forex_diffusion/
├── adapters/               # External service adapters
├── analysis/               # Analysis modules (order flow, correlation)
├── api/                    # FastAPI REST API
├── backtest/               # Backtesting utilities (legacy)
├── backtesting/            # Advanced backtesting engine
├── broker/                 # Broker adapters (cTrader, FxPro)
├── cache/                  # Caching utilities
├── cli/                    # Command-line interface
├── config/                 # Configuration models
├── credentials/            # Secure credential management
├── data/                   # Data adapters (legacy)
├── execution/              # Smart execution algorithms
├── features/               # Feature engineering pipeline
├── inference/              # Model inference system
├── models/                 # ML/DL model definitions
├── monitoring/             # Monitoring and logging
├── patterns/               # Pattern recognition (30+ patterns)
├── portfolio/              # Portfolio optimization
├── providers/              # Multi-provider data system
├── regime/                 # Regime detection
├── risk/                   # Risk management
├── services/               # Core services (MarketData, DB, Aggregators)
├── trading/                # Automated trading engine
├── train/                  # Training loop (legacy)
├── training/               # Training pipeline
│   ├── optimization/       # Hyperparameter optimization
│   └── training_pipeline/  # Training orchestration
├── ui/                     # PySide6 GUI
│   ├── chart_components/   # Chart-related components
│   ├── chart_tab/          # Chart tab implementation
│   ├── controllers/        # UI controllers
│   ├── handlers/           # Event handlers
│   └── workers/            # Background workers
├── utils/                  # Utility functions
├── validation/             # Model validation
└── visualization/          # Visualization tools
```

---

## Technology Stack

### Core Languages & Frameworks

**Python 3.10+**:
- Type hints with Pydantic models
- Async/await with asyncio
- Threading for CPU-bound tasks

### ML/AI Stack

**Deep Learning**:
- **PyTorch 2.x**: Primary DL framework
- **PyTorch Lightning**: Training orchestration
- **TensorFlow 2.x**: Alternative backend
- **xFormers**: Memory-efficient transformers
- **Flash Attention 2**: Fast attention (Ampere+ GPUs)

**Traditional ML**:
- **scikit-learn**: Ridge, Lasso, RF
- **XGBoost**: Gradient boosting
- **LightGBM**: Fast gradient boosting

**Optimization**:
- **Pymoo**: Multi-objective optimization (NSGA-II)
- **Optuna**: Bayesian optimization
- **DEAP**: Genetic algorithms

### Data Processing

**Core Libraries**:
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **BTA-Lib**: Technical indicators

**Database**:
- **SQLite**: Primary database
- **SQLAlchemy**: ORM
- **Alembic**: Migrations
- **DuckDB**: Fast analytics (optional)

**Backtesting**:
- **VectorBT Pro**: Advanced backtesting
- **backtesting.py**: Strategy backtesting

### GUI & Visualization

**GUI Framework**:
- **PySide6**: Qt 6.5+ bindings
- **FinPlot**: Financial charting

**Visualization**:
- **Plotly**: Interactive 3D charts
- **Matplotlib**: Static plots

### Networking & Async

**HTTP Clients**:
- **httpx**: Async HTTP client
- **requests**: Sync HTTP client

**WebSocket**:
- **asyncio**: Async I/O
- **Twisted**: cTrader WebSocket bridge

**Message Encoding**:
- **Protobuf**: cTrader API messages

### Security & Credentials

**Encryption**:
- **cryptography**: Fernet symmetric encryption
- **keyring**: OS-level credential storage

**OAuth**:
- **httpx**: OAuth 2.0 Authorization Code Flow with PKCE

### Configuration & Logging

**Configuration**:
- **PyYAML**: YAML parsing
- **Pydantic**: Configuration validation
- **python-dotenv**: Environment variables

**Logging**:
- **Loguru**: Structured logging
- **colorama**: Colored console output

### Time Series & Finance

**Time Series**:
- **statsmodels**: Statistical models (ARCH/GARCH)
- **hmmlearn**: Hidden Markov Models

**Portfolio**:
- **Riskfolio-Lib**: Portfolio optimization
- **cvxpy**: Convex optimization

### Development Tools

**Testing**:
- **pytest**: Test framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting

**Code Quality**:
- **Black**: Code formatting
- **Ruff**: Fast linter

**Build**:
- **setuptools**: Package building
- **pip**: Dependency management

---

## Data Flow

### Real-Time Data Flow

```
1. Provider WebSocket
   ↓
2. Twisted Callback (sync) → asyncio.Queue
   ↓
3. Stream Consumer (async)
   ↓
4. Aggregator (AggregatorService)
   ↓
5. DBWriter Queue
   ↓
6. Database (bulk insert every 60s / 500 records)
   ↓
7. GUI Update (via Qt signals)
```

### Historical Data Flow

```
1. GUI: User clicks "Backfill"
   ↓
2. MarketDataService.backfill_symbol_timeframe()
   ↓
3. ProviderManager.get_primary_provider()
   ↓
4. Provider.get_historical_bars() [REST API]
   ↓
5. Gap detection (compare with DB)
   ↓
6. DBWriter Queue
   ↓
7. Database (batch insert)
   ↓
8. Derive higher timeframes (1m → 5m, 15m, 1h, 4h, 1d, 1w)
   ↓
9. Progress callback → GUI
```

### Training Data Flow

```
1. GUI: Training Tab → Start Training
   ↓
2. TrainingController → TrainingOrchestrator
   ↓
3. Load data from DB (symbol, timeframe, days)
   ↓
4. Feature pipeline (indicators × timeframes)
   ↓
5. Train/test split (causal)
   ↓
6. Model training (Ridge/RF/VAE/Diffusion)
   ↓
7. Save model + scaler + metadata
   ↓
8. Register in training_queue (status=completed)
   ↓
9. Progress signals → GUI
```

### Inference Data Flow

```
1. GUI: Forecast Tab → Generate Prediction
   ↓
2. ForecastWorker (background thread)
   ↓
3. Load model + scaler from artifacts/
   ↓
4. Fetch latest OHLCV + features from DB
   ↓
5. Standardize features (using saved scaler)
   ↓
6. Model inference (single/multi-horizon)
   ↓
7. Generate quantiles (5%, 50%, 95%)
   ↓
8. Return prediction → GUI
   ↓
9. Draw on chart (confidence bands)
```

### Pattern Detection Data Flow

```
1. GUI: Patterns Tab → Start Scan
   ↓
2. DetectionWorker (background thread)
   ↓
3. Fetch OHLCV from DB (visible range)
   ↓
4. For each pattern type:
   ↓
5. ProgressiveFormationScanner.scan()
   ↓
6. Calculate confidence (ConfidenceCalibrator)
   ↓
7. DOM confirmation (if available)
   ↓
8. Return detected patterns
   ↓
9. Draw on chart (rectangles, labels)
```

---

## Key Workflows

### Workflow 1: Initial Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ForexGPT.git
cd ForexGPT

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e .

# 4. Install PyTorch with CUDA (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install VectorBT Pro (if available)
pip install ./VectorBt_PRO/vectorbtpro-2025.7.27-py3-none-any.whl

# 6. Run database migrations
alembic upgrade head

# 7. Configure providers
export TIINGO_API_KEY="your_key"
python -m forex_diffusion.cli.providers add ctrader

# 8. Backfill historical data
python -m forex_diffusion.cli.data backfill --provider tiingo --symbol EURUSD --days 30

# 9. Launch GUI
python -m forex_diffusion.ui.app
```

### Workflow 2: Train a Model

```
1. Launch GUI: python -m forex_diffusion.ui.app
2. Go to "Training" tab
3. Configure:
   - Symbol: EUR/USD
   - Base TF: 1m
   - Days history: 30
   - Horizon: 60 (bars)
   - Model: ridge
   - Encoder: pca
   - Optimization: none
4. Select indicators × timeframes (use Default buttons)
5. Set parameters:
   - warmup: 64
   - atr_n: 14
   - rsi_n: 14
   - bb_n: 20
   - encoder_dim: 64
6. Click "Start Training"
7. Monitor progress bar and logs
8. Model saved to artifacts/models/
```

### Workflow 3: Generate Predictions

```
1. Ensure model is trained (Workflow 2)
2. Go to "Forecast" tab
3. Configure:
   - Prediction mode: multi-horizon
   - Horizons: 1m, 5m, 15m, 30m, 1h
   - Ensemble: mean
   - Quantiles: 0.05, 0.5, 0.95
4. Click "Generate Forecast"
5. View predictions on chart (confidence bands)
6. Check forecast table for numeric values
```

### Workflow 4: Pattern Trading

```
1. Go to "Patterns" tab
2. Configure pattern parameters (or use defaults)
3. Run historical scan (optimization):
   - Click "Historical Scan"
   - Select pattern type
   - Set optimization objective (profit, accuracy)
   - Wait for results
4. Enable live scanning:
   - Toggle "Live Scan"
   - Patterns detected in visible range
   - Progressive formation tracking
5. Configure trading:
   - Go to "Live Trading" tab
   - Enable "Auto-Trade from Patterns"
   - Set risk parameters
6. Monitor:
   - Detected patterns appear on chart
   - Trades executed automatically
   - Positions tracked in Live Trading tab
```

---

## Configuration Management

### Configuration Hierarchy

1. **Default Config**: `configs/default.yaml`
2. **Environment Variables**: Override YAML values
3. **User Settings**: Persistent UI preferences (`user_settings.py`)
4. **Credentials**: Secure storage (OS keyring + Fernet)

### Main Configuration File

**Location**: `configs/default.yaml`

**Key Sections**:

```yaml
app:
  name: "magicforex"
  debug: false
  seed: 42

db:
  dialect: "sqlite"
  database_url: "sqlite:///./data/forex_diffusion.db"

providers:
  default: "tiingo"
  secondary: "ctrader"

  tiingo:
    enabled: true
    key: "${TIINGO_API_KEY}"

  ctrader:
    enabled: false
    environment: "demo"

model:
  artifacts_dir: "./artifacts/models"
  max_saved: 10

training:
  batch_size: 64
  max_epochs: 50
  learning_rate: 0.001

sampler:
  default: "ddim"
  ddim:
    steps: 20
    eta: 0.0
```

### Environment Variables

**Required**:
- `TIINGO_API_KEY`: Tiingo API key
- `CTRADER_CLIENT_ID`: cTrader OAuth client ID (if using cTrader)
- `CTRADER_CLIENT_SECRET`: cTrader OAuth client secret (if using cTrader)

**Optional**:
- `ALPHAVANTAGE_KEY`: AlphaVantage API key
- `API_PORT`: API server port (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)

### User Settings

**Location**: `%APPDATA%/ForexGPT/user_settings.json` (Windows) or `~/.config/ForexGPT/user_settings.json` (Linux/Mac)

**Stored Settings**:
- UI preferences (window size, splitter positions)
- Selected symbols, timeframes
- Indicator configurations
- Chart colors
- Risk profiles

---

## Development Guidelines

### Code Style

**Follow PEP 8**:
- 4-space indentation
- Max line length: 120 characters
- Type hints for function signatures
- Docstrings for classes and public methods

**Naming Conventions**:
- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants

### Project Organization

**Module Structure**:
```python
# Good: Clear responsibility
src/forex_diffusion/services/marketdata.py  # Market data service
src/forex_diffusion/providers/tiingo.py     # Tiingo provider

# Bad: Ambiguous naming
src/forex_diffusion/stuff.py                # Too vague
src/forex_diffusion/helpers.py              # What kind of helpers?
```

### Testing

**Test Location**: `tests/` (mirrors `src/forex_diffusion/`)

**Test Types**:
- Unit tests: Test individual functions/classes in isolation
- Integration tests: Test component interactions
- Manual tests: `tests/manual_tests/` (opt-in)

**Running Tests**:
```bash
# Fast suite
pytest tests -q

# All tests with coverage
pytest --cov=src/forex_diffusion --cov-report=html

# Specific module
pytest tests/test_providers.py -v
```

### Commit Guidelines

**Commit Format**:
```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Example**:
```
feat: Add multi-horizon prediction support

- Implement multi-horizon scaling with regime detection
- Add quantile predictions (5%, 50%, 95%)
- Update GUI to display confidence bands

Closes #123
```

### Git Workflow

**Branch Naming**:
- `feature/add-pattern-recognition`
- `fix/memory-leak-aggregator`
- `docs/update-architecture`

**Pull Request Process**:
1. Create feature branch from `main`
2. Make changes with tests
3. Run full test suite
4. Update documentation
5. Submit PR with description
6. Address review comments
7. Merge after approval

---

## Additional Resources

- **API Documentation**: `API_README.md`
- **Training Guide**: `Training.md`
- **Architecture Details**: `docs/ARCHITECTURE.md`
- **Provider Setup**: `docs/PROVIDERS.md`
- **Database Schema**: `docs/DATABASE.md`
- **NSGA-II Optimization**: `NSGA-II.md`

---

**For questions or contributions, see the main `README.md` or contact the development team.**
