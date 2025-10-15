# ForexGPT Trading Engine - Complete Documentation

**Version**: 1.0  
**Date**: 2025-10-13  
**Status**: Comprehensive System Review

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Training Pipeline](#training-pipeline)
3. [Backtesting System](#backtesting-system)
4. [Optimization Engine](#optimization-engine)
5. [Inference & Prediction](#inference--prediction)
6. [Order Execution](#order-execution)
7. [Risk Management](#risk-management)
8. [Complete Workflow](#complete-workflow)
9. [Parameter Flow](#parameter-flow)
10. [Critical Issues Identified](#critical-issues-identified)

---

## System Architecture Overview

### Directory Structure

```
src/forex_diffusion/
├── training/                    # Training system
│   ├── train.py                # Main training script
│   ├── train_sklearn.py        # Sklearn models training
│   ├── train_sklearn_btalib.py # BTALib features training
│   ├── train_optimized.py      # Optimized training with DALI
│   ├── train_sssd.py           # SSSD diffusion training
│   ├── optimized_trainer.py    # Performance-optimized trainer
│   ├── auto_retrain.py         # Automatic retraining system
│   ├── optimization/           # Parameter optimization
│   │   ├── engine.py          # NSGA-II optimization engine
│   │   ├── backtest_runner.py # Backtest runner for optimization
│   │   ├── parameter_space.py # Parameter space definitions
│   │   └── multi_objective.py # Multi-objective optimization
│   └── training_pipeline/      # Training orchestration
│       ├── training_orchestrator.py
│       ├── inference_backtester.py
│       └── regime_manager.py
│
├── training_pipeline/           # ⚠️ DUPLICATE directory
│   ├── training_orchestrator.py
│   ├── inference_backtester.py
│   └── data_loader.py
│
├── backtest/                    # Original backtest system
│   ├── engine.py               # Walk-forward backtest engine
│   ├── integrated_backtest.py  # Integrated system backtest
│   ├── worker.py               # Async backtest worker
│   ├── db.py                   # Backtest database
│   ├── genetic_optimizer.py    # Genetic optimization
│   ├── hybrid_optimizer.py     # Hybrid optimization
│   └── resumable_optimizer.py  # Resumable optimization
│
├── backtesting/                 # ⚠️ DUPLICATE backtesting system
│   ├── forecast_backtest_engine.py  # Forecast-specific backtest
│   ├── advanced_backtest_engine.py  # Advanced backtest engine
│   ├── risk_management.py
│   └── multi_horizon_validator.py
│
├── inference/                   # Prediction/inference system
│   ├── parallel_inference.py   # Parallel model inference
│   ├── service.py              # Inference API service
│   └── backtest_api.py         # Backtest API endpoints
│
├── trading/                     # Live trading engine
│   └── automated_trading_engine.py  # Automated trading system
│
├── execution/                   # Order execution
│   └── smart_execution.py      # Smart execution optimizer
│
├── risk/                        # Risk management (directory 1)
│   ├── position_sizer.py       # Position sizing calculator
│   ├── regime_position_sizer.py # Regime-aware position sizing
│   ├── multi_level_stop_loss.py # Multi-level stop loss
│   └── adaptive_stop_loss_manager.py  # Adaptive stops
│
├── portfolio/                   # ⚠️ DUPLICATE position sizing
│   ├── position_sizer.py       # Portfolio position sizer
│   ├── optimizer.py            # Portfolio optimizer
│   └── risk_metrics.py
│
├── broker/                      # Broker integration (directory 1)
│   └── ctrader_broker.py
│
├── brokers/                     # ⚠️ DUPLICATE broker directory
│   ├── base.py
│   ├── paper_broker.py
│   └── fxpro_ctrader.py
│
└── services/                    # Services and utilities
    ├── brokers.py
    ├── parameter_loader.py
    └── dom_aggregator.py
```

### ⚠️ Critical Structural Issues

1. **THREE different backtest engines** with overlapping functionality
2. **TWO position sizer implementations** (risk/ vs portfolio/)
3. **TWO training pipeline directories** (training/training_pipeline/ vs training_pipeline/)
4. **TWO broker directories** (broker/ vs brokers/)
5. **SEVEN different training scripts** with unclear differentiation
6. **Mixed import patterns** causing confusion

---

## Training Pipeline

### Overview

The training pipeline transforms raw OHLCV data into trained models capable of making predictions. Multiple training scripts exist with overlapping functionality.

### Training Scripts Comparison

| Script | Purpose | Model Types | Features | Status |
|--------|---------|-------------|----------|--------|
| `train.py` | Base training script | Diffusion, VAE | Basic features | ⚠️ Legacy |
| `train_sklearn.py` | Sklearn models | Ridge, Random Forest | Standard features | ✅ Active |
| `train_sklearn_btalib.py` | Sklearn + BTALib | Ridge with BTALib | Technical indicators | ✅ Active |
| `train_optimized.py` | GPU-optimized training | Diffusion, VAE | DALI loader, mixed precision | ⚠️ Experimental |
| `train_sssd.py` | SSSD diffusion | SSSD specific | Diffusion-specific | ⚠️ Specialized |
| `optimized_trainer.py` | Performance training | Multiple | Optimized pipeline | ❓ Unclear |
| `auto_retrain.py` | Automatic retraining | Multiple | Scheduled retraining | ✅ Utility |

### Training Workflow

```
1. DATA PREPARATION
   ├── Load OHLCV from database/CSV
   ├── Feature engineering (technical indicators, BTALib)
   ├── Horizon target calculation (1h, 4h, 1d, 1w)
   ├── Standardization/normalization
   └── Train/val/test split

2. MODEL TRAINING
   ├── Model selection (Ridge, RF, Diffusion, VAE, SSSD)
   ├── Hyperparameter configuration
   ├── Training loop (with checkpointing)
   ├── Validation monitoring
   └── Early stopping

3. MODEL SAVING
   ├── Model weights/parameters
   ├── Feature list
   ├── Scaler (mean/std)
   ├── Optional: PCA/encoder
   └── Metadata (horizons, version, timestamp)

4. VALIDATION
   ├── Backtest on test set
   ├── Performance metrics calculation
   ├── Multi-horizon validation
   └── Results storage
```

### Key Parameters

#### train_sklearn.py Parameters

```python
symbol: str = "EUR/USD"           # Trading pair
timeframe: str = "15m"            # Data timeframe
horizon: int = 4                  # Prediction horizon (bars)
algo: str = "ridge"               # Algorithm: ridge, rf, xgboost
artifacts_dir: str = "artifacts"  # Output directory
mode: str = "train"               # train or backtest
n_estimators: int = 100           # For RF/XGBoost
max_depth: int = 10               # Tree depth
alpha: float = 1.0                # Ridge regularization
```

#### train_sklearn_btalib.py Parameters

```python
# Same as train_sklearn.py plus:
use_btalib: bool = True           # Enable BTALib indicators
btalib_period: int = 14           # Indicator period
include_volume: bool = True       # Include volume features
```

#### train_sssd.py Parameters

```python
# Diffusion-specific:
timesteps: int = 1000             # Diffusion timesteps
beta_schedule: str = "linear"     # Noise schedule
learning_rate: float = 1e-4       # Adam learning rate
batch_size: int = 32              # Training batch size
n_epochs: int = 100               # Training epochs
embedding_dim: int = 128          # Latent dimension
```

### Feature Engineering

**Standard Features** (train_sklearn.py):
- Returns (1-bar, 5-bar, 20-bar)
- Volatility (rolling std)
- Volume changes
- Time-based features (hour, day of week)

**BTALib Features** (train_sklearn_btalib.py):
- SMA, EMA (multiple periods)
- RSI, MACD, Stochastic
- Bollinger Bands
- ATR, ADX
- Volume indicators (OBV, MFI)

**Diffusion Features** (train_sssd.py):
- Raw OHLCV embeddings
- Temporal embeddings
- Latent representations

---

## Backtesting System

### Multiple Backtest Engines

**⚠️ CRITICAL ISSUE**: Three separate backtest engines with overlapping functionality and no clear hierarchy.

#### Engine 1: `backtest/engine.py`

**Purpose**: Walk-forward backtesting with quantile-based strategy

**Features**:
- Walk-forward validation (configurable splits)
- Quantile-based entry/exit (q50, q05, q95)
- Random walk baseline comparison
- Transaction costs modeling (spread, slippage)
- First-passage simulation with max_hold timeout

**Key Parameters**:
```python
n_splits: int = 5                 # Walk-forward splits
train_days: int = 730             # Training window (2 years)
val_days: int = 90                # Validation window
test_days: int = 90               # Test window
entry_threshold: float = 0.0      # Entry trigger threshold
max_hold: int = 20                # Max holding period (bars)
target_from: str = "q95"          # Target from quantile
stop_from: str = "q05"            # Stop from quantile
spread_pips: float = 0.5          # Spread cost
slippage_pips: float = 0.2        # Slippage cost
```

**Strategy Logic**:
```
Entry (Long):
  IF median_prediction (q50) >= current_close + entry_threshold:
    ENTER at next bar's open + spread + slippage
    SET target = q95 quantile
    SET stop = q05 quantile

Exit:
  IF high >= target: EXIT at target (win)
  IF low <= stop: EXIT at stop (loss)
  IF bars_held >= max_hold: EXIT at current price (timeout)
```

**Metrics**:
- Sharpe ratio (annualized)
- Maximum drawdown
- Turnover
- Net P&L
- Win rate
- Average win/loss

#### Engine 2: `backtesting/forecast_backtest_engine.py`

**Purpose**: Specialized backtesting for probabilistic forecasts

**Features**:
- Multi-horizon evaluation (1h, 4h, 1d, 1w, 1m)
- Probabilistic metrics (CRPS, PIT, interval coverage)
- Directional accuracy
- Model comparison and significance testing

**Key Metrics**:
```python
# Basic accuracy
mae: float                        # Mean absolute error
rmse: float                       # Root mean squared error
mape: float                       # Mean absolute percentage error

# Probabilistic metrics
crps: float                       # Continuous Ranked Probability Score
pit_uniformity: float             # PIT uniformity test
interval_coverage: Dict[str, float]  # Coverage rates

# Directional
directional_accuracy: float       # Direction prediction accuracy
hit_rate: float                   # Hit rate for targets

# Multi-horizon
horizon_performance: Dict[str, float]  # Performance by horizon
```

**Does NOT execute trades** - focused on forecast quality only.

#### Engine 3: `backtest/integrated_backtest.py`

**Purpose**: End-to-end system backtest with ALL components

**Features**:
- Multi-timeframe ensemble predictions
- Multi-model stacked ensemble
- Regime detection (HMM)
- Multi-level risk management
- Regime-aware position sizing
- Smart execution optimization
- Transaction cost modeling

**Components Integration**:
```python
# Prediction
multi_timeframe_ensemble: MultiTimeframeEnsemble
ml_ensemble: StackedMLEnsemble

# Context
regime_detector: HMMRegimeDetector

# Risk management
risk_manager: MultiLevelStopLoss
position_sizer: RegimePositionSizer

# Execution
execution_optimizer: SmartExecutionOptimizer
```

**Configuration**:
```python
# Data
symbol: str
start_date: datetime
end_date: datetime
timeframes: List[str] = ['5m', '15m', '1h']

# Models
use_multi_timeframe: bool = True
use_stacked_ensemble: bool = True
use_regime_detection: bool = True
use_smart_execution: bool = True

# Capital management
initial_capital: float = 10000.0
max_positions: int = 3
base_risk_per_trade_pct: float = 1.0

# Risk management
use_multi_level_stops: bool = True
max_holding_hours: int = 24
daily_loss_limit_pct: float = 3.0

# Costs
spread_pct: float = 0.0002        # 2 pips
commission_pct: float = 0.0001    # 0.01%
slippage_pct: float = 0.0001      # 0.01%

# Walk-forward
train_size_days: int = 30
test_size_days: int = 7
step_size_days: int = 7
```

**Full Trade Lifecycle**:
```
1. SIGNAL GENERATION
   ├── Multi-timeframe predictions
   ├── Ensemble aggregation
   ├── Confidence scoring
   └── Regime detection

2. POSITION SIZING
   ├── Regime-aware sizing
   ├── Kelly criterion (optional)
   ├── Volatility adjustment
   └── Max exposure constraints

3. EXECUTION
   ├── Smart execution optimizer
   ├── Spread/slippage modeling
   ├── Commission calculation
   └── Order placement

4. RISK MANAGEMENT
   ├── Multi-level stop loss
   │   ├── Technical stop
   │   ├── Trailing stop
   │   ├── Time-based stop
   │   └── Volatility stop
   ├── Take profit targets
   ├── Max holding period
   └── Daily loss limits

5. EXIT
   ├── Stop hit
   ├── Target reached
   ├── Time expired
   ├── Daily limit reached
   └── Regime change
```

### Backtest Result Comparison

| Metric | engine.py | forecast_backtest_engine.py | integrated_backtest.py |
|--------|-----------|----------------------------|------------------------|
| **Execution** | ✅ Yes | ❌ No (forecast only) | ✅ Yes |
| **Multi-horizon** | ❌ No | ✅ Yes | ⚠️ Partial |
| **Regime-aware** | ❌ No | ⚠️ Optional | ✅ Yes |
| **Ensemble** | ❌ No | ❌ No | ✅ Yes |
| **Smart execution** | ❌ No | N/A | ✅ Yes |
| **Multi-level stops** | ❌ No | N/A | ✅ Yes |
| **Transaction costs** | ✅ Basic | N/A | ✅ Comprehensive |

---

## Optimization Engine

### NSGA-II Multi-Objective Optimization

Located in `training/optimization/engine.py`

**Purpose**: Find optimal parameters for pattern detection using multi-objective genetic algorithm

**Objectives** (simultaneous optimization):
1. **Maximize** success rate (win rate)
2. **Maximize** profit factor
3. **Minimize** max drawdown
4. **Maximize** Sharpe ratio
5. **Minimize** median time to target/invalidation

**Parameter Space**:
```python
# Pattern-specific parameters
min_span: (int, 5, 500)           # Min pattern span
max_span: (int, 10, 1000)         # Max pattern span
min_touches: (int, 2, 10)         # Min touch points
tolerance: (float, 0.001, 0.5)    # Pattern tolerance
tightness: (float, 0.1, 2.0)      # Pattern tightness

# Entry/exit parameters
entry_threshold: (float, 0.0, 1.0)
target_multiplier: (float, 1.0, 5.0)
stop_multiplier: (float, 0.5, 2.0)
```

**Optimization Workflow**:
```
1. INITIALIZATION
   ├── Define parameter space per pattern
   ├── Load historical data (2+ years)
   ├── Setup multi-objective evaluator
   └── Initialize population (NSGA-II)

2. EVOLUTION
   FOR each generation (max 1000):
     ├── Generate trial parameters (crossover + mutation)
     ├── Run backtest with trial parameters
     ├── Calculate multi-objective fitness
     ├── Update Pareto frontier
     ├── Selection (tournament)
     └── Check early stopping

3. PARETO FRONTIER
   ├── Extract non-dominated solutions
   ├── Rank by preference weights
   ├── Validate on out-of-sample data
   └── Select best compromise solution

4. PARAMETER PROMOTION
   ├── Validate parameters (PROC-001)
   ├── Store in database
   ├── Associate with asset/timeframe/regime
   └── Enable for live trading
```

**Backtest Runner** (`training/optimization/backtest_runner.py`):

**Features**:
- Walk-forward validation (6-month windows)
- Purge period (1 day) + embargo (2 days)
- Transaction cost modeling
- Recency weighting (recent performance weighted higher)
- Invalidation rule enforcement

**Key Parameters**:
```python
walk_forward_months: int = 6
purge_days: int = 1
embargo_days: int = 2
initial_capital: float = 100000.0
risk_per_trade: float = 0.02      # 2% risk
max_position_size: float = 0.1    # 10% max
spread_bps: float = 1.0           # 1 basis point
slippage_bps: float = 0.5         # 0.5 basis point
max_daily_drawdown: float = 0.05  # 5% daily DD limit
```

**Trade Simulation**:
```
FOR each pattern signal in walk-forward window:
  1. Check if pattern meets parameters
  2. Calculate position size (risk-based)
  3. Simulate entry (with costs)
  4. Track until target/stop/timeout
  5. Calculate P&L
  6. Update metrics
```

**Metrics Calculated**:
- Total trades, win rate, profit factor
- Average win, average loss
- Max drawdown, volatility
- Sharpe, Sortino, Calmar ratios
- Expectancy, win/loss ratio
- Temporal coverage, consistency score
- Invalidation statistics
- Regime breakdown

---

## Inference & Prediction

### Parallel Inference System

Located in `inference/parallel_inference.py`

**Purpose**: Efficient multi-model predictions with GPU support

**Features**:
- Parallel model execution (ThreadPoolExecutor)
- GPU acceleration (if available)
- Model caching
- Ensemble predictions
- Batch processing

**Workflow**:
```
1. MODEL LOADING
   ├── Load models from artifacts directory
   ├── Parse model metadata
   ├── Setup device (CPU/GPU)
   └── Cache loaded models

2. FEATURE PREPARATION
   ├── Load latest OHLCV data
   ├── Calculate features (matching training)
   ├── Apply standardization (from model scaler)
   ├── Apply dimensionality reduction (PCA/VAE)
   └── Format for model input

3. PARALLEL PREDICTION
   WITH ThreadPoolExecutor (max 4-8 workers):
     FOR each model:
       ├── Predict on latest data
       ├── Generate quantiles (q05, q25, q50, q75, q95)
       ├── Calculate confidence score
       └── Return predictions

4. ENSEMBLE AGGREGATION
   ├── Collect all model predictions
   ├── Weight by model confidence/performance
   ├── Calculate ensemble quantiles
   ├── Determine directional consensus
   └── Generate final signal

5. SIGNAL OUTPUT
   ├── Predicted price levels (per horizon)
   ├── Direction (long/short/neutral)
   ├── Confidence score (0-1)
   ├── Ensemble agreement
   └── Individual model predictions
```

**Model Executor** (per model):
```python
class ModelExecutor:
    model_path: str
    model_config: Dict
    model_data: Dict              # Loaded model + metadata
    is_loaded: bool
    device: torch.device          # CPU or CUDA
    
    def load_model() -> None
    def predict(features_df, candles_df) -> Dict
```

**Prediction Output**:
```python
{
    "model_name": "ridge_EURUSD_15m_4h",
    "predictions": {
        "1h": {"q05": 1.0950, "q50": 1.0980, "q95": 1.1010},
        "4h": {"q05": 1.0940, "q50": 1.0990, "q95": 1.1040},
        "1d": {"q05": 1.0920, "q50": 1.1000, "q95": 1.1080}
    },
    "direction": "long",
    "confidence": 0.75,
    "inference_time_ms": 15.3
}
```

### Multi-Timeframe Ensemble

Located in `models/multi_timeframe_ensemble.py`

**Purpose**: Aggregate predictions across multiple timeframes for robust signals

**Timeframes**:
```python
class Timeframe(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
```

**Aggregation Strategy**:
```
Weighted Average by Timeframe:
  - 1m: weight = 0.05  (short-term noise)
  - 5m: weight = 0.10
  - 15m: weight = 0.15
  - 1h: weight = 0.25  (primary signal)
  - 4h: weight = 0.25
  - 1d: weight = 0.20  (trend filter)

Consensus Requirement:
  - Minimum 3/6 timeframes agreeing on direction
  - Higher timeframe veto (1d can override shorter)
```

### ML Stacked Ensemble

Located in `models/ml_stacked_ensemble.py`

**Purpose**: Combine multiple model types (Ridge, RF, XGBoost, Neural Net, Diffusion)

**Stacking Levels**:
```
Level 1 (Base Models):
  ├── Ridge Regression
  ├── Random Forest
  ├── XGBoost
  ├── Neural Network
  └── Diffusion Model

Level 2 (Meta-Model):
  └── Gradient Boosting (combines Level 1 predictions)
```

**Prediction Flow**:
```
1. Each base model predicts independently
2. Predictions stacked as meta-features
3. Meta-model predicts final output
4. Confidence calculated from base model agreement
```

---

## Order Execution

### Smart Execution Optimizer

Located in `execution/smart_execution.py`

**Purpose**: Optimize order execution to minimize market impact and slippage

**Features**:
- TWAP/VWAP execution strategies
- Market impact estimation
- Spread analysis
- Liquidity-aware execution
- DOM (Depth of Market) integration

**Execution Strategies**:

#### 1. Market Order
```python
strategy: "market"
# Immediate execution at best available price
# Pros: Fast, guaranteed fill
# Cons: Higher slippage, market impact
```

#### 2. Limit Order
```python
strategy: "limit"
limit_price: float                # Max price for buy, min for sell
timeout_seconds: int = 60         # Order timeout
# Pros: Price control
# Cons: May not fill
```

#### 3. TWAP (Time-Weighted Average Price)
```python
strategy: "twap"
total_quantity: float
num_slices: int = 10              # Split into N orders
duration_seconds: int = 300       # Execute over 5 minutes
# Pros: Reduces market impact
# Cons: Longer execution time
```

#### 4. VWAP (Volume-Weighted Average Price)
```python
strategy: "vwap"
total_quantity: float
target_participation_rate: float = 0.1  # 10% of volume
# Pros: Follows market volume
# Cons: Requires volume data
```

#### 5. Iceberg Order
```python
strategy: "iceberg"
total_quantity: float
visible_quantity: float           # Visible portion
# Pros: Hides true order size
# Cons: More complex
```

**Smart Execution Logic**:
```
INPUT: Order (symbol, direction, quantity, urgency)

1. ANALYZE MARKET CONDITIONS
   ├── Current spread (bid-ask)
   ├── Order book depth (DOM)
   ├── Recent volatility
   ├── Average volume
   └── Market impact estimate

2. SELECT STRATEGY
   IF urgency == "high":
     RETURN market order
   ELIF quantity > 10% of avg volume:
     IF tight spread:
       RETURN twap order
     ELSE:
       RETURN iceberg order
   ELSE:
     IF spread < threshold:
       RETURN limit order (mid-price)
     ELSE:
       RETURN market order

3. EXECUTE
   ├── Place order via broker API
   ├── Monitor fill status
   ├── Track execution price
   └── Calculate realized slippage

4. METRICS
   ├── Execution time
   ├── Realized slippage vs expected
   ├── Market impact
   └── Fill rate
```

---

## Risk Management

### ⚠️ Position Sizer Duplication

**CRITICAL ISSUE**: Two separate position sizer implementations with overlapping functionality

#### Position Sizer 1: `risk/position_sizer.py`

**Methods**:
- Fixed fractional (risk fixed % per trade)
- Kelly Criterion (mathematically optimal from backtest data)
- Optimal f (Larry Williams method)
- Volatility-adjusted (based on ATR)

**Key Parameters**:
```python
base_risk_pct: float = 2.0        # Base risk per trade (2%)
kelly_fraction: float = 0.25      # Quarter Kelly (conservative)
max_position_size_pct: float = 5.0   # Max 5% per position
min_position_size_pct: float = 0.1   # Min 0.1%
max_total_exposure_pct: float = 20.0 # Max 20% total exposure
drawdown_reduction_enabled: bool = True
```

**Kelly Criterion**:
```
Kelly % = W - (1 - W) / R
Where:
  W = Win rate
  R = Average win / Average loss (win/loss ratio)

Example:
  W = 0.55 (55% win rate)
  R = 1.5 (avg win is 1.5x avg loss)
  Kelly % = 0.55 - (0.45 / 1.5) = 0.25 = 25%
  
  With quarter Kelly (conservative):
  Position size = 25% / 4 = 6.25% of capital
```

**Optimal f**:
```
Optimal f = -1 / (max consecutive losses * avg loss)
```

**Volatility-Adjusted**:
```
Position size = base_risk / (ATR * volatility_multiplier)
```

#### Position Sizer 2: `portfolio/position_sizer.py`

**⚠️ DUPLICATE with different API**

Implements similar methods but with portfolio-level constraints.

**Additional Features**:
- Portfolio-level risk limits
- Correlation-based sizing
- Sector exposure limits

**RECOMMENDATION**: Consolidate into single implementation in `risk/position_sizer.py`

### Multi-Level Stop Loss

Located in `risk/multi_level_stop_loss.py`

**Stop Types**:

#### 1. Technical Stop
```python
stop_type: "technical"
atr_multiplier: float = 2.0       # Stop at entry - (2 * ATR)
```

#### 2. Trailing Stop
```python
stop_type: "trailing"
trail_pct: float = 2.0            # Trail 2% behind highest price
activation_pct: float = 1.0       # Activate after 1% profit
```

#### 3. Time-Based Stop
```python
stop_type: "time"
max_bars: int = 20                # Exit after 20 bars
```

#### 4. Volatility Stop
```python
stop_type: "volatility"
volatility_multiplier: float = 1.5
# Adjusts stop based on current volatility
```

**Stop Logic**:
```
FOR each open position:
  current_price = market.get_price(position.symbol)
  
  # Calculate all stop levels
  technical_stop = entry_price - (atr * atr_multiplier)
  trailing_stop = highest_price * (1 - trail_pct / 100)
  volatility_stop = entry_price - (volatility * vol_multiplier)
  
  # Take tightest (closest to current price)
  effective_stop = max(technical_stop, trailing_stop, volatility_stop)
  
  # Check if triggered
  IF current_price <= effective_stop:
    CLOSE position at effective_stop
    RECORD exit reason and stop type
```

### Regime-Aware Position Sizing

Located in `risk/regime_position_sizer.py`

**Regimes**:
```python
class MarketRegime(Enum):
    TRENDING_BULL = "trending_bull"      # Strong uptrend
    TRENDING_BEAR = "trending_bear"      # Strong downtrend
    RANGE_BOUND = "range_bound"          # Sideways
    HIGH_VOLATILITY = "high_volatility"  # Choppy
    LOW_VOLATILITY = "low_volatility"    # Quiet
```

**Regime-Based Adjustments**:
```python
base_risk: float = 2.0%

Adjustments:
  trending_bull:      1.5x  (3.0% risk) - favorable for longs
  trending_bear:      1.5x  (3.0% risk) - favorable for shorts
  range_bound:        1.0x  (2.0% risk) - neutral
  high_volatility:    0.5x  (1.0% risk) - reduce risk
  low_volatility:     1.2x  (2.4% risk) - slightly increase
```

**Logic**:
```
1. DETECT REGIME
   regime = regime_detector.detect(market_data)

2. ADJUST BASE RISK
   risk_pct = base_risk * regime_multiplier[regime]

3. DIRECTIONAL ADJUSTMENT
   IF (regime == trending_bull AND direction == "long") OR
      (regime == trending_bear AND direction == "short"):
     risk_pct *= 1.2  # Bonus for trading with trend

4. CALCULATE POSITION SIZE
   position_size = (account_balance * risk_pct) / stop_distance
```

### Adaptive Stop Loss Manager

Located in `risk/adaptive_stop_loss_manager.py`

**Purpose**: Dynamically adjust stops based on market conditions and position performance

**Adaptation Factors**:
```python
@dataclass
class AdaptationFactors:
    regime: str                   # Current market regime
    volatility: float             # Current volatility (ATR-based)
    time_in_trade: int            # Bars since entry
    profit_pct: float             # Unrealized profit %
    win_streak: int               # Consecutive wins
    drawdown_pct: float           # Current drawdown from peak
```

**Adaptive Logic**:
```
INITIAL STOP:
  stop = entry_price - (atr * 2.0)

ADAPTATION:
  IF profit_pct > 2.0%:
    # Move to breakeven
    stop = max(stop, entry_price)
  
  IF profit_pct > 5.0%:
    # Lock in profit (trail at 3%)
    stop = max(stop, highest_price * 0.97)
  
  IF volatility > avg_volatility * 1.5:
    # Widen stop in high volatility
    stop = entry_price - (atr * 2.5)
  
  IF time_in_trade > max_holding_period / 2:
    # Tighten stop after half of max time
    stop = max(stop, current_price * 0.99)
  
  IF drawdown_pct > 3.0%:
    # Global tightening during drawdown
    stop *= 0.95  # 5% tighter
```

---

## Complete Workflow

### End-to-End Trading System Flow

```
═══════════════════════════════════════════════════════════
                    OFFLINE (Training)
═══════════════════════════════════════════════════════════

1. DATA COLLECTION
   ├── Download historical OHLCV (2+ years)
   ├── Store in database
   └── Quality checks

2. FEATURE ENGINEERING
   ├── Calculate technical indicators (BTALib)
   ├── Time-based features
   ├── Regime features
   └── Store feature set

3. MODEL TRAINING
   ├── FOR each (symbol, timeframe, horizon):
   │   ├── Load data + features
   │   ├── Train model (Ridge, RF, Diffusion, etc.)
   │   ├── Validate performance
   │   └── Save model + metadata
   └── Store in artifacts/

4. PATTERN OPTIMIZATION (Optional)
   ├── Define parameter space per pattern
   ├── Run NSGA-II optimization
   │   ├── Generate trial parameters
   │   ├── Backtest with parameters
   │   ├── Calculate multi-objective fitness
   │   └── Evolve population
   ├── Extract Pareto frontier
   ├── Validate parameters
   └── Store optimal parameters in database

5. BACKTESTING
   ├── Load models from artifacts/
   ├── Load optimal parameters (if available)
   ├── Run walk-forward backtest
   │   ├── Generate predictions
   │   ├── Simulate trades (with costs)
   │   ├── Apply risk management
   │   └── Calculate metrics
   ├── Evaluate performance
   │   ├── Sharpe, Sortino, Calmar
   │   ├── Max drawdown
   │   ├── Win rate, profit factor
   │   └── Regime breakdown
   └── Approve/reject for live trading

═══════════════════════════════════════════════════════════
                    ONLINE (Live Trading)
═══════════════════════════════════════════════════════════

6. REAL-TIME DATA INGESTION
   ├── Subscribe to broker data feed
   ├── Receive OHLCV updates (every bar close)
   ├── Update database
   └── Trigger prediction pipeline

7. INFERENCE
   ├── Load latest N bars (lookback window)
   ├── Calculate features
   ├── Load models from artifacts/
   ├── Parallel inference
   │   ├── Multi-timeframe predictions
   │   ├── Multi-model ensemble
   │   └── Aggregate predictions
   ├── Detect regime (HMM)
   └── Generate signal
       ├── Direction (long/short/neutral)
       ├── Confidence score
       ├── Target levels (from quantiles)
       └── Invalidation level

8. SIGNAL FILTERING
   ├── Minimum confidence check (e.g., > 0.6)
   ├── Regime alignment check
   ├── Timeframe consensus check
   ├── Pattern confirmation (if using patterns)
   └── Max positions check

9. POSITION SIZING
   ├── Load account balance
   ├── Detect current regime
   ├── Calculate base risk (1-2%)
   ├── Apply regime multiplier
   ├── Apply Kelly/Optimal f (if enabled)
   ├── Check max exposure limits
   └── Determine position size (lots)

10. RISK MANAGEMENT SETUP
    ├── Calculate stop loss
    │   ├── Technical (ATR-based)
    │   ├── Trailing (if profit > threshold)
    │   ├── Time-based (max holding period)
    │   └── Volatility-adjusted
    ├── Set take profit targets
    │   ├── Primary target (q95/q05)
    │   ├── Secondary target (stretch)
    │   └── Partial exit levels (optional)
    └── Set max holding period

11. ORDER EXECUTION
    ├── Analyze market conditions
    │   ├── Current spread
    │   ├── Order book depth (DOM)
    │   ├── Recent volatility
    │   └── Average volume
    ├── Select execution strategy
    │   ├── Market (urgent)
    │   ├── Limit (patient)
    │   ├── TWAP (large order)
    │   └── Iceberg (hide size)
    ├── Place order via broker API
    ├── Monitor fill
    └── Confirm execution

12. POSITION MONITORING
    FOR each open position (continuous):
      ├── Update current price
      ├── Check stop loss (all types)
      ├── Check take profit
      ├── Check max holding period
      ├── Update trailing stop (if active)
      ├── Check invalidation level
      ├── Check daily loss limit
      ├── Adapt stops (if enabled)
      └── Log position metrics

13. POSITION EXIT
    WHEN exit condition triggered:
      ├── Determine exit reason
      │   ├── Stop hit
      │   ├── Target reached
      │   ├── Time expired
      │   ├── Invalidated
      │   └── Daily limit
      ├── Execute exit order
      ├── Calculate P&L
      ├── Log trade details
      ├── Update statistics
      └── Update model performance tracking

14. PERFORMANCE TRACKING
    ├── Calculate real-time metrics
    │   ├── Win rate
    │   ├── Profit factor
    │   ├── Sharpe ratio (rolling)
    │   ├── Max drawdown (current)
    │   └── Expectancy
    ├── Compare to backtest results
    ├── Detect performance degradation
    ├── Trigger alerts (if needed)
    └── Store in database

15. AUTO-RETRAINING (Scheduled)
    ├── Check if retraining needed
    │   ├── Performance degradation > threshold
    │   ├── Time since last training > N days
    │   └── Regime change detected
    ├── Collect new data
    ├── Retrain models
    ├── Validate new models
    ├── Replace old models (if better)
    └── Restart inference pipeline

16. PARAMETER REFRESH (Scheduled)
    ├── Check parameter age
    ├── Evaluate recent performance with current parameters
    ├── IF degradation detected:
    │   ├── Queue re-optimization
    │   ├── Run NSGA-II on recent data
    │   ├── Validate new parameters
    │   └── Replace old parameters (if better)
    └── Log refresh decision
```

---

## Parameter Flow

### Training → Inference Flow

```
TRAINING OUTPUT (per model):
  artifacts/
    └── ridge_EURUSD_15m_4h/
        ├── model.pkl              # Trained model weights
        ├── features.json          # ["return_1", "return_5", "rsi_14", ...]
        ├── scaler.json            # {mu: {...}, sigma: {...}}
        ├── pca.pkl                # (Optional) PCA/VAE encoder
        ├── metadata.json          # {horizon: "4h", version: "1.0", ...}
        └── performance.json       # Backtest metrics

INFERENCE INPUT:
  1. Load model.pkl
  2. Read features.json → know which features needed
  3. Calculate features on latest data
  4. Load scaler.json → standardize features
  5. Load pca.pkl → apply dimensionality reduction
  6. Predict with model
  7. Read metadata.json → interpret prediction (horizon, etc.)
```

### Optimization → Execution Flow

```
OPTIMIZATION OUTPUT:
  database: optimization_studies table
    study_id: 12345
    pattern_key: "wedge_ascending"
    asset: "EURUSD"
    timeframe: "15m"
    regime: "trending_bull"
    best_parameters: {
      "min_span": 35,
      "max_span": 180,
      "min_touches": 5,
      "tolerance": 0.08,
      "entry_threshold": 0.002,
      "target_multiplier": 2.5,
      "stop_multiplier": 1.0
    }
    performance: {
      "success_rate": 0.62,
      "profit_factor": 1.85,
      "sharpe_ratio": 1.42,
      "max_drawdown": 0.08
    }
    last_updated: "2025-10-13"

EXECUTION USAGE:
  1. Detect pattern (live)
  2. Query database for optimal parameters:
     WHERE pattern_key = "wedge_ascending"
       AND asset = "EURUSD"
       AND timeframe = "15m"
       AND regime = current_regime
  3. Load best_parameters
  4. Apply parameters to pattern detection
  5. If pattern confirmed:
     - entry_threshold → signal filter
     - target_multiplier → set take profit
     - stop_multiplier → set stop loss
```

### Risk Parameters Flow

```
CONFIGURATION (config file or database):
  risk_config:
    base_risk_per_trade_pct: 2.0
    position_sizing_method: "kelly"
    kelly_fraction: 0.25
    max_position_size_pct: 5.0
    max_total_exposure_pct: 20.0
    
  stop_loss_config:
    use_multi_level: true
    atr_multiplier: 2.0
    trail_activation_pct: 1.0
    trail_pct: 2.0
    max_holding_hours: 24
    
  regime_adjustments:
    trending_bull: 1.5
    trending_bear: 1.5
    range_bound: 1.0
    high_volatility: 0.5

EXECUTION USAGE:
  1. Signal generated (long EURUSD, confidence 0.75)
  2. Detect regime → "trending_bull"
  3. Calculate position size:
     a. Base risk = 2.0% * 1.5 (regime) = 3.0%
     b. Apply directional bonus (long in bull) = 3.0% * 1.2 = 3.6%
     c. Apply Kelly adjustment (if enabled)
     d. Check max limits (5.0%, 20.0% total)
     e. Final: 3.6% of account
  4. Calculate stops:
     a. Technical = entry - (atr * 2.0)
     b. Trailing (activate after 1% profit, trail 2%)
     c. Time-based (exit after 24 hours)
     d. Use tightest of all stops
```

---

## Critical Issues Identified

### 1. MASSIVE CODE DUPLICATION

#### Backtest Engines (3 separate)
- `backtest/engine.py`
- `backtesting/forecast_backtest_engine.py`  
- `backtest/integrated_backtest.py`

**Impact**: HIGH  
**Issue**: Unclear which to use, inconsistent results, maintenance nightmare

**Recommendation**: 
1. Consolidate into single `backtest/engine.py`
2. Support multiple strategies as plugins
3. Clear hierarchy: basic → forecast → integrated

#### Position Sizers (2 separate)
- `risk/position_sizer.py`
- `portfolio/position_sizer.py`

**Impact**: MEDIUM  
**Issue**: Duplicate logic, inconsistent APIs, confusion

**Recommendation**: 
1. Keep `risk/position_sizer.py` (more comprehensive)
2. Move portfolio-specific logic to `portfolio/optimizer.py`
3. Delete duplicate

#### Broker Directories (2 separate)
- `broker/` (single file)
- `brokers/` (multiple files)

**Impact**: LOW  
**Issue**: Organizational confusion

**Recommendation**: 
1. Consolidate to `brokers/`
2. Delete `broker/` directory

#### Training Pipeline Directories (2 separate)
- `training/training_pipeline/`
- `training_pipeline/`

**Impact**: HIGH  
**Issue**: Critical duplication, import confusion

**Recommendation**: 
1. Keep `training/training_pipeline/` (better organized)
2. Move useful code from root `training_pipeline/`
3. Delete root directory

### 2. TRAINING SCRIPT PROLIFERATION

**7 different training scripts** with overlapping functionality:
- `train.py`
- `train_sklearn.py`
- `train_sklearn_btalib.py`
- `train_optimized.py`
- `train_sssd.py`
- `optimized_trainer.py`
- `auto_retrain.py`

**Impact**: HIGH  
**Issue**: 
- Unclear which to use
- No clear documentation
- Different feature sets
- Inconsistent outputs

**Recommendation**:
1. **Keep 3 main scripts**:
   - `train_sklearn.py` → Ridge, RF, XGBoost (standard)
   - `train_sssd.py` → Diffusion models only
   - `auto_retrain.py` → Scheduled retraining

2. **Consolidate features**:
   - Merge BTALib features into standard pipeline
   - Make GPU optimization a flag, not separate script

3. **Deprecate/remove**:
   - `train.py` → merge into train_sklearn.py
   - `train_optimized.py` → merge GPU features into train_sklearn.py
   - `optimized_trainer.py` → unclear purpose, remove if duplicate

### 3. IMPORT INCONSISTENCIES

**Mixed imports**:
```python
# Some files use:
from ..backtest.engine import BacktestEngine

# Others use:
from ..backtesting.advanced_backtest_engine import AdvancedBacktestEngine

# No clear standard
```

**Impact**: MEDIUM  
**Issue**: Confusion, import errors, circular dependencies

**Recommendation**:
1. Standardize on `backtest/` after consolidation
2. Update all imports
3. Add import guidelines to documentation

### 4. MISSING INTEGRATION POINTS

**Parameter Refresh Manager** (`patterns/parameter_refresh_manager.py`):
- Core logic implemented
- **NOT connected** to database queries
- **NOT connected** to OptimizationEngine
- **NOT connected** to live trading

**Impact**: MEDIUM  
**Issue**: Feature exists but not functional

**Recommendation**:
1. Implement database queries (OptimizationStudy model)
2. Connect to OptimizationEngine for reoptimization
3. Add scheduler for automatic checks

### 5. BACKTEST RESULT INCONSISTENCIES

**Different metrics** across engines:

| Metric | engine.py | forecast_backtest_engine.py | integrated_backtest.py |
|--------|-----------|----------------------------|------------------------|
| Sharpe | ✅ | ❌ | ✅ |
| CRPS | ❌ | ✅ | ❌ |
| Regime breakdown | ❌ | ❌ | ✅ |
| Transaction costs | ✅ | N/A | ✅ |

**Impact**: HIGH  
**Issue**: Cannot compare results across engines

**Recommendation**:
1. Define standard metric set
2. Implement in all engines
3. Create unified result format

### 6. NO UNIFIED CONFIGURATION

**Multiple config locations**:
- `configs/` directory
- Hardcoded in scripts
- Database parameters
- Environment variables

**Impact**: MEDIUM  
**Issue**: Hard to understand what parameters are being used

**Recommendation**:
1. Create `TradingSystemConfig` dataclass
2. Centralize all configuration
3. Support overrides from CLI/env

### 7. BROKER API ABSTRACTION INCOMPLETE

**Multiple broker integrations** with inconsistent APIs:
- `brokers/base.py` - Abstract base class
- `brokers/fxpro_ctrader.py` - cTrader implementation
- `broker/ctrader_broker.py` - Another cTrader implementation

**Impact**: LOW  
**Issue**: Incomplete abstraction, duplication

**Recommendation**:
1. Consolidate cTrader implementations
2. Complete base class interface
3. Add paper trading broker for testing

### 8. MISSING ERROR HANDLING IN LIVE TRADING

**`automated_trading_engine.py`** has minimal error handling:
```python
# Current:
try:
    execute_trade()
except Exception as e:
    logger.error(f"Error: {e}")  # Just log, no recovery

# Should be:
try:
    execute_trade()
except BrokerConnectionError:
    reconnect_broker()
    retry_execute()
except InsufficientFundsError:
    reduce_position_size()
    retry_execute()
except Exception as e:
    emergency_close_all_positions()
    alert_administrator()
```

**Impact**: CRITICAL  
**Issue**: Silent failures in live trading can cause losses

**Recommendation**:
1. Add comprehensive error handling
2. Implement retry logic
3. Add emergency procedures
4. Real-time alerting

### 9. NO PERFORMANCE DEGRADATION DETECTION

**Missing**: System to detect when live performance diverges from backtest

**Impact**: HIGH  
**Issue**: No early warning of system failure

**Recommendation**:
1. Track rolling metrics (Sharpe, win rate, etc.)
2. Compare to backtest expectations
3. Alert if degradation > threshold
4. Auto-pause trading if severe

### 10. TRANSACTION COST MODELING INCONSISTENT

**Different costs** across engines:
```python
# engine.py
spread_pips: float = 0.5
slippage_pips: float = 0.2

# integrated_backtest.py
spread_pct: float = 0.0002
commission_pct: float = 0.0001
slippage_pct: float = 0.0001

# Not clear if these are consistent
```

**Impact**: MEDIUM  
**Issue**: Backtest accuracy depends on realistic costs

**Recommendation**:
1. Measure actual costs from broker
2. Standardize cost model
3. Make configurable per asset

---

## Summary Statistics

**Total Files Analyzed**: 50+  
**Duplicated Modules**: 8  
**Import Inconsistencies**: 20+  
**Critical Issues**: 10  
**Code Cleanup Needed**: HIGH  

**Estimated Effort to Fix All Issues**: 40-60 hours

---

**Document Generated**: 2025-10-13  
**Analyzer**: Factory AI Droid  
**Status**: COMPREHENSIVE REVIEW COMPLETE
