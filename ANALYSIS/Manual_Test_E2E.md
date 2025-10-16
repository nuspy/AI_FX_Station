# ForexGPT - Manual End-to-End Test Procedure

**Document Version**: 1.0  
**Date**: 2025-01-08  
**Purpose**: Comprehensive manual testing procedure for complete system validation  
**Test Duration**: ~8-12 hours (full suite)  
**Tester Prerequisites**: Windows 10+, Python 3.12+, 16GB RAM, cTrader account

---

## Table of Contents

1. [Test Environment Setup](#1-test-environment-setup)
2. [Test 1: Manual Model Training](#test-1-manual-model-training)
3. [Test 2: Model Backtesting](#test-2-model-backtesting)
4. [Test 3: Live Models Training & Optimization](#test-3-live-models-training--optimization)
5. [Test 4: Pattern Training & Optimization](#test-4-pattern-training--optimization)
6. [Test 5: Trading Engine Configuration](#test-5-trading-engine-configuration)
7. [Test 6: Integrated System Test](#test-6-integrated-system-test)
8. [Test 7: Real-Time Data Flow](#test-7-real-time-data-flow)
9. [Results Validation](#results-validation)
10. [Troubleshooting Guide](#troubleshooting-guide)

---

## 1. Test Environment Setup

### 1.1 Prerequisites Check

**Command Line**:
```bash
# Check Python version
python --version
# Expected: Python 3.12.x

# Check installed packages
python -m pip list | findstr "xgboost lightgbm scikit-learn PySide6"
# Expected: All packages present

# Check database
ls forexgpt.db
# Expected: File exists (or will be created)

# Check git status
git status
# Expected: On branch Debug-2025108 or main
```

**Expected Output**:
```
Python 3.12.10
xgboost        3.0.0
lightgbm       4.0.0
scikit-learn   1.3.x
PySide6        6.5.x
forexgpt.db exists
```

### 1.2 Database Initialization

**Steps**:
1. Open PowerShell in project root: `D:\Projects\ForexGPT`
2. Run: `python -c "from src.forex_diffusion.services.database_service import DatabaseService; db = DatabaseService(); print('DB initialized')"`
3. Verify tables exist: `python -c "from src.forex_diffusion.services.database_service import DatabaseService; db = DatabaseService(); from sqlalchemy import inspect; insp = inspect(db.engine); print('Tables:', insp.get_table_names())"`

**Expected Output**:
```
DB initialized
Tables: ['bars_1m', 'bars_5m', 'bars_15m', 'bars_1h', 'bars_4h', 'bars_1d', 
         'forecast_results', 'model_metadata', 'optimization_results', 
         'pattern_configs', 'sentiment_data', 'dom_data', 'vix_data', ...]
```

### 1.3 Data Download (if needed)

**For Test Data**:
```bash
# Download EUR/USD 1H data for last 2 years
python scripts/download_historical_data.py --symbol EUR/USD --timeframe 1h --days 730

# Verify data
python -c "from src.forex_diffusion.data.sqlite_data_adapter import SQLiteDataAdapter; adapter = SQLiteDataAdapter('forexgpt.db'); df = adapter.get_bars('EUR/USD', '1h', limit=100); print(f'Data rows: {len(df)}'); print(df.tail())"
```

**Expected Output**:
```
Data rows: 17520 (730 days * 24 hours)
Last 5 bars displayed with ts_utc, open, high, low, close, volume
```

---

## Test 1: Manual Model Training

**Objective**: Train a single Ridge regression model for EUR/USD 1H with 4-hour horizon.

### 1.1 Launch Application

**Command**:
```bash
python run_forexgpt.py
```

**Expected**:
- Main window opens
- Chart tab visible
- Training tab visible in main tab bar

### 1.2 Navigate to Training Tab

**Steps**:
1. Click "Training" tab in main tab bar
2. Verify training interface loads with:
   - Symbol selector (dropdown)
   - Timeframe selector (dropdown)
   - Algorithm selector (dropdown)
   - Horizon selector (spinbox)
   - Feature selectors (checkboxes)
   - Training controls (buttons)

### 1.3 Configure Training Parameters

**Parameters to Set**:

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| **Symbol** | EUR/USD | Top dropdown | Primary trading pair |
| **Timeframe** | 1h | Timeframe dropdown | 1-hour candles |
| **Algorithm** | Ridge Regression | Algorithm dropdown | Baseline model |
| **Horizon** | 4 | Horizon spinbox (hours) | 4-hour forecast |
| **Lookback Bars** | 2000 | Lookback spinbox | Training data size |
| **Train/Test Split** | 80/20 | Split slider or spinbox | Standard split |

**Features to Enable** (checkboxes):
- ✅ ATR (Average True Range)
- ✅ RSI (Relative Strength Index)
- ✅ MACD (Moving Average Convergence Divergence)
- ✅ Bollinger Bands
- ✅ EMA (Exponential Moving Average)
- ✅ SMA (Simple Moving Average)
- ✅ ADX (Average Directional Index)
- ✅ Stochastic
- ⬜ Leave others unchecked for baseline test

**Additional Features**:
- ✅ Returns & Volatility
- ✅ Trading Sessions
- ⬜ Candlestick Patterns (skip for speed)
- ⬜ Volume Profile (skip for speed)

### 1.4 Execute Training

**Steps**:
1. Click "Start Training" button
2. Observe progress bar appear
3. Monitor log output in bottom panel

**Expected Progress**:
```
[Training] Loading data for EUR/USD @ 1h...
[Training] Data loaded: 2000 bars
[Training] Extracting features... (8 indicators + 2 additional)
[Training] Feature extraction complete: 45 features
[Training] Train/Test split: 1600 train, 400 test
[Training] Training Ridge Regression model...
[Training] Training complete in 3.2 seconds
[Training] Validation metrics:
  - MAE: 0.000234 (23.4 pips on EUR/USD)
  - RMSE: 0.000412 (41.2 pips)
  - R²: 0.234
  - Directional Accuracy: 58.3%
[Training] Model saved to: artifacts/ridge_eurusd_1h_4h_20250108.pkl
[Training] Metadata saved to database: model_id=1
```

**Validation Criteria**:
- ✅ Training completes without errors
- ✅ Directional Accuracy >= 52% (above random 50%)
- ✅ R² > 0.10 (some predictive power)
- ✅ Model file exists in artifacts/
- ✅ Metadata saved to database (check model_metadata table)

### 1.5 Verify Model Saved

**Database Check**:
```bash
python -c "from src.forex_diffusion.services.database_service import DatabaseService; from sqlalchemy import text; db = DatabaseService(); with db.engine.connect() as conn: result = conn.execute(text('SELECT * FROM model_metadata ORDER BY created_at DESC LIMIT 1')); print(dict(result.fetchone()))"
```

**Expected Output**:
```python
{
    'id': 1,
    'model_type': 'ridge',
    'symbol': 'EUR/USD',
    'timeframe': '1h',
    'horizon': 4,
    'features': ['atr_14', 'rsi_14', 'macd', ...],
    'mae': 0.000234,
    'rmse': 0.000412,
    'r2': 0.234,
    'directional_accuracy': 0.583,
    'file_path': 'artifacts/ridge_eurusd_1h_4h_20250108.pkl',
    'created_at': '2025-01-08 10:30:15'
}
```

**File Check**:
```bash
ls artifacts/ridge_eurusd_1h_4h_*.pkl
# Expected: File exists, size ~50KB-5MB
```

---

## Test 2: Model Backtesting

**Objective**: Backtest the trained Ridge model on out-of-sample data.

### 2.1 Navigate to Backtesting Tab

**Steps**:
1. Click "Backtesting" tab in main tab bar
2. Wait for tab to load (may take 2-3 seconds)
3. Verify interface shows:
   - Model selector dropdown (populated with available models)
   - Date range pickers
   - Backtest configuration panel
   - Results display area

### 2.2 Configure Backtest Parameters

**Parameters**:

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| **Model** | ridge_eurusd_1h_4h_20250108 | Model dropdown | Select trained model |
| **Start Date** | 2024-06-01 | Date picker | Out-of-sample period |
| **End Date** | 2024-12-31 | Date picker | Recent data |
| **Initial Capital** | $10,000 | Capital spinbox | Standard test amount |
| **Risk Per Trade** | 1.0% | Risk spinbox | Conservative risk |
| **Position Sizing** | Fixed Fractional | Dropdown | Simple method |
| **Stop Loss** | 1.5× ATR | SL config | Risk management |
| **Take Profit** | 3.0× ATR | TP config | 2:1 reward/risk |
| **Commission** | 0.0002 | Commission spinbox | 2 pips |
| **Slippage** | 0.0001 | Slippage spinbox | 1 pip |

**Execution Rules**:
- ✅ Only trade predicted moves >0.3%
- ✅ Max 1 position per symbol
- ✅ No trading during news events (if calendar integrated)
- ✅ Close positions before weekend

### 2.3 Run Backtest

**Steps**:
1. Click "Run Backtest" button
2. Observe progress indicators:
   - Progress bar (0-100%)
   - Current date being processed
   - Trades executed count
   - Current P&L
3. Wait for completion (estimated 2-5 minutes for 6 months of 1H data)

**Expected Progress Output**:
```
[Backtest] Initializing engine...
[Backtest] Loading model: ridge_eurusd_1h_4h_20250108
[Backtest] Loading data: EUR/USD 1h from 2024-06-01 to 2024-12-31
[Backtest] Data loaded: 4380 bars (6 months)
[Backtest] Starting simulation...
[Backtest] Progress: 10% | Date: 2024-06-15 | Trades: 5 | P&L: +$123.45
[Backtest] Progress: 25% | Date: 2024-07-15 | Trades: 12 | P&L: +$287.90
[Backtest] Progress: 50% | Date: 2024-09-01 | Trades: 24 | P&L: +$542.33
[Backtest] Progress: 75% | Date: 2024-10-15 | Trades: 38 | P&L: +$821.67
[Backtest] Progress: 100% | Date: 2024-12-31 | Trades: 52 | P&L: +$1,234.56
[Backtest] Simulation complete!
[Backtest] Calculating metrics...
```

### 2.4 Review Backtest Results

**Expected Metrics**:

```
=== BACKTEST SUMMARY ===

Performance Metrics:
  Total Return: +12.35% ($1,234.56)
  Annualized Return: ~24.7%
  Total Trades: 52
  Winning Trades: 30 (57.7%)
  Losing Trades: 22 (42.3%)
  Win Rate: 57.7%
  
  Avg Win: +$65.23 (+0.65%)
  Avg Loss: -$32.11 (-0.32%)
  Win/Loss Ratio: 2.03:1
  Profit Factor: 1.52
  Expectancy: +$23.74 per trade
  
Risk Metrics:
  Sharpe Ratio: 1.38
  Sortino Ratio: 2.05
  Calmar Ratio: 2.73
  Max Drawdown: -$456.78 (-4.57%)
  Max Drawdown Duration: 12 days
  Recovery Time: 8 days
  
  VaR (95%): -$78.90 (-0.79%)
  CVaR (95%): -$112.34 (-1.12%)
  
Trade Analysis:
  Avg Trade Duration: 6.2 hours
  Max Trade Duration: 18 hours
  Longest Winning Streak: 6 trades
  Longest Losing Streak: 4 trades
  
  Best Trade: +$145.67 (+1.46%)
  Worst Trade: -$89.23 (-0.89%)
```

**Validation Criteria**:
- ✅ Total Return > 0% (profitable)
- ✅ Win Rate > 52% (edge over random)
- ✅ Profit Factor > 1.3 (consistent edge)
- ✅ Sharpe Ratio > 1.0 (risk-adjusted return acceptable)
- ✅ Max Drawdown < 15% (risk manageable)
- ⚠️ If any criteria fails, model may need retraining or parameter tuning

### 2.5 Examine Equity Curve

**Steps**:
1. Scroll to "Equity Curve" chart in results panel
2. Verify chart shows:
   - Smooth upward trend (no vertical drops)
   - Drawdown periods recover within reasonable time
   - No prolonged flat periods

**Red Flags**:
- ❌ Vertical drops >10% (stop loss not working)
- ❌ Prolonged flat periods >2 months (model stopped working)
- ❌ Equity curve resembles staircase (overfitting to specific periods)

### 2.6 Export Backtest Report

**Steps**:
1. Click "Export Results" button
2. Save to: `D:\Projects\ForexGPT\analysis\backtest_ridge_eurusd_20250108.json`
3. Verify file contains complete backtest data

**File Verification**:
```bash
python -c "import json; data = json.load(open('analysis/backtest_ridge_eurusd_20250108.json')); print('Trades:', len(data['trades'])); print('Final Capital:', data['final_capital'])"
```

**Expected Output**:
```
Trades: 52
Final Capital: 11234.56
```

---

## Test 3: Live Models Training & Optimization

**Objective**: Train and optimize the complete ensemble used in live trading.

### 3.1 Multi-Timeframe Ensemble Training

**Navigate to**: Training Tab → "Multi-Timeframe Ensemble" section

**Parameters**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Base Models** | Ridge, Lasso, ElasticNet, Random Forest, XGBoost, LightGBM | All 6 models |
| **Timeframes** | 1h, 4h, 1d | 3 timeframes (can expand to 6) |
| **Horizons** | 1h, 4h, 8h | 3 horizons |
| **Training Period** | 2022-01-01 to 2024-06-30 | 2.5 years |
| **Validation Period** | 2024-07-01 to 2024-12-31 | 6 months |
| **Walk-Forward** | Enabled | Rolling window validation |
| **WF Window** | 90 days | 3-month training window |
| **WF Step** | 30 days | 1-month step |

**Expected Training Matrix**:
```
Total Combinations: 6 models × 3 TFs × 3 horizons = 54 individual models

Training Schedule:
  1h models: 18 models × ~5 min = 90 min
  4h models: 18 models × ~3 min = 54 min
  1d models: 18 models × ~1 min = 18 min
  Total estimated time: 162 minutes (~2.7 hours)
```

### 3.2 Execute Ensemble Training

**Steps**:
1. Click "Train All Combinations" button
2. Confirm warning dialog (long training time)
3. Monitor progress:
   - Overall progress bar
   - Current model being trained
   - Estimated time remaining
   - Model validation metrics as they complete

**Console Output Sample**:
```
[MTF Ensemble] Training 54 model combinations...
[MTF Ensemble] Progress: 1/54 | Ridge @ 1h → 1h | MAE: 0.000234 | Dir Acc: 58.3%
[MTF Ensemble] Progress: 2/54 | Ridge @ 1h → 4h | MAE: 0.000456 | Dir Acc: 57.1%
...
[MTF Ensemble] Progress: 18/54 | LightGBM @ 1h → 8h | MAE: 0.000198 | Dir Acc: 61.2%
[MTF Ensemble] Progress: 19/54 | Ridge @ 4h → 1h | MAE: 0.000312 | Dir Acc: 56.8%
...
[MTF Ensemble] Progress: 54/54 | LightGBM @ 1d → 8h | MAE: 0.000289 | Dir Acc: 59.4%
[MTF Ensemble] All models trained successfully!
[MTF Ensemble] Saving ensemble metadata...
[MTF Ensemble] Training complete in 2h 38m
```

### 3.3 Meta-Learner Training (Stacked Ensemble)

**Automatic After Ensemble Training**:

**Steps**:
1. After base models complete, system automatically trains meta-learner
2. Meta-learner uses predictions from all 54 base models as features
3. Trains Ridge regression to combine predictions

**Expected Output**:
```
[Stacked Ensemble] Training meta-learner...
[Stacked Ensemble] Base predictions: 54 features
[Stacked Ensemble] Additional features: 15 aggregate metrics
[Stacked Ensemble] Total input features: 69
[Stacked Ensemble] Training Ridge meta-learner...
[Stacked Ensemble] Meta-learner validation:
  - MAE: 0.000187 (18.7 pips) - BEST!
  - Directional Accuracy: 62.4% - BEST!
  - Sharpe (validation): 1.67
[Stacked Ensemble] Meta-learner saved
[Stacked Ensemble] Ensemble ready for production
```

### 3.4 Walk-Forward Validation

**Automatic During Training**:

**Results to Check**:
```
=== WALK-FORWARD VALIDATION RESULTS ===

Window 1 (Train: 2022-01-01 to 2022-03-31, Test: 2022-04-01 to 2022-04-30):
  Train MAE: 0.000198 | Test MAE: 0.000234 | Degradation: +18.2%
  Train Acc: 61.2% | Test Acc: 58.7% | Degradation: -2.5%

Window 2 (Train: 2022-02-01 to 2022-04-30, Test: 2022-05-01 to 2022-05-31):
  Train MAE: 0.000203 | Test MAE: 0.000241 | Degradation: +18.7%
  Train Acc: 60.8% | Test Acc: 57.9% | Degradation: -2.9%

... (more windows) ...

Window 9 (Train: 2024-04-01 to 2024-06-30, Test: 2024-07-01 to 2024-07-31):
  Train MAE: 0.000189 | Test MAE: 0.000229 | Degradation: +21.2%
  Train Acc: 62.1% | Test Acc: 59.3% | Degradation: -2.8%

Average Degradation:
  MAE: +19.4% (acceptable, <25%)
  Accuracy: -2.7% (acceptable, <5%)

✅ Walk-forward validation PASSED
```

**Validation Criteria**:
- ✅ Average MAE degradation <25% (train to test)
- ✅ Average accuracy degradation <5%
- ✅ No single window with >50% degradation (overfitting)
- ✅ Consistent performance across all windows

### 3.5 Hyperparameter Optimization (NSGA-II)

**Navigate to**: Training Tab → "Optimization" section

**Parameters**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimization Method** | NSGA-II | Multi-objective genetic algorithm |
| **Objectives** | Sharpe Ratio (max), Max Drawdown (min) | Pareto optimization |
| **Population Size** | 50 | Individuals per generation |
| **Generations** | 30 | Evolution iterations |
| **Crossover Prob** | 0.8 | Genetic crossover rate |
| **Mutation Prob** | 0.2 | Genetic mutation rate |
| **Eval Method** | Walk-Forward | Backtest each individual |

**Parameters to Optimize**:
- `alpha` (regularization): [0.001, 10.0]
- `l1_ratio` (ElasticNet): [0.0, 1.0]
- `max_depth` (tree models): [3, 15]
- `n_estimators` (ensemble): [50, 500]
- `learning_rate` (boosting): [0.01, 0.3]

**Execution**:
1. Click "Start Optimization" button
2. Estimated time: 4-6 hours (50 pop × 30 gen × ~15 seconds per eval)
3. Can pause/resume if needed

**Expected Output**:
```
[NSGA-II] Generation 1/30 | Best Sharpe: 1.42 | Best DD: -18.3%
[NSGA-II] Generation 5/30 | Best Sharpe: 1.58 | Best DD: -16.7%
[NSGA-II] Generation 10/30 | Best Sharpe: 1.71 | Best DD: -14.2%
[NSGA-II] Generation 20/30 | Best Sharpe: 1.83 | Best DD: -12.8%
[NSGA-II] Generation 30/30 | Best Sharpe: 1.89 | Best DD: -11.5%
[NSGA-II] Optimization complete!
[NSGA-II] Pareto front contains 8 solutions
[NSGA-II] Recommended solution: Sharpe=1.89, DD=-11.5%
  alpha: 0.123
  l1_ratio: 0.456
  max_depth: 7
  n_estimators: 234
  learning_rate: 0.078
[NSGA-II] Promoting optimized parameters to production...
✅ Optimization results saved to database
```

### 3.6 Verify Optimized Models

**Database Check**:
```bash
python -c "from src.forex_diffusion.services.database_service import DatabaseService; from sqlalchemy import text; db = DatabaseService(); with db.engine.connect() as conn: result = conn.execute(text('SELECT COUNT(*) FROM model_metadata WHERE is_optimized = 1')); print('Optimized models:', result.fetchone()[0])"
```

**Expected Output**:
```
Optimized models: 54
```

---

## Test 4: Pattern Training & Optimization

**Objective**: Train and optimize chart and candlestick pattern parameters using genetic algorithm.

### 4.1 Navigate to Pattern Training Tab

**Steps**:
1. Click "Pattern Training" tab in main tab bar
2. Wait for interface to load
3. Verify sections visible:
   - Training Type (Chart vs Candlestick)
   - Dataset Configuration
   - Parameter Space
   - Optimization Configuration
   - Execution Controls
   - Results Display

### 4.2 Chart Patterns Training

**Configuration**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Training Type** | Chart Patterns | Radio button |
| **Start Date** | 2020-01-01 | 5 years of data |
| **End Date** | 2024-12-31 | Recent data |
| **Assets** | EUR/USD, GBP/USD, USD/JPY | Major pairs |
| **Timeframes** | 1h, 4h | Multi-timeframe |
| **Patterns** | Head & Shoulders, Double Top/Bottom, Triangles, Flags | 4 major patterns |

**Parameter Space** (ranges to optimize):

```yaml
Template Matching Threshold: [0.75, 0.95]
  Current: 0.85
  Optimal: Find best balance (false positives vs false negatives)

Neckline Tolerance: [0.01, 0.10]  # 1-10%
  Current: 0.05
  Optimal: Pattern boundary flexibility

Symmetry Requirement: [0.60, 0.95]
  Current: 0.80
  Optimal: How symmetric pattern must be

Volume Confirmation: [1.2, 2.5]  # Multiplier of average volume
  Current: 1.5
  Optimal: Breakout volume requirement

Fibonacci Tolerance: [0.01, 0.05]  # 1-5%
  Current: 0.03
  Optimal: Harmonic pattern ratio tolerance

Min Pattern Bars: [10, 50]
  Current: 20
  Optimal: Minimum candles for valid pattern

Max Pattern Bars: [30, 200]
  Current: 100
  Optimal: Maximum candles for pattern completion
```

**Genetic Algorithm Config**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Population** | 40 | Individuals per generation |
| **Generations** | 25 | Evolution iterations |
| **Objective 1** | Maximize Profit Factor | Primary goal |
| **Objective 2** | Minimize False Positives | Secondary goal |
| **Crossover** | 0.75 | Breeding rate |
| **Mutation** | 0.25 | Random variation rate |

**Execution**:
1. Select "Chart Patterns" radio button
2. Configure all parameters above
3. Click "Start Training" button
4. Monitor progress (estimated 2-3 hours)

**Expected Output**:
```
[Pattern Training] Initializing genetic algorithm...
[Pattern Training] Loading historical data...
[Pattern Training] Data loaded: 3 symbols × 2 TFs × 5 years = 87,600 bars
[Pattern Training] Detecting patterns with baseline parameters...
[Pattern Training] Baseline: 247 patterns detected
[Pattern Training] Starting evolution...

Generation 1/25:
  Best Individual: Profit Factor=1.42, False Positive Rate=28.3%
  Population Stats: Avg PF=1.18, Avg FPR=35.7%

Generation 5/25:
  Best Individual: Profit Factor=1.67, False Positive Rate=22.1%
  Top 5 Solutions in Pareto Front

Generation 10/25:
  Best Individual: Profit Factor=1.83, False Positive Rate=18.5%
  Pareto Front: 7 solutions

Generation 25/25:
  Best Individual: Profit Factor=2.04, False Positive Rate=15.2%
  Pareto Front: 12 solutions

[Pattern Training] Evolution complete!
[Pattern Training] Recommended solution (balanced trade-off):
  Template Threshold: 0.88 (was 0.85)
  Neckline Tolerance: 0.038 (was 0.05)
  Symmetry: 0.85 (was 0.80)
  Volume Confirmation: 1.82 (was 1.5)
  Fibonacci Tolerance: 0.024 (was 0.03)
  Min Bars: 15 (was 20)
  Max Bars: 78 (was 100)

Performance Improvement:
  Profit Factor: 1.42 → 2.04 (+43.7%)
  False Positive Rate: 28.3% → 15.2% (-46.3%)
  Win Rate: 54.7% → 61.3% (+6.6%)
  Avg Win/Loss: 1.8:1 → 2.4:1 (+33.3%)

✅ Optimized parameters saved to database
```

**Validation**:
```bash
python -c "from src.forex_diffusion.services.database_service import DatabaseService; from sqlalchemy import text; db = DatabaseService(); with db.engine.connect() as conn: result = conn.execute(text(\"SELECT pattern_type, profit_factor, false_positive_rate FROM pattern_configs WHERE is_optimized = 1\")); print('Optimized patterns:'); [print(dict(row)) for row in result]"
```

### 4.3 Candlestick Patterns Training

**Configuration**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Training Type** | Candlestick Patterns | Radio button |
| **Patterns** | Doji, Hammer, Engulfing, Morning Star, Evening Star, Three White Soldiers | 6 common patterns |

**Parameter Space**:

```yaml
Body/Shadow Ratio: [0.1, 0.5]
  Current: 0.3
  Optimal: Doji/Hammer body size

Engulfing Percentage: [0.5, 1.0]  # How much body must engulf
  Current: 0.8
  Optimal: Engulfing pattern strictness

Confirmation Bars: [1, 5]
  Current: 2
  Optimal: Bars to wait for confirmation

ATR Multiplier: [0.5, 3.0]
  Current: 1.0
  Optimal: Stop loss distance
```

**Execution**:
1. Select "Candlestick Patterns" radio button
2. Configure parameters
3. Click "Start Training"
4. Monitor (estimated 1-2 hours, faster than chart patterns)

**Expected Output**:
```
[Candlestick Training] Generation 20/20 complete
[Candlestick Training] Optimized parameters:
  Doji Body Ratio: 0.15 (very small body)
  Hammer Body Ratio: 0.28
  Engulfing: 0.92 (strict requirement)
  Confirmation Bars: 3 (wait 3 bars)
  ATR Multiplier: 1.4

Performance:
  Win Rate: 56.8% → 62.1% (+5.3%)
  Profit Factor: 1.38 → 1.74 (+26.1%)

✅ Parameters saved
```

### 4.4 Validate Pattern Performance

**Backtest Optimized Patterns**:

Navigate to: Backtesting Tab → Pattern Backtest

**Parameters**:
- Period: Last 6 months (out-of-sample)
- Symbols: EUR/USD, GBP/USD
- Use optimized parameters: ✅ Enabled
- Position size: 1% risk per trade

**Expected Results**:
```
=== PATTERN BACKTEST RESULTS ===

Chart Patterns:
  Total Signals: 43
  Win Rate: 60.5% (26 wins, 17 losses)
  Profit Factor: 1.96
  Avg R:R: 2.2:1

Candlestick Patterns:
  Total Signals: 87
  Win Rate: 61.2% (53 wins, 34 losses)
  Profit Factor: 1.68
  Avg R:R: 1.9:1

Combined Performance:
  Total Return: +8.9%
  Sharpe Ratio: 1.52
  Max Drawdown: -3.4%

✅ Patterns performing above baseline
```

---

## Test 5: Trading Engine Configuration

**Objective**: Configure and optimize trading engine parameters for live use.

### 5.1 Navigate to Trading Engine Config

**Location**: Settings → Trading Engine (or dedicated Trading Engine tab)

### 5.2 Core Configuration

**Parameters**:

```yaml
# === ACCOUNT SETTINGS ===
Initial Capital: $10,000
  Location: Account section
  Test Value: 10000

Account Currency: USD
  Location: Account section
  Test Value: USD

Max Account Risk: 15%
  Location: Risk limits
  Test Value: 0.15
  Notes: Total exposure limit across all positions

# === POSITION SIZING ===
Position Sizing Method: Kelly Criterion (Modified)
  Location: Position sizing dropdown
  Options: Fixed Fractional, Kelly, Optimal F, Volatility Adjusted
  Test Value: Kelly

Kelly Fraction: 0.25
  Location: Kelly config
  Test Value: 0.25
  Notes: Quarter Kelly for safety

Risk Per Trade: 1.0%
  Location: Position sizing
  Test Value: 0.01
  Notes: Base risk before adjustments

Max Position Size: 5%
  Location: Position sizing
  Test Value: 0.05
  Notes: Hard cap per position

# === SIGNAL FILTERING ===
Signal Quality Threshold: 0.70
  Location: Signal filtering
  Test Value: 0.70
  Notes: Minimum quality score to trade

MTF Confirmation: Enabled
  Location: Signal filtering
  Test Value: True
  Notes: Require multiple timeframe agreement

MTF Agreement: 70%
  Location: MTF config
  Test Value: 0.70
  Notes: Percentage of TFs that must agree

# === RISK MANAGEMENT ===
Stop Loss Method: ATR-based
  Location: Risk management dropdown
  Test Value: ATR

Stop Loss Multiplier: 2.0× ATR
  Location: Stop loss config
  Test Value: 2.0
  Notes: Dynamic stop based on volatility

Take Profit Multiplier: 3.0× ATR
  Location: Take profit config
  Test Value: 3.0
  Notes: Target 1.5:1 reward/risk

Trailing Stop: Enabled
  Location: Risk management
  Test Value: True

Trailing Stop Activation: 1.5× initial risk
  Location: Trailing config
  Test Value: 1.5
  Notes: Activate when profit >1.5R

Trailing Stop Step: 0.3× ATR
  Location: Trailing config
  Test Value: 0.3
  Notes: Trail by 0.3 ATR per 1R profit

# === REGIME AWARENESS ===
Use Regime Detection: Enabled
  Location: Advanced settings
  Test Value: True

Regime Detector: HMM (Hidden Markov Model)
  Location: Regime dropdown
  Test Value: HMM

Regime Multipliers:
  Location: Regime config
  Values:
    Trending: 1.2× size
    Ranging: 0.8× size
    High Volatility: 0.5× size

# === SENTIMENT & VIX ===
Use Sentiment Data: Enabled
  Location: Data sources
  Test Value: True

Sentiment Adjustment Range: 0.8× - 1.2×
  Location: Sentiment config
  Test Values: [0.8, 1.2]

Use VIX Filter: Enabled
  Location: Data sources
  Test Value: True

VIX Adjustment Levels:
  Location: VIX config
  Values:
    VIX < 12 (Complacency): 0.95× size
    VIX 12-20 (Normal): 1.0× size
    VIX 20-30 (Concern): 0.85× size
    VIX > 30 (Fear): 0.7× size

# === EXECUTION ===
Execution Method: Smart Execution
  Location: Execution dropdown
  Test Value: Smart

Max Slippage: 2 pips
  Location: Execution config
  Test Value: 0.0002
  Notes: Reject fills with >2 pips slippage

Order Timeout: 30 seconds
  Location: Execution config
  Test Value: 30
  Notes: Cancel if not filled in 30s

# === TIME FILTERS ===
Trading Hours: London + NY sessions
  Location: Time filters
  Test Value: [08:00-21:00 GMT]

Avoid News Events: Enabled
  Location: Calendar integration
  Test Value: True

News Buffer: 30 minutes before/after
  Location: Calendar config
  Test Value: 30
  Notes: No trading 30min around high-impact news

Weekend Close: Enabled
  Location: Time filters
  Test Value: True

Weekend Close Time: Friday 21:00 GMT
  Location: Weekend config
  Test Value: 21:00

# === CORRELATION MANAGEMENT ===
Max Correlated Positions: 3
  Location: Portfolio management
  Test Value: 3

Correlation Threshold: 0.7
  Location: Correlation config
  Test Value: 0.7
  Notes: Pairs with r>0.7 considered correlated

Correlation Check Pairs:
  EUR/USD ↔ GBP/USD: r=0.82 (limit exposure)
  USD/JPY ↔ EUR/USD: r=-0.41 (diversification OK)
  AUD/USD ↔ EUR/USD: r=0.73 (limit exposure)

# === DRAWDOWN PROTECTION ===
Daily Loss Limit: -3%
  Location: Circuit breaker
  Test Value: -0.03
  Action: Stop trading for the day

Weekly Loss Limit: -8%
  Location: Circuit breaker
  Test Value: -0.08
  Action: Reduce size 50% until recovery

Max Drawdown Circuit Breaker: -25%
  Location: Circuit breaker
  Test Value: -0.25
  Action: Stop all trading, manual review required

# === RETRAINING ===
Auto Retraining: Enabled
  Location: Model management
  Test Value: True

Retraining Trigger: Performance degradation >5%
  Location: Retraining config
  Test Value: 0.05

Retraining Frequency: Maximum 30 days
  Location: Retraining config
  Test Value: 30
  Notes: Retrain at least monthly

Model Validation Required: Enabled
  Location: Retraining config
  Test Value: True
  Notes: New model must beat old model to deploy
```

### 5.3 Save Configuration

**Steps**:
1. Review all parameters entered
2. Click "Save Configuration" button
3. Confirm save dialog
4. Verify settings saved

**Verification**:
```bash
python -c "from src.forex_diffusion.utils.user_settings import get_setting; print('Risk per trade:', get_setting('trading_engine.risk_per_trade')); print('Kelly fraction:', get_setting('trading_engine.kelly_fraction')); print('Stop loss ATR:', get_setting('trading_engine.stop_loss_atr'))"
```

**Expected Output**:
```
Risk per trade: 0.01
Kelly fraction: 0.25
Stop loss ATR: 2.0
```

### 5.4 Test Configuration with Paper Trading

**Steps**:
1. Navigate to: Trading Engine → Paper Trading Mode
2. Enable paper trading: ✅
3. Click "Start Engine" button
4. Let run for 1-2 hours

**Monitor**:
- Real-time signal generation
- Position entries/exits
- P&L updates
- Risk metrics

**Expected Behavior**:
- Signals generated when confidence >0.70
- Positions sized correctly (1% risk, Kelly adjusted)
- Stops and targets placed at 2× and 3× ATR
- No more than 3 correlated positions open
- Regime adjustments applied (size varies by regime)

**Sample Output**:
```
[12:34:56] Signal: EUR/USD LONG | Confidence: 0.78 | Quality: GOOD
[12:35:02] Position opened: EUR/USD LONG | Entry: 1.0850 | Size: 0.23 lots | Stop: 1.0820 | Target: 1.0910
[12:35:02] Regime: Trending Up | Size multiplier: 1.2×
[12:35:02] VIX: 18.5 (Normal) | VIX multiplier: 1.0×
[12:35:02] Sentiment: Neutral | Sentiment multiplier: 1.0×
[12:35:02] Final size: 0.23 lots (base 0.19 × 1.2)

[13:45:23] Position closed: EUR/USD LONG | Exit: 1.0905 | P&L: +$127.50 (+1.27%) | R: +1.83
[13:45:23] Reason: Take profit target reached
[13:45:23] Duration: 1h 10m

[14:20:15] Signal: GBP/USD LONG | Confidence: 0.72
[14:20:15] ⚠️  Correlation check: EUR/USD (r=0.82) already open
[14:20:15] Max correlated positions: 3 | Current: 2
[14:20:15] ✅ Signal accepted (under limit)
[14:20:20] Position opened: GBP/USD LONG | Entry: 1.2650 | Size: 0.18 lots
```

---

## Test 6: Integrated System Test

**Objective**: Run complete end-to-end system with all components integrated.

### 6.1 Start Complete System

**Command Line**:
```bash
# Option 1: GUI mode
python run_forexgpt.py

# Option 2: Headless mode (for server deployment)
python run_forexgpt.py --headless --config configs/production.yaml
```

### 6.2 Verify All Services Started

**Check System Status Tab**:

Navigate to: Help → System Status (or dedicated Status tab)

**Expected Services Running**:
```
✅ Database Service: Connected
✅ Data Provider (cTrader): Connected
✅ DOM Aggregator Service: Running (last update: 2s ago)
✅ Sentiment Aggregator Service: Running (last update: 15s ago)
✅ VIX Service: Running (last VIX: 18.5, updated 3m ago)
✅ Pattern Detection Service: Running
✅ Forecast Service: Running (54 models loaded)
✅ Regime Detector: Running (current: Trending Up, conf: 0.87)
✅ Trading Engine: Running (paper mode)
✅ Risk Manager: Active

System Health: HEALTHY
Uptime: 0h 05m 23s
Memory: 2.3GB / 16GB (14%)
CPU: 15% avg
Latency: 1.2s (data → signal → execution)
```

### 6.3 Monitor Data Flow

**Check Data Flow Tab**:

**Real-Time Ticks**:
```
[15:23:45.123] EUR/USD | 1.0850 / 1.0851 | Spread: 1.0 pips
[15:23:46.234] GBP/USD | 1.2650 / 1.2652 | Spread: 2.0 pips
[15:23:47.345] USD/JPY | 145.23 / 145.25 | Spread: 2.0 pips
```

**Aggregated Bars**:
```
[15:24:00] EUR/USD 1m | O: 1.0850 H: 1.0852 L: 1.0849 C: 1.0851 V: 1250
[15:25:00] EUR/USD 1m | O: 1.0851 H: 1.0853 L: 1.0850 C: 1.0852 V: 980
```

**DOM Updates**:
```
[15:24:30] EUR/USD DOM updated | Bid depth: 125 lots | Ask depth: 98 lots | Imbalance: +27.5%
```

**Sentiment Updates**:
```
[15:24:00] EUR/USD Sentiment | Long: 62% | Short: 38% | Contrarian Signal: -0.24 (fade long)
```

**VIX Updates**:
```
[15:20:00] VIX updated: 18.5 (Normal) | No adjustment needed
```

### 6.4 Test Signal Generation Pipeline

**Monitor Signals Tab**:

**Wait for Signal** (may take 15-60 minutes):

**Expected Signal**:
```
=== SIGNAL GENERATED ===
Timestamp: 2025-01-08 15:45:23
Symbol: EUR/USD
Direction: LONG
Entry Price: 1.0852

Forecast Confidence: 0.78
  1h prediction: +0.42% (confidence: 0.75)
  4h prediction: +0.58% (confidence: 0.79)
  8h prediction: +0.71% (confidence: 0.80)
  MTF Agreement: 85% (3/3 timeframes agree)

Pattern Confirmation: YES
  Pattern: Ascending Triangle (4h)
  Pattern Confidence: 0.82
  Breakout confirmed: Volume 1.9× average

Regime Context: Trending Up
  Regime Confidence: 0.87
  Recommended Strategy: Trend Following
  Position Size Multiplier: 1.2×

Quality Score: 0.81 (EXCELLENT)
  Pattern Strength: 0.82
  MTF Agreement: 0.85
  Regime Confidence: 0.87
  Volume Confirmation: 0.75
  Sentiment Alignment: 0.65
  Correlation Safety: 0.90

Risk Management:
  Stop Loss: 1.0822 (-30 pips, 2.0× ATR)
  Take Profit: 1.0912 (+60 pips, 3.0× ATR)
  Risk/Reward: 1:2.0

Position Sizing:
  Base Size (1% risk): 0.19 lots
  Kelly Adjustment (0.25): 0.21 lots
  Regime Multiplier (1.2×): 0.25 lots
  VIX Adjustment (1.0×): 0.25 lots
  Sentiment Adjustment (0.95×): 0.24 lots
  Final Size: 0.24 lots

DECISION: ✅ EXECUTE TRADE
```

### 6.5 Verify Trade Execution

**Check Positions Tab**:

**Expected Open Position**:
```
Position #1 - EUR/USD LONG
  Entry Time: 15:45:25
  Entry Price: 1.0852
  Current Price: 1.0855 (+3 pips)
  Size: 0.24 lots
  P&L: +$7.20 (+0.07%)
  Stop Loss: 1.0822 (-30 pips)
  Take Profit: 1.0912 (+60 pips)
  Risk: $72.00 (-0.72%)
  Potential Reward: $144.00 (+1.44%)
  R Multiple: +0.1R
  Duration: 2m 15s
```

### 6.6 Test Risk Management

**Scenario A: Stop Loss Hit**

**Simulate** (in paper trading):
1. Wait for price to move against position
2. Or manually trigger stop loss test

**Expected Behavior**:
```
[15:58:42] EUR/USD price: 1.0821 | Stop triggered at 1.0822
[15:58:43] Position closed: EUR/USD LONG | Exit: 1.0822 | P&L: -$72.00 (-0.72%)
[15:58:43] Reason: Stop loss hit
[15:58:43] R Multiple: -1.0R (as expected)
[15:58:43] ✅ Risk management working correctly
```

**Scenario B: Take Profit Hit**

**Simulate**:
1. Wait for price to reach target
2. Or manually trigger TP test

**Expected Behavior**:
```
[16:15:22] EUR/USD price: 1.0912 | Take profit triggered
[16:15:23] Position closed: EUR/USD LONG | Exit: 1.0912 | P&L: +$144.00 (+1.44%)
[16:15:23] Reason: Take profit target
[16:15:23] R Multiple: +2.0R (as expected)
[16:15:23] ✅ Trade executed perfectly
```

**Scenario C: Circuit Breaker**

**Simulate Daily Loss Limit**:

**Steps**:
1. Manually adjust account balance in paper trading
2. Set balance to trigger -3% loss
3. Verify engine stops trading

**Expected Behavior**:
```
[16:30:45] ⚠️  CIRCUIT BREAKER LEVEL 1 TRIGGERED
[16:30:45] Daily Loss: -3.1% ($310 of $10,000)
[16:30:45] Action: STOP TRADING FOR TODAY
[16:30:45] Reason: Daily loss limit exceeded (-3%)
[16:30:45] All pending orders cancelled
[16:30:45] New signals will be blocked until next trading day
[16:30:45] ✅ Risk protection activated
```

---

## Test 7: Real-Time Data Flow

**Objective**: Verify complete data pipeline from cTrader to UI.

### 7.1 Enable Real-Time Connection

**Prerequisites**:
- cTrader account credentials configured
- API access enabled
- Network connection stable

**Steps**:
1. Navigate to: Settings → Data Providers
2. Select cTrader provider
3. Enter credentials (if not saved)
4. Click "Connect" button

**Expected Output**:
```
[Data Provider] Connecting to cTrader...
[Data Provider] Authentication successful
[Data Provider] Subscribing to symbols: EUR/USD, GBP/USD, USD/JPY
[Data Provider] ✅ Connected and streaming
```

### 7.2 Verify Tick Flow

**Check Chart**:
1. Open chart for EUR/USD 1m
2. Enable "Follow" mode (right-most candle always visible)
3. Watch for real-time tick updates

**Expected Behavior**:
- Bid/Ask prices update every 100-500ms
- Current candle updates in real-time (wick extends, close moves)
- New candle appears exactly at minute boundary
- Volume increments with each tick
- No gaps or freezes

**Console Verification**:
```bash
python -c "from src.forex_diffusion.services.database_service import DatabaseService; from sqlalchemy import text; import time; db = DatabaseService(); t1 = time.time(); time.sleep(5); with db.engine.connect() as conn: result = conn.execute(text('SELECT COUNT(*) FROM bars_1m WHERE symbol = \"EURUSD\" AND ts_utc > :t'), {'t': int(t1*1000)}); print('New bars in 5s:', result.fetchone()[0])"
```

**Expected Output**:
```
New bars in 5s: 0-1 (depends on timing)
```

### 7.3 Verify DOM Updates

**Check Order Books Widget** (left panel):

**Expected Display**:
```
Order Books - EUR/USD

  Ask  | Size
─────────────
1.0854 | 45.2
1.0853 | 67.8
1.0852 | 123.4  ← Best Ask
─────────────
1.0851 | 98.7   ← Best Bid
1.0850 | 54.3
1.0849 | 32.1
  Bid  | Size

Spread: 1.0 pips
Imbalance: +25.3% BID
```

**Updates should occur every 1-3 seconds**.

### 7.4 Verify Sentiment Updates

**Check Sentiment Panel** (right side):

**Expected Display**:
```
Market Sentiment - EUR/USD

Long Positions: 62% ████████████▌
Short Positions: 38% ███████▋

Contrarian Signal: -0.24
  Interpretation: Fade Long (crowd too bullish)
  Confidence: 0.68

Sentiment Change (1h): +4% ↑
Total Traders: 12,847
```

**Updates every 30 seconds**.

### 7.5 Verify VIX Updates

**Check VIX Widget** (left panel):

**Expected Display**:
```
Volatility
━━━━━━━━━━━━━━━━━━
VIX: 18.5
[████████░░░░░░░░░░] 37%
Normal
```

**Color**: Green (Normal)  
**Updates**: Every 5 minutes

---

## Results Validation

### Acceptance Criteria

**Test 1 - Manual Training**: ✅ PASS if:
- Training completes without errors
- Model achieves directional accuracy >52%
- Model file saved and database entry created

**Test 2 - Backtesting**: ✅ PASS if:
- Backtest completes for full date range
- Profit Factor >1.3
- Sharpe Ratio >1.0
- Max Drawdown <15%

**Test 3 - Live Models Training**: ✅ PASS if:
- All 54 model combinations trained successfully
- Meta-learner achieves accuracy >60%
- Walk-forward degradation <25% MAE, <5% accuracy
- Optimization improves Sharpe by >10%

**Test 4 - Pattern Training**: ✅ PASS if:
- Genetic algorithm completes all generations
- Profit factor improves >20%
- False positive rate reduces >30%
- Win rate improves >5%

**Test 5 - Trading Engine Config**: ✅ PASS if:
- All parameters saved correctly
- Paper trading executes trades
- Risk management works (stops, limits, circuit breakers)

**Test 6 - Integrated System**: ✅ PASS if:
- All services start and remain healthy
- Signal generation pipeline works end-to-end
- Trades execute with correct sizing
- Risk management activates when needed

**Test 7 - Real-Time Data**: ✅ PASS if:
- Tick data flows continuously
- DOM updates in real-time
- Sentiment and VIX update on schedule
- No data gaps or freezes for >30 seconds

### Overall System Health

**SYSTEM HEALTHY** if:
- ✅ 6/7 tests pass (min 85%)
- ✅ No critical errors in logs
- ✅ Memory usage stable (<80%)
- ✅ CPU usage reasonable (<60% sustained)
- ✅ Latency <5 seconds (data → decision → execution)

---

## Troubleshooting Guide

### Issue 1: Training Fails - "No data available"

**Symptoms**:
```
[ERROR] No data found for EUR/USD @ 1h
[ERROR] Training aborted
```

**Solutions**:
1. Check data exists:
   ```bash
   python -c "from src.forex_diffusion.data.sqlite_data_adapter import SQLiteDataAdapter; adapter = SQLiteDataAdapter('forexgpt.db'); df = adapter.get_bars('EUR/USD', '1h', limit=10); print(len(df))"
   ```
2. Download data if missing:
   ```bash
   python scripts/download_historical_data.py --symbol EUR/USD --timeframe 1h --days 730
   ```
3. Check symbol format (EUR/USD vs EURUSD):
   ```bash
   python -c "from src.forex_diffusion.data.sqlite_data_adapter import SQLiteDataAdapter; adapter = SQLiteDataAdapter('forexgpt.db'); from sqlalchemy import text; with adapter.engine.connect() as conn: result = conn.execute(text('SELECT DISTINCT symbol FROM bars_1h')); print([row[0] for row in result])"
   ```

### Issue 2: Backtest Fails - "Model not found"

**Symptoms**:
```
[ERROR] Model file not found: artifacts/ridge_eurusd_1h_4h_20250108.pkl
```

**Solutions**:
1. List available models:
   ```bash
   ls artifacts/*.pkl
   ```
2. Check database for model path:
   ```bash
   python -c "from src.forex_diffusion.services.database_service import DatabaseService; from sqlalchemy import text; db = DatabaseService(); with db.engine.connect() as conn: result = conn.execute(text('SELECT file_path FROM model_metadata')); [print(row[0]) for row in result]"
   ```
3. Re-train model if missing (see Test 1)

### Issue 3: Real-Time Connection Fails

**Symptoms**:
```
[ERROR] cTrader connection timeout
[ERROR] Authentication failed
```

**Solutions**:
1. Check credentials:
   - Settings → Data Providers → cTrader
   - Verify Client ID, Secret, Access Token
2. Test connection manually:
   ```bash
   python -c "from src.forex_diffusion.providers.ctrader_provider import CTraderProvider; provider = CTraderProvider(); provider.connect(); print('Connected:', provider.is_connected())"
   ```
3. Check firewall/proxy settings
4. Verify cTrader API status (check website)

### Issue 4: High Memory Usage

**Symptoms**:
- Memory >80% (>12GB on 16GB system)
- Application slowdown
- "Out of memory" errors

**Solutions**:
1. Reduce lookback window:
   - Training tab → Lookback bars: 2000 → 1000
2. Disable unused features:
   - Uncheck heavy indicators (Volume Profile, All candlestick patterns)
3. Clear cache:
   ```bash
   python -c "from src.forex_diffusion.cache.simple_cache import SimpleCache; cache = SimpleCache(); cache.clear(); print('Cache cleared')"
   ```
4. Restart application

### Issue 5: Pattern Training Extremely Slow

**Symptoms**:
- Generation 1 takes >30 minutes
- CPU at 100% for extended period

**Solutions**:
1. Reduce population size: 40 → 20
2. Reduce generations: 25 → 15
3. Reduce data period: 5 years → 2 years
4. Reduce assets: 3 symbols → 1 symbol
5. Use fewer timeframes: 2 TFs → 1 TF

### Issue 6: Trading Engine Not Generating Signals

**Symptoms**:
- Engine running but no signals for >2 hours
- Signal quality always <0.70

**Solutions**:
1. Lower quality threshold:
   - Settings → Trading Engine → Signal threshold: 0.70 → 0.60
2. Disable MTF confirmation temporarily:
   - Settings → Trading Engine → MTF confirmation: ☐
3. Check if models loaded:
   ```bash
   python -c "from src.forex_diffusion.inference.service import InferenceService; service = InferenceService(); print('Models loaded:', len(service._models))"
   ```
4. Retrain models if old (>60 days)

### Issue 7: Circuit Breaker Triggered Incorrectly

**Symptoms**:
```
[WARNING] Circuit breaker activated at -2.5% loss
```
(But threshold is -3%)

**Solutions**:
1. Check calculation includes commission/slippage
2. Review closed trades for unexpected losses
3. Verify circuit breaker settings:
   ```bash
   python -c "from src.forex_diffusion.utils.user_settings import get_setting; print('Daily limit:', get_setting('trading_engine.daily_loss_limit'))"
   ```
4. May be correct if multiple rapid losses occurred

---

## Test Execution Log

**Date**: _________________  
**Tester**: _________________  
**Environment**: Windows ___ / Python ___ / RAM ___ GB

| Test | Status | Duration | Notes |
|------|--------|----------|-------|
| 1. Manual Training | ☐ PASS ☐ FAIL | _____ min | |
| 2. Backtesting | ☐ PASS ☐ FAIL | _____ min | |
| 3. Live Models Training | ☐ PASS ☐ FAIL | _____ hours | |
| 4. Pattern Training | ☐ PASS ☐ FAIL | _____ hours | |
| 5. Trading Engine Config | ☐ PASS ☐ FAIL | _____ min | |
| 6. Integrated System | ☐ PASS ☐ FAIL | _____ min | |
| 7. Real-Time Data Flow | ☐ PASS ☐ FAIL | _____ min | |

**Overall Result**: ☐ PASS ☐ FAIL  
**Pass Rate**: _____ / 7 (_____ %)

**Critical Issues**:
1. ___________________________________________
2. ___________________________________________
3. ___________________________________________

**Recommendations**:
1. ___________________________________________
2. ___________________________________________
3. ___________________________________________

---

## Appendix A: Quick Reference

### Key File Locations

```
D:\Projects\ForexGPT\
├── forexgpt.db              # Main database
├── artifacts/               # Trained models
├── configs/                 # Configuration files
│   └── default.yaml         # Default config
├── analysis/                # Test results
│   └── backtest_*.json      # Backtest exports
└── run_forexgpt.py          # Main application
```

### Key Tables in Database

```sql
-- Model metadata
SELECT * FROM model_metadata ORDER BY created_at DESC LIMIT 5;

-- Optimization results
SELECT * FROM optimization_results WHERE objective_1 > 1.5;

-- Pattern configurations
SELECT * FROM pattern_configs WHERE is_optimized = 1;

-- Recent bars
SELECT * FROM bars_1h WHERE symbol = 'EURUSD' ORDER BY ts_utc DESC LIMIT 10;
```

### Performance Benchmarks

| Operation | Expected Duration | Max Acceptable |
|-----------|-------------------|----------------|
| Single model training | 3-10 seconds | 30 seconds |
| Ensemble training (54 models) | 2-3 hours | 4 hours |
| Pattern training (25 gen) | 2-3 hours | 5 hours |
| Backtest (6 months 1H) | 2-5 minutes | 10 minutes |
| Signal generation | <2 seconds | 5 seconds |
| Tick processing | <100ms | 500ms |

---

**Document End**

*ForexGPT Manual Test Procedure v1.0*  
*For questions or issues: Check troubleshooting guide or review AGENTS.md*
