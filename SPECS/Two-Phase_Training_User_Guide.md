# Two-Phase Training System - User Guide

**Version**: 1.0
**Date**: 2025-10-07
**Status**: Production-Ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Creating Training Queues](#creating-training-queues)
5. [Monitoring Training Progress](#monitoring-training-progress)
6. [Regime Analysis](#regime-analysis)
7. [Training History](#training-history)
8. [Crash Recovery](#crash-recovery)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

The Two-Phase Training System is an intelligent model training pipeline that:

- **Automates** multi-configuration model training
- **Optimizes** inference strategies through internal backtesting
- **Selects** best models per market regime automatically
- **Manages** storage efficiently (90-95% reduction)
- **Recovers** from crashes automatically

### How It Works

**External Loop** (Phase 1):
- Trains models for each configuration in your grid
- Tests each model with multiple inference strategies
- Evaluates performance across different market regimes

**Internal Loop** (Phase 2):
- Backtests each trained model with various inference methods
- Tests different prediction approaches (direct, recursive, multi-step)
- Applies ensemble techniques and confidence filtering
- Identifies best inference configuration per regime

**Decision Logic**:
- Keeps only models that improve performance in ≥1 regime
- Automatically deletes non-improving models
- Updates best model tracking per regime
- Saves 90-95% disk space

---

## Quick Start

### 1. Access the Training Manager

From the main ForexGPT GUI:
1. Navigate to the **Training** tab
2. Click **"Grid Training Manager"** button
3. The Grid Training Manager dialog opens with 3 tabs:
   - **Training Queue**: Create and run training grids
   - **Regime Analysis**: View best models per regime
   - **Training History**: Browse historical runs

### 2. Create Your First Training Queue

1. In the **Training Queue** tab:
   - Select model types (e.g., RandomForest, XGBoost)
   - Choose symbols (e.g., EURUSD, GBPUSD)
   - Select timeframes (e.g., 1H, 4H)
   - Configure parameters (days history, horizons)

2. Click **"Generate Queue"**
   - System calculates total configurations
   - Checks for already-trained models
   - Creates queue in database

3. Click **"Start Training"**
   - Training begins in background
   - Progress updates in real-time
   - Can pause/cancel anytime

### 3. Monitor Progress

- **Progress Bar**: Shows completion percentage
- **Status Log**: Real-time training updates
- **Results Table**: Models trained, kept, deleted
- **Current Index**: Which configuration is training

### 4. View Results

**Regime Analysis Tab**:
- See best models for each market regime
- View performance metrics (Sharpe, Drawdown, Win Rate)
- Click **"View Performance Charts"** for visualizations
- Use **"Manage Regimes"** to customize regime definitions

**Training History Tab**:
- Search and filter all training runs
- Export results to CSV
- View detailed metrics for each run

---

## Configuration

Configuration is managed via **YAML file**:
```
configs/training_pipeline/default_config.yaml
```

### Key Configuration Sections

#### Storage Settings
```yaml
storage:
  artifacts_dir: "./artifacts"           # Where models are stored
  checkpoints_dir: "./checkpoints/training_pipeline"  # Checkpoints
  compress_old_models: true              # Compress old models
```

#### Queue Settings
```yaml
queue:
  max_parallel_queues: 1                 # Max simultaneous queues
  auto_checkpoint_interval: 10           # Checkpoint every N models
  max_inference_workers: 4               # Parallel inference backtests
```

#### Recovery Settings
```yaml
recovery:
  auto_resume_on_crash: false            # Auto-resume crashed queues
  detect_crash_after_minutes: 5          # Crash detection threshold
```

#### Model Management
```yaml
model_management:
  delete_non_best_models: true           # Auto-delete non-improving models
  keep_top_n_per_regime: 1               # Keep top N models per regime
  min_improvement_threshold: 0.01        # 1% minimum improvement to replace
```

#### Inference Grid
```yaml
inference_grid:
  prediction_methods:                    # Prediction approaches
    - "direct"                          # Single-step
    - "recursive"                       # Multi-step recursive
    - "direct_multi"                    # Direct multi-step

  ensemble_methods:                      # Ensemble techniques
    - "mean"                            # Simple average
    - "weighted"                        # Performance-weighted
    - "stacking"                        # Meta-model stacking

  confidence_thresholds:                 # Filter predictions
    - 0.0                               # No filtering
    - 0.3                               # Low confidence
    - 0.5                               # Medium confidence
    - 0.7                               # High confidence
    - 0.9                               # Very high confidence

  lookback_windows:                      # Feature lookback periods
    - 50
    - 100
    - 200
```

---

## Creating Training Queues

### Configuration Grid Builder

#### 1. Select Model Types
Multi-select list with options:
- **random_forest**: Ensemble tree-based
- **gradient_boosting**: Gradient-boosted trees
- **xgboost**: Extreme gradient boosting
- **lightgbm**: Light gradient boosting
- **linear_regression**: Simple linear model
- **ridge**: L2 regularized linear
- **lasso**: L1 regularized linear
- **elasticnet**: L1+L2 regularized linear

*Tip: Start with XGBoost and LightGBM for best performance*

#### 2. Choose Symbols
Multi-select from available currency pairs:
- EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, etc.

*Tip: Train on correlated pairs together (e.g., EURUSD + GBPUSD)*

#### 3. Select Timeframes
Checkboxes for:
- 1H: Hourly data
- 4H: 4-hour data
- 1D: Daily data

*Tip: Higher timeframes require less compute but may miss intraday patterns*

#### 4. Configure Parameters

**Days History** (checkboxes):
- 30: Short-term patterns
- 60: Medium-term
- 90: Quarterly patterns
- 180: Half-year patterns

**Horizons** (checkboxes):
- 1: Next bar prediction
- 3: 3-bar ahead
- 6: 6-bar ahead
- 12: 12-bar ahead

**Encoder** (dropdown):
- time_delta: Time-based encoding
- absolute: Absolute price encoding
- normalized: Normalized features

#### 5. Generate Queue

Click **"Generate Queue"**:
- System calculates Cartesian product (all combinations)
- Shows total configurations
- Deduplicates identical configurations (SHA256 hashing)
- Filters already-trained models (optional)
- Displays filtered count

Example:
```
2 models × 2 symbols × 1 timeframe × 2 days × 2 horizons = 16 configurations
- 3 already trained
= 13 new configurations to train
```

#### 6. Start Training

Click **"Start Training"**:
- Queue created in database
- Training worker thread starts
- Background execution (GUI remains responsive)
- Auto-checkpoints every N models (configurable)

---

## Monitoring Training Progress

### Real-Time Updates

**Progress Bar**:
- Visual completion percentage
- Updates after each model trained

**Status Log**:
- Real-time messages:
  - "Starting training queue..."
  - "Training model 5/13..."
  - "Model kept (improves: bull_trending, calm_ranging)"
  - "Model deleted (no regime improvements)"
  - "Checkpoint saved"
  - "Training completed!"

**Results Summary**:
- **Total Configs**: Number of models to train
- **Completed**: Models finished training
- **Kept Models**: Models that improved ≥1 regime
- **Deleted Models**: Non-improving models
- **Current Index**: Which model is training now

### Control Buttons

**Pause**:
- Pauses after current model finishes
- State saved to database
- Can resume later

**Cancel**:
- Graceful cancellation
- Completes current model
- Saves progress
- Confirmation dialog

**Resume**:
- Continues from last checkpoint
- Available after pause/crash
- Loads queue state from database

---

## Regime Analysis

### Best Models Per Regime

**Default Regimes**:
1. **Bull Trending**: Strong upward trend (trend_strength > 0.7, returns > 0)
2. **Bear Trending**: Strong downward trend (trend_strength > 0.7, returns < 0)
3. **Volatile Ranging**: High volatility, no trend (volatility > 75th percentile)
4. **Calm Ranging**: Low volatility consolidation (volatility < 50th percentile)

### Viewing Best Models

**Regime Table** shows:
- **Regime Name**: Market condition category
- **Has Best Model**: ✓ if model assigned, ✗ if empty
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Achieved At**: When model became best

### Performance Charts

Click **"View Performance Charts"** to see:

**Chart 1: Sharpe Ratio Comparison**
- Bar chart comparing all regimes
- Green bars: regimes with models
- Gray bars: empty regimes
- Baseline at 0

**Chart 2: Max Drawdown Comparison**
- Risk metrics across regimes
- Lower is better

**Chart 3: Win Rate Comparison**
- Winning trade percentages
- Red dashed line at 50% baseline

**Chart 4: Performance Profile (Radar)**
- Multi-dimensional view of best model
- Shows Sharpe, Drawdown Control, Win Rate
- Normalized 0-1 scale

### Managing Regimes

Click **"Manage Regimes"** to:
- **Add** new regime definitions
- **Edit** existing regimes
- **Delete** unused regimes
- **Toggle** active/inactive status

**Adding a Regime**:
1. Click "Add New Regime"
2. Enter regime name (e.g., "breakout_volatile")
3. Add description
4. Define detection rules (JSON format):
   ```json
   {
     "trend_strength": "> 0.5",
     "volatility": "> 90th percentile",
     "volume": "> 2x average"
   }
   ```
5. Check "Active" to enable
6. Click "Save"

---

## Training History

### Searching and Filtering

**Filter Options**:
- **Symbol**: Filter by currency pair (All, EURUSD, GBPUSD, etc.)
- **Model Type**: Filter by algorithm (All, XGBoost, LightGBM, etc.)
- **Status**: Filter by outcome (All, completed, failed, cancelled)
- **Limit**: Results per query (10-1000)

**Search Button**:
- Applies current filters
- Loads matching runs
- Updates table in background

**Clear Filters**:
- Resets all filters to "All"
- Sets limit to 100

### Results Table

**Columns**:
- **ID**: Training run database ID
- **UUID**: Unique identifier (first 8 chars)
- **Model Type**: Algorithm used
- **Symbol**: Currency pair
- **Timeframe**: Data interval
- **Status**: completed (green), failed (red), running (blue)
- **Kept**: ✓ if model kept, ✗ if deleted
- **Best Regimes**: Regimes where this model is best
- **Created At**: When training started

**Interactions**:
- **Click row**: Selects for details
- **Double-click**: Opens detail dialog
- **"View Details" button**: Shows full run information

### Exporting Results

Click **"Export Results"**:
- Exports current filtered results to CSV
- Saves to `~/.forexgpt/training_history_YYYYMMDD_HHMMSS.csv`
- All columns included
- Success notification with file path

---

## Crash Recovery

### Automatic Detection

On application startup, the system:
1. Scans for queues with status='running'
2. Checks last activity timestamp
3. If no activity for >5 minutes, marks as crashed
4. Shows recovery dialog (if `auto_resume_on_crash: false`)

### Recovery Dialog

**Information Shown**:
- Queue Name
- Progress percentage
- Remaining configurations
- Recommendation based on progress:
  - <10%: "Consider restarting from scratch"
  - 10-50%: "Resume to save partial progress"
  - >50%: "Resume to complete remaining work"

**Actions**:
- **Resume**: Continues from last checkpoint
- **Close**: Ignore for now (can resume later)

### Auto-Resume

Enable in config:
```yaml
recovery:
  auto_resume_on_crash: true
```

Behavior:
- Automatically marks crashed queues as 'paused'
- Shows info notification on startup
- No user interaction required
- Can still resume manually from Training Queue tab

---

## Advanced Features

### Checkpoint System

**Auto-Checkpointing**:
- Saves queue state every N models (default: 10)
- Stores to JSON file with version validation
- Includes:
  - Queue configuration
  - Current index
  - Trained configurations
  - Results so far

**Manual Checkpointing**:
- Pause button triggers checkpoint
- Cancel button saves before exit
- Safe interruption at any time

**Checkpoint Files**:
- Location: `checkpoints/training_pipeline/`
- Format: `queue_{id}_checkpoint_YYYYMMDD_HHMMSS.json`
- Auto-cleanup after 30 days (configurable)

### Configuration Deduplication

**SHA256 Hashing**:
- Each configuration gets unique hash
- Identical configs detected and skipped
- Hash based on:
  - Model type, encoder, symbol, timeframe
  - Days history, horizon, indicator timeframes
  - Hyperparameters (if specified)

**Already-Trained Filtering**:
- Checkbox: "Skip already-trained configurations"
- Queries database for existing config hashes
- Skips training if match found
- Saves time on repeated runs

### Inference Strategy Optimization

**Prediction Methods**:
- **Direct**: Single-step prediction (fastest)
- **Recursive**: Multi-step with feedback (long horizons)
- **Direct Multi**: Smoothed multi-step (stable)

**Ensemble Methods**:
- **Mean**: Simple average (baseline)
- **Weighted**: Performance-weighted (adaptive)
- **Stacking**: Variance-based weighting (advanced)

**Confidence Filtering**:
- Filters low-confidence predictions
- Combines 3 metrics:
  - Magnitude confidence (40%): Stronger signals prioritized
  - Consistency confidence (40%): Agrees with recent predictions
  - Feature confidence (20%): Lower volatility favored
- Thresholds: 0.0 (none), 0.3 (low), 0.5 (med), 0.7 (high), 0.9 (strict)

### Storage Efficiency

**Model File Management**:
- **Keep**: Models improving ≥1 regime
- **Delete**: Non-improving models
- **Threshold**: Configurable (default: 1% improvement)

**Space Savings**:
- Typical: 90-95% reduction
- Example: 100 models trained → 5-10 models kept
- Automatic cleanup after each training

**Manual Cleanup**:
```python
from forex_diffusion.training.training_pipeline.model_file_manager import ModelFileManager

manager = ModelFileManager()
stats = manager.cleanup_old_models(days_old=30)
print(f"Freed {stats['freed_space_mb']:.2f} MB")
```

---

## Troubleshooting

### Common Issues

**Q: Training queue not starting**
- Check if another queue is already running (max_parallel_queues=1)
- Verify database connection
- Check logs in `logs/training_pipeline.log`

**Q: "No configurations to train after filtering"**
- All configurations already trained
- Uncheck "Skip already-trained configurations"
- Or change grid parameters

**Q: Models not being kept**
- Check if models actually improve performance
- Verify regime definitions are appropriate
- Lower `min_improvement_threshold` in config

**Q: Crash recovery not detecting**
- Check `detect_crash_after_minutes` (default: 5)
- Verify queue status in database
- Check last updated_at timestamp

**Q: Charts not displaying**
- Install matplotlib: `pip install matplotlib`
- Check if regime data is loaded
- Verify at least one regime has a best model

**Q: Out of disk space**
- Check `delete_non_best_models: true` in config
- Run manual cleanup for old models
- Increase `min_improvement_threshold` (more deletions)

### Performance Tuning

**Faster Training**:
- Increase `max_inference_workers` (more parallel backtests)
- Reduce `lookback_windows` (fewer inference configs)
- Use fewer `confidence_thresholds`

**Better Results**:
- Increase `days_history` (more training data)
- Use multiple `ensemble_methods`
- Lower `min_improvement_threshold` (keep more models)

**Less Disk Usage**:
- Enable `delete_non_best_models: true`
- Set `keep_top_n_per_regime: 1`
- Increase `min_improvement_threshold`

### Logs and Debugging

**Log Locations**:
- Application: `logs/training_pipeline.log`
- Individual runs: Check `training_runs` table in database

**Enable Debug Logging**:
```yaml
logging:
  level: "DEBUG"
  log_to_file: true
  max_log_size_mb: 100
```

**Database Inspection**:
```python
from forex_diffusion.training.training_pipeline.database import session_scope, get_all_training_queues

with session_scope() as session:
    queues = get_all_training_queues(session)
    for q in queues:
        print(f"{q.id}: {q.queue_name} - {q.status} - {q.current_index}/{q.total_configs}")
```

---

## Getting Help

**Documentation**:
- Specification: `SPECS/New_AI_Training_Specs_10-07.md`
- Implementation Report: `SPECS/New_AI_Training_Implemented_10-07.md`
- Verification Report: `SPECS/Verification_Report_10-07.md`

**Support**:
- Check logs first: `logs/training_pipeline.log`
- Review database state via GUI tabs
- Consult specification for detailed algorithm explanations

---

**Last Updated**: 2025-10-07
**Version**: 1.0.0
**Status**: Production-Ready ✅
