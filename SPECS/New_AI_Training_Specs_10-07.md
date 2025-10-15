# New AI Training System Specification - October 7, 2025

## Executive Summary

This document specifies a comprehensive redesign of the AI training pipeline to implement an efficient two-phase training and backtesting architecture. The new system separates computationally expensive model training from fast inference backtesting, enabling rapid optimization without redundant training.

### Core Innovation

**Current Problem**: We backtest on trained models by varying all parameters, requiring full model retraining for each parameter combination. This is extremely slow (hours to days).

**Proposed Solution**: 
1. **External Loop (Training)**: Train all possible model configurations once (slow, done once)
2. **Internal Loop (Inference Backtest)**: Test inference combinations on pre-trained models (fast, iterative)
3. **Regime-based Selection**: Keep only models that outperform previous best for at least one market regime
4. **Efficient Storage**: Store only winning models, archive metadata for all configurations

### Expected Impact

- **Training Speed**: 10-100x faster backtesting (minutes vs hours)
- **Storage Efficiency**: 90% reduction in model storage (only keep best performers)
- **Reproducibility**: Complete audit trail of all training attempts
- **Flexibility**: Easy to test new inference strategies without retraining
- **Resumability**: Interrupt and resume training at any point

---

## 1. System Architecture

### 1.1 Two-Phase Pipeline

#### Phase 1: Model Training (External Loop)
```
FOR each model_configuration:
    1. Train model with specific parameters
    2. Execute Phase 2 (Inference Backtest) on trained model
    3. Compare results against current best_per_regime
    4. IF improvement in ANY regime:
         SAVE model file + metadata
       ELSE:
         SAVE metadata only to DB
         DELETE model file
    5. UPDATE regime performance tracking
```

#### Phase 2: Inference Backtest (Internal Loop)
```
FOR each inference_configuration:
    1. Load trained model (fast)
    2. Generate predictions with inference params
    3. Calculate performance metrics
    4. Evaluate across all regimes
    5. RETURN performance_by_regime
```

### 1.2 Parameter Classification

Parameters are classified into three categories:

#### A. Mutually Exclusive Parameters (External Loop)
These define distinct model configurations. Only ONE value can be selected:

1. **model_type**: ridge | lasso | elasticnet | rf | lightning | diffusion-ddpm | diffusion-ddim | sssd
2. **encoder**: none | pca | autoencoder | vae | latents
3. **symbol**: EUR/USD | GBP/USD | AUX/USD | GBP/NZD | AUD/JPY | GBP/EUR | GBP/AUD
4. **base_timeframe**: 1m | 5m | 15m | 30m | 1h | 4h | 1d
5. **days_history**: integer (1-3650)
6. **horizon**: integer (1-500)

#### B. Combinable Parameters (External Loop)
These can be combined within a single model:

1. **indicator_timeframes**: Dictionary mapping each indicator to list of timeframes
   - Example: {"rsi": ["5m", "15m", "1h"], "macd": ["15m", "1h", "4h"]}
   - Each indicator can have multiple timeframes simultaneously
   
2. **additional_features**: Boolean flags, can be combined
   - returns_volatility: True/False
   - trading_sessions: True/False
   - candlestick_patterns: True/False
   - volume_profile: True/False
   - vsa_analysis: True/False

3. **preprocessing_params**: All preprocessing can be combined
   - warmup_bars, rv_window, min_coverage, etc.

#### C. Inference Parameters (Internal Loop)
These affect only prediction generation, not model training:

1. **prediction_method**:
   - direct: Single-step prediction
   - recursive: Multi-step recursive
   - direct_multi: Multiple direct predictors
   
2. **ensemble_method**:
   - mean: Simple average
   - weighted: Performance-weighted
   - stacking: Meta-model stacking
   
3. **confidence_threshold**: float (0.0-1.0)
   - Minimum confidence to execute trade

4. **lookback_window**: integer
   - Number of historical bars for inference context

### 1.3 Configuration Space Example

**External Loop (Model Training)**:
- 8 model types
- 5 encoders  
- 7 symbols
- 7 timeframes
- 5 common day ranges
- 5 common horizons
= 8 × 5 × 7 × 7 × 5 × 5 = **49,000 potential model configurations**

**Internal Loop (Inference Backtest)**:
- 3 prediction methods
- 3 ensemble methods
- 5 confidence thresholds
- 3 lookback windows
= 3 × 3 × 5 × 3 = **135 inference configurations per model**

**Total Tests**: 49,000 models × 135 inferences = 6,615,000 combinations
- Current system: ~6.6M full model trainings (IMPOSSIBLE)
- New system: 49K trainings + 6.6M fast inferences (FEASIBLE)

---

## 2. Database Schema

### 2.1 New Tables (Alembic Migration)

#### Table: training_runs
Tracks every model training attempt

```sql
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uuid TEXT UNIQUE NOT NULL,  -- UUID for this training run
    status TEXT NOT NULL,  -- 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
    
    -- Model Configuration (External Loop Parameters)
    model_type TEXT NOT NULL,
    encoder TEXT NOT NULL,
    symbol TEXT NOT NULL,
    base_timeframe TEXT NOT NULL,
    days_history INTEGER NOT NULL,
    horizon INTEGER NOT NULL,
    
    -- Feature Configuration
    indicator_tfs JSON,  -- {"rsi": ["5m", "15m"], "macd": ["1h"]}
    additional_features JSON,  -- {"returns": true, "sessions": false}
    preprocessing_params JSON,  -- All preprocessing hyperparameters
    
    -- Model Hyperparameters
    model_hyperparams JSON,  -- Model-specific params (learning_rate, etc.)
    
    -- Training Results
    training_metrics JSON,  -- MAE, RMSE, R2, etc. on validation set
    feature_count INTEGER,
    training_duration_seconds REAL,
    
    -- File Management
    model_file_path TEXT,  -- NULL if model deleted
    model_file_size_bytes INTEGER,
    is_model_kept BOOLEAN DEFAULT FALSE,  -- True if model file still exists
    
    -- Regime Performance
    best_regimes JSON,  -- List of regimes where this model is best ["bull", "volatile"]
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    
    -- Provenance
    created_by TEXT DEFAULT 'system',  -- 'ui' | 'api' | 'scheduled'
    config_hash TEXT NOT NULL,  -- SHA256 hash of configuration for deduplication
    
    INDEX idx_status (status),
    INDEX idx_config_hash (config_hash),
    INDEX idx_symbol_timeframe (symbol, base_timeframe),
    INDEX idx_model_type (model_type)
);
```

#### Table: inference_backtests
Tracks every inference backtest on trained models

```sql
CREATE TABLE inference_backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_uuid TEXT UNIQUE NOT NULL,
    training_run_id INTEGER NOT NULL,
    
    -- Inference Configuration (Internal Loop Parameters)
    prediction_method TEXT NOT NULL,
    ensemble_method TEXT,
    confidence_threshold REAL,
    lookback_window INTEGER,
    inference_params JSON,  -- Other inference-specific parameters
    
    -- Backtest Results
    backtest_metrics JSON,  -- Sharpe, max_drawdown, win_rate, etc.
    backtest_duration_seconds REAL,
    
    -- Regime Performance
    regime_metrics JSON,  -- Performance broken down by regime
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    
    -- Relationships
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE,
    
    INDEX idx_training_run (training_run_id),
    INDEX idx_prediction_method (prediction_method)
);
```

#### Table: regime_definitions
Defines market regimes for performance tracking

```sql
CREATE TABLE regime_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    regime_name TEXT UNIQUE NOT NULL,  -- 'bull' | 'bear' | 'volatile' | 'calm' | 'trending' | 'ranging'
    description TEXT,
    detection_rules JSON,  -- Rules for identifying regime from market data
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

#### Table: regime_best_models
Tracks best performing model for each regime

```sql
CREATE TABLE regime_best_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    regime_name TEXT NOT NULL,
    training_run_id INTEGER NOT NULL,
    inference_backtest_id INTEGER NOT NULL,
    
    -- Performance Metrics
    performance_score REAL NOT NULL,  -- Primary metric (e.g., Sharpe ratio)
    secondary_metrics JSON,  -- Other relevant metrics
    
    -- Timestamps
    achieved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Relationships
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (inference_backtest_id) REFERENCES inference_backtests(id) ON DELETE CASCADE,
    FOREIGN KEY (regime_name) REFERENCES regime_definitions(regime_name) ON DELETE CASCADE,
    
    UNIQUE(regime_name),  -- Only one best model per regime
    INDEX idx_regime (regime_name)
);
```

#### Table: training_queue
Manages queued training jobs for interruption/resume

```sql
CREATE TABLE training_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    queue_uuid TEXT UNIQUE NOT NULL,
    
    -- Configuration
    config_grid JSON NOT NULL,  -- Full grid of configurations to train
    current_index INTEGER DEFAULT 0,  -- Index in grid currently processing
    total_configs INTEGER NOT NULL,
    
    -- Status
    status TEXT NOT NULL,  -- 'pending' | 'running' | 'paused' | 'completed' | 'cancelled'
    
    -- Progress Tracking
    completed_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    skipped_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    paused_at DATETIME,
    completed_at DATETIME,
    
    -- Settings
    priority INTEGER DEFAULT 0,  -- Higher = more important
    max_parallel INTEGER DEFAULT 1,  -- Max parallel training jobs
    
    INDEX idx_status (status),
    INDEX idx_priority (priority DESC)
);
```

### 2.2 Schema Migration Strategy

1. **Create migration file**: `alembic revision -m "add_new_training_system"`
2. **Implement upgrade()**: Create all 5 new tables
3. **Implement downgrade()**: Drop all 5 tables
4. **Data migration**: If existing models, migrate to new schema
5. **Apply migration**: `alembic upgrade head`

---

## 3. Core Training Pipeline Logic

### 3.1 Training Orchestrator Class

```
Class: TrainingOrchestrator

Responsibilities:
- Generate configuration grid from user selections
- Iterate over external loop (model training)
- For each trained model, execute internal loop (inference backtest)
- Compare performance against regime bests
- Decide whether to keep or delete model file
- Handle interruption and resume logic
- Update database and file system accordingly

Key Methods:
- generate_config_grid()
- train_single_config()
- backtest_all_inference_configs()
- evaluate_regime_performance()
- update_best_models()
- cleanup_model_files()
- save_checkpoint()
- resume_from_checkpoint()
```

### 3.2 External Loop: Model Training

#### Algorithm

```
FUNCTION train_models_grid(config_grid, resume_from=None):
    
    # Initialize or resume queue
    IF resume_from:
        queue = load_training_queue(resume_from)
        start_index = queue.current_index
    ELSE:
        queue = create_training_queue(config_grid)
        start_index = 0
    
    # Load current regime bests
    regime_bests = load_regime_best_models()
    
    # External loop: iterate model configurations
    FOR index = start_index TO len(config_grid):
        config = config_grid[index]
        
        # Check for cancellation
        IF check_cancellation_requested():
            save_checkpoint(queue, index)
            RETURN "PAUSED"
        
        # Skip if already trained (deduplication)
        IF config_exists_in_db(config):
            CONTINUE
        
        # Train model
        TRY:
            training_run = train_model(config)
            
            # Internal loop: inference backtest
            inference_results = backtest_all_inference_configs(training_run)
            
            # Evaluate against regimes
            improvements = evaluate_regime_improvements(inference_results, regime_bests)
            
            # Decide: keep or delete model
            IF improvements:
                keep_model_file(training_run)
                update_regime_bests(improvements)
                log("✓ Model kept - improved regimes: {}", improvements)
            ELSE:
                delete_model_file(training_run)
                log("✗ Model deleted - no improvements")
            
            # Update database
            save_training_run(training_run, improvements)
            save_inference_backtests(inference_results)
            
            # Update queue progress
            update_queue_progress(queue, index + 1)
            
        CATCH Exception as e:
            log_error("Training failed: {}", e)
            mark_training_failed(config, error=e)
            CONTINUE
    
    # Mark queue as completed
    mark_queue_completed(queue)
    RETURN "COMPLETED"
```

#### Deduplication Logic

Before training, compute configuration hash:

```
FUNCTION compute_config_hash(config):
    # Canonical JSON representation
    canonical = sort_dict_recursively(config)
    json_str = json.dumps(canonical, sort_keys=True)
    hash_value = sha256(json_str)
    RETURN hash_value

FUNCTION config_exists_in_db(config):
    hash_value = compute_config_hash(config)
    existing = query_db("SELECT id FROM training_runs WHERE config_hash = ?", hash_value)
    RETURN existing IS NOT NULL
```

### 3.3 Internal Loop: Inference Backtest

#### Algorithm

```
FUNCTION backtest_all_inference_configs(training_run):
    
    # Load trained model
    model = load_model(training_run.model_file_path)
    
    # Load test data
    test_data = load_test_data(training_run.symbol, training_run.base_timeframe)
    
    # Define inference configuration grid
    inference_grid = generate_inference_grid()
    
    results = []
    
    # Internal loop: iterate inference configurations
    FOR inference_config IN inference_grid:
        
        # Generate predictions
        predictions = model.predict(test_data, **inference_config)
        
        # Simulate trading
        trades = simulate_trading(predictions, test_data)
        
        # Calculate metrics
        metrics = calculate_metrics(trades)
        
        # Evaluate by regime
        regime_metrics = calculate_regime_metrics(trades, test_data)
        
        # Store results
        result = {
            'training_run_id': training_run.id,
            'inference_config': inference_config,
            'metrics': metrics,
            'regime_metrics': regime_metrics
        }
        results.append(result)
    
    RETURN results
```

#### Inference Configuration Grid

```
FUNCTION generate_inference_grid():
    grid = []
    
    FOR prediction_method IN ['direct', 'recursive', 'direct_multi']:
        FOR ensemble_method IN ['mean', 'weighted', 'stacking']:
            FOR confidence_threshold IN [0.0, 0.3, 0.5, 0.7, 0.9]:
                FOR lookback_window IN [50, 100, 200]:
                    config = {
                        'prediction_method': prediction_method,
                        'ensemble_method': ensemble_method,
                        'confidence_threshold': confidence_threshold,
                        'lookback_window': lookback_window
                    }
                    grid.append(config)
    
    RETURN grid
```

### 3.4 Regime Performance Evaluation

#### Algorithm

```
FUNCTION evaluate_regime_improvements(inference_results, current_regime_bests):
    
    improvements = {}
    
    # Aggregate results by regime
    regime_aggregated = aggregate_by_regime(inference_results)
    
    FOR regime_name, metrics IN regime_aggregated:
        current_best = current_regime_bests.get(regime_name)
        
        IF current_best IS NULL:
            # No existing best for this regime
            improvements[regime_name] = {
                'inference_backtest_id': metrics.id,
                'performance_score': metrics.primary_metric,
                'improvement': 'NEW'
            }
        ELSE:
            # Compare against current best
            IF metrics.primary_metric > current_best.performance_score:
                improvement_pct = (metrics.primary_metric - current_best.performance_score) / current_best.performance_score * 100
                improvements[regime_name] = {
                    'inference_backtest_id': metrics.id,
                    'performance_score': metrics.primary_metric,
                    'improvement': f"+{improvement_pct:.2f}%"
                }
    
    RETURN improvements
```

#### Regime Detection

```
FUNCTION classify_regime(market_data, timestamp):
    
    # Calculate regime indicators
    volatility = calculate_realized_volatility(market_data, window=20)
    trend = calculate_trend_strength(market_data, window=50)
    
    # Classify regime
    IF trend > 0.7:
        IF returns > 0:
            RETURN 'bull_trending'
        ELSE:
            RETURN 'bear_trending'
    ELSE IF volatility > percentile(volatility, 75):
        RETURN 'volatile_ranging'
    ELSE:
        RETURN 'calm_ranging'
```

---

## 4. Interruption and Resume Logic

### 4.1 Checkpoint System

#### Checkpoint Structure

```json
{
    "queue_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "checkpoint_version": 1,
    "created_at": "2025-10-07T14:30:00Z",
    
    "config_grid": [...],  // Full configuration grid
    "current_index": 42,    // Index currently processing
    "total_configs": 200,
    
    "completed_runs": [
        {"index": 0, "training_run_id": 1, "kept": true},
        {"index": 1, "training_run_id": 2, "kept": false},
        ...
    ],
    
    "failed_runs": [
        {"index": 15, "error": "GPU out of memory"},
        ...
    ],
    
    "regime_bests": {
        "bull_trending": {"training_run_id": 5, "score": 1.87},
        ...
    },
    
    "statistics": {
        "models_trained": 42,
        "models_kept": 8,
        "models_deleted": 34,
        "elapsed_time_seconds": 14523
    }
}
```

#### Save Checkpoint

```
FUNCTION save_checkpoint(queue, current_index):
    
    checkpoint = {
        'queue_uuid': queue.queue_uuid,
        'checkpoint_version': 1,
        'created_at': datetime.now().isoformat(),
        'config_grid': queue.config_grid,
        'current_index': current_index,
        'total_configs': queue.total_configs,
        'completed_runs': load_completed_runs(queue.id),
        'failed_runs': load_failed_runs(queue.id),
        'regime_bests': load_regime_bests(),
        'statistics': calculate_queue_statistics(queue)
    }
    
    checkpoint_path = get_checkpoint_path(queue.queue_uuid)
    save_json(checkpoint_path, checkpoint)
    
    # Update database
    update_queue_status(queue.id, status='paused', paused_at=datetime.now())
    
    log("Checkpoint saved: {} ({}/{} complete)", 
        checkpoint_path, current_index, queue.total_configs)
```

#### Resume from Checkpoint

```
FUNCTION resume_from_checkpoint(checkpoint_path):
    
    # Load checkpoint
    checkpoint = load_json(checkpoint_path)
    
    # Validate checkpoint version
    IF checkpoint.checkpoint_version != CURRENT_VERSION:
        RAISE Error("Incompatible checkpoint version")
    
    # Recreate queue
    queue = TrainingQueue(
        queue_uuid=checkpoint.queue_uuid,
        config_grid=checkpoint.config_grid,
        current_index=checkpoint.current_index,
        total_configs=checkpoint.total_configs
    )
    
    # Resume training from current index
    log("Resuming from checkpoint: {}/{} complete", 
        checkpoint.current_index, checkpoint.total_configs)
    
    result = train_models_grid(
        config_grid=checkpoint.config_grid,
        resume_from=queue
    )
    
    RETURN result
```

### 4.2 Cancellation Handling

#### User-Initiated Cancellation

```
FUNCTION handle_user_cancellation(queue_uuid):
    
    # Set cancellation flag (atomic)
    set_cancellation_flag(queue_uuid)
    
    # Wait for graceful shutdown
    WHILE queue_is_running(queue_uuid):
        sleep(1)
        IF timeout_exceeded():
            BREAK
    
    # Verify checkpoint saved
    checkpoint = load_latest_checkpoint(queue_uuid)
    IF checkpoint.current_index >= 0:
        log("Training paused successfully at {}/{}", 
            checkpoint.current_index, checkpoint.total_configs)
        RETURN True
    ELSE:
        log_error("Failed to save checkpoint")
        RETURN False
```

#### Automatic Checkpoint Intervals

```
FUNCTION train_with_auto_checkpoints(queue):
    
    last_checkpoint_index = 0
    CHECKPOINT_INTERVAL = 10  // Every 10 models
    
    FOR index IN range(queue.current_index, len(queue.config_grid)):
        
        # Train model
        train_single_config(queue.config_grid[index])
        
        # Auto-checkpoint every N models
        IF index - last_checkpoint_index >= CHECKPOINT_INTERVAL:
            save_checkpoint(queue, index)
            last_checkpoint_index = index
        
        # Check cancellation
        IF check_cancellation_requested():
            save_checkpoint(queue, index)
            BREAK
```

### 4.3 Crash Recovery

```
FUNCTION detect_and_recover_from_crash():
    
    # Find interrupted queues
    interrupted_queues = query_db(
        "SELECT * FROM training_queue WHERE status = 'running' AND updated_at < datetime('now', '-5 minutes')"
    )
    
    FOR queue IN interrupted_queues:
        # Find latest checkpoint
        checkpoint = find_latest_checkpoint(queue.queue_uuid)
        
        IF checkpoint:
            log("Detected crashed training: {}", queue.queue_uuid)
            log("Last checkpoint: {}/{} complete", checkpoint.current_index, queue.total_configs)
            
            # Prompt user or auto-resume
            IF config.auto_resume_on_crash:
                resume_from_checkpoint(checkpoint)
            ELSE:
                notify_user_of_recovery_option(queue, checkpoint)
        ELSE:
            log_error("No checkpoint found for crashed queue: {}", queue.queue_uuid)
            mark_queue_failed(queue.id)
```

---

## 5. GUI Integration

### 5.1 New Training Tab Components

#### Section: Training Queue Manager

**Location**: New tab "Training Queue" or section in existing Training Tab

**Components**:

1. **Configuration Grid Builder**
   - Multi-select for mutually exclusive parameters:
     - Model types: [x] ridge [x] lasso [x] rf [ ] lightning
     - Encoders: [x] none [x] pca [ ] autoencoder
     - Symbols: [x] EUR/USD [x] GBP/USD
     - Timeframes: [x] 1m [x] 5m [x] 15m
     - Day ranges: [x] 7 [x] 30 [x] 90
     - Horizons: [x] 5 [x] 10 [x] 20
   
   - Combined parameters (same as current UI):
     - Indicator timeframes grid (unchanged)
     - Additional features checkboxes (unchanged)
     - Preprocessing parameters (unchanged)
   
   - **Calculate Grid Size** button:
     - Shows: "This will create X model configurations"
     - Estimates: "Total training time: ~Y hours"

2. **Queue Control Panel**
   - **Start Training Queue** button
     - Creates training_queue entry
     - Begins external loop
   
   - **Pause Training** button
     - Requests graceful stop
     - Saves checkpoint
     - Shows: "Pausing after current model completes..."
   
   - **Resume Training** button
     - Lists available checkpoints
     - Loads selected checkpoint
     - Resumes from saved index
   
   - **Cancel Training** button
     - Stops immediately
     - Keeps completed work
     - Deletes checkpoint

3. **Progress Display**
   - Overall progress bar: "Training model 42 of 200 (21%)"
   - Current model: "EUR/USD 1h horizon=5 model=rf encoder=pca"
   - Current inference backtest: "Testing inference config 15 of 135"
   - Regime improvements: "✓ bull_trending +12.5% | ✓ volatile_ranging +3.2%"
   - Models kept/deleted: "Kept: 8 models | Deleted: 34 models"
   - Elapsed time: "Runtime: 4h 23m | Est. remaining: 18h 15m"

4. **Recent Results Table**
   - Columns: Model Config | Training Time | Inference Tests | Best Regimes | Status | Actions
   - Row actions:
     - View Details: Opens detail dialog
     - Compare: Multi-select for comparison
     - Export: Save configuration to JSON

#### Section: Regime Performance Dashboard

**Location**: New tab "Regime Analysis"

**Components**:

1. **Regime Best Models Table**
   - Columns: Regime | Best Model | Score | Improvement | Last Updated
   - Rows:
     - bull_trending | EUR/USD_1h_rf_pca | Sharpe=1.87 | +12.5% | 2025-10-07 14:30
     - bear_trending | GBP/USD_5m_ridge_none | Sharpe=1.23 | +3.2% | 2025-10-07 13:15
     - ...
   
   - Actions:
     - View: Shows full model details
     - Backtest: Run additional backtests
     - Deploy: Add to production ensemble

2. **Regime Performance Charts**
   - Time series: Regime detection over time
   - Heatmap: Model performance across all regimes
   - Scatter: Risk vs Return by regime

3. **Regime Definition Manager**
   - Add/Edit/Delete custom regimes
   - Configure detection rules
   - Test regime detection on historical data

#### Section: Training History Explorer

**Location**: New tab "Training History"

**Components**:

1. **Search and Filter**
   - Filters:
     - Status: [x] Completed [ ] Failed [ ] Running
     - Model type: [x] All [ ] ridge [ ] rf [ ] lightning
     - Date range: [From] [To]
     - Performance: [Min Sharpe] [Max Drawdown]
   
   - Search: "Search by symbol, config, notes..."

2. **Results Table**
   - Columns: Timestamp | Config | Performance | Regimes | Model File | Actions
   - Sortable by any column
   - Pagination: 50 per page
   
   - Row actions:
     - View Details
     - Re-run Inference
     - Export Config
     - Delete (if not best for any regime)

3. **Bulk Actions**
   - Select multiple rows
   - Compare selected models
   - Export configurations
   - Delete models (with confirmation)

### 5.2 Modified Existing Components

#### Training Tab Changes

1. **Mode Selector** (NEW at top)
   - Radio buttons:
     - ( ) Single Training: Train one model (current behavior)
     - (●) Grid Training: Train multiple configurations (new)

2. **Grid Training Mode** (NEW panel, shown when Grid mode selected)
   - All existing single-model controls become multi-selectable
   - Removed: "Start Training" button
   - Added: "Add to Queue" button → Opens queue manager

3. **Single Training Mode** (existing behavior, no changes)
   - All current controls unchanged
   - "Start Training" button works as before

### 5.3 Workflow Example: User Creates Training Queue

#### User Actions:

1. User opens Training Tab
2. Selects "Grid Training" mode
3. Configures parameter grid:
   - Models: [x] ridge [x] rf
   - Encoders: [x] none [x] pca
   - Symbols: [x] EUR/USD
   - Timeframes: [x] 1h [x] 4h
   - Days: [x] 30
   - Horizons: [x] 5 [x] 10
   
   **Grid size: 2×2×1×2×1×2 = 16 configurations**

4. Clicks "Calculate Estimates"
   - System shows:
     - "16 models will be trained"
     - "Each model: ~135 inference backtests"
     - "Total: 2,160 backtest combinations"
     - "Estimated time: 8 hours (ridge), 32 hours (rf)"

5. Clicks "Create Training Queue"
   - Opens queue manager dialog
   - Shows configuration summary
   - User clicks "Start Training"

6. System begins external loop:
   - Progress bar appears
   - Log shows:
     ```
     [14:30:00] Starting training queue: 16 configurations
     [14:30:05] Training model 1/16: EUR/USD_1h_h5_ridge_none
     [14:32:15] Model trained in 2m 10s
     [14:32:15] Running 135 inference backtests...
     [14:35:45] Backtests complete in 3m 30s
     [14:35:45] Best regimes: bull_trending (Sharpe=1.42)
     [14:35:45] ✓ Model kept - new best for bull_trending
     [14:35:46] Training model 2/16: EUR/USD_1h_h5_ridge_pca
     ...
     ```

7. User clicks "Pause Training" after 8 models
   - System finishes current model
   - Saves checkpoint
   - Shows: "Paused at 8/16 models complete"

8. User clicks "Resume Training" next day
   - System loads checkpoint
   - Continues from model 9/16
   - Completes remaining 8 models

9. User views results in "Regime Analysis" tab
   - Sees best model for each regime
   - 16 models trained, 4 kept, 12 deleted
   - Storage: 40 MB (models) + 2 MB (metadata)

### 5.4 Error Handling and User Feedback

#### Training Failures

```
IF training fails for a configuration:
    1. Show error dialog:
       - "Training failed for EUR/USD_1h_rf_pca"
       - "Error: GPU out of memory"
       - Options: [Skip] [Retry] [Cancel Queue]
    
    2. Log error to database:
       - training_runs.status = 'failed'
       - training_runs.error_message = "GPU OOM"
    
    3. Continue to next configuration (if Skip selected)
    4. Update progress display:
       - "Models: 8 completed, 1 failed, 7 remaining"
```

#### Invalid Configurations

```
IF configuration is invalid:
    1. Show validation error:
       - "Invalid configuration: horizon > days_history"
       - "Fix: Increase days_history or decrease horizon"
    
    2. Highlight problematic fields in UI
    3. Prevent queue creation until fixed
```

#### Disk Space Issues

```
IF disk space low:
    1. Show warning:
       - "Warning: Only 2 GB free space remaining"
       - "Training may need 5 GB for this queue"
       - Options: [Continue Anyway] [Cancel] [Change Output Dir]
    
    2. If space exhausted during training:
       - Pause queue
       - Save checkpoint
       - Show error: "Disk full - training paused"
       - "Free up space and click Resume"
```

---

## 6. Implementation Checklist

### 6.1 Phase 1: Database Foundation

- [ ] Create Alembic migration: `20251007_add_new_training_system.py`
- [ ] Implement `upgrade()`: Create 5 new tables
- [ ] Implement `downgrade()`: Drop tables safely
- [ ] Add indexes for performance
- [ ] Test migration on development database
- [ ] Add database models (SQLAlchemy ORM classes)
- [ ] Write database access layer (DAL) functions

### 6.2 Phase 2: Core Training Logic

- [ ] Create `TrainingOrchestrator` class
  - [ ] `generate_config_grid()`
  - [ ] `create_training_queue()`
  - [ ] `train_single_config()`
  - [ ] `compute_config_hash()`
  - [ ] `config_exists_in_db()`

- [ ] Create `InferenceBacktester` class
  - [ ] `generate_inference_grid()`
  - [ ] `backtest_single_inference()`
  - [ ] `calculate_metrics()`
  - [ ] `calculate_regime_metrics()`

- [ ] Create `RegimeManager` class
  - [ ] `classify_regime()`
  - [ ] `evaluate_regime_improvements()`
  - [ ] `update_regime_bests()`
  - [ ] `load_regime_definitions()`

- [ ] Create `ModelFileManager` class
  - [ ] `keep_model_file()`
  - [ ] `delete_model_file()`
  - [ ] `cleanup_orphaned_files()`

### 6.3 Phase 3: Interruption/Resume System

- [ ] Create `CheckpointManager` class
  - [ ] `save_checkpoint()`
  - [ ] `load_checkpoint()`
  - [ ] `validate_checkpoint()`
  - [ ] `list_available_checkpoints()`

- [ ] Implement cancellation handling
  - [ ] Atomic cancellation flag (threading.Event)
  - [ ] Graceful shutdown in training loop
  - [ ] Auto-checkpoint on cancellation

- [ ] Implement crash recovery
  - [ ] Detect interrupted queues on startup
  - [ ] Prompt user to resume
  - [ ] Auto-resume option (config setting)

### 6.4 Phase 4: GUI Implementation

#### Training Tab Modifications

- [ ] Add training mode selector (Single vs Grid)
- [ ] Convert single-select controls to multi-select for Grid mode
- [ ] Add "Calculate Grid Size" button
- [ ] Add "Create Training Queue" button
- [ ] Keep existing Single Training mode unchanged

#### New Training Queue Tab

- [ ] Create QTabWidget for "Training Queue"
- [ ] Implement Configuration Grid Builder section
- [ ] Implement Queue Control Panel
  - [ ] Start/Pause/Resume/Cancel buttons
  - [ ] Checkpoint selector dialog
  - [ ] Confirmation dialogs
- [ ] Implement Progress Display
  - [ ] Overall progress bar
  - [ ] Current model/inference display
  - [ ] Regime improvements display
  - [ ] Statistics display
- [ ] Implement Recent Results Table
  - [ ] Table model for training_runs
  - [ ] Detail view dialog
  - [ ] Compare functionality
  - [ ] Export to JSON

#### New Regime Analysis Tab

- [ ] Create QTabWidget for "Regime Analysis"
- [ ] Implement Regime Best Models Table
  - [ ] Table model for regime_best_models
  - [ ] Actions: View/Backtest/Deploy
- [ ] Implement Regime Performance Charts
  - [ ] Time series chart (matplotlib/pyqtgraph)
  - [ ] Heatmap chart
  - [ ] Scatter chart
- [ ] Implement Regime Definition Manager
  - [ ] CRUD dialogs for regimes
  - [ ] Detection rule editor
  - [ ] Test detection tool

#### New Training History Tab

- [ ] Create QTabWidget for "Training History"
- [ ] Implement Search and Filter section
  - [ ] Filter controls (comboboxes, date pickers)
  - [ ] Search text field
  - [ ] Apply/Reset buttons
- [ ] Implement Results Table
  - [ ] Paginated table model
  - [ ] Sortable columns
  - [ ] Row actions menu
- [ ] Implement Bulk Actions toolbar
  - [ ] Multi-select checkboxes
  - [ ] Bulk action buttons

### 6.5 Phase 5: Worker Threads and Async

- [ ] Create `TrainingWorker` (QThread)
  - [ ] Emit progress signals
  - [ ] Emit log signals
  - [ ] Handle cancellation
  - [ ] Exception handling

- [ ] Create `BacktestWorker` (QThread)
  - [ ] Parallel inference backtests
  - [ ] Progress reporting
  - [ ] Results aggregation

- [ ] Connect signals to GUI
  - [ ] Progress bar updates
  - [ ] Log view updates
  - [ ] Table updates on completion
  - [ ] Error message dialogs

### 6.6 Phase 6: Testing

- [ ] Unit tests for core logic
  - [ ] `test_config_hash_deterministic()`
  - [ ] `test_checkpoint_save_load()`
  - [ ] `test_regime_classification()`
  - [ ] `test_model_file_cleanup()`

- [ ] Integration tests
  - [ ] `test_full_training_pipeline()`
  - [ ] `test_resume_from_checkpoint()`
  - [ ] `test_regime_improvements_detection()`

- [ ] GUI tests
  - [ ] `test_create_training_queue_ui()`
  - [ ] `test_pause_resume_ui()`
  - [ ] `test_regime_dashboard_ui()`

### 6.7 Phase 7: Documentation

- [ ] Update USER_TRAINING_GUIDE.md
  - [ ] New Grid Training mode section
  - [ ] Interruption/Resume tutorial
  - [ ] Regime Analysis tutorial

- [ ] Create TRAINING_PIPELINE_COMPLETE_GUIDE_V2.md
  - [ ] Architecture explanation
  - [ ] Parameter classification guide
  - [ ] Best practices for grid configuration

- [ ] Create REGIME_ANALYSIS_GUIDE.md
  - [ ] What are regimes?
  - [ ] How regime detection works
  - [ ] How to define custom regimes
  - [ ] Interpreting regime performance

- [ ] Update API_README.md (if API access needed)

---

## 7. Configuration and Settings

### 7.1 Configuration File: config/training_pipeline.yaml

```yaml
training_pipeline:
  # Storage settings
  artifacts_dir: "./artifacts"
  checkpoints_dir: "./checkpoints"
  max_checkpoint_age_days: 30  # Auto-delete old checkpoints
  
  # Queue settings
  max_parallel_queues: 1  # Max simultaneous queues
  auto_checkpoint_interval: 10  # Checkpoint every N models
  max_inference_workers: 4  # Parallel inference backtests
  
  # Recovery settings
  auto_resume_on_crash: false  # Auto-resume interrupted queues
  detect_crash_after_minutes: 5  # Consider crashed if no activity for N minutes
  
  # Model file management
  delete_non_best_models: true  # Delete models not best for any regime
  keep_top_n_per_regime: 1  # Keep top N models per regime
  compress_old_models: true  # Compress models older than 30 days
  
  # Performance settings
  model_cache_size_mb: 1000  # Cache recent models in memory
  database_connection_pool_size: 5
  
  # Regime definitions
  regimes:
    - name: "bull_trending"
      description: "Strong upward trend"
      detection_rules:
        trend_strength: "> 0.7"
        returns: "> 0"
    
    - name: "bear_trending"
      description: "Strong downward trend"
      detection_rules:
        trend_strength: "> 0.7"
        returns: "< 0"
    
    - name: "volatile_ranging"
      description: "High volatility, no clear trend"
      detection_rules:
        trend_strength: "< 0.3"
        volatility: "> 75th percentile"
    
    - name: "calm_ranging"
      description: "Low volatility, no clear trend"
      detection_rules:
        trend_strength: "< 0.3"
        volatility: "< 50th percentile"
  
  # Inference grid defaults
  inference_grid:
    prediction_methods:
      - "direct"
      - "recursive"
      - "direct_multi"
    
    ensemble_methods:
      - "mean"
      - "weighted"
      - "stacking"
    
    confidence_thresholds:
      - 0.0
      - 0.3
      - 0.5
      - 0.7
      - 0.9
    
    lookback_windows:
      - 50
      - 100
      - 200
  
  # Notification settings
  notifications:
    enabled: true
    email_on_queue_complete: true
    email_on_queue_failure: true
    email_on_new_regime_best: true
```

### 7.2 User Settings: ~/.forexgpt/training_settings.json

This file stores user preferences (as currently implemented):
- Last selected parameters
- UI state (window sizes, splitter positions)
- Recent checkpoints list
- Custom regime definitions

---

## 8. Performance Optimization

### 8.1 Database Optimization

1. **Indexes**: Add composite indexes for common queries
   ```sql
   CREATE INDEX idx_runs_symbol_tf_status 
   ON training_runs(symbol, base_timeframe, status);
   
   CREATE INDEX idx_backtests_run_method 
   ON inference_backtests(training_run_id, prediction_method);
   ```

2. **Connection Pooling**: Use SQLAlchemy connection pool
3. **Batch Inserts**: Insert multiple inference results at once
4. **Vacuum**: Periodically run `VACUUM` to defragment database

### 8.2 Training Optimization

1. **Model Caching**: Keep recently trained models in memory
2. **Parallel Inference**: Run inference backtests in parallel (ThreadPoolExecutor)
3. **GPU Utilization**: Use GPU for neural network models (lightning, diffusion)
4. **Data Caching**: Cache preprocessed data for same symbol/timeframe

### 8.3 File System Optimization

1. **Compressed Models**: Use `joblib` compression for model files
2. **Lazy Loading**: Load models only when needed
3. **Async File Deletion**: Delete model files asynchronously
4. **SSD Recommendation**: Store artifacts on SSD for fast I/O

---

## 9. Testing Strategy

### 9.1 Unit Tests

Location: `tests/training_pipeline/`

Files:
- `test_config_hash.py`: Test configuration hashing and deduplication
- `test_checkpoint.py`: Test checkpoint save/load/validate
- `test_regime_detection.py`: Test regime classification logic
- `test_model_file_mgmt.py`: Test model file keep/delete logic
- `test_inference_grid.py`: Test inference configuration generation

### 9.2 Integration Tests

Location: `tests/integration/training_pipeline/`

Files:
- `test_full_pipeline.py`: End-to-end training pipeline test
- `test_resume.py`: Test interruption and resume
- `test_regime_updates.py`: Test regime best model updates
- `test_database_integrity.py`: Test database constraints and relationships

### 9.3 GUI Tests

Location: `tests/gui/training_pipeline/`

Files:
- `test_queue_creation_ui.py`: Test queue creation workflow
- `test_pause_resume_ui.py`: Test pause/resume buttons
- `test_regime_dashboard_ui.py`: Test regime analysis tab
- `test_progress_updates_ui.py`: Test progress display updates

### 9.4 Performance Tests

Location: `tests/performance/training_pipeline/`

Files:
- `test_large_grid.py`: Test with 1000+ configurations
- `test_parallel_inference.py`: Test parallel backtest performance
- `test_database_throughput.py`: Test DB insert/query performance

### 9.5 Test Data

Create synthetic test data:
- Small grid: 4 configurations (quick test, <1 minute)
- Medium grid: 50 configurations (integration test, ~10 minutes)
- Large grid: 500 configurations (stress test, ~2 hours)

---

## 10. Migration Plan

### 10.1 Backward Compatibility

**Goal**: New system should work alongside existing single-training workflow.

**Strategy**:
1. Keep existing single-training code path intact
2. Add new grid-training code path as separate module
3. GUI mode selector chooses which path to use
4. Existing trained models remain usable

### 10.2 Data Migration

If existing trained models need to be imported into new system:

```sql
-- Create migration script
INSERT INTO training_runs (
    run_uuid,
    status,
    model_type,
    encoder,
    symbol,
    base_timeframe,
    days_history,
    horizon,
    model_file_path,
    is_model_kept,
    created_at
)
SELECT
    uuid(),
    'completed',
    COALESCE(metadata.model_type, 'unknown'),
    COALESCE(metadata.encoder, 'none'),
    metadata.symbol,
    metadata.base_timeframe,
    metadata.days_history,
    metadata.horizon_bars,
    model_path,
    TRUE,  -- All existing models are kept
    metadata.created_at
FROM existing_models_metadata;
```

### 10.3 Rollback Plan

If new system has critical issues:

1. **Code Rollback**: Git revert to pre-implementation commit
2. **Database Rollback**: `alembic downgrade -1`
3. **Model Files**: No impact (existing models untouched)
4. **User Data**: Training settings file unchanged

---

## 11. Future Enhancements

### 11.1 Distributed Training

**Goal**: Train multiple models in parallel across multiple machines.

**Implementation**:
- Replace local queue with distributed queue (Celery + Redis)
- Each worker pulls next configuration from queue
- Centralized database for results
- Requires: Redis, Celery, shared file system

### 11.2 AutoML Integration

**Goal**: Automatically suggest optimal configurations based on historical performance.

**Implementation**:
- Analyze training_runs and inference_backtests tables
- Use Bayesian optimization to suggest next configurations
- Prioritize configurations likely to improve regime performance
- Requires: scikit-optimize or Optuna

### 11.3 Real-time Regime Detection

**Goal**: Detect regime changes in live trading and switch models.

**Implementation**:
- Monitor live market data
- Classify current regime every N minutes
- Automatically use best model for current regime
- Requires: Real-time data feed, regime detection service

### 11.4 Model Ensemble Across Regimes

**Goal**: Create meta-model that selects best model based on regime.

**Implementation**:
- Train ensemble model: input=regime features, output=model selection
- Use historical regime_best_models data for training
- Deploy ensemble in production
- Requires: Ensemble training module

### 11.5 Cloud Deployment

**Goal**: Run training queues on cloud GPUs (AWS, GCP, Azure).

**Implementation**:
- Dockerize training pipeline
- Submit queue to cloud (e.g., AWS Batch, Google Cloud AI Platform)
- Stream logs and results back to local database
- Requires: Cloud provider account, Docker, cloud SDK

---

## 12. Success Metrics

### 12.1 Performance Metrics

- **Training Speed**: 10-100x faster backtesting vs old system
- **Storage Efficiency**: 90% reduction in model storage
- **Time to Best Model**: <4 hours for 100-configuration grid (vs 40 hours old system)

### 12.2 Usability Metrics

- **Time to Create Queue**: <2 minutes (user sets up grid, clicks start)
- **Resume Success Rate**: >99% (checkpoints work reliably)
- **Error Recovery Rate**: >95% (system handles failures gracefully)

### 12.3 Quality Metrics

- **Regime Detection Accuracy**: >85% (regimes correctly identified)
- **Best Model Improvement**: >10% average improvement over baseline per regime
- **Model Retention Rate**: 5-15% of trained models kept (target: low to save storage)

---

## 13. Risk Mitigation

### 13.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Database corruption | High | Low | Daily backups, transaction safety |
| Checkpoint file corruption | Medium | Low | Versioning, checksum validation |
| Out of disk space | High | Medium | Monitor disk, auto-pause if low |
| GPU out of memory | Medium | Medium | Catch exception, skip config, log error |
| Training never completes | High | Low | Timeout per model, auto-skip after N failures |

### 13.2 Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| User loses progress | High | Medium | Auto-checkpoint every 10 models |
| Unclear which model to deploy | Medium | High | Regime dashboard, clear "best" indication |
| Too many stored models | Medium | Medium | Auto-delete non-best models |
| Can't reproduce results | High | Low | Store complete config + hash in DB |

### 13.3 User Experience Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Complex UI confuses users | Medium | Medium | Tutorial, tooltips, mode selector |
| Long wait times frustrate users | High | High | Progress display, time estimates |
| Difficult to interpret results | Medium | High | Clear charts, metric explanations |

---

## 14. Documentation Requirements

### 14.1 User Documentation

1. **Quick Start Guide**: Grid Training in 5 Minutes
2. **Complete Tutorial**: Step-by-step queue creation
3. **Regime Analysis Guide**: Understanding and using regimes
4. **Troubleshooting Guide**: Common errors and fixes

### 14.2 Developer Documentation

1. **Architecture Overview**: System design, data flow diagrams
2. **API Reference**: All classes and methods
3. **Database Schema**: ERD, table descriptions
4. **Extension Guide**: How to add new regime types, inference methods

### 14.3 Operational Documentation

1. **Deployment Guide**: Installation and configuration
2. **Backup and Recovery**: Database and checkpoint backup procedures
3. **Performance Tuning**: Optimizing for large grids
4. **Monitoring**: Key metrics to watch

---

## 15. Conclusion

This specification defines a comprehensive, production-ready AI training pipeline that dramatically improves training efficiency while maintaining complete auditability and recoverability. The two-phase architecture (training + inference backtest) leverages the fact that inference is much faster than training, enabling rapid iteration on inference strategies without costly retraining.

Key benefits:
- **10-100x faster** optimization through inference-only backtesting
- **90% storage reduction** by keeping only best models
- **Complete auditability** of all training attempts
- **Robust interruption/resume** for long-running queues
- **Regime-aware selection** for robust production deployment

The system is designed for **production use**, with proper error handling, database integrity, checkpoint recovery, and user-friendly GUI integration.

**Estimated implementation time**: 3-4 weeks
- Week 1: Database and core training logic
- Week 2: Interruption/resume and file management
- Week 3: GUI implementation and integration
- Week 4: Testing, documentation, refinement

---

## Appendix A: Configuration Parameter Reference

### External Loop Parameters (Mutually Exclusive)

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| model_type | string | ridge, lasso, elasticnet, rf, lightning, diffusion-ddpm, diffusion-ddim, sssd | ML algorithm |
| encoder | string | none, pca, autoencoder, vae, latents | Dimensionality reduction |
| symbol | string | EUR/USD, GBP/USD, ... | Currency pair |
| base_timeframe | string | 1m, 5m, 15m, 30m, 1h, 4h, 1d | Candlestick resolution |
| days_history | integer | 1-3650 | Days of historical data |
| horizon | integer | 1-500 | Prediction horizon in bars |

### External Loop Parameters (Combinable)

| Parameter | Type | Description |
|-----------|------|-------------|
| indicator_tfs | dict | Mapping of indicators to timeframe lists |
| returns_volatility | boolean | Include returns/volatility features |
| trading_sessions | boolean | Include session indicators |
| candlestick_patterns | boolean | Include pattern recognition |
| volume_profile | boolean | Include volume distribution |
| vsa_analysis | boolean | Include VSA features |
| warmup_bars | integer | Bars to discard for indicator warmup |
| rv_window | integer | Window for realized volatility |
| min_coverage | float | Min fraction of non-NaN values |
| ... | ... | (All preprocessing parameters) |

### Inference Loop Parameters

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| prediction_method | string | direct, recursive, direct_multi | How to generate predictions |
| ensemble_method | string | mean, weighted, stacking | How to combine predictions |
| confidence_threshold | float | 0.0-1.0 | Min confidence for trade execution |
| lookback_window | integer | 20-500 | Historical bars for inference context |

---

## Appendix B: Database Schema ERD

```
┌────────────────────────┐
│   training_queue       │
│────────────────────────│
│ id (PK)                │
│ queue_uuid (UNIQUE)    │
│ config_grid (JSON)     │
│ current_index          │
│ total_configs          │
│ status                 │
│ created_at             │
└────────────────────────┘
            │
            │ 1:N
            ▼
┌────────────────────────┐       ┌──────────────────────┐
│   training_runs        │──────▶│  inference_backtests │
│────────────────────────│ 1:N   │──────────────────────│
│ id (PK)                │       │ id (PK)              │
│ run_uuid (UNIQUE)      │       │ backtest_uuid        │
│ status                 │       │ training_run_id (FK) │
│ model_type             │       │ prediction_method    │
│ encoder                │       │ ensemble_method      │
│ symbol                 │       │ confidence_threshold │
│ base_timeframe         │       │ backtest_metrics     │
│ days_history           │       │ regime_metrics       │
│ horizon                │       │ created_at           │
│ indicator_tfs (JSON)   │       └──────────────────────┘
│ training_metrics       │                │
│ model_file_path        │                │
│ is_model_kept          │                │
│ best_regimes (JSON)    │                │
│ config_hash            │                │ N:1
│ created_at             │                ▼
└────────────────────────┘       ┌──────────────────────┐
            │                     │ regime_best_models   │
            │ N:1                 │──────────────────────│
            └────────────────────▶│ id (PK)              │
                                  │ regime_name (FK)     │
                                  │ training_run_id (FK) │
                                  │ inference_bt_id (FK) │
                                  │ performance_score    │
                                  │ achieved_at          │
                                  └──────────────────────┘
                                            │
                                            │ N:1
                                            ▼
                                  ┌──────────────────────┐
                                  │ regime_definitions   │
                                  │──────────────────────│
                                  │ id (PK)              │
                                  │ regime_name (UNIQUE) │
                                  │ description          │
                                  │ detection_rules      │
                                  │ is_active            │
                                  └──────────────────────┘
```

---

*End of Specification Document*
