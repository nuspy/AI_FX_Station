# ForexGPT - Generative Forecast System: Complete Workflow

**Version**: 2.0.0  
**Last Updated**: 2025-10-13  
**Status**: Production

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Training System Overview](#training-system-overview)
4. [Training Workflows](#training-workflows)
5. [Model Types & Capabilities](#model-types--capabilities)
6. [Feature Engineering Pipeline](#feature-engineering-pipeline)
7. [Data Standardization](#data-standardization)
8. [Optimization Systems](#optimization-systems)
9. [Inference Pipeline](#inference-pipeline)
10. [Backtesting & Validation](#backtesting--validation)
11. [Complete Parameter Reference](#complete-parameter-reference)
12. [Workflow Diagrams](#workflow-diagrams)

---

## Executive Summary

The **Generative Forecast System** is ForexGPT's core ML/AI pipeline for probabilistic time-series prediction. It supports multiple model architectures (traditional ML, VAE+Diffusion, SSSD) with comprehensive training, optimization, and inference capabilities.

**Key Components**:
- **7 Training Scripts**: Traditional ML, Deep Learning, SSSD, Optimized variants
- **3 Main Entry Points**: GUI (Training Tab), CLI, Training Orchestrator
- **80+ Technical Indicators**: Multi-timeframe feature engineering
- **4 Optimization Methods**: Grid Search, Genetic Algorithm, NSGA-II, Optuna
- **Probabilistic Inference**: Multi-horizon, ensemble, quantile predictions

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Entry Points                     │
│  GUI (Training Tab) | CLI Scripts | Training Orchestrator    │
└────────────┬────────────────────────────────────────────────┘
             │
      ┌──────┴──────┬──────────┬──────────┬──────────┐
      │             │          │          │          │
┌─────▼──────┐ ┌────▼────┐ ┌──▼───┐ ┌────▼────┐ ┌──▼────┐
│train_sklearn│ │train.py│ │SSSD  │ │inproc.py│ │Optim. │
│  (CLI)      │ │(Lightn)│ │Trainer│ │ (GUI)   │ │ Engine│
└─────┬──────┘ └────┬────┘ └──┬───┘ └────┬────┘ └──┬────┘
      │             │          │          │          │
      └─────────────┴──────────┴──────────┴──────────┘
                           │
              ┌────────────▼────────────┐
              │  Feature Engineering    │
              │  (Indicators, Features) │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Data Standardization  │
              │   (Mean/Std, Scaler)    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Model Training        │
              │   (Fit, Validate, Save) │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Artifact Storage      │
              │   (Model + Metadata)    │
              └─────────────────────────┘
```

---

## Training System Overview

### Training Scripts Matrix

| Script | Purpose | Model Type | Entry Point | GPU Support | Status |
|--------|---------|------------|-------------|-------------|--------|
| `train_sklearn.py` | Traditional ML (Ridge, RF, etc.) | scikit-learn | CLI | ❌ | ✅ Production |
| `train_sklearn_btalib.py` | Enhanced ML with 80+ indicators | scikit-learn | CLI | ❌ | ✅ Production |
| `train.py` | VAE + Diffusion | PyTorch Lightning | CLI/GUI | ✅ | ✅ Production |
| `train_sssd.py` | SSSD (S4 + Diffusion) | PyTorch | CLI | ✅ | ⚠️ Experimental |
| `train_optimized.py` | NVIDIA-optimized training | PyTorch Lightning | CLI | ✅ | ✅ Production |
| `inproc.py` | In-process training wrapper | scikit-learn | GUI only | ❌ | ✅ Production |
| `loop.py` | PyTorch Lightning module | PyTorch Lightning | Internal | ✅ | ✅ Production |

### Entry Points

#### 1. GUI Training Tab (`ui/training_tab.py`)

**Path**: User Interface → Training Tab → Start Training

**Workflow**:
```python
TrainingTab
  ↓
TrainingController (controllers/training_controller.py)
  ↓
train_sklearn_inproc() (training/inproc.py)
  ↓
train_sklearn module functions
  ↓
Model artifact saved
```

**Features**:
- Interactive parameter configuration
- Real-time progress tracking
- Indicator × Timeframe selection grid (18 indicators × 7 timeframes = 126 checkboxes)
- Advanced parameter tuning
- Genetic Algorithm optimization (optional)

#### 2. CLI Direct Training

**Path**: Command line → Training script

**Examples**:
```bash
# Traditional ML (Ridge)
python -m forex_diffusion.training.train_sklearn \
  --symbol "EUR/USD" \
  --timeframe "1m" \
  --horizon 60 \
  --days_history 90 \
  --algo ridge \
  --artifacts_dir ./artifacts

# VAE + Diffusion
python -m forex_diffusion.training.train \
  --symbol "EUR/USD" \
  --timeframe "1m" \
  --horizon 60 \
  --patch_len 64 \
  --epochs 30 \
  --artifacts_dir ./artifacts

# SSSD
python -m forex_diffusion.training.train_sssd \
  --config configs/sssd/default_config.yaml \
  --asset EUR/USD
```

#### 3. Training Orchestrator

**Path**: Database queue → Orchestrator → Grid training

**Workflow**:
```python
TrainingOrchestrator.create_training_queue(grid_params)
  ↓
TrainingOrchestrator.train_models_grid(queue_id)
  ↓
For each config:
  train_single_config()
    ↓
  InferenceBacktester.backtest_all_inference_configs()
    ↓
  RegimeManager.evaluate_regime_performance()
    ↓
  ModelFileManager.keep_or_delete_model()
```

**Features**:
- Grid search over hyperparameters
- Automatic inference backtesting (Internal Loop)
- Regime-aware model selection
- Auto-delete non-best models

---

## Training Workflows

### Workflow 1: Traditional ML (Ridge/RF/ElasticNet)

**Script**: `train_sklearn.py` or `inproc.py` (via GUI)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Loading                                              │
│    fetch_candles_from_db(symbol, timeframe, days_history)   │
│    → DataFrame[ts_utc, open, high, low, close, volume]      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 2. Feature Engineering                                       │
│    _build_features(candles, args)                           │
│    ├─ Relative OHLC: log(price / prev_close)               │
│    ├─ Temporal Features: hour_sin, hour_cos, dow_sin, etc.  │
│    ├─ Realized Volatility: rolling std of log returns       │
│    └─ Multi-TF Indicators:                                  │
│        For each (indicator, timeframe):                     │
│          - Fetch/resample timeframe data                    │
│          - Compute indicator (ATR, RSI, MACD, etc.)         │
│          - Merge back to base timeframe                     │
│    → X (features), y (targets), metadata                    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 3. Data Standardization (CRITICAL: No Look-Ahead Bias)      │
│    _standardize_train_val(X, y, val_frac)                   │
│    ├─ Split: train_size = n × (1 - val_frac)               │
│    ├─ Compute stats ONLY on training set:                  │
│    │    mu = mean(X_train), sigma = std(X_train)           │
│    ├─ Apply to both:                                        │
│    │    X_train_norm = (X_train - mu) / sigma              │
│    │    X_val_norm = (X_val - mu) / sigma                  │
│    ├─ Verification: KS test for distribution difference     │
│    │    (p-value < 0.5 expected, warns if > 0.8)           │
│    └─ Save scaler: (mu, sigma) for inference               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 4. Optional PCA Encoding                                     │
│    if pca_components > 0:                                    │
│      pca = PCA(n_components=pca_components)                 │
│      X_train_pca = pca.fit_transform(X_train_norm)          │
│      X_val_pca = pca.transform(X_val_norm)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 5. Model Training                                            │
│    _fit_model(algo, X_train, y_train, args)                │
│    ├─ Ridge: Ridge(alpha=args.alpha)                       │
│    ├─ Lasso: Lasso(alpha=args.alpha)                       │
│    ├─ ElasticNet: ElasticNet(alpha, l1_ratio)              │
│    └─ RandomForest: RF(n_estimators, max_depth)            │
│    model.fit(X_train, y_train)                             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 6. Validation                                                │
│    y_pred_val = model.predict(X_val)                        │
│    val_mae = mean_absolute_error(y_val, y_pred_val)        │
│    val_r2 = r2_score(y_val, y_pred_val)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 7. Artifact Saving                                           │
│    Path: artifacts_dir / {symbol}_{tf}_{algo}_h{horizon}.pkl│
│    ├─ Save model: joblib.dump(model, path)                 │
│    ├─ Save metadata (sidecar JSON):                        │
│    │    - symbol, timeframe, horizon_bars                  │
│    │    - feature_columns                                  │
│    │    - mu, sigma (scaler stats)                         │
│    │    - pca_model (if used)                              │
│    │    - indicator_tfs (multi-TF config)                  │
│    │    - val_mae, val_r2                                  │
│    │    - warmup_bars, rv_window (for inference)           │
│    └─ Log: "Model saved to {path}, val_mae={val_mae:.4f}"  │
└─────────────────────────────────────────────────────────────┘
```

**Key Functions**:
- `fetch_candles_from_db()`: SQLAlchemy DB query
- `_build_features()`: Feature engineering pipeline
- `_indicators()`: Multi-TF indicator computation
- `_standardize_train_val()`: Standardization with bias detection
- `_fit_model()`: Model instantiation and fitting

---

### Workflow 2: Deep Learning (VAE + Diffusion)

**Script**: `train.py` + `loop.py` (ForexDiffusionLit)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Loading                                              │
│    fetch_candles_from_db(symbol, timeframe, days_history)   │
│    _add_time_features(df) → hour_sin, hour_cos             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 2. Patch Construction                                        │
│    _build_arrays(df, patch_len, horizon, warmup)           │
│    For each valid window:                                   │
│      patch = df[start:start+patch_len]                     │
│      target = df.close[start+patch_len+horizon-1]          │
│    → patches (B, C, L), targets (B, 1), cond (B, cond_dim) │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 3. Temporal Split Standardization (NO Look-Ahead)           │
│    _standardize_train_val(patches, val_frac)               │
│    ├─ Split: train_size = n × (1 - val_frac)               │
│    ├─ Compute stats ONLY on train:                         │
│    │    mu = mean(patches_train, axis=(0, 2))              │
│    │    sigma = std(patches_train, axis=(0, 2))            │
│    ├─ Normalize:                                            │
│    │    train_norm = (train - mu) / sigma                  │
│    │    val_norm = (val - mu) / sigma                      │
│    ├─ KS test verification (p < 0.5 expected)              │
│    └─ Save to model.dataset_stats                          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 4. DataLoader Creation                                       │
│    train_ds = CandlePatchDataset(train_norm, train_y)      │
│    val_ds = CandlePatchDataset(val_norm, val_y)            │
│    train_loader = DataLoader(train_ds, batch_size, ...)    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 5. Model Initialization                                      │
│    model = ForexDiffusionLit(config)                       │
│    ├─ VAE: Encoder/Decoder (Conv1D blocks)                 │
│    │   Encoder: (B,C,L) → (B,z_dim) via convs + flatten    │
│    │   Decoder: (B,z_dim) → (B,C,L) via linear + upconvs   │
│    ├─ DiffusionModel: v-prediction MLP                     │
│    │   Input: z_t + t_emb + cond → v (velocity)            │
│    ├─ Scheduler: Cosine noise schedule (T=1000)            │
│    └─ Losses: L_v + L_crps + L_kl                          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 6. PyTorch Lightning Training                                │
│    trainer = pl.Trainer(max_epochs, callbacks, ...)        │
│    trainer.fit(model, train_loader, val_loader)            │
│                                                              │
│    Training Loop (ForexDiffusionLit.training_step):        │
│      1. Forward pass:                                       │
│         x_hat, mu, logvar, z = vae(x)                      │
│      2. Sample diffusion timestep t ~ U(0, T)              │
│      3. Add noise to z: z_t = sqrt(a_t)*z_0 + ...          │
│      4. Predict velocity: v_pred = diffusion(z_t, t, cond) │
│      5. Compute losses:                                     │
│         L_recon = MSE(x_hat, x)                            │
│         L_kl = KL(q(z|x) || p(z))                          │
│         L_v = MSE(v_pred, v_true)                          │
│         L_crps = CRPS(decoded_samples, y_target)           │
│      6. Combined: L = λ_v*L_v + λ_crps*L_crps + λ_kl*L_kl │
│      7. Backprop and optimize                              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 7. Checkpoint Saving                                         │
│    ModelCheckpoint callback saves:                          │
│      - Best model (lowest val/loss)                         │
│      - Last checkpoint                                      │
│      - Top-k checkpoints                                    │
│    Sidecar metadata JSON:                                   │
│      - symbol, timeframe, horizon_bars                      │
│      - patch_len, channel_order                             │
│      - mu, sigma (for inference denormalization)            │
│      - indicator_tfs, warmup_bars                           │
└─────────────────────────────────────────────────────────────┘
```

**Key Components**:
- `CandlePatchDataset`: PyTorch Dataset for patches
- `ForexDiffusionLit`: Lightning module with VAE + Diffusion
- `VAE`: 1D Conv encoder/decoder with reparameterization
- `DiffusionModel`: MLP for v-prediction
- `cosine_alphas()`: Cosine noise schedule
- `crps_sample_estimator()`: CRPS loss for probabilistic evaluation

---

### Workflow 3: SSSD (Structured State Space Diffusion)

**Script**: `train_sssd.py` + `models/sssd.py`

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Configuration Loading                                     │
│    config = load_sssd_config(config_path)                  │
│    - Multi-timeframe config (5m, 15m, 1h, 4h, 1d)          │
│    - S4 layer config (state_dim, n_layers)                 │
│    - Diffusion config (steps_train, steps_inference)       │
│    - Horizon config (minutes list, weights)                │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 2. Data Module Setup                                         │
│    data_module = SSSDDataModule(data_path, config)         │
│    ├─ Load pre-computed features from unified pipeline     │
│    ├─ Create datasets per timeframe                        │
│    └─ DataLoaders with collate_fn                          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 3. Model Initialization                                      │
│    model = SSSDModel(config)                                │
│    ├─ MultiScaleEncoder (S4 layers per timeframe)          │
│    │   S4Layer: Structured state space model                │
│    │   Cross-TF attention for context integration           │
│    ├─ HorizonEmbeddings: Learnable per horizon             │
│    ├─ DiffusionHead: MLP noise predictor                   │
│    │   Input: latent_t + t_emb + context + horizon_emb     │
│    └─ CosineNoiseScheduler: Diffusion process              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 4. Training Loop                                             │
│    trainer = SSSDTrainer(config)                            │
│    trainer.train(data_module)                               │
│                                                              │
│    For each epoch:                                          │
│      For each batch:                                        │
│        1. Multi-scale encoding:                             │
│           features_per_tf = {                               │
│             "5m": (B, seq_len, feat_dim),                   │
│             "15m": (B, seq_len, feat_dim), ...              │
│           }                                                  │
│           context = encoder(features_per_tf)  # (B, ctx_dim)│
│        2. For each horizon in batch:                        │
│           - Get horizon embedding                           │
│           - Sample diffusion timestep t                     │
│           - Add noise to target                             │
│           - Predict noise: diffusion_head(noisy, t, ctx+h) │
│           - Compute MSE loss                                │
│        3. Weighted loss across horizons                     │
│        4. Gradient clipping + optimizer step                │
│        5. LR scheduler step (CosineAnnealing)               │
│      Validation every epoch                                 │
│      Checkpoint if best val loss                            │
│      Early stopping if no improvement                       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 5. Checkpoint Saving                                         │
│    SSSDModel.save_checkpoint(path, epoch, optimizer, metrics)│
│    - Model state_dict                                       │
│    - Config (serialized)                                    │
│    - Optimizer state                                        │
│    - Training metrics                                       │
│    - Epoch number                                           │
└─────────────────────────────────────────────────────────────┘
```

**Key Features**:
- **S4 Layers**: Structured state space models for long-range dependencies
- **Multi-Scale Context**: Combines 5 timeframes (5m to 1d)
- **Multi-Horizon**: Predicts 1m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d simultaneously
- **Consistency Loss**: Penalizes contradictory predictions across horizons
- **Mixed Precision**: AMP for 2x speedup

---

## Model Types & Capabilities

### 1. Traditional ML Models

| Model | Algorithm | Pros | Cons | Use Case |
|-------|-----------|------|------|----------|
| **Ridge** | L2 regularization | Fast, interpretable, stable | Linear only | Baseline, feature importance |
| **Lasso** | L1 regularization | Feature selection, sparse | Can be unstable | High-dim features |
| **ElasticNet** | L1 + L2 | Balanced regularization | Extra hyperparameter | General purpose |
| **RandomForest** | Ensemble trees | Non-linear, robust | Overfits easily, slow | Complex patterns |

**Hyperparameters**:
- `alpha`: Regularization strength (Ridge/Lasso/ElasticNet)
- `l1_ratio`: L1/L2 balance (ElasticNet)
- `n_estimators`: Number of trees (RF)
- `max_depth`: Tree depth (RF)
- `pca_components`: Dimensionality reduction

### 2. VAE + Diffusion

**Architecture**:
- **VAE Encoder**: (B, C, L) → (B, z_dim) via 1D convolutions
- **VAE Decoder**: (B, z_dim) → (B, C, L) via transposed convolutions
- **Diffusion Model**: Predicts velocity v for denoising

**Capabilities**:
- Latent representation learning
- Probabilistic forecasting (sample N trajectories)
- CRPS-based evaluation
- Multi-modal predictions

**Hyperparameters**:
- `patch_len`: Input sequence length (default: 64)
- `z_dim`: Latent dimensionality (default: 128)
- `T`: Diffusion steps (default: 1000)
- `lambda_v`, `lambda_crps`, `lambda_kl`: Loss weights

### 3. SSSD (Experimental)

**Architecture**:
- **Multi-Scale Encoder**: S4 layers for 5m, 15m, 1h, 4h, 1d
- **Cross-Timeframe Attention**: Integrates information across timeframes
- **Diffusion Head**: Noise predictor conditioned on context + horizon

**Capabilities**:
- Multi-horizon predictions (10 horizons)
- Long-range dependencies via S4
- Consistency enforcement across horizons

**Hyperparameters**:
- `s4_state_dim`: S4 hidden state size (default: 64)
- `s4_layers`: Number of S4 layers per timeframe (default: 4)
- `diffusion_steps_train`: Training timesteps (default: 1000)
- `diffusion_steps_inference`: Inference timesteps (default: 50)

---

## Feature Engineering Pipeline

### Feature Categories

#### 1. Relative OHLC Features
```python
rel_open = log(open / prev_close)
rel_high = log(high / open)
rel_low = log(low / open)
rel_close = log(close / open)
```

**Purpose**: Normalize price movements, remove absolute price dependency

#### 2. Temporal Features
```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
dow_sin = sin(2π × day_of_week / 7)
dow_cos = cos(2π × day_of_week / 7)
```

**Purpose**: Capture cyclical patterns (daily/weekly)

#### 3. Realized Volatility
```python
log_returns = log(close / close.shift(1))
rv = sqrt(sum(log_returns^2, window))
```

**Purpose**: Quantify recent volatility for risk adjustment

#### 4. Multi-Timeframe Indicators

**Process**:
1. For each selected (indicator, timeframe) pair:
2. Fetch or resample data to target timeframe
3. Compute indicator (e.g., RSI_15m_14)
4. Merge back to base timeframe (forward-fill)

**Supported Indicators** (18 total):
- **Volatility**: ATR, Bollinger Bands, Keltner Channels
- **Momentum**: RSI, Stochastic, CCI, Williams%R, MFI
- **Trend**: MACD, ADX, EMA, SMA, TRIX
- **Volume**: OBV, VWAP
- **Channels**: Donchian, Keltner
- **Advanced**: Hurst Exponent, Ultimate Oscillator

**Example Configuration** (indicator_tfs):
```json
{
  "atr": ["1m", "5m", "15m"],
  "rsi": ["1m", "5m", "15m", "1h"],
  "bollinger": ["5m", "15m"],
  "macd": ["15m", "1h"]
}
```

**Result**: Each combination creates a feature column
- `atr_1m_14`, `atr_5m_14`, `atr_15m_14`
- `rsi_1m_14`, `rsi_5m_14`, `rsi_15m_14`, `rsi_1h_14`
- `bb_m_5m_20_2.0`, `bb_h_5m_20_2.0`, `bb_l_5m_20_2.0`, ...

---

## Data Standardization

### Critical: No Look-Ahead Bias

**Problem**: If standardization uses statistics from the entire dataset (including validation), the model indirectly sees future data during training.

**Solution**: Compute mean/std **ONLY on training set**, then apply to validation.

**Implementation** (`_standardize_train_val()`):

```python
def _standardize_train_val(X, y, val_frac):
    n = len(X)
    val_size = int(n * val_frac)
    train_size = n - val_size
    
    # Temporal split (NO shuffling)
    X_train = X[:train_size]
    X_val = X[train_size:]
    
    # Compute stats ONLY on training set
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma[sigma == 0] = 1.0  # Avoid division by zero
    
    # Apply to both
    X_train_norm = (X_train - mu) / sigma
    X_val_norm = (X_val - mu) / sigma
    
    # Verification: KS test (distributions should differ)
    # p-value < 0.5 expected for temporal data
    # p-value > 0.8 WARNING: potential look-ahead bias
    
    return X_train_norm, X_val_norm, mu, sigma
```

**Verification**:
- Kolmogorov-Smirnov test compares train/val distributions
- Different time periods → different distributions → low p-value
- If p > 0.8: distributions too similar → potential bias

**For Inference**:
- Load saved `(mu, sigma)` from metadata
- Standardize new data: `X_new_norm = (X_new - mu) / sigma`

---

## Optimization Systems

### 1. Grid Search (Training Orchestrator)

**Path**: `training/training_pipeline/training_orchestrator.py`

**Workflow**:
```python
# Define parameter grid
grid_params = {
    "symbol": ["EUR/USD", "GBP/USD"],
    "timeframe": ["1m", "5m"],
    "horizon": [30, 60, 120],
    "algo": ["ridge", "rf"],
    "alpha": [0.001, 0.01, 0.1]
}

# Create queue
orchestrator = TrainingOrchestrator()
queue_id = orchestrator.create_training_queue(grid_params)

# Train all combinations
results = orchestrator.train_models_grid(queue_id)
```

**Features**:
- **Resume/Idempotency**: Skips already-trained configs (by hash)
- **Parallel Execution**: 4-32 workers (configurable)
- **Automatic Backtesting**: Each model tested with all inference configs
- **Regime-Aware Selection**: Keeps models that improve at least 1 regime
- **Auto-Cleanup**: Deletes non-best models

### 2. Genetic Algorithm (Single-Objective)

**Path**: `training/optimization/genetic_algorithm.py`

**Workflow**:
```python
# Define parameter space
param_space = {
    "alpha": (0.0001, 1.0, "log"),
    "l1_ratio": (0.0, 1.0, "linear"),
    "n_estimators": (50, 500, "int"),
    "max_depth": (3, 20, "int")
}

# Run GA
ga = GeneticAlgorithm(
    param_space=param_space,
    population_size=50,
    generations=100,
    objective="maximize_r2"
)

best_params, best_score = ga.optimize()
```

**Genetic Operators**:
- **Selection**: Tournament (size=3)
- **Crossover**: Blend (alpha=0.5) or uniform
- **Mutation**: Gaussian (sigma=0.1) or uniform

**Objective Functions**:
- Maximize R²
- Minimize MAE
- Maximize Sharpe ratio (if backtest enabled)

### 3. NSGA-II (Multi-Objective)

**Path**: `training/optimization/multi_objective.py`

**Objectives**:
- Minimize: -R² (maximize accuracy)
- Minimize: MAE (minimize error)
- Minimize: Max Drawdown (minimize risk)

**Output**: Pareto frontier of non-dominated solutions

**Workflow**:
```python
evaluator = MultiObjectiveEvaluator()
pareto_optimizer = ParetoOptimizer()

# Run NSGA-II
pareto_front = pareto_optimizer.optimize_nsga2(
    param_space=param_space,
    evaluator=evaluator,
    pop_size=100,
    generations=200
)

# Get knee point (best compromise)
knee_solution = pareto_optimizer.get_knee_point(pareto_front)
```

### 4. Optuna (Bayesian Optimization)

**Status**: Configured but not actively used

**Potential Use**:
```python
import optuna

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.0001, 1.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    
    return mae  # Minimize

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

---

## Inference Pipeline

### Inference Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Model + Metadata                                     │
│    model = joblib.load(model_path)                          │
│    metadata = json.load(metadata_path)                      │
│    mu, sigma = metadata["mu"], metadata["sigma"]            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 2. Fetch Recent Data                                         │
│    candles = fetch_candles_from_db(                         │
│        symbol, timeframe,                                    │
│        days_history=metadata["days_history"]                │
│    )                                                         │
│    # Need enough bars for warmup + indicators               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 3. Feature Engineering (Same as Training)                    │
│    X_new = _build_features(                                 │
│        candles,                                              │
│        indicator_tfs=metadata["indicator_tfs"],             │
│        warmup=metadata["warmup_bars"]                       │
│    )                                                         │
│    # Use same indicator config as training                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 4. Standardization (Use Saved Scaler)                       │
│    X_new_norm = (X_new - mu) / sigma                        │
│    # CRITICAL: Use training set stats, NOT recompute        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 5. Optional PCA Transform                                    │
│    if metadata["pca_model"]:                                │
│        pca = metadata["pca_model"]                          │
│        X_new_norm = pca.transform(X_new_norm)               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 6. Prediction                                                │
│    # For traditional ML:                                     │
│    y_pred = model.predict(X_new_norm[-1:])  # Last row     │
│                                                              │
│    # For diffusion models:                                   │
│    samples = model.sample(X_new_norm[-1:], N=100)          │
│    quantiles = np.percentile(samples, [5, 50, 95])         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ 7. Post-Processing                                           │
│    # Convert from normalized to actual price                │
│    if metadata["use_relative_ohlc"]:                        │
│        # y_pred is relative change                          │
│        pred_price = last_close * exp(y_pred)                │
│    else:                                                     │
│        pred_price = y_pred                                  │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Horizon Inference

**Path**: `inference/service.py` + `utils/horizon_converter.py`

**Workflow**:
```python
# Load multi-horizon model
service = InferenceService(model_dir)

# Request predictions for multiple horizons
horizons = ["1m", "5m", "15m", "30m", "1h"]
predictions = service.predict_multi_horizon(
    symbol="EUR/USD",
    timeframe="1m",
    horizons=horizons,
    mode="ensemble",  # mean/median/voting
    quantiles=[0.05, 0.5, 0.95]
)

# Result structure:
{
    "1m": {"mean": 1.0805, "q05": 1.0798, "q50": 1.0805, "q95": 1.0812},
    "5m": {"mean": 1.0810, "q05": 1.0795, "q50": 1.0810, "q95": 1.0825},
    ...
}
```

**Ensemble Methods**:
- **Mean**: Average of all model predictions
- **Median**: Robust to outliers
- **Voting**: Classification-based (up/down/neutral)

---

## Backtesting & Validation

### Walk-Forward Validation

**Path**: `validation/walk_forward.py`

**Workflow**:
```python
validator = WalkForwardValidator(
    window_months=6,
    purge_days=1,
    embargo_days=2
)

results = validator.validate(
    model=model,
    X=features,
    y=targets,
    n_splits=10
)

# Results: metrics per fold
# - In-sample metrics
# - Out-of-sample metrics
# - Degradation analysis
```

**Split Strategy**:
```
|----Train----|--Purge--|--Embargo--|----Test----|
|    6 months |  1 day  |  2 days   |  1 month   |
```

**Purpose**:
- **Purge**: Remove overlap between train/test
- **Embargo**: Prevent information leakage from correlated data

### Combinatorial Purged CV

**Path**: `validation/walk_forward.py` → `CombinatorialPurgedCV`

**Workflow**:
```python
cv = CombinatorialPurgedCV(
    n_splits=5,
    n_test_splits=2,
    purge_days=1,
    embargo_days=2
)

# Generates all combinations of test splits
# Purges overlapping data
# Applies embargo
```

### Probabilistic Metrics

**Path**: `backtesting/probabilistic_metrics.py`

**CRPS** (Continuous Ranked Probability Score):
```python
crps = crps_sample_np(samples, y)
# Lower is better
# Measures calibration of probabilistic forecasts
```

**PIT-KS** (Probability Integral Transform - Kolmogorov-Smirnov):
```python
pit_values, ks_stat, ks_pvalue = pit_ks_np(samples, y)
# pit_values should be uniform [0, 1] if well-calibrated
# ks_pvalue > 0.05: well-calibrated
```

---

## Complete Parameter Reference

### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | str | Required | Trading pair (e.g., "EUR/USD") |
| `timeframe` | str | Required | Base timeframe (1m, 5m, 15m, 1h, 4h, 1d) |
| `days_history` | int | 90 | Days of historical data |
| `warmup_bars` | int | 64 | Initial bars to discard for indicator warmup |
| `val_frac` | float | 0.2 | Validation set fraction (0.0-1.0) |

### Model Parameters (Traditional ML)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algo` | str | "ridge" | Model type: ridge, lasso, elasticnet, rf |
| `alpha` | float | 0.001 | Regularization strength (Ridge/Lasso/EN) |
| `l1_ratio` | float | 0.5 | L1/L2 ratio (ElasticNet only) |
| `n_estimators` | int | 400 | Number of trees (RandomForest) |
| `max_depth` | int | None | Max tree depth (RandomForest) |
| `pca_components` | int | 0 | PCA dimensions (0=disabled) |

### Model Parameters (Deep Learning)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patch_len` | int | 64 | Input sequence length |
| `z_dim` | int | 128 | Latent dimensionality |
| `hidden_channels` | int | 256 | VAE encoder channels |
| `n_down` | int | 6 | VAE encoder depth |
| `T` | int | 1000 | Diffusion timesteps |
| `epochs` | int | 30 | Training epochs |
| `batch_size` | int | 64 | Batch size |
| `learning_rate` | float | 0.001 | Optimizer LR |

### Feature Engineering Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atr_n` | int | 14 | ATR period |
| `rsi_n` | int | 14 | RSI period |
| `bb_n` | int | 20 | Bollinger Bands period |
| `hurst_window` | int | 64 | Hurst exponent window |
| `rv_window` | int | 60 | Realized volatility window |
| `min_feature_coverage` | float | 0.15 | Min non-NaN fraction (0.0-1.0) |
| `indicator_tfs` | dict | {} | Multi-TF config (JSON) |

### Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimization_method` | str | "none" | none, genetic-basic, nsga2, optuna |
| `population_size` | int | 50 | GA population |
| `generations` | int | 100 | GA generations |
| `mutation_rate` | float | 0.1 | GA mutation probability |
| `crossover_rate` | float | 0.8 | GA crossover probability |

### NVIDIA Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_nvidia_opts` | bool | False | Enable all NVIDIA optimizations |
| `use_amp` | bool | False | Automatic Mixed Precision |
| `precision` | str | "fp16" | fp16, bf16, fp32 |
| `compile_model` | bool | False | torch.compile (PyTorch 2.0+) |
| `use_fused_optimizer` | bool | False | APEX fused optimizer |
| `use_flash_attention` | bool | False | Flash Attention 2 (Ampere+ GPU) |
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation |

---

## Workflow Diagrams

### Training System Decision Tree

```
User initiates training
         │
         ├─ From GUI?
         │  ├─ Yes → TrainingTab → TrainingController
         │  │         → inproc.py (train_sklearn_inproc)
         │  │         → Traditional ML only
         │  │
         │  └─ No → From CLI?
         │           ├─ Traditional ML → train_sklearn.py
         │           ├─ VAE+Diffusion → train.py
         │           ├─ SSSD → train_sssd.py
         │           └─ Optimized → train_optimized.py
         │
         └─ Grid training?
            └─ Yes → TrainingOrchestrator
                     → Database queue
                     → train_single_config() loop
                     → Internal loop (backtesting)
```

### Feature Engineering Flow

```
Raw OHLCV Data
    │
    ├─→ Relative OHLC
    │   (log(price / prev_close))
    │
    ├─→ Temporal Features
    │   (hour_sin, hour_cos, dow_sin, dow_cos)
    │
    ├─→ Realized Volatility
    │   (rolling std of log returns)
    │
    └─→ Multi-TF Indicators
        │
        For each (indicator, timeframe):
            │
            ├─ Fetch timeframe data (DB or resample)
            ├─ Compute indicator (ATR, RSI, MACD, etc.)
            └─ Merge back to base timeframe (forward-fill)
        │
        Result: X (features matrix)
```

### Inference Decision Flow

```
Load model + metadata
    │
    ├─ Model type?
    │  ├─ Traditional ML (joblib)
    │  │  └─ Single point prediction
    │  │
    │  ├─ VAE+Diffusion (PyTorch Lightning)
    │  │  └─ Sample N trajectories → quantiles
    │  │
    │  └─ SSSD (PyTorch)
    │     └─ Multi-horizon predictions
    │
    └─ Apply standardization (use saved mu, sigma)
       │
       └─ Return prediction(s)
```

---

## Conclusion

This document provides a **complete reference** for the ForexGPT Generative Forecast System, covering all workflows, parameters, and implementation details. For issues, optimizations, and code improvements, refer to the companion document: `/SPECS/1_Generative_Forecast.txt`.

**Next Steps**:
1. Review `/SPECS/1_Generative_Forecast.txt` for identified issues
2. Implement recommended fixes (consolidate duplicated code, refactor imports)
3. Add unit tests for standardization (verify no look-ahead bias)
4. Document which training script to use for which use case
5. Create feature engineering utility module (eliminate duplication)

---

**Document Version**: 1.0  
**Generated**: 2025-10-13  
**Maintainer**: ForexGPT Development Team
