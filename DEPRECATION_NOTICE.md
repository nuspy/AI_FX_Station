# Deprecated Training Scripts

**Date**: 2025-10-13  
**Status**: DEPRECATED - Use consolidated scripts

---

## ⚠️ Deprecated Scripts (DO NOT USE)

The following training scripts are **deprecated** and will be removed in a future release.

### 1. `training/train.py` → DEPRECATED

**Status**: ❌ DEPRECATED  
**Replacement**: Use `training/train_sklearn.py` instead

**Migration**:
```bash
# OLD (deprecated)
python src/forex_diffusion/training/train.py --symbol EUR/USD

# NEW (use this)
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD \
  --timeframe 15m \
  --horizon 4 \
  --algo ridge
```

**Reason**: Overlaps with train_sklearn.py functionality.

---

### 2. `training/train_optimized.py` → DEPRECATED

**Status**: ❌ DEPRECATED  
**Replacement**: Use `training/train_sklearn.py --use-gpu`

**Migration**:
```bash
# OLD (deprecated)
python src/forex_diffusion/training/train_optimized.py --symbol EUR/USD

# NEW (use this)
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD \
  --timeframe 15m \
  --horizon 4 \
  --algo ridge \
  --use-gpu  # GPU optimization included
```

**Reason**: GPU features merged into train_sklearn.py.

---

### 3. `training/optimized_trainer.py` → DEPRECATED

**Status**: ❌ DEPRECATED  
**Replacement**: Use `training/parallel_trainer.py` or `training/train_sklearn.py`

**Migration**:
```bash
# OLD (deprecated)
python src/forex_diffusion/training/optimized_trainer.py

# NEW (use this - for parallel training)
python -m forex_diffusion.training.parallel_trainer \
  --symbols EUR/USD,GBP/USD \
  --timeframes 15m,1h \
  --horizons 4,8 \
  --algos ridge,lasso
```

**Reason**: Parallel training now handled by parallel_trainer.py.

---

## ✅ Active Training Scripts (USE THESE)

### Primary Scripts

#### 1. `train_sklearn.py` - Main Training Script

**Purpose**: Train sklearn-based models (ridge, lasso, random forest, etc.)

**Features**:
- All major algorithms (ridge, lasso, rf, xgboost, lightgbm)
- GPU acceleration support (--use-gpu)
- BTAlib indicators support (--use-btalib)
- Walk-forward validation
- Comprehensive metrics

**Usage**:
```bash
# Basic training
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD \
  --timeframe 15m \
  --horizon 4 \
  --algo ridge \
  --artifacts-dir artifacts/

# With GPU acceleration
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD \
  --timeframe 15m \
  --horizon 4 \
  --algo xgboost \
  --use-gpu

# With BTAlib indicators
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD \
  --timeframe 15m \
  --horizon 4 \
  --algo ridge \
  --use-btalib
```

**Algorithms Supported**:
- `ridge` - Ridge Regression (default, fast, robust)
- `lasso` - Lasso Regression (feature selection)
- `elastic_net` - Elastic Net (ridge + lasso)
- `rf` - Random Forest
- `xgboost` - XGBoost (GPU support)
- `lightgbm` - LightGBM (fast)
- `catboost` - CatBoost

---

#### 2. `train_sssd.py` - Diffusion Model Training

**Purpose**: Train SSSD diffusion models for probabilistic forecasting

**Features**:
- Multi-horizon forecasting (1h, 4h, 1d)
- Probabilistic predictions
- Conditional generation
- Advanced deep learning

**Usage**:
```bash
# Train diffusion model
python -m forex_diffusion.training.train_sssd \
  --symbol EUR/USD \
  --timeframe 15m \
  --horizons 1h,4h,1d \
  --epochs 100 \
  --artifacts-dir artifacts/
```

**When to Use**:
- Need probabilistic forecasts (confidence intervals)
- Multi-horizon predictions
- Advanced deep learning approach
- More complex than sklearn models

---

#### 3. `parallel_trainer.py` - Parallel Training

**Purpose**: Train multiple models simultaneously (8x speedup)

**Features**:
- Parallel execution across CPU cores
- Progress tracking
- Error handling per job
- Automatic retry on failure

**Usage**:
```bash
# Train all combinations in parallel
python -m forex_diffusion.training.parallel_trainer \
  --symbols EUR/USD,GBP/USD,USD/JPY \
  --timeframes 15m,1h,4h \
  --horizons 1,4,8,24 \
  --algos ridge,lasso,rf \
  --max-workers 8
```

**Example Speedup**:
- Sequential: 36 models × 10 min = 360 min (6 hours)
- Parallel (8 cores): 36 / 8 = 45 minutes (8x speedup)

---

#### 4. `auto_retrain.py` - Automated Retraining

**Purpose**: Automatically retrain models on schedule

**Features**:
- Scheduled retraining (daily, weekly, monthly)
- Performance monitoring
- Automatic model updates
- Email notifications

**Usage**:
```bash
# Start auto-retraining daemon
python -m forex_diffusion.training.auto_retrain \
  --config configs/auto_retrain.yaml \
  --check-interval 24h
```

**Config Example** (`configs/auto_retrain.yaml`):
```yaml
symbols:
  - EUR/USD
  - GBP/USD

timeframes:
  - 15m
  - 1h

horizons:
  - 4
  - 8

retrain_schedule: daily
performance_threshold: 0.55  # Retrain if win rate drops below 55%
```

---

## 🔄 Migration Timeline

### Phase 1 (Current - 2 weeks)
- ✅ Create DEPRECATION_NOTICE.md
- ✅ Add deprecation warnings to old scripts
- ⚠️ Both old and new scripts work (backward compatibility)

### Phase 2 (2-4 weeks)
- 📝 Update all documentation to use new scripts
- 📝 Update GUI to use new scripts
- 🔔 Log warnings when deprecated scripts used

### Phase 3 (1-2 months)
- ❌ Remove deprecated scripts
- ✅ Keep only 4 main scripts

---

## 📊 Script Comparison

| Feature | train.py (OLD) | train_sklearn.py (NEW) |
|---------|---------------|----------------------|
| Algorithms | Limited | All sklearn + XGBoost + LightGBM |
| GPU Support | ❌ No | ✅ Yes (--use-gpu) |
| BTAlib | ❌ No | ✅ Yes (--use-btalib) |
| Walk-forward | ❌ No | ✅ Yes |
| Parallel | ❌ No | ✅ Yes (via parallel_trainer.py) |
| Status | ❌ Deprecated | ✅ Active |

| Feature | train_optimized.py (OLD) | train_sklearn.py --use-gpu (NEW) |
|---------|------------------------|--------------------------------|
| GPU Support | ✅ Yes | ✅ Yes |
| Algorithms | Limited | All |
| Maintenance | ❌ No longer updated | ✅ Actively maintained |
| Status | ❌ Deprecated | ✅ Active |

| Feature | optimized_trainer.py (OLD) | parallel_trainer.py (NEW) |
|---------|--------------------------|-------------------------|
| Parallel | ✅ Yes | ✅ Yes (better) |
| Progress | ⚠️ Limited | ✅ Detailed |
| Error Handling | ⚠️ Basic | ✅ Comprehensive |
| Retry Logic | ❌ No | ✅ Yes |
| Status | ❌ Deprecated | ✅ Active |

---

## 🆘 Support

If you encounter issues during migration:

1. Check this document for correct usage
2. Review logs for error messages
3. Open issue on GitHub with:
   - Old command you were using
   - Error message
   - Expected behavior

---

**Last Updated**: 2025-10-13  
**Maintained by**: ForexGPT Development Team
