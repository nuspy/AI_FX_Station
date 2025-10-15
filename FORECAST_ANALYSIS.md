# Forecast Analysis - Why Forecasts Look Similar

## Issue Reported
"I forecast sono sempre uguali. Non potrebbe essere che abbiamo un metodo duplicato e modifichiamo quello sbagliato?"

## Root Causes Found

### 1. **Bug in parallel_inference.py** ✅ FIXED
**Location**: `src/forex_diffusion/inference/parallel_inference.py:323-330`

**Bug**: Linear scaling ancora presente
```python
# OLD (WRONG):
scale_factor = bars / horizon_bars[0]
scaled_preds.append(base_pred * scale_factor)

# NEW (CORRECT):
preds = np.full(len(horizon_bars), base_pred)
```

This was a **duplicate** of the bug we fixed in `forecast_worker.py`, but in a different file that handles parallel inference.

### 2. **Only ONE Model Available** ⚠️ PRIMARY CAUSE
**Finding**: Only 1 model file exists:
```
artifacts/models/EURUSD_1m_d500_h180_rf_vae16.pkl
```

**Impact**:
- When you request multiple forecasts, they ALL use the same model
- Same model → same predictions → forecasts look identical
- Different colors but same trajectory

**Why This Happens**:
- Model was trained with specific parameters (RF, VAE encoder, horizon 180)
- All forecast requests load this same model
- Even parallel inference uses the same model multiple times

## Solution: Train Multiple Models

To get **diverse forecasts**, you need multiple models with:

### Different Algorithms
```bash
# Ridge regression
python -m forex_diffusion.training.train_sklearn --algo ridge --horizon 30 ...

# Random Forest
python -m forex_diffusion.training.train_sklearn --algo rf --horizon 30 ...

# Elastic Net
python -m forex_diffusion.training.train_sklearn --algo elasticnet --horizon 30 ...
```

### Different Horizons
```bash
# Short-term (30 bars = 30 minutes)
--horizon 30

# Medium-term (60 bars = 1 hour)
--horizon 60

# Long-term (180 bars = 3 hours)
--horizon 180
```

### Different Feature Sets
```bash
# Without PCA
--encoder none

# With PCA compression
--encoder pca --latent_dim 10

# With VAE encoder
--encoder vae --latent_dim 16

# With Autoencoder
--encoder autoencoder --latent_dim 20
```

### Different Indicators
```bash
# Minimal indicators
--indicator_tfs '{"atr": ["1m"], "rsi": ["1m"]}'

# Full indicators
--indicator_tfs '{"atr": ["1m","5m","15m"], "rsi": ["1m","5m"], "macd": ["15m","30m"]}'
```

## Expected Results After Training Multiple Models

### Before (Current State):
```
Forecast 1: EURUSD_1m_h180_rf_vae16 → prediction 0.0025
Forecast 2: EURUSD_1m_h180_rf_vae16 → prediction 0.0025 (SAME!)
Forecast 3: EURUSD_1m_h180_rf_vae16 → prediction 0.0025 (SAME!)
```

### After (Multiple Models):
```
Forecast 1: EURUSD_1m_h30_ridge_none → prediction 0.0018
Forecast 2: EURUSD_1m_h60_rf_pca10 → prediction 0.0032
Forecast 3: EURUSD_1m_h180_rf_vae16 → prediction 0.0025
Forecast 4: EURUSD_1m_h90_elasticnet_ae20 → prediction 0.0021
```

Now you'll see **different trajectories** because:
- Different algorithms make different assumptions
- Different horizons predict different timeframes
- Different encoders capture different patterns

## Training Strategy

### Quick Test (3 models, ~10 min total):
```bash
# Model 1: Ridge, short-term
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD --timeframe 1m --horizon 30 --algo ridge \
  --days_history 30 --encoder none

# Model 2: Random Forest, medium-term
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD --timeframe 1m --horizon 60 --algo rf \
  --days_history 30 --encoder pca --latent_dim 10

# Model 3: Elastic Net, long-term
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD --timeframe 1m --horizon 180 --algo elasticnet \
  --days_history 30 --encoder vae --latent_dim 16
```

### Production Ensemble (8-10 models, ~30-40 min):
Train models with:
- 3 different algorithms (ridge, rf, elasticnet)
- 3 different horizons (30, 60, 180)
- 2 different encoders (none, vae)
- Result: 3×3×2 = 18 combinations, pick best 8-10

## Verification

After training multiple models, verify diversity:
```bash
# List all models
ls -lh artifacts/models/

# You should see:
# EURUSD_1m_d30_h30_ridge_none.pkl
# EURUSD_1m_d30_h60_rf_pca10.pkl
# EURUSD_1m_d30_h180_rf_vae16.pkl
# EURUSD_1m_d30_h90_elasticnet_ae20.pkl
# ... etc
```

Then in the UI:
1. Click "Adv Forecast"
2. Select multiple models
3. Check "Combine models" or uncheck to see separate forecasts
4. Each forecast should have different trajectory and color

## Summary

**The forecasts look identical because you're using the same model repeatedly.**

This is NOT a bug - it's expected behavior when only one model exists.

**Solution**: Train multiple models with different configurations to get diverse predictions.

All the scaling bugs have been fixed - now the system is ready for multi-model ensemble forecasting!
