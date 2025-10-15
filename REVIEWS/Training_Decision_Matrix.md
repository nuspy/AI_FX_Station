# Training Script Decision Matrix

## Overview
ForexGPT provides three main training scripts, each optimized for different use cases and model types. This guide helps you choose the right script for your needs.

---

## Quick Decision Tree

```
START
  |
  ├─ Need probabilistic/generative forecasts? → train.py (PyTorch Lightning + Diffusion)
  |
  ├─ Need 80+ professional technical indicators? → train_sklearn_btalib.py (BTALib)
  |
  └─ Need custom features + optimization? → train_sklearn.py (Sklearn + Advanced Features)
```

---

## Training Scripts Comparison

| Feature | train.py | train_sklearn.py | train_sklearn_btalib.py |
|---------|----------|------------------|------------------------|
| **Model Type** | Diffusion (PyTorch Lightning) | Sklearn (Ridge/Lasso/ElasticNet/RF) | Sklearn (Ridge/Lasso/ElasticNet/RF) |
| **Primary Use Case** | Probabilistic forecasting | Custom feature engineering | Professional indicator-based trading |
| **Indicators** | Limited (time features) | ~8 basic indicators (RSI, ATR, MACD, etc.) | 80+ professional indicators (bta-lib) |
| **Optimization** | PyTorch optimizers + Lightning | Genetic Algorithm, NSGA-II | None (but easy to add) |
| **Advanced Features** | ✅ AMP, torch.compile, Flash Attention | ✅ PCA, Autoencoder, VAE | ❌ Basic features only |
| **Volume Features** | ❌ | ✅ Volume Profile, VSA, Smart Money | ❌ |
| **Regime Detection** | ❌ | ✅ HMM-based | ❌ |
| **GPU Support** | ✅ Native PyTorch | ✅ For encoders only | ❌ |
| **Training Time** | Hours (depends on epochs) | Minutes | Minutes |
| **Model Size** | Large (neural network) | Small (linear/tree) | Small (linear/tree) |
| **Interpretability** | Low (black box) | Medium | High (indicator-based) |

---

## Detailed Use Cases

### 1. train.py - PyTorch Lightning Diffusion Model

**File**: `src/forex_diffusion/training/train.py`

**When to use**:
- Need **probabilistic predictions** (not just point estimates)
- Want to generate **multiple forecast scenarios**
- Have GPU available for faster training
- Need NVIDIA optimization stack (AMP, torch.compile, Flash Attention)
- Forecasting longer horizons where uncertainty matters

**Strengths**:
- Captures complex non-linear patterns
- Provides uncertainty estimates
- State-of-the-art generative modeling
- PyTorch Lightning training pipeline with callbacks
- NVIDIA optimization stack for 2-3x speedup

**Weaknesses**:
- Longer training time
- More complex to tune
- Requires more data for good performance
- Black box (hard to interpret)

**Example**:
```bash
fx-train \
  --symbol EUR/USD \
  --timeframe 5m \
  --horizon 12 \
  --epochs 30 \
  --batch_size 64 \
  --use_nvidia_opts \
  --artifacts_dir artifacts/diffusion/
```

**Output**: PyTorch Lightning checkpoint + metadata JSON

---

### 2. train_sklearn.py - Sklearn with Advanced Features

**File**: `src/forex_diffusion/training/train_sklearn.py`

**When to use**:
- Need **custom feature engineering**
- Want **volume-based features** (Volume Profile, VSA, Smart Money)
- Need **regime detection** (HMM-based market state classification)
- Want **hyperparameter optimization** (Genetic Algorithm or NSGA-II)
- Need **dimensionality reduction** (PCA, Autoencoder, VAE)
- Fast prototyping with interpretable models

**Strengths**:
- Extensive feature engineering capabilities
- Multiple optimization strategies
- Fast training (minutes)
- Supports neural encoders for feature compression
- Volume analysis features
- Market regime detection
- Easy to interpret feature importance

**Weaknesses**:
- Linear assumptions (except RandomForest)
- Limited to basic indicators
- No built-in multi-timeframe indicator support

**Features Available**:
- **Basic**: Relative OHLC, temporal features, realized volatility
- **Indicators**: RSI, ATR, Bollinger Bands, MACD, Donchian, Keltner, Hurst exponent, EMA
- **Volume**: Volume Profile (VP), Volume Spread Analysis (VSA), Smart Money Detection
- **Regime**: HMM-based market regime classification
- **Encoders**: PCA, Autoencoder, VAE

**Example with optimization**:
```bash
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD \
  --timeframe 5m \
  --horizon 12 \
  --algo rf \
  --use_vsa \
  --use_smart_money \
  --use_regime_detection \
  --optimization genetic-basic \
  --gen 10 \
  --pop 20 \
  --encoder vae \
  --latent_dim 16 \
  --artifacts_dir artifacts/sklearn/
```

**Output**: Joblib pickle with model + preprocessing + metadata JSON

---

### 3. train_sklearn_btalib.py - Professional Indicators

**File**: `src/forex_diffusion/training/train_sklearn_btalib.py`

**When to use**:
- Need **professional technical indicators** (80+ from bta-lib)
- Want **indicator-based trading strategies**
- Need **smart data filtering** (automatically handles missing OHLCV data)
- Want **configuration management** for indicators
- Building traditional technical analysis systems
- Need multi-timeframe indicator analysis

**Strengths**:
- 80+ professional indicators from bta-lib
- Intelligent data requirements filtering
- Multi-timeframe indicator support with caching
- Category-based indicator organization
- Configuration-driven indicator selection
- High interpretability (indicator-based decisions)

**Weaknesses**:
- No advanced features (VSA, Smart Money, Regime Detection)
- No hyperparameter optimization (yet)
- No dimensionality reduction (yet)

**Indicator Categories**:
- **Overlap**: SMA, EMA, DEMA, TEMA, WMA, VWMA, etc.
- **Momentum**: RSI, MFI, STOCH, Williams %R, ROC, etc.
- **Volatility**: ATR, Bollinger Bands, Keltner Channels, Donchian Channels
- **Trend**: MACD, ADX, Aroon, Parabolic SAR, etc.
- **Volume**: OBV, AD, CMF, MFI, VWAP, etc.
- **Price Transform**: Median Price, Typical Price, Weighted Close
- **Statistics**: Beta, Correlation, Variance, Standard Deviation
- **Cycle**: HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE

**Example with indicator configuration**:
```bash
python -m forex_diffusion.training.train_sklearn_btalib \
  --symbol EUR/USD \
  --timeframe 5m \
  --horizon 12 \
  --algo rf \
  --indicators_config configs/indicators_btalib.json \
  --indicator_tfs '{"rsi": ["5m", "15m"], "macd": ["15m", "1h"]}' \
  --artifacts_dir artifacts/btalib/
```

**Output**: Joblib pickle with model + preprocessing + indicator metadata JSON

---

## Configuration Examples

### Indicator Configuration (train_sklearn_btalib.py)

Create `configs/indicators_btalib.json`:
```json
{
  "available_data": ["open", "high", "low", "close", "volume"],
  "indicators": {
    "rsi": {
      "enabled": true,
      "params": {"period": 14},
      "weight": 1.0
    },
    "macd": {
      "enabled": true,
      "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
      "weight": 1.0
    },
    "bbands": {
      "enabled": true,
      "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
      "weight": 1.0
    }
  }
}
```

---

## Performance Benchmarks (Indicative)

| Script | Training Time | Memory | GPU Required | Model Size |
|--------|--------------|--------|--------------|------------|
| train.py | 1-3 hours | 2-4 GB | Recommended | 10-50 MB |
| train_sklearn.py | 2-10 min | 0.5-2 GB | Optional (for encoders) | 1-5 MB |
| train_sklearn_btalib.py | 2-10 min | 0.5-2 GB | No | 1-5 MB |

*Based on 90 days history, 5m timeframe, EUR/USD*

---

## Integration Notes

### All Scripts Support (PROC-001)
- ✅ Train/Val/Test split (60/20/20)
- ✅ KS test for look-ahead bias detection
- ✅ Timeframe caching for 30-50% speedup
- ✅ Comprehensive metadata saving

### Centralized Modules (ISSUE-001)
All scripts now use:
- `data_loader.py`: Unified data fetching
- `feature_utils.py`: Timeframe utilities
- `feature_engineering.py`: Feature computation

---

## Migration Guide

### From train_sklearn.py to train_sklearn_btalib.py
**When**: You want professional indicators instead of basic ones

**Steps**:
1. Create indicator configuration JSON
2. Map basic indicators to bta-lib equivalents:
   - RSI → rsi
   - ATR → atr
   - Bollinger Bands → bbands
   - MACD → macd
3. Test with same data to verify consistency

### From train_sklearn*.py to train.py
**When**: You need probabilistic forecasts or longer horizons

**Steps**:
1. Increase history days (diffusion needs more data)
2. Add GPU if available
3. Tune epochs (start with 30)
4. Enable NVIDIA optimizations for 2-3x speedup

---

## Future Enhancements

### Planned (from SPECS)
- [ ] ISSUE-001b: Consolidate `_indicators()` function across scripts
- [ ] OPT-002: Parallel indicator computation (2-4x speedup)
- [ ] OPT-003: Lazy loading for inference
- [ ] PROC-002: Hyperparameter search GUI
- [ ] Add optimization support to train_sklearn_btalib.py

---

## Support

For issues or questions:
- Check SPECS/1_Generative_Forecast.txt for detailed specifications
- Review REVIEWS/ folder for implementation notes
- GitHub: https://github.com/anthropics/claude-code/issues

---

**Last Updated**: 2025-10-13
**Version**: 1.0
**Status**: ✅ Implemented (ISSUE-003)
