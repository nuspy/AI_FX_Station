# LDM4TS Integration Specification

**Vision-Enhanced Time Series Forecasting for ForexGPT Trading Engine**

Reference Paper: [Vision-Enhanced Time Series Forecasting via Latent Diffusion Models](https://arxiv.org/html/2502.14887v1)

---

## 🎯 OBJECTIVE

Integrate LDM4TS (Latent Diffusion Models for Time Series) into ForexGPT to generate **probabilistic multi-horizon forecasts** with **uncertainty quantification** for the automated trading engine.

### Key Benefits:
- ✅ **Vision-enhanced patterns**: Leverage pre-trained vision models for chart pattern recognition
- ✅ **Uncertainty quantification**: Get prediction intervals (not just point estimates)
- ✅ **Multi-horizon native**: Single model for multiple forecast horizons
- ✅ **Improved accuracy**: Paper shows 65.2% MSE reduction vs baseline diffusion models

---

## 🏗️ ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│                     REAL-TIME MARKET DATA                   │
│              (cTrader WebSocket → MarketDataService)        │
└────────────────────┬────────────────────────────────────────┘
                     │ OHLCV Candles
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATABASE: market_data_candles              │
│                    (1.9M+ historical candles)                │
└────────────────────┬────────────────────────────────────────┘
                     │ Query last N candles
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               VISION TRANSFORMS MODULE (NEW)                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ TimeSeriesVisionEncoder                             │  │
│  │  - Segmentation (SEG): Periodic patterns            │  │
│  │  - GAF: Long-range correlations                     │  │
│  │  - Recurrence Plot: Cyclical behaviors              │  │
│  │                                                       │  │
│  │  Input:  [L, 5] OHLCV → Output: [3, 224, 224] RGB   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │ RGB Image
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  LDM4TS MODEL (NEW)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. VAE Encoder (Stable Diffusion)                    │  │
│  │    - Encode RGB → Latent space [B, C, H/8, W/8]     │  │
│  │                                                       │  │
│  │ 2. Cross-Modal Conditioning                          │  │
│  │    - Frequency: FFT embeddings                       │  │
│  │    - Text: Statistical descriptions (CLIP)           │  │
│  │                                                       │  │
│  │ 3. Latent Diffusion Process                          │  │
│  │    - Iterative denoising (T=50 steps)                │  │
│  │    - U-Net with cross-attention                      │  │
│  │                                                       │  │
│  │ 4. VAE Decoder                                        │  │
│  │    - Decode latent → Reconstructed RGB               │  │
│  │                                                       │  │
│  │ 5. Temporal Fusion Module                            │  │
│  │    - Gated fusion: Explicit + Implicit patterns      │  │
│  │    - Projection: RGB → Future time series            │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │ Probabilistic Forecast
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              LDM4TS INFERENCE SERVICE (NEW)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ LDM4TSPrediction:                                     │  │
│  │  - mean: Dict[horizon → mean price]                  │  │
│  │  - std: Dict[horizon → uncertainty]                  │  │
│  │  - q05, q50, q95: Quantiles for each horizon         │  │
│  │  - inference_time_ms: ~80-190ms                      │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │ Prediction Dict
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         UNIFIED SIGNAL FUSION (EXISTING)                    │
│  Integrates:                                                │
│  - LDM4TS forecasts (new)                                   │
│  - Pattern signals (existing)                               │
│  - Order flow (existing)                                    │
│  - Sentiment (existing)                                     │
│  - Regime detection (existing)                              │
└────────────────────┬────────────────────────────────────────┘
                     │ FusedSignal
                     ▼
┌─────────────────────────────────────────────────────────────┐
│        AUTOMATED TRADING ENGINE (EXISTING)                  │
│  - Position sizing (regime-aware)                           │
│  - Risk management (multi-level stops)                      │
│  - Smart execution optimization                             │
│  - Broker integration (cTrader)                             │
└────────────────────┬────────────────────────────────────────┘
                     │ Orders
                     ▼
                  BROKER
```

---

## 📁 FILE STRUCTURE

```
src/forex_diffusion/
├── models/
│   ├── vision_transforms.py           ✅ CREATED (SEG, GAF, RP)
│   ├── ldm4ts.py                       🆕 TO CREATE (main model)
│   ├── ldm4ts_vae.py                   🆕 TO CREATE (VAE wrapper)
│   ├── ldm4ts_conditioning.py          🆕 TO CREATE (frequency + text)
│   ├── ldm4ts_temporal_fusion.py       🆕 TO CREATE (gated fusion)
│   └── sssd.py                         ✅ EXISTING (reuse components)
│
├── inference/
│   ├── ldm4ts_inference.py             🆕 TO CREATE (service)
│   ├── sssd_inference.py               ✅ EXISTING (reference)
│   └── service.py                      🔧 TO MODIFY (add LDM4TS)
│
├── training/
│   ├── train_ldm4ts.py                 🆕 TO CREATE (training script)
│   └── train_sssd.py                   ✅ EXISTING (reference)
│
├── intelligence/
│   └── unified_signal_fusion.py        🔧 TO MODIFY (add LDM4TS source)
│
├── trading/
│   └── automated_trading_engine.py     🔧 TO MODIFY (use LDM4TS forecasts)
│
└── ui/
    ├── prediction_settings_dialog.py   🔧 TO MODIFY (add LDM4TS toggle)
    ├── chart_components/services/
    │   └── forecast_service.py         🔧 TO MODIFY (visualize RGB)
    └── ldm4ts_visualization_widget.py  🆕 TO CREATE (show RGB channels)
```

---

## 🔌 INTEGRATION POINTS

### 1. **Data Pipeline** (EXISTING → LDM4TS)

**From:**
```python
# services/marketdata.py
candles_df = marketdata_service.get_candles(
    symbol="EUR/USD",
    timeframe="1m",
    start_date="2025-01-08",
    end_date="2025-01-08"
)
# → DataFrame: [ts_utc, open, high, low, close, volume]
```

**To:**
```python
# models/vision_transforms.py
from forex_diffusion.models.vision_transforms import ohlcv_to_vision

ohlcv = candles_df[["open", "high", "low", "close", "volume"]].values  # [L, 5]
rgb_image = ohlcv_to_vision(ohlcv, image_size=(224, 224))  # [3, 224, 224]
```

---

### 2. **Inference Service** (NEW)

**API:**
```python
# inference/ldm4ts_inference.py
from forex_diffusion.inference.ldm4ts_inference import LDM4TSInferenceService, LDM4TSPrediction

service = LDM4TSInferenceService(
    checkpoint_path="artifacts/ldm4ts_eurusd_1m.ckpt",
    device="cuda"
)

prediction: LDM4TSPrediction = service.predict(
    ohlcv=candles_df[["open", "high", "low", "close", "volume"]].values,
    horizons=[15, 30, 60, 240],  # Minutes
    num_samples=50  # Monte Carlo samples for uncertainty
)

# prediction.mean = {15: 1.0523, 30: 1.0528, 60: 1.0534, 240: 1.0541}
# prediction.std = {15: 0.0003, 30: 0.0005, 60: 0.0008, 240: 0.0015}
# prediction.q05 = {15: 1.0518, ...}
# prediction.q50 = {15: 1.0523, ...}
# prediction.q95 = {15: 1.0528, ...}
```

**Output Format (compatible with existing pipeline):**
```python
@dataclass
class LDM4TSPrediction:
    asset: str
    timestamp: pd.Timestamp
    horizons: List[int]  # Minutes
    
    # Predictions for each horizon
    mean: Dict[int, float]
    std: Dict[int, float]
    q05: Dict[int, float]
    q50: Dict[int, float]
    q95: Dict[int, float]
    
    # Metadata
    inference_time_ms: float
    model_name: str = "LDM4TS"
    num_samples: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (for JSON serialization)"""
        ...
    
    def to_quantiles_format(self) -> Dict[str, List[float]]:
        """Convert to ForecastService format for chart overlay"""
        return {
            "q05": [self.q05[h] for h in self.horizons],
            "q50": [self.q50[h] for h in self.horizons],
            "q95": [self.q95[h] for h in self.horizons],
            "future_ts": [
                self.timestamp + pd.Timedelta(minutes=h) 
                for h in self.horizons
            ]
        }
```

---

### 3. **Signal Fusion Integration**

**Modify:** `intelligence/unified_signal_fusion.py`

```python
# ADD to SignalSource enum
class SignalSource(Enum):
    # ... existing sources ...
    LDM4TS_FORECAST = "ldm4ts_forecast"  # 🆕 NEW

# ADD to UnifiedSignalFusion._collect_forecast_signals()
def _collect_forecast_signals(self, symbol: str, timeframe: str) -> List[FusedSignal]:
    """Collect forecast signals from all models."""
    signals = []
    
    # ... existing SSSD/ensemble logic ...
    
    # 🆕 NEW: LDM4TS forecasts
    if self.config.use_ldm4ts:
        try:
            from ..inference.ldm4ts_inference import LDM4TSInferenceService
            
            # Get latest candles
            ohlcv = self._get_latest_ohlcv(symbol, timeframe, lookback=100)
            
            # Predict
            ldm4ts_service = LDM4TSInferenceService.get_instance()
            prediction = ldm4ts_service.predict(ohlcv, horizons=[15, 60, 240])
            
            # Convert to FusedSignal
            for horizon in prediction.horizons:
                mean_pred = prediction.mean[horizon]
                uncertainty = prediction.std[horizon]
                
                # Direction: compare mean prediction with current price
                current_price = ohlcv[-1, 3]  # Close
                direction = "bull" if mean_pred > current_price else "bear"
                
                # Strength: normalized price change / uncertainty (Sharpe-like)
                price_change = abs(mean_pred - current_price) / current_price
                strength = min(price_change / uncertainty, 1.0) if uncertainty > 0 else 0.5
                
                signal = FusedSignal(
                    signal_id=f"ldm4ts_{symbol}_{timeframe}_{horizon}m",
                    source=SignalSource.LDM4TS_FORECAST,
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    strength=strength,
                    entry_price=current_price,
                    target_price=mean_pred,
                    stop_price=prediction.q05[horizon] if direction == "bull" else prediction.q95[horizon],
                    timestamp=int(prediction.timestamp.timestamp() * 1000),
                    metadata={
                        "horizon_minutes": horizon,
                        "mean_pred": mean_pred,
                        "uncertainty": uncertainty,
                        "q05": prediction.q05[horizon],
                        "q50": prediction.q50[horizon],
                        "q95": prediction.q95[horizon],
                    }
                )
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"LDM4TS forecast failed: {e}")
    
    return signals
```

---

### 4. **Trading Engine Integration**

**Modify:** `trading/automated_trading_engine.py`

```python
@dataclass
class TradingConfig:
    # ... existing fields ...
    use_ldm4ts: bool = True  # 🆕 NEW: Use LDM4TS forecasts
    ldm4ts_horizons: List[int] = field(default_factory=lambda: [15, 60, 240])
    ldm4ts_uncertainty_threshold: float = 0.5  # Max uncertainty (% of price)
    ldm4ts_min_strength: float = 0.3  # Minimum signal strength

class AutomatedTradingEngine:
    def _process_signals_for_symbol(self, symbol: str, timeframe: str):
        """Process signals and generate trading decisions."""
        
        # 1. Collect all signals (including LDM4TS)
        all_signals = self.signal_fusion.collect_signals(symbol, timeframe)
        
        # 2. Filter by quality and regime
        filtered_signals = self.signal_fusion.filter_and_rank(
            all_signals,
            regime=self.current_regime,
            min_quality=0.6
        )
        
        # 3. 🆕 NEW: Special handling for LDM4TS uncertainty
        ldm4ts_signals = [s for s in filtered_signals if s.source == SignalSource.LDM4TS_FORECAST]
        
        for signal in ldm4ts_signals:
            uncertainty = signal.metadata.get("uncertainty", 0)
            current_price = signal.entry_price
            
            # Skip if uncertainty too high
            if uncertainty / current_price > self.config.ldm4ts_uncertainty_threshold:
                logger.info(f"Skipping LDM4TS signal: uncertainty {uncertainty:.5f} too high")
                filtered_signals.remove(signal)
                continue
            
            # Adjust position size based on uncertainty
            # Higher uncertainty → smaller position
            uncertainty_factor = 1.0 - (uncertainty / current_price) / self.config.ldm4ts_uncertainty_threshold
            signal.metadata["position_size_multiplier"] = uncertainty_factor
        
        # 4. Execute top signals
        for signal in filtered_signals[:self.config.max_positions]:
            self._execute_signal(signal)
```

---

### 5. **UI Integration**

#### **5.1 Prediction Settings Dialog**

**Modify:** `ui/prediction_settings_dialog.py`

```python
class PredictionSettingsDialog(QDialog):
    def _build_model_selection_tab(self):
        # ... existing model selection ...
        
        # 🆕 NEW: LDM4TS section
        ldm4ts_group = QGroupBox("LDM4TS (Vision-Enhanced Diffusion)")
        ldm4ts_layout = QVBoxLayout()
        
        self.use_ldm4ts_cb = QCheckBox("Use LDM4TS Model")
        self.use_ldm4ts_cb.setChecked(True)
        self.use_ldm4ts_cb.setToolTip(
            "Vision-enhanced latent diffusion model with uncertainty quantification.\n"
            "Converts time series to images for better pattern recognition."
        )
        
        self.ldm4ts_path_edit = QLineEdit()
        self.ldm4ts_path_edit.setPlaceholderText("Path to LDM4TS checkpoint...")
        
        self.ldm4ts_horizons_edit = QLineEdit("15,60,240")
        self.ldm4ts_horizons_edit.setToolTip("Forecast horizons in minutes (comma-separated)")
        
        self.ldm4ts_samples_spin = QSpinBox()
        self.ldm4ts_samples_spin.setRange(10, 100)
        self.ldm4ts_samples_spin.setValue(50)
        self.ldm4ts_samples_spin.setToolTip("Number of Monte Carlo samples for uncertainty")
        
        ldm4ts_layout.addWidget(self.use_ldm4ts_cb)
        ldm4ts_layout.addWidget(QLabel("Model Path:"))
        ldm4ts_layout.addWidget(self.ldm4ts_path_edit)
        ldm4ts_layout.addWidget(QLabel("Horizons (minutes):"))
        ldm4ts_layout.addWidget(self.ldm4ts_horizons_edit)
        ldm4ts_layout.addWidget(QLabel("MC Samples:"))
        ldm4ts_layout.addWidget(self.ldm4ts_samples_spin)
        
        ldm4ts_group.setLayout(ldm4ts_layout)
        main_layout.addWidget(ldm4ts_group)
```

#### **5.2 Forecast Visualization**

**Modify:** `ui/chart_components/services/forecast_service.py`

```python
def _plot_forecast_overlay(self, quantiles: dict, source: str = "basic"):
    """Plot quantiles with LDM4TS-specific styling."""
    
    # ... existing plotting logic ...
    
    # 🆕 NEW: If LDM4TS, show uncertainty bands
    if "ldm4ts" in quantiles.get("model_name", "").lower():
        # Plot uncertainty band (std deviation)
        if "std" in quantiles:
            mean = quantiles["q50"]
            std = quantiles["std"]
            upper = [m + s for m, s in zip(mean, std)]
            lower = [m - s for m, s in zip(mean, std)]
            
            # Fill between (lighter shade)
            self.ax_main.fill_between(
                future_ts,
                lower,
                upper,
                alpha=0.1,
                color=color,
                label=f"{model_name} ±1σ"
            )
```

#### **5.3 Vision Preview Widget (NEW)**

**Create:** `ui/ldm4ts_visualization_widget.py`

```python
"""
LDM4TS Visualization Widget

Shows the three RGB channels (SEG, GAF, RP) for debugging and insight.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QImage, QPixmap
import numpy as np

class LDM4TSVisualizationWidget(QWidget):
    """Widget to visualize LDM4TS RGB channels."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout()
        
        self.seg_label = QLabel("Segmentation (R)")
        self.gaf_label = QLabel("GAF (G)")
        self.rp_label = QLabel("Recurrence Plot (B)")
        
        layout.addWidget(QLabel("LDM4TS Vision Transforms:"))
        layout.addWidget(self.seg_label)
        layout.addWidget(self.gaf_label)
        layout.addWidget(self.rp_label)
        
        self.setLayout(layout)
    
    def update_channels(self, rgb_tensor):
        """Update visualization with new RGB tensor [3, H, W]."""
        rgb_np = rgb_tensor.cpu().numpy()
        
        # Convert each channel to QPixmap
        self.seg_label.setPixmap(self._array_to_pixmap(rgb_np[0]))
        self.gaf_label.setPixmap(self._array_to_pixmap(rgb_np[1]))
        self.rp_label.setPixmap(self._array_to_pixmap(rgb_np[2]))
    
    def _array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """Convert 2D numpy array to QPixmap."""
        # Normalize to [0, 255]
        array = ((array - array.min()) / (array.max() - array.min() + 1e-8) * 255).astype(np.uint8)
        
        h, w = array.shape
        qimage = QImage(array.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage).scaled(200, 200)
```

---

## 🧪 TESTING STRATEGY

### **Phase 1: Vision Transforms (COMPLETE)**
- [x] Test SEG, GAF, RP on sample data
- [x] Verify output shapes: [3, 224, 224]
- [x] Check normalization: values in [0, 1]

### **Phase 2: LDM4TS Model**
- [ ] Load pre-trained Stable Diffusion VAE
- [ ] Test frequency conditioning (FFT)
- [ ] Test text conditioning (CLIP)
- [ ] Test denoising loop (50 steps)
- [ ] Test temporal fusion
- [ ] Verify inference time <200ms

### **Phase 3: Inference Service**
- [ ] Test with EUR/USD 1m data (1.9M candles)
- [ ] Compare with SSSD baseline
- [ ] Benchmark: MSE, MAE, CRPS
- [ ] Test uncertainty calibration (PIT uniformity)

### **Phase 4: Trading Integration**
- [ ] Test signal generation from forecasts
- [ ] Test uncertainty-based position sizing
- [ ] Backtest on historical data (walk-forward)
- [ ] Paper trading (1 week)

### **Phase 5: UI Testing**
- [ ] Test settings dialog (model selection)
- [ ] Test forecast overlay (quantiles + uncertainty bands)
- [ ] Test vision preview widget

---

## 📊 EXPECTED METRICS

### **Forecast Accuracy (Paper Benchmarks):**
| Dataset | Baseline MSE | LDM4TS MSE | Improvement |
|---------|--------------|------------|-------------|
| ETTh1   | 0.421       | 0.147      | **65.2%** ↓ |
| ETTh2   | 0.358       | 0.125      | **65.1%** ↓ |
| Weather | 0.196       | 0.152      | **22.4%** ↓ |

### **Inference Time (ETTh1, GPU):**
| Horizon | LDM4TS | TimeGrad | SSSD |
|---------|--------|----------|------|
| H=96    | 76ms   | 870ms    | 418ms |
| H=192   | 80ms   | 1854ms   | 645ms |
| H=336   | 193ms  | 3119ms   | 1054ms |
| H=720   | 192ms  | 6724ms   | 2516ms |

### **ForexGPT Expected:**
- **Forecast Accuracy**: 15-30% improvement over SSSD
- **Uncertainty Quality**: Better calibrated intervals (PIT uniformity >0.9)
- **Trading Performance**: 5-10% increase in Sharpe ratio (uncertainty-aware sizing)
- **Inference Latency**: <200ms per forecast (acceptable for 1m trading)

---

## 🚀 DEPLOYMENT PLAN

### **Stage 1: Development (1 week)**
- Implement LDM4TS model components
- Train on EUR/USD 1m data
- Unit tests + integration tests

### **Stage 2: Validation (3 days)**
- Walk-forward backtest (2023-2024)
- Compare vs SSSD baseline
- Calibrate uncertainty thresholds

### **Stage 3: Paper Trading (1 week)**
- Deploy to paper trading account
- Monitor forecasts vs actuals
- Tune position sizing multipliers

### **Stage 4: Live Trading (gradual)**
- Start with 10% of account
- Monitor for 2 weeks
- Scale to 50% if performance holds
- Full deployment after 1 month

---

## ⚙️ CONFIGURATION

### **Model Config (YAML):**
```yaml
ldm4ts:
  model:
    vision:
      image_size: [224, 224]
      period_detection: auto  # or fixed integer
      normalize: true
    
    vae:
      pretrained: "stabilityai/sd-vae-ft-mse"  # Hugging Face model
      latent_channels: 4
      latent_scale_factor: 0.18215
    
    conditioning:
      use_frequency: true
      use_text: true
      text_encoder: "openai/clip-vit-base-patch32"
    
    diffusion:
      timesteps: 1000
      sampling_steps: 50
      beta_schedule: "cosine"
    
    temporal_fusion:
      hidden_dim: 256
      num_layers: 2
      gating: true
  
  training:
    batch_size: 16
    learning_rate: 1e-4
    num_epochs: 100
    warmup_steps: 1000
    gradient_clip: 1.0
    
  inference:
    num_samples: 50  # Monte Carlo samples
    horizons: [15, 30, 60, 120, 240]  # Minutes
    device: "cuda"
    compile: true  # torch.compile for speed
```

### **Trading Config:**
```yaml
trading:
  use_ldm4ts: true
  ldm4ts_horizons: [15, 60, 240]
  ldm4ts_uncertainty_threshold: 0.5  # % of price
  ldm4ts_min_strength: 0.3
  ldm4ts_position_size_scaling: true  # Scale by uncertainty
```

---

## 📚 DEPENDENCIES (NEW)

```toml
[project.dependencies]
# Existing...
torch = ">=2.0.0"
numpy = ">=1.24.0"
pandas = ">=2.0.0"
loguru = ">=0.7.0"

# NEW for LDM4TS:
diffusers = ">=0.25.0"           # Stable Diffusion VAE
transformers = ">=4.36.0"        # CLIP text encoder
accelerate = ">=0.25.0"          # Efficient inference
scikit-image = ">=0.22.0"        # Image transforms (optional, we use torch)
```

---

## 🎓 TRAINING INSTRUCTIONS

### **Step 1: Prepare Training Data**
```bash
python scripts/prepare_ldm4ts_data.py \
    --symbols EUR/USD,GBP/USD,USD/JPY \
    --timeframe 1m \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --output-dir data/ldm4ts_training
```

### **Step 2: Train Model**
```bash
python -m forex_diffusion.training.train_ldm4ts \
    --config configs/ldm4ts_eurusd_1m.yaml \
    --data-dir data/ldm4ts_training \
    --output-dir artifacts/ldm4ts \
    --gpus 1
```

### **Step 3: Evaluate**
```bash
python scripts/evaluate_ldm4ts.py \
    --checkpoint artifacts/ldm4ts/ldm4ts_eurusd_1m_best.ckpt \
    --test-data data/ldm4ts_training/test.parquet \
    --horizons 15,60,240
```

---

## ✅ ACCEPTANCE CRITERIA

### **Model Performance:**
- [x] Vision transforms generate valid RGB images [0, 1]
- [ ] VAE encode/decode cycle: reconstruction RMSE <0.01
- [ ] Diffusion denoising: converges in <50 steps
- [ ] Temporal fusion: output matches target horizons
- [ ] Inference time: <200ms on GPU

### **Forecast Quality:**
- [ ] MSE: 20% better than SSSD baseline
- [ ] Directional accuracy: >55%
- [ ] Uncertainty calibration: PIT uniformity 0.85-0.95
- [ ] Quantile coverage: 90% interval captures 85-95% of actuals

### **Integration:**
- [ ] Seamless integration with UnifiedSignalFusion
- [ ] Trading engine uses LDM4TS forecasts correctly
- [ ] UI shows forecasts + uncertainty bands
- [ ] Settings dialog allows model selection

### **Production Readiness:**
- [ ] Model checkpoint <500MB
- [ ] Inference latency: p95 <250ms
- [ ] Memory usage: <2GB GPU
- [ ] Error handling: graceful degradation if model unavailable

---

## 🐛 KNOWN LIMITATIONS

1. **Computational Cost**: Requires GPU for real-time inference (~80-190ms)
2. **Model Size**: ~500MB checkpoint (VAE + U-Net + Temporal Fusion)
3. **Training Time**: ~12 hours on single GPU (100 epochs, 1M samples)
4. **Cold Start**: First inference takes ~2s (model loading)
5. **Lookback Window**: Requires 100+ candles for stable vision transforms

---

## 🔮 FUTURE ENHANCEMENTS

1. **Multi-Asset Training**: Train on multiple FX pairs simultaneously
2. **Adaptive Horizons**: Dynamic horizon selection based on regime
3. **Attention Visualization**: Show which parts of image contribute to forecast
4. **Model Distillation**: Compress model to <100MB for faster inference
5. **Online Learning**: Fine-tune on recent data (continual learning)
6. **Ensemble**: Combine LDM4TS + SSSD + ML models

---

## 📞 CONTACT & SUPPORT

- **Paper Authors**: yuxliang@outlook.com
- **Implementation**: ForexGPT Team
- **Issues**: GitHub Issues

---

**STATUS**: Vision Transforms COMPLETE ✅ | Model Implementation IN PROGRESS 🚧

**Next Steps**:
1. Implement LDM4TS model core (VAE + Diffusion + Fusion)
2. Create inference service
3. Integrate with signal fusion
4. Train first model on EUR/USD 1m
5. Backtest and validate
