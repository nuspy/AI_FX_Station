# LDM4TS Integration TODO

**Status**: Core implementation COMPLETE ✅ | Integration IN PROGRESS 🚧

---

## ✅ COMPLETATO (Phase 1-5)

### **1. Dependencies & Database**
- [x] pyproject.toml: Added diffusers, transformers, accelerate, safetensors
- [x] Alembic migration 0020: ldm4ts_predictions, ldm4ts_model_metadata, ldm4ts_inference_metrics
- [x] trading_engine_configs: Added LDM4TS columns
- [x] Migration executed: `alembic upgrade head` ✅

### **2. Model Core**
- [x] vision_transforms.py: SEG, GAF, RP encoding
- [x] ldm4ts_vae.py: Stable Diffusion VAE wrapper
- [x] ldm4ts_conditioning.py: Frequency + Text embeddings
- [x] ldm4ts_temporal_fusion.py: Gated fusion module
- [x] ldm4ts.py: Complete model pipeline

### **3. Inference Service**
- [x] ldm4ts_inference.py: Singleton service
- [x] LDM4TSPrediction dataclass
- [x] to_dict() and to_quantiles_format() methods

### **4. Signal Fusion**
- [x] signal_quality_scorer.py: Added SignalSource.LDM4TS_FORECAST
- [x] unified_signal_fusion.py: _collect_ldm4ts_signals() method
- [x] unified_signal_fusion.py: fuse_signals() LDM4TS integration ✅ NEW
- [x] Uncertainty-based signal creation

### **5. Trading Engine (COMPLETE)** ✅ NEW
- [x] TradingConfig: Added LDM4TS settings (disabilitabile)
- [x] AutomatedTradingEngine.__init__: LDM4TS service initialization
- [x] _process_signals_for_symbol(): OHLCV fetching ✅ NEW
- [x] _calculate_position_size_with_uncertainty() method ✅ NEW
- [x] Signal fusion call with ldm4ts_ohlcv ✅ NEW

---

## 🚧 DA COMPLETARE (Phase 6-8)

### **STATUS UPDATE**: Trading Engine Integration COMPLETE ✅

~~**6. Trading Engine - Signal Collection**~~ ✅ DONE (commit: e442722)
~~**7. Trading Engine - Position Sizing**~~ ✅ DONE (commit: e442722)

**Remaining Work:**

### **8. Backtesting Integration**

**File**: `src/forex_diffusion/backtest/engine.py` (or similar)

**Modifiche necessarie**:

```python
def _process_signals_for_symbol(self, symbol: str, market_data: Dict):
    # ... existing code ...
    
    # NEW: Fetch OHLCV for LDM4TS
    ldm4ts_ohlcv = None
    if self.config.use_ldm4ts and self.ldm4ts_service:
        try:
            from ..services.marketdata import MarketDataService
            mkt_service = MarketDataService()
            
            candles_df = mkt_service.get_candles(
                symbol=symbol,
                timeframe='1m',
                limit=100  # Last 100 candles for vision encoding
            )
            
            if candles_df is not None and len(candles_df) >= 100:
                ldm4ts_ohlcv = candles_df[['open', 'high', 'low', 'close', 'volume']].values
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for LDM4TS: {e}")
    
    # Collect signals (pass ldm4ts_ohlcv to signal_fusion)
    if self.signal_fusion:
        fused_signals = self.signal_fusion.fuse_signals(
            pattern_signals=pattern_signals,
            ensemble_predictions=ensemble_predictions,
            orderflow_signals=orderflow_signals,
            correlation_signals=correlation_signals,
            event_signals=event_signals,
            ldm4ts_ohlcv=ldm4ts_ohlcv,  # NEW
            market_data={'symbol': symbol, 'timeframe': '1m'},
            sentiment_score=sentiment_score
        )
```

---

### **7. Trading Engine - Position Sizing**

**Metodo**: `_calculate_position_size_with_uncertainty(signal, account_balance)`

**Aggiungi logica**:

```python
def _calculate_position_size_with_uncertainty(self, signal: FusedSignal, account_balance: float) -> float:
    """Calculate position size with uncertainty adjustment."""
    
    # Base size
    base_size = self.advanced_position_sizer.calculate_position_size(
        account_balance=account_balance,
        entry_price=signal.entry_price,
        stop_loss_price=signal.stop_price,
        symbol=signal.symbol,
        method=self.config.position_sizing_method
    )
    
    # NEW: Uncertainty adjustment for LDM4TS
    if signal.source == SignalSource.LDM4TS_FORECAST and self.config.ldm4ts_position_scaling:
        uncertainty_pct = signal.metadata.get('uncertainty_pct', 0)
        threshold = self.config.ldm4ts_uncertainty_threshold
        
        # Factor: 1.0 (low uncertainty) → 0.0 (high uncertainty)
        uncertainty_factor = max(0, 1.0 - uncertainty_pct / threshold)
        
        adjusted_size = base_size * uncertainty_factor
        
        logger.debug(
            f"LDM4TS position sizing: base={base_size:.2f}, "
            f"uncertainty={uncertainty_pct:.3f}%, factor={uncertainty_factor:.2f}, "
            f"adjusted={adjusted_size:.2f}"
        )
        
        return adjusted_size
    
    return base_size
```

**Chiamare in**: `_execute_signal(signal)`

```python
def _execute_signal(self, signal: FusedSignal):
    # ... existing code ...
    
    # Calculate position size (with uncertainty adjustment)
    position_size = self._calculate_position_size_with_uncertainty(
        signal=signal,
        account_balance=self.account_balance
    )
    
    # ... rest of execution ...
```

---

### **8. Backtesting Integration**

**File**: `src/forex_diffusion/backtest/engine.py` (or similar)

**Modifiche**:

```python
class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        # ... existing ...
        
        # NEW: LDM4TS support
        self.use_ldm4ts = config.use_ldm4ts
        self.ldm4ts_service = None
        
        if self.use_ldm4ts and config.ldm4ts_checkpoint_path:
            from ..inference.ldm4ts_inference import LDM4TSInferenceService
            self.ldm4ts_service = LDM4TSInferenceService.get_instance()
            self.ldm4ts_service.load_model(config.ldm4ts_checkpoint_path)
    
    def run_backtest(self, ...):
        # ... existing logic ...
        
        # NEW: Generate LDM4TS signals during backtest
        if self.use_ldm4ts and self.ldm4ts_service:
            # Fetch historical OHLCV window
            ohlcv_window = historical_data.iloc[i-100:i][['open', 'high', 'low', 'close', 'volume']].values
            
            if len(ohlcv_window) >= 100:
                # Predict
                prediction = self.ldm4ts_service.predict(ohlcv_window, ...)
                
                # Create signals
                ldm4ts_signals = self._create_ldm4ts_signals(prediction, ...)
                
                # Add to signal pool
                all_signals.extend(ldm4ts_signals)
```

---

### **9. E2E Optimization Integration**

**File**: `src/forex_diffusion/optimization/e2e_optimizer.py`

**Parameter Space Extension**:

```python
class E2EParameterSpace:
    def __init__(self):
        # ... existing parameters ...
        
        # NEW: LDM4TS parameters
        self.ldm4ts_params = {
            'use_ldm4ts': [True, False],
            'ldm4ts_uncertainty_threshold': [0.3, 0.4, 0.5, 0.6, 0.7],
            'ldm4ts_min_strength': [0.2, 0.3, 0.4, 0.5],
            'ldm4ts_position_scaling': [True, False],
            'ldm4ts_horizons': [
                [15, 60, 240],
                [15, 60],
                [60, 240],
                [15, 30, 60, 120, 240]
            ]
        }
```

**Objective Function**:

```python
def evaluate_ldm4ts_config(params: Dict) -> Dict[str, float]:
    """Evaluate LDM4TS configuration."""
    
    # Create trading engine with LDM4TS config
    config = TradingConfig(
        symbols=params['symbols'],
        use_ldm4ts=params['use_ldm4ts'],
        ldm4ts_checkpoint_path=params['ldm4ts_checkpoint_path'],
        ldm4ts_uncertainty_threshold=params['ldm4ts_uncertainty_threshold'],
        ldm4ts_min_strength=params['ldm4ts_min_strength'],
        ldm4ts_position_scaling=params['ldm4ts_position_scaling'],
        ldm4ts_horizons=params['ldm4ts_horizons'],
        ...
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest(start_date, end_date)
    
    return {
        'sharpe_ratio': results['sharpe'],
        'total_return': results['return'],
        'max_drawdown': results['max_dd'],
        'win_rate': results['win_rate'],
        'avg_inference_time': results.get('ldm4ts_avg_inference_ms', 0)
    }
```

---

### **10. UI Integration**

**File**: `src/forex_diffusion/ui/prediction_settings_dialog.py`

**Add LDM4TS Section**:

```python
# In PredictionSettingsDialog.__init__()

ldm4ts_group = QGroupBox("LDM4TS (Vision-Enhanced Diffusion)")
ldm4ts_layout = QVBoxLayout()

self.use_ldm4ts_cb = QCheckBox("Enable LDM4TS Forecasts")
self.use_ldm4ts_cb.setToolTip(
    "Vision-enhanced latent diffusion model with uncertainty quantification.\\n"
    "Converts time series to RGB images for better pattern recognition."
)

self.ldm4ts_checkpoint_edit = QLineEdit()
self.ldm4ts_checkpoint_edit.setPlaceholderText("Path to LDM4TS checkpoint...")

ldm4ts_browse_btn = QPushButton("Browse...")
ldm4ts_browse_btn.clicked.connect(self._browse_ldm4ts_checkpoint)

self.ldm4ts_horizons_edit = QLineEdit("15,60,240")
self.ldm4ts_horizons_edit.setToolTip("Forecast horizons in minutes (comma-separated)")

self.ldm4ts_uncertainty_spin = QDoubleSpinBox()
self.ldm4ts_uncertainty_spin.setRange(0.1, 1.0)
self.ldm4ts_uncertainty_spin.setValue(0.5)
self.ldm4ts_uncertainty_spin.setSingleStep(0.1)
self.ldm4ts_uncertainty_spin.setToolTip("Maximum uncertainty % to accept")

self.ldm4ts_samples_spin = QSpinBox()
self.ldm4ts_samples_spin.setRange(10, 100)
self.ldm4ts_samples_spin.setValue(50)
self.ldm4ts_samples_spin.setToolTip("Monte Carlo samples for uncertainty")

ldm4ts_layout.addWidget(self.use_ldm4ts_cb)
ldm4ts_layout.addWidget(QLabel("Checkpoint Path:"))
ldm4ts_layout.addWidget(self.ldm4ts_checkpoint_edit)
ldm4ts_layout.addWidget(ldm4ts_browse_btn)
ldm4ts_layout.addWidget(QLabel("Horizons (minutes):"))
ldm4ts_layout.addWidget(self.ldm4ts_horizons_edit)
ldm4ts_layout.addWidget(QLabel("Max Uncertainty %:"))
ldm4ts_layout.addWidget(self.ldm4ts_uncertainty_spin)
ldm4ts_layout.addWidget(QLabel("MC Samples:"))
ldm4ts_layout.addWidget(self.ldm4ts_samples_spin)

ldm4ts_group.setLayout(ldm4ts_layout)
main_layout.addWidget(ldm4ts_group)
```

---

### **11. ForecastService - Uncertainty Bands**

**File**: `src/forex_diffusion/ui/chart_components/services/forecast_service.py`

**Modify**: `_plot_forecast_overlay(quantiles, source)`

```python
def _plot_forecast_overlay(self, quantiles: dict, source: str = "basic"):
    # ... existing code ...
    
    # NEW: Plot uncertainty bands for LDM4TS
    if "ldm4ts" in quantiles.get("model_name", "").lower():
        if "std" in quantiles:
            mean = quantiles["q50"]
            std = quantiles["std"]
            
            # ±1σ band
            upper_1sigma = [m + s for m, s in zip(mean, std)]
            lower_1sigma = [m - s for m, s in zip(mean, std)]
            
            self.ax_main.fill_between(
                future_ts,
                lower_1sigma,
                upper_1sigma,
                alpha=0.15,
                color=color,
                label=f"{model_name} ±1σ"
            )
            
            # ±2σ band
            upper_2sigma = [m + 2*s for m, s in zip(mean, std)]
            lower_2sigma = [m - 2*s for m, s in zip(mean, std)]
            
            self.ax_main.fill_between(
                future_ts,
                lower_2sigma,
                upper_2sigma,
                alpha=0.08,
                color=color,
                label=f"{model_name} ±2σ"
            )
```

---

### **12. Training Script**

**File**: `src/forex_diffusion/training/train_ldm4ts.py` (NEW)

**TODO**: Creare training script completo con:
- Data loading (historical OHLCV)
- Vision transformation batch processing
- Training loop (VAE frozen, U-Net + Temporal Fusion trainable)
- Validation metrics (MSE, MAE, Directional Accuracy)
- Checkpointing
- TensorBoard logging

**Example outline**:

```python
def train_ldm4ts(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4
):
    # Load data
    train_loader, val_loader = load_forex_data(data_dir, batch_size)
    
    # Create model
    model = LDM4TSModel(horizons=[15, 60, 240], device='cuda')
    
    # Optimizer (only trainable params: U-Net + Temporal Fusion)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_metrics = validate(model, val_loader)
        
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_mse={val_metrics['mse']:.4f}")
        
        # Save checkpoint
        if val_metrics['mse'] < best_mse:
            save_checkpoint(model, output_dir / f"ldm4ts_best.ckpt")
```

---

## 📋 CHECKLIST FINALE

### **Testing**
- [ ] Test vision transforms on EUR/USD 1m data
- [ ] Test inference service (cold start + warm inference)
- [ ] Test signal generation from predictions
- [ ] Test position sizing with uncertainty
- [ ] Backtest vs SSSD baseline (walk-forward)
- [ ] Paper trading (1 week minimum)

### **Performance Benchmarks**
- [ ] Inference time: <200ms p95
- [ ] Memory usage: <2GB GPU
- [ ] Forecast accuracy: >baseline + 15%
- [ ] Uncertainty calibration: PIT uniformity 0.85-0.95

### **Documentation**
- [ ] Update README with LDM4TS instructions
- [ ] Add training guide
- [ ] Add configuration examples
- [ ] Add troubleshooting section

### **Production Readiness**
- [ ] Error handling (graceful degradation)
- [ ] Logging (debug, info, error levels)
- [ ] Monitoring (inference metrics to DB)
- [ ] Config validation
- [ ] Unit tests (model components)
- [ ] Integration tests (end-to-end)

---

## 🚀 QUICK START (After Integration Complete)

### **1. Install Dependencies**
```bash
pip install -e .  # Installs diffusers, transformers, etc.
```

### **2. Run Alembic Migration**
```bash
alembic upgrade head  # Creates ldm4ts_* tables
```

### **3. Train Model (or download checkpoint)**
```bash
python -m forex_diffusion.training.train_ldm4ts \
    --data-dir data/eurusd_1m \
    --output-dir artifacts/ldm4ts \
    --epochs 100
```

### **4. Configure Trading Engine**
```python
config = TradingConfig(
    symbols=['EUR/USD'],
    timeframes=['1m'],
    use_ldm4ts=True,
    ldm4ts_checkpoint_path='artifacts/ldm4ts/ldm4ts_best.ckpt',
    ldm4ts_horizons=[15, 60, 240],
    ldm4ts_uncertainty_threshold=0.5,
    ldm4ts_position_scaling=True
)

engine = AutomatedTradingEngine(config)
engine.start()
```

### **5. Monitor Performance**
```sql
-- Check predictions
SELECT * FROM ldm4ts_predictions 
WHERE symbol = 'EUR/USD' 
ORDER BY forecast_time DESC 
LIMIT 10;

-- Check accuracy
SELECT 
    horizon_minutes,
    AVG(mae) as avg_mae,
    AVG(CAST(direction_correct AS FLOAT)) as directional_accuracy
FROM ldm4ts_predictions
WHERE actual_price IS NOT NULL
GROUP BY horizon_minutes;

-- Check inference metrics
SELECT * FROM ldm4ts_inference_metrics 
ORDER BY metric_time DESC 
LIMIT 5;
```

---

## 📞 SUPPORT

**Issues**: Check logs in `app_debug.log`  
**Performance**: Query `ldm4ts_inference_metrics` table  
**Accuracy**: Query `ldm4ts_predictions` table with actuals

**Status**: 70% COMPLETE | Remaining: 30% (integration + testing)
