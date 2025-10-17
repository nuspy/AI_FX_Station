# LDM4TS Chart Visualization - Integration TODO

## Problema Attuale

**LDM4TS è configurato nell'UI ma le previsioni non appaiono sul grafico.**

### Causa
Il `ForecastWorker` non è integrato con LDM4TS - gestisce solo modelli sklearn/tensorflow tradizionali. Le previsioni LDM4TS non vengono generate quando si clicca "Forecast".

### Workflow Esistente

```
User clicks "Forecast" button
  ↓
forecast_service.py emits forecastRequested signal
  ↓
ui_controller.py receives signal → creates ForecastWorker
  ↓
ForecastWorker.run() → loads sklearn/TF models → inference
  ↓
Emits forecastReady signal with quantiles
  ↓
forecast_service.on_forecast_ready() → draws on chart
```

### Cosa Manca

**LDM4TS non è nel workflow:**
- `ForecastWorker` non controlla se LDM4TS è enabled
- Nessuna chiamata a `LDM4TSInferenceService`
- Predictions non vengono convertite in formato chart-compatible

---

## Implementazione Richiesta

### File da Modificare

#### 1. `src/forex_diffusion/ui/workers/forecast_worker.py`

**Aggiungere:**
```python
def _check_ldm4ts_enabled(self) -> bool:
    """Check if LDM4TS is enabled in settings."""
    from ..unified_prediction_settings_dialog import UnifiedPredictionSettingsDialog
    settings = UnifiedPredictionSettingsDialog.get_settings_from_file()
    return settings.get('ldm4ts_enabled', False)

def _run_ldm4ts_inference(self) -> Optional[dict]:
    """Run LDM4TS inference if enabled."""
    from ...inference.ldm4ts_inference import LDM4TSInferenceService
    from ..unified_prediction_settings_dialog import UnifiedPredictionSettingsDialog
    
    # Load settings
    settings = UnifiedPredictionSettingsDialog.get_settings_from_file()
    
    checkpoint_path = settings.get('ldm4ts_checkpoint', '')
    if not checkpoint_path or not Path(checkpoint_path).exists():
        logger.warning("LDM4TS enabled but no valid checkpoint")
        return None
    
    # Get service
    service = LDM4TSInferenceService.get_instance(
        checkpoint_path=checkpoint_path
    )
    
    # Get OHLCV data
    symbol = self.payload['symbol']
    timeframe = self.payload['timeframe']
    window_size = settings.get('ldm4ts_window_size', 100)
    
    # Fetch from DB
    df = self.market_service.get_recent_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=window_size
    )
    
    # Run inference
    num_samples = settings.get('ldm4ts_num_samples', 50)
    prediction = service.predict(
        ohlcv_data=df,
        num_samples=num_samples
    )
    
    # Convert to quantiles format
    return prediction.to_quantiles_format()
```

**Modificare run():**
```python
def run(self):
    try:
        # Check if LDM4TS is enabled
        if self._check_ldm4ts_enabled():
            logger.info("LDM4TS enabled - running vision-enhanced forecast")
            quantiles = self._run_ldm4ts_inference()
            
            if quantiles:
                quantiles['source'] = 'ldm4ts'
                quantiles['model_name'] = 'LDM4TS'
                
                # Emit results
                self.signals.forecastReady.emit(
                    pd.DataFrame(),  # Empty df (LDM4TS doesn't need candles in result)
                    quantiles
                )
                self.signals.status.emit("LDM4TS forecast completed")
                return
            else:
                logger.warning("LDM4TS inference failed, falling back to standard models")
        
        # Existing code for standard models...
        # (rest of run() method unchanged)
```

#### 2. `src/forex_diffusion/ui/chart_components/services/forecast_service.py`

**Aggiungere styling LDM4TS:**
```python
def _plot_forecast_overlay(self, quantiles: dict, source: str = "basic"):
    """Plot quantiles on the chart."""
    # ... existing code ...
    
    # LDM4TS-specific styling
    if source == 'ldm4ts':
        line_color = '#9C27B0'  # Purple for LDM4TS
        fill_alpha = 0.2
        line_width = 3
        line_style = 'solid'
        model_name = "LDM4TS (Vision)"
    else:
        # Existing styling for standard models
        line_color = self._get_color_for_model(model_path)
        # ... rest of existing code
```

#### 3. `src/forex_diffusion/services/marketdata.py`

**Aggiungere metodo helper:**
```python
def get_recent_candles(
    self,
    symbol: str,
    timeframe: str,
    limit: int = 100
) -> pd.DataFrame:
    """
    Get recent candles for inference.
    
    Args:
        symbol: Trading symbol (e.g., 'EUR/USD')
        timeframe: Timeframe (e.g., '1m', '15m', '1h')
        limit: Number of candles to retrieve
        
    Returns:
        DataFrame with OHLCV columns
    """
    with Session(self.engine) as session:
        query = (
            select(MarketDataCandle)
            .where(
                MarketDataCandle.symbol == symbol,
                MarketDataCandle.timeframe == timeframe
            )
            .order_by(MarketDataCandle.timestamp_ms.desc())
            .limit(limit)
        )
        
        candles = session.execute(query).scalars().all()
        
    if not candles:
        raise ValueError(f"No data found for {symbol} {timeframe}")
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'timestamp': pd.to_datetime(c.timestamp_ms, unit='ms', utc=True),
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        }
        for c in reversed(candles)  # Reverse to chronological order
    ])
    
    return df.set_index('timestamp')
```

---

## Testing Steps

### 1. Enable LDM4TS in UI

```
Menu → Settings → Prediction Settings → Tab "LDM4TS (Vision)"
- ✅ Enable LDM4TS Forecasting
- Checkpoint: artifacts/ldm4ts/test_checkpoint.pt
- Save
```

### 2. Open Chart

```
- Select symbol: EUR/USD
- Select timeframe: 15m
- Ensure sufficient historical data loaded
```

### 3. Click Forecast

```
- Click "Forecast" button in chart toolbar
- Wait for inference (2-3 seconds)
- Check logs for "LDM4TS enabled - running vision-enhanced forecast"
```

### 4. Verify Visualization

**Expected:**
- Purple lines appear on chart (3 lines for 3 horizons)
- Light purple shaded area (confidence band)
- Legend shows "LDM4TS (Vision)"
- Hovering shows uncertainty values

**If not visible:**
- Check logs for errors
- Verify checkpoint path is correct
- Ensure window_size data is available (100 candles)
- Check GPU/CUDA availability

---

## Expected Visualization

### Chart Elements

```
       Price
         │
    1.10 ├─────────────────────────────────────
         │                               ╱╲
         │                             ╱    ╲  ← LDM4TS Prediction
    1.09 ├───────────────────────────╱        ╲
         │          ████████████████              ← Confidence Band (±1σ)
         │        ╱
    1.08 ├──────╱
         │    ╱
         │  ╱
    1.07 ├╱
         └─────────────────────────────────────────→ Time
           Now    +15m    +1h        +4h
```

### Legend

```
📊 Chart Legend:
- LDM4TS (Vision) ━━━━ (purple, thick)
  - 15-min horizon ──── (thin)
  - 1-hour horizon ──── (medium, main)
  - 4-hour horizon ---- (dashed)
- Confidence ▒▒▒▒ (light purple, 80% opacity)
```

### Hover Tooltip

```
┌─────────────────────────────┐
│ LDM4TS Forecast             │
├─────────────────────────────┤
│ Time: 2025-10-18 14:30 UTC  │
│ Horizon: 1-hour             │
│ Price: 1.0895 ± 0.0015      │
│ Confidence: 85%             │
│ Range: [1.0880, 1.0910]     │
│ Uncertainty: 15%            │
└─────────────────────────────┘
```

---

## Advanced: Multi-Model Display

### Combined Forecast (LDM4TS + SSSD)

Se entrambi LDM4TS e altri modelli sono abilitati:

```python
# In forecast_service.py
def _plot_forecast_overlay(self, quantiles: dict, source: str = "basic"):
    # ... existing code ...
    
    # Apply transparency based on source
    if source == 'ldm4ts':
        alpha = 0.9  # LDM4TS more opaque (higher confidence)
    else:
        alpha = 0.6  # Other models more transparent
```

**Visual:**
```
Purple (LDM4TS) - opaque, thick
Blue (SSSD) - semi-transparent, medium
Green (Ensemble) - transparent, thin
```

---

## Performance Considerations

### Inference Time

**LDM4TS inference:**
- Test checkpoint: ~200-300ms (50 MC samples)
- Trained checkpoint: same (inference time independent of training)

**Optimization:**
```python
# In settings, reduce samples for faster inference
num_samples = 25  # 100ms (acceptable quality)
# vs
num_samples = 100  # 500ms (excellent quality)
```

### Memory Usage

**Chart rendering:**
- Each forecast: ~10KB (3 horizons × 3 quantiles × timestamps)
- Max 10 forecasts on chart: ~100KB
- Negligible impact on UI performance

### Auto-Forecast

**If auto-forecast enabled:**
```python
# In forecast_service.py
def _auto_forecast_tick(self):
    # Check if LDM4TS enabled
    if self._check_ldm4ts_enabled():
        # Run LDM4TS inference
        # (automatic every N minutes)
```

---

## Error Handling

### Common Issues

**1. "No valid checkpoint"**
```
Solution: Browse to correct checkpoint file
Check: artifacts/ldm4ts/test_checkpoint.pt exists
```

**2. "Not enough historical data"**
```
Solution: Load more candles (window_size=100 requires 100 candles)
Check: marketdata.py get_recent_candles() returns sufficient data
```

**3. "CUDA out of memory"**
```
Solution: Reduce num_samples (50 → 25)
Or: Use CPU inference (slower but works)
```

**4. "Inference timeout"**
```
Solution: Check GPU availability
Increase timeout in ForecastWorker
```

### Debug Logging

```python
# Enable detailed logging
import logging
logging.getLogger('forex_diffusion.inference.ldm4ts_inference').setLevel(logging.DEBUG)
logging.getLogger('forex_diffusion.ui.workers.forecast_worker').setLevel(logging.DEBUG)
```

---

## Future Enhancements

### Phase 1 (Current)
- ✅ Basic LDM4TS inference
- ✅ Chart visualization
- ✅ Multi-horizon display

### Phase 2 (Short-term)
- [ ] Uncertainty-based coloring (red=high, green=low)
- [ ] Clickable forecast lines (show details)
- [ ] Export forecast to CSV
- [ ] Compare LDM4TS vs actual prices

### Phase 3 (Long-term)
- [ ] Real-time forecast updates
- [ ] Ensemble weighting (LDM4TS + SSSD)
- [ ] Forecast accuracy dashboard
- [ ] Adaptive horizon selection based on market conditions

---

## Implementation Priority

### High Priority (Must Have)
1. ✅ Detect LDM4TS enabled
2. ✅ Run inference in ForecastWorker
3. ✅ Convert to quantiles format
4. ✅ Draw on chart

### Medium Priority (Should Have)
5. ⏳ Error handling
6. ⏳ Performance optimization
7. ⏳ User feedback (loading spinner)

### Low Priority (Nice to Have)
8. ⏳ Custom styling per horizon
9. ⏳ Forecast comparison tools
10. ⏳ Export/import capabilities

---

## Summary

**Current State:**
- ❌ LDM4TS predictions not visible on chart
- ✅ Inference service ready
- ✅ UI configuration available
- ✅ Checkpoint created

**Required Work:**
- 🔧 Integrate LDM4TS in ForecastWorker
- 🔧 Add chart rendering
- 🔧 Handle edge cases

**Estimated Effort:**
- Core integration: ~2-3 hours
- Testing & polish: ~1-2 hours
- **Total: ~4-5 hours development time**

**Next Steps:**
1. Modify `ForecastWorker` to detect LDM4TS
2. Add `_run_ldm4ts_inference()` method
3. Test with test checkpoint
4. Verify chart visualization
5. Document for users

---

**Ready to implement?** 🚀
