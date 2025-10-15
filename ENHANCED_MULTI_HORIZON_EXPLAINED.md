# Enhanced Multi-Horizon System - Features Removed

## Cosa era l'Enhanced Multi-Horizon System?

Sistema avanzato per convertire previsioni single-horizon in multi-horizon usando:
- **Regime detection** (trending vs mean-reverting)
- **Smart adaptive scaling** basato su volatilità
- **Trading scenario support** (conservative, aggressive, balanced)

## Features Rimosse (erano in `_local_infer()`, linee 749-809)

### 1. **Regime Detection** ❌
```python
# Analizzava le ultime 100 candele per determinare:
- Trending market: espande la previsione con scaling geometrico
- Mean-reverting: comprime la previsione (ritorno alla media)
- Sideways: mantiene previsione flat

# Esempio:
# Se market è trending UP e modello prevede +0.5% a 30min
# → 1h prediction = +1.0% (compound growth)
# → 3h prediction = +2.2% (geometric scaling)

# Se market è mean-reverting
# → 1h prediction = +0.3% (decay verso media)
# → 3h prediction = +0.1% (forte decay)
```

**Perché era utile**: Modelli allenati su un horizon specifico (es. 30min) NON sanno predire orizzonti diversi. Il regime detection adattava la previsione in base al comportamento del mercato.

### 2. **Smart Adaptive Scaling** ❌
```python
# 3 modalità di scaling:

# A) Conservative (default)
scaling_factors = {
    "5m": 0.8,    # Sconta previsione -20%
    "15m": 0.9,
    "1h": 1.0,    # Mantiene originale
    "4h": 1.05    # Scala +5%
}

# B) Aggressive
scaling_factors = {
    "5m": 1.0,
    "15m": 1.1,
    "1h": 1.2,
    "4h": 1.5     # Scala +50%!
}

# C) Smart_Adaptive (migliore)
# Usa volatilità realizzata + Hurst exponent
vol = realized_volatility(last_100_bars)
hurst = calculate_hurst_exponent(last_100_bars)

if hurst > 0.6:  # Trending
    factor = 1.0 + (horizon_ratio * vol_adjustment)
else:  # Mean-reverting
    factor = 1.0 - (horizon_ratio * 0.3)
```

**Perché era utile**:
- Preveniva previsioni irrealistiche su orizzonti lunghi
- Adattava lo scaling alla volatilità corrente del mercato
- Considerava la persistenza del trend (Hurst exponent)

### 3. **Trading Scenario Support** ❌
```python
# Scenario preconfigurati:

scenarios = {
    "day_trading": {
        "horizons": ["5m", "15m", "30m"],
        "scaling": "aggressive",
        "confidence": 0.8  # Bande strette
    },
    "swing_trading": {
        "horizons": ["1h", "4h", "1d"],
        "scaling": "conservative",
        "confidence": 0.6  # Bande larghe
    },
    "scalping": {
        "horizons": ["1m", "3m", "5m"],
        "scaling": "aggressive",
        "confidence": 0.9  # Altissima confidenza
    }
}
```

**Perché era utile**: Adattava automaticamente parametri di previsione al tuo stile di trading.

### 4. **Volatility-Based Adjustments** ❌
```python
# Aggiustamento bande di confidenza in base a volatilità:

vol_percentile = current_vol / historical_vol_90th

if vol_percentile > 1.5:  # Alta volatilità
    confidence_bands *= 1.8  # Allarga bande
    prediction *= 0.9  # Sconta previsione
elif vol_percentile < 0.5:  # Bassa volatilità
    confidence_bands *= 0.6  # Restringe bande
    prediction *= 1.1  # Aumenta previsione
```

**Perché era utile**:
- Evitava previsioni troppo aggressive in mercati volatili
- Aumentava confidenza in mercati calmi
- Bande dinamiche riflettevano l'incertezza reale

### 5. **Uncertainty Bands with Metadata** ❌
```python
# Per ogni orizzonte, ritornava:
{
    "prediction": 0.0025,  # Previsione principale
    "lower": 0.0018,       # Banda inferiore
    "upper": 0.0032,       # Banda superiore
    "confidence": 0.75,    # Confidenza (0-1)
    "regime": "trending",  # Regime rilevato
    "scaling_mode": "smart_adaptive",  # Modalità usata
    "volatility": 0.015    # Volatilità normalizzata
}
```

**Perché era utile**: UI poteva mostrare colori diversi per regime, spessore linea basato su confidenza, etc.

### 6. **Performance Registry Integration** ❌
```python
# Salvava ogni previsione per tracking futuro:
performance_registry.record_prediction(
    model_name=display_name,
    symbol=sym,
    timeframe=tf,
    horizon=horizon,
    prediction=float(prediction),
    regime=regime,  # ← Metadata Enhanced
    volatility=volatility,  # ← Metadata Enhanced
    confidence=confidence,  # ← Metadata Enhanced
    scaling_mode=scaling_mode,  # ← Metadata Enhanced
    metadata={...}
)
```

**Perché era utile**: Permetteva analisi post-mortem per capire quali regimi/volatilità il modello prevede meglio.

---

## Sistema Attuale (dopo unificazione)

### Cosa rimane ✅
1. **Replicazione semplice**: `np.full(N, base_prediction)` - copia la stessa previsione su tutti gli orizzonti
2. **Ensemble averaging**: Se hai 10 modelli, fa media ponderata delle loro previsioni
3. **Confidence bands**: Bande basate su deviazione standard delle previsioni (se ensemble)
4. **Performance tracking**: Salva previsioni ma SENZA metadata Enhanced (regime, confidence, scaling_mode)

### Cosa manca ❌
1. ❌ Regime detection
2. ❌ Smart scaling basato su volatilità
3. ❌ Trading scenarios
4. ❌ Adaptive confidence bands
5. ❌ Metadata ricchi per performance analysis

---

## Impatto della Rimozione

### Caso 1: Modello allenato per 30min
**Prima (con Enhanced)**:
```
Previsione base: +0.5% a 30min
↓
5min:  +0.1% (scaled down, conservative)
15min: +0.3% (interpolated)
30min: +0.5% (originale)
1h:    +0.9% (scaled up, trending regime)
4h:    +1.8% (compound growth, alta confidenza)
```

**Adesso (senza Enhanced)**:
```
Previsione base: +0.5% a 30min
↓
5min:  +0.5% (replicated, STESSO VALORE!)
15min: +0.5% (replicated)
30min: +0.5% (replicated)
1h:    +0.5% (replicated, TROPPO BASSO!)
4h:    +0.5% (replicated, MOLTO BASSO!)
```

**Problema**: Previsioni multi-horizon sono FLAT - matematicamente incorrette per orizzonti lunghi!

### Caso 2: Alta Volatilità (VIX > 30)
**Prima (con Enhanced)**:
```
Volatilità rilevata: ALTA (90° percentile)
↓
Prediction: +0.8% → scontata a +0.72% (10% discount)
Confidence bands: ±0.3% → allargata a ±0.54% (80% più larga)
Metadata: {regime: "volatile", confidence: 0.4}
```

**Adesso (senza Enhanced)**:
```
Volatilità: IGNORATA
↓
Prediction: +0.8% (nessun discount)
Confidence bands: ±0.2% (banda fissa, troppo stretta!)
Metadata: {} (vuoto)
```

**Problema**: Nessun adattamento alla volatilità - previsioni troppo aggressive in mercati volatili!

---

## Come Ripristinarlo (se necessario)

### Opzione A: Porta Enhanced a `_parallel_infer()`
**File**: `src/forex_diffusion/ui/workers/forecast_worker.py`

Dopo linea 630 (dove estrae predictions), aggiungi:
```python
# Get ensemble predictions
mean_returns = np.array(ensemble_preds["mean"])

# Apply Enhanced Multi-Horizon if enabled
if self.payload.get("use_enhanced_scaling", True) and len(mean_returns) == 1:
    from ...utils.horizon_converter import convert_single_to_multi_horizon

    base_pred = mean_returns[0]
    market_data = df_candles.tail(100) if len(df_candles) >= 100 else df_candles

    multi_horizon_results = convert_single_to_multi_horizon(
        base_prediction=base_pred,
        base_timeframe=tf,
        target_horizons=horizons_time_labels,
        scenario=self.payload.get("trading_scenario"),
        scaling_mode=self.payload.get("scaling_mode", "smart_adaptive"),
        market_data=market_data,
        uncertainty_bands=True
    )

    # Extract scaled predictions
    scaled_preds = []
    for horizon in horizons_time_labels:
        if horizon in multi_horizon_results:
            scaled_preds.append(multi_horizon_results[horizon]["prediction"])

    mean_returns = np.array(scaled_preds)
```

### Opzione B: Crea nuovo checkbox in UI
**File**: `src/forex_diffusion/ui/unified_prediction_settings_dialog.py`

Aggiungi checkbox "Usa Enhanced Multi-Horizon" con tooltip che spiega i benefici.

---

## Raccomandazione

**Se hai solo 1 modello per forecast**: Enhanced Multi-Horizon è CRITICO!

**Se hai ensemble di 10+ modelli**: Enhanced meno critico (la diversità dei modelli copre gli orizzonti)

**Per produzione seria**: Ripristina Enhanced - le previsioni flat sono matematicamente sbagliate!
