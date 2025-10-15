# Inference Settings - Guida Completa

## Indice
1. [Overview](#overview)
2. [Model Selection](#model-selection)
3. [Horizons](#horizons)
4. [Core Settings](#core-settings)
5. [Advanced Settings](#advanced-settings)
6. [GPU Acceleration](#gpu-acceleration)
7. [Best Practices](#best-practices)

---

## Overview

### Flusso Inference
```
1. SELECT MODELS
   ↓ Scegli 1+ modelli addestrati
   ↓
2. PREPARE FEATURES
   ↓ Fetch candles, calcola indicators, apply encoder
   ↓
3. PARALLEL INFERENCE
   ↓ Ogni modello predice in parallelo (CPU o GPU)
   ↓
4. ENSEMBLE (opzionale)
   ↓ Combina predizioni → media ponderata + confidence bands
   ↓
5. MULTI-HORIZON CONVERSION
   ↓ Converte returns → prezzi futuri per ogni orizzonte
   ↓
6. DISPLAY
   ↓ Mostra forecast sul grafico
```

---

## Model Selection

### Combinazione Modelli
```
checkbox "Combina modelli (Ensemble)"

✅ Checked (default):
- Combina tutti modelli in UNA forecast
- Media ponderata (weight = 1/execution_time)
- Confidence bands da variance predizioni
- Risultato: 1 linea forecast con bande

❌ Unchecked:
- Genera forecast separato per OGNI modello
- Ogni modello = linea colorata diversa
- Utile per confrontare modelli
- Risultato: N linee forecast
```

**Quando combinare**:
- ✅ Produzione (massima robustezza)
- ✅ Hai 3+ modelli diversi (algo, encoder, horizon)
- ✅ Vuoi ridurre variance

**Quando NON combinare**:
- ✅ Debug (capire quale modello performa)
- ✅ Hai solo 1-2 modelli
- ✅ Modelli troppo diversi (confonde)

### Selezione Modelli

**Strategia 1: Diversità**
```
Modello A: EURUSD_1m_h30_ridge_none.pkl
Modello B: EURUSD_1m_h60_rf_pca10.pkl
Modello C: EURUSD_1m_h180_elasticnet_vae16.pkl

→ 3 algo diversi (ridge, rf, elasticnet)
→ 3 encoder diversi (none, pca, vae)
→ 3 horizon diversi (30, 60, 180)
→ ENSEMBLE POTENTE!
```

**Strategia 2: Specializzazione**
```
Modello A: EURUSD_1m_h30_rf_vae16.pkl  (short-term)
Modello B: EURUSD_1m_h30_rf_vae16.pkl  (stesso, per confirmation)
→ ❌ NO diversità, ensemble inutile
```

**Regola**: Ensemble funziona SE modelli sono **indipendenti** (errori non correlati)

---

## Horizons

### Formato
```
Campo "Horizons": "1m, 5m, 15m, 1h"

Formati supportati:
1. Lista semplice: "1m, 5m, 15m"
2. Range: "1-5m" → 1m, 2m, 3m, 4m, 5m
3. Range con step: "15-30m/5m" → 15m, 20m, 25m, 30m
4. Range tra unità: "30m-2h" → 30m, 1h, 1h30m, 2h
5. Mix: "1-5m, 10m, 1h-3h/30m"
```

### Scelta Strategica

**Scalping (orizzonti brevi)**:
```
Horizons: "1m, 2m, 3m, 5m"

Pro: Massima granularità, entry preciso
Contro: Rumoroso, commissioni pesano
Quando: Trading ad alta frequenza
```

**Day Trading (orizzonti medi)**:
```
Horizons: "5m, 15m, 30m, 1h"

Pro: Buon trade-off signal/noise
Contro: Richiede monitoring
Quando: Intraday trading standard
```

**Swing (orizzonti lunghi)**:
```
Horizons: "1h, 4h, 1d, 3d"

Pro: Pochi falsi segnali, robusto
Contro: Meno opportunità
Quando: Holding posizioni giorni/settimane
```

**Regola**: `max(horizon) <= 2x model_horizon_trained`

Esempio:
```
Modello addestrato con horizon=30 (bars)
TF=1m → horizon=30min

Horizons OK:
"5m, 15m, 30m, 1h" ✅ (max=1h = 2x 30min)

Horizons SBAGLIATO:
"5m, 1h, 4h, 1d" ❌ (max=1d >> 2x 30min)
→ Previsioni a 1d sono extrapolazioni irrealistiche!
```

---

## Core Settings

### N Samples
```
Spinbox: 1 - 10000 (default: 1000)

Cosa fa: Numero sample generati per distribuzione

Valori bassi (100-500):
Pro: Veloce
Contro: Distribuzione grossolana, bande instabili

Valori medi (1000-2000):
Pro: Buon compromesso
Contro: --

Valori alti (5000-10000):
Pro: Distribuzione smooth, bande precise
Contro: Lento (5-10x)
```

**Raccomandazione**:
- Test rapidi: 500
- Produzione: 1000
- Analisi statistica: 5000

### Quantiles
```
Campo: "0.05, 0.50, 0.95"

Cosa fa: Percentili per confidence bands

Valori standard:
0.05, 0.50, 0.95 → 90% confidence interval
0.10, 0.50, 0.90 → 80% confidence interval
0.25, 0.50, 0.75 → 50% confidence interval (IQR)

Quando usare:
90% CI (0.05, 0.95): Produzione (captures most scenarios)
80% CI (0.10, 0.90): Conservative (tighter bands)
50% CI (0.25, 0.75): Very conservative (solo core prediction)
```

**Multi-quantiles**:
```
"0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99"

→ Genera 7 bande:
- q01-q99: 98% CI (estremi)
- q05-q95: 90% CI (standard)
- q25-q75: 50% CI (core)
- q50: mediana

Utile per: Visualizzare incertezza completa
```

### Warmup Bars
```
Spinbox: 10 - 200 (default: 64)

Cosa fa: Scarta prime N barre per stabilizzare indicatori

Formula: warmup >= max(indicator_windows)

Esempio:
RSI(14) + Bollinger(20) + Hurst(64)
→ warmup >= 64

Valori bassi (10-30):
Pro: Massimizza dati
Contro: Indicatori instabili

Valori alti (100-200):
Pro: Indicatori perfetti
Contro: Spreca dati
```

### RV Window
```
Spinbox: 30 - 240 minuti (default: 60)

Cosa fa: Finestra realized volatility

Valori bassi (30-60):
Pro: Reattivo a spike volatilità
Contro: Rumoroso

Valori alti (120-240):
Pro: Volatilità smooth
Contro: Lag su cambi regime
```

**Raccomandazione per TF**:
```
TF=1m  → rv_window=30-60
TF=5m  → rv_window=60-120
TF=15m → rv_window=120-180
TF=1h  → rv_window=180-240
```

---

## Advanced Settings

### Indicators Multi-Timeframe

**Sistema**:
```json
{
  "atr": ["1m", "5m", "15m"],
  "rsi": ["1m", "5m"],
  "macd": ["5m", "15m"]
}
```

**Effetto**:
```
ATR:
- atr_1m  (volatilità immediata)
- atr_5m  (volatilità breve)
- atr_15m (volatilità medio)

RSI:
- rsi_1m  (momentum immediato)
- rsi_5m  (momentum breve)

MACD:
- macd_5m  (trend breve)
- macd_15m (trend medio)

→ Totale: 7 features da 3 indicatori
```

**Best Practices**:
```
1. Include TF base + superiori (NO inferiori)
   ✅ TF=5m → ["5m", "15m", "1h"]
   ❌ TF=5m → ["1m", "5m", "15m"]  (1m crea noise)

2. Max 3 TF per indicatore
   ✅ ["5m", "15m", "1h"]
   ❌ ["5m", "10m", "15m", "30m", "1h"]  (troppi)

3. Gap TF ragionevole (2-4x)
   ✅ 5m → 15m (3x) → 1h (4x)
   ❌ 1m → 1h (60x)  (gap eccessivo)
```

### Additional Features

**Returns & Volatility**:
```
checkbox: Include calcolo returns e realized volatility

Quando usare: SEMPRE
Fondamentali per modelli di prezzo
```

**Trading Sessions**:
```
checkbox: Include features sessione (Tokyo/London/NY)

Effetto:
- is_tokyo: 1 se 00:00-09:00 UTC
- is_london: 1 se 08:00-17:00 UTC
- is_ny: 1 se 13:00-22:00 UTC
- overlap_london_ny: 1 se 13:00-17:00 UTC (alta volatilità!)

Quando usare:
✅ Intraday trading
✅ Strategia session-based
❌ Daily/Weekly TF (irrilevante)
```

**Candlestick Patterns**:
```
checkbox: Include pattern detection su TF superiore

Higher TF: Dropdown (5m, 15m, 1h, 4h)

Patterns rilevati:
- Engulfing (bullish/bearish)
- Hammer / Hanging Man
- Doji
- Morning/Evening Star

Quando usare:
✅ Reversal strategies
✅ Entry timing
❌ Trend following puro
```

**Volume Profile**:
```
checkbox: Include distribuzione volume per prezzo

VL Bins: 50 (default)

Effetto:
Calcola 50 livelli di prezzo, conta volume per livello
→ Identifica support/resistance da volume

Quando usare:
✅ Asset liquidi (Forex major)
✅ Hai dati volume affidabili
❌ Low volume assets
```

### Forecast Types

**Basic (default)**:
- Usa modelli standard addestrati
- Solo predizioni dirette

**Advanced**:
- Include regime detection
- Feature weights dinamici
- Conformal prediction

**Random Walk (RW)**:
- Baseline: predice flat (no change)
- Utile per benchmark

**Raccomandazione**: Basic (Advanced attualmente disabilitato in unified path)

---

## GPU Acceleration

### Inference GPU
```
checkbox "Usa GPU per inference"

Quando abilitato:
- Modelli PyTorch → GPU
- Modelli sklearn → CPU (no GPU support)

Speedup (RTX 4090):
Singolo modello: 10ms → 2ms (5x)
Ensemble 10 modelli: 100ms → 20ms (5x)
```

**Quando usare**:
- ✅ Hai modelli PyTorch (Autoencoder/VAE)
- ✅ Ensemble > 5 modelli
- ✅ Inference real-time critico
- ❌ Solo modelli sklearn (no benefit)

**Memoria GPU**:
```
1 modello PyTorch: ~100-200 MB VRAM
10 modelli ensemble: ~1-2 GB VRAM

RTX 4090 (24GB): Può gestire 100+ modelli!
```

---

## Best Practices

### Setup 1: Scalping (1 modello, veloce)
```
Models: EURUSD_1m_h30_rf_vae16.pkl
Combine models: N/A (solo 1)
Horizons: "1m, 2m, 3m, 5m"
N Samples: 500
Quantiles: "0.10, 0.50, 0.90"
Warmup: 64
RV Window: 30
Indicators: {"atr": ["1m","5m"], "rsi": ["1m"]}
GPU: ✅ (se modello PyTorch)

→ Latency: 5-10ms, Accuracy: buona
```

### Setup 2: Day Trading (ensemble robusto)
```
Models:
  - EURUSD_5m_h12_ridge_none.pkl
  - EURUSD_5m_h12_rf_pca10.pkl
  - EURUSD_5m_h24_elasticnet_vae16.pkl
Combine models: ✅
Horizons: "5m, 15m, 30m, 1h"
N Samples: 1000
Quantiles: "0.05, 0.50, 0.95"
Warmup: 80
RV Window: 60
Indicators: {"atr": ["5m","15m","1h"], "rsi": ["5m","15m"], "macd": ["15m"]}
Additional: Returns, Sessions
GPU: ✅

→ Latency: 20-30ms, Accuracy: ottima
```

### Setup 3: Swing (massima robustezza)
```
Models: 8-10 modelli diversi (algo, encoder, horizon)
Combine models: ✅
Horizons: "1h, 4h, 1d, 3d"
N Samples: 2000
Quantiles: "0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99"
Warmup: 100
RV Window: 180
Indicators: Multi-TF completo (4-5 TF per indicatore)
Additional: Tutti
GPU: ✅

→ Latency: 50-100ms, Accuracy: massima
```

### Checklist Pre-Inference
```
□ Modelli compatibili (stesso symbol/timeframe)?
□ Horizons <= 2x model horizon?
□ Warmup >= max indicator window?
□ GPU enabled se hai PyTorch models?
□ Indicators match training config?
□ N samples adeguato (1000+ per produzione)?
```

### Troubleshooting

**Forecast flat (linea orizzontale)**:
```
Causa: 1 modello single-horizon, no Enhanced scaling
Fix: Usa ensemble 3+ modelli o attiva Enhanced (se disponibile)
```

**Bande troppo strette**:
```
Causa: Quantiles conservative o pochi samples
Fix: Usa 0.05/0.95, aumenta n_samples a 2000
```

**Bande troppo larghe**:
```
Causa: Alta variance ensemble o alta volatilità
Fix: Normale se mercato volatile, o riduci modelli non allineati
```

**Latency alta**:
```
Causa: Troppi modelli, no GPU, troppi samples
Fix: Riduci a 5 modelli, abilita GPU, usa samples=500
```

**Predizioni irrealistiche**:
```
Causa: Horizons fuori range modello, bug scaling
Fix: Usa horizons <= 2x model horizon
```

---

## Interpretazione Output

### Forecast Line
```
Linea centrale = q50 (mediana)

Se sopra prezzo corrente → Prediction BUY
Se sotto prezzo corrente → Prediction SELL
```

### Confidence Bands
```
Banda superiore = q95
Banda inferiore = q05

Ampiezza banda = incertezza

Bande strette → Alta confidenza
Bande larghe → Bassa confidenza (volatilità alta, modelli disagree)
```

### Colori (se modelli separati)
```
Ogni modello = colore diverso

Convergenza colori → Modelli agree (forte signal)
Divergenza colori → Modelli disagree (weak signal)
```

---

## Performance Attesa

### Latency
```
Setup            | CPU      | GPU (RTX 4090)
-----------------|----------|---------------
1 modello        | 10-20ms  | 2-5ms
5 modelli        | 50-100ms | 10-20ms
10 modelli       | 100-200ms| 20-40ms
```

### Accuracy
```
1 modello:       MAE ~0.8% (baseline)
3-5 modelli:     MAE ~0.5% (ensemble benefit)
10+ modelli:     MAE ~0.3% (diminishing returns)
```

**Regola**: Oltre 10 modelli, benefit marginale (ma latency aumenta)

---

## Conclusione

**Pipeline ottimale**:
1. Seleziona 3-5 modelli DIVERSI (algo, encoder, horizon)
2. Combina in ensemble
3. Horizons appropriati (5-7 punti, max 2x model horizon)
4. N samples 1000+
5. Quantiles 0.05/0.50/0.95
6. Indicators multi-TF allineati a training
7. GPU enabled

**Result atteso**: Forecast robusto, MAE <0.5%, latency <50ms

Buon trading! 📈
