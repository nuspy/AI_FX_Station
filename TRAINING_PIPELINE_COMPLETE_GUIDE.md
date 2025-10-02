# Training Pipeline - Guida Completa

## Indice
1. [Overview Pipeline](#overview-pipeline)
2. [Parametri Base](#parametri-base)
3. [Encoders](#encoders)
4. [Modelli](#modelli)
5. [Ottimizzatori](#ottimizzatori)
6. [Timeframes](#timeframes)
7. [Indicatori](#indicatori)
8. [GPU Acceleration](#gpu-acceleration)
9. [Best Practices](#best-practices)

---

## Overview Pipeline

### Flusso Completo
```
1. FETCH DATA
   ↓ Scarica candele dal DB (days_history)
   ↓
2. FEATURE ENGINEERING
   ↓ OHLC relativo + Temporale + Realized Vol + Indicatori Multi-TF
   ↓
3. WARMUP & COVERAGE
   ↓ Scarta prime N barre (warmup_bars)
   ↓ Rimuovi features con coverage < min_coverage
   ↓
4. ENCODER (opzionale)
   ↓ PCA / Autoencoder / VAE → Compressione features
   ↓
5. TRAIN/VAL SPLIT
   ↓ val_frac = 0.2 (20% validation)
   ↓
6. HYPERPARAMETER OPTIMIZATION (opzionale)
   ↓ Genetic / NSGA-II → Best params
   ↓
7. MODEL TRAINING
   ↓ Ridge / Lasso / ElasticNet / RandomForest
   ↓
8. SAVE MODEL
   ↓ artifacts/models/SYMBOL_TF_dDAYS_hHORIZON_ALGO_ENCODER.pkl
```

---

## Parametri Base

### Symbol & Timeframe
```bash
--symbol EUR/USD    # Coppia da tradare
--timeframe 1m      # Timeframe base dati

# Scelte comuni:
1m  → Scalping (richiede bassa latenza, molti dati)
5m  → Day trading (buon compromesso)
15m → Swing trading breve termine
1h  → Swing trading medio termine
4h  → Position trading
1d  → Investing a lungo termine
```

**Come scegliere**:
- **1m-5m**: Se fai scalping/day trading, hai GPU potente, vuoi reattività immediata
- **15m-1h**: Se fai swing trading, vuoi meno noise, hai dati limitati
- **4h-1d**: Se fai position trading, vuoi stabilità, pochi trade

### Horizon (Target di Previsione)
```bash
--horizon 30    # Prevedi a 30 barre nel futuro

# Esempi pratici:
TF=1m, horizon=30  → Prevedi a 30 minuti
TF=5m, horizon=12  → Prevedi a 1 ora (12*5min)
TF=1h, horizon=24  → Prevedi a 1 giorno (24 ore)
```

**Regola d'oro**: `horizon` deve essere 20-50% del `days_history` in barre.

**Esempi corretti**:
```bash
# Scalping (1m)
--timeframe 1m --horizon 30 --days_history 7    # 30min forecast, 10k barre training

# Day trading (5m)
--timeframe 5m --horizon 12 --days_history 14   # 1h forecast, 4k barre training

# Swing (1h)
--timeframe 1h --horizon 24 --days_history 60   # 1d forecast, 1.4k barre training
```

**Esempi sbagliati**:
```bash
--timeframe 1m --horizon 180 --days_history 1   # ❌ Horizon troppo lungo (3h su 1 giorno dati)
--timeframe 1h --horizon 5 --days_history 365   # ❌ Horizon troppo corto (5h su 1 anno dati)
```

### Days History
```bash
--days_history 30    # Scarica ultimi 30 giorni

# Valori bassi (1-7 giorni):
Pro: Training veloce, cattura regime recente
Contro: Poco generalizzabile, rischio overfitting

# Valori medi (14-60 giorni):
Pro: Buon compromesso, cattura patterns stagionali
Contro: Potrebbe includere regimi obsoleti

# Valori alti (90-365 giorni):
Pro: Robusto, generalizza bene
Contro: Training lento, potrebbe diluire signal recente
```

**Raccomandazione**:
- **Scalping (1m-5m)**: 3-14 giorni (mercato cambia velocemente)
- **Day trading (15m-1h)**: 14-60 giorni (cattura patterns settimanali)
- **Swing (4h-1d)**: 60-180 giorni (cattura stagionalità)

### Warmup Bars
```bash
--warmup_bars 64    # Scarta prime 64 barre

# Valori bassi (10-30):
Pro: Massimizza dati training
Contro: Indicatori instabili all'inizio

# Valori medi (30-100):
Pro: Indicatori stabilizzati, buon trade-off
Contro: Perde alcune barre iniziali

# Valori alti (100-200):
Pro: Indicatori perfettamente stabi, massima qualità
Contro: Spreca molti dati
```

**Formula**: `warmup_bars >= max(indicator_windows)`

Esempio:
```bash
# Se usi:
--rsi_n 14        # RSI window = 14
--bb_n 20         # Bollinger window = 20
--hurst_window 64 # Hurst window = 64

# Allora:
--warmup_bars 64  # >= max(14, 20, 64) = 64
```

### Min Feature Coverage
```bash
--min_feature_coverage 0.15    # Rimuovi features con >85% NaN

# Valori bassi (0.05-0.10):
Pro: Mantiene più features, più info
Contro: Rischio features rumorose con molti NaN

# Valori medi (0.15-0.30):
Pro: Rimuove features instabili, buon trade-off
Contro: Potrebbe scartare features utili ma sparse

# Valori alti (0.40-0.50):
Pro: Solo features di alta qualità
Contro: Perde molte features, meno diversità
```

**Quando alzare**: Se hai molte features (>200) e vuoi speed
**Quando abbassare**: Se hai poche features (<50) e vuoi massimizzare info

---

## Encoders

### 1. None (Nessun Encoder)
```bash
--encoder none
```

**Cosa fa**: Usa features raw senza compressione

**Pro**:
- Semplicità massima
- Interpretabilità features
- Training velocissimo
- Nessuna perdita informazione

**Contro**:
- Curse of dimensionality con >100 features
- Rischio overfitting con molte features
- Nessuna denoising

**Quando usarlo**:
- ✅ Features < 50
- ✅ Vuoi interpretabilità
- ✅ Hai pochi dati (<1000 samples)
- ❌ Features > 100

### 2. PCA (Principal Component Analysis)
```bash
--encoder pca --latent_dim 10
```

**Cosa fa**: Compressione lineare - proietta features su componenti principali

**Pro**:
- Velocissimo (CPU)
- Deterministico (stesso risultato sempre)
- Mantiene varianza massima
- Rimuove correlazioni

**Contro**:
- Solo trasformazioni lineari
- Perde info non-lineare
- Difficile interpretare componenti

**Quando usarlo**:
- ✅ Features 50-200
- ✅ Vuoi velocità
- ✅ Features sono correlate
- ❌ Patterns non-lineari complessi

**Parametri**:
```bash
--latent_dim 10   # Comprimi a 10 componenti

# Valori bassi (5-10):
Pro: Massima compressione, speed
Contro: Perde molte info

# Valori medi (10-30):
Pro: Mantiene 80-90% varianza
Contro: Compressione moderata

# Valori alti (30-50):
Pro: Mantiene 95%+ varianza
Contro: Poca compressione, lento
```

**Formula euristica**: `latent_dim = sqrt(n_features)`

Esempio:
```bash
# Se hai 100 features:
--latent_dim 10    # sqrt(100) = 10

# Se hai 400 features:
--latent_dim 20    # sqrt(400) = 20
```

### 3. Autoencoder (Neural, Non-Lineare)
```bash
--encoder autoencoder --latent_dim 16 --encoder_epochs 50
```

**Cosa fa**: Rete neurale encoder-decoder, compressione non-lineare

**Pro**:
- Cattura patterns non-lineari
- Denoising automatico
- Più potente di PCA

**Contro**:
- Training lento (senza GPU: 5-10min, con GPU: 30-60sec)
- Non deterministico (varia ad ogni run)
- Rischio overfitting

**Quando usarlo**:
- ✅ Features > 200
- ✅ Hai GPU (IMPORTANTE!)
- ✅ Patterns non-lineari complessi
- ✅ Dati > 5000 samples
- ❌ Pochi dati (<1000 samples)

**Parametri**:
```bash
--latent_dim 16         # Dimensione bottleneck
--encoder_epochs 50     # Epoche training
--use-gpu              # Usa GPU (10-15x speedup!)

# latent_dim:
Valori bassi (8-16): Compressione aggressiva, rischio underfitting
Valori medi (16-32): Buon compromesso
Valori alti (32-64): Massima capacità, rischio overfitting

# encoder_epochs:
Valori bassi (20-30): Underfit, encoder non converge
Valori medi (50-100): Converge bene
Valori alti (100-200): Rischio overfit, ma migliore denoising
```

**Speedup GPU**:
```
CPU (RTX 4090):
latent_dim=16, epochs=50 → 8 minuti

GPU (RTX 4090):
latent_dim=16, epochs=50 → 35 secondi  (13x faster!)
```

### 4. VAE (Variational Autoencoder)
```bash
--encoder vae --latent_dim 16 --encoder_epochs 100
```

**Cosa fa**: Autoencoder probabilistico con regolarizzazione KL divergence

**Pro**:
- Più robusto di autoencoder standard
- Meno overfitting (grazie a KL loss)
- Genera latent space smooth
- Migliore denoising

**Contro**:
- Training molto lento (2x autoencoder)
- Più complesso
- Richiede più epochs per convergere

**Quando usarlo**:
- ✅ Features > 500
- ✅ Hai GPU + tempo
- ✅ Serve robustezza massima
- ✅ Dati molto rumorosi
- ❌ Training veloce richiesto

**Parametri**:
```bash
--latent_dim 16         # Dimensione latent space
--encoder_epochs 100    # VAE richiede più epochs
--use-gpu              # Quasi obbligatorio!

# Regola: VAE richiede 2x epochs di Autoencoder
Autoencoder: 50 epochs → VAE: 100 epochs
Autoencoder: 100 epochs → VAE: 200 epochs
```

**Confronto Encoder - Riassunto**:
```
Scenario 1: Features < 50, vuoi velocità
→ none

Scenario 2: Features 50-200, vuoi velocità
→ pca (latent_dim = sqrt(features))

Scenario 3: Features > 200, hai GPU, vuoi potenza
→ autoencoder (latent_dim=16-32, epochs=50-100, --use-gpu)

Scenario 4: Features > 500, hai GPU + tempo, vuoi robustezza
→ vae (latent_dim=16-32, epochs=100-200, --use-gpu)
```

---

## Modelli

### 1. Ridge Regression
```bash
--algo ridge --alpha 0.001
```

**Cosa fa**: Regressione lineare con regolarizzazione L2

**Pro**:
- Velocissimo (secondi)
- Stabile (no overfitting estremo)
- Funziona bene con features correlate
- Interpetabile

**Contro**:
- Solo relazioni lineari
- Performance limitata con patterns complessi
- Non cattura interazioni

**Quando usarlo**:
- ✅ Baseline rapido
- ✅ Features linearmente separabili
- ✅ Vuoi interpretabilità
- ❌ Patterns non-lineari

**Parametri**:
```bash
--alpha 0.001    # Forza regolarizzazione

# Valori bassi (0.0001-0.001):
Pro: Più flessibile, più capacità
Contro: Rischio overfitting

# Valori medi (0.001-0.01):
Pro: Buon trade-off
Contro: Potrebbe underfit con pochi dati

# Valori alti (0.01-0.1):
Pro: Massima regolarizzazione, robusto
Contro: Underfitting
```

### 2. Lasso Regression
```bash
--algo lasso --alpha 0.001
```

**Cosa fa**: Regressione lineare con regolarizzazione L1 (feature selection)

**Pro**:
- Feature selection automatica
- Zeroes out features inutili
- Modello sparse (poche features attive)
- Veloce

**Contro**:
- Solo relazioni lineari
- Può scartare features utili
- Instabile con features correlate

**Quando usarlo**:
- ✅ Molte features (>100)
- ✅ Vuoi capire quali features contano
- ✅ Sospetti molte features sono noise
- ❌ Poche features critiche

**Parametri**: Stessi di Ridge

### 3. ElasticNet
```bash
--algo elasticnet --alpha 0.001 --l1_ratio 0.5
```

**Cosa fa**: Combinazione Ridge (L2) + Lasso (L1)

**Pro**:
- Best of both: regolarizzazione + feature selection
- Gestisce features correlate meglio di Lasso
- Flessibile con l1_ratio

**Contro**:
- Due iperparametri da tunare
- Più lento di Ridge/Lasso

**Quando usarlo**:
- ✅ Features correlate + serve feature selection
- ✅ Non sai se L1 o L2 è meglio
- ✅ Hai optimization attivo (trova best l1_ratio)
- ❌ Vuoi semplicità

**Parametri**:
```bash
--alpha 0.001      # Forza regolarizzazione totale
--l1_ratio 0.5     # Mix L1/L2

# l1_ratio:
0.0 → Ridge puro (solo L2)
0.5 → 50% L1, 50% L2 (bilanciato)
1.0 → Lasso puro (solo L1)

# Quando usare:
l1_ratio=0.0-0.3 → Poche features da eliminare
l1_ratio=0.3-0.7 → Moderata feature selection
l1_ratio=0.7-1.0 → Aggressiva feature selection
```

### 4. Random Forest
```bash
--algo rf --n_estimators 400
```

**Cosa fa**: Ensemble di decision trees con bagging

**Pro**:
- Cattura interazioni non-lineari
- Robusto a outliers
- No assunzioni sui dati
- Feature importance gratis

**Contro**:
- Lento con molti estimators
- Tende a overfittare se non tunato
- Black box (meno interpretabile)

**Quando usarlo**:
- ✅ Patterns non-lineari complessi
- ✅ Features con interazioni
- ✅ Hai tempo/risorse
- ✅ Non serve interpretabilità perfetta
- ❌ Vuoi velocità massima

**Parametri**:
```bash
--n_estimators 400    # Numero alberi

# Valori bassi (50-100):
Pro: Training veloce
Contro: Underfitting, alta variance

# Valori medi (200-400):
Pro: Buon trade-off
Contro: Training 5-30 secondi

# Valori alti (500-1000):
Pro: Massima performance, bassa variance
Contro: Training lento (30-120 sec)

# Regola: Più estimators = migliore SEMPRE (con diminishing returns dopo 400)
```

**Confronto Modelli - Riassunto**:
```
Scenario 1: Baseline veloce, vuoi capire features
→ ridge (alpha=0.001)

Scenario 2: Molte features, serve feature selection
→ lasso (alpha=0.001) o elasticnet (l1_ratio=0.7)

Scenario 3: Features correlate + feature selection
→ elasticnet (alpha=0.001, l1_ratio=0.5)

Scenario 4: Patterns non-lineari, massima performance
→ rf (n_estimators=400)
```

---

## Ottimizzatori

### 1. None (Manual)
```bash
--optimization none
```

**Cosa fa**: Usa parametri di default, nessuna ricerca

**Pro**:
- Training velocissimo (1x)
- Semplice
- Deterministico

**Contro**:
- Performance sub-ottimale
- Devi tunare manualmente

**Quando usarlo**:
- ✅ Test rapidi
- ✅ Conosci già best params
- ✅ Vuoi velocità
- ❌ Prima volta con questi dati

### 2. Genetic-Basic (Single Objective)
```bash
--optimization genetic-basic --gen 5 --pop 8
```

**Cosa fa**: Algoritmo genetico che ottimizza solo MAE

**Pro**:
- Trova parametri buoni
- Training moderato (5-10x rispetto a none)
- Semplice (1 obiettivo)

**Contro**:
- Ignora complessità/overfitting
- Può trovare soluzioni fragili
- Non multi-objective

**Quando usarlo**:
- ✅ Vuoi performance migliore
- ✅ Hai 10-30min
- ✅ Non serve Pareto optimization
- ❌ Serve trade-off performance/complessità

**Parametri**:
```bash
--gen 5     # Generazioni
--pop 8     # Popolazione

# gen (generazioni):
Valori bassi (3-5): Converge veloce, potrebbe non trovare ottimo
Valori medi (5-10): Buon trade-off
Valori alti (10-20): Massima esplorazione, molto lento

# pop (popolazione):
Valori bassi (4-8): Veloce, esplorazione limitata
Valori medi (8-16): Buon compromesso
Valori alti (16-32): Massima diversità, lentissimo

# Tempo stimato:
gen=5, pop=8 → 40 training runs → 5-15 minuti (Ridge/Lasso)
gen=5, pop=8 → 40 training runs → 20-60 minuti (RandomForest)
```

### 3. NSGA-II (Multi-Objective)
```bash
--optimization nsga2 --gen 5 --pop 8
```

**Cosa fa**: Non-dominated Sorting Genetic Algorithm - ottimizza MAE + Complessità simultaneamente

**Pro**:
- Trova Pareto front (trade-off ottimali)
- Evita overfitting (penalizza complessità)
- Soluzioni robuste
- Migliore generalizzazione

**Contro**:
- Lento come genetic-basic
- Più complesso (2 obiettivi)
- Richiede expertise per scegliere

**Quando usarlo**:
- ✅ Produzione seria
- ✅ Serve robustezza
- ✅ Vuoi multiple opzioni (Pareto front)
- ✅ Hai tempo
- ❌ Test rapidi

**Output Pareto Front**:
```
Soluzione A: MAE=0.0015, Complexity=50  (semplice, accurato)
Soluzione B: MAE=0.0012, Complexity=100 (complesso, molto accurato)
Soluzione C: MAE=0.0014, Complexity=70  (trade-off)

→ Scegli based on:
- Se serve velocità inference → A
- Se serve massima accuratezza → B
- Se serve compromesso → C
```

**Confronto Ottimizzatori - Riassunto**:
```
Scenario 1: Test rapido, conosci parametri
→ none

Scenario 2: Vuoi migliorare, hai 15-30min
→ genetic-basic (gen=5, pop=8)

Scenario 3: Produzione, serve robustezza, hai 30-60min
→ nsga2 (gen=5, pop=8)

Scenario 4: Massima performance, hai ore
→ nsga2 (gen=10, pop=16)
```

---

## Timeframes

### Relazione TF - Horizon - Days
```
Timeframe | Horizon (bars) | Horizon (tempo) | Days History | Samples
----------|---------------|-----------------|--------------|--------
1m        | 30            | 30min           | 7            | ~10k
5m        | 12            | 1h              | 14           | ~4k
15m       | 16            | 4h              | 30           | ~3k
1h        | 24            | 1d              | 60           | ~1.4k
4h        | 12            | 2d              | 90           | ~500
1d        | 5             | 5d              | 180          | ~180
```

### Scelta Strategica

**Scalping (1m-5m)**:
```bash
--timeframe 1m --horizon 30 --days_history 7
# Pro: Massima reattività, molti trade
# Contro: Rumore alto, commissioni pesano, latency critica
```

**Day Trading (5m-15m)**:
```bash
--timeframe 5m --horizon 12 --days_history 14
# Pro: Buon signal/noise, gestibile
# Contro: Richiede monitoring attivo
```

**Swing Trading (1h-4h)**:
```bash
--timeframe 1h --horizon 24 --days_history 60
# Pro: Meno rumore, pochi trade, più affidabile
# Contro: Meno opportunità, holding risk
```

**Position (1d)**:
```bash
--timeframe 1d --horizon 5 --days_history 180
# Pro: Massima stabilità, fundamentals contano
# Contro: Poche predizioni, lento
```

---

## Indicatori

### Sistema Multi-Timeframe
```bash
--indicator_tfs '{"atr": ["1m", "5m", "15m"], "rsi": ["1m", "5m"]}'
```

**Cosa fa**: Calcola stesso indicatore su timeframe diversi → cattura patterns multi-scala

**Esempio**:
```
atr_1m  → Volatilità immediata (scalping signal)
atr_5m  → Volatilità breve termine
atr_15m → Volatilità medio termine (swing context)

rsi_1m  → Ipercomprato/venduto immediato
rsi_5m  → Trend breve termine
```

**Regola**: Includi sempre TF base + 2-3 TF superiori
```bash
# Se TF base = 1m:
{"atr": ["1m", "5m", "15m"]}  # ✅ Buono

# Se TF base = 1h:
{"atr": ["1h", "4h", "1d"]}   # ✅ Buono

# Se TF base = 1m:
{"atr": ["1m", "1h", "1d"]}   # ❌ Gap troppo grande (1m → 1h)
```

### Indicatori Disponibili

**Volatilità**:
```bash
"atr": ["1m", "5m", "15m"]     # Average True Range
"bollinger": ["5m", "15m"]     # Bande Bollinger
"keltner": ["15m"]             # Keltner Channels

# Quando usarli:
- ATR: Sempre (sizing posizioni, stop loss)
- Bollinger: Mean reversion strategies
- Keltner: Breakout strategies
```

**Momentum**:
```bash
"rsi": ["1m", "5m"]            # Relative Strength Index
"macd": ["15m", "1h"]          # MACD
"stoch": ["5m"]                # Stochastic
"cci": ["15m"]                 # Commodity Channel Index
"williams_r": ["5m"]           # Williams %R

# Quando usarli:
- RSI: Sempre (ipercomprato/venduto)
- MACD: Trend following
- Stoch/Williams: Entry/exit timing
- CCI: Cyclic markets
```

**Trend**:
```bash
"adx": ["15m", "1h"]           # Average Directional Index
"ema": ["5m", "15m", "1h"]     # Exponential MA
"sma": ["15m", "1h"]           # Simple MA

# Quando usarli:
- ADX: Detect trend strength (>25 = trend)
- EMA/SMA: Trend direction, support/resistance
```

**Volume**:
```bash
"mfi": ["5m"]                  # Money Flow Index
"obv": ["15m"]                 # On Balance Volume
"vwap": ["1h"]                 # Volume Weighted Avg Price

# Quando usarli:
- MFI: Pressure acquisto/vendita
- OBV: Conferma trend
- VWAP: Intraday benchmark
```

**Pattern**:
```bash
"donchian": ["15m"]            # Donchian Channels
"hurst": ["1h"]                # Hurst Exponent

# Quando usarli:
- Donchian: Breakout su max/min
- Hurst: Detect trending vs mean-reverting
```

### Parametri Indicatori
```bash
--atr_n 14         # ATR period
--rsi_n 14         # RSI period
--bb_n 20          # Bollinger period
--hurst_window 64  # Hurst window

# Valori bassi (7-10):
Pro: Più reattivo, cattura cambi veloci
Contro: Rumoroso, falsi segnali

# Valori medi (14-20):
Pro: Standard industry, buon trade-off
Contro: Potrebbe essere lento in mercati veloci

# Valori alti (25-50):
Pro: Smooth, pochi falsi segnali
Contro: Lag elevato, perde entry
```

**Raccomandazione**:
```bash
# Scalping:
--atr_n 10 --rsi_n 10 --bb_n 15

# Day trading:
--atr_n 14 --rsi_n 14 --bb_n 20  # Standard

# Swing:
--atr_n 20 --rsi_n 21 --bb_n 25
```

---

## GPU Acceleration

### Training
```bash
--use-gpu    # Abilita GPU per encoder training
```

**Speedup (RTX 4090)**:
```
Autoencoder (latent_dim=16, epochs=50):
CPU: 8 min → GPU: 35 sec  (13x faster)

VAE (latent_dim=16, epochs=100):
CPU: 18 min → GPU: 1 min 20 sec  (13x faster)

PCA/Ridge/Lasso/ElasticNet/RF:
Rimangono su CPU (non supportano GPU in sklearn)
```

**Quando usare GPU**:
- ✅ encoder=autoencoder
- ✅ encoder=vae
- ❌ encoder=pca (CPU comunque velocissimo)
- ❌ encoder=none

### Inference
```bash
# In UI: checkbox "Usa GPU per inference"
```

**Speedup (RTX 4090)**:
```
Singolo modello PyTorch:
CPU: 10ms → GPU: 2ms  (5x faster)

Ensemble 10 modelli PyTorch:
CPU: 100ms → GPU: 20ms  (5x faster)

Modelli sklearn:
Rimangono su CPU
```

### Requisiti GPU
```
Minimo:
- NVIDIA GPU CUDA 3.5+
- 4GB VRAM
- CUDA 11.8 o 12.x

Raccomandato:
- NVIDIA GPU CUDA 6.0+ (Pascal+)
- 8GB+ VRAM
- CUDA 12.x

Optimal (questo sistema):
- RTX 4090 (24GB VRAM)
- CUDA 12.1
- PyTorch 2.5.1+cu121
```

---

## Best Practices

### 1. Pipeline Minima (Test Rapido)
```bash
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD --timeframe 5m --horizon 12 \
  --algo ridge --encoder none \
  --days_history 7 --artifacts_dir ./artifacts \
  --optimization none

# Tempo: 30 secondi
# Uso: Test rapido, baseline
```

### 2. Pipeline Ottimale (Produzione Scalping)
```bash
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD --timeframe 1m --horizon 30 \
  --algo rf --n_estimators 400 \
  --encoder vae --latent_dim 16 --encoder_epochs 100 \
  --use-gpu \
  --days_history 14 --artifacts_dir ./artifacts \
  --warmup_bars 64 --min_feature_coverage 0.15 \
  --indicator_tfs '{"atr":["1m","5m","15m"],"rsi":["1m","5m"],"macd":["5m","15m"]}' \
  --atr_n 10 --rsi_n 10 --bb_n 15 \
  --optimization nsga2 --gen 5 --pop 8

# Tempo: 30-45 min (con GPU)
# Uso: Produzione scalping
```

### 3. Pipeline Ottimale (Produzione Swing)
```bash
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD --timeframe 1h --horizon 24 \
  --algo rf --n_estimators 400 \
  --encoder autoencoder --latent_dim 20 --encoder_epochs 50 \
  --use-gpu \
  --days_history 60 --artifacts_dir ./artifacts \
  --warmup_bars 100 --min_feature_coverage 0.20 \
  --indicator_tfs '{"atr":["1h","4h"],"rsi":["1h","4h"],"macd":["4h","1d"],"adx":["4h"]}' \
  --atr_n 14 --rsi_n 14 --bb_n 20 \
  --optimization nsga2 --gen 8 --pop 12

# Tempo: 45-90 min (con GPU)
# Uso: Produzione swing trading
```

### 4. Checklist Pre-Training
```
□ Hai dati sufficienti? (days_history * bars_per_day > 1000)
□ Horizon sensato? (20-50% dei dati in barre)
□ Warmup >= max indicator window?
□ GPU abilitata per encoder neural?
□ Indicatori multi-TF coerenti con strategia?
□ Optimization attivo se prima volta?
□ Encoder appropriato per n_features?
```

### 5. Troubleshooting

**Training troppo lento**:
```
Causa: RF con n_estimators alto + optimization
Fix: Riduci n_estimators a 200, gen=3
```

**MAE alto (>0.01)**:
```
Causa: Pochi dati o horizon troppo lungo
Fix: Aumenta days_history, riduci horizon
```

**Overfitting (train MAE << val MAE)**:
```
Causa: Encoder troppo potente o alpha troppo basso
Fix: Riduci latent_dim, aumenta alpha, usa optimization
```

**Out of Memory (GPU)**:
```
Causa: Encoder troppo grande
Fix: Riduci latent_dim, riduci encoder_epochs
```

**Features tutte NaN**:
```
Causa: Warmup troppo corto
Fix: Aumenta warmup_bars >= max(indicator_windows)
```

---

## Comandi Completi - Templates

### Template 1: Scalping (1m)
```bash
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD --timeframe 1m --horizon 30 \
  --algo rf --n_estimators 400 \
  --encoder vae --latent_dim 16 --encoder_epochs 100 --use-gpu \
  --days_history 14 --warmup_bars 64 --min_feature_coverage 0.15 \
  --indicator_tfs '{"atr":["1m","5m","15m"],"rsi":["1m","5m"],"macd":["5m"]}' \
  --atr_n 10 --rsi_n 10 --bb_n 15 \
  --optimization nsga2 --gen 5 --pop 8 \
  --artifacts_dir ./artifacts
```

### Template 2: Day Trading (5m)
```bash
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD --timeframe 5m --horizon 12 \
  --algo rf --n_estimators 400 \
  --encoder autoencoder --latent_dim 20 --encoder_epochs 50 --use-gpu \
  --days_history 30 --warmup_bars 80 --min_feature_coverage 0.15 \
  --indicator_tfs '{"atr":["5m","15m","1h"],"rsi":["5m","15m"],"macd":["15m","1h"],"adx":["1h"]}' \
  --atr_n 14 --rsi_n 14 --bb_n 20 \
  --optimization nsga2 --gen 5 --pop 8 \
  --artifacts_dir ./artifacts
```

### Template 3: Swing Trading (1h)
```bash
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD --timeframe 1h --horizon 24 \
  --algo rf --n_estimators 400 \
  --encoder autoencoder --latent_dim 24 --encoder_epochs 50 --use-gpu \
  --days_history 60 --warmup_bars 100 --min_feature_coverage 0.20 \
  --indicator_tfs '{"atr":["1h","4h","1d"],"rsi":["1h","4h"],"macd":["4h","1d"],"adx":["4h","1d"]}' \
  --atr_n 14 --rsi_n 14 --bb_n 20 \
  --optimization nsga2 --gen 8 --pop 12 \
  --artifacts_dir ./artifacts
```

### Template 4: Baseline Rapido (qualsiasi TF)
```bash
python -m forex_diffusion.training.train_sklearn \
  --symbol EUR/USD --timeframe 5m --horizon 12 \
  --algo ridge --encoder pca --latent_dim 10 \
  --days_history 14 --warmup_bars 50 \
  --indicator_tfs '{"atr":["5m"],"rsi":["5m"]}' \
  --optimization none \
  --artifacts_dir ./artifacts
```

---

## Conclusione

**Regola d'oro**: Inizia con baseline (Template 4), poi ottimizza incrementalmente:
1. Baseline Ridge+PCA → capire se c'è signal
2. Passa a RF → cattura non-linearità
3. Aggiungi Autoencoder+GPU → comprimi features
4. Attiva NSGA-II → trova best params
5. Affina indicatori multi-TF → cattura multi-scala

**Performance attesa**:
- **MAE < 0.005**: Ottimo (0.5% error su returns)
- **MAE 0.005-0.01**: Buono (0.5-1% error)
- **MAE 0.01-0.02**: Accettabile (1-2% error)
- **MAE > 0.02**: Scarso (>2% error, rivedi setup)

Buon training! 🚀
