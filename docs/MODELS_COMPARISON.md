# Confronto: Supervised Models vs Diffusion Models

## Introduzione

ForexGPT supporta due approcci fondamentalmente diversi al machine learning per la previsione forex:

1. **Supervised Models** (Modelli Supervisionati) - modelli classici di regressione
2. **Diffusion Models** (Modelli di Diffusione) - modelli generativi di tipo probabilistico

Questa guida spiega le differenze, vantaggi/svantaggi, e quando usare ciascun approccio.

---

## Supervised Models

### Cosa Sono

I modelli supervisionati apprendono una funzione diretta `f(X) → y` che mappa input (features) a output (target). Il training consiste nel minimizzare l'errore tra predizioni e valori reali.

**Modelli disponibili**:
- **Ridge Regression**: regressione lineare con regolarizzazione L2
- **Lasso Regression**: regressione lineare con regolarizzazione L1 (feature selection)
- **ElasticNet**: combinazione di Ridge e Lasso
- **Random Forest (RF)**: ensemble di decision trees
- **Lightning**: neural network con PyTorch Lightning (MLP/LSTM/Transformer)

### Come Funzionano

1. **Input**: Features multi-timeframe (indicatori tecnici, returns, volatility, etc.)
2. **Training**: Il modello apprende la relazione statistica tra features e target (es. prezzo futuro)
3. **Predizione**: Dato un nuovo input X, il modello restituisce direttamente la predizione y

**Esempio**:
```
Input: [RSI_1m=65, MACD_5m=0.003, ATR_15m=0.0012, ...]
Output: Prezzo previsto tra 5 candele = 1.0856
```

### Vantaggi

✅ **Velocità**: Training e inference molto rapidi (secondi o minuti)
✅ **Interpretabilità**: Pesi/importanze delle features sono ispezionabili
✅ **Stabilità**: Convergenza garantita per modelli lineari
✅ **Efficienza dati**: Funzionano anche con dataset piccoli (migliaia di sample)
✅ **Semplicità**: Pochi iperparametri da ottimizzare
✅ **Robustezza**: Meno sensibili a overfitting (con regolarizzazione)
✅ **Deployment**: Modelli leggeri, facili da servire in produzione

### Svantaggi

❌ **Assunzioni lineari**: Ridge/Lasso assumono relazioni lineari (limitazione superata da RF/Lightning)
❌ **Predizioni puntuali**: Output deterministico, nessuna quantificazione incertezza
❌ **Difficoltà con multi-step**: Previsioni a lungo termine degradano rapidamente
❌ **Feature engineering manuale**: Servono features ben progettate
❌ **Sensibilità a outlier**: Soprattutto modelli lineari
❌ **Nessuna generazione di scenari**: Solo 1 predizione, non distribuzione di possibilità

### Quando Usarli

🎯 **Use cases ideali**:
- Previsioni a breve termine (1-20 candele)
- Prototipazione rapida e iterazione veloce
- Production con latenza critica (<100ms)
- Dataset limitati (giorni/settimane di dati)
- Necessità di interpretabilità (compliance, audit)
- Baseline per confronti con modelli più complessi

🎯 **Parametri critici**:
- **days_history**: 7-60 giorni (più dati = migliore generalizzazione, ma più lento)
- **horizon**: 1-20 candele (oltre 20, la precisione degrada)
- **optimization**: `genetic-basic` o `nsga2` per hyperparameter tuning automatico

---

## Diffusion Models

### Cosa Sono

I modelli di diffusione sono modelli generativi che apprendono a "denoising" (rimuovere rumore) da una distribuzione casuale per generare samples realistici. Invece di predire direttamente il target, apprendono il processo inverso di aggiunta graduale di rumore.

**Architetture disponibili**:
- **DDPM** (Denoising Diffusion Probabilistic Models): processo Markoviano con T timesteps
- **DDIM** (Denoising Diffusion Implicit Models): versione deterministica/accelerata
- **UNet-based**: architettura convoluzionale con skip connections
- **Transformer-based**: architettura attention-based per catturare dipendenze temporali

### Come Funzionano

1. **Forward Process (Training)**: Aggiungi rumore Gaussiano progressivamente ai dati reali in T steps
   ```
   x₀ (dato reale) → x₁ → x₂ → ... → x_T (rumore puro)
   ```

2. **Reverse Process (Inference)**: Il modello apprende a rimuovere il rumore step-by-step
   ```
   z (rumore casuale) → x_{T-1} → x_{T-2} → ... → x₀ (predizione)
   ```

3. **Output**: Distribuzione di possibili scenari futuri (non singolo valore)

**Esempio**:
```
Input: Sequenza storica + rumore casuale z
Output: 100 scenari possibili per i prossimi 50 candele
→ Scenario 1: [1.0856, 1.0862, 1.0858, ...]
→ Scenario 2: [1.0854, 1.0851, 1.0849, ...]
→ Scenario 3: [1.0857, 1.0865, 1.0872, ...]
```

### Vantaggi

✅ **Incertezza quantificata**: Genera distribuzione di probabilità, non singolo valore
✅ **Multi-step naturale**: Può generare lunghe sequenze (50-500 candele) in un solo forward pass
✅ **Cattura non-linearità complesse**: Architetture profonde (UNet/Transformer)
✅ **Robustezza a outlier**: Il processo di denoising è intrinsecamente robusto
✅ **Generazione scenari**: Utile per risk management e portfolio optimization
✅ **Flessibilità**: Può condizionare su features esterne (regime, volatility, news)
✅ **State-of-the-art**: Performance superiori su benchmark time-series complessi

### Svantaggi

❌ **Lentezza**: Training richiede ore/giorni anche su GPU
❌ **Inference lenta**: Sampling richiede T forward passes (10-1000 steps)
❌ **Complessità**: Molti iperparametri (timesteps, noise schedule, architecture, etc.)
❌ **Requisiti hardware**: GPU obbligatoria per training efficiente
❌ **Dati massivi**: Servono centinaia di migliaia / milioni di samples
❌ **Overfitting facile**: Architetture profonde con milioni di parametri
❌ **Debugging difficile**: Black box, difficile interpretare cosa ha appreso
❌ **Instabilità**: Training può divergere senza tuning attento (learning rate, batch size, etc.)

### Quando Usarli

🎯 **Use cases ideali**:
- Previsioni a lungo termine (50-500 candele)
- Generazione di scenari multipli per risk analysis
- Dataset molto grandi (anni di dati ad alta frequenza)
- Ricerca e sperimentazione (non production critica)
- Quando l'incertezza è fondamentale (VaR, stress testing)
- Quando GPU è disponibile

🎯 **Parametri critici**:
- **num_diffusion_timesteps**: 50-1000 (più alto = migliore qualità, ma più lento)
- **model_channels**: 64-256 (capacità modello, attenzione a overfitting)
- **learning_rate**: 1e-5 a 1e-3 (critico per stabilità)
- **batch_size**: 32-256 (più alto = gradiente stabile, ma serve più RAM)
- **dropout**: 0.1-0.3 (regolarizzazione contro overfitting)
- **num_heads**: 4-16 (per Transformer, più heads = cattura pattern complessi)

---

## Confronto Diretto

| Dimensione | Supervised | Diffusion |
|------------|-----------|-----------|
| **Training Time** | Minuti - Ore | Ore - Giorni |
| **Inference Time** | <1ms | 100ms - 10s |
| **Dataset Size** | 1K - 100K samples | 100K - 10M samples |
| **Hardware** | CPU ok | GPU consigliata |
| **Precisione Short-term** | ⭐⭐⭐⭐⭐ Alta | ⭐⭐⭐ Media |
| **Precisione Long-term** | ⭐⭐ Bassa | ⭐⭐⭐⭐ Alta |
| **Incertezza** | ❌ No | ✅ Sì (distribuzione) |
| **Interpretabilità** | ⭐⭐⭐⭐ Alta (RF, Linear) | ⭐ Bassa |
| **Overfitting Risk** | ⭐⭐ Basso (con regolarizzazione) | ⭐⭐⭐⭐ Alto |
| **Production Ready** | ✅ Sì | ⚠️ Dipende (latency) |
| **Manutenzione** | ⭐⭐⭐⭐⭐ Semplice | ⭐⭐ Complessa |

---

## Esempi Pratici

### Caso 1: Day Trading (1m timeframe, horizon 5-10 candele)

**Raccomandazione**: Supervised (Ridge/RF/Lightning)

**Motivo**: Velocità critica, previsioni puntuali sufficienti, dataset piccolo (ore/giorni di dati). Un modello Ridge con indicatori multi-TF offre ottimo rapporto performance/complessità.

**Configurazione esempio**:
```
Model: ridge
Encoder: none
Days: 7
Horizon: 5
Optimization: genetic-basic
Features: RSI, MACD, ATR (1m, 5m, 15m)
```

### Caso 2: Swing Trading (1h timeframe, horizon 50-100 candele)

**Raccomandazione**: Diffusion (DDPM con UNet)

**Motivo**: Previsioni a lungo termine (2-4 giorni), beneficio da scenari multipli per risk management. Dataset ampio disponibile (anni di dati 1h).

**Configurazione esempio**:
```
Model: diffusion-ddpm
Architecture: unet
Timesteps: 200
Horizon: 100
Learning Rate: 1e-4
Batch Size: 64
Model Channels: 128
```

### Caso 3: Portfolio Optimization (1d timeframe, multiple assets)

**Raccomandazione**: Diffusion (DDIM con Transformer)

**Motivo**: Necessità di distribuzione completa per calcolare correlazioni e VaR. DDIM è più veloce di DDPM, Transformer cattura cross-asset dependencies.

**Configurazione esempio**:
```
Model: diffusion-ddim
Architecture: transformer
Timesteps: 50 (DDIM accelerato)
Horizon: 30
Num Heads: 8
Learning Rate: 5e-5
```

---

## Workflow Consigliato

### 1. Fase di Esplorazione (1-2 giorni)

1. Inizia con **Ridge Regression** (baseline velocissima)
2. Prova **Random Forest** (cattura non-linearità senza tuning complesso)
3. Se performance insufficiente, passa a **Lightning** (neural network)

### 2. Fase di Ottimizzazione (3-7 giorni)

1. Usa **optimization: nsga2** per trovare best hyperparameters
2. Testa diverse combinazioni di features (indicatori multi-TF)
3. Valida su out-of-sample data (train/val/test split)

### 3. Fase Avanzata (2-4 settimane)

1. Se hai dataset massivo (>1M samples) e GPU, sperimenta **Diffusion**
2. Inizia con **DDPM** semplice (UNet, timesteps=100)
3. Se troppo lento, passa a **DDIM** (10x più veloce a parità di qualità)
4. Se serve catturare dipendenze lunghe, usa **Transformer** invece di UNet

---

## Metriche di Valutazione

### Per Supervised

- **MAE** (Mean Absolute Error): errore medio in pips
- **RMSE** (Root Mean Squared Error): penalizza outlier
- **R²** (Coefficient of Determination): % varianza spiegata
- **Directional Accuracy**: % di direzione corretta (up/down)

### Per Diffusion

- **CRPS** (Continuous Ranked Probability Score): misura qualità distribuzione
- **Calibration**: quanto la distribuzione predicted matcha la reale
- **Sharpness**: quanto stretta è la distribuzione (preferibile stretta ma calibrata)
- **Coverage**: % di valori reali dentro prediction interval (es. 95%)

---

## Conclusioni

### Quando Usare Supervised

✅ Produzione con latenza critica
✅ Dataset piccoli/medi (<100K samples)
✅ Previsioni short-term (1-20 steps)
✅ Interpretabilità richiesta
✅ Team piccolo con competenze ML standard

### Quando Usare Diffusion

✅ Ricerca e sperimentazione
✅ Dataset molto grandi (>1M samples)
✅ Previsioni long-term (50-500 steps)
✅ Risk management (scenari multipli)
✅ Team con competenze deep learning + GPU disponibile

### Approccio Ibrido (Best Practice)

🎯 **Usa entrambi**:
1. **Supervised** per predizioni rapide in produzione
2. **Diffusion** per generare scenari overnight per risk analysis
3. **Ensemble**: combina predizioni supervised (punto) con intervalli di confidenza da diffusion

---

## Risorse e Riferimenti

### Papers Fondamentali

- **Supervised**: Breiman (2001) "Random Forests", Zou & Hastie (2005) "ElasticNet"
- **Diffusion**: Ho et al. (2020) "Denoising Diffusion Probabilistic Models", Song et al. (2021) "DDIM"

### Tutorials

- Supervised: Scikit-learn documentation
- Diffusion: Hugging Face Diffusers library, Labml.ai tutorials

### Benchmark Datasets

- Forex: Dukascopy historical data, FXCM API
- Time Series: M4 Competition, Electricity, Traffic datasets

---

**Ultima modifica**: 2025-10-02
**Versione**: 1.0
**Autore**: ForexGPT Team
