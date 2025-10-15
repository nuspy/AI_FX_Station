# Changelog: Training UI & Model Enhancements

**Data**: 2025-10-02
**Versione**: 2.0
**Autore**: Claude Code (Autonomous Session)

---

## Modifiche Completate

### 1. Documento Modelli: `docs/MODELS_COMPARISON.md`

Creato documento completo che spiega:
- Differenze tra Supervised Models e Diffusion Models
- Vantaggi/svantaggi di ciascun approccio
- Quando usare supervised vs diffusion
- Metriche di valutazione appropriate
- Esempi pratici per day trading, swing trading, portfolio optimization
- Workflow raccomandato (esplorazione → ottimizzazione → produzione)
- Tabelle comparative dettagliate
- Riferimenti a papers scientifici

### 2. Training Tab (`src/forex_diffusion/ui/training_tab.py`)

#### 2.1 Nuove Funzionalità

**Model Name Field**:
- Campo testo per specificare nome custom del modello
- Se vuoto: auto-generazione da symbol_timeframe_model_features
- Tooltip dettagliato che spiega uso e best practices

**Load/Save Config (JSON)**:
- Pulsante "📂 Load Config": carica configurazione completa da JSON
- Pulsante "💾 Save Config": salva configurazione attuale in JSON
- Include tutti i parametri: features, indicatori, hyperparameters, diffusion params
- Metadata: config_version, created_at timestamp
- File dialog con filtro `.json`
- Gestione errori con messaggi informativi

#### 2.2 Parametri Diffusion Models Aggiunti

Nuova sezione "Diffusion Model Parameters" in Advanced Params:

1. **Diffusion timesteps** (10-5000, default 200)
   - Numero di steps per denoising process
   - DDPM: 1000 steps (alta qualità)
   - DDIM: 50-200 steps (10x più veloce)

2. **Learning rate** (1e-6 - 1e-1, default 1e-4)
   - Step size gradient descent
   - Critico per stabilità training

3. **Batch size (DL)** (4-512, default 64)
   - Samples per batch per deep learning models
   - Separato da "Lightning batch" (supervised)

4. **Model channels** (32-512, default 128)
   - Capacità UNet/Transformer
   - Determina numero parametri (≈channels² × layers)

5. **Dropout** (0.0-0.6, default 0.1)
   - Regolarizzazione anti-overfitting
   - 0.0 = no regularization, 0.3 = forte regularization

6. **Attention heads** (1-16, default 8)
   - Numero teste per Transformer architecture
   - Vincolo: model_channels % num_heads == 0

#### 2.3 Tooltips Dettagliati Aggiunti

**TUTTI i controlli ora hanno tooltips che spiegano**:
- Cosa è il parametro
- Perché è importante
- Cosa succede con valori bassi
- Cosa succede con valori alti
- Best practices e raccomandazioni
- Esempi concreti

**Esempi di tooltip migliorati**:

**Symbol**:
```
Coppia valutaria da usare per il training.
Cosa è: asset finanziario su cui addestrare il modello predittivo.
Perché è importante: ogni coppia ha caratteristiche uniche (volatilità, correlazioni).
Best practice: addestra modelli separati per ciascuna coppia (no mixing).

EUR/USD: coppia più liquida, spread bassi, volatilità media.
GBP/USD: alta volatilità, buona per swing trading.
Exotic pairs (AUD/JPY, etc.): spread alti, pattern diversi.
```

**Days**:
```
Numero di giorni storici da usare per l'addestramento.
Cosa è: quanti giorni passati includere nel dataset di training.
Perché è importante: più dati = migliore generalizzazione, ma training più lento.
Valori bassi (1-7): training veloce, rischio overfitting, buono per test rapidi.
Valori medi (30-90): bilanciamento speed/quality, uso standard.
Valori alti (365-1825): migliore generalizzazione, cattura cicli stagionali, lento.
Best practice: almeno 1000 samples per feature (es. 100 features → 7 giorni su 1m).

1-7: test rapido, pochi dati, rischio overfitting.
30-90: standard, bilanciamento qualità/velocità.
365+: massima qualità, cattura stagionalità, training lungo (ore).
Nota: con TF=1m, 7 giorni ≈ 10K samples. Con TF=1h, 7 giorni ≈ 168 samples.
```

**Model Combo** (ora include diffusion):
```
Algoritmi disponibili:

SUPERVISED (veloce, interpretabile, short-term):
• ridge: regressione lineare L2, velocissimo, baseline ottimo.
• lasso: regressione lineare L1, feature selection automatica.
• elasticnet: combina ridge+lasso, bilanciamento L1/L2.
• rf: Random Forest, cattura non-linearità, robusto, lento.
• lightning: neural network (MLP/LSTM), molto flessibile, richiede GPU.

DIFFUSION (lento, generativo, long-term, incertezza):
• diffusion-ddpm: Denoising Diffusion Probabilistic Model, alta qualità.
• diffusion-ddim: DDIM (deterministic), 10x più veloce di DDPM.

Raccomandazioni:
- Test rapido: ridge (secondi)
- Production: ridge/rf (minuti, interpretabile)
- Ricerca: lightning/diffusion (ore, GPU consigliata)
- Long-term forecast: diffusion-ddim (genera scenari multipli)
```

**Optimization**:
```
Metodi di ottimizzazione automatica:

• none: usa parametri di default, training veloce (1x).
  Quando: test rapido, parametri già noti.

• genetic-basic: algoritmo genetico single-objective.
  Cosa ottimizza: solo MAE (errore).
  Pro: semplice, funziona bene.
  Contro: ignora trade-off (es. accuracy vs complexity).
  Tempo: ~5-20x più lento di 'none' (dipende da gen×pop).
  Quando: vuoi best accuracy, hai tempo.

• nsga2: NSGA-II multi-objective optimization.
  Cosa ottimizza: MAE + complessità + robustezza.
  Pro: trova Pareto front, bilanciamento obiettivi.
  Contro: molto lento, complesso interpretare.
  Tempo: ~10-50x più lento di 'none'.
  Quando: ricerca avanzata, vuoi trade-off espliciti.

Nota: gen=10, pop=20 → 200 training runs → 200x tempo!
```

#### 2.4 Persistenza Configurazioni

**Aggiornato `_save_settings()` e `_load_settings()`**:
- Includono tutti i nuovi parametri (model_name, diffusion params)
- Salvataggio automatico on tab change/close
- Ripristino automatico all'apertura

### 3. Bug Fixes

#### 3.1 `train_sklearn.py` - AttributeError 'days'
**Errore**: `args.days` non esisteva, causava crash
**Fix**: Cambiato a `args.days_history` (line 363)

#### 3.2 `train_sklearn.py` - TypeError: cannot unpack non-iterable NoneType
**Errore**: `_build_features()` non ritornava valore se `min_cov <= 0`
**Fix**: De-indentato il blocco `X.dropna()` ... `return X, y, meta` (lines 380-392)

#### 3.3 PyTorch/torchvision Compatibility
**Errore**: `torch.library.register_fake` AttributeError (torch 2.2.2 + torchvision 0.22.1)
**Fix**: Upgrade a torch 2.8.0 + torchvision 0.23.0 + torchaudio 2.8.0

### 4. Model Types Support

Training tab ora supporta esplicitamente:
- **Supervised Models**: ridge, lasso, elasticnet, rf, lightning
- **Diffusion Models**: diffusion-ddpm, diffusion-ddim

Dropdown model aggiornato con tutti i tipi disponibili.

---

## Modifiche Pending (Non Completate)

### 5. Prediction Dialog Sync

**TODO**: Aggiornare `unified_prediction_settings_dialog.py` con:
- Load/Save config JSON
- Parametri diffusion (se model loaded è diffusion-*)
- Tooltips dettagliati consistenti con training tab
- Model name display

### 6. Date Range Instead of Days

**TODO** (richiesto dall'utente):
- Rimuovere campo "Days"
- Aggiungere campi "Train Start Date" e "Train End Date"
- Modificare train_sklearn.py per accettare date invece di days_history
- Aggiornare tutti i riferimenti a `days` / `days_history`

### 7. Prediction Settings Updates

**TODO**: Sincronizzare prediction dialog con training tab per consistenza UX

---

## Test Raccomandati

Prima di rilasciare in produzione:

1. **Test Load/Save Config**:
   - Salvare config con tutti parametri settati
   - Chiudere app
   - Riaprire e caricare config
   - Verificare che TUTTI i parametri siano ripristinati correttamente

2. **Test Training Ridge (quick)**:
   ```
   Symbol: EUR/USD
   TF: 1h
   Days: 7
   Horizon: 5
   Model: ridge
   Opt: none
   ```
   - Dovrebbe completare in <1 min
   - Verificare che modello sia salvato con nome corretto

3. **Test Training RF (con optimization)**:
   ```
   Symbol: EUR/USD
   TF: 1h
   Days: 30
   Horizon: 10
   Model: rf
   Opt: genetic-basic
   Gen: 5
   Pop: 8
   ```
   - Dovrebbe completare in ~30-60 min
   - Verificare 40 training runs (5×8)
   - Verificare best model selection

4. **Test Diffusion Params Visibility**:
   - Selezionare model=diffusion-ddpm
   - Verificare che parametri diffusion siano visibili e attivi
   - Selezionare model=ridge
   - Parametri diffusion ignorati ma visibili (non nascosti)

5. **Test Tooltips**:
   - Hover su OGNI controllo
   - Verificare tooltip mostri info dettagliate
   - Nessun tooltip deve essere vuoto o generico

---

## Statistiche Modifiche

- **File modificati**: 3
  - `src/forex_diffusion/ui/training_tab.py` (+400 righe, tooltips completi)
  - `src/forex_diffusion/training/train_sklearn.py` (2 bug fixes)
  - `docs/MODELS_COMPARISON.md` (nuovo, ~500 righe)

- **Nuove funzionalità**: 9
  1. Model Name field
  2. Load Config JSON
  3. Save Config JSON
  4. Diffusion timesteps param
  5. Learning rate param
  6. Batch size DL param
  7. Model channels param
  8. Dropout param
  9. Attention heads param

- **Tooltips aggiunti/migliorati**: 30+
  - Tutti i top controls (symbol, tf, days, horizon, model, encoder, opt, gen, pop)
  - Tutti i parametri avanzati
  - Tutti i parametri diffusion
  - Tutte le additional features

- **Bug fixes**: 3
  - PyTorch compatibility
  - args.days AttributeError
  - _build_features indentation TypeError

---

## Note Implementazione

### Scelte Design

1. **Diffusion params sempre visibili**: Non nascosti quando model != diffusion, per evitare confusione (cosa è nascosto?). Semplicemente ignorati se model supervised.

2. **Config JSON vs Settings Persistence**:
   - Settings persistence (auto-save): mantiene UX session-to-session
   - Config JSON (manual): per condivisione, A/B testing, reproducibility

3. **Tooltips ultra-dettagliati**: User potrebbe essere principiante ML, quindi tooltip spiega concetti base + advanced insieme.

4. **Model name optional**: Se vuoto, auto-generation evita nomi troppo lunghi ma mantiene leggibilità.

### Future Enhancements Suggerite

1. **Preset Configs**: Bottoni per caricare config predefinite (es. "Day Trading 1m", "Swing Trading 1h", "Research Diffusion")

2. **Config Comparison**: Tool per confrontare 2 config JSON side-by-side

3. **Training History**: Log di tutti i training run con timestamp, parametri, performance

4. **Model Registry**: Database locale di tutti i modelli trained con metadata searchable

5. **Auto-Tuning**: Integration con Optuna/Hyperopt per tuning automatico avanzato (oltre genetic)

6. **Distributed Training**: Support per training multi-GPU o distributed (Ray, Horovod)

7. **MLflow Integration**: Tracking automatico experiments con MLflow

---

**Session completata autonomamente durante sleep dell'utente**
**Status**: Training UI v2.0 pronto per testing
**Next**: Prediction dialog sync + date range implementation
