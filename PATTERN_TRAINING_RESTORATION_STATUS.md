# Pattern Training/Backtest Tab - Restoration Status

## ✅ Completato

### 1. Ristrutturazione UI
- ✅ Tab UNO rinominato in "Generative Forecast"
- ✅ Rimossi bottoni "Settings" e "Adv Settings" dal Chart
- ✅ Riattivato bottone "Config" per i pattern (ora funzionante con PatternsConfigDialog)
- ✅ Tab Patterns spostato sotto DUE come tab di secondo livello
- ✅ Tab Logs spostato a primo livello (era secondo livello in Chart)
- ✅ Controlli pattern (Chart/Candle/History) rimossi dalla toolbar Chart e spostati in DUE > Patterns

### 2. Nuova Struttura UI Finale
```
Primo Livello:
├── Chart
│   └── Secondo livello:
│       ├── Chart (grafico principale)
│       └── Training/Backtest ← RIPRISTINATO (pattern training)
├── Generative Forecast
│   └── Secondo livello:
│       ├── Training (modelli ML)
│       ├── Forecast Settings (Base + Advanced Settings)
│       └── Backtesting (modelli ML)
├── DUE
│   └── Secondo livello:
│       └── Patterns (detection e configurazione)
├── Logs (monitoring e system logs)
├── Signals (Temp)
└── 3D Reports (Temp)
```

### 3. File Creati/Modificati

**File Nuovi:**
- `src/forex_diffusion/ui/pattern_training_tab.py` - Tab standalone per training pattern
- `src/forex_diffusion/ui/patterns_tab.py` - Tab per pattern detection (sotto DUE)
- `src/forex_diffusion/ui/logs_tab.py` - Tab logs di primo livello
- `src/forex_diffusion/ui/forecast_settings_tab.py` - Forecast settings embedded

**File Modificati:**
- `src/forex_diffusion/ui/app.py` - Nuova struttura tab
- `src/forex_diffusion/ui/chart_tab/ui_builder.py` - Integrato PatternTrainingTab
- `src/forex_diffusion/ui/chart_tab/event_handlers.py` - Rimossi handler forecast settings
- `src/forex_diffusion/ui/chart_tab/patterns_mixin.py` - Collegato Config button
- `src/forex_diffusion/utils/device_manager.py` - Rimossi emoji (fix encoding Windows)
- `src/forex_diffusion/training/encoders.py` - Rimossi emoji GPU logs

## 🚧 In Corso

### Pattern Training Tab - Implementazione Parziale

**Stato Attuale:**
Il tab `pattern_training_tab.py` è stato creato come skeleton con:
- ✅ Struttura UI completa (6 sezioni)
- ✅ Integrato come tab di secondo livello in Chart
- ✅ Placeholder per tutti i controlli
- ⏳ Implementazione UI completa da commit 11d3627 (in sospeso)
- ⏳ Metodi di training execution (in sospeso)

**Sezioni Create (placeholder):**
1. Training Setup - Selezione Chart/Candlestick patterns, periodo, assets, timeframes
2. Optimization Targets (D1/D2) - Multi-objective optimization
3. Parameter Space - Form parameters e Action parameters
4. GA Configuration - Popolazione, generazioni, trials
5. Execution Control - Start/Pause/Resume/Stop + progress tracking
6. Results - Status, best parameters, performance analysis

**Codice Sorgente Disponibile:**
- Commit: `11d3627` - "Training parametri patterns"
- File: `/tmp/chart_tab_ui_with_training.py` (2769 righe)
- Sezioni training: righe 277-2770 (~2500 righe di implementazione)

## 📋 Prossimi Passi

### Fase 1: Completare UI del Training Tab
1. **Popolare `_create_study_setup_section()`**
   - Radio buttons Chart vs Candlestick patterns
   - Date pickers (training start/end)
   - Asset selection (text edit)
   - Timeframe selection (text edit)
   - Info label con features automatiche

2. **Popolare `_create_dataset_config_section()`**
   - D1/D2 explanation label
   - Multi-objective checkbox
   - D1 frame (profit maximization)
   - D2 frame (risk minimization)
   - Strategy preference combo
   - D1 weight slider

3. **Popolare `_create_parameter_space_section()`**
   - Form parameters group:
     * Min touches (spinboxes)
     * Confidence threshold (double spinboxes)
     * Pattern tolerance (double spinboxes)
     * ATR period (spinboxes)
   - Action parameters group:
     * Target modes (text edit con lista completa)
     * Stop loss modes (text edit)
     * Trail modes (text edit)
     * Confirmation indicators (text edit)

4. **Popolare `_create_optimization_config_section()`**
   - Population size spinbox
   - Generations spinbox
   - Trials per pattern spinbox
   - GA note label

5. **Completare `_create_execution_control_section()`**
   - Progress bars già presenti
   - Aggiungere time estimates:
     * Elapsed time label
     * Remaining time label
     * Completion ETA label
     * Trials per minute label
     * Best score label
     * Worker status label
   - Auto-refresh checkbox

### Fase 2: Implementare Training Execution
1. **Estrai metodi dal commit 11d3627:**
   - `_start_optimization()` - line 2167
   - `_pause_optimization()` - line 2174
   - `_resume_optimization()` - line 2184
   - `_resume_interrupted_training()` - line 2194
   - `_stop_optimization()` - line 2597
   - `_start_pattern_family_training()` - line 2088
   - `_start_comprehensive_training()` - line 2129
   - `_execute_pattern_family_training()` - line 2469
   - `_run_genetic_optimization()` - line 2539

2. **Implementa progress tracking:**
   - `_refresh_progress()` - Update progress bars
   - `_refresh_status()` - Update status text
   - `_update_training_ui_state()` - line 2330

3. **Implementa parameter management:**
   - `_promote_parameters()` - line 2633
   - `_rollback_parameters()` - line 2649
   - `_collect_optimization_config()` - line 2665
   - `_validate_training_config()` - line 2287

4. **Implementa helpers:**
   - `_estimate_training_time()` - line 2316
   - `_create_individual_study_config()` - line 2523
   - `_collect_comprehensive_training_config()` - line 2224

### Fase 3: Integrare con Genetic Algorithm
1. Verificare esistenza di `src/forex_diffusion/training/optimization/genetic_algorithm.py`
2. Importare `GAConfig` e classi correlate
3. Collegare UI con engine di ottimizzazione
4. Implementare callback per progress updates

### Fase 4: Testing
1. Test UI rendering
2. Test validation configurazione
3. Test start/pause/resume training
4. Test parameter promotion/rollback
5. Test risultati e performance display

## 🔧 Come Procedere

### Opzione A: Copia Manuale Sezioni
Estrarre manualmente ogni metodo dal file temporaneo e copiarlo in `pattern_training_tab.py`:
```bash
# File disponibile:
/tmp/chart_tab_ui_with_training.py

# Sezioni da copiare:
# - Righe 277-377: _create_study_setup_section
# - Righe 378-478: _create_dataset_config_section
# - Righe 479-647: _create_parameter_space_section
# - Righe 648-751: _create_optimization_config_section
# - Righe 752-877: _create_execution_control_section (parziale)
# - Righe 2088-2770: Training execution methods
```

### Opzione B: Script di Estrazione
Creare uno script Python che estrae automaticamente i metodi dal commit:
```python
import subprocess

# Estrai metodi specifici dal commit
methods_to_extract = [
    '_create_study_setup_section',
    '_create_dataset_config_section',
    '_create_parameter_space_section',
    # ... etc
]

for method in methods_to_extract:
    # Extract using git show and sed
    pass
```

### Opzione C: Cherry-pick Commit (Sconsigliato)
Potrebbe causare conflitti con i cambiamenti recenti. Meglio estrazione manuale.

## 📝 Note Tecniche

### Dipendenze da Verificare:
- `src/forex_diffusion/training/optimization/genetic_algorithm.py` (menzionato nel commit)
- `src/forex_diffusion/training/optimization/task_manager.py` (modificato nel commit)
- `GAConfig` class
- Pattern registry e boundary config

### Modifiche UI Necessarie:
Il codice originale usava `self` (riferimento a ChartTabUI). Nel nuovo file standalone dovremmo:
1. Mantenere riferimenti a widget come variabili di istanza
2. Implementare signal/slot connections separate
3. Gestire comunicazione con chart_controller se necessario

### Compatibilità:
Il codice è del 28 Settembre 2025, quindi relativamente recente. Dovrebbe essere compatibile con la codebase attuale.

## 📊 Stima Lavoro

- **Fase 1 (UI completa)**: 2-3 ore
- **Fase 2 (Training logic)**: 3-4 ore
- **Fase 3 (GA integration)**: 1-2 ore
- **Fase 4 (Testing)**: 1-2 ore

**Totale**: 7-11 ore di lavoro

## 🎯 Obiettivo Finale

Un tab completamente funzionale per:
1. Configurare training di pattern (chart e candlestick)
2. Ottimizzare parametri con algoritmo genetico
3. Multi-objective optimization (profitti vs rischi)
4. Monitorare progress in real-time
5. Gestire parametri (promote/rollback)
6. Analizzare risultati e performance
