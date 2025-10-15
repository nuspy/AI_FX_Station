# Fake/Placeholder Implementations - Elenco Completo

Generato: 2025-10-02
Scope: Identificare tutte le implementazioni fake/placeholder da completare

---

## 1. **Backtest - Adherence Metrics**
**File**: `src/forex_diffusion/backtest/db.py:204`
**Status**: Placeholder baseline adherence for RW
**Criticità**: MEDIA
**Descrizione**: Metriche di aderenza per baseline Random Walk non implementate
**Implementazione richiesta**: Calcolo CRPS, calibration, sharpness per RW baseline

---

## 2. **Backtest - Worker Adherence**
**File**: `src/forex_diffusion/backtest/worker.py:172-173`
**Status**: TODO compute adherence on slice
**Criticità**: MEDIA
**Descrizione**: Calcolo adherence su time slice non implementato, usa dati triviali
**Implementazione richiesta**: merge_asof alignment e calcolo metriche reali

---

## 3. **Backtesting - Multi-Horizon Model Type Dispatch**
**File**: `src/forex_diffusion/backtesting/multi_horizon_validator.py:346`
**Status**: Placeholder - depends on model type
**Criticità**: ALTA
**Descrizione**: Dispatch per type-specific validation non implementato
**Implementazione richiesta**: Switch su model type (supervised/diffusion/ensemble) con validation appropriata

---

## 4. **Backtesting - Pattern Benchmark Time Estimation**
**File**: `src/forex_diffusion/backtesting/pattern_benchmark_suite.py:636-638`
**Status**: Placeholder calculation (10ms per pattern fixed)
**Criticità**: BASSA
**Descrizione**: Stima tempo pattern detection usa valore fisso
**Implementazione richiesta**: Stima dinamica basata su profiling reale

---

## 5. **Backtesting - Double Top Detection**
**File**: `src/forex_diffusion/backtesting/pattern_benchmark_suite.py:754`
**Status**: Simple double top detection (placeholder)
**Criticità**: MEDIA
**Descrizione**: Algoritmo semplificato per double top, non production-ready
**Implementazione richiesta**: Algoritmo robusto con threshold dinamici, volume confirmation

---

## 6. **Backtesting - ML Pattern Logic**
**File**: `src/forex_diffusion/backtesting/pattern_benchmark_suite.py:783`
**Status**: ML-based logic (placeholder)
**Criticità**: ALTA
**Descrizione**: ML-based pattern detection non implementato
**Implementazione richiesta**: Modello ML (CNN/Transformer) per pattern recognition

---

## 7. **Inference - Conformal Calibration**
**File**: `src/forex_diffusion/inference/service.py:311`
**Status**: MVP - delta=0, no historical quantiles
**Criticità**: ALTA
**Descrizione**: Conformal prediction calibration non funzionale
**Implementazione richiesta**:
- Store historical predicted quantiles
- Implement weighted ICP (Inductive Conformal Prediction)
- Apply delta adjustment basato su calibration set

---

## 8. **Inference - Model Artifact Loading**
**File**: `src/forex_diffusion/inference/service.py:143`
**Status**: Not implemented
**Criticità**: MEDIA
**Descrizione**: Caricamento model artifact da storage non implementato
**Implementazione richiesta**: Load model from registry/storage con fallback

---

## 9. **Diffusion - DPM++ Sampler**
**File**: `src/forex_diffusion/models/diffusion.py:258`
**Status**: Heun-style integrator placeholder
**Criticità**: MEDIA
**Descrizione**: Sampler DPM++ è placeholder con Heun semplificato
**Implementazione richiesta**: Full DPM++ (2M/3M) sampler implementation

---

## 10. **Diffusion - Temporal UNet Architecture**
**File**: `src/forex_diffusion/models/diffusion.py:85`
**Status**: Placeholder for Temporal U-Net / DiT
**Criticità**: ALTA
**Descrizione**: Architettura placeholder, non ottimizzata per time series
**Implementazione richiesta**: Temporal UNet o DiT (Diffusion Transformer) architecture

---

## 11. **Patterns - Detector Base Class**
**File**: `src/forex_diffusion/patterns/engine.py:33`
**Status**: NotImplementedError on detect()
**Criticità**: BASSA (design pattern corretto)
**Descrizione**: Abstract base class, deve essere subclassato
**Implementazione richiesta**: NESSUNA (design intenzionale)

---

## 12. **Brokers - Interface Methods**
**File**: `src/forex_diffusion/services/brokers.py:28,30,32,93,158`
**Status**: Multiple NotImplementedError + placeholder PnL
**Criticità**: ALTA
**Descrizione**: Broker interface non implementata (IB, MT4/MT5)
**Implementazione richiesta**:
- `connect()`: Implement IB/MT client initialization
- `get_positions()`: Query real positions
- `place_order()`: Execute orders
- `get_pnl()`: Real-time P&L calculation
- Real client libraries integration (ib_insync, MetaTrader5)

---

## 13. **Model Service - Conformal TODO**
**File**: `src/forex_diffusion/services/model_service.py:297`
**Status**: TODO - integrate historical quantiles
**Criticità**: ALTA
**Descrizione**: Weighted ICP non integrato
**Implementazione richiesta**: Same as #7 (conformal calibration)

---

## 14. **Training - Loop NotImplemented**
**File**: `src/forex_diffusion/train/loop.py:125`
**Status**: NotImplementedError on step()
**Criticità**: BASSA (abstract method)
**Descrizione**: Abstract training loop, deve essere implementato in subclass
**Implementazione richiesta**: NESSUNA (design intenzionale)

---

## 15. **Training - Optimization Placeholders**
**File**: `src/forex_diffusion/training/optimization/engine.py:548,555`
**Status**: Two placeholders in optimization engine
**Criticità**: MEDIA
**Descrizione**: Optimization callbacks/metrics placeholder
**Implementazione richiesta**: Investigate context and implement proper metrics

---

## 16. **Training - Task Manager DB Models**
**File**: `src/forex_diffusion/training/optimization/task_manager.py:817`
**Status**: Placeholder classes for DB models
**Criticità**: BASSA
**Descrizione**: Dovrebbe importare da migration, usa placeholder temporanei
**Implementazione richiesta**: Import proper SQLAlchemy models from migrations

---

## 17. **UI - Chart Data Service TODO**
**File**: `src/forex_diffusion/ui/chart_components/services/data_service.py:143`
**Status**: TODO - Reimplement for PyQtGraph
**Criticità**: MEDIA
**Descrizione**: Migrazione da matplotlib a PyQtGraph incompleta
**Implementazione richiesta**: Port matplotlib code to PyQtGraph API

---

## 18. **UI - Enhanced Chart Placeholder**
**File**: `src/forex_diffusion/ui/chart_components/services/enhanced_chart_service.py:248`
**Status**: Simple line plot as placeholder
**Criticità**: BASSA
**Descrizione**: Chart enhancement placeholder
**Implementazione richiesta**: Full chart feature implementation

---

## 19. **UI - Pattern Detection Placeholder**
**File**: `src/forex_diffusion/ui/chart_components/services/enhanced_finplot_service.py:89,246,352`
**Status**: Multiple pattern detection placeholders
**Criticità**: MEDIA
**Descrizione**: Pattern detection UI non funzionale, simula detection
**Implementazione richiesta**: Integrate real pattern detection engine

---

## 20. **UI - Finplot Chart Adapter**
**File**: `src/forex_diffusion/ui/chart_components/services/finplot_chart_adapter.py:137-151`
**Status**: Placeholder QLabel instead of finplot chart
**Criticità**: MEDIA
**Descrizione**: Finplot non integrato, mostra solo label
**Implementazione richiesta**: Real finplot chart widget integration

---

## 21. **UI - Drawing Tools TODO**
**File**: `src/forex_diffusion/ui/chart_components/services/interaction_service.py:40,52`
**Status**: TODO - Implement drawing tools
**Criticità**: BASSA
**Descrizione**: Drawing tools (trendlines, etc.) non implementati
**Implementazione richiesta**: Implement finplot drawing API

---

## 22. **UI - Pattern Background Processing**
**File**: `src/forex_diffusion/ui/chart_components/services/patterns/patterns_service.py:746`
**Status**: TODO - Implement proper background processing
**Criticità**: MEDIA
**Descrizione**: Pattern detection non usa background worker
**Implementazione richiesta**: QThreadPool worker for async pattern detection

---

## 23. **UI - Manual Overlay TODO**
**File**: `src/forex_diffusion/ui/chart_components/services/plot_service.py:1969`
**Status**: TODO: Inserire manualmente
**Criticità**: BASSA
**Descrizione**: Manual insertion placeholder
**Implementazione richiesta**: Completare logica manual overlay

---

## 24. **UI - Overlay Decompression**
**File**: `src/forex_diffusion/ui/chart_tab/overlay_manager.py:112`
**Status**: Need decompression method - placeholder
**Criticità**: BASSA
**Descrizione**: Decompressione overlay non implementata
**Implementazione richiesta**: Implement decompression for compressed overlays

---

## 25. **UI - Viewer Plot Placeholder**
**File**: `src/forex_diffusion/ui/viewer.py:18,20`
**Status**: No-op placeholder, no rendering
**Criticità**: BASSA
**Descrizione**: Chart viewer non renderizza
**Implementazione richiesta**: Real chart rendering logic

---

## 26. **Utils - Logging Placeholder**
**File**: `src/forex_diffusion/utils/logging.py:45`
**Status**: Placeholder for loguru logger object
**Criticità**: BASSA (commento documentazione)
**Descrizione**: Comment placeholder, not actual code issue
**Implementazione richiesta**: NESSUNA

---

## Riepilogo per Criticità

### ALTA (Action Required) - 7 items
3. Multi-Horizon Model Type Dispatch
6. ML Pattern Logic
7. Conformal Calibration (Inference)
10. Temporal UNet Architecture
12. Broker Interface Methods
13. Model Service Conformal

### MEDIA (Important but not blocking) - 11 items
1. Backtest Adherence Metrics
2. Backtest Worker Adherence
4. Pattern Benchmark Time Estimation
5. Double Top Detection
8. Model Artifact Loading
9. DPM++ Sampler
15. Optimization Engine Placeholders
17. Chart Data Service PyQtGraph
19. Pattern Detection Placeholders
20. Finplot Chart Adapter
22. Pattern Background Processing

### BASSA (Nice to have) - 8 items
11. Patterns Detector (intentional abstract)
14. Training Loop (intentional abstract)
16. Task Manager DB Models
18. Enhanced Chart Placeholder
21. Drawing Tools
23. Manual Overlay TODO
24. Overlay Decompression
25. Viewer Plot
26. Logging Placeholder (comment only)

---

## Priorità Implementazione Consigliata

### Fase 1 (Immediate - Blocking Production)
1. **Conformal Calibration** (#7, #13) - Criticalfor forecast credibility
2. **Broker Integration** (#12) - Needed for live trading
3. **Multi-Horizon Validation** (#3) - Needed for model selection

### Fase 2 (Short Term - Enhance Quality)
4. **ML Pattern Detection** (#6) - Differentiate product
5. **Temporal UNet** (#10) - Improve diffusion model quality
6. **Pattern Detection UI** (#19) - User-facing feature

### Fase 3 (Medium Term - Polish)
7. **DPM++ Sampler** (#9) - Faster inference
8. **Finplot Integration** (#20) - Better charts
9. **Adherence Metrics** (#1, #2) - Better backtesting

### Fase 4 (Long Term - Nice to Have)
10. Drawing Tools (#21)
11. Background Processing (#22)
12. Minor UI improvements (#17, #18, #23-25)

---

**Note**: Molti "placeholder" sono solo QLineEdit.setPlaceholderText() per UI hints - questi NON sono fake implementations, solo testo di aiuto all'utente.

**Total Fake Implementations**: 26
**Critical (must fix)**: 7
**Important (should fix)**: 11
**Low priority**: 8
