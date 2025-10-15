# ForexGPT - Pattern Recognition System: Complete Workflow

**Version**: 2.0.0  
**Last Updated**: 2025-10-13  
**Status**: Production

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Pattern Detection Workflows](#pattern-detection-workflows)
4. [Pattern Types & Implementations](#pattern-types--implementations)
5. [Real-Time Pattern Scanning](#real-time-pattern-scanning)
6. [Historical Pattern Optimization](#historical-pattern-optimization)
7. [Pattern Confidence & Calibration](#pattern-confidence--calibration)
8. [Multi-Timeframe Analysis](#multi-timeframe-analysis)
9. [DOM Confirmation](#dom-confirmation)
10. [Parameter Selection & Optimization](#parameter-selection--optimization)
11. [Complete Parameter Reference](#complete-parameter-reference)
12. [Workflow Diagrams](#workflow-diagrams)

---

## Executive Summary

The **Pattern Recognition System** is ForexGPT's comprehensive chart pattern detection and analysis engine. It provides real-time and historical pattern scanning with advanced features:

**Key Components**:
- **30+ Pattern Detectors**: Chart patterns, candlestick patterns, harmonic patterns, Elliott Wave
- **2 Scanning Modes**: Real-time (10-second intervals) and Historical (multi-timeframe optimization)
- **3 Confirmation Systems**: Confidence calibration, DOM confirmation, multi-timeframe confluence
- **NSGA-II Optimization**: Multi-objective genetic algorithm for parameter optimization
- **Progressive Formation**: Detect patterns at 60%+ confidence (forming stage)
- **Database-Driven Parameters**: Optimal parameters per asset/timeframe/regime

**Architecture**:
- **Multi-threaded**: Separate threads for chart patterns, candle patterns, and detection batches
- **Cached**: Redis + LRU caching for performance
- **Configurable**: YAML-based pattern boundaries and parameters
- **Extensible**: Easy to add new patterns

---

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     GUI Layer (PySide6)                           │
│  Pattern Overlay | Patterns Tab | Config Dialog | Training Tab   │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                  PatternsService (Main Orchestrator)              │
│  - Real-time scanning (chart/candle threads)                     │
│  - Historical scanning (multi-timeframe worker)                  │
│  - Detection coordination (batch processing)                     │
│  - Cache management (Redis + LRU)                                │
└────────────────────────┬─────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┬─────────────────┐
         │               │               │                 │
┌────────▼────────┐ ┌────▼──────┐ ┌─────▼─────┐ ┌────────▼────────┐
│ Pattern Registry│ │Scan Workers│ │Detectors  │ │Optimization Eng.│
│ (30+ patterns)  │ │(Background)│ │(30+ impl.)│ │(NSGA-II/GA)     │
└────────┬────────┘ └────┬──────┘ └─────┬─────┘ └────────┬────────┘
         │               │               │                 │
         └───────────────┴───────────────┴─────────────────┘
                                 │
                ┌────────────────▼───────────────────┐
                │  Support Systems                   │
                │  - Confidence Calibrator           │
                │  - Strength Calculator             │
                │  - DOM Confirmation                │
                │  - Multi-Timeframe Analyzer        │
                │  - Progressive Formation Detector  │
                │  - Boundary Config                 │
                │  - Parameter Selector (DB-driven)  │
                └────────────────────────────────────┘
```

### Component Matrix

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| **PatternRegistry** | patterns/registry.py | Central pattern catalog | ✅ Production |
| **PatternsService** | ui/chart_components/services/patterns/patterns_service.py | Main orchestration | ✅ Production |
| **ScanWorker** | ui/chart_components/services/patterns/scan_worker.py | Background scanning | ✅ Production |
| **DetectionWorker** | ui/chart_components/services/patterns/detection_worker.py | Batch detection | ✅ Production |
| **ProgressiveFormation** | patterns/progressive_formation.py | Formation tracking | ✅ Production |
| **ConfidenceCalibrator** | patterns/confidence_calibrator.py | Historical calibration | ✅ Production |
| **StrengthCalculator** | patterns/strength_calculator.py | Pattern strength | ✅ Production |
| **DOMConfirmation** | patterns/dom_confirmation.py | Order book validation | ✅ Production |
| **MultiTimeframeAnalyzer** | patterns/multi_timeframe.py | TF confluence | ✅ Production |
| **ParameterSelector** | patterns/parameter_selector.py | DB-driven params | ✅ Production |
| **BoundaryConfig** | patterns/boundary_config.py | Pattern boundaries | ✅ Production |
| **OptimizationEngine** | training/optimization/engine.py | NSGA-II optimization | ✅ Production |

---

## Pattern Detection Workflows

### Workflow 1: Real-Time Pattern Detection

**Trigger**: Every 10 seconds (configurable) or chart update

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. Trigger Event                                                  │
│    - Timer tick (every 10s)                                       │
│    - Chart update (manual)                                        │
│    - New candle received                                          │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 2. Resource Check                                                 │
│    - Check CPU usage (< 30% threshold)                           │
│    - Check memory usage (< 80% threshold)                        │
│    - Check if market is open                                     │
│    → If constrained: Increase interval, skip scan               │
│    → If market closed: Increase interval to 5+ minutes           │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 3. Data Preparation                                               │
│    - Get current dataframe from chart                            │
│    - Apply pattern-specific boundaries:                          │
│      * Fast patterns (candles): 30-50 candles                    │
│      * Medium patterns (chart): 80-200 candles                   │
│      * Slow patterns (Elliott/Harmonic): 300-500 candles         │
│    - Check timeframe (1m, 5m, 15m, 1h, 4h, 1d)                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 4. Pattern Registry Query                                         │
│    registry = PatternRegistry()                                  │
│    detectors = registry.detectors(kinds=['chart', 'candle'])    │
│    → Returns 30+ pattern detectors                               │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 5. Batch Detection (Background Thread)                           │
│    For each detector batch (size=5):                             │
│      1. Apply boundary to dataframe                              │
│         df_limited = df.tail(boundary_candles)                   │
│      2. Run detector                                             │
│         events = detector.detect(df_limited)                     │
│      3. Emit progress (0-100%)                                   │
│      4. Yield to GUI (sleep 1ms)                                 │
│    → Returns List[PatternEvent]                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 6. Pattern Enrichment                                            │
│    For each detected pattern:                                    │
│      A. Calculate Strength:                                      │
│         - Raw confidence (detector output)                       │
│         - Volatility score (ATR-based)                          │
│         - Historical success rate (from DB)                     │
│         - Price action quality                                  │
│         → Combined score + Star rating (1-5)                    │
│      B. Calibrate Confidence:                                   │
│         - Get historical win rate for pattern type             │
│         - Apply adjustment factor (0.5-1.5x)                   │
│         → Calibrated confidence                                 │
│      C. DOM Confirmation (if available):                        │
│         - Check order book imbalance                            │
│         - Bullish pattern → expect bid > ask                   │
│         - Bearish pattern → expect ask > bid                   │
│         → Adjusted score ± 20%                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 7. Progressive Formation Check                                    │
│    progressive_detector.update_patterns(events)                  │
│    For each pattern:                                             │
│      - Determine formation stage:                                │
│        * Early: < 40% confidence                                 │
│        * Developing: 40-60%                                      │
│        * Forming: 60-80% (show with dashed lines)               │
│        * Mature: 80-95%                                          │
│        * Completed: > 95%                                        │
│      - Track formation progress                                  │
│      - Estimate completion time                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 8. Multi-Timeframe Confluence (Optional)                         │
│    If enabled:                                                   │
│      - Check patterns on 1m, 5m, 15m, 1h                        │
│      - Calculate alignment:                                      │
│        * Bullish confluence: All timeframes bullish             │
│        * Bearish confluence: All timeframes bearish             │
│        * Mixed signals: Conflicting directions                  │
│      - Boost confidence if confluence detected                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 9. Filter & Sort                                                  │
│    - Filter patterns below minimum confidence (default: 60%)     │
│    - Sort by combined score (strength × confidence)             │
│    - Limit to top N patterns per direction (default: 5)         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 10. Cache Results                                                 │
│     - Redis cache: key = f"{symbol}_{timeframe}_patterns"       │
│     - LRU cache: in-memory for fast access                      │
│     - TTL: 30 minutes                                            │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 11. Draw on Chart                                                 │
│     For each pattern:                                            │
│       - Draw pattern boundaries (rectangles, trendlines)        │
│       - Add confidence badge (1-5 stars)                        │
│       - Add labels (pattern name, direction, confidence)        │
│       - Show target/stop levels (if available)                  │
│       - Use dashed lines for forming patterns (< 80%)           │
└──────────────────────────────────────────────────────────────────┘
```

**Key Features**:
- **Non-blocking**: Detection runs in background thread
- **Adaptive Intervals**: Increases during market closure or high CPU
- **Boundary-Aware**: Each pattern limited to relevant candle count
- **Multi-Confirmation**: Strength + Calibration + DOM + Multi-TF

---

### Workflow 2: Historical Pattern Optimization

**Trigger**: User clicks "Scan Historical" button

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. User Configuration                                             │
│    - Select pattern type (e.g., "head_shoulders")               │
│    - Choose direction (bull/bear/both)                           │
│    - Set asset & timeframe                                       │
│    - Optional: Select regime (trending/ranging)                  │
│    - Define date range (e.g., "last 6 months")                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 2. Parameter Space Definition                                     │
│    ParameterSpace.define_for_pattern(pattern_type)              │
│    Example for Head & Shoulders:                                │
│      - shoulder_tolerance: (0.02, 0.10, 'float')                │
│      - neck_angle_tolerance: (5.0, 30.0, 'float')               │
│      - head_prominence: (1.05, 1.30, 'float')                   │
│      - symmetry_ratio: (0.7, 1.0, 'float')                      │
│      - min_bars: (20, 100, 'int')                               │
│    → Returns parameter ranges for optimization                   │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 3. Historical Data Loading                                        │
│    - Fetch OHLCV data for date range                            │
│    - Load volume data (if available)                            │
│    - Load DOM data (if available)                               │
│    - Apply timeframe resampling if needed                       │
│    → DataFrame with full historical data                         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 4. Walk-Forward Split                                             │
│    - Training period: First 60% of data                         │
│    - Validation period: Next 20% of data                        │
│    - Test period: Last 20% of data                              │
│    - Apply purge (1 day) and embargo (2 days)                  │
│    → Train/Val/Test splits                                       │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 5. Optimization Method Selection                                  │
│    User chooses:                                                 │
│      A. Grid Search: Exhaustive parameter combinations          │
│      B. Genetic Algorithm (GA): Single-objective (max profit)   │
│      C. NSGA-II: Multi-objective (profit vs. risk)              │
│    → Most common: NSGA-II for Pareto frontier                   │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 6. NSGA-II Optimization Loop                                      │
│    config = OptimizationConfig(                                 │
│        pattern_key=pattern_type,                                │
│        asset=asset,                                              │
│        timeframe=timeframe,                                      │
│        max_trials=1000,                                          │
│        max_parallel_workers=32                                   │
│    )                                                             │
│                                                                   │
│    engine = OptimizationEngine(db_service, data_root)          │
│    study = engine.create_study(config)                          │
│                                                                   │
│    For generation in range(max_generations):                    │
│      1. Generate population (N=100 individuals)                 │
│      2. Parallel evaluation (32 workers):                       │
│         For each parameter set:                                 │
│           a. Create detector with parameters                    │
│           b. Run detection on training data                     │
│           c. Backtest detected patterns:                        │
│              - Calculate returns (actual move vs. target)       │
│              - Track success rate (hit target vs. hit stop)    │
│              - Calculate Sharpe ratio                          │
│              - Measure max drawdown                            │
│           d. Compute objective scores:                          │
│              - Objective 1: Maximize total return              │
│              - Objective 2: Minimize max drawdown              │
│              - Objective 3: Maximize success rate              │
│      3. Non-dominated sorting (Pareto ranking)                 │
│      4. Crowding distance calculation                          │
│      5. Selection (tournament)                                  │
│      6. Crossover (blend/uniform)                              │
│      7. Mutation (polynomial/Gaussian)                         │
│      8. Early stopping check:                                   │
│         - If no improvement for 20 generations                 │
│         - If insufficient signals (< 20 patterns detected)     │
│         - If poor performance (success rate < 40%)             │
│    → Pareto frontier of optimal parameter sets                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 7. Validation & Test                                              │
│    For each Pareto solution:                                    │
│      - Run detection on validation set                          │
│      - Calculate validation metrics                             │
│      - Run detection on test set                                │
│      - Calculate test metrics                                   │
│      - Check for overfitting:                                   │
│        * Test performance < 70% of training → overfitted       │
│        * Discard if overfitted                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 8. Strategy Selection                                             │
│    User selects optimal strategy based on:                      │
│      - High Return: Max total return (aggressive)               │
│      - Low Risk: Min drawdown + high Sharpe (conservative)     │
│      - Balanced: Best overall performance                       │
│    → Selected parameter set                                      │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 9. Database Storage                                               │
│    Store in optimization_results table:                         │
│      - pattern_key, asset, timeframe, regime                    │
│      - optimal_parameters (JSON)                                │
│      - performance_metrics (JSON)                               │
│      - created_at, last_used                                    │
│    → Parameters available for real-time scanning                │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ 10. Confidence Calibration Update                                 │
│     For each detected pattern in historical data:               │
│       - Record outcome (success/failure/timeout)                │
│       - Calculate actual win rate                               │
│       - Update calibration curve                                │
│       - Store in pattern_outcomes table                         │
│     → Calibration data for future confidence adjustment         │
└──────────────────────────────────────────────────────────────────┘
```

**Key Features**:
- **Multi-Objective**: Optimizes for profit AND risk simultaneously
- **Walk-Forward**: Prevents overfitting with proper train/val/test splits
- **Parallel Execution**: 32-worker ThreadPoolExecutor for speed
- **Database-Driven**: Results stored for real-time parameter selection
- **Adaptive**: Different parameters per asset/timeframe/regime

---

## Pattern Types & Implementations

### Pattern Hierarchy

```
PatternDetectorBase (patterns/engine.py)
    │
    ├─ Chart Patterns (20+)
    │  ├─ Reversal Patterns
    │  │  ├─ Head & Shoulders (hns.py)
    │  │  ├─ Double/Triple Top/Bottom (double_triple.py)
    │  │  ├─ Rounding Top/Bottom (rounding.py)
    │  │  ├─ Diamond Top/Bottom (diamond.py)
    │  │  └─ Cup & Handle (cup_handle.py)
    │  │
    │  ├─ Continuation Patterns
    │  │  ├─ Flags (flags.py)
    │  │  ├─ Pennants (flags.py)
    │  │  ├─ Rectangles (rectangle.py)
    │  │  └─ Channels (channels.py)
    │  │
    │  ├─ Triangle Patterns
    │  │  ├─ Ascending Triangle (triangles.py)
    │  │  ├─ Descending Triangle (triangles.py)
    │  │  └─ Symmetrical Triangle (triangles.py)
    │  │
    │  ├─ Wedge Patterns
    │  │  ├─ Rising Wedge (wedges.py)
    │  │  └─ Falling Wedge (wedges.py)
    │  │
    │  ├─ Broadening Patterns
    │  │  ├─ Broadening Top (broadening.py)
    │  │  └─ Broadening Bottom (broadening.py)
    │  │
    │  ├─ Harmonic Patterns (harmonic_patterns.py)
    │  │  ├─ Gartley (Bull/Bear)
    │  │  ├─ Bat (Bull/Bear)
    │  │  ├─ Butterfly (Bull/Bear)
    │  │  ├─ Crab (Bull/Bear)
    │  │  ├─ Cypher (Bull/Bear)
    │  │  ├─ Shark (Bull/Bear)
    │  │  └─ ABCD (Bull/Bear)
    │  │
    │  ├─ Elliott Wave (elliott_wave.py)
    │  │  ├─ Impulse Waves (5-wave)
    │  │  └─ Corrective Waves (ABC, WXY)
    │  │
    │  └─ Advanced Patterns (advanced_chart_patterns.py)
    │     ├─ Island Reversal
    │     ├─ Measured Move
    │     └─ Three Drives
    │
    └─ Candlestick Patterns (20+)
       ├─ Single Candle
       │  ├─ Hammer (candles.py)
       │  ├─ Shooting Star (candles.py)
       │  ├─ Doji (candles.py)
       │  ├─ Dragonfly Doji (candles.py)
       │  └─ Gravestone Doji (candles.py)
       │
       ├─ Two Candle
       │  ├─ Bullish/Bearish Engulfing (candles.py)
       │  ├─ Harami Bull/Bear (candles.py)
       │  ├─ Tweezer Top/Bottom (candles.py)
       │  ├─ Piercing Line (candles.py)
       │  └─ Dark Cloud Cover (candles.py)
       │
       ├─ Three Candle
       │  ├─ Three White Soldiers (candles.py)
       │  ├─ Three Black Crows (candles.py)
       │  ├─ Morning Star (candles.py)
       │  ├─ Evening Star (candles.py)
       │  ├─ Rising Three Methods (candles.py)
       │  └─ Falling Three Methods (candles.py)
       │
       └─ Advanced Candles (candles_advanced.py)
          ├─ Belt Hold
          ├─ Kicker
          └─ Tasuki
```

### Pattern Detection Algorithm (Generic)

```python
class PatternDetectorBase:
    key: str = "base"
    kind: Literal["chart", "candle"] = "chart"
    
    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        """
        Generic pattern detection algorithm.
        
        CRITICAL: Must be causal - only use data <= confirm_ts
        """
        events = []
        
        # 1. Preprocessing
        df = self._validate_dataframe(df)
        if df.empty or len(df) < self.min_bars:
            return events
        
        # 2. Feature Extraction
        features = self._extract_features(df)
        # Examples:
        # - Peaks/troughs (for H&S, double top, etc.)
        # - Trendlines (for triangles, wedges, channels)
        # - Support/resistance levels
        # - Fibonacci ratios (for harmonics)
        # - ATR for volatility normalization
        
        # 3. Pattern Matching
        for i in range(self.min_bars, len(df)):
            window = df.iloc[max(0, i - self.lookback):i+1]
            
            if self._matches_pattern(window, features):
                # 4. Validation
                if not self._validate_pattern(window, features):
                    continue
                
                # 5. Score Calculation
                score = self._calculate_score(window, features)
                
                # 6. Target/Stop Calculation
                target, stop = self._calculate_targets(window, features)
                
                # 7. Create Pattern Event
                event = PatternEvent(
                    pattern_key=self.key,
                    kind=self.kind,
                    direction=self._determine_direction(window),
                    start_ts=window.index[0],
                    confirm_ts=window.index[-1],
                    state="confirmed",
                    score=score,
                    target_price=target,
                    failure_price=stop,
                    overlay=self._create_overlay(window, features)
                )
                events.append(event)
        
        return events
```

---

## Real-Time Pattern Scanning

### Scan Architecture

**Two Parallel Threads**:

1. **Chart Pattern Thread** (`ScanWorker` with kind='chart')
   - Scans every 10 seconds (configurable)
   - Detects: H&S, triangles, wedges, flags, channels, etc.
   - Limited to 150-300 candles per pattern

2. **Candle Pattern Thread** (`ScanWorker` with kind='candle')
   - Scans every 10 seconds (same as chart)
   - Detects: Hammer, doji, engulfing, star patterns, etc.
   - Limited to 25-50 candles per pattern

**Dynamic Interval Adjustment**:

```python
# In ScanWorker._tick():

# Check market status
if is_market_closed():
    # Increase interval to 5+ minutes during closure
    self._timer.setInterval(max(300000, current_interval * 2))
    return  # Skip scanning

# Check resource constraints
if cpu_usage > 30% or memory_usage > 80%:
    # Increase interval to reduce load
    self._timer.setInterval(min(current_interval * 1.5, 300000))
    return

# Restore normal interval if resources available
if current_interval > original_interval:
    self._timer.setInterval(max(current_interval * 0.9, original_interval))
```

**Batch Detection** (`DetectionWorker`):

- Processes detectors in batches of 5
- Runs in background QThread
- Emits progress signals every batch
- Applies pattern-specific boundaries
- Yields to GUI every 1ms

### Progressive Formation Detection

**Purpose**: Show patterns forming in real-time (60-80% confidence)

**Stages**:
1. **Early** (<40%): Not shown
2. **Developing** (40-60%): Internal tracking
3. **Forming** (60-80%): **Shown with dashed lines**
4. **Mature** (80-95%): Shown with solid lines
5. **Completed** (>95%): Fully confirmed

**Update Frequency**: Every 1-minute candle

**Implementation**:

```python
class ProgressivePatternDetector:
    def update_patterns(self, current_data, asset, timeframe, detectors):
        forming_patterns = []
        
        for detector in detectors:
            # Run with partial formation mode
            patterns = self._run_progressive_detection(detector, current_data)
            
            for pattern in patterns:
                if pattern['confidence'] >= 0.60:  # Forming threshold
                    # Estimate formation progress
                    progress = self._estimate_formation_progress(pattern)
                    
                    # Determine stage
                    if progress >= 0.80:
                        stage = FormationStage.MATURE
                    elif progress >= 0.60:
                        stage = FormationStage.FORMING  # Show with dashes
                    elif progress >= 0.40:
                        stage = FormationStage.DEVELOPING
                    else:
                        stage = FormationStage.EARLY
                    
                    progressive_pattern = ProgressivePattern(
                        pattern_key=pattern['key'],
                        stage=stage,
                        confidence=pattern['confidence'],
                        completion_percent=progress,
                        visual_elements={'line_style': 'dashed' if stage == FormationStage.FORMING else 'solid'}
                    )
                    forming_patterns.append(progressive_pattern)
        
        return forming_patterns
```

---

## Historical Pattern Optimization

### NSGA-II Multi-Objective Optimization

**Objectives** (minimize all):
1. **-Total Return**: Maximize cumulative profit
2. **Max Drawdown**: Minimize worst peak-to-trough decline
3. **-Success Rate**: Maximize hit rate (target vs. stop)

**Algorithm**:

```
Initialize population P(0) of size N
Evaluate objectives for each individual

For generation t = 1 to T:
    1. Non-Dominated Sorting:
       - Rank individuals into Pareto fronts (F1, F2, ...)
       - F1: Non-dominated solutions (best)
       - F2: Dominated only by F1
       - Etc.
    
    2. Crowding Distance:
       - For each front, calculate crowding distance
       - Prefer individuals in sparse regions (diversity)
    
    3. Selection:
       - Tournament selection (size=3)
       - Compare rank first, then crowding distance
    
    4. Crossover:
       - Blend crossover (BLX-α) for float params
       - Uniform crossover for int params
       - Rate: 80%
    
    5. Mutation:
       - Polynomial mutation for bounded params
       - Gaussian mutation for unbounded
       - Rate: 10%
    
    6. Combine: Q(t) = P(t) ∪ Offspring
    
    7. Select next generation:
       - Fill P(t+1) from Q(t) using rank + crowding
       - Size N
    
    8. Early Stopping:
       - If hypervolume unchanged for 20 generations
       - If min_signals not met (< 20 patterns)
       - If performance unacceptable (success_rate < 40%)

Return Pareto frontier (F1)
```

**Hypervolume Indicator**:

Used to measure convergence and quality of Pareto frontier:

```python
def hypervolume(pareto_front, reference_point):
    """
    Calculate hypervolume dominated by Pareto front.
    Higher is better (larger dominated space).
    """
    # Monte Carlo approximation for 3+ objectives
    n_samples = 100000
    dominated_count = 0
    
    for _ in range(n_samples):
        # Sample random point in objective space
        point = np.random.uniform(
            low=[min(f.obj1) for f in pareto_front],
            high=reference_point
        )
        
        # Check if dominated by any Pareto solution
        if any(dominates(solution, point) for solution in pareto_front):
            dominated_count += 1
    
    volume = (dominated_count / n_samples) * reference_volume
    return volume
```

### Parameter Space Examples

**Head & Shoulders**:
```python
{
    'shoulder_tolerance': (0.02, 0.10, 'float'),      # 2-10% tolerance
    'neck_angle_tolerance': (5.0, 30.0, 'float'),    # 5-30 degrees
    'head_prominence': (1.05, 1.30, 'float'),        # 5-30% higher
    'symmetry_ratio': (0.7, 1.0, 'float'),           # 70-100% symmetric
    'min_bars': (20, 100, 'int')                     # 20-100 bars duration
}
```

**Triangle**:
```python
{
    'min_touches': (4, 8, 'int'),                    # 4-8 touches
    'trendline_tolerance': (0.005, 0.03, 'float'),   # 0.5-3% tolerance
    'convergence_angle': (10.0, 60.0, 'float'),      # 10-60 degrees
    'breakout_threshold': (0.01, 0.05, 'float'),     # 1-5% breakout
    'min_bars': (30, 150, 'int')                     # 30-150 bars
}
```

**Harmonic (Gartley)**:
```python
{
    'xab_ratio': (0.618, 0.618, 'fixed'),            # Fibonacci 0.618
    'abc_ratio': (0.382, 0.886, 'float'),            # 0.382-0.886
    'bcd_ratio': (1.272, 1.618, 'float'),            # 1.272-1.618
    'xad_ratio': (0.786, 0.786, 'fixed'),            # Fibonacci 0.786
    'ratio_tolerance': (0.01, 0.05, 'float'),        # 1-5% tolerance
    'min_bars_per_leg': (10, 50, 'int')             # 10-50 bars per leg
}
```

---

## Pattern Confidence & Calibration

### Confidence Calculation Pipeline

```
Raw Detector Score (0-100)
    ↓
Strength Calculator
    ├─ Confidence Component (35%)
    ├─ Volatility Component (25%)
    ├─ Historical Success (25%)
    └─ Price Action Quality (15%)
    ↓
Combined Strength Score (0-1)
    ↓
Confidence Calibrator
    - Get historical win rate
    - Apply adjustment factor (0.5-1.5x)
    ↓
Calibrated Confidence (0-100)
    ↓
DOM Confirmation (optional)
    - Check order book imbalance
    - Adjust ±20% based on alignment
    ↓
Final Confidence Score
    ↓
Star Rating (1-5 stars)
```

### Strength Calculator Components

**1. Confidence Score**:
```python
raw_confidence = pattern_event.confidence  # From detector
if raw_confidence < min_confidence (0.3):
    return 0.0

# Normalize to 0-1
normalized = (raw_confidence - 0.3) / 0.7
return clip(normalized, 0.0, 1.0)
```

**2. Volatility Score**:
```python
# Calculate ATR during formation
high = df['high']
low = df['low']
close = df['close']
prev_close = close.shift(1)

true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
atr = true_range.rolling(14).mean()

# Normalize by price
normalized_vol = atr / close.mean()

# Score volatility (moderate is best)
if normalized_vol < 0.01:  # Too low
    return 0.3
elif normalized_vol < 0.03:  # Good
    return 0.9
elif normalized_vol < 0.06:  # High but OK
    return 0.7
else:  # Too high
    return 0.4
```

**3. Historical Success Score**:
```python
# Query database for historical win rate
win_rate = db.query(
    "SELECT AVG(target_hit) FROM pattern_outcomes "
    "WHERE pattern_key = ? AND asset = ? AND timeframe = ?",
    (pattern_key, asset, timeframe)
)

if win_rate is None:
    return 0.5  # Default

# Convert to score (0-1)
# 50% win rate → 0.5 score
# 70% win rate → 0.9 score
# 30% win rate → 0.1 score
score = min(1.0, max(0.0, win_rate))
return score
```

**4. Price Action Quality**:
```python
# Momentum consistency
returns = close.pct_change()
sign_changes = (returns.shift(1) * returns < 0).sum()
momentum_score = 1.0 - (sign_changes / len(returns))

# Range expansion
range_ratio = (high - low) / close
range_consistency = 1.0 - range_ratio.std()

# Trend clarity
ma_20 = close.rolling(20).mean()
distance_from_ma = abs(close - ma_20) / close
trend_clarity = distance_from_ma.mean()

# Combined
quality_score = (momentum_score * 0.4 + 
                range_consistency * 0.3 + 
                trend_clarity * 0.3)
return clip(quality_score, 0.0, 1.0)
```

### Calibration Adjustment

```python
def calibrate_confidence(initial_score, historical_win_rate):
    """
    Adjust confidence based on historical performance.
    """
    baseline = 0.55  # Expected baseline for valid patterns
    
    if historical_win_rate >= 0.60:
        # Strong pattern, boost confidence
        adjustment = 1.0 + 0.3 * (historical_win_rate - baseline) / (1.0 - baseline)
    elif historical_win_rate < 0.50:
        # Weak pattern, reduce confidence
        adjustment = 0.5 + 0.5 * (historical_win_rate / 0.50)
    else:
        # Moderate pattern, slight adjustment
        adjustment = 1.0 + 0.2 * (historical_win_rate - baseline) / (baseline - 0.50)
    
    adjustment = clip(adjustment, 0.5, 1.5)
    calibrated = initial_score * adjustment
    return clip(calibrated, 0.0, 100.0)
```

### Star Rating Conversion

```python
def strength_to_stars(strength_score):
    """Convert strength score (0-1) to star rating (1-5)"""
    if strength_score >= 0.85:
        return 5  # Excellent
    elif strength_score >= 0.70:
        return 4  # Good
    elif strength_score >= 0.55:
        return 3  # Fair
    elif strength_score >= 0.40:
        return 2  # Weak
    else:
        return 1  # Very weak
```

---

## Multi-Timeframe Analysis

### Timeframe Combinations

**Predefined Strategies**:

1. **Scalping** (Short-term):
   - Timeframes: 1m, 5m, 15m
   - Weights: [0.2, 0.3, 0.5]
   - Focus: Fast entries/exits

2. **Intraday** (Medium-term):
   - Timeframes: 15m, 1h, 4h
   - Weights: [0.2, 0.4, 0.4]
   - Focus: Same-day trades

3. **Swing** (Long-term):
   - Timeframes: 1h, 4h, 1d
   - Weights: [0.3, 0.4, 0.3]
   - Focus: Multi-day positions

### Confluence Detection

```python
def detect_confluence(patterns_by_timeframe, strategy):
    """
    Detect multi-timeframe confluence.
    
    Bullish Confluence: All timeframes show bullish patterns
    Bearish Confluence: All timeframes show bearish patterns
    Mixed Signals: Conflicting directions
    """
    timeframes = strategy['timeframes']
    weights = strategy['weights']
    
    # Get patterns for each timeframe
    tf_directions = []
    for tf in timeframes:
        patterns = patterns_by_timeframe.get(tf, [])
        if not patterns:
            tf_directions.append('neutral')
        else:
            # Dominant direction for this timeframe
            bullish_count = sum(1 for p in patterns if p.direction == 'bull')
            bearish_count = sum(1 for p in patterns if p.direction == 'bear')
            
            if bullish_count > bearish_count:
                tf_directions.append('bull')
            elif bearish_count > bullish_count:
                tf_directions.append('bear')
            else:
                tf_directions.append('neutral')
    
    # Check confluence
    if all(d == 'bull' for d in tf_directions):
        return PatternAlignment.BULLISH_CONFLUENCE
    elif all(d == 'bear' for d in tf_directions):
        return PatternAlignment.BEARISH_CONFLUENCE
    elif 'neutral' in tf_directions:
        return PatternAlignment.NEUTRAL
    else:
        return PatternAlignment.MIXED_SIGNALS
```

### Composite Patterns

**Examples**:

1. **Head & Shoulders in Triangle**:
   - Primary: H&S reversal pattern
   - Secondary: Triangle consolidation
   - Interpretation: High reversal probability
   - Expected move: 15%
   - Risk level: Medium

2. **Flag in Wedge**:
   - Primary: Flag continuation
   - Secondary: Wedge structure
   - Interpretation: Strong momentum continuation
   - Expected move: 12%
   - Risk level: Low

3. **Elliott Wave with Harmonic**:
   - Primary: Elliott wave completion
   - Secondary: Harmonic ratio confluence
   - Interpretation: Precise entry opportunity
   - Expected move: 20%
   - Risk level: Medium

---

## DOM Confirmation

### Order Book Validation

**Purpose**: Validate pattern signals with real-time order flow

**Logic**:
```python
def confirm_with_dom(pattern_direction, symbol):
    """
    Check if order book aligns with pattern direction.
    """
    dom_snapshot = dom_service.get_latest_dom_snapshot(symbol)
    
    depth_imbalance = dom_snapshot['depth_imbalance']
    # Imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    # Positive: More bids (buying pressure)
    # Negative: More asks (selling pressure)
    
    if pattern_direction == 'bull':
        # Expect positive imbalance (bid > ask)
        aligned = depth_imbalance > 0
        imbalance_strength = depth_imbalance
    elif pattern_direction == 'bear':
        # Expect negative imbalance (ask > bid)
        aligned = depth_imbalance < 0
        imbalance_strength = abs(depth_imbalance)
    else:
        return {'adjusted_score': original_score, 'aligned': None}
    
    # Calculate confidence boost/penalty
    if aligned:
        if imbalance_strength > 0.30:  # Strong confirmation
            confidence_boost = +0.20  # +20%
            reasoning = f"✅ STRONG DOM confirmation: {imbalance_strength*100:.1f}% imbalance"
        elif imbalance_strength > 0.15:  # Moderate confirmation
            confidence_boost = +0.10  # +10%
            reasoning = f"✅ Moderate DOM confirmation: {imbalance_strength*100:.1f}% imbalance"
        else:
            confidence_boost = 0.0
            reasoning = f"→ Weak DOM alignment: {imbalance_strength*100:.1f}% imbalance"
    else:
        # DOM opposes pattern - penalty
        confidence_boost = -0.05  # -5%
        reasoning = f"⚠️ DOM contradiction: {depth_imbalance*100:.1f}% (opposing flow)"
    
    adjusted_score = clip(original_score + confidence_boost, 0.0, 1.0)
    
    return {
        'adjusted_score': adjusted_score,
        'dom_aligned': aligned,
        'confidence_boost': confidence_boost,
        'imbalance': depth_imbalance,
        'reasoning': reasoning
    }
```

**Thresholds**:
- **Strong confirmation**: |imbalance| > 30%
- **Moderate confirmation**: |imbalance| > 15%
- **Weak/neutral**: |imbalance| < 15%

---

## Parameter Selection & Optimization

### Database-Driven Parameter Selection

**Priority Order** (specific to general):

1. **asset_timeframe_regime** (most specific)
   - Example: EUR/USD + 5m + trending
   
2. **asset_timeframe**
   - Example: EUR/USD + 5m
   
3. **timeframe_regime**
   - Example: 5m + trending
   
4. **asset_regime**
   - Example: EUR/USD + trending
   
5. **timeframe** (general defaults)
   - Example: 5m
   
6. **default** (global fallback)
   - Hardcoded defaults

**Fallback Strategy**:
- If no parameters found: **Skip pattern** and log warning
- Never use random or unvalidated parameters

**Implementation**:

```python
class DatabaseParameterSelector:
    def get_optimal_parameters(self, context: ParameterContext):
        """
        Get optimal parameters using priority-based selection.
        
        context contains:
        - asset (e.g., "EUR/USD")
        - timeframe (e.g., "5m")
        - regime (e.g., "trending")
        - pattern_type (e.g., "head_shoulders")
        """
        # Try each priority level
        for priority in self.priority_order:
            params = self._get_parameters_for_priority(context, priority)
            
            if params and self._validate_parameter_set(params):
                logger.info(f"Using {priority} parameters for {context.pattern_type}")
                return params
        
        # No parameters found
        logger.warning(
            f"No parameters found for {context.pattern_type} "
            f"({context.asset}, {context.timeframe}, {context.regime}). "
            f"Skipping pattern."
        )
        return None
```

**Validation**:
```python
def _validate_parameter_set(self, parameter_set):
    """Ensure parameter set is valid for use."""
    # Check minimum trials
    if parameter_set.trial_count < self.min_trials:
        return False
    
    # Check minimum success rate
    if parameter_set.success_rate < self.min_success_rate:
        return False
    
    # Check age (not older than 6 months)
    age = datetime.now() - parameter_set.last_updated
    if age > timedelta(days=180):
        return False
    
    # Check confidence
    if parameter_set.confidence < 0.7:
        return False
    
    return True
```

---

## Complete Parameter Reference

### System Configuration (patterns.yaml)

```yaml
patterns:
  # Pattern scanning intervals
  current_scan:
    interval_seconds: 10  # Real-time scan every 10 seconds
    enabled_chart: true
    enabled_candle: true
  
  # Historical scanning
  historical_patterns:
    enabled: true
    start_time: "30d"  # Start 30 days ago
    end_time: "7d"     # End 7 days ago (avoid most recent)
  
  # Progressive formation
  progressive:
    enabled: true
    confidence_threshold: 60  # Show patterns at 60%+ confidence
    update_frequency: "1min"
    visual:
      line_style: "dashed"
      alpha: 0.6
  
  # Multi-timeframe analysis
  multi_timeframe:
    enabled: true
    combinations:
      - name: "scalping"
        timeframes: ["1m", "5m", "15m"]
        weights: [0.2, 0.3, 0.5]
      - name: "intraday"
        timeframes: ["15m", "1h", "4h"]
        weights: [0.2, 0.4, 0.4]
      - name: "swing"
        timeframes: ["1h", "4h", "1d"]
        weights: [0.3, 0.4, 0.3]
  
  # Strength calculation
  strength:
    weights:
      confidence: 0.35
      volatility: 0.25
      historical_success: 0.25
      price_action: 0.15
    min_confidence: 0.3
    volatility_window: 20
  
  # Confidence calibration
  calibration:
    min_samples: 30
    calibration_window_days: 180
    n_bins: 10
  
  # DOM confirmation
  dom_confirmation:
    enabled: true
    strong_threshold: 0.30
    moderate_threshold: 0.15
    strong_boost: 0.20
    moderate_boost: 0.10
    weak_penalty: -0.05
  
  # Resource limits
  resources:
    pattern_detection:
      max_cpu_percent: 30
    memory:
      max_usage_percent: 80

# Optimization configuration
optimization:
  engine: "nsga2"  # nsga2, ga, grid
  
  nsga2:
    population_size: 100
    max_generations: 200
    crossover_rate: 0.8
    mutation_rate: 0.1
    tournament_size: 3
  
  early_stopping:
    enabled: true
    patience: 20
    min_improvement: 0.001
  
  constraints:
    min_signals: 20
    min_success_rate: 0.4
    max_drawdown: 0.3
  
  database:
    parameters:
      selection_strategy: "historical_performance"
      fallback_strategy: "skip_pattern"
      priority_order:
        - "asset_timeframe_regime"
        - "asset_timeframe"
        - "timeframe_regime"
        - "asset_regime"
        - "timeframe"
        - "default"
      optimization:
        min_trials: 100
        min_success_rate: 0.4
        performance_window: 90
```

### Pattern Boundaries (pattern_boundaries.json)

```json
{
  "hammer": {
    "1m": 45,
    "5m": 30,
    "15m": 18,
    "1h": 9,
    "4h": 3,
    "1d": 2
  },
  "head_shoulders": {
    "1m": 270,
    "5m": 180,
    "15m": 108,
    "1h": 54,
    "4h": 18,
    "1d": 9
  },
  "harmonic_gartley_bull": {
    "1m": 525,
    "5m": 350,
    "15m": 210,
    "1h": 105,
    "4h": 35,
    "1d": 18
  }
}
```

---

## Workflow Diagrams

### Pattern Detection Decision Tree

```
Real-time scan triggered
    │
    ├─ Resources OK?
    │  ├─ No → Increase interval, skip scan
    │  └─ Yes → Continue
    │
    ├─ Market open?
    │  ├─ No → Increase interval to 5+ min
    │  └─ Yes → Continue
    │
    ├─ Get detector list
    │  └─ PatternRegistry.detectors(['chart', 'candle'])
    │
    ├─ For each detector:
    │  ├─ Apply boundary (pattern-specific candle count)
    │  ├─ Run detection
    │  └─ Collect events
    │
    ├─ For each event:
    │  ├─ Calculate strength
    │  ├─ Calibrate confidence
    │  ├─ DOM confirmation (optional)
    │  └─ Multi-TF confluence (optional)
    │
    ├─ Filter & sort
    │  ├─ Remove below threshold (60%)
    │  └─ Sort by combined score
    │
    └─ Draw on chart
       ├─ Boundaries (rectangles, lines)
       ├─ Badges (stars)
       ├─ Labels (name, direction, confidence)
       └─ Targets/stops
```

### Optimization Workflow

```
User clicks "Scan Historical"
    │
    ├─ Configure:
    │  ├─ Pattern type
    │  ├─ Asset & timeframe
    │  ├─ Regime (optional)
    │  └─ Date range
    │
    ├─ Define parameter space
    │  └─ Pattern-specific ranges
    │
    ├─ Load historical data
    │  ├─ OHLCV
    │  ├─ Volume (if available)
    │  └─ DOM (if available)
    │
    ├─ Walk-forward split
    │  ├─ Train: 60%
    │  ├─ Val: 20%
    │  └─ Test: 20%
    │
    ├─ NSGA-II loop (200 generations):
    │  ├─ Generate population (100)
    │  ├─ Evaluate objectives:
    │  │  ├─ Total return
    │  │  ├─ Max drawdown
    │  │  └─ Success rate
    │  ├─ Non-dominated sorting
    │  ├─ Crowding distance
    │  ├─ Selection (tournament)
    │  ├─ Crossover + mutation
    │  └─ Early stopping check
    │
    ├─ Validate on test set
    │  └─ Check overfitting
    │
    ├─ User selects strategy:
    │  ├─ High return
    │  ├─ Low risk
    │  └─ Balanced
    │
    └─ Store in database
       ├─ Optimal parameters
       ├─ Performance metrics
       └─ Calibration data
```

---

## Conclusion

This document provides a **complete reference** for the ForexGPT Pattern Recognition System, covering:

- **30+ pattern implementations** with detection algorithms
- **Real-time scanning** with adaptive intervals and resource management
- **Historical optimization** with NSGA-II multi-objective genetic algorithm
- **Confidence calibration** based on historical win rates
- **Multi-timeframe confluence** for signal validation
- **DOM confirmation** using order book data
- **Database-driven parameter selection** with priority-based fallback

**For issues, bugs, and optimizations**, see `/SPECS/2_PatternRecognition.txt`.

**Next Steps**:
1. Review identified issues in specs document
2. Consolidate duplicated primitive functions
3. Standardize imports across pattern modules
4. Add unit tests for calibration and strength calculation
5. Optimize batch detection for < 1s latency

---

**Document Version**: 1.0  
**Generated**: 2025-10-13  
**Maintainer**: ForexGPT Development Team
