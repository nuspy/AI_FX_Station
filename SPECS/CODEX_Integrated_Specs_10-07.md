# CODEX Integrated Specification — 07 Oct

## Scope
Consolidated plan to elevate ForexGPT toward production readiness across training, forecasting, backtesting, pattern analytics, autotrading, monitoring, and data semantics. Each workstream links to observed code realities.

## Workstreams

### 1. Training Pipeline Hardening
- Goals: introduce robust validation, algorithm diversity, and feature governance.
- Actions:
  1. Implement walk-forward cross-validation utility (wrapper around TimeSeriesSplit) and expose it in CLI/UI; replace the single train_test_split path (current use: src/forex_diffusion/training/train_sklearn.py:549).
  2. Extend _fit_model to register gradient boosting (LightGBM / XGBoost) and, where available, lightweight neural regressors (src/forex_diffusion/training/train_sklearn.py:592). Ensure serialization supports new models.
  3. Add optional feature selection stage (e.g. RFE or mutual-information filter) before standardisation; surface min/max feature caps in CLI/UI.
  4. Formalise volume semantics by renaming columns to tick_volume and documenting provider provenance before feature expansion (current ingestion: src/forex_diffusion/training/train_sklearn.py:393).
- Acceptance: unit tests covering CV splitter correctness, model loader persistence for new estimators, and feature-selection toggles; documentation update for volume semantics.

### 2. Forecast Reliability and Uncertainty
- Goals: deliver calibrated multi-horizon forecasts with quantified uncertainty.
- Actions:
  1. Replace horizon replication with either (a) multi-output regression heads or (b) iterative forecasting loop; retire the padding logic in end_to_end_predict and ensemble aggregation (current behaviour: src/forex_diffusion/inference/prediction.py:151 and src/forex_diffusion/inference/parallel_inference.py:353).
  2. Introduce conformal prediction or quantile regression for intervals; expose in UI and API payloads.
  3. Add weekly automated retrospective validation comparing predicted vs realised returns; persist MAE/RMSE deltas for monitoring.
- Acceptance: integration test that multi-horizon outputs differ per-step, interval coverage within expected quantiles, and monitoring job writing metrics to DB/artifacts.

### 3. Backtesting Realism Upgrade
- Goals: align simulations with executable trading strategies.
- Actions:
  1. Generalise simulate_trades to support short entries, dynamic sizing, and parameterised entry/exit rules (current long-only logic: src/forex_diffusion/backtest/engine.py:181).
  2. Add pluggable slippage/spread models (volatility-aware, session-aware) instead of static constants.
  3. Compute additional metrics (Sortino, Calmar, profit factor, win rate) and make them available to UI/API consumers.
  4. Provide bootstrap or Monte Carlo analysis for distributional insights.
- Acceptance: regression tests validating short-path PnL, new metrics, and scenario slippage curves; documentation summarising strategy templates.

### 4. Pattern Analytics Integration
- Goals: bridge pattern detections with ML training and finish optimisation tooling.
- Actions:
  1. Wire PatternEvent outputs into feature engineering (e.g. binary flags and confidence scores appended before standardisation) ensuring causal alignment.
  2. Complete UI and logic placeholders in PatternTrainingTab so GA optimisation and progress monitoring function end-to-end (current placeholders: src/forex_diffusion/ui/pattern_training_tab.py:188).
  3. Establish statistical evaluation (false-positive benchmarking, success rates) persisted per pattern and timeframe.
- Acceptance: automated tests verifying pattern-derived features in training matrices, GA workflow smoke tests, and historical stats accessible from the UI/API.

### 5. Autotrading and Decision Engine Maturation
- Goals: transition the automated trading engine from prototype to paper-trading readiness.
- Actions:
  1. Replace simulated market data paths with broker adapters and add graceful degradation only for explicit test modes (current simulation hook: src/forex_diffusion/trading/automated_trading_engine.py:234).
  2. Ensure signal generation degrades to ensemble fallback with documented confidence thresholds rather than hard zeros (neutral-return guard: src/forex_diffusion/trading/automated_trading_engine.py:379).
  3. Calculate P&L using broker fills or market price snapshots instead of synthetic multipliers (current stub: src/forex_diffusion/trading/automated_trading_engine.py:491).
  4. Instrument latency, slippage, and execution outcomes; stream metrics into monitoring dashboard.
- Acceptance: paper-trade dry run capturing fills and latency, P&L reconciliation against broker statements, and telemetry charts populated in monitoring UI.

### 6. Monitoring and Auto-Retrain Orchestration
- Goals: activate existing monitoring infrastructure and automate retraining safely.
- Actions:
  1. Integrate DriftDetector scoring into the weekly validation job and persist alerts.
  2. Connect AutoRetrainingPipeline triggers to actual training invocations with approval checkpoints (current dormant pipeline: src/forex_diffusion/training/auto_retrain.py:131).
  3. Implement A/B routing that respects configured traffic splits and promotes/rolls back models based on statistical tests.
- Acceptance: end-to-end rehearsal where drift triggers retraining, AB metrics recorded, and rollout status reflected in model registry.

### 7. Data Semantics and Documentation
- Goals: ensure users understand provider limitations and configuration flows.
- Actions:
  1. Update schema docs and load-time validators to flag tick-volume vs true volume and note provider provenance.
  2. Document workflow for enabling live providers, model monitoring, retraining, and paper trading in README/quick-start guides.
  3. Add configuration validation for missing API keys or misconfigured databases, failing fast with actionable errors.
- Acceptance: doc review checklist complete, CI lint for config schemas, and runtime validation tests.

## Cross-Cutting
- Introduce tracing/logging correlation IDs across services to trace a forecast through backtesting, signal generation, and execution.
- Expand test matrix to cover GPU/non-GPU paths and key timeframes to prevent regressions.

## Dependencies and Sequencing
1. Training hardening and forecast reliability (Workstreams 1–2) unblock most downstream tasks.
2. Backtesting realism (Workstream 3) should precede autotrading maturity to ensure strategies are trustworthy.
3. Pattern integration (Workstream 4) can proceed in parallel but depends on feature pipeline hooks from Workstream 1.
4. Monitoring and auto-retrain (Workstream 6) should follow once validation metrics (Workstream 2) are available.

## Deliverables
- Updated training CLI/UI with CV, new estimators, feature selection, and volume semantics.
- Forecast service emitting calibrated multi-horizon distributions with intervals.
- Enhanced backtest reports and pattern optimisation UI.
- Autotrading engine ready for paper trading with telemetry.
- Operational monitoring pipeline with drift alerts and controlled retraining.
- Revised documentation and configuration guides.
