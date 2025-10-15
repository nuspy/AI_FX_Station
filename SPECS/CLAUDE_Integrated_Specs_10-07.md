# CLAUDE Integrated Implementation Specifications

## Document Control
- **Version**: 1.0
- **Date**: October 7, 2025
- **Status**: Proposed
- **Supersedes**: CODEX_Integrated_Specs_10-07.md (enhances and expands)

---

## Executive Framework

### Purpose
This specification defines the logical architecture, process flows, data transformations, and integration patterns required to elevate ForexGPT from prototype to production-ready quantitative trading platform. It expands upon the CODEX specifications with enhanced detail on system behavior, operational procedures, and quality assurance.

### Guiding Principles
1. **Safety First**: All changes prioritize preventing financial loss and system instability
2. **Incremental Value**: Each workstream delivers independently testable capabilities
3. **Observable Systems**: Every component emits metrics and logs for debugging
4. **Reversible Decisions**: Architecture choices include fallback and rollback mechanisms
5. **Data Integrity**: Quality validation occurs at every transformation boundary

### Success Definition
Production readiness is achieved when:
- Models can be trained, validated, and deployed without manual intervention
- Forecasts include calibrated uncertainty estimates validated against realized outcomes
- Backtests accurately simulate executable trading strategies
- Autotrading operates in paper-trade mode with full telemetry and safety controls
- Monitoring detects drift and triggers remediation workflows
- All critical paths have ≥80% test coverage and documented operational procedures

---

## Core Workstream Specifications

## Workstream 1: Training Pipeline Hardening

### 1.1 Time-Series Cross-Validation Framework

**Objective**: Replace single train-test split with walk-forward validation that respects temporal dependencies.

**Logical Requirements**:

**1.1.1 Splitter Configuration**
- Define a configurable time-series splitter that creates overlapping or non-overlapping training windows
- Each split must maintain chronological order: training data always precedes validation data
- Support multiple split strategies: expanding window (grows training set), sliding window (fixed size), and gap-based (skip data between train/val to prevent leakage)
- Allow specification of minimum training window size (e.g., 6 months) and validation window size (e.g., 1 month)
- Enable configuration of number of splits (default 5 for computational efficiency)

**1.1.2 Integration Points**
- Expose splitter configuration through CLI arguments: `--cv-strategy [expanding|sliding|gap]`, `--cv-splits N`, `--cv-train-months M`, `--cv-val-months V`
- Add UI components to training tab showing split visualization: timeline chart with colored segments indicating train/validation periods for each fold
- Store CV strategy metadata in model artifacts for reproducibility

**1.1.3 Performance Aggregation**
- Compute per-fold metrics (MAE, RMSE, R²) and aggregate using mean and standard deviation
- Flag folds with anomalous performance (>2 standard deviations from mean) for investigation
- Persist per-fold predictions for detailed error analysis
- Calculate forecast stability metric: correlation of predictions across folds for same test periods

**1.1.4 Validation Logic**
- Verify that no data leakage occurs: assert that max(train_dates) < min(val_dates) for each fold
- Ensure feature standardization parameters are computed only on training data within each fold
- Confirm that market regime shifts are represented across folds (bull, bear, sideways)

---

### 1.2 Algorithm Portfolio Expansion

**Objective**: Add gradient boosting and ensemble methods to increase model expressiveness for non-linear forex patterns.

**Logical Requirements**:

**1.2.1 Gradient Boosting Integration**
- Integrate LightGBM as primary gradient boosting implementation due to speed and memory efficiency
- Add XGBoost as alternative for comparison and ensemble diversity
- Define default hyperparameters suitable for time-series regression: 
  - Learning rate: 0.05 (conservative to prevent overfitting)
  - Max depth: 5 (moderate to capture interactions without overfitting)
  - Number of estimators: 100-500 (early stopping based on validation)
  - Subsample: 0.8 (introduces randomness for regularization)
- Implement categorical feature handling for currency pairs, timeframes, and market sessions

**1.2.2 Neural Network Pathway**
- Add support for simple feedforward neural networks as optional estimator
- Architecture should be shallow (2-3 hidden layers) to prevent overfitting on limited forex data
- Use dropout (0.3) and L2 regularization for generalization
- Implement early stopping based on validation loss with patience parameter
- Support GPU acceleration when available but gracefully degrade to CPU

**1.2.3 Model Selection and Comparison**
- Enable training multiple algorithms simultaneously for comparison
- Generate comparison report showing:
  - Cross-validation performance metrics per algorithm
  - Training time and inference latency benchmarks
  - Feature importance rankings (for tree-based and linear models)
  - Prediction distribution analysis (identify if models systematically over/under-predict)
- Allow selection of best model based on configurable metric (default: validation RMSE)
- Support ensemble creation by averaging top-N performers weighted by validation performance

**1.2.4 Serialization and Loading**
- Extend model persistence to handle new model types with proper versioning
- Store model type identifier, hyperparameters, and training metadata in manifest file
- Implement validation on load to ensure model was trained with compatible data schema
- Handle backward compatibility: old models should load and inference with warnings if schema evolved

---

### 1.3 Feature Selection and Dimensionality Management

**Objective**: Prevent feature bloat and improve model interpretability through principled feature selection.

**Logical Requirements**:

**1.3.1 Feature Selection Strategies**
- Implement three selection approaches:
  1. **Recursive Feature Elimination (RFE)**: Iteratively remove least important features based on model coefficients/importances
  2. **Mutual Information**: Rank features by dependency with target variable, select top-K
  3. **Variance Threshold**: Remove features with near-zero variance (indicate little signal)
- Allow configuration of target feature count or percentage (e.g., keep top 50% or top 100 features)
- Provide option to disable selection for maximum expressiveness

**1.3.2 Selection Process Integration**
- Feature selection occurs after feature engineering but before standardization
- Selection criteria computed only on training data to prevent leakage
- Selected feature subset persisted in model artifact for consistent inference
- Generate feature selection report showing:
  - Importance scores for all features
  - Selected vs rejected feature lists
  - Correlation matrix of selected features to detect redundancy

**1.3.3 Feature Caps and Governance**
- Establish maximum feature count based on sample size (heuristic: max features ≤ sqrt(n_samples))
- Alert if feature count exceeds recommended threshold
- Track feature count trend over time to detect feature creep
- Implement mandatory justification process for adding new feature categories

**1.3.4 Domain-Specific Considerations**
- Prioritize retention of feature groups with known forex relevance (price action, volume profile, VSA)
- Ensure at least one feature from each category survives selection
- Consider temporal stability: features that appear/disappear between retraining create inconsistency

---

### 1.4 Volume Semantics Formalization

**Objective**: Clarify whether volume represents tick count or actual traded volume and document provider limitations.

**Logical Requirements**:

**1.4.1 Data Schema Enhancement**
- Rename ambiguous `volume` column to `tick_volume` in internal schema
- Add metadata field `volume_type` with enumeration: [TICK_COUNT, ACTUAL_VOLUME, UNKNOWN]
- Persist data provider identifier (MT4, MT5, broker name) in metadata
- Flag whether provider supplies true volume or tick count approximation

**1.4.2 Feature Engineering Adjustments**
- Volume-derived features (volume profile, VSA indicators) should document assumptions
- If tick volume: document that features measure trading activity frequency, not capital flow
- If actual volume: validate magnitude consistency (detect obviously erroneous values)
- Consider separate feature branches for tick vs actual volume with appropriate naming

**1.4.3 Documentation and User Communication**
- Update data ingestion documentation explaining tick volume vs actual volume distinction
- Add warnings in UI when loading data from providers known to supply only tick volume
- Generate data quality report showing volume statistics and anomaly detection
- Provide guidance on interpreting model predictions trained on tick volume data

**1.4.4 Validation and Quality Checks**
- Implement sanity checks: volume cannot be negative, zero volume should be rare
- Detect suspiciously constant volume (indicates feed issue)
- Flag volume spikes exceeding 5x typical range for manual review
- Cross-validate volume against known market events (NFP releases should show volume increase)

---

## Workstream 2: Forecast Reliability and Uncertainty Quantification

### 2.1 True Multi-Horizon Forecasting

**Objective**: Replace horizon replication with genuine multi-step predictions that acknowledge increasing uncertainty.

**Logical Requirements**:

**2.1.1 Multi-Output Architecture**
- Transition to multi-output regression where a single model predicts all horizons simultaneously
- Each horizon (1-step, 5-step, 10-step, etc.) becomes a separate output dimension
- Alternatively, implement auto-regressive approach: 1-step prediction feeds into 2-step prediction recursively
- For ensemble approaches, each member should predict all horizons independently

**2.1.2 Training Modifications**
- Construct target matrix with columns for each forecast horizon
- Account for data availability: 10-step ahead targets require 10 fewer samples than 1-step
- Consider loss function weighting: penalize near-term errors more heavily than distant ones
- Validate that longer horizons show wider prediction distributions (increasing uncertainty)

**2.1.3 Inference Changes**
- Remove padding/replication logic that copies last forecast value to all horizons
- Return distinct prediction for each horizon with associated timestamps
- Compute prediction deltas between horizons to identify expected directional changes
- Flag if consecutive horizons show implausible discontinuities (suggests model instability)

**2.1.4 Performance Evaluation**
- Measure per-horizon accuracy separately: expect RMSE to increase with horizon length
- Track horizon-specific bias: are 10-step forecasts systematically over-optimistic?
- Compute directional accuracy: percentage of correct up/down predictions per horizon
- Analyze error decomposition: is error growth due to bias or variance increase?

---

### 2.2 Uncertainty Quantification via Prediction Intervals

**Objective**: Provide calibrated confidence intervals that honestly represent forecast uncertainty.

**Logical Requirements**:

**2.2.1 Conformal Prediction Implementation**
- Adopt conformal prediction framework for distribution-free uncertainty quantification
- Split training data into proper training set and calibration set (80/20 split)
- Train model on training set, compute absolute residuals on calibration set
- For new predictions, construct intervals using calibration residual quantiles

**2.2.2 Interval Construction**
- Generate 90% prediction intervals by default (5th to 95th percentile)
- Support multiple confidence levels (50%, 80%, 90%, 95%) simultaneously for visualization
- Ensure intervals are asymmetric if residual distribution is skewed
- Widen intervals for longer horizons to reflect increasing uncertainty

**2.2.3 Quantile Regression Alternative**
- Provide quantile regression as alternative approach for algorithms that support it
- Train separate models predicting 10th, 50th (median), and 90th percentiles
- Validate that quantiles maintain proper ordering: Q10 < Q50 < Q90
- Compare conformal and quantile approaches to assess agreement

**2.2.4 Interval Calibration Monitoring**
- Track interval coverage: percentage of actual outcomes falling within predicted intervals
- Expected coverage should match nominal level (90% intervals should contain 90% of outcomes)
- Detect miscalibration: coverage significantly below nominal suggests overconfident intervals
- Recompute calibration periodically as new data arrives to maintain accuracy

**2.2.5 API and UI Integration**
- Forecast API should return: `{point_prediction, lower_bound, upper_bound, confidence_level}`
- UI visualizes forecasts with shaded interval regions
- Allow toggling between confidence levels interactively
- Display historical interval calibration statistics on monitoring dashboard

---

### 2.3 Forecast Validation and Performance Tracking

**Objective**: Systematically compare predictions to realized outcomes to detect degradation.

**Logical Requirements**:

**2.3.1 Retrospective Validation Pipeline**
- Schedule weekly job comparing past forecasts to actual market outcomes
- Match forecast timestamps to corresponding realized values accounting for horizon
- Compute forecast errors (predicted - actual) and aggregate into performance metrics
- Store results in time-series database for trend analysis

**2.3.2 Performance Metrics Tracking**
- Calculate rolling MAE, RMSE, and R² over 30-day, 90-day, and 365-day windows
- Track directional accuracy: correct prediction of price increase/decrease
- Measure forecast bias: systematic over/under prediction indicates drift
- Compute forecast skill relative to naive baseline (e.g., random walk, simple moving average)

**2.3.3 Degradation Detection**
- Alert if current 30-day performance drops >20% below 90-day historical average
- Flag if directional accuracy falls below 50% (worse than random guessing)
- Identify if degradation is uniform across currencies/timeframes or localized
- Trigger investigation workflow when performance anomalies detected

**2.3.4 Root Cause Analysis Support**
- Decompose errors by market regime (trending vs ranging), volatility level, and time of day
- Identify if errors correlate with specific events (news releases, market opens/closes)
- Compare performance of different model types to isolate algorithm-specific issues
- Generate diagnostic report for human review when automatic retraining is triggered

---

## Workstream 3: Backtesting Realism and Strategy Evaluation

### 3.1 Generalized Trading Strategy Framework

**Objective**: Support diverse trading strategies beyond long-only, fixed-size entries.

**Logical Requirements**:

**3.1.1 Directional Flexibility**
- Extend simulation to support short positions: enter when forecast < threshold, exit when target reached or stop-loss hit
- Implement bi-directional strategies: simultaneously hold long and short positions in correlated pairs
- Handle position flattening: close all positions at designated times (e.g., before weekend)
- Support position reversal: exit long and enter short in single operation

**3.1.2 Position Sizing Algorithms**
- Fixed fractional: position size = account_equity * risk_fraction
- Volatility-adjusted: position size inversely proportional to ATR or recent volatility
- Kelly criterion: size based on estimated win probability and average win/loss ratio
- Maximum position limits per instrument and aggregate across portfolio
- Margin requirement checking: ensure sufficient equity for all open positions

**3.1.3 Entry and Exit Logic**
- Parameterize entry conditions: forecast threshold, confirmation signals, timing constraints
- Support multiple exit reasons: profit target, stop loss, trailing stop, time-based, signal reversal
- Implement partial exits: close portion of position as price moves favorably
- Allow entry filters: don't trade during low-liquidity sessions, major news events, high spread periods

**3.1.4 Multi-Instrument Portfolio Logic**
- Simulate correlated instruments simultaneously with shared capital pool
- Apply portfolio-level position limits and risk constraints
- Implement correlation-based position sizing: reduce size in highly correlated positions
- Support currency exposure limits: total EUR exposure across EUR/USD, EUR/GBP, etc.

---

### 3.2 Realistic Execution Cost Modeling

**Objective**: Replace static slippage constants with dynamic, regime-aware cost models.

**Logical Requirements**:

**3.2.1 Spread Modeling**
- Capture bid-ask spread variation by time of day: wider during Asian session, tighter during London/NY overlap
- Model spread widening during high volatility periods (proportional to ATR)
- Incorporate spread expansion during news releases (maintain event calendar)
- Use historical spread data from provider if available, otherwise use provider-specific typical spreads

**3.2.2 Slippage Estimation**
- Base slippage on order size relative to typical market depth
- Increase slippage non-linearly for larger orders (market impact)
- Model adverse selection: larger slippage in direction of rapid price movement
- Distinguish between market orders (higher slippage) and limit orders (lower but partial fill risk)

**3.2.3 Partial Fill Simulation**
- For large orders, simulate partial fills requiring multiple transactions
- Model limit order fill probability as function of how far limit price is from market
- Account for fill delay: limit orders may fill minutes or hours after placement
- Stop orders may not fill at exact stop price due to gap/slippage

**3.2.4 Commission Structure**
- Apply broker-specific commission schedules: flat per-trade, per-lot, or percentage-based
- Include swap/rollover costs for positions held overnight
- Model minimum commission charges that affect small positions disproportionately
- Account for currency conversion fees if trading crosses in non-account currency

**3.2.5 Configuration and Calibration**
- Expose all cost model parameters through configuration files
- Provide cost model calibration utility that fits parameters to actual trade fills
- Generate execution quality reports showing estimated vs actual costs
- Support broker-specific profiles with pre-calibrated parameters

---

### 3.3 Advanced Performance Metrics

**Objective**: Extend beyond Sharpe ratio to capture risk-adjusted performance comprehensively.

**Logical Requirements**:

**3.3.1 Downside Risk Metrics**
- Sortino Ratio: adjust for downside volatility only (negative returns)
- Compute downside deviation relative to minimum acceptable return (MAR)
- Clearly communicate that Sortino >1.5 is good, >2.0 is excellent in forex context

**3.3.2 Drawdown Analysis**
- Maximum drawdown: largest peak-to-trough decline during backtest period
- Average drawdown: mean of all drawdown episodes
- Drawdown duration: time required to recover from each drawdown
- Calmar Ratio: annualized return / maximum drawdown
- Underwater plot: visualize cumulative time spent in drawdown state

**3.3.3 Win/Loss Statistics**
- Win rate: percentage of profitable trades
- Profit factor: gross profits / gross losses
- Average win vs average loss: measure reward/risk per trade
- Consecutive wins/losses: detect winning/losing streaks
- Expectancy: average profit per trade including frequency

**3.3.4 Consistency Metrics**
- Monthly/quarterly return distribution: detect if returns are steady or sporadic
- Percentage of profitable months/quarters
- Worst month/quarter: stress test mental resilience
- Correlation of returns across different market regimes

**3.3.5 Statistical Robustness**
- Bootstrap analysis: resample returns to estimate confidence intervals on metrics
- Monte Carlo simulation: randomize trade sequence to assess order dependency
- Walk-forward efficiency: ratio of out-of-sample to in-sample Sharpe
- Parameter sensitivity: how much do results change with small strategy modifications?

**3.3.6 Reporting and Visualization**
- Generate comprehensive PDF report with all metrics and plots
- Equity curve with drawdown overlay
- Return distribution histogram with normality test
- Calendar heat map showing return by month/year
- Trade analysis scatter plot: holding time vs return

---

### 3.4 Strategy Comparison and Optimization

**Objective**: Enable systematic comparison of strategy variants and hyperparameter optimization.

**Logical Requirements**:

**3.4.1 Batch Backtesting**
- Support running multiple strategy configurations in parallel
- Each configuration produces complete performance report
- Generate comparison table ranking strategies by selected metric
- Identify common characteristics of top performers

**3.4.2 Parameter Space Exploration**
- Define parameter ranges for key strategy components (entry thresholds, stop levels, position sizes)
- Implement grid search or random search over parameter space
- Visualize performance landscape as heatmap for 2D parameter pairs
- Detect parameter cliff edges: small changes causing large performance swings (indicates overfitting)

**3.4.3 Robustness Testing**
- Run each strategy variant across multiple time periods to test consistency
- Test on different currency pairs to assess generalization
- Apply regime filtering: performance in trending vs ranging markets
- Sensitivity analysis: vary execution costs to stress-test profitability

**3.4.4 Overfitting Prevention**
- Reserve portion of data for final validation never used during optimization
- Penalize strategies with excessive parameters (Bayesian Information Criterion)
- Require minimum trade count for statistical validity (e.g., ≥30 trades)
- Flag if in-sample/out-of-sample performance ratio is suspiciously high (>1.5 suggests overfitting)

---

## Workstream 4: Pattern Analytics Integration

### 4.1 Pattern Event Ingestion into Feature Pipeline

**Objective**: Bridge technical pattern detection with ML feature engineering to realize hybrid approach.

**Logical Requirements**:

**4.1.1 Pattern Feature Schema**
- Define standardized pattern event structure: `{pattern_type, timestamp, confidence, direction, completion_price}`
- Pattern types enumeration: HEAD_SHOULDERS, DOUBLE_TOP, ASCENDING_TRIANGLE, BULL_FLAG, etc.
- Confidence score: 0.0-1.0 representing detector certainty
- Direction: BULLISH, BEARISH, or NEUTRAL

**4.1.2 Temporal Alignment**
- Synchronize pattern events with OHLC bars: each bar receives pattern flags active at that timestamp
- Handle multi-bar patterns: flag persists until pattern completes or invalidates
- Implement lookback window: feature vector includes patterns detected in past N bars
- Ensure no lookahead bias: pattern recognized at bar T cannot use information from bar T+1

**4.1.3 Feature Encoding**
- Binary flags: one-hot encoding for pattern types (BULL_FLAG_DETECTED, BEAR_FLAG_DETECTED)
- Confidence features: numeric confidence score if pattern detected, 0 otherwise
- Count features: number of bullish patterns in past N bars, number of bearish patterns
- Directional aggregation: net bullish pattern score - net bearish pattern score

**4.1.4 Pattern-Price Interaction Features**
- Distance from pattern completion: bars elapsed since pattern completed
- Price relative to pattern completion level: current_price / completion_price - 1
- Pattern success indicator: did price move in expected direction post-completion?
- Pattern failure indicator: did price move against expected direction?

**4.1.5 Integration into Training**
- Pattern features appended to existing feature matrix before standardization
- Standardization parameters computed to normalize pattern confidence scores
- Feature importance analysis reveals which patterns contribute to predictions
- Ablation study: compare model performance with and without pattern features

---

### 4.2 Pattern Optimization and Discovery

**Objective**: Complete UI and backend logic for genetic algorithm-based pattern parameter optimization.

**Logical Requirements**:

**4.2.1 Optimization Objective Function**
- Define fitness as pattern detection accuracy: true positives / (true positives + false positives)
- True positive: pattern followed by expected price movement exceeding threshold within timeout
- False positive: pattern not followed by expected movement
- Fitness function should penalize patterns that trigger too frequently (many false signals) or too rarely (insufficient utility)

**4.2.2 Genetic Algorithm Configuration**
- Population size: 50-100 pattern parameter sets
- Generations: 20-50 iterations (balance compute cost vs convergence)
- Selection: tournament selection favoring high-fitness individuals
- Crossover: blend parameters from two parents to create offspring
- Mutation: randomly adjust parameters within bounds to maintain diversity
- Elitism: preserve top 10% of population to ensure monotonic improvement

**4.2.3 Parameter Encoding**
- Each pattern type has configurable parameters (e.g., triangle: breakout threshold, minimum height, maximum duration)
- Parameters encoded as chromosome with genes corresponding to each tunable value
- Establish bounds for each parameter based on domain knowledge
- Implement constraint satisfaction: parameter combinations must be logically valid

**4.2.4 Fitness Evaluation Strategy**
- Evaluate each individual on historical data using walk-forward approach
- Compute fitness on out-of-sample data to prevent overfitting
- Average fitness across multiple currency pairs and timeframes for robustness
- Penalize runtime: patterns requiring excessive computation receive fitness discount

**4.2.5 Progress Monitoring and Visualization**
- UI displays real-time optimization progress: generation number, best fitness, average fitness
- Plot fitness evolution over generations to confirm convergence
- Show best-performing parameter sets and their detection examples
- Allow early termination if fitness plateaus (no improvement for 5 generations)

**4.2.6 Result Persistence and Deployment**
- Save optimized pattern parameters to configuration file
- Tag with optimization date, data range, and fitness metrics
- Allow A/B testing: compare original vs optimized parameters in live detection
- Provide rollback mechanism if optimized parameters degrade performance

---

### 4.3 Pattern Performance Analytics

**Objective**: Quantify pattern detection accuracy and profitability to guide strategy development.

**Logical Requirements**:

**4.3.1 Detection Statistics Database**
- Record every pattern detection: timestamp, type, confidence, completion price
- Track outcome: price change 1/5/10 bars after detection
- Classify outcome: success (moved as expected), failure (moved opposite), neutral (insufficient movement)
- Persist in time-series database for historical analysis

**4.3.2 Performance Metrics**
- True positive rate: percentage of detected patterns followed by expected moves
- False positive rate: percentage of detected patterns not followed by expected moves
- Precision-recall curve: tradeoff between catching all patterns vs reducing false alarms
- Stratify metrics by pattern type, currency pair, timeframe, and market regime

**4.3.3 Profitability Analysis**
- Simulate simple strategy: enter trade on pattern detection, exit after N bars
- Compute win rate, average profit per trade, maximum drawdown
- Compare profitability across pattern types to identify most reliable
- Assess if pattern profitability persists over time or degrades (market adaptation)

**4.3.4 False Positive Investigation**
- Identify common characteristics of false positive detections
- Visualize false positive examples to understand failure modes
- Develop filters to suppress low-quality detections (e.g., patterns during low volume)
- Iterate on pattern definition to reduce false positives

**4.3.5 Reporting and Dashboards**
- Generate monthly pattern performance report
- Dashboard showing pattern detection frequency, accuracy, and profitability trends
- Alert if pattern accuracy drops significantly (suggests market regime change or pattern obsolescence)
- Provide drill-down capability: click pattern type to see all recent detections and outcomes

---

## Workstream 5: Autotrading Engine Maturation

### 5.1 Broker Adapter Architecture

**Objective**: Replace simulated data with real broker integrations supporting multiple providers.

**Logical Requirements**:

**5.1.1 Broker Abstraction Layer**
- Define unified broker interface: `{connect(), get_quotes(), place_order(), get_positions(), get_account()}`
- Implement adapters for common brokers: OANDA, Interactive Brokers, MetaTrader 5
- Each adapter translates broker-specific APIs to common interface
- Handle authentication, connection management, and error recovery

**5.1.2 Quote Streaming**
- Establish persistent WebSocket or FIX connection for real-time quotes
- Maintain local order book cache with bid/ask/timestamp
- Detect stale quotes (no update for >5 seconds) and mark connection unhealthy
- Implement reconnection logic with exponential backoff
- Normalize quote format across brokers to common schema

**5.1.3 Order Management**
- Translate strategy signals to broker-specific order formats
- Support market, limit, stop, and trailing stop order types
- Implement order state tracking: pending, filled, partially filled, rejected, canceled
- Handle asynchronous fill notifications and update position state
- Reconcile internal position state with broker-reported positions periodically

**5.1.4 Account Monitoring**
- Query account equity, margin, and open positions at regular intervals
- Implement balance change alerts if unexpected equity movements detected
- Track realized and unrealized P&L separately
- Validate margin sufficiency before placing orders
- Detect and alert on margin calls or forced liquidations

**5.1.5 Error Handling and Resilience**
- Retry failed operations with exponential backoff and maximum retry limit
- Classify errors: transient (retry), permanent (alert and stop), broker-side (wait)
- Log all broker interactions with full request/response for debugging
- Implement circuit breaker: stop trading after consecutive failures exceed threshold
- Provide manual override: operator can disable autotrading immediately

**5.1.6 Test Mode and Simulation**
- Maintain simulated broker adapter for testing that doesn't require live credentials
- Simulated adapter uses historical data or random walk for price generation
- Clearly mark simulated mode in logs and UI to prevent confusion
- Require explicit configuration flag to enable live broker connection

---

### 5.2 Signal Generation and Execution Logic

**Objective**: Enhance decision-making to incorporate ensemble forecasts, risk limits, and position management.

**Logical Requirements**:

**5.2.1 Signal Aggregation**
- Collect forecasts from all available models in ensemble
- Compute ensemble prediction as weighted average (weights based on validation performance)
- Calculate ensemble disagreement: standard deviation of predictions across models
- Flag low-confidence situations: high disagreement or low individual model confidence
- Implement signal thresholds: enter long if ensemble > upper_threshold, short if < lower_threshold

**5.2.2 Position Sizing Integration**
- Apply selected position sizing algorithm (fixed, volatility-adjusted, Kelly)
- Respect maximum position size limits configured per instrument
- Adjust size based on account equity: larger account allows larger positions
- Reduce size if recent performance has been poor (dynamic risk management)
- Override size to zero if confidence below minimum threshold

**5.2.3 Entry Filtering**
- Check time-of-day constraints: avoid trading during low-liquidity periods
- Consult economic calendar: skip trades near major news releases
- Verify spread is within acceptable range: wide spreads invalidate entry
- Ensure no conflicting positions: don't enter long if already long
- Validate risk exposure: total portfolio risk within acceptable bounds

**5.2.4 Exit Logic**
- Monitor positions continuously for exit conditions
- Profit target: close position when unrealized profit exceeds target
- Stop loss: close position when unrealized loss exceeds stop
- Trailing stop: adjust stop loss as position moves favorably
- Signal reversal: close long if new signal is bearish (and vice versa)
- Time-based exit: close positions before weekend or low-liquidity periods

**5.2.5 Risk Management Overrides**
- Daily loss limit: stop trading if daily losses exceed threshold
- Maximum drawdown: stop trading if equity drops below percentage from peak
- Correlation check: avoid concentrated exposure in correlated instruments
- Manual emergency stop: operator can trigger immediate position flattening

---

### 5.3 Execution Quality Monitoring

**Objective**: Measure and optimize trade execution to minimize costs and slippage.

**Logical Requirements**:

**5.3.1 Fill Analysis**
- Record for every trade: intended price, fill price, fill timestamp, order type
- Compute slippage: fill_price - intended_price (negative for favorable, positive for unfavorable)
- Calculate average slippage by instrument, time of day, and market condition
- Track fill latency: time from signal generation to order fill
- Identify patterns: slippage worse during high volatility or at certain brokers

**5.3.2 Cost Attribution**
- Decompose total costs into: spread, slippage, commission, swap
- Calculate cost as percentage of position value for normalization
- Compare actual costs to backtest assumptions and alert if materially higher
- Identify cost outliers: trades with unusually high costs for investigation

**5.3.3 Execution Alerts**
- Alert if slippage on recent trades exceeds historical average by >2 standard deviations
- Flag if fill latency increases significantly (connectivity issue?)
- Notify if broker rejects orders (margin issue, invalid parameters?)
- Warn if spread widening reduces trade frequency (market condition change)

**5.3.4 Optimization Feedback Loop**
- Use execution data to calibrate backtest execution cost models
- Identify whether market or limit orders provide better net outcomes
- Determine optimal trade timing: early session vs mid-session vs close
- Inform position sizing: higher costs justify smaller positions

---

### 5.4 Paper Trading Mode

**Objective**: Validate autotrading logic with real market data but simulated execution.

**Logical Requirements**:

**5.4.1 Paper Trading Infrastructure**
- Subscribe to live market data from broker or data provider
- Generate signals using production forecasting models
- Simulate order execution using current bid/ask with realistic slippage model
- Track paper positions and P&L as if trades were real
- Do not transmit orders to broker

**5.4.2 Realism Enhancements**
- Apply same execution costs as live trading: spread, commission, slippage
- Respect position limits and risk constraints exactly as live mode
- Simulate fill delays: market orders fill immediately, limit orders may delay or not fill
- Model partial fills and order rejections based on historical frequencies

**5.4.3 Performance Tracking**
- Record paper trading results separately from live trading
- Compare paper trading P&L to backtest expectations
- Compute correlation between paper and backtest returns (high correlation validates backtest)
- Alert if paper trading significantly underperforms backtest (suggests model degradation or unrealistic assumptions)

**5.4.4 Transition Criteria**
- Define graduation requirements: e.g., 30 days paper trading with Sharpe >1.0
- Require paper trading drawdown remain below threshold
- Validate that execution quality metrics match backtest assumptions
- Obtain operator approval before enabling live trading

---

## Workstream 6: Monitoring and Auto-Retrain Orchestration

### 6.1 Model Drift Detection System

**Objective**: Activate drift detection infrastructure to identify when model predictions degrade.

**Logical Requirements**:

**6.1.1 Drift Detection Metrics**
- Distribution shift: compare feature distributions in production vs training data
- Use Kolmogorov-Smirnov test to detect significant distribution changes
- Prediction shift: compare recent prediction distributions to historical
- Performance degradation: monitor rolling forecast accuracy metrics
- Track separate drift scores per feature and aggregate for overall score

**6.1.2 Drift Scoring and Thresholds**
- Compute composite drift score: weighted combination of distribution and performance shifts
- Establish warning threshold (drift_score > 0.3) and critical threshold (drift_score > 0.5)
- Warning generates alert for review, critical triggers auto-retrain pipeline
- Configurable sensitivity: conservative (fewer retrains) vs aggressive (more responsive)

**6.1.3 Integration with Validation Pipeline**
- Weekly retrospective validation job computes drift metrics alongside performance
- Drift scores persisted to time-series database for trend analysis
- Dashboard visualization: drift score time series with threshold lines
- Alert notification: email/Slack message when threshold exceeded

**6.1.4 Drift Investigation**
- Detailed report identifying which features show largest distribution shifts
- Compare recent vs historical distributions with overlaid histograms
- Highlight potential causes: market regime change, data quality issue, external factor
- Provide recommendation: retrain, adjust features, or investigate further

---

### 6.2 Automated Retraining Pipeline

**Objective**: Wire existing auto-retrain infrastructure to enable hands-off model updates.

**Logical Requirements**:

**6.2.1 Retrain Trigger Logic**
- Trigger conditions: drift score exceeds critical threshold, scheduled periodic retrain (e.g., quarterly)
- Manual trigger: operator can force retrain on demand
- Cooldown period: prevent consecutive retrains within minimum interval (e.g., 7 days)
- Retrain suppression: disable during high-volatility events or market hours

**6.2.2 Training Job Orchestration**
- Auto-retrain pipeline invokes full training workflow with current configuration
- Use refreshed data: fetch latest historical data up to current date
- Apply same preprocessing, feature engineering, and validation procedures
- Train multiple model candidates using current hyperparameters
- Select best model based on validation metrics

**6.2.3 Model Staging and Validation**
- New model enters staging environment, not production
- Run extended validation: compare predictions to recent actuals
- Compute A/B metrics: new model vs current production model on same data
- Require new model to outperform production by minimum margin (e.g., 5% RMSE improvement)
- Manual approval gate: operator reviews validation report and approves promotion

**6.2.4 Promotion and Rollback**
- Upon approval, promote staged model to production
- Update model registry with new model metadata and version
- Route percentage of traffic to new model (gradual rollout)
- Monitor performance during rollout, ready to rollback if issues detected
- After successful rollout, archive old model but retain for potential rollback

**6.2.5 Retraining History and Audit**
- Log every retraining event: trigger reason, training duration, validation metrics
- Persist model lineage: which model version replaced which, when, and why
- Generate retraining report for stakeholders
- Track retraining frequency and success rate over time

---

### 6.3 A/B Testing and Model Comparison

**Objective**: Systematically compare model variants to identify improvements and regressions.

**Logical Requirements**:

**6.3.1 Traffic Splitting**
- Configure traffic split percentages: e.g., 80% to model A, 20% to model B
- Route incoming forecast requests randomly according to split
- Tag each request with model variant for analysis
- Support multi-way splits: A/B/C testing of three models simultaneously

**6.3.2 Metric Collection**
- Record predictions from both models for each request
- Track realized outcomes when available
- Compute per-variant metrics: accuracy, latency, confidence calibration
- Aggregate metrics daily and weekly for comparison

**6.3.3 Statistical Testing**
- Apply statistical significance test (t-test, Mann-Whitney) to determine if performance difference is real
- Require minimum sample size before declaring winner (avoid premature conclusions)
- Compute confidence intervals on metric differences
- Account for multiple testing: adjust significance threshold for multiple comparisons

**6.3.4 Decision Automation**
- If variant B significantly outperforms variant A, automatically promote B to 100% traffic
- If variant B underperforms, roll back to 100% variant A
- If results are inconclusive, extend test duration or increase traffic to variant B
- Notify operators of test outcomes and decisions

**6.3.5 Reporting and Visualization**
- Dashboard showing A/B test status: variants, traffic split, current metrics
- Comparison table: side-by-side metrics for each variant
- Time series plots: metric evolution during test period
- Confidence interval visualization: is the difference meaningful?

---

## Workstream 7: Data Semantics and Documentation

### 7.1 Data Quality Validation Framework

**Objective**: Prevent garbage-in-garbage-out by validating data integrity at ingestion.

**Logical Requirements**:

**7.1.1 OHLC Consistency Checks**
- Validate OHLC relationships: Low ≤ Open, Close ≤ High; Open, Close ∈ [Low, High]
- Detect impossible bars: Close significantly different from Open/High/Low (data corruption?)
- Flag suspicious bars: overnight gaps >5%, intrabar volatility >10%
- Check timestamp monotonicity: each bar timestamp > previous bar timestamp
- Verify timeframe consistency: bars at expected intervals (no missing bars, no duplicate timestamps)

**7.1.2 Volume and Spread Validation**
- Volume cannot be negative or implausibly large (e.g., >1000x typical)
- Detect constant volume across many bars (feed issue)
- Flag zero volume bars (rare in liquid markets)
- If spread data available, validate: Spread = Ask - Bid > 0
- Alert on spread outliers: >5x typical spread (liquidity crisis or data error)

**7.1.3 Cross-Instrument Consistency**
- For currency crosses, validate triangular arbitrage relationships approximately hold
- EUR/USD * USD/JPY ≈ EUR/JPY (allowing for small deviations due to spreads)
- Detect if instrument correlations break: typically correlated pairs suddenly uncorrelated (data issue)

**7.1.4 Provider Metadata Enrichment**
- Tag each data record with provider identifier and ingestion timestamp
- Maintain data quality metrics per provider: error rate, update frequency, staleness
- Generate provider comparison report: which provider has cleanest data
- Support switching providers if quality degrades

**7.1.5 Error Handling and Reporting**
- Validation failures logged with severity: warning (continue with caution) vs error (reject data)
- Generate daily data quality report summarizing validation results
- Alert on critical issues: extended data gaps, high error rates
- Provide data correction utilities: interpolate missing values, flag for manual review

---

### 7.2 Configuration Management and Validation

**Objective**: Prevent runtime failures due to misconfiguration.

**Logical Requirements**:

**7.2.1 Configuration Schema Definition**
- Define JSON schema specifying all configuration parameters and types
- Document each parameter: purpose, valid range, default value, examples
- Organize into sections: data, training, forecasting, backtesting, trading, monitoring
- Support environment-specific overrides: dev, staging, production

**7.2.2 Validation on Load**
- Validate configuration file against schema before application startup
- Check required parameters are present
- Verify parameter values are within valid ranges
- Detect deprecated parameters and issue warnings
- Fail fast with actionable error messages if configuration invalid

**7.2.3 Dependency Checking**
- Validate broker API credentials if autotrading enabled
- Check database connection if persistence enabled
- Verify model files exist at specified paths
- Ensure data directories are accessible and contain expected files

**7.2.4 Configuration Versioning**
- Tag configuration with version number
- Maintain changelog documenting configuration changes
- Support migration utilities to upgrade old configurations
- Warn if configuration version is outdated

---

### 7.3 User Documentation and Guides

**Objective**: Enable users to understand and operate the system confidently.

**Logical Requirements**:

**7.3.1 Quick Start Guide**
- Step-by-step tutorial for installing and running first training job
- Example commands for common workflows: train, forecast, backtest
- Screenshots of UI showing key features
- Troubleshooting section for common errors

**7.3.2 Conceptual Documentation**
- Explain system architecture: components and their interactions
- Describe data flow: ingestion → preprocessing → training → forecasting → trading
- Document key concepts: walk-forward validation, drift detection, conformal prediction
- Provide background on forex trading: pips, spreads, margin, leverage

**7.3.3 API Reference**
- Document all CLI arguments and configuration parameters
- Describe API endpoints: request/response formats, examples
- List supported models, features, and pattern types
- Explain output formats and interpretation

**7.3.4 Operational Procedures**
- Runbook for monitoring and responding to alerts
- Procedure for manually triggering retraining
- Steps for enabling/disabling autotrading
- Incident response playbook: what to do if trading losses spike

**7.3.5 FAQ and Best Practices**
- Address common questions: which timeframe to trade, how to set stop losses
- Recommend parameter values for different risk tolerances
- Explain limitations: tick volume semantics, forecast uncertainty
- Warn about pitfalls: overfitting, ignoring transaction costs

---

## Additional Critical Workstreams

## Workstream 8: System Observability and Debugging

### 8.1 Logging and Tracing

**Objective**: Ensure all system behavior is observable for debugging and auditing.

**Logical Requirements**:

**8.1.1 Structured Logging**
- Adopt structured logging format (JSON) for machine parsing
- Include standard fields in every log: timestamp, level, component, correlation_id
- Log key events: training starts/completes, forecasts generated, trades executed
- Parameterize log levels: DEBUG for development, INFO for production
- Implement log rotation to prevent disk exhaustion

**8.1.2 Correlation IDs**
- Generate unique ID for each forecast request and thread it through system
- Include correlation ID in all related logs: feature computation, model inference, signal generation, order execution
- Enable tracing request flow across components
- Facilitate debugging by filtering logs for specific request

**8.1.3 Audit Trail**
- Log all critical actions: model training, model promotion, configuration changes, manual interventions
- Record actor (user or system), timestamp, action, outcome
- Persist audit logs separately from application logs (immutability, retention)
- Support compliance requirements and post-incident analysis

---

### 8.2 Metrics and Alerting

**Objective**: Surface system health and performance through real-time metrics.

**Logical Requirements**:

**8.2.1 Application Metrics**
- Instrument code to emit metrics: forecast latency, training duration, backtest runtime
- Track error rates: failed API calls, rejected orders, data validation failures
- Monitor resource utilization: CPU, memory, disk, network
- Record business metrics: forecasts per day, trades per day, portfolio value

**8.2.2 Custom Dashboards**
- Build dashboards visualizing key metrics: forecast accuracy, trading P&L, drift scores
- Provide drill-down capability: click metric to see underlying data
- Support time range selection: view last hour, day, week, month
- Enable comparison: current period vs previous period

**8.2.3 Alerting Rules**
- Define alert conditions: metric exceeds threshold, error rate spikes, service unavailable
- Configure alert severity: info, warning, critical
- Route alerts to appropriate channels: email, Slack, PagerDuty
- Implement alert throttling to prevent notification storms

---

## Workstream 9: Security and Compliance

### 9.1 Secrets Management

**Objective**: Protect sensitive credentials from exposure.

**Logical Requirements**:

**9.1.1 Credential Storage**
- Never store API keys or passwords in code or configuration files
- Use environment variables or secure vaults (AWS Secrets Manager, HashiCorp Vault)
- Encrypt secrets at rest and in transit
- Rotate credentials periodically

**9.1.2 Access Control**
- Implement role-based access control (RBAC): separate roles for admin, operator, viewer
- Require authentication for all API endpoints and UI access
- Log all access attempts and authorization decisions
- Support multi-factor authentication for administrative actions

---

### 9.2 Regulatory Compliance

**Objective**: Ensure system meets financial regulation requirements.

**Logical Requirements**:

**9.2.1 Audit Requirements**
- Maintain immutable audit logs of all trades and orders
- Record decision rationale: which model, what forecast, what signal
- Support regulatory reporting: transaction records, position snapshots
- Implement data retention policies: 7 years for financial records

**9.2.2 Risk Disclosures**
- Document system limitations and risk factors
- Provide clear disclaimers: past performance not indicative of future results
- Warn users of leverage and margin risks
- Ensure users acknowledge risks before enabling autotrading

---

## Workstream 10: Testing and Quality Assurance

### 10.1 Automated Testing Strategy

**Objective**: Ensure code changes don't introduce regressions.

**Logical Requirements**:

**10.1.1 Unit Tests**
- Test individual functions and classes in isolation
- Mock external dependencies (broker APIs, databases)
- Achieve ≥80% code coverage for critical paths
- Run unit tests on every commit via CI/CD

**10.1.2 Integration Tests**
- Test interactions between components: training pipeline → model registry → inference engine
- Use test databases and brokers (paper trading mode)
- Validate end-to-end workflows: data ingestion → training → forecasting → signal generation
- Run integration tests nightly or before deployments

**10.1.3 Performance Tests**
- Benchmark forecast latency, training duration, backtest runtime
- Establish performance budgets: forecast must complete within 100ms
- Detect performance regressions: alert if latency increases >20%
- Load testing: ensure system handles peak request rates

**10.1.4 Validation Tests**
- Verify model predictions are deterministic given same inputs
- Check that forecast intervals contain expected percentage of outcomes
- Validate backtest metrics match manual calculations
- Ensure drift detection triggers under synthetic distribution shifts

---

### 10.2 Continuous Integration and Deployment

**Objective**: Automate testing and deployment to reduce manual errors.

**Logical Requirements**:

**10.2.1 CI Pipeline**
- Trigger on every commit to main branch
- Run linters and code formatters
- Execute unit test suite
- Report test results and coverage metrics
- Block merge if tests fail

**10.2.2 CD Pipeline**
- Deploy to staging environment automatically after CI passes
- Run integration and performance tests in staging
- Require manual approval before production deployment
- Deploy to production with blue-green or canary strategy
- Monitor for errors post-deployment, rollback if issues detected

---

## Integration Architecture

### System-Wide Coordination

**Objective**: Ensure all workstreams integrate cohesively into unified system.

**Logical Requirements**:

**Component Interfaces**
- Define clear APIs between components: training → model registry, forecast service → signal generator
- Use standard data formats: JSON for configs, Parquet for data, pickle for models
- Version all interfaces to manage breaking changes
- Document dependencies and calling conventions

**Data Flow Orchestration**
- Implement workflow orchestration (Apache Airflow, Prefect) for multi-stage pipelines
- Define DAGs for: data ingestion → training → validation → deployment
- Handle failures gracefully: retry transient errors, alert on permanent failures
- Support manual intervention: pause workflows, replay from checkpoint

**State Management**
- Centralize state in shared database or model registry
- Avoid local state that can diverge across instances
- Implement locking for concurrent access to shared resources
- Provide state inspection utilities for debugging

---

## Quality Gates and Acceptance Criteria

### Gate 1: Training Pipeline (Workstreams 1, 7)
**Criteria**:
- Walk-forward CV implemented with configurable strategies
- LightGBM and XGBoost train successfully and persist correctly
- Feature selection reduces dimensionality by 30-50% while maintaining performance
- Volume semantics documented and validation checks implemented
- Unit test coverage ≥80% for training module
- Documentation updated with CV and feature selection guidance

**Validation**:
- Run training pipeline on sample data with CV enabled
- Verify per-fold metrics are computed and aggregated
- Confirm new models load and inference without errors
- Check that volume validation detects injected anomalies

---

### Gate 2: Forecasting (Workstream 2)
**Criteria**:
- Multi-horizon forecasts differ per step (no replication)
- Prediction intervals provided for all forecasts
- Interval coverage ≥85% for 90% confidence level on validation data
- Retrospective validation job runs successfully
- API returns intervals in documented format

**Validation**:
- Generate forecasts for multiple horizons and verify non-replication
- Compute empirical interval coverage on historical data
- Confirm retrospective validation persists metrics to database
- Test API with sample requests and verify response format

---

### Gate 3: Backtesting (Workstream 3)
**Criteria**:
- Short positions simulated correctly with profit/loss computed accurately
- Volatility-aware slippage model implemented
- Sortino and Calmar ratios computed and displayed in reports
- Bootstrap analysis provides confidence intervals on Sharpe ratio
- Report generation completes without errors

**Validation**:
- Run backtest with long-only and long-short strategies, compare results
- Verify slippage increases during high volatility periods
- Check that Sortino ratio differs from Sharpe ratio as expected
- Review generated PDF report for completeness

---

### Gate 4: Pattern Integration (Workstream 4)
**Criteria**:
- Pattern events encoded as features in training data
- GA optimization completes and persists best parameters
- Pattern performance statistics accessible via API
- UI displays optimization progress and results

**Validation**:
- Train model with pattern features, verify feature importance shows patterns contribute
- Run GA optimization for one pattern type, confirm fitness improves
- Query pattern statistics API, confirm data returned
- Use UI to start optimization and view progress

---

### Gate 5: Autotrading (Workstream 5)
**Criteria**:
- Broker adapter connects to paper trading account successfully
- Orders placed and fills recorded accurately
- Execution quality metrics computed and logged
- Paper trading runs for 7 days with P&L correlation >0.7 to backtest

**Validation**:
- Connect to paper trading broker and verify quotes streaming
- Place test order and confirm fill recorded with correct details
- Review execution quality report for slippage and cost breakdown
- Compare paper trading results to backtest expectations

---

### Gate 6: Monitoring (Workstream 6)
**Criteria**:
- Drift detection computes scores and logs to database
- Auto-retrain triggers when drift threshold exceeded
- A/B test routes traffic and records variant-specific metrics
- Dashboard displays drift scores and A/B test status

**Validation**:
- Inject synthetic distribution shift and verify drift score increases
- Trigger manual retrain and confirm new model enters staging
- Configure A/B test and verify traffic split as expected
- Review dashboard and confirm visualizations render correctly

---

### Gate 7: Documentation (Workstream 7)
**Criteria**:
- Quick start guide enables new user to run training within 30 minutes
- API reference documents all endpoints and parameters
- Operational runbook provides clear response procedures for alerts

**Validation**:
- Fresh user follows quick start guide, confirms success
- Developer uses API reference to integrate system, confirms completeness
- Operator reviews runbook and confirms procedures are actionable

---

## Deployment Strategy

### Phase 1: Foundation (Months 1-2)
**Scope**: Workstreams 1, 7 (Training Pipeline, Documentation)
**Deliverables**:
- Hardened training pipeline with CV and new algorithms
- Data quality validation preventing bad data ingestion
- Comprehensive documentation enabling self-service

**Risk Mitigation**:
- Extensive testing before deploying new training code
- Feature flags to toggle between old and new training paths
- Rollback plan if training failures increase

---

### Phase 2: Forecasting (Month 3)
**Scope**: Workstream 2 (Forecast Reliability)
**Deliverables**:
- Multi-horizon forecasts with calibrated intervals
- Retrospective validation monitoring forecast quality

**Risk Mitigation**:
- Parallel run: generate forecasts with both old and new methods, compare
- Gradual API cutover: migrate consumers one at a time
- Monitor for forecast quality degradation post-deployment

---

### Phase 3: Evaluation (Month 4)
**Scope**: Workstreams 3, 4 (Backtesting, Patterns)
**Deliverables**:
- Realistic backtests informing strategy development
- Pattern features integrated into ML pipeline

**Risk Mitigation**:
- Backtests remain offline tool, low deployment risk
- Pattern features optional: can disable if performance degrades

---

### Phase 4: Autotrading (Months 5-6)
**Scope**: Workstream 5 (Autotrading Maturation)
**Deliverables**:
- Paper trading with full telemetry
- Execution quality monitoring

**Risk Mitigation**:
- Mandatory paper trading period before live trading allowed
- Manual approval gate for transitioning to live
- Circuit breakers to halt trading on anomalies

---

### Phase 5: Observability (Month 7)
**Scope**: Workstreams 6, 8 (Monitoring, Observability)
**Deliverables**:
- Drift detection and auto-retrain operational
- Comprehensive dashboards and alerting

**Risk Mitigation**:
- Monitor system stability during auto-retrain rollout
- Tune alert thresholds to reduce noise
- Ensure runbooks are validated before production

---

## Operational Procedures

### Incident Response Playbook

**High Forecast Error Rate**
1. Check data quality: any anomalies or gaps in recent data?
2. Review drift metrics: is distribution shift occurring?
3. Examine model performance: has validation accuracy dropped?
4. Decision: trigger manual investigation or auto-retrain

**Autotrading Loss Spike**
1. Immediately review open positions and recent trades
2. Check execution quality: is slippage unusually high?
3. Verify broker connectivity and quote accuracy
4. Decision: halt trading, close positions, or continue with increased monitoring

**Model Retraining Failure**
1. Review training logs for errors
2. Verify data availability and quality
3. Check resource availability: sufficient memory, disk, CPU?
4. Decision: retry with adjusted parameters, alert engineers, or skip retrain

**Drift Alert**
1. Review drift report: which features show largest shifts?
2. Correlate with market events: major news, regime change?
3. Assess forecast quality: has accuracy degraded?
4. Decision: proceed with auto-retrain, investigate further, or adjust thresholds

---

## Maintenance and Evolution

### Quarterly Reviews
- Assess system performance against objectives
- Identify bottlenecks and optimization opportunities
- Review technical debt and prioritize remediation
- Update documentation to reflect changes

### Annual Strategy
- Evaluate whether system meets business needs
- Consider major architecture upgrades
- Plan next-generation capabilities
- Budget for infrastructure and staffing

---

## Conclusion

This specification provides a comprehensive blueprint for evolving ForexGPT from prototype to production-ready platform. Each workstream defines clear objectives, detailed requirements, acceptance criteria, and risk mitigation strategies. Successful implementation requires disciplined execution, continuous validation, and adaptation based on operational experience.

The path forward emphasizes:
1. **Safety and Reliability**: Preventing financial loss through robust testing and monitoring
2. **Incremental Progress**: Delivering value at each phase rather than monolithic deployment
3. **Observability**: Ensuring system behavior is transparent and debuggable
4. **Quality Assurance**: Maintaining high standards through automated testing and review gates

With commitment to these principles and systematic execution of defined workstreams, ForexGPT will transform into a production system that confidently supports quantitative trading operations.
