# ForexGPT - Action Plan Summary
**Date:** October 6, 2025  
**Based on:** Complete System Review

---

## Quick Assessment

**Overall System Rating:** ⭐⭐⭐½☆ (3.5/5)

**Production Ready:** ❌ NO - Requires 9-15 months of work

**Strengths:**
- ✅ Excellent architecture (multi-provider, causal features, walk-forward validation)
- ✅ Comprehensive feature engineering (50+ indicators, multi-timeframe)
- ✅ Advanced volume analysis (Profile, VSA, Smart Money)
- ✅ Proper GPU acceleration for neural encoders

**Critical Issues:**
- ❌ Never validated with live trading
- ❌ Pattern optimization incomplete
- ❌ No model performance monitoring
- ❌ Simplified forecasting (constant return assumption)
- ❌ No uncertainty quantification

---

## IMMEDIATE FIXES (Week 1-2)

### 1. Fix Volume Labeling (2 hours)
**File:** `src/forex_diffusion/training/train_sklearn.py`
```python
# Change line ~80
df['tick_volume'] = df['volume']  # Rename to be explicit

# Add comment
# NOTE: This is TICK VOLUME, not real traded volume
# Forex has no centralized exchange, so this is a proxy
```

### 2. Add Model Monitoring (1 week)
**New file:** `src/forex_diffusion/monitoring/model_validator.py`
```python
def validate_weekly_predictions():
    """Compare last week's predictions with actual returns"""
    predictions = fetch_predictions(days=7)
    actuals = fetch_actual_returns(predictions)
    
    mae = mean_absolute_error(actuals, predictions['value'])
    
    if mae > threshold * 1.5:
        logger.critical(f"Model degraded! MAE={mae:.6f}")
        send_email_alert("admin@forexgpt.com", "Model Degradation Alert")
        
    # Save metrics
    save_metric('weekly_mae', mae)
    save_metric('weekly_sharpe', calculate_sharpe(actuals - predictions))
```

**Integration:** Add cron job to run every Monday.

### 3. Fix Forecast Documentation (1 hour)
**File:** `src/forex_diffusion/ui/workers/forecast_worker.py`
```python
# Add clear warning at top of file:
# WARNING: Current forecasting assumes constant returns per step.
# This is a SIMPLIFICATION. Real returns vary.
# 
# For production use, implement:
# 1. Multi-output models (predict N returns for N horizons)
# 2. Autoregressive forecasting
# 3. Recurrent models (LSTM/GRU)
```

---

## HIGH PRIORITY (Month 1-2)

### 4. Complete Pattern Training Tab (7-11 hours)
**Status:** Skeleton only, ~2500 lines of code available

**Steps:**
1. Extract UI code from commit `11d3627`
2. Integrate genetic algorithm
3. Add progress tracking
4. Test with chart patterns
5. Test with candlestick patterns

**Files to modify:**
- `src/forex_diffusion/ui/pattern_training_tab.py` (add 2500 lines)
- `src/forex_diffusion/training/optimization/genetic_algorithm.py` (verify exists)

**Estimated time:** 2 days

### 5. Implement Uncertainty Quantification (2 weeks)
**New dependency:** `pip install mapie`

**New file:** `src/forex_diffusion/postproc/conformal_prediction.py`
```python
from mapie.regression import MapieRegressor

class ConformalPredictor:
    def __init__(self, base_model, alpha=0.05):
        self.mapie = MapieRegressor(estimator=base_model, method="plus")
        self.alpha = alpha
    
    def fit(self, X_train, y_train):
        self.mapie.fit(X_train, y_train)
    
    def predict_with_intervals(self, X_test):
        y_pred, y_intervals = self.mapie.predict(X_test, alpha=self.alpha)
        return {
            'prediction': y_pred,
            'lower_95': y_intervals[:, 0, 0],
            'upper_95': y_intervals[:, 1, 0]
        }
```

**Integration points:**
- `train_sklearn.py`: Wrap model with ConformalPredictor
- `forecast_worker.py`: Return confidence intervals
- `chart_tab/`: Display intervals as shaded area

**Estimated time:** 2 weeks

### 6. Add Feature Importance Validation (1 week)
**New file:** `src/forex_diffusion/analysis/feature_analysis.py`
```python
import shap

def analyze_feature_importance(model, X_train, feature_names):
    """
    Validate which features actually matter.
    
    Returns:
        DataFrame with columns: [feature, importance, rank]
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    importance = np.abs(shap_values).mean(axis=0)
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    df['rank'] = range(1, len(df) + 1)
    
    # Flag volume features
    df['is_volume'] = df['feature'].str.contains('volume|vp_|vsa_|sm_')
    
    return df
```

**Usage:**
```bash
python -m forex_diffusion.analysis.feature_analysis \
  --model_path artifacts/models/EURUSD_1m_d60_h30_rf_pca10.pkl \
  --output_dir analysis/
```

**Estimated time:** 1 week

---

## MEDIUM PRIORITY (Month 3-6)

### 7. Add Gradient Boosting Models (2 weeks)
**New dependency:** `pip install lightgbm xgboost catboost`

**Modify:** `src/forex_diffusion/training/train_sklearn.py`
```python
def _fit_model(algo: str, Xtr, ytr, args):
    if algo == "lightgbm":
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.random_state
        )
    elif algo == "xgboost":
        return XGBRegressor(...)
    elif algo == "catboost":
        return CatBoostRegressor(...)
```

### 8. Implement LSTM Baseline (3 weeks)
**New file:** `src/forex_diffusion/training/train_lstm.py`
```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class ForexLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat.squeeze(), y)
        self.log('train_loss', loss)
        return loss
```

### 9. Implement Walk-Forward Cross-Validation (1 week)
**Modify:** `src/forex_diffusion/training/train_sklearn.py`
```python
from sklearn.model_selection import TimeSeriesSplit

def _walk_forward_cv(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        score = mean_absolute_error(y_val, model.predict(X_val))
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

### 10. Add Feature Selection (1 week)
**New file:** `src/forex_diffusion/features/feature_selector.py`
```python
from sklearn.feature_selection import RFE, SelectKBest, f_regression

class FeatureSelector:
    def __init__(self, method='rfe', n_features=50):
        self.method = method
        self.n_features = n_features
    
    def fit_transform(self, X, y, estimator=None):
        if self.method == 'rfe':
            selector = RFE(estimator, n_features_to_select=self.n_features)
        elif self.method == 'kbest':
            selector = SelectKBest(f_regression, k=self.n_features)
        
        X_selected = selector.fit_transform(X, y)
        self.selected_features_ = selector.get_support(indices=True)
        return X_selected
```

---

## VALIDATION PHASE (Month 5-10)

### 11. Paper Trading Setup (Month 5)
**New file:** `src/forex_diffusion/trading/paper_trader.py`
```python
class PaperTrader:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.positions = []
        self.trades = []
    
    def execute_signal(self, signal):
        """
        Execute trading signal with simulated slippage/spread.
        
        Log everything to database for later analysis.
        """
        # Calculate position size (1% risk per trade)
        risk_amount = self.capital * 0.01
        position_size = risk_amount / signal.stop_distance
        
        # Simulate execution
        entry_price = signal.price + self.estimate_slippage()
        
        # Record trade
        trade = Trade(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=entry_price,
            size=position_size,
            stop=signal.stop,
            target=signal.target,
            timestamp=datetime.now()
        )
        
        self.trades.append(trade)
        self.positions.append(trade)
        
        # Log to database
        db.insert_trade(trade)
```

**Run paper trading for 3-6 months:**
```bash
# Start paper trading daemon
python -m forex_diffusion.trading.paper_trader \
  --model artifacts/models/best_model.pkl \
  --capital 10000 \
  --risk_per_trade 0.01 \
  --log_path paper_trading.db
```

### 12. Weekly Performance Reports (Months 5-10)
**Automated report generation:**
```python
def generate_weekly_report():
    """
    Generate PDF report with:
    - Predictions vs Actuals scatter plot
    - MAE trend over time
    - Sharpe ratio trend
    - Drawdown chart
    - Feature importance changes
    """
    pass
```

**Email to team every Monday.**

### 13. Performance Threshold Gates
**Define success criteria for going live:**
```python
PRODUCTION_GATES = {
    'min_sharpe': 1.5,           # Annualized Sharpe > 1.5
    'max_drawdown': 0.15,        # Max DD < 15%
    'min_win_rate': 0.52,        # Win rate > 52%
    'min_profit_factor': 1.5,    # Gross profit / loss > 1.5
    'min_weeks_profitable': 24,  # At least 6 months
    'max_weeks_consecutive_loss': 4
}

def check_production_ready():
    metrics = calculate_paper_trading_metrics()
    
    for gate, threshold in PRODUCTION_GATES.items():
        if not meets_threshold(metrics[gate], threshold):
            logger.warning(f"Failed gate: {gate}")
            return False
    
    return True
```

---

## PRODUCTION PHASE (Month 11-15)

### 14. Live Trading with Minimal Capital (Month 11)
```python
# Start with $1000-2000
# Max 1% risk per trade = $10-20 per trade

live_trader = LiveTrader(
    capital=1000,
    max_risk_per_trade=0.01,
    max_daily_loss=0.03,
    broker='ctrader'
)
```

**Monitor daily for first month:**
- Check all fills vs expected
- Measure actual slippage
- Validate latency
- Track spreads during different sessions

### 15. Gradual Capital Scaling (Months 12-15)
```python
CAPITAL_SCALING_RULES = {
    'increase_by': 1000,          # Add $1000 each time
    'min_weeks_profitable': 4,    # Must be profitable 4 weeks
    'max_dd_threshold': 0.10,     # Max DD < 10%
    'min_sharpe_threshold': 1.2   # Sharpe > 1.2
}

def should_increase_capital():
    last_4_weeks = get_metrics(weeks=4)
    
    if all([
        last_4_weeks['sharpe'] > 1.2,
        last_4_weeks['max_dd'] < 0.10,
        last_4_weeks['net_pnl'] > 0
    ]):
        return True
    
    return False
```

---

## CRITICAL WARNINGS

### ❌ DO NOT SKIP THESE STEPS

1. **Never skip paper trading**
   - Backtests ≠ Real performance
   - Must validate with live data first
   - Minimum 6 months paper trading

2. **Never trade without monitoring**
   - Implement weekly validation FIRST
   - Set up alerts for degradation
   - Manual review of all signals initially

3. **Never use full capital immediately**
   - Start with 1-2% of intended capital
   - Scale up slowly over 6-12 months
   - Be prepared to lose it all (learning cost)

4. **Never ignore risk management**
   - Max 1-2% risk per trade
   - Max 5% daily loss limit
   - Stop trading if DD > 20%

---

## SUCCESS METRICS

### Phase 1: Foundation (Weeks 1-8)
- ✅ Pattern training tab completed
- ✅ Model monitoring implemented
- ✅ Uncertainty quantification added
- ✅ Volume labeling fixed

### Phase 2: Model Diversity (Weeks 9-16)
- ✅ LightGBM/XGBoost added
- ✅ LSTM baseline implemented
- ✅ Feature selection working
- ✅ Walk-forward CV implemented

### Phase 3: Validation (Weeks 17-43)
- ✅ 6 months paper trading completed
- ✅ Sharpe > 1.5 over 6 months
- ✅ Max DD < 15%
- ✅ Win rate > 52%

### Phase 4: Production (Weeks 44-65)
- ✅ Live trading started with $1000-2000
- ✅ 3 months profitable
- ✅ Capital scaled to $5000-10000
- ✅ Automated monitoring stable

---

## RESOURCES NEEDED

### People
- 1 ML Engineer (full-time, Months 1-4)
- 1 Trader (part-time, Months 5-15, for validation)
- 1 DevOps (part-time, for monitoring setup)

### Infrastructure
- GPU server for training (RTX 4090 or better)
- Database for tracking (already have SQLite)
- Monitoring tools (Grafana + Prometheus)
- Paper trading account (cTrader demo)
- Live trading account (start with $1000-2000)

### Software
- `mapie` (conformal prediction)
- `lightgbm`, `xgboost`, `catboost` (gradient boosting)
- `pytorch` (LSTM)
- `shap` (feature importance)

---

## ESTIMATED COSTS

### Development (Months 1-4)
- ML Engineer: $15,000/month × 4 = **$60,000**
- DevOps (20% time): $3,000/month × 2 = **$6,000**
- Infrastructure: $500/month × 4 = **$2,000**
- **Subtotal: $68,000**

### Validation (Months 5-10)
- Trader (50% time): $10,000/month × 6 = **$60,000**
- Infrastructure: $500/month × 6 = **$3,000**
- **Subtotal: $63,000**

### Production Start (Months 11-15)
- Trader (full-time): $15,000/month × 5 = **$75,000**
- Infrastructure: $500/month × 5 = **$2,500**
- Trading capital (learning cost): **$5,000** (assume total loss)
- **Subtotal: $82,500**

**TOTAL ESTIMATED COST: $213,500**

---

## TIMELINE SUMMARY

| Phase | Duration | Key Deliverables | Cost |
|-------|----------|-----------------|------|
| Foundation | Weeks 1-8 | Monitoring, patterns, uncertainty | $34k |
| Model Diversity | Weeks 9-16 | GBM, LSTM, feature selection | $34k |
| Paper Trading | Weeks 17-43 | 6 months validation | $63k |
| Live Trading | Weeks 44-65 | Production with $1-10k capital | $82.5k |
| **TOTAL** | **15 months** | **Production-ready system** | **$213.5k** |

---

## FINAL RECOMMENDATION

**Current Status:** Excellent prototype, NOT production-ready

**Path Forward:**
1. Complete immediate fixes (Weeks 1-2)
2. Build foundation (Weeks 3-8)
3. Expand models (Weeks 9-16)
4. Validate with paper trading (Weeks 17-43)
5. Deploy to production gradually (Weeks 44-65)

**Risk Assessment:**
- High chance of success in development phases (80%)
- Medium chance of passing paper trading (50-60%)
- Low-medium chance of profitability in production (30-40%)

**Key Success Factors:**
1. Don't skip paper trading
2. Implement proper monitoring
3. Start with minimal capital
4. Be prepared to iterate and learn
5. Have realistic expectations (Forex is hard!)

---

**END OF ACTION PLAN**
