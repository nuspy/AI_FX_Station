# Meccanismo di Trading Engine e Backtesting

## 1. AUTOMATED TRADING ENGINE

### Architettura
```
AutomatedTradingEngine
├── Trading Loop (thread separato)
├── Componenti ML/AI
│   ├── Multi-timeframe ensemble
│   ├── Stacked ML ensemble
│   └── Regime detector
├── Risk Management
│   ├── Multi-level stop loss
│   ├── Regime position sizer
│   └── Daily loss limiter
└── Execution
    └── Smart execution optimizer
```

### Ciclo di Trading (Loop Principale)

**Ogni X secondi** (configurabile, default 60s):

```python
while not stopped:
    1. FETCH MARKET DATA
       ↓
    2. MANAGE EXISTING POSITIONS
       - Aggiorna trailing stops
       - Controlla 6 tipi di stop loss
       - Chiudi se triggered
       ↓
    3. CHECK NEW OPPORTUNITIES (se sotto max_positions)
       - Genera segnale da ensemble
       - Rileva regime corrente
       - Calcola position size
       - Apri posizione se segnale valido
       ↓
    4. UPDATE METRICS
       - Account balance
       - Daily P&L
       ↓
    sleep(60 secondi)
```

### Generazione Segnale

**Multi-Timeframe Ensemble**:
```python
# Per ogni timeframe (5m, 15m, 1h, 4h, 1d, 1w)
for tf in timeframes:
    prediction = model[tf].predict(data[tf])
    confidence = model[tf].confidence

# Weighted voting
weights = accuracy × confidence × regime_multiplier
final_signal = weighted_sum(predictions, weights)

# Require consensus (60%+)
if consensus >= 0.6:
    return signal  # +1 (long), -1 (short)
else:
    return 0  # No trade
```

### Position Sizing

**Regime-Aware + Kelly Criterion**:
```python
# 1. Base risk (es: 1% del capitale)
base_risk = account_balance × 0.01

# 2. Regime multiplier
if regime == TRENDING_UP and signal == LONG:
    multiplier = 1.2  # Aumenta size
elif regime == VOLATILE:
    multiplier = 0.5  # Riduce size

# 3. Confidence adjustment
confidence_adj = 0.5 + (confidence × 0.5)

# 4. Kelly Criterion
win_rate = 0.55
avg_win = 100
avg_loss = 50
kelly_fraction = (win_rate × avg_win - (1-win_rate) × avg_loss) / avg_win

# 5. Final size
position_size = base_risk × multiplier × confidence_adj × kelly_fraction
position_size = min(position_size, max_risk_per_trade)
```

### Multi-Level Stop Loss (6 Tipi)

**Priority order** (da più alta a più bassa):

```python
1. DAILY_LOSS (3% del capitale/giorno)
   → Chiude TUTTE le posizioni

2. CORRELATION_STOP
   → Se correlazione mercato < -0.7 (rischio sistemico)

3. TIME_STOP
   → Max 24 ore di holding

4. VOLATILITY_STOP (ATR-based)
   → Entry ± (2 × ATR)

5. TECHNICAL_STOP
   → Pattern invalidation (es: supporto rotto)

6. TRAILING_STOP
   → Segue il prezzo quando in profitto
```

### Esempio Pratico di Trade

```python
# T=0: Segnale di LONG su EURUSD
Signal: +1, Confidence: 75%
Price: 1.1000
Regime: TRENDING_UP

# Calculate position size
stop_distance = 1.1000 × 0.02 = 0.0220 (2%)
stop_loss = 1.1000 - 0.0220 = 1.0780

base_size = 10000 / (1.1000 - 1.0780) = 454 units
regime_mult = 1.2 (trending up)
final_size = 454 × 1.2 × 0.75 = 408 units

# Entry execution (smart execution optimizer)
spread = 0.0002 (2 pips)
entry_cost = 408 × 1.1000 × 0.0002 = $0.09
execution_price = 1.1002 (price + spread/2)

Position opened:
- Entry: 1.1002
- Size: 408 units
- Stop: 1.0780
- Take Profit: 1.1440 (2:1 risk/reward)

# T=1h: Price = 1.1050 (+48 pips)
trailing_stop = 1.1050 - 0.0220 = 1.0830 (aggiornato)

# T=2h: Price = 1.1080 (+78 pips)
trailing_stop = 1.1080 - 0.0220 = 1.0860 (aggiornato)

# T=3h: Price drops to 1.1060
trailing_stop triggered at 1.0860? No, price still above

# T=4h: Price = 1.1020
trailing_stop triggered at 1.0860? No

# T=5h: Signal reverses to -1 (SHORT)
→ Close position at 1.1020

P&L = (1.1020 - 1.1002) × 408 = $7.34
Exit cost = $0.09
Net P&L = $7.34 - 0.09 - 0.09 = $7.16
```

### Controllo del Trading Engine

```python
from src.forex_diffusion.trading import (
    AutomatedTradingEngine,
    TradingConfig
)

# 1. Configurazione
config = TradingConfig(
    symbols=['EURUSD', 'GBPUSD'],
    timeframes=['5m', '15m', '1h'],
    update_interval_seconds=60,
    max_positions=5,
    account_balance=10000.0,
    risk_per_trade_pct=1.0,
    use_multi_timeframe=True,
    use_stacked_ensemble=True,
    use_regime_detection=True,
    use_smart_execution=True
)

# 2. Inizializzazione
engine = AutomatedTradingEngine(
    config=config,
    broker_api=your_broker_api  # Optional
)

# 3. Carica modelli pre-trained
engine.load_models(
    mtf_ensemble=trained_mtf_ensemble,
    ml_ensemble=trained_ml_ensemble,
    regime_detector=trained_regime_detector
)

# 4. Start trading
engine.start()

# 5. Monitora status
status = engine.get_status()
print(f"State: {status['state']}")
print(f"Balance: ${status['account_balance']}")
print(f"Open positions: {status['open_positions']}")

# 6. Pausa/Riprendi
engine.pause()   # Mantiene posizioni aperte
engine.resume()  # Riprende trading

# 7. Stop (chiude tutte le posizioni)
engine.stop()
```

---

## 2. INTEGRATED BACKTEST SYSTEM

### Architettura Walk-Forward

```
Historical Data (es: 1 anno)
├── Window 1: Train 30d → Test 7d
├── Window 2: Train 30d → Test 7d (spostato 7d)
├── Window 3: Train 30d → Test 7d (spostato 7d)
└── ... (finché dati disponibili)

Per ogni window:
1. TRAIN models su training period
2. TEST su test period (simulazione trading)
3. RECORD trades e performance
```

### Processo per Ogni Window

```python
# TRAINING PHASE
train_data = data[0:8640]  # 30 giorni × 288 candles (5min)

# Train stacked ensemble
ensemble = StackedMLEnsemble()
ensemble.fit(features_train, labels_train)

# Train regime detector
regime_detector = HMMRegimeDetector()
regime_detector.fit(data_train)

# TESTING PHASE (simulazione trading)
test_data = data[8640:10656]  # 7 giorni × 288

for i in range(len(test_data)):
    current_price = test_data[i]['close']

    # 1. Manage existing positions
    for position in open_positions:
        check_stops(position, current_price)
        update_trailing(position, current_price)

    # 2. Check new entries
    if len(positions) < max_positions:
        prediction = ensemble.predict(features[i])
        regime = regime_detector.predict(data[i-100:i])

        if valid_signal(prediction, confidence):
            size = position_sizer.calculate(price, regime)
            open_position(price, size, regime)
```

### Simulazione Realistica

**Transaction Costs**:
```python
# ENTRY
entry_price = market_price × (1 + spread/2)
entry_cost = size × price × (spread + commission)

Example:
- Market: 1.1000
- Spread: 0.0002 (2 pips)
- Commission: 0.0001 (0.01%)
→ Execution: 1.1001
→ Cost: 408 × 1.1000 × 0.0003 = $0.13

# EXIT (stesso processo)
exit_price = market_price × (1 - spread/2)
exit_cost = size × price × (spread + commission)

# NET P&L
gross_pnl = (exit_price - entry_price) × size
net_pnl = gross_pnl - entry_cost - exit_cost
```

**Slippage** (se abilitato):
```python
# Market impact model
size_ratio = order_size / avg_volume
slippage = base_slippage × sqrt(size_ratio) × volatility
execution_price += slippage
```

### Tracking Dettagliato

**Per ogni trade registra**:
```python
Trade {
    entry_time: 2024-01-15 10:30:00
    entry_price: 1.1002
    exit_time: 2024-01-15 15:45:00
    exit_price: 1.1020

    size: 408
    direction: "long"

    # Context
    entry_regime: "trending_up"
    entry_confidence: 0.75

    # Costs
    entry_cost: 0.13
    exit_cost: 0.13
    slippage: 0.02

    # P&L
    gross_pnl: 7.34
    net_pnl: 7.06

    # Analytics
    mae: 0.0015  # Maximum Adverse Excursion (quanto è andato contro)
    mfe: 0.0082  # Maximum Favorable Excursion (quanto è andato a favore)
    holding_time_hours: 5.25

    # Exit
    exit_reason: "signal_reversal"
    stop_type: "trailing_stop"
}
```

### Metriche Calcolate

**1. Performance Metrics**:
```python
Win Rate = winning_trades / total_trades
Profit Factor = total_wins / total_losses
Expectancy = (win_rate × avg_win) + ((1-win_rate) × avg_loss)

Example:
- 100 trades: 60 wins, 40 losses
- Avg win: $50, Avg loss: $30
→ Win Rate = 60%
→ Profit Factor = (60×50) / (40×30) = 2.5
→ Expectancy = (0.6×50) + (0.4×-30) = $18/trade
```

**2. Risk Metrics**:
```python
# Sharpe Ratio (annualized)
returns = [trade.net_pnl for trade in trades]
sharpe = (mean(returns) / std(returns)) × sqrt(252)

# Max Drawdown
equity_curve = cumulative_sum(returns)
running_max = maximum.accumulate(equity_curve)
drawdowns = running_max - equity_curve
max_dd = max(drawdowns)
max_dd_pct = (max_dd / initial_capital) × 100

# Sortino Ratio (penalizza solo downside)
downside_returns = [r for r in returns if r < 0]
sortino = (mean(returns) / std(downside_returns)) × sqrt(252)

# Calmar Ratio
calmar = annual_return / max_dd_pct
```

**3. Regime Analysis**:
```python
regime_performance = {
    "trending_up": {
        trades: 25,
        win_rate: 68%,
        avg_pnl: $45,
        total_pnl: $1125
    },
    "ranging": {
        trades: 40,
        win_rate: 45%,
        avg_pnl: -$5,
        total_pnl: -$200
    },
    "volatile": {
        trades: 15,
        win_rate: 40%,
        avg_pnl: -$10,
        total_pnl: -$150
    }
}

→ Insight: Sistema performa bene in trending, male in ranging/volatile
```

### Output Finale

```python
BacktestResult {
    # Capital
    initial: $10,000
    final: $12,450
    return: $2,450
    return_pct: 24.5%

    # Trades
    total: 120
    wins: 68 (56.7%)
    losses: 52

    # Performance
    sharpe: 1.85
    sortino: 2.31
    profit_factor: 2.1
    expectancy: $20.42

    # Risk
    max_drawdown: $845 (8.45%)
    calmar: 2.90

    # Time
    avg_holding: 8.5 hours

    # Costs
    total_costs: $156
    avg_cost_per_trade: $1.30

    # Files generated
    - summary.json
    - trades.csv (tutti i 120 trades)
    - equity_curve.csv
}
```

### Utilizzo del Backtest System

```python
from src.forex_diffusion.backtest import (
    IntegratedBacktester,
    BacktestConfig
)
from datetime import datetime

# 1. Configurazione
config = BacktestConfig(
    symbol='EURUSD',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1),
    timeframes=['5m', '15m', '1h'],

    # Components
    use_multi_timeframe=True,
    use_stacked_ensemble=True,
    use_regime_detection=True,
    use_smart_execution=True,

    # Capital
    initial_capital=10000.0,
    max_positions=3,
    base_risk_per_trade_pct=1.0,

    # Risk
    use_multi_level_stops=True,
    max_holding_hours=24,
    daily_loss_limit_pct=3.0,

    # Costs
    spread_pct=0.0002,      # 2 pips
    commission_pct=0.0001,  # 0.01%
    slippage_pct=0.0001,    # 0.01%

    # Execution
    min_signal_confidence=0.6,

    # Walk-forward
    train_size_days=30,
    test_size_days=7,
    step_size_days=7
)

# 2. Inizializzazione
backtester = IntegratedBacktester(config)

# 3. Run backtest
result = backtester.run(
    data=ohlcv_data,
    features=calculated_features,
    labels=target_labels,
    verbose=True
)

# 4. Analizza risultati
print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
print(f"Win Rate: {result.win_rate:.2%}")
print(f"Profit Factor: {result.profit_factor:.2f}")

# 5. Regime performance
for regime, perf in result.regime_performance.items():
    print(f"\n{regime}:")
    print(f"  Trades: {perf['trades']}")
    print(f"  Win Rate: {perf['win_rate']:.2%}")
    print(f"  Avg P&L: ${perf['avg_pnl']:.2f}")

# 6. Salva risultati
backtester.save_results(result, output_path='backtest_results/')
# Genera:
# - backtest_results/summary.json
# - backtest_results/trades.csv
# - backtest_results/equity_curve.csv

# 7. Analizza trade individuali
for trade in result.trades[:10]:  # Prime 10 trades
    print(f"\nTrade {trade.trade_id}:")
    print(f"  {trade.direction.upper()} @ {trade.entry_price:.5f}")
    print(f"  Exit @ {trade.exit_price:.5f}")
    print(f"  P&L: ${trade.net_pnl:.2f}")
    print(f"  Regime: {trade.entry_regime}")
    print(f"  Holding: {trade.holding_time_hours:.1f}h")
    print(f"  Exit reason: {trade.exit_reason}")
```

---

## 3. DIFFERENZE E COMPLEMENTARIETÀ

### Differenze Chiave

| Aspetto | Trading Engine | Backtest |
|---------|---------------|----------|
| **Dati** | Real-time (broker API) | Storico (database) |
| **Esecuzione** | Threading continuo | Loop su finestre temporali |
| **Training** | Modelli pre-trained | Re-train ogni window |
| **Costi** | Real broker costs | Simulati realisticamente |
| **Slippage** | Reale | Modellato |
| **Scope** | Trading live | Validazione strategia |
| **Rischio** | Capitale reale | Simulato |
| **Feedback** | Immediato | Post-analisi |

### Componenti Condivisi (Identici)

Entrambi i sistemi usano **esattamente gli stessi componenti**:

1. **Multi-Timeframe Ensemble**
   - Stesso algoritmo di weighted voting
   - Stesse soglie di consensus
   - Stessi timeframes

2. **Stacked ML Ensemble**
   - Stessi 5 base models
   - Stesso meta-learner
   - Stessa architettura

3. **Regime Detector**
   - Stesso HMM con 4 regimi
   - Stessi criteri di classificazione
   - Stessi multipliers

4. **Multi-Level Stop Loss**
   - Stessi 6 tipi di stop
   - Stesso priority order
   - Stessi parametri ATR

5. **Regime Position Sizer**
   - Stesso Kelly Criterion
   - Stessi regime multipliers
   - Stesso Risk Parity

6. **Smart Execution Optimizer**
   - Stessi modelli di costo
   - Stesso time-of-day optimization
   - Stessi parametri TWAP/VWAP

### Workflow Completo

```
1. BACKTEST (Validazione)
   ↓
   Risultati: Sharpe 1.85, Win Rate 56.7%, Max DD 8.45%
   ↓
   Decisione: Strategia valida ✓
   ↓
2. TRAINING (Produzione)
   ↓
   Train modelli su ultimi 3 mesi di dati
   ↓
3. TRADING ENGINE (Live)
   ↓
   Deploy con modelli trained
   ↓
   Monitoring continuo
   ↓
4. PERIODIC RE-BACKTEST
   ↓
   Ogni mese: nuovo backtest con dati aggiornati
   ↓
   Se performance degrada → Retrain → Redeploy
```

---

## 4. ESEMPIO COMPLETO END-TO-END

### Step 1: Backtest della Strategia

```python
# Backtest su 1 anno di dati storici
config = BacktestConfig(
    symbol='EURUSD',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1),
    initial_capital=10000,
    train_size_days=30,
    test_size_days=7
)

backtester = IntegratedBacktester(config)
result = backtester.run(data, features, labels)

# Risultati
print(f"Return: {result.total_return_pct:.2f}%")  # 24.5%
print(f"Sharpe: {result.sharpe_ratio:.2f}")       # 1.85
print(f"Win Rate: {result.win_rate:.2%}")         # 56.7%
print(f"Max DD: {result.max_drawdown_pct:.2f}%")  # 8.45%

# Decision: Deploy? ✓ (Sharpe > 1.5, Win Rate > 50%, Max DD < 15%)
```

### Step 2: Train Modelli per Produzione

```python
# Train su ultimi 3 mesi (dati più recenti)
recent_data = data[-25920:]  # 90 giorni × 288 candles
recent_features = features[-25920:]
recent_labels = labels[-25920:]

# Train ensemble
ensemble = StackedMLEnsemble(n_folds=5)
ensemble.fit(recent_features, recent_labels)

# Train regime detector
regime_detector = HMMRegimeDetector(n_regimes=4)
regime_detector.fit(recent_data)

# Save models
ensemble.save('models/production_ensemble.pkl')
regime_detector.save('models/production_regime.pkl')
```

### Step 3: Deploy Trading Engine

```python
# Load models
ensemble = StackedMLEnsemble.load('models/production_ensemble.pkl')
regime_detector = HMMRegimeDetector.load('models/production_regime.pkl')

# Configure engine
config = TradingConfig(
    symbols=['EURUSD'],
    timeframes=['5m', '15m', '1h'],
    account_balance=10000,
    max_positions=3,
    risk_per_trade_pct=1.0
)

# Initialize
engine = AutomatedTradingEngine(config, broker_api=my_broker)
engine.load_models(
    ml_ensemble=ensemble,
    regime_detector=regime_detector
)

# Start trading
engine.start()
print("✓ Trading engine started")
```

### Step 4: Monitoring

```python
# Check status periodicamente
while True:
    status = engine.get_status()

    print(f"\n=== Status at {datetime.now()} ===")
    print(f"State: {status['state']}")
    print(f"Balance: ${status['account_balance']:,.2f}")
    print(f"Open Positions: {status['open_positions']}")
    print(f"Total Trades: {status['total_trades']}")
    print(f"Daily P&L: ${status['daily_pnl']:.2f}")

    # Check positions
    for pos in status['positions']:
        print(f"  {pos['symbol']} {pos['direction']} @ {pos['entry_price']:.5f}")

    time.sleep(3600)  # Check ogni ora
```

### Step 5: Periodic Re-validation

```python
# Ogni mese: nuovo backtest
monthly_config = BacktestConfig(
    symbol='EURUSD',
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    initial_capital=engine.current_capital
)

monthly_result = backtester.run(recent_data, recent_features, recent_labels)

# Check performance
if monthly_result.sharpe_ratio < 1.0:
    print("⚠️ Performance degraded - consider retraining")
    engine.pause()

    # Retrain models
    new_ensemble = StackedMLEnsemble()
    new_ensemble.fit(features_last_3_months, labels_last_3_months)

    # Reload and resume
    engine.load_models(ml_ensemble=new_ensemble)
    engine.resume()
    print("✓ Models retrained and reloaded")
else:
    print("✓ Performance stable - continue trading")
```

---

## 5. VANTAGGI DEL SISTEMA INTEGRATO

### 1. Validazione Robusta
- Walk-forward previene overfitting
- Costi realistici (spread, commission, slippage)
- Regime analysis identifica quando funziona
- MAE/MFE tracking per ottimizzazione stops

### 2. Deployment Sicuro
- Stessi componenti tra backtest e live
- Parametri validati su dati storici
- Multi-level risk management
- Daily loss limits

### 3. Adattabilità
- Regime detection per adattamento dinamico
- Position sizing varia con market conditions
- Smart execution ottimizza timing
- Periodic retraining

### 4. Trasparenza
- Ogni trade tracciato completamente
- Breakdown dettagliato costi
- Performance per regime
- Equity curve completa

### 5. Scalabilità
- Multi-symbol support
- Multi-timeframe analysis
- Parallel window processing (backtest)
- Broker API agnostic

---

## 6. BEST PRACTICES

### Backtest
1. **Usa walk-forward**: Mai train e test sugli stessi dati
2. **Include costi realistici**: Spread + commission + slippage
3. **Molteplici metriche**: Non solo return, ma anche Sharpe, DD, win rate
4. **Regime analysis**: Capisci QUANDO la strategia funziona
5. **Out-of-sample**: Riserva ultimi 20% dati per final validation

### Live Trading
1. **Start small**: Inizia con capitale ridotto
2. **Monitor continuously**: Check status ogni ora
3. **Respect limits**: Daily loss limit è critico
4. **Log everything**: Salva tutti i trades per analisi
5. **Periodic revalidation**: Backtest mensile con dati recenti
6. **Paper trading first**: Testa in demo prima di live

### Risk Management
1. **Position sizing**: Mai oltre 2% del capitale per trade
2. **Max positions**: Limita esposizione totale (3-5 positions)
3. **Correlation check**: Evita posizioni altamente correlate
4. **Drawdown limits**: Stop se DD > 15%
5. **Regime awareness**: Riduci size in volatile/ranging

### Model Management
1. **Regular retraining**: Ogni 1-3 mesi
2. **Version control**: Salva ogni versione dei modelli
3. **A/B testing**: Compara nuovi vs vecchi modelli
4. **Ensemble diversification**: Mantieni diversi model types
5. **Feature monitoring**: Track feature importance over time

---

## 7. TROUBLESHOOTING

### Backtest Issues

**Problema**: Win rate basso (<40%)
```python
# Soluzioni:
1. Aumenta min_signal_confidence (0.6 → 0.7)
2. Aggiungi più features
3. Retrain con più dati
4. Check regime - forse strategia non funziona in ranging
```

**Problema**: Max drawdown alto (>20%)
```python
# Soluzioni:
1. Riduci risk_per_trade_pct (1.0% → 0.5%)
2. Abilita correlation_stop
3. Riduci max_positions
4. Aumenta stop loss distance
```

**Problema**: Sharpe ratio basso (<1.0)
```python
# Soluzioni:
1. Filtra trades con bassa confidence
2. Usa regime_aware sizing più aggressivo
3. Ottimizza stops (reduce false exits)
4. Check transaction costs (potrebbero essere troppo alti)
```

### Live Trading Issues

**Problema**: Engine si ferma frequentemente
```python
# Check:
1. Broker API connection
2. Daily loss limit non troppo basso
3. Log errors nel trading loop
4. Data fetching funziona correttamente
```

**Problema**: Posizioni non si aprono
```python
# Debug:
1. Check signal generation (confidence >= threshold?)
2. Verify regime detection funziona
3. Check capital sufficiente
4. Verify max_positions non raggiunto
5. Log predictions e decisions
```

**Problema**: Stop loss troppo frequenti
```python
# Soluzioni:
1. Aumenta ATR multiplier per volatility stops
2. Disabilita time_stop se troppo aggressivo
3. Review trailing stop parameters
4. Check se regime detection corretto
```

---

## 8. FILE LOCATIONS

```
ForexGPT/
├── src/forex_diffusion/
│   ├── trading/
│   │   ├── __init__.py
│   │   └── automated_trading_engine.py  # Trading Engine
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── integrated_backtest.py       # Backtest System
│   │
│   ├── models/
│   │   ├── multi_timeframe_ensemble.py
│   │   └── ml_stacked_ensemble.py
│   │
│   ├── regime/
│   │   └── hmm_detector.py
│   │
│   ├── risk/
│   │   ├── multi_level_stop_loss.py
│   │   └── regime_position_sizer.py
│   │
│   └── execution/
│       └── smart_execution.py
│
└── AUTOTRADING_AND_BACKTESTING.md        # Questa documentazione
```

---

## CONCLUSIONE

Il sistema fornisce:

✅ **Automated Trading Engine** completo e production-ready
✅ **Integrated Backtest System** con walk-forward validation
✅ **Componenti condivisi** per coerenza backtest↔live
✅ **Risk management** multi-livello
✅ **Performance tracking** dettagliato
✅ **Regime awareness** per adattamento dinamico

Pronto per deployment su dati reali con appropriato risk management.
