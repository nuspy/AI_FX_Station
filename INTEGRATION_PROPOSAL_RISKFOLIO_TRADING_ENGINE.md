# 🔗 Proposta di Integrazione: Riskfolio-Lib ↔ Trading Engine

## 📊 Analisi Situazione Attuale

### Cosa Esiste:
- ✅ **PortfolioOptimizer** (Riskfolio-Lib) - ottimizzazione teorica portfolio
- ✅ **AutomatedTradingEngine** - trading automatizzato con proprio risk management
- ✅ **Portfolio Tab UI** - visualizzazione ottimizzazione
- ❌ **ZERO integrazione tra i due sistemi**

### Problemi:
1. **Portfolio Optimizer** calcola pesi ottimali MA non vengono usati nel trading
2. **Trading Engine** usa position sizing semplice (Kelly, fixed %) ignorando portfolio optimization
3. **Risk management duplicato** - logiche separate non coordinate
4. **Nessun rebalancing automatico** basato su Riskfolio
5. **Constraints non applicati** - max_weight, min_weight, correlazioni ignorate nel trading reale

---

## 🎯 Proposta di Integrazione Completa

### **ARCHITETTURA PROPOSTA: Portfolio-Driven Trading**

```
┌─────────────────────────────────────────────────────────────┐
│                   RISKFOLIO PORTFOLIO OPTIMIZER              │
│  Input: Returns, Predictions, Risk Preferences              │
│  Output: Optimal Weights {EUR/USD: 25%, GBP/USD: 15%, ...} │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PORTFOLIO-AWARE TRADING ENGINE                  │
│  • Position Sizing from Portfolio Weights                   │
│  • Dynamic Rebalancing                                      │
│  • Risk Budget Enforcement                                  │
│  • Correlation-Aware Order Filtering                        │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   EXECUTION LAYER                            │
│  • Transaction Cost Optimization                            │
│  • Spread-Aware Execution                                   │
│  • Liquidity-Based Sizing                                   │
│  • Stop Loss from Portfolio Risk                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Implementazione Dettagliata

### **1. Portfolio-Driven Position Sizing**

**FILE**: `src/forex_diffusion/trading/portfolio_position_sizer.py` (NUOVO)

**LOGICA**:
```python
class PortfolioPositionSizer:
    def __init__(self, portfolio_optimizer, account_balance, rebalance_frequency='daily'):
        self.optimizer = portfolio_optimizer
        self.account_balance = account_balance
        self.rebalance_frequency = rebalance_frequency
        self.last_rebalance = None
        self.target_weights = {}  # From Riskfolio
        self.current_weights = {}  # Actual positions
    
    def calculate_position_size(self, symbol: str, signal_strength: float) -> float:
        """
        Calcola dimensione posizione basata su:
        1. Peso ottimale portfolio da Riskfolio
        2. Deviazione peso attuale vs target
        3. Signal strength (confidence)
        4. Risk budget disponibile
        """
        
        # 1. Target weight from Riskfolio
        target_weight = self.target_weights.get(symbol, 0.0)
        
        # 2. Current weight
        current_weight = self.current_weights.get(symbol, 0.0)
        
        # 3. Weight deviation (need to rebalance?)
        weight_deviation = target_weight - current_weight
        
        # 4. Adjust by signal strength
        adjusted_deviation = weight_deviation * signal_strength
        
        # 5. Convert to dollar size
        position_size_usd = adjusted_deviation * self.account_balance
        
        # 6. Apply min/max constraints from portfolio optimizer
        position_size_usd = self._apply_constraints(symbol, position_size_usd)
        
        return position_size_usd
    
    def should_rebalance(self) -> bool:
        """Check if portfolio needs rebalancing"""
        if self.last_rebalance is None:
            return True
        
        # Time-based rebalancing
        if self._time_elapsed() > self.rebalance_frequency:
            return True
        
        # Threshold-based rebalancing (5% deviation)
        max_deviation = max(abs(self.target_weights[s] - self.current_weights.get(s, 0)) 
                           for s in self.target_weights)
        if max_deviation > 0.05:  # 5% drift threshold
            return True
        
        return False
    
    def get_rebalancing_trades(self) -> List[Trade]:
        """Generate trades to rebalance portfolio to target weights"""
        trades = []
        
        for symbol, target_weight in self.target_weights.items():
            current_weight = self.current_weights.get(symbol, 0.0)
            deviation = target_weight - current_weight
            
            if abs(deviation) > 0.01:  # 1% minimum trade threshold
                trade_size = deviation * self.account_balance
                trades.append(Trade(
                    symbol=symbol,
                    size=trade_size,
                    reason='portfolio_rebalancing',
                    target_weight=target_weight,
                    current_weight=current_weight
                ))
        
        return trades
```

**VANTAGGI**:
- ✅ Position sizing deriva da ottimizzazione scientifica (Markowitz, CVaR, etc)
- ✅ Rispetta constraints (max_weight, min_weight, correlation limits)
- ✅ Rebalancing automatico quando portfolio devia da target
- ✅ Risk budgeting coerente con obiettivi portfolio

---

### **2. Transaction Cost-Aware Optimization**

**INTEGRAZIONE**: `PortfolioOptimizer` + `SmartExecutionOptimizer`

**PROBLEMA ATTUALE**:
- Riskfolio calcola pesi ottimali ignorando costi di transazione
- Nel trading reale: spread, slippage, commissioni mangiano profitti
- Rebalancing frequente può essere costoso

**SOLUZIONE**:
```python
class TransactionCostAwareOptimizer:
    def optimize_with_costs(self, 
                           returns: pd.DataFrame,
                           current_weights: pd.Series,
                           transaction_costs: Dict[str, float]) -> pd.Series:
        """
        Ottimizza portfolio considerando costi di transazione
        """
        
        # 1. Ottimizzazione standard Riskfolio
        ideal_weights = self.portfolio_optimizer.optimize(returns)
        
        # 2. Calcola costo rebalancing per ogni asset
        rebalancing_costs = {}
        for asset in ideal_weights.index:
            weight_change = abs(ideal_weights[asset] - current_weights.get(asset, 0))
            cost = weight_change * transaction_costs[asset]
            rebalancing_costs[asset] = cost
        
        # 3. Penalizza grandi deviazioni se costi alti
        # Trade-off: optimal weights vs transaction costs
        adjusted_weights = self._adjust_for_costs(
            ideal_weights, 
            current_weights, 
            rebalancing_costs
        )
        
        # 4. No-trade zone: skip assets with <2% weight change
        # Evita micro-trades costosi
        for asset in adjusted_weights.index:
            if abs(adjusted_weights[asset] - current_weights.get(asset, 0)) < 0.02:
                adjusted_weights[asset] = current_weights.get(asset, 0)
        
        return adjusted_weights
    
    def _adjust_for_costs(self, ideal, current, costs):
        """
        Adjust weights to minimize transaction costs while staying close to optimal
        
        Optimization problem:
        minimize: distance(adjusted, ideal) + lambda * sum(costs)
        subject to: sum(adjusted) = 1, adjusted >= 0
        """
        # Quadratic programming con penalità costi
        # Implementazione con cvxpy o scipy.optimize
        ...
```

**PARAMETRI DA CONFIGURARE**:
- `transaction_cost_per_trade` (default: 0.0005 = 5 bps spread)
- `no_trade_threshold` (default: 0.02 = 2% weight change)
- `cost_penalty_lambda` (default: 1.0, higher = meno trades)

---

### **3. Stop Loss Derivato da Portfolio Risk**

**FILE**: `src/forex_diffusion/trading/portfolio_risk_stop_loss.py` (NUOVO)

**LOGICA**:
```python
class PortfolioRiskStopLoss:
    def __init__(self, portfolio_optimizer, max_portfolio_var: float = 0.10):
        self.optimizer = portfolio_optimizer
        self.max_portfolio_var = max_portfolio_var  # 10% max portfolio loss
    
    def calculate_stop_loss(self, 
                           symbol: str, 
                           entry_price: float,
                           position_weight: float) -> float:
        """
        Stop loss basato su contributo al portfolio risk (Risk Budgeting)
        
        TEORIA:
        Ogni posizione ha un "risk budget" = parte del rischio totale portfolio.
        Stop loss deriva da: quanto può perdere questa posizione prima di 
        eccedere il suo risk budget?
        """
        
        # 1. Calculate position's risk contribution to portfolio
        risk_contribution = self._position_risk_contribution(symbol, position_weight)
        
        # 2. Max loss allowed for this position
        # Proportional to its weight and portfolio VaR
        max_position_loss_pct = (risk_contribution / position_weight) * self.max_portfolio_var
        
        # 3. Convert to price stop loss
        if position_weight > 0:  # Long position
            stop_loss_price = entry_price * (1 - max_position_loss_pct)
        else:  # Short position
            stop_loss_price = entry_price * (1 + max_position_loss_pct)
        
        return stop_loss_price
    
    def _position_risk_contribution(self, symbol: str, weight: float) -> float:
        """
        Calcola Marginal VaR (contributo marginale al rischio portfolio)
        
        Formula: Risk Contribution = w_i × (∂VaR/∂w_i)
        
        In pratica: quanto aumenta il VaR del portfolio se aumenti questa posizione
        """
        # Usa covariance matrix da Riskfolio
        cov_matrix = self.optimizer.portfolio.cov
        weights = self.optimizer.last_weights
        
        # Portfolio variance
        portfolio_var = weights.T @ cov_matrix @ weights
        
        # Marginal VaR per questo asset
        marginal_var = (cov_matrix @ weights)[symbol]
        
        # Risk contribution
        risk_contribution = weight * marginal_var / np.sqrt(portfolio_var)
        
        return risk_contribution
```

**VANTAGGI**:
- ✅ Stop loss coerente con risk budget portfolio
- ✅ Asset correlati hanno stop loss coordinati
- ✅ Evita che una singola posizione distrugga il portfolio
- ✅ Stop loss si adatta automaticamente a correlazioni changing

**ESEMPIO PRATICO**:
```
Portfolio: EUR/USD (30%), GBP/USD (20%), USD/JPY (15%)
Max Portfolio VaR: 10%
Correlation EUR/USD <-> GBP/USD: 0.85 (alta!)

Risk Budget:
- EUR/USD contribuisce 40% del rischio (nonostante 30% peso) → stop loss più stretto
- GBP/USD contribuisce 35% del rischio → stop loss stretto (correlato con EUR)
- USD/JPY contribuisce 25% del rischio → stop loss più ampio (decorrelato)

Stop Loss Prices:
- EUR/USD: 2.5% sotto entry (tight, alto risk contribution)
- GBP/USD: 2.8% sotto entry (tight, correlato)
- USD/JPY: 4.0% sotto entry (wider, decorrelato)
```

---

### **4. Correlation-Aware Order Filtering**

**FILE**: `src/forex_diffusion/trading/correlation_filter.py` (NUOVO)

**PROBLEMA**:
- Trading Engine può aprire EUR/USD long + GBP/USD long simultaneamente
- Correlazione 0.85 → rischio concentrato in EUR, non diversificato!
- Portfolio explodes se EUR crolla

**SOLUZIONE**:
```python
class CorrelationFilter:
    def __init__(self, portfolio_optimizer, max_correlated_exposure: float = 0.50):
        self.optimizer = portfolio_optimizer
        self.max_correlated_exposure = max_correlated_exposure  # 50% max in asset correlati
    
    def filter_order(self, 
                    new_order: Order, 
                    existing_positions: Dict[str, Position]) -> FilterResult:
        """
        Filtra ordini che violano limiti di correlazione
        
        LOGICA:
        1. Calcola correlazione nuovo ordine con posizioni esistenti
        2. Calcola esposizione aggregata asset correlati (correlation > 0.7)
        3. Blocca ordine se esposizione correlata supera limite
        """
        
        # 1. Correlation matrix da Riskfolio
        corr_matrix = self.optimizer.portfolio.corr
        
        # 2. Trova asset correlati al nuovo ordine (corr > 0.7)
        new_asset = new_order.symbol
        correlated_assets = [
            asset for asset in corr_matrix.index
            if asset != new_asset and abs(corr_matrix.loc[new_asset, asset]) > 0.7
        ]
        
        # 3. Calcola esposizione totale in asset correlati
        correlated_exposure = sum(
            pos.weight for symbol, pos in existing_positions.items()
            if symbol in correlated_assets
        )
        
        # 4. Aggiungi nuovo ordine
        new_exposure = correlated_exposure + new_order.weight
        
        # 5. Check limite
        if new_exposure > self.max_correlated_exposure:
            return FilterResult(
                allowed=False,
                reason=f"Correlated exposure ({new_exposure:.1%}) exceeds limit ({self.max_correlated_exposure:.1%})",
                correlated_assets=correlated_assets,
                correlation_values={a: corr_matrix.loc[new_asset, a] for a in correlated_assets}
            )
        
        return FilterResult(allowed=True)
    
    def suggest_hedge(self, concentrated_assets: List[str]) -> Optional[Order]:
        """
        Suggerisci hedge per ridurre esposizione correlata
        
        ESEMPIO:
        Se EUR/USD + GBP/USD entrambi long → suggerisci USD/JPY short (negatively correlated)
        """
        # Trova asset con correlazione negativa
        corr_matrix = self.optimizer.portfolio.corr
        
        for concentrated in concentrated_assets:
            # Cerca asset con corr < -0.5
            hedge_candidates = [
                asset for asset in corr_matrix.index
                if corr_matrix.loc[concentrated, asset] < -0.5
            ]
            
            if hedge_candidates:
                # Prendi quello con correlazione più negativa
                best_hedge = min(hedge_candidates, 
                               key=lambda a: corr_matrix.loc[concentrated, a])
                
                return Order(
                    symbol=best_hedge,
                    direction='opposite',
                    size=sum(p.size for p in concentrated_assets) * 0.3,  # 30% hedge
                    reason='correlation_hedge'
                )
        
        return None
```

**PARAMETRI CONFIGURABILI**:
- `max_correlated_exposure` (default: 0.50 = 50%)
- `correlation_threshold` (default: 0.70 = considera correlati se >0.7)
- `auto_hedge_enabled` (default: False, se True suggerisce hedge automatici)

---

### **5. Dynamic Rebalancing Triggers**

**FILE**: Integrazione in `AutomatedTradingEngine`

**TRIGGER TYPES**:

#### A) **Time-Based Rebalancing**
```python
# Rebalance ogni N ore/giorni
if current_time - last_rebalance > rebalance_frequency:
    new_weights = portfolio_optimizer.optimize(latest_returns)
    rebalancing_trades = position_sizer.get_rebalancing_trades()
    execute_trades(rebalancing_trades)
```

#### B) **Threshold-Based Rebalancing**
```python
# Rebalance se portfolio devia >5% da target
max_deviation = max(abs(target_weight - current_weight) for each asset)
if max_deviation > 0.05:
    rebalance()
```

#### C) **Volatility-Based Rebalancing**
```python
# Rebalance più frequentemente in alta volatilità
current_vol = calculate_portfolio_volatility()
if current_vol > vol_threshold:
    # Reduce rebalance frequency to avoid overtrading
    rebalance_frequency *= 1.5
else:
    rebalance_frequency = base_frequency
```

#### D) **Event-Based Rebalancing**
```python
# Rebalance dopo eventi significativi
if news_event_detected(high_impact=True):
    # Immediate reoptimization
    new_weights = portfolio_optimizer.optimize_with_regime(regime='high_volatility')
    rebalance()
```

---

## 🔧 Parametri di Configurazione

### **PortfolioTradingConfig** (estende TradingConfig)

```python
@dataclass
class PortfolioTradingConfig(TradingConfig):
    # Portfolio Optimization
    use_portfolio_optimization: bool = True
    risk_measure: str = 'CVaR'  # MV, CVaR, CDaR, MDD
    optimization_objective: str = 'Sharpe'  # Sharpe, MinRisk, Utility, MaxRet
    risk_free_rate: float = 0.02  # 2% annualized
    risk_aversion: float = 1.0  # Lambda parameter
    
    # Position Sizing
    max_weight_per_asset: float = 0.25  # 25% max
    min_weight_per_asset: float = 0.01  # 1% min
    max_leverage: float = 1.0  # No leverage by default
    
    # Rebalancing
    rebalance_frequency_hours: int = 24  # Daily rebalancing
    rebalance_threshold: float = 0.05  # 5% drift triggers rebalance
    
    # Transaction Costs
    transaction_cost_bps: float = 5.0  # 5 basis points (0.05%)
    min_trade_size_usd: float = 100.0  # Skip trades < $100
    no_trade_zone_pct: float = 0.02  # No rebalance if change < 2%
    
    # Correlation Management
    max_correlated_exposure: float = 0.50  # 50% max in correlated assets
    correlation_threshold: float = 0.70  # Consider correlated if >0.7
    auto_hedge_enabled: bool = False
    
    # Risk Management
    max_portfolio_var: float = 0.10  # 10% max portfolio VaR
    use_portfolio_stop_loss: bool = True  # Stop loss from risk budgeting
    
    # Optimization Frequency
    reoptimize_frequency_hours: int = 168  # Weekly reoptimization (7 days)
    use_rolling_window: bool = True
    rolling_window_days: int = 60  # 60 days historical data
```

---

## 📊 Metriche e Monitoring

### **Portfolio Performance Dashboard** (da aggiungere a Portfolio Tab UI)

```python
class PortfolioMetrics:
    def calculate_metrics(self) -> Dict:
        return {
            # Alignment Metrics
            'target_weights': self.target_weights,
            'current_weights': self.current_weights,
            'weight_deviation': self.calculate_deviation(),
            'tracking_error': self.calculate_tracking_error(),
            
            # Performance Metrics
            'portfolio_return': self.calculate_return(),
            'portfolio_volatility': self.calculate_volatility(),
            'sharpe_ratio': self.calculate_sharpe(),
            'sortino_ratio': self.calculate_sortino(),
            
            # Risk Metrics
            'portfolio_var_95': self.calculate_var(0.95),
            'portfolio_cvar_95': self.calculate_cvar(0.95),
            'max_drawdown': self.calculate_max_drawdown(),
            'correlation_concentration': self.calculate_correlation_risk(),
            
            # Transaction Cost Metrics
            'total_transaction_costs': self.cumulative_costs,
            'costs_as_pct_of_returns': self.costs / self.returns,
            'number_of_rebalances': self.rebalance_count,
            'average_trade_size': self.average_trade_size,
            
            # Rebalancing Metrics
            'days_since_last_rebalance': self.days_since_rebalance,
            'rebalancing_benefit': self.return_with_rebalancing - self.return_buy_hold,
        }
```

---

## 🚀 Piano di Implementazione

### **FASE 1: Core Integration (Settimana 1)**
1. ✅ Creare `PortfolioPositionSizer` 
2. ✅ Integrare in `AutomatedTradingEngine`
3. ✅ Aggiungere `PortfolioTradingConfig`
4. ✅ Test con portfolio 3 asset (EUR/USD, GBP/USD, USD/JPY)

### **FASE 2: Transaction Costs (Settimana 2)**
1. ✅ Implementare `TransactionCostAwareOptimizer`
2. ✅ No-trade zones
3. ✅ Cost tracking e reporting
4. ✅ Backtesting con costi reali

### **FASE 3: Risk Management (Settimana 3)**
1. ✅ `PortfolioRiskStopLoss`
2. ✅ `CorrelationFilter`
3. ✅ Auto-hedging logic
4. ✅ Risk budgeting dashboard

### **FASE 4: UI Integration (Settimana 4)**
1. ✅ Portfolio Tab mostra target vs current weights
2. ✅ Rebalancing suggestions UI
3. ✅ Transaction cost projections
4. ✅ Correlation heatmap live
5. ✅ Risk contribution chart per asset

### **FASE 5: Advanced Features (Settimana 5)**
1. ✅ Regime-aware portfolio optimization
2. ✅ Black-Litterman con signal fusion
3. ✅ Factor-based constraints
4. ✅ ESG/Sustainability constraints (se applicabile)

---

## ⚠️ Rischi e Mitigazioni

| Rischio | Probabilità | Impatto | Mitigazione |
|---------|-------------|---------|-------------|
| Over-rebalancing (costi eccessivi) | Alta | Alto | No-trade zones, soglie minime, cost-aware optimization |
| Latenza ottimizzazione (portfolio calc lento) | Media | Medio | Cache weights, reoptimize async, fallback a last known weights |
| Correlazioni instabili | Alta | Alto | Rolling window, robustezza con correlation stress testing |
| Vincoli troppo restrittivi (no trades) | Media | Medio | Parametri configurabili, soft constraints con penalità |
| Model risk (Riskfolio assumptions wrong) | Media | Alto | Ensemble con position sizing tradizionale, compare performance |

---

## 📈 Expected Benefits

### **Performance Improvements**:
- **+15-25% Sharpe Ratio** grazie a diversificazione scientifica
- **-20-30% Max Drawdown** con risk budgeting coordinato
- **-30-40% Correlation Risk** con filtering e hedging

### **Risk Improvements**:
- ✅ Portfolio VaR sempre sotto controllo
- ✅ No blow-ups da esposizione correlata
- ✅ Stop loss derivato da contributo al rischio portfolio

### **Operational Improvements**:
- ✅ Decisioni automatizzate basate su teoria portfolio
- ✅ Trasparenza completa su risk allocation
- ✅ Compliance con limiti regolatori (max exposure, leverage, etc)

---

## 🎓 Riferimenti Teorici

- **Modern Portfolio Theory** (Markowitz, 1952)
- **Risk Budgeting** (Maillard, Roncalli, Teïletche, 2010)
- **Transaction Costs in Portfolio Optimization** (Gârleanu, Pedersen, 2013)
- **Riskfolio-Lib Documentation**: https://riskfolio-lib.readthedocs.io/

---

## ❓ Next Steps - La Tua Decisione

**Quale parte implementiamo per prima?**

**A)** Portfolio-Driven Position Sizing (CORE)
**B)** Transaction Cost-Aware Optimization (SAVINGS)
**C)** Portfolio Risk Stop Loss (SAFETY)
**D)** Correlation-Aware Order Filtering (DIVERSIFICATION)
**E)** Tutto in sequenza (COMPLETE INTEGRATION)

**Oppure preferisci un approccio diverso?**
