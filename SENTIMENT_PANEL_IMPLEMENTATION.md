# Sentiment Panel - Implementazione Completa

## 📊 Panoramica

Il **SentimentPanel** è ora completamente integrato e funzionante in ForexGPT. Visualizza metriche di market sentiment in real-time derivate dal order flow di cTrader.

## ✅ Componenti Implementati

### 1. **SentimentAggregatorService** (`src/forex_diffusion/services/sentiment_aggregator.py`)
- ✅ Servizio background che elabora dati sentiment ogni 30 secondi
- ✅ Legge da tabella `sentiment_data` popolata da CTraderWebSocketService
- ✅ Calcola metriche aggregate: long/short %, contrarian signals, confidence
- ✅ Cache in-memory per performance ottimali
- ✅ Fallback automatico a database quando cache vuota

### 2. **SentimentPanel** (`src/forex_diffusion/ui/sentiment_panel.py`)
- ✅ Pannello UI già esistente nel ChartTab (right splitter)
- ✅ Visualizza:
  - **Current Sentiment**: Bullish/Bearish/Neutral con color coding
  - **Long/Short Positioning**: Progress bars con percentuali
  - **Contrarian Signals**: Indicatore per fade opportunities
  - **Trading Alerts**: Extreme positioning, sentiment shifts, opportunities
- ✅ Auto-refresh ogni 5 secondi
- ✅ Selector per cambiare simbolo (EURUSD, GBPUSD, USDJPY, etc.)

### 3. **Database Schema** (`migrations/versions/0015_add_sentiment_data.py`)
- ✅ Tabella `sentiment_data` con campi:
  - `symbol`, `ts_utc`, `long_pct`, `short_pct`
  - `total_traders`, `confidence`, `sentiment`, `ratio`
  - `buy_volume`, `sell_volume`, `provider`
- ✅ Indici ottimizzati: `idx_sentiment_symbol_ts`, `idx_sentiment_ts`

### 4. **Integrazione CTrader** (`src/forex_diffusion/services/ctrader_websocket.py`)
- ✅ `_update_sentiment()`: Calcola sentiment da bid/ask volume imbalance
- ✅ `_store_sentiment()`: Salva dati nel database (non-blocking)
- ✅ Classificazione: bullish (ratio > 0.3), bearish (< -0.3), neutral
- ✅ Confidence score basato su magnitude del ratio

## 🔄 Flusso Dati

```
cTrader WebSocket
       ↓
   Order Flow (bid/ask volumes)
       ↓
_update_sentiment() → calcola ratio
       ↓
sentiment_data table (SQLite)
       ↓
SentimentAggregatorService (ogni 30s)
       ↓
get_latest_sentiment_metrics()
       ↓
SentimentPanel (refresh ogni 5s)
       ↓
UI Visualization
```

## 🎯 Metriche Visualizzate

### Sentiment Classification
- **Bullish**: Long > 60% (verde)
- **Bearish**: Long < 40% (rosso)
- **Neutral**: 40-60% (grigio)

### Contrarian Signals
- **Positive (+0.5 to +1.0)**: Crowd troppo short → Consider LONG
- **Negative (-0.5 to -1.0)**: Crowd troppo long → Consider SHORT
- **Neutral (-0.3 to +0.3)**: Posizionamento bilanciato

### Alerts
1. **Extreme Positioning**: Long o Short > 75%
2. **Sentiment Shift**: Cambio significativo (future implementation)
3. **Contrarian Opportunity**: |Signal| > 0.5 && Confidence > 0.6

## 🧪 Testing

### Test Scripts
1. **`test_sentiment_panel.py`**
   - Verifica database schema
   - Testa SentimentAggregatorService
   - Mostra formato dati UI-compatible
   
2. **`create_test_sentiment_data.py`**
   - Genera 300 record di test (100 per simbolo)
   - Scenari realistici: EURUSD bullish, GBPUSD bearish, USDJPY contrarian
   - Utile per development senza connessione cTrader

### Esecuzione Test
```bash
# Genera dati di test
python create_test_sentiment_data.py

# Verifica funzionamento
python test_sentiment_panel.py
```

## 📍 Posizione nell'UI

Il **SentimentPanel** si trova nel **ChartTab**, nello **splitter di destra**, sotto l'**OrderFlowPanel**.

**Percorso UI**: Chart → Right Panel → Sentiment Analysis

## 🔧 Configurazione

### Avvio Automatico
Il servizio viene avviato automaticamente in `app.py`:

```python
sentiment_aggregator = SentimentAggregatorService(
    engine=db_service.engine,
    symbols=["EURUSD", "GBPUSD", "USDJPY"],
    interval_seconds=30
)
sentiment_aggregator.start()
```

### Shutdown Automatico
Incluso nel graceful shutdown di ForexGPT:
```python
if sentiment_aggregator: 
    sentiment_aggregator.stop()
```

## 📊 Formato Dati

### Output `get_latest_sentiment_metrics()`
```python
{
    "sentiment": "bullish",        # bullish/bearish/neutral
    "confidence": 0.51,            # 0.0-1.0
    "ratio": 0.51,                 # -1.0 to +1.0
    "total_traders": 1000,         # Volume count
    "long_pct": 75.7,              # 0-100
    "short_pct": 24.3,             # 0-100
    "contrarian_signal": -0.51     # -1.0 to +1.0
}
```

## 🚀 Utilizzo

### In Real-Time con cTrader
1. Avvia ForexGPT con cTrader configurato
2. CTraderWebSocketService processa order flow
3. Sentiment generato automaticamente da volume imbalance
4. SentimentPanel si aggiorna ogni 5 secondi

### Con Dati di Test
1. `python create_test_sentiment_data.py`
2. Avvia ForexGPT
3. Apri Chart tab → visualizza pannello Sentiment
4. Cambia simbolo con dropdown per vedere scenari diversi

## 🎨 Stile UI

### Color Coding
- **Bullish**: Verde (`#D4EDDA` / `#155724`)
- **Bearish**: Rosso (`#F8D7DA` / `#721C24`)
- **Neutral**: Grigio (`#E0E0E0` / `#333`)

### Alerts
- **Extreme**: Giallo (`#FFF3CD` / `#856404`)
- **Shift**: Blu (`#D1ECF1` / `#0C5460`)
- **Opportunity**: Verde (`#D4EDDA` / `#155724`)

## 📝 Commits Correlati

- `9480f88` - feat: Attiva SentimentPanel con SentimentAggregatorService
- `3b29095` - test: Aggiunti script test per SentimentPanel e creazione dati mock

## 🔮 Future Enhancements

1. **Sentiment History Chart**: Grafico trend sentiment ultimi 60 minuti
2. **Multi-Provider Sentiment**: Integrazione news sentiment, social media
3. **Sentiment Shift Detection**: Alert su cambi rapidi di sentiment
4. **Correlation Analysis**: Sentiment vs price action correlation
5. **Custom Thresholds**: Configurazione user-defined per contrarian levels

## ✅ Status

**IMPLEMENTATO E TESTATO** ✅

Il SentimentPanel è completamente funzionante e pronto per l'uso in produzione. Tutti i test passano e l'integrazione con cTrader order flow è operativa.
