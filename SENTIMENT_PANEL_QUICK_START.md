# SentimentPanel - Guida Rapida

## 🎯 Dove Trovare il Pannello

Il **SentimentPanel** si trova nel **ChartTab**, in un **pannello mobile a destra del grafico**, nella sezione inferiore (sotto Order Flow Panel):

```
┌─────────────┬──────────────────────┬─────────────────┐
│             │                      │ ╔═════════════╗ │
│ Market      │                      │ ║ 📊 ORDER    ║ │
│ Watch       │      GRAFICO         │ ║ FLOW PANEL  ║ │
│             │                      │ ║             ║ │
│             │                      │ ╠═════════════╣ │ ← Handle Movibile (BLU)
├─────────────┤                      │ ║ 💭 SENTIMEN ║ │
│             │                      │ ║ T PANEL     ║ │
│ Order       │                      │ ║             ║ │
│ Books       │  Orders Table (sotto)│ ║   ← QUI!    ║ │
│             │                      │ ╚═════════════╝ │
└─────────────┴──────────────────────┴─────────────────┘
                                       ↑
                                Panel Analisi Mobile
```

### Caratteristiche del Pannello

- **Frame visibile** grigio (#3a3a3a) attorno al pannello
- **Handle blu** (4px) tra Order Flow e Sentiment - **trascina per ridimensionare**
- **Larghezza minima**: 350px
- **Split iniziale**: 50% Order Flow, 50% Sentiment
- **Completamente ridimensionabile** trascinando i separatori

## 🔍 Il Pannello Non è Visibile?

Se non vedi il SentimentPanel, segui questi passi:

### 1. **Cerca il Frame Grigio a Destra del Grafico**
Il pannello analisi ha un **frame grigio (#3a3a3a)** ben visibile sul lato destro del grafico. Il SentimentPanel ha anche un **bordo blu (#0078d7)** interno.

### 2. **Trascina i Separatori dello Splitter**
Gli splitter possono essere collassati. Cerca le **linee sottili** tra i pannelli e trascinale per espandere lo spazio.

### 3. **Resetta lo Stato degli Splitter**
Se il pannello è stato nascosto da uno stato salvato precedente:

```bash
python reset_splitter_state.py
```

Poi riavvia ForexGPT.

## 📊 Cosa Mostra il Pannello

### Current Market Sentiment
- **Sentiment**: Bullish (verde) / Bearish (rosso) / Neutral (grigio)
- **Confidence**: 0-100% basato su quanto il posizionamento si discosta dal 50%
- **Sentiment Ratio**: -1.0 a +1.0
- **Traders**: Numero totale di posizioni (volume)

### Long/Short Positioning
- **Barre Progress** che mostrano la percentuale di posizioni Long vs Short
- **Color-coded**: Verde per Long, Rosso per Short

### Contrarian Signals
- **Indicatore -1.0 a +1.0**: 
  - **Positivo (+0.5 a +1.0)**: Crowd troppo short → Consider LONG
  - **Negativo (-0.5 a -1.0)**: Crowd troppo long → Consider SHORT
  - **Neutral**: Posizionamento bilanciato

### Trading Alerts
- **⚠️ Extreme Positioning**: Long o Short > 75%
- **✅ Contrarian Opportunity**: Segnale forte con alta confidence

## 🧪 Test con Dati Mock

Se non hai cTrader configurato o vuoi testare il pannello:

```bash
# 1. Genera 300 record di test
python create_test_sentiment_data.py

# 2. Avvia ForexGPT
python run_forexgpt.py

# 3. Vai al Chart tab e scorri verso il basso nel pannello di destra
```

Gli scenari di test sono:
- **EURUSD**: 65% Long (Moderately Bullish)
- **GBPUSD**: 35% Long (Moderately Bearish)
- **USDJPY**: 78% Long (Extremely Bullish - **CONTRARIAN SHORT SIGNAL!**)

## 🔄 Dati Real-Time

In produzione con cTrader:
1. Il `CTraderWebSocketService` processa l'order flow
2. Calcola sentiment da bid/ask volume imbalance
3. Salva in `data/forex_diffusion.db` → `sentiment_data` table
4. `SentimentAggregatorService` elabora i dati ogni 30 secondi
5. `SentimentPanel` si aggiorna automaticamente ogni 5 secondi

## 🎨 Personalizzazione

### Cambiare Simbolo
Usa il **dropdown "Symbol:"** in alto nel pannello per cambiare tra:
- EURUSD
- GBPUSD
- USDJPY
- AUDUSD
- USDCAD

### Dimensioni Iniziali
Le dimensioni di default sono:
- **Altezza minima**: 250px
- **Larghezza minima**: 200px

Puoi ridimensionare trascinando i separatori dello splitter.

## ⚙️ Configurazione Tecnica

### Stretch Factors (ui_builder.py)
```python
right_splitter.setStretchFactor(0, 6)  # Chart area (più spazio)
right_splitter.setStretchFactor(1, 1)  # Orders table
right_splitter.setStretchFactor(2, 2)  # Order Flow Panel
right_splitter.setStretchFactor(3, 2)  # Sentiment Panel
```

### Dimensioni Iniziali
```python
right_splitter.setSizes([400, 100, 200, 250])
```

## 🐛 Troubleshooting

### Il pannello appare ma è vuoto
- Verifica che `sentiment_service` sia collegato:
  ```python
  # Nel log all'avvio dovrebbe apparire:
  "Sentiment Aggregator Service started for ['EURUSD', 'GBPUSD', 'USDJPY']"
  ```
- Verifica dati nel database:
  ```bash
  python test_sentiment_panel.py
  ```

### Il pannello non si aggiorna
- Controlla che il timer sia attivo (auto-refresh ogni 5s)
- Verifica che ci siano dati recenti (< 1 ora) nel database
- Guarda i log per errori nel processing

### Errore "No DOM data available"
Questo è **normale**: il SentimentPanel usa `sentiment_data`, **non** `market_depth`.
Il sentiment viene calcolato dal **volume imbalance**, non dal DOM.

## 📚 Documentazione Completa

Per dettagli tecnici completi, vedi:
- `SENTIMENT_PANEL_IMPLEMENTATION.md` - Architettura e flusso dati
- `src/forex_diffusion/ui/sentiment_panel.py` - Codice sorgente UI
- `src/forex_diffusion/services/sentiment_aggregator.py` - Logica elaborazione

## ✅ Checklist Veloce

- [ ] Ho avviato ForexGPT
- [ ] Sono nella tab **Chart**
- [ ] Ho scrollato verso il basso nel pannello di destra
- [ ] Vedo un bordo blu con "Market Sentiment Analysis"
- [ ] Se non lo vedo, ho provato a trascinare i separatori dello splitter
- [ ] Se ancora non lo vedo, ho eseguito `python reset_splitter_state.py`

Se dopo questi passi non vedi ancora il pannello, apri un issue con screenshot.
