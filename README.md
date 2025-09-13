MagicForex — guida rapida (locale)
# MagicForex – Training e Forecast

Questo progetto fornisce una pipeline completa per:
- acquisizione dati (realtime e storici),
- feature engineering multi‑timeframe,
- addestramento modelli,
- previsione con bande di incertezza e calibrazione,
- ottimizzazione genetica (GA, NSGA‑II) degli iperparametri.

Requisiti principali:
- Python 3.12
- Librerie indicate in requirements (scikit‑learn, numpy, pandas, PySide6, ecc.)

## Panoramica UI

- Tab “Chart”
  - Grafico con toolbar; zoom/pan non si resettano agli update.
  - Overlay di previsioni q05/q50/q95 con legenda, colori deterministici per “fonte”.
  - Pulsante “Backfill Missing” con progress bar determinata; esclusione automatica dei weekend; ricarica del grafico a completamento.
  - “Clear Forecasts”, impostazioni di prediction (peso modello, indicatori×TF), auto‑previsioni, Alt+Click (basic) e Shift+Alt+Click (advanced) per “TestingPoint”.
- Tab “Training”
  - Selezione Symbol/TF base, giorni di storico, horizon in barre.
  - Griglia “Indicatori × Timeframe” con checkbox per i TF pertinenti a ciascun indicatore, bottoni “Default” per riga; la scelta è salvata tra sessioni.
  - Scelta Model (ridge/lasso/elasticnet/rf) ed Encoder (none/pca/latents).
  - Parametri pipeline (warmup, atr_n, rsi_n, bb_n, hurst_window, rv_window).
  - Ottimizzazione:
    - none: singolo training;
    - genetic-basic: GA single‑objective (max R2);
    - nsga2: multi‑obiettivo con ordinamento non‑dominato e crowding distance (min [-R2, MAE]).
  - Lancio asincrono con log in tempo reale e progress bar determinata (gens×pop).

Il menu “Model → Train” porta direttamente alla tab di Training.

## Forecast multi‑modello e multi‑tipo

Nel dialog “Prediction Settings”:
- Modelli multipli: inserire i percorsi (uno per riga). Se valorizzati, la previsione verrà calcolata per ciascun modello.
- Tipi di previsione:
  - Basic: pipeline standard con ensure e standardizzazione;
  - Advanced: usa parametri avanzati (indicatori estesi);
  - Baseline RW: Random Walk baseline (deterministico q50), utile come confronto.
- È possibile combinare più modelli con più tipi simultaneamente; ogni combinazione produce un overlay con etichetta “nomeModello:tipo” e colore coerente.

Suggerimenti:
- Impostare “Model weight (%)” per attenuare/rafforzare l’uscita del modello rispetto al last_close.
- Limitare il numero massimo di overlay visibili per mantenere il grafico leggibile (settings).

## Workflow scientifico

1. Dati e pre‑processing
   - Timestamp UTC, esclusione weekend nelle richieste storiche.
   - Feature causali (r, ATR, RSI, MACD, Bollinger, Donchian, Hurst, ecc.), multi‑timeframe come canali del tensore.
2. Addestramento
   - Modelli supervisati (ridge/lasso/elasticnet/rf) su feature/encoder con standardizzazione coerente.
   - Encoder: none, PCA (riduzione dimensionale), latents (se presenti).
   - GA/NSGA‑II:
     - Genetic‑basic: selezione, crossover, mutazione; fitness = R2.
     - NSGA‑II: valutazione (obj1=-R2, obj2=MAE), sorting non‑dominato e crowding distance; torneo binario e popolazione successiva, fronte di Pareto loggato a ogni generazione.
3. Inferenza e incertezza
   - Allineamento feature→ordine→standardizzazione a quelle del training.
   - Ricostruzione prezzi sui prossimi orizzonti; bande q05/q50/q95 e area tra q05‑q95.
   - Calibrazione conformale (ICP pesata) opzionale.
4. Regime e latenti (opzionale)
   - Clustering k‑means sugli z latenti e indice ANN (HNSW) per query kNN; score di regime come conditioning.

## Uso rapido

- Backfill: Chart → Backfill Missing (progress bar determinata).
- Training: Tab Training → configura indicatori×TF, modello/encoder, ottimizzazione → Start.
- Forecast:
  - Prediction Settings → inserisci modelli (uno per riga), seleziona tipi → Make Prediction.
  - Overlay multipli con legenda “modello:tipo”; trimming automatico oltre la soglia massima.

## Multi‑simbolo

L’interfaccia consente di selezionare la coppia di valute direttamente nella Tab “Chart” (combo in alto).
Le coppie supportate out‑of‑the‑box:
- AUX/USD
- GBP/NZD
- AUD/JPY
- GBP/EUR
- GBP/AUD

Tutte le funzioni (storici/backfill, addestramento, forecast) operano in base al simbolo selezionato. Il controller usa il symbol/timeframe correnti del grafico come default per le richieste di previsione.

## Note

- Tutte le operazioni di input/output sono in UTC.
- La progress bar del backfill è determinata sui subrange (split weekend).
- Le scelte UI vengono salvate tra sessioni (settings locali).

1. Dati e pre‑processing
   - Le candele sono in UTC.
   - La pipeline costruisce feature causali (r, ATR, RSI, MACD, Bollinger, Donchian, Hurst, ecc.).
   - Multi‑timeframe: la selezione indicatori×TF definisce il tensore finale (stack coerente per canali/scala).

2. Addestramento
   - Script base: weighted_forecast.py (metodi supervisati su features/encoder con standardizzazione fissata).
   - Encoder:
     - none: usa feature selezionate;
     - pca: proiezione a bassa dimensione per mitigare collinearità/rumore;
     - latents: vettori latenti persistiti in DB (se disponibili).
   - Modelli: ridge/lasso/elasticnet (regressione lineare penalizzata) e random forest.
   - Ricerca iperparametri:
     - baseline: parametri UI;
     - estendibile con ricerca evolutiva (strategia genetica) per scenari avanzati.
   - Metriche: R2/MAE in validazione; salvataggio artefatti in artifacts/models, nome includente parametri.

3. Inferenza e incertezza
   - Forecast locale: costruzione feature coerenti con il modello (ordine/standardizzazione), scoring e ricostruzione prezzi sui prossimi orizzonti.
   - Bande q05/q50/q95: overlay sul grafico e area tra q05‑q95.
   - Peso modello: nei settings è possibile attenuare o amplificare la previsione (0‑100%), utile per blending con baseline price o ensemble.
   - Calibrazione conformale (ICP pesata) opzionale per allineare la copertura empirica.

4. Regime e latenti (opzionale)
   - Persistenza dei latenti (VAE/diffusion) per clustering e ANN (HNSW), con query k‑NN e score di regime come conditioning addizionale.

## Uso pratico

- Backfill:
  - Chart → Backfill Missing: scarica solo gap mancanti per TF e aggiorna il grafico (barra di progresso indeterminata durante l’operazione).
- Training:
  - Tab Training → seleziona indicatori×TF, modello, encoder e parametri → Start Training.
  - Log in tempo reale e barra di progresso (indeterminata o approssimata da epoch).
  - Il modello è salvato in artifacts/models con nome contenente i parametri.
- Forecast:
  - Prediction Settings: selezione modello, orizzonti, N_samples, calibrazione, peso modello, indicatori/TF di inferenza.
  - Make Prediction / Advanced Forecast: overlay di nuove previsioni (non cancella le precedenti; trimming in base al limite massimo impostato).
  - Alt+Click (e Shift+Alt+Click) sul grafico: crea una previsione “TestingPoint” utilizzando le barre precedenti al punto cliccato.

## Note tecniche

- Zoom/pan non vengono resettati agli update del grafico.
- I weekend sono esclusi dal backfill, e le timeline di visualizzazione evitano periodi off‑trading (UTC).
- Tutta la logica è causale e le finestre di rolling evitano leakage.
- Le scelte “Indicatori×TF” vengono salvate in settings utente e ripristinate all’avvio.

Per dettagli implementativi, vedere i moduli UI (Chart/Training), controllers e la pipeline delle feature.
1) Crea e attiva l'ambiente virtuale:
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -e .
   pip install websocket-client

2) Provider realtime:
   - Default: tiingo (REST). Se provider espone WebSocket, RealTimeIngestionService userà lo streaming.
   - Puoi cambiare provider dalla UI (SignalsTab) o in configs/default.yaml -> providers.default

3) Avvia GUI:
   python .\scripts\run_gui.py

4) Test e Backfill:
   - Per testare la scrittura dei tick: `python .\tests\manual_tests\write_3_ticks.py`
   - Corretta la logica di ingestione realtime: ora salva i tick grezzi con il timeframe corretto ('tick') e l'aggregatore li processa correttamente per creare le candele.

GUI tabs:
 - Signals: recent signals + admin controls
 - History: historical candles table per symbol/timeframe (Refresh, Backfill)
 - Chart: matplotlib chart with pan/zoom (update via HistoryTab refresh or programmatically)

5) Avvia realtime helper (foreground):
   python .\scripts\start_realtime.py

Per operazioni avanzate vedere la cartella scripts/ e configs/.
