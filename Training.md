# Manuale Utente – Training

Questo manuale spiega in modo dettagliato come configurare ed eseguire l’addestramento dei modelli, come scegliere encoder e indicatori tecnici multi‑timeframe, come usare l’ottimizzazione genetica (GA) e come sfruttare previsioni multi‑modello. Include procedure passo‑passo, consigli pratici e una sezione di troubleshooting.

Indice
- Concetti chiave
- Panoramica UI del Training
- Modelli ed Encoder
- Indicatori × Timeframe e Pipeline
- Ottimizzazione (None, GA)
- Multi‑modello e Multi‑simbolo
- Procedure passo‑passo (How‑To)
- Best practices e checklist
- Troubleshooting

## Concetti chiave

- Feature causali: tutte le features sono calcolate in modo causale (no leakage).
- Multi‑timeframe: puoi selezionare per ciascun indicatore i timeframe (es. 1m, 5m, …) per costruire un tensore multiscala coerente.
- Standardizzazione: viene salvata (mu/sigma) insieme al modello per garantire inferenza coerente.
- Encoder: riduzione dimensionale (PCA) o latenti (se salvati in DB).
- Ottimizzazione: disponibile ricerca genetica single‑objective (GA); NSGA‑II è documentato separatamente (NSGA-II.md).

## Panoramica UI del Training

Apri la Tab “Training”. La UI è composta da:

1) Controlli principali
- Symbol: coppia FX su cui addestrare (selezione tra: EUR/USD, GBP/USD, AUX/USD, GBP/NZD, AUD/JPY, GBP/EUR, GBP/AUD).
- Base TF: timeframe principale (es. 1m).
- Days history: numero di giorni di storico da utilizzare.
- Horizon (bars): orizzonte target in barre (per supervised learning).
- Model: ridge, lasso, elasticnet, rf.
- Encoder: none, pca, latents.
- Optimization: none, genetic‑basic (per NSGA‑II vedi NSGA-II.md).
- Gen / Pop: generazioni e popolazione per GA.

2) Griglia Indicatori × Timeframe
- Una matrice di checkbox: per ogni indicatore (ATR, RSI, Bollinger, MACD, Donchian, Keltner, Hurst) scegli i TF in cui includerlo.
- Pulsante Default per riga: ripristina la selezione consigliata per quell’indicatore.
- La selezione viene salvata tra sessioni.

3) Parametri Pipeline
- warmup: barre iniziali da scartare per stabilizzare indicatori.
- atr_n, rsi_n, bb_n: parametri classici degli indicatori.
- hurst_window, rv_window: dimensioni finestra per stime di Hurst/volatilità.

4) Output e Avvio
- Output dir: cartella destinazione artefatti/modelli.
- Start Training: avvio asincrono con log in tempo reale e progress bar determinata (per GA: gens × pop).

## Modelli ed Encoder

Modelli (supervised):
- ridge (L2), lasso (L1), elasticnet (L1+L2): adatti a features numeriche; ottimi come baseline e interpretabilità (coefficiente → importanza).
- rf (Random Forest): non lineare, robusto al rumore; attenzione a overfitting se iperparametri non regolati.

Encoder:
- none: usa direttamente le features scelte.
- pca: riduzione dimensionale; param: encoder_dim (es. 64). Riduce collinearità/rumore.
- latents: usa vettori latenti salvati in DB (se disponibili). Richiede pipeline di salvataggio latenti.

Consigli:
- Inizia con ridge/elasticnet e encoder=none o pca.
- Se molte features, PCA può migliorare stabilità.
- RF utile per catturare non linearità; controlla n_estimators e max_depth.

## Indicatori × Timeframe e Pipeline

Indicatori disponibili:
- ATR, RSI, Bollinger, MACD, Donchian, Keltner, Hurst.

Selezione TF:
- Spunta i timeframe per indicatore coerenti con il setup (es. intraday: 1m/5m/15m per ATR/RSI).
- Usa Default per un punto di partenza equilibrato.

Parametri pipeline:
- warmup (es. 16, 64, 128): evita fasi transitorie di indicatori.
- atr_n: default 14.
- rsi_n: default 14.
- bb_n: default 20.
- hurst_window: default 64 (intraday).
- rv_window: default 60 (rolling std returns).

## Ottimizzazione (None, GA)

- none: esegue un singolo training con i parametri scelti.
- genetic‑basic (GA single‑objective): ottimizza iperparametri per massimizzare R2.
  - Parametri GA: Gen (generazioni), Pop (dimensione popolazione).
  - Operatori: torneo, crossover, mutazione su alpha, l1_ratio, n_estimators, max_depth, encoder_dim (se PCA).
  - Progress: determinato su gens × pop; log dettaglia fitness e best config.

Per ottimizzazione multi‑obiettivo (NSGA‑II: min[-R2, MAE]), vedi NSGA-II.md.

## Multi‑modello e Multi‑simbolo

Multi‑simbolo:
- Scegli la coppia FX nella Tab “Chart” (combo in alto). Backfill, training e forecast usano il simbolo correntemente selezionato.

Multi‑modello (in Forecast):
- Nella finestra “Prediction Settings”, sezione “Modelli multipli”, inserisci percorsi modello (uno per riga). La previsione verrà calcolata per ciascun modello e ogni combinazione di “tipo” (basic, advanced, baseline RW).
- Gli overlay sul grafico sono colorati in modo deterministico e rimangono fino al trimming (impostazione numero massimo).

## Procedure passo‑passo (How‑To)

A) Preparare i dati e backfill
1. Vai su Tab “Chart” → seleziona Symbol (es. GBP/NZD).
2. Clicca “Backfill Missing” per scaricare i gap mancanti (progress bar determinata). Fine → il grafico si aggiorna.

B) Configurare un training semplice (ridge + none)
1. Apri Tab “Training”.
2. Imposta: Symbol (es. GBP/NZD), Base TF (1m), Days history (es. 7), Horizon (es. 5).
3. Model = ridge; Encoder = none; Optimization = none.
4. Nella griglia Indicatori×TF: lascia Default o affina le selezioni.
5. Parametri pipeline: warmup=16, atr_n=14, rsi_n=14, bb_n=20, hurst_window=64, rv_window=60.
6. Output dir: conferma.
7. Start Training: osserva log e attendi completamento. Il modello viene salvato negli artefatti.

C) Training con PCA + elasticnet
1. Come sopra, ma Encoder=pca (encoder_dim ~ 64) e Model=elasticnet (l1_ratio=0.5).
2. Start Training.

D) GA per cercare iperparametri (genetic‑basic)
1. Optimization=genetic‑basic; Gen=5; Pop=8 (adatta a risorse/tempo).
2. Start Training: la GA esegue valutazioni; guarda lo stream di R2 e best config.
3. A fine GA, viene rieseguito il training con la best config.

E) Forecast multi‑modello e multi‑tipo
1. Prediction Settings:
   - Inserisci Modelli multipli (ogni riga un file).
   - Seleziona tipi: Basic, Advanced, Baseline RW.
   - Imposta Model weight (%), indicatori/TF di inferenza se necessario.
2. Make Prediction: il grafico mostra overlay per ogni (modello × tipo). La legenda include “nomeModello:tipo”.

F) TestingPoint sul grafico
1. Su Chart: Alt+Click → previsione Basic a partire dal punto; Shift+Alt+Click → Advanced.
2. Il numero di barre storiche usate è nei settings (Test history bars).

## Best practices e checklist

- Dati: assicurati che backfill sia aggiornato per il simbolo/TF.
- Selezione TF: non esagerare con troppi canali; usa Default come base e aggiungi gradualmente.
- Standardizzazione: mantieni coerenti feature, mu/sigma e ordine tra training e inference (la pipeline lo gestisce).
- GA: inizia con Pop piccola (8–16) e poche generazioni (5–10). Espandi se serve.
- Multi‑modello: confronta sempre con Baseline RW per valutare il valore aggiunto del modello.

## Troubleshooting

- “Poche features/NaN”: aumenta warmup, riduci indicatori o controlla che i TF selezionati abbiano dati a sufficienza.
- “Modello non caricato in forecast”: verifica percorsi nel dialog (model_paths) e i permessi file.
- “Zero overlay o overlay confusi”: aumenta il limite max overlay o usa meno modelli/tipi contemporaneamente.
- “GA lenta”: riduci popolazione/gen o restringi lo spazio iperparametri (ad es. range alpha).
