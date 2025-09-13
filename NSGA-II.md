# Manuale Utente – NSGA‑II (Multi‑Obiettivo)

Questo manuale spiega l’uso di NSGA‑II per ottimizzare i modelli in modo multi‑obiettivo, la semantica dei parametri, e il flusso completo: dal training al forecast e alla lettura dei segnali, con esempi passo‑passo.

Indice
- Cos’è NSGA‑II
- Obiettivi e metriche
- Parametri e operatori
- Flusso end‑to‑end
- Esempi passo‑passo
- Interpretare il fronte di Pareto
- Best practices
- FAQ

## Cos’è NSGA‑II

NSGA‑II è un algoritmo evolutivo multi‑obiettivo che evolve una popolazione di soluzioni verso un insieme di modelli “non‑dominati” (fronte di Pareto). Ogni individuo rappresenta una configurazione (modello + iperparametri + encoder). L’algoritmo:
- Valuta ogni individuo su più obiettivi (es. massimizzare R2 e minimizzare MAE).
- Ordina la popolazione in fronti non‑dominati (fast non‑dominated sorting).
- Assegna una crowding distance per preservare diversità sul fronte.
- Applica torneo binario basato su rank e crowding per selezione.
- Applica crossover e mutazione per generare i figli.
- Combina popolazione attuale + figli e seleziona la prossima generazione mantenendo il fronte migliore con massima diversità.

## Obiettivi e metriche

Per questa implementazione:
- obj1 = −R2 (da minimizzare) → equivale a massimizzare R2.
- obj2 = MAE (da minimizzare).

R2 e MAE sono loggati durante la fase di training; NSGA‑II estrae dai log i valori per comporre la fitness multi‑obiettivo.

## Parametri e operatori

- Generations (Gen): numero di generazioni evolutive (tipico: 5–20).
- Population (Pop): dimensione popolazione per generazione (tipico: 8–32).
- Selezione: torneo binario con crowded‑comparison (confronto su rank del fronte; pareggio → crowding distance).
- Crossover: ricombinazione di iperparametri (mix da due genitori).
- Mutazione: piccole perturbazioni su iperparametri:
  - ridge/lasso/elasticnet: alpha, l1_ratio (solo elasticnet).
  - rf: n_estimators, max_depth.
  - encoder pca: encoder_dim.

Spazio di ricerca (indicativo e modificabile):
- model ∈ {ridge, lasso, elasticnet, rf}
- alpha ∈ [1e−3, 10] (log‑uniforme).
- l1_ratio ∈ [0.1, 0.9] (solo elasticnet).
- n_estimators ∈ [100, 500].
- max_depth ∈ {None, 8, 12, 16, 20}.
- encoder ∈ {none, pca}, encoder_dim ∈ {32, 64, 96, 128}.

## Flusso end‑to‑end

1) Preparazione
- Assicurati di avere storico sufficiente per il simbolo/TF (usa “Backfill Missing” su Chart).
- Apri la Tab “Training”.

2) Configurazione NSGA‑II
- Symbol/TF: scegli coppia FX e timeframe base.
- Days/Horizon: imposta finestra temporale e orizzonte target.
- Indicatori×TF: seleziona i timeframe per ciascun indicatore.
- Pipeline: warmup/atr_n/rsi_n/bb_n/hurst_window/rv_window.
- Optimization: seleziona “nsga2”.
- Gen/Pop: imposta generazioni/popolazione.

3) Avvio
- Clicca “Start Training”: partirà l’ottimizzazione. La progress bar è determinata (generazioni × popolazione).
- Durante l’esecuzione vengono mostrati i log, inclusi R2 e MAE degli individui e il “best front” della generazione.

4) Output finale
- Al termine, la popolazione finale contiene il fronte di Pareto. È possibile scegliere manualmente il compromesso preferito (es. R2 elevato con MAE moderato) oppure ri‑eseguire training mirato con la configurazione desiderata.

## Esempi passo‑passo

A) Ottimizzazione NSGA‑II su GBP/EUR 1m
1. Chart → seleziona GBP/EUR, backfill se necessario.
2. Training:
   - Symbol=GBP/EUR, Base TF=1m, Days=10, Horizon=5.
   - Indicatori×TF: lascia Default (modifica solo se sai cosa fare).
   - Pipeline: warmup=64, atr_n=14, rsi_n=14, bb_n=20, hurst_window=64, rv_window=60.
   - Optimization=nsga2, Gen=8, Pop=12.
3. Start Training. Attendi che compaiano i log del best front (pareto).
4. Leggi il best front nel log, per es.:
   - (R2=0.3150, MAE=1.2e−03), (R2=0.2900, MAE=1.0e−03), (R2=0.3400, MAE=1.6e−03)
5. Scegli la configurazione coerente con il tuo obiettivo (es. preferisci minor MAE) e salva il relativo modello (se non già salvato).
6. Vai su Prediction Settings per il forecast.

B) Dalla configurazione NSGA‑II al Forecast multi‑modello
1. Prediction Settings:
   - Inserisci più modelli (uno per riga): ad esempio quelli scelti dal best front (varie trade‑off).
   - Seleziona tipi di previsione: Basic, Advanced, Baseline RW.
   - Imposta Model weight (%) per controllare l’impatto della previsione sul prezzo (blending).
2. Make Prediction: vedrai overlay multipli sul grafico, uno per ogni (modello × tipo).
3. Confronta le traiettorie ed eventualmente limita il numero massimo di overlay.

C) Dalla previsione ai segnali (linee guida)
- Sogliare il ritorno atteso a orizzonte d’interesse (es. 5/10/20 barre) oppure usare bande q05/q95:
  - Segnale long se q50 cresce e q05>ultimo prezzo; short viceversa.
  - Integrare con regime/volatilità per filtrare rumore.
- Con multi‑modello: usare voting o media pesata (coerente con Model weight).

## Interpretare il fronte di Pareto

- Ogni punto del fronte è “non‑dominato”: non esiste altra soluzione che abbia entrambi (−R2) e MAE migliori.
- Scegli il punto che massimizza la tua utilità:
  - massimizza R2 a fronte di MAE accettabile;
  - minimizza MAE a fronte di R2 accettabile;
  - oppure scorri il fronte alla ricerca del “gomito” (knee).

## Best practices

- Generazioni e popolazione: aumentano robustezza ma anche tempo. Inizia con 5×8 e scala.
- Spazio iperparametri: restringi se il tempo è limitato; lascia più libertà se hai risorse.
- Indicatori×TF: parti da Default; aggiungi canali solo con una giustificazione (evita overfitting).
- Encoder: prova PCA quando hai molte features; latents richiede pipeline di salvataggio attiva.
- Benchmark: confronta sempre con Baseline RW per quantificare il guadagno del modello.

## FAQ

D: Non vedo miglioramenti netti generazione su generazione.
- Verifica la variabilità introdotta da mutazione/crossover e l’ampiezza dello spazio di ricerca.
- Aumenta Pop o Gen, o rivedi gli obiettivi (es. normalizzazione delle metriche).

D: Il fronte di Pareto è troppo fitto e poco differenziato.
- Aumenta crowding (pop più grande) o amplia la diversità (più ampia mutazione).

D: Modelli NSGA‑II peggio del baseline.
- Controlla i dati (backfill completo), warmup sufficiente, e verifica indicatori/TF.
- Eventualmente semplifica la configurazione e allarga gradualmente.
