Sistema_previsionale — documento operativo (breve)

1) Scopo
- Descrive l'architettura del sistema di previsione, i metodi usati (nearest-neighbor su latenti, possibili regressori), i parametri rilevanti e i comandi PowerShell per testare e debuggare.

2) Architettura (sintesi)
- Fonte: market_data_candles (OHLCV, ts_utc).
- Pipeline features: src/forex_diffusion/features/pipeline.py -> features_df (indicatori multi‑timeframe, time-features).
- Tensori/patches: sliding windows (patch_len) -> tensori [batch, channels, patch_len].
- Encoder: VAE/encoder -> vettori latenti (z_dim) salvati in latents.latent_json.
- Clustering + ANN index: RegimeService.fit_clusters_and_index -> KMeans + hnswlib index (regime_index.bin) + mapping (regime_mapping.json).
- Query / previsione: RegimeService.query_regime(query_vec, k) -> neighbor ids + voting per regime.

3) Metodi di previsione disponibili
A) Nearest-neighbor sui latenti (naïve, storico)
 - Idea: prendi l'ultimo latent q, cerca k vicini nell'indice, leggi per ciascun vicino i ritorni successivi (horizon in barre) e aggrega (media/mediana/weighted by distance).
 - Parametri: k, HORIZON_BARS (es. 5,10,20), weighting by distance, choice di usare nearest by id o excluding recent.
 - Script di test: tmp/nn_forecast.py (fornisce esempio NN forecasting).

B) Regressore supervisionato (consigliato per produzione)
 - Idea: usare features/patterns per predire ritorno a orizzonte H o classificare direzione.
 - Candidate models: LightGBM/XGBoost, MLP, 1D-CNN / Transformer su patches.
 - Parametri: patch_len, channels, z_dim, feature set, standardizer parameters, train/val windows, target normalization.
 - Note: attivare rolling/backtesting e careful leakage control (warmup_bars).

4) Indicatori tecnici e time‑features inclusi
- Time: day_of_week, hour (no minuti), hour_sin/hour_cos, session dummies (Tokyo, London, New York).
- Indicatori (multi-timeframe): Log-return, High–Low Range, ATR, rolling volatility, Garman–Klass, EMA fast/slow, EMA slope, MACD, RSI (Wilder), Bollinger (upper/lower/width/%B), Keltner, Donchian, realized skew/kurtosis, Hurst (raw + aggvar + R/S windows), volume rolling mean.
- Prefisso colonna: "<tf>_<indicator>" (es. "1m_r", "15m_atr", "1d_bb_pctb").

5) Parametri principali (configurabili)
- features_config: warmup_bars, standardization.window_bars.
- vae: patch_len, channels, z_dim.
- hurst: window (default intraday = 64).
- clustering/index: n_clusters, index_space, ef_construction, M, ef (query).
- forecasting: K (neighbors), horizons (bars list), weighting, regressore hyperparams.

6) Comandi rapidi PowerShell (tutti caricano .env automaticamente)
- Eseguire check completo ML workflow:
  .\scripts\ml_workflow_check.ps1 -Symbol "EUR/USD" -Timeframe "1m" -DaysBackfill 3 -NClusters 8

- Test Hurst diagnostico (per un ts specifico):
  .\scripts\check_hurst_debug.ps1 -Symbol "EUR/USD" -Timeframe "1m" -TsUtc 1750376100000

- Esempio NN forecast (se tmp/nn_forecast.py è presente):
  python tmp\nn_forecast.py

7) Come interpretare i risultati dei check
- PASS per indicatori = il valore della pipeline (standardizzato) corrisponde alla versione ricomputata e standardizzata.
- MISMATCH su Hurst tipicamente indica:
  - differente window usata o metodo (aggvar vs R/S),
  - trasformazione/standardizzazione applicata nella pipeline,
  - data preprocessing (detrending) differente.
- Per Hurst il documento espone ora colonne raw: hurst, hurst_raw, hurst_aggvar_window, hurst_rs_window (controllare i valori grezzi in tmp/features_sample.csv).

8) Debug e next steps consigliati
- Se MISMATCH significativo per Hurst: confronta hurst_aggvar_window e hurst_rs_window; se entrambi concordano ma "hurst" della pipeline diverge, la pipeline applica una trasformazione: aggiungi hurst_raw e rivedi standardizer.
- Per migliorare previsione: costruire regressore supervisionato per ogni horizon e validare con backtesting walk-forward.

9) TODO rapidi
- [TODO] Aggiungere tmp/nn_forecast.py e wrapper PS se vuoi un tool integrato di prova (posso aggiungerlo).
- [TODO] Salvare parametri Standardizer (mu/sigma) su disco con modello per riproducibilità.

Fine.
