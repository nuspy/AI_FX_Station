MagicForex — Forecast intraday FX con VAE + Diffusion (MVP)

Tutto implementato: ingest, pipeline features (persistite), VAE+diffusion training/inference, DBWriter asincrono, bulk insert ottimizzato per Postgres (COPY), compaction scheduler e benchmark CI.
- Provider: Alpha Vantage (alpha_vantage + fallback HTTP).
- DB: SQLite per sviluppo; Postgres supportato con JSONB + GIN per features.
- Script utili: scripts/benchmark_bulk.py, scripts/do_all.sh, scripts/auto_tune.py

Uso GUI (SignalsTab)
1) Avvio: python -m src.forex_diffusion.ui.app (richiede PySide6 installato).
2) Main view: SignalsTab mostra signals recenti con Refresh; usa filtri/paginazione per dataset grandi.
3) Admin panel (SignalsTab):
   - Rebuild Regime Index: ricostruisce clustering + ANN (async).
   - Incremental Update: aggiunge nuovi latenti all'indice (batch).
   - Start/Stop Scheduler: avvia scheduler che esegue aggiornamenti incrementali periodici.
   - Show Index / Monitor Metrics: visualizza metriche ann (file size, elementi) e monitor runtime.
4) Operazioni runtime:
   - Backfill: avvia ripopolamento storico dal provider.
   - Benchmark / Autotune: esegue script locali per sizing batch e tuning.
   - Export Features / Show Predictions: esporta CSV e visualizza predizioni recenti.
   - Run Forecast: chiama API /forecast e mostra risultato.
5) Realtime & DBWriter:
   - Pulsanti per avviare/fermare il servizio realtime e il DBWriter (locale/global).
6) Logging & Feedback:
   - Area log integrata mostra output operazioni; popup per messaggi importanti.
7) Sicurezza:
   - Endpoint admin protetti da ADMIN_TOKENS (token:role); per GUI locale si può usare dialogo token (non abilitato per default).
   - Impostare ADMIN_TOKENS env var per permessi admin (es. token1:admin,token2:operator).

Ambiente e variabili
- DATABASE_URL (es. postgresql://fx:fxpass@127.0.0.1:5432/magicforex)
- ALPHAVANTAGE_KEY (API key per Alpha Vantage)
- ADMIN_TOKENS (comma-separated token:role)
- ARTIFACTS_DIR (cartella per index/checkpoint; default ./artifacts)

One-shot (locale, Postgres via Docker):
1) docker run -d --name pg -e POSTGRES_USER=fx -e POSTGRES_PASSWORD=fxpass -e POSTGRES_DB=magicforex -p 5432:5432 postgres:15
2) export DATABASE_URL=postgresql://fx:fxpass@127.0.0.1:5432/magicforex
3) export ADMIN_TOKENS=yourtoken:admin
4) python -m venv .venv && source .venv/bin/activate
5) pip install -e .
6) ./scripts/do_all.sh --db-url $DATABASE_URL --benchmark-rows 50000 --benchmark-batch 1000

Prometheus & monitoring
- Endpoint metrics: /metrics/prometheus
- Scheduler/incremental metrics persisted in DB (metrics table) e disponibili via admin endpoints.

CONTINUA_NECESSARIA: se vuoi posso aggiungere una mini-guida video o screenshot per i flussi GUI principali.

Note ottimizzazione:
- GIN/JSONB index migration applicata condizionalmente per Postgres.
- write_features_bulk usa COPY per Postgres (psycopg[binary]); fallback per altri DB.
- DBWriter chunked bulk flush e FeatureCompactor (retention) attivi; configurare bulk_batch_size e retention_days in configs/default.yaml.

CONTINUA_NECESSARIA:
- tuning dei parametri batch/interval in ambiente Postgres reale; posso aggiungere runner CI specifico e script di tuning auto.
