MagicForex — Forecast intraday FX con VAE + Diffusion (MVP)

Tutto implementato: ingest, pipeline features (persistite), VAE+diffusion training/inference, DBWriter asincrono, bulk insert ottimizzato per Postgres (COPY), compaction scheduler e benchmark CI.
- Provider: Alpha Vantage (bridge ufficiale alpha_vantage + fallback HTTP).
- DB: SQLite per sviluppo; Postgres supportato con JSONB + GIN per features.
- Script utili: scripts/benchmark_bulk.py, scripts/do_all.sh (one-shot).

One-shot (locale, Postgres via Docker):
1) docker run -d --name pg -e POSTGRES_USER=fx -e POSTGRES_PASSWORD=fxpass -e POSTGRES_DB=magicforex -p 5432:5432 postgres:15
2) export DATABASE_URL=postgresql://fx:fxpass@127.0.0.1:5432/magicforex
3) export ADMIN_TOKEN=YOUR_SECRET_TOKEN
4) python -m venv .venv && source .venv/bin/activate
5) pip install -e .
6) ./scripts/do_all.sh --db-url $DATABASE_URL --benchmark-rows 50000 --benchmark-batch 1000

Admin endpoints (protected by X-ADMIN-TOKEN header or ADMIN_TOKENS env list):
- POST /admin/regime/rebuild?n_clusters=8
- POST /admin/regime/incremental?batch_size=1000
- POST /admin/regime/incremental (automated by scheduler)
- GET  /admin/regime/status
- GET  /admin/regime/metrics
Prometheus metrics exposed at: /metrics/prometheus (scrape target)
Set ADMIN_TOKENS (comma-separated) env var to control admin access.

Note ottimizzazione:
- GIN/JSONB index migration applicata condizionalmente per Postgres.
- write_features_bulk usa COPY per Postgres (psycopg[binary]); fallback per altri DB.
- DBWriter chunked bulk flush e FeatureCompactor (retention) attivi; configurare bulk_batch_size e retention_days in configs/default.yaml.

CONTINUA_NECESSARIA:
- tuning dei parametri batch/interval in ambiente Postgres reale; posso aggiungere runner CI specifico e script di tuning auto.
