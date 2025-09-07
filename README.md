MagicForex — Forecast intraday FX con VAE + Diffusion (MVP)

MVP per forecasting FX intraday con VAE latente + diffusion, calibrazione conformale e GUI/ API.
- Backfill automatico (Alpha Vantage), DB SQLite e migrazioni Alembic.
- API FastAPI (/forecast) e GUI desktop (PySide6 + pyqtgraph).
- Moduli: data/io, features, models (VAE/diffusion), training, calibration, backtest.

Avvio rapido:
1) python3.12 -m venv .venv && source .venv/bin/activate
2) pip install -e .
3) impostare ALPHAVANTAGE_KEY nell'ambiente
4) alembic upgrade head
5) avviare API: uvicorn src.forex_diffusion.inference.service:app --host 0.0.0.0 --port 8000
6) avviare GUI: python -m src.forex_diffusion.ui.app

Test:
- pytest

Stato: MVP funzionale con fallback RW; integrazioni modello addestrato, calibrazione persistente e miglior sampler sono TODO.
