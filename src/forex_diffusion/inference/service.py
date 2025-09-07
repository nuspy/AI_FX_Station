"""
FastAPI inference service for MagicForex.

- POST /forecast accepts {symbol, timeframe, horizons[], N_samples, apply_conformal}
- ModelService (MVP) tries to load last close and volatility from DB and samples N trajectories per horizon
- Returns quantiles {q05,q50,q95} per horizon, conformal bands (if applied), credibility (placeholder), diagnostics (placeholders)
- On startup attempts alembic upgrade head (best-effort)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger
from sqlalchemy import create_engine, select, Table, MetaData

from ..utils.config import get_config
from ..postproc import uncertainty as unc
from ..data import io as data_io

from contextlib import asynccontextmanager

cfg = get_config()

# Define lifespan manager to ensure alembic and DBWriter start/stop correctly per worker process
@asynccontextmanager
async def lifespan(app):
    # run alembic upgrade head (best-effort)
    try:
        if getattr(cfg.app, "alembic_upgrade_on_start", True) or (isinstance(cfg, dict) and cfg.get("app", {}).get("alembic_upgrade_on_start", True)):
            # run synchronously; acceptable for startup preparation
            _run_alembic_upgrade_head()
    except Exception as e:
        logger.warning("Alembic upgrade attempt failed in lifespan: {}", e)

    # start DBWriter if present
    try:
        if "db_writer" in globals() and db_writer is not None:
            try:
                db_writer.start()
                logger.info("Background DBWriter started (lifespan)")
            except Exception as e:
                logger.exception("Failed to start DBWriter in lifespan: {}", e)
    except Exception:
        pass

    # start real-time ingest service to receive tick data immediately
    try:
        from ..services.realtime import RealTimeIngestService
        # symbols and timeframe from config
        symbols = getattr(cfg.data, "symbols", None) or (cfg.data.get("symbols") if isinstance(cfg.data, dict) else [])
        tf_default = None
        try:
            tf_default = cfg.timeframes.native[0] if hasattr(cfg.timeframes, "native") else (cfg.timeframes.get("native", [])[0] if isinstance(cfg.timeframes, dict) else "1m")
        except Exception:
            tf_default = "1m"
        rt = RealTimeIngestService(engine=_engine, market_service=None, symbols=symbols, timeframe=tf_default, poll_interval=cfg.get("providers", {}).get("poll_interval", 2.0) if isinstance(cfg, dict) else 2.0, db_writer=db_writer)
        # expose globally so other components may reference or stop it
        globals()["rt_service"] = rt
        rt.start()
        logger.info("RealTimeIngestService started in lifespan for symbols: {}", symbols)
    except Exception as e:
        logger.exception("Failed to start RealTimeIngestService in lifespan: {}", e)

    try:
        yield
    finally:
        # stop DBWriter gracefully
        try:
            if "db_writer" in globals() and db_writer is not None:
                db_writer.stop(flush=True, timeout=5.0)
                logger.info("Background DBWriter stopped (lifespan)")
        except Exception as e:
            logger.exception("Failed to stop DBWriter in lifespan: {}", e)


# Create FastAPI app with lifespan manager so uvicorn/ASGI server triggers start/stop per worker
app = FastAPI(title="MagicForex Inference Service", lifespan=lifespan)


class ForecastRequest(BaseModel):
    symbol: str
    timeframe: str
    horizons: List[str] = Field(..., description="List of horizon labels, e.g. ['1m','5m','15m']")
    N_samples: int = Field(200, ge=1, le=2000)
    apply_conformal: bool = Field(True)


class HorizonQuantiles(BaseModel):
    q05: float
    q50: float
    q95: float


class ForecastResponse(BaseModel):
    quantiles: Dict[str, HorizonQuantiles]
    bands_conformal: Optional[Dict[str, Dict[str, float]]] = None
    credibility: Dict[str, float]
    diagnostics: Dict[str, Optional[float]]


def _run_alembic_upgrade_head():
    """
    Best-effort: run 'alembic upgrade head' in project root.
    Non-fatal: logs errors but does not block startup.
    """
    try:
        logger.info("Running alembic upgrade head (best-effort)...")
        # Try subprocess call; assumes alembic.ini configured
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        logger.info("Alembic upgrade head completed.")
    except Exception as e:
        logger.warning("Alembic upgrade head failed or not available: {}", e)


def _get_engine() -> Any:
    db_url = getattr(cfg.db, "database_url", None) or (cfg.db.get("database_url") if isinstance(cfg.db, dict) else None)
    if not db_url:
        raise RuntimeError("Database URL not configured")
    engine = create_engine(db_url, future=True)
    return engine


@dataclass
class ModelService:
    """
    Minimal ModelService for MVP forecasting.

    Behavior:
      - If model artifact exists (placeholder), use it (not implemented).
      - Otherwise use RW log-normal sampling from last_close with sigma estimated from recent returns.
    """
    engine: Any

    def _get_last_close_and_recent(self, symbol: str, timeframe: str, n: int = 500) -> Dict[str, Any]:
        """
        Query DB for last close and recent closes (up to n bars).
        Returns dict with last_close (float) and recent_closes (np.array)
        """
        # Ensure table exists
        metadata = MetaData()
        metadata.reflect(bind=self.engine, only=["market_data_candles"])
        if "market_data_candles" not in metadata.tables:
            raise HTTPException(status_code=500, detail="market_data_candles table not found in DB")
        tbl = metadata.tables["market_data_candles"]
        with self.engine.connect() as conn:
            stmt = (
                select(tbl.c.ts_utc, tbl.c.close)
                .where(tbl.c.symbol == symbol)
                .where(tbl.c.timeframe == timeframe)
                .order_by(tbl.c.ts_utc.desc())
                .limit(n)
            )
            res = conn.execute(stmt).fetchall()
            if not res:
                raise HTTPException(status_code=404, detail=f"No historical data for {symbol} {timeframe}")
            rows = list(res)
            closes = np.array([float(r[1]) for r in rows[::-1]])  # ascending order
            last_close = float(rows[0][1])
            timestamps = np.array([int(r[0]) for r in rows[::-1]])
        return {"last_close": last_close, "recent_closes": closes, "recent_ts": timestamps}

    def _estimate_sigma(self, recent_closes: np.ndarray, window: int = 100) -> float:
        """
        Estimate 1-bar volatility (std of log returns) using recent_closes.
        """
        if len(recent_closes) < 2:
            return 1e-6
        r = np.log(recent_closes).diff() if isinstance(recent_closes, pd.Series) else np.diff(np.log(recent_closes))
        r = np.asarray(r)
        if r.size < 1:
            return 1e-6
        # rolling std over last 'window' bars (use numpy)
        if r.size > window:
            r = r[-window:]
        sigma = float(np.nanstd(r, ddof=1))
        if sigma <= 0:
            sigma = 1e-6
        return sigma

    def forecast(self, symbol: str, timeframe: str, horizons: List[str], N_samples: int = 200, apply_conformal: bool = True) -> Dict[str, Any]:
        """
        Generate forecast quantiles for requested horizons.
        Horizons are labels; map labels to minutes via config horizons list.
        Returns a dict with quantiles per horizon and additional diagnostics.
        """
        # Resolve horizons mapping from cfg
        cfg_horizons = getattr(cfg, "horizons", None)
        label2min = {}
        if isinstance(cfg_horizons, dict):
            # handle 'list' section
            for h in cfg_horizons.get("list", []):
                label2min[str(h.get("label"))] = int(h.get("minutes"))
        else:
            # fallback: try to parse simple list-like structure
            try:
                for item in cfg.horizons.list:
                    label2min[item["label"]] = int(item["minutes"])
            except Exception:
                label2min = {}

        # Get last close and recent closes
        d = self._get_last_close_and_recent(symbol, timeframe, n=1024)
        last_close = d["last_close"]
        recent_closes = d["recent_closes"]
        recent_ts = d["recent_ts"]

        # Build conditioning vector h from recent candles using pipeline.build_conditioning
        try:
            # fetch last n rows of candles to construct df for conditioning
            from ..data import io as data_io_local
            from ..features.pipeline import build_conditioning
            # query recent candles as DataFrame via engine
            with self.engine.connect() as conn:
                tbl = MetaData()
                tbl.reflect(bind=self.engine, only=["market_data_candles"])
                mdt = tbl.tables.get("market_data_candles")
                if mdt is not None:
                    stmt = select(mdt).where(mdt.c.symbol == symbol).where(mdt.c.timeframe == timeframe).order_by(mdt.c.ts_utc.desc()).limit(1024)
                    rows = conn.execute(stmt).fetchall()
                    if rows:
                        df_recent = pd.DataFrame(rows, columns=rows[0]._mapping.keys()).sort_values("ts_utc").reset_index(drop=True)
                        cond_df = build_conditioning(df_recent)
                        # take last row as conditioning vector
                        cond_vec = cond_df.iloc[-1].to_numpy(dtype=float)
                    else:
                        cond_vec = None
                else:
                    cond_vec = None
        except Exception:
            cond_vec = None

        sigma_1 = self._estimate_sigma(recent_closes, window=cfg.features.get("standardization", {}).get("window_bars", 1000) if isinstance(cfg.features, dict) else 100)
        # Generate samples per horizon using log-normal RW: price_h = last_close * exp(sigma_1 * sqrt(h_minutes) * Z)
        quantiles_out = {}
        samples_dict = {}
        for h_label in horizons:
            # map label to minutes
            h_min = label2min.get(h_label)
            if h_min is None:
                # try simple parse like '1m','5m','2h','1d'
                if h_label.endswith("m"):
                    h_min = int(h_label[:-1])
                elif h_label.endswith("h"):
                    h_min = int(h_label[:-1]) * 60
                elif h_label.endswith("d"):
                    h_min = int(h_label[:-1]) * 1440
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown horizon label: {h_label}")
            # compute horizon factor: sqrt(h in minutes) relative to 1-minute sigma
            factor = np.sqrt(max(1.0, float(h_min)))
            # sample returns
            Z = np.random.randn(N_samples)
            # use mu=0 drift
            sim_prices = last_close * np.exp(sigma_1 * factor * Z)
            samples_dict[h_label] = sim_prices  # shape (N,)
            qs = np.quantile(sim_prices, [0.05, 0.5, 0.95])
            quantiles_out[h_label] = {"q05": float(qs[0]), "q50": float(qs[1]), "q95": float(qs[2])}

        # Conformal calibration (MVP): no historical predicted quantiles stored -> delta=0; implement placeholder
        bands = {}
        credibility = {}
        for h_label in horizons:
            q = quantiles_out[h_label]
            # apply delta = 0 by default (no historical calibration applied)
            delta = 0.0
            bands[h_label] = {"low": float(q["q05"] - delta), "high": float(q["q95"] + delta)}
            # credibility: use simplified heuristic from uncertainty. Here compute width normalized by sigma
            width = q["q95"] - q["q05"]
            w_norm = width / (sigma_1 * 1.0 + 1e-9)
            # crude credibility: sigmoid of inverse width and coverage heuristic
            cred = 1.0 / (1.0 + float(w_norm))
            credibility[h_label] = float(max(0.0, min(1.0, cred)))

        # Diagnostics: CRPS not computable without true y; compute CRPS vs RW baseline by comparing sample distributions
        diagnostics = {"CRPS_model": None, "CRPS_RW": None, "PIT_pvalue": None}
        # but compute sample-based variance etc for debugging
        diagnostics["model_sigma_estimate"] = float(sigma_1)

        # return structured response
        return {"quantiles": quantiles_out, "bands_conformal": bands, "credibility": credibility, "diagnostics": diagnostics}


# initialize services with a shared engine
_engine = _get_engine()
# local imports to avoid top-level circulars in some environments
from ..services.model_service import ModelService
from ..services.db_service import DBService
from ..services.calibration import CalibrationService
from ..services.db_writer import DBWriter
from ..services.db_writer import DBWriter
from ..services.db_writer import DBWriter

db_service = DBService(engine=_engine)
calib_service = CalibrationService(engine=_engine)
model_service = ModelService(engine=_engine)

# instantiate DBWriter (background writer) - lifecycle managed by the FastAPI lifespan
db_writer = DBWriter(db_service=db_service)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """
    Forecast endpoint that:
      - invokes ModelService.forecast
      - optionally applies last calibration (if apply_conformal and a calibration record exists)
      - persists predictions in DBService
      - returns structured ForecastResponse
    """
    try:
        res = model_service.forecast(req.symbol, req.timeframe, req.horizons, N_samples=req.N_samples, apply_conformal=req.apply_conformal)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Forecast error: {}", e)
        raise HTTPException(status_code=500, detail=str(e))

    quantiles_raw = res.get("quantiles", {})
    bands_raw = res.get("bands_conformal", {})
    credibility = res.get("credibility", {})
    diagnostics = res.get("diagnostics", {})

    # If requested, try to apply latest calibration adjustments (if present)
    adjusted_bands = {}
    adjusted_quantiles = {}
    try:
        if req.apply_conformal:
            last_cal = calib_service.get_last_calibration(req.symbol, req.timeframe)
            if last_cal:
                delta = last_cal.delta_global
                # apply delta to each horizon quantiles
                for h, q in quantiles_raw.items():
                    q05 = float(q["q05"])
                    q50 = float(q["q50"])
                    q95 = float(q["q95"])
                    adj = unc.apply_conformal_adjustment(q05=q05, q50=q50, q95=q95, delta=delta, asymmetric=True)
                    adjusted_quantiles[h] = {"q05": float(adj["q05_adj"]), "q50": float(adj["q50"]), "q95": float(adj["q95_adj"])}
                    adjusted_bands[h] = {"low": float(adj["q05_adj"]), "high": float(adj["q95_adj"])}
                diagnostics["calibration_applied"] = True
                diagnostics["calibration_delta"] = float(delta)
            else:
                # no calibration available; use raw quantiles/bands
                adjusted_quantiles = {h: {"q05": float(q["q05"]), "q50": float(q["q50"]), "q95": float(q["q95"])} for h, q in quantiles_raw.items()}
                adjusted_bands = bands_raw
                diagnostics["calibration_applied"] = False
        else:
            adjusted_quantiles = {h: {"q05": float(q["q05"]), "q50": float(q["q50"]), "q95": float(q["q95"])} for h, q in quantiles_raw.items()}
            adjusted_bands = bands_raw
            diagnostics["calibration_applied"] = False
    except Exception as e:
        logger.exception("Failed to apply calibration: {}", e)
        adjusted_quantiles = {h: {"q05": float(q["q05"]), "q50": float(q["q50"]), "q95": float(q["q95"])} for h, q in quantiles_raw.items()}
        adjusted_bands = bands_raw
        diagnostics["calibration_error"] = str(e)

    # Persist predictions asynchronously via DBWriter (one task per horizon)
    try:
        for h_label, qdict in adjusted_quantiles.items():
            enqueued = db_writer.write_prediction_async(
                symbol=req.symbol,
                timeframe=req.timeframe,
                horizon=h_label,
                q05=float(qdict["q05"]),
                q50=float(qdict["q50"]),
                q95=float(qdict["q95"]),
                meta={"diagnostics": diagnostics, "credibility": credibility.get(h_label)},
            )
            if not enqueued:
                logger.warning("DBWriter queue full: failed to enqueue prediction for {} {} {}", req.symbol, req.timeframe, h_label)
        diagnostics["persisted_predictions"] = True
    except Exception as e:
        logger.exception("Failed to enqueue predictions: {}", e)
        diagnostics["persisted_predictions"] = False
        diagnostics["persist_error"] = str(e)

    # build response model
    quantiles_resp = {h: HorizonQuantiles(**adjusted_quantiles[h]) for h in adjusted_quantiles.keys()}
    bands_resp = adjusted_bands if adjusted_bands else None

    return ForecastResponse(quantiles=quantiles_resp, bands_conformal=bands_resp, credibility=credibility, diagnostics=diagnostics)


@app.get("/health")
def health():
    """
    Simple liveness probe. Returns status ok and basic service flags.
    """
    try:
        model_loaded = bool(model_service.is_model_loaded())
    except Exception:
        model_loaded = False
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/ready")
def ready():
    """
    Readiness probe: verifies DB connectivity and that migrations/tables exist.
    """
    try:
        # quick DB check: count predictions
        with db_service.engine.connect() as conn:
            res = conn.execute("SELECT 1").fetchone()
            ready_db = res is not None
    except Exception as e:
        logger.warning("Readiness DB check failed: {}", e)
        return {"ready": False, "reason": "db_unavailable"}
    # optionally ensure model loaded or fallback allowed
    model_ok = bool(model_service.is_model_loaded())
    return {"ready": True, "db": True, "model_loaded": model_ok}


@app.get("/metrics")
def metrics():
    """
    Lightweight metrics endpoint returning counts of persisted entities.
    Suitable for scraping by simple monitors.
    """
    try:
        with db_service.engine.connect() as conn:
            pred_count = conn.execute("SELECT COUNT(1) FROM predictions").scalar() or 0
            cal_count = conn.execute("SELECT COUNT(1) FROM calibration_records").scalar() or 0
            sig_count = conn.execute("SELECT COUNT(1) FROM signals").scalar() or 0
    except Exception as e:
        logger.warning("Metrics query failed: {}", e)
        return {"error": "db_unavailable"}
    return {"predictions": int(pred_count), "calibrations": int(cal_count), "signals": int(sig_count)}


def main():
    import uvicorn
    import os

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))
    uvicorn.run("src.forex_diffusion.inference.service:app", host=host, port=port, workers=workers, reload=False)


if __name__ == "__main__":
    main()
