from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from ..backtest.schemas import BacktestConfigPayload
from ..backtest.horizons import parse_horizons
from ..backtest.worker import Worker, TrialConfig
from ..backtest.db import BacktestDB
from ..backtest.queue import BacktestQueue
from ..ui.prediction_settings_dialog import PredictionSettingsDialog


router = APIRouter(prefix="/backtests", tags=["backtests"])


class CreateBacktestRequest(BacktestConfigPayload):
    pass


class CreateBacktestResponse(BaseModel):
    job_id: int
    n_configs: int


@router.post("")
def create_backtest(req: CreateBacktestRequest) -> CreateBacktestResponse:
    try:
        # Expand horizons
        _, horizons_sec = parse_horizons(req.horizons_raw)
        if not horizons_sec:
            raise HTTPException(status_code=400, detail="No valid horizons parsed")
        # Build configs (cartesian of models × prediction_types)
        configs: List[TrialConfig] = []
        for model_name in (req.models or ["baseline_rw"]):
            for ptype in req.prediction_types:
                tc = TrialConfig(
                    model_name=model_name,
                    prediction_type=ptype,
                    timeframe=req.timeframe,
                    horizons_sec=horizons_sec,
                    samples_range=tuple(req.samples_range),
                    indicators=req.indicators.model_dump(),
                    interval=req.interval.model_dump(),
                    data_version=req.data_version,
                    symbol=req.symbol,
                )
                configs.append(tc)
        # Create job and run (with caching/dedup: reuse existing result by fingerprint)
        btdb = BacktestDB()
        job_id = int(req.job_id) if req.job_id is not None else btdb.create_job(status="running")
        # Check dedup: if all configs already have results, skip run and just bind them to job via read-only response
        fingerprints = []
        for cfg in configs:
            fingerprints.append(cfg.fingerprint())
        have_all = all(btdb.has_result_for_fingerprint(fp) for fp in fingerprints)
        if have_all and req.use_cache:
            # bind existing configs to this job for visibility
            try:
                btdb.update_job_for_fingerprints(fingerprints, job_id)
            except Exception:
                pass
            btdb.update_job_status(job_id, status="done")
            return CreateBacktestResponse(job_id=job_id, n_configs=len(configs))
        # Enqueue async job: store configs and mark job pending
        # Upsert configs is already done in worker.run_job flow; here we store rows with job_id and payload
        for cfg in configs:
            _ = btdb.upsert_config({
                "job_id": job_id,
                "fingerprint": cfg.fingerprint(),
                "payload_json": {
                    "model": cfg.model_name,
                    "ptype": cfg.prediction_type,
                    "timeframe": cfg.timeframe,
                    "horizons_sec": cfg.horizons_sec,
                    "samples_range": cfg.samples_range,
                    "indicators": cfg.indicators,
                    "interval": cfg.interval,
                    "data_version": cfg.data_version,
                    "extra": cfg.extra,
                },
            })
        btdb.update_job_status(job_id, status="pending")
        # Ensure queue is running (singleton per process)
        try:
            global _bt_queue
        except Exception:
            _bt_queue = None
        if _bt_queue is None:
            _bt_queue = BacktestQueue(poll_interval=0.5)
            _bt_queue.start()
        return CreateBacktestResponse(job_id=job_id, n_configs=len(configs))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("create_backtest failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/status")
def get_status(job_id: int) -> Dict[str, Any]:
    btdb = BacktestDB()
    counts = btdb.job_status_counts(job_id)
    # include job status and simple progress ratio
    job_status = btdb.get_job_status(job_id) or "unknown"
    # derive simple status
    status = "running"
    if counts.get("n_configs", 0) > 0 and counts.get("n_results", 0) >= counts.get("n_configs", 0):
        status = "done"
    progress = 0.0
    try:
        ncfg = max(1, int(counts.get("n_configs", 0)))
        progress = float(min(1.0, (counts.get("n_results", 0) + counts.get("n_dropped", 0)) / ncfg))
    except Exception:
        progress = 0.0
    return {"job_id": job_id, "status": status, "job_status": job_status, "progress": progress, **counts}


class CancelJobRequest(BaseModel):
    reason: str | None = None


@router.post("/{job_id}/cancel")
def cancel_job(job_id: int, req: CancelJobRequest) -> Dict[str, Any]:
    btdb = BacktestDB()
    try:
        btdb.update_job_status(job_id, status="cancelled")
        return {"ok": True, "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{job_id}/pause")
def pause_job(job_id: int) -> Dict[str, Any]:
    btdb = BacktestDB()
    try:
        btdb.update_job_status(job_id, status="paused")
        return {"ok": True, "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{job_id}/resume")
def resume_job(job_id: int) -> Dict[str, Any]:
    btdb = BacktestDB()
    try:
        btdb.update_job_status(job_id, status="running")
        return {"ok": True, "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CancelConfigRequest(BaseModel):
    config_id: int
    reason: str | None = None


@router.post("/{job_id}/config/cancel")
def cancel_config(job_id: int, req: CancelConfigRequest) -> Dict[str, Any]:
    db = BacktestDB()
    try:
        db.cancel_config(int(req.config_id), reason=req.reason or "user_cancel")
        return {"ok": True, "config_id": int(req.config_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/config/{config_id}/profiles")
def get_profiles(job_id: int, config_id: int) -> Dict[str, Any]:
    db = BacktestDB()
    r = db.result_for_config(int(config_id))
    if not r:
        raise HTTPException(status_code=404, detail="result not found")
    return {
        "horizon_profile": r.get("horizon_profile_json"),
        "time_profile": r.get("time_profile_json"),
    }


@router.get("/{job_id}/results")
def get_results(job_id: int, top_k: int = 20) -> Dict[str, Any]:
    btdb = BacktestDB()
    rows = btdb.results_for_job(job_id)
    if not rows:
        return {"job_id": job_id, "top_k": top_k, "results": []}
    # Robust normalization: compute p10-p90 band for metrics used in CompositeScore
    import numpy as np
    def _collect(key: str) -> List[float]:
        vals = []
        for r in rows:
            v = r.get(key, None)
            try:
                vals.append(float(v))
            except Exception:
                continue
        return vals
    def _robust_norm(val: float, arr: List[float]) -> float:
        if not arr:
            return 0.0
        p10 = float(np.percentile(arr, 10))
        p90 = float(np.percentile(arr, 90))
        if p90 <= p10:
            return 0.0
        return float((val - p10) / (p90 - p10))

    adh_arr = _collect("adherence_mean")
    std_arr = _collect("adherence_std")
    skill_arr = _collect("skill_rw")
    cov_err_arr = _collect("coverage_abs_error")
    be_arr = _collect("band_efficiency")

    scored = []
    for r in rows:
        adh = float(r.get("adherence_mean") or 0.0)
        std = float(r.get("adherence_std") or 0.0)
        win = float(r.get("win_rate_delta") or 0.0)
        skill = float(r.get("skill_rw") or 0.0)
        cov_err = float(r.get("coverage_abs_error") or 0.0)
        be = float(r.get("band_efficiency") or 0.0)
        s = 0.40 * _robust_norm(adh, adh_arr) \
            + 0.20 * win \
            + 0.10 * _robust_norm(skill, skill_arr) \
            + 0.10 * (1.0 - _robust_norm(std, std_arr)) \
            + 0.10 * (1.0 - _robust_norm(cov_err, cov_err_arr)) \
            + 0.05 * (1.0 - _robust_norm(be, be_arr)) \
            + 0.05 * 0.0
        r_out = {"config_id": r["config_id"], "composite_score": float(s), "payload": r.get("payload_json"),
                 "adherence_mean": adh, "adherence_std": std, "p50": float(r.get("p50") or 0.0),
                 "win_rate_delta": win, "n_points": int(r.get("n_points") or 0),
                 "coverage_observed": r.get("coverage_observed"), "coverage_target": r.get("coverage_target"),
                 "coverage_abs_error": r.get("coverage_abs_error"),
                 "band_efficiency": r.get("band_efficiency"),
                 }
        scored.append(r_out)
    # Tie-breakers: p50 desc -> p10 desc -> complexity_penalty asc(placeholder=0) -> n_points desc
    def _key(x: Dict[str, Any]):
        return (
            float(x.get("composite_score", 0.0)),
            float(x.get("p50", 0.0)),
            float(x.get("adherence_mean", 0.0)),
            -float(x.get("n_points", 0)),
        )
    ranked = sorted(scored, key=_key, reverse=True)[: max(1, int(top_k))]
    return {"job_id": job_id, "top_k": top_k, "results": ranked}


class ApplyConfigRequest(BaseModel):
    config_id: int
    target: str  # "Basic" | "Advanced"


@router.post("/{job_id}/apply-config")
def apply_config(job_id: int, req: ApplyConfigRequest) -> Dict[str, Any]:
    """Apply the selected configuration to Prediction Settings (stub mapping)."""
    try:
        db = BacktestDB()
        # fetch config payload
        rows = db.results_for_job(job_id)
        row = next((r for r in rows if int(r.get("config_id")) == int(req.config_id)), None)
        if not row:
            raise HTTPException(status_code=404, detail="config not found in job")
        payload = row.get("payload_json") or {}
        # build mapping into dialog settings structure
        mapped = {
            "horizons": [str(h) for h in (payload.get("horizons_sec") or [])],
            "indicator_tfs": payload.get("indicators") or {},
            "forecast_types": [req.target.lower()],
            "model_paths": [payload.get("model")] if payload.get("model") else [],
        }
        # persist by loading current and merging
        current = PredictionSettingsDialog.get_settings()
        current.update(mapped)
        from pathlib import Path
        import json as _json
        cfg_file = PredictionSettingsDialog.CONFIG_FILE if hasattr(PredictionSettingsDialog, "CONFIG_FILE") else None
        if cfg_file is None:
            # fallback to known path
            from pathlib import Path
            cfg_file = Path(__file__).resolve().parents[3] / "configs" / "prediction_settings.json"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(_json.dumps(current, indent=4), encoding="utf-8")
        return {"ok": True, "applied_to": req.target}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



