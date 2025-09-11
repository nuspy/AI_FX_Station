# src/forex_diffusion/inference/prediction.py

"""
src/forex_diffusion/inference/prediction.py

Utility helper end_to_end_predict to:
 - load a saved model payload (pickle)
 - compute features via pipeline_process from candles dataframe
 - apply saved standardization (std_mu/std_sigma)
 - handle encoder types (none / pca / latents)
 - return predictions (returns) and forecast prices (for given horizon)

This helper is synchronous and minimal; caller is responsible to supply candles_df with
columns ['ts_utc','open','high','low','close'] and for latents mode the DB must contain matching latents.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Optional, Tuple, Sequence

import numpy as np
import pandas as pd

from forex_diffusion.features.pipeline import pipeline_process
from forex_diffusion.services.db_service import DBService
from forex_diffusion.inference.prediction_config import ensure_features_for_prediction

# Optional sklearn / torch will be used if present at runtime
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

def _apply_standardization_row(row: pd.Series, mu: Dict[str, float], sigma: Dict[str, float], features: Sequence[str]) -> np.ndarray:
    vals = []
    for f in features:
        v = row.get(f, 0.0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        mu_v = mu.get(f, None) if isinstance(mu, dict) else None
        sig_v = sigma.get(f, None) if isinstance(sigma, dict) else None
        if mu_v is not None and sig_v is not None:
            denom = float(sig_v) if float(sig_v) != 0.0 else 1.0
            vals.append((v - float(mu_v)) / denom)
        else:
            vals.append(v)
    return np.asarray(vals, dtype=float).reshape(1, -1)

def end_to_end_predict(
    model_path: str | Path,
    candles_df: pd.DataFrame,
    timeframe: str,
    features_config: Optional[Dict] = None,
    horizon: int = 1,
    ensure_cfg: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    End-to-end predict helper.

    Returns dict with keys:
      - "pred_returns": numpy array length==horizon of predicted returns (if model provides sequence or single value repeated)
      - "pred_prices": numpy array of forecast prices (for given horizon)
      - "future_ts": list of pd.Timestamp for predicted bars
      - "payload": loaded model payload (for debugging)
    """
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {mp}")

    payload = pickle_load_safe(mp)
    model = payload.get("model")
    features = payload.get("features") or []
    std_mu = payload.get("std_mu") or {}
    std_sigma = payload.get("std_sigma") or {}
    encoder = payload.get("encoder", None)

    # compute features via pipeline
    feats_cfg = features_config or {"warmup_bars": 16, "indicators": {}}
    feats_df, _std = pipeline_process(candles_df.copy(), timeframe=timeframe, features_config=feats_cfg)
    if "ts_utc" not in feats_df.columns:
        feats_df = feats_df.reset_index(drop=True)
    if feats_df.empty:
        raise RuntimeError("pipeline produced empty features_df")

    # ensure required features exist
    try:
        feats_df = ensure_features_for_prediction(feats_df, timeframe, features, adv_cfg=ensure_cfg)
    except Exception:
        pass

    # build X depending on encoder
    if encoder and isinstance(encoder, dict) and encoder.get("type") == "latents":
        # need to fetch latent for last ts from DB
        last_ts = int(candles_df["ts_utc"].iat[-1])
        db = DBService()
        with db.engine.connect() as conn:
            r = conn.execute(
                "SELECT latent_json FROM latents WHERE ts_utc = :t AND timeframe=:tf LIMIT 1",
                {"t": last_ts, "tf": timeframe},
            ).fetchone()
            if r is None:
                r = conn.execute(
                    "SELECT latent_json FROM latents WHERE timeframe=:tf ORDER BY ts_utc DESC LIMIT 1",
                    {"tf": timeframe},
                ).fetchone()
            if r is None:
                raise RuntimeError("No latent vector available for query")
            try:
                lj = r._mapping["latent_json"]
            except Exception:
                lj = r[0] if len(r) > 0 else None
            vec = np.asarray(json.loads(lj), dtype=float).reshape(1, -1)
            X_arr = vec
    else:
        if not features:
            raise RuntimeError("Model payload missing 'features' list; cannot build input X")
        # >>> FIX: build multiple rows, standardize like the simple-forecast path
        rows_needed = max(int(horizon), 1)
        sub_df = feats_df.tail(rows_needed).copy()
        missing_cols = [f for f in features if f not in sub_df.columns]
        for col in missing_cols:
            try:
                fill_val = float(std_mu.get(col, 0.0)) if isinstance(std_mu, dict) else 0.0
            except Exception:
                fill_val = 0.0
            sub_df[col] = fill_val
        X_df = sub_df[features].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in features:
            mu_v = std_mu.get(col)
            sig_v = std_sigma.get(col)
            if mu_v is not None and sig_v is not None:
                denom = float(sig_v) if float(sig_v) != 0.0 else 1.0
                X_df[col] = (X_df[col] - float(mu_v)) / denom
        X_arr = X_df.to_numpy(dtype=float)
        # <<< FIX

    # predict: support sklearn-like and torch
    preds_seq = None
    if TORCH_AVAILABLE and (hasattr(model, "forward") or isinstance(model, object) and "torch" in str(type(model)).lower()):
        try:
            model.eval()
            import torch
            with torch.no_grad():
                t_in = torch.tensor(X_arr, dtype=torch.float32)
                out = model(t_in)
                preds = out.cpu().numpy()
                preds_seq = np.ravel(preds)
        except Exception:
            preds_seq = None

    if preds_seq is None:
        if hasattr(model, "predict"):
            preds = model.predict(X_arr)
            preds_seq = np.ravel(preds)
        else:
            try:
                val = float(model)
                preds_seq = np.array([val])
            except Exception:
                raise RuntimeError("Unsupported model type for prediction")

    # build horizon sequence
    if preds_seq.size == 0:
        raise RuntimeError("Model returned empty prediction")
    if preds_seq.size >= horizon:
        seq = preds_seq[-horizon:]
    else:
        seq = np.pad(preds_seq, (horizon - preds_seq.size, 0), mode="edge")

    last_close = float(candles_df["close"].iat[-1])
    last_ts = pd.to_datetime(candles_df["ts_utc"].astype("int64"), unit="ms", utc=True).iat[-1]
    delta = timeframe_to_pandas_timedelta(timeframe)
    future_ts = [last_ts + delta * (i + 1) for i in range(len(seq))]

    # convert returns to prices (coerente con il percorso semplice)
    prices = []
    p = last_close
    for r in seq:
        p = p * (1.0 + float(r))
        prices.append(p)

    return {
        "pred_returns": np.asarray(seq, dtype=float),
        "pred_prices": np.asarray(prices, dtype=float),
        "future_ts": future_ts,
        "payload": payload,
    }


# helpers

def pickle_load_safe(path: Path):
    import pickle, json
    b = path.read_bytes()
    try:
        return pickle.loads(b)
    except Exception:
        # try JSON (fallback)
        try:
            return json.loads(b.decode("utf-8"))
        except Exception as e:
            raise

def timeframe_to_pandas_timedelta(tf: str):
    tf = tf.strip().lower()
    if tf.endswith("m"):
        try:
            v = int(tf[:-1]); return pd.to_timedelta(v, unit="m")
        except Exception:
            return pd.to_timedelta(1, unit="m")
    if tf.endswith("h"):
        try:
            v = int(tf[:-1]); return pd.to_timedelta(v, unit="h")
        except Exception:
            return pd.to_timedelta(1, unit="h")
    if tf.endswith("d"):
        try:
            v = int(tf[:-1]); return pd.to_timedelta(v, unit="d")
        except Exception:
            return pd.to_timedelta(1, unit="d")
    # fallback 1 minute
    return pd.to_timedelta(1, unit="m")
