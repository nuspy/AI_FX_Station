# src/forex_diffusion/inference/prediction.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable
_ensure_features_for_prediction: Optional[Callable] = None
try:
    from forex_diffusion.inference.prediction_config import ensure_features_for_prediction as _ensure_features_for_prediction
except Exception:
    try:
        from .prediction_config import ensure_features_for_prediction as _ensure_features_for_prediction  # type: ignore
    except Exception:
        _ensure_features_for_prediction = None

import hashlib
import numpy as np
import pandas as pd

from forex_diffusion.features.pipeline import pipeline_process, Standardizer
from forex_diffusion.inference.prediction_config import ensure_features_for_prediction
from loguru import logger

try:
    import torch
    TORCH = True
except Exception:
    TORCH = False


def _pickle_load(path: Path):
    import pickle, json
    b = path.read_bytes()
    try:
        return pickle.loads(b)
    except Exception:
        try:
            return json.loads(b.decode("utf-8"))
        except Exception as e:
            raise e

def _prepare_stats(payload: Dict[str, Any], feature_names: Sequence[str]) -> tuple[Dict[str, float], Dict[str, float]]:
    mu = payload.get("std_mu")
    sg = payload.get("std_sigma")
    if isinstance(mu, dict) and isinstance(sg, dict) and mu and sg:
        return dict(mu), dict(sg)
    legacy = payload.get("std")
    if isinstance(legacy, dict) and legacy:
        return {k: float(v) for k, v in legacy.items()}, {f: 1.0 for f in feature_names}
    return {}, {}

def _zscore(df: pd.DataFrame, mu: Dict[str, float], sg: Dict[str, float], cols: Sequence[str]) -> pd.DataFrame:
    X = df.loc[:, list(cols)].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    for c in cols:
        if c in mu and c in sg:
            d = float(sg[c]) if float(sg[c]) != 0.0 else 1.0
            X[c] = (X[c] - float(mu[c])) / d
    return X

def timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    s = (tf or "").strip().lower()
    if s.endswith("m"):
        v = int(s[:-1]) if s[:-1].isdigit() else 1
        return pd.to_timedelta(v, unit="m")
    if s.endswith("h"):
        v = int(s[:-1]) if s[:-1].isdigit() else 1
        return pd.to_timedelta(v, unit="h")
    if s.endswith("d"):
        v = int(s[:-1]) if s[:-1].isdigit() else 1
        return pd.to_timedelta(v, unit="d")
    return pd.to_timedelta(1, unit="m")

def end_to_end_predict(
    model_path: str | Path,
    candles_df: pd.DataFrame,
    timeframe: str,
    features_config: Optional[Dict[str, Any]] = None,
    horizon: int = 1,
    ensure_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Carica artifact, costruisce features (senza rifit standardizer), allinea schema, z-score con μ/σ di training e predice.
    Ritorna: pred_returns, pred_prices, future_ts, payload, model_path_used, model_sha16
    """
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {mp}")

    payload = _pickle_load(mp)
    model = payload.get("model")
    features_list = payload.get("features") or []
    if model is None or not features_list:
        raise RuntimeError("Model payload missing 'model' or 'features'")

    mu, sg = _prepare_stats(payload, features_list)
    try:
        sha16 = hashlib.sha256(mp.read_bytes()).hexdigest()[:16]
    except Exception:
        sha16 = None

    # pipeline: NO fit standardizer (no leakage)
    no_std = Standardizer(cols=[], mu={}, sigma={})
    feats_cfg = features_config or {"warmup_bars": 16, "indicators": {}}
    feats_df, _ = pipeline_process(candles_df.copy(), timeframe=timeframe, features_config=feats_cfg, standardizer=no_std)
    if feats_df is None or feats_df.empty:
        raise RuntimeError("pipeline_process produced empty features_df")

    # ensure + fill + z-score + ordine colonne

    # Ensure expected features from training exist
    if callable(_ensure_features_for_prediction):
        try:
            feats_df = _ensure_features_for_prediction(
                feats_df,
                timeframe=timeframe,
                features_list=features_list,
                adv_cfg=ensure_cfg or {
                    "rv_window": 60, "rsi_n": 14, "bb_n": 20, "don_n": 20, "hurst_window": 64,
                },
            )
        except Exception:
            pass


    for c in features_list:
        if c not in feats_df.columns:
            feats_df[c] = float(mu.get(c, 0.0))
    X_df = _zscore(feats_df, mu, sg, features_list)
    rows_needed = max(1, int(horizon))
    X_arr = X_df.tail(rows_needed).to_numpy(dtype=float)

    # inferenza
    preds = None
    if TORCH and hasattr(model, "eval"):
        try:
            model.eval()
            with torch.no_grad():  # type: ignore
                t_in = torch.tensor(X_arr, dtype=torch.float32)  # type: ignore
                out = model(t_in)
                preds = np.ravel(out.detach().cpu().numpy())
        except Exception:
            preds = None
    if preds is None:
        if hasattr(model, "predict"):
            preds = np.ravel(model.predict(X_arr))
        else:
            raise RuntimeError(f"Unsupported model type for prediction: {type(model)}")

    if preds.size == 0:
        raise RuntimeError("Model returned empty prediction")

    # allinea a horizon
    if preds.size >= rows_needed:
        seq = preds[-rows_needed:]
    else:
        pad = rows_needed - preds.size
        seq = np.concatenate([preds, np.repeat(preds[-1], pad)])

    # prezzi e ts futuri
    last_close = float(candles_df["close"].iat[-1])
    prices = []
    p = last_close
    for r in seq:
        p *= (1.0 + float(r))
        prices.append(p)
    pred_prices = np.asarray(prices, dtype=float)

    last_ts = pd.to_datetime(candles_df["ts_utc"].astype("int64"), unit="ms", utc=True).iat[-1]
    delta = timeframe_to_timedelta(timeframe)
    future_ts = [last_ts + delta * (i + 1) for i in range(rows_needed)]

    return {
        "pred_returns": np.asarray(seq, dtype=float),
        "pred_prices": pred_prices,
        "future_ts": future_ts,
        "payload": payload,
        "model_path_used": str(mp),
        "model_sha16": sha16,
    }
