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
    End-to-end predict helper (Advanced).
    Restituisce:
      - "pred_returns": np.ndarray (horizon,) con i ritorni previsti (o valori)
      - "pred_prices": np.ndarray (horizon,) con i prezzi ricostruiti
      - "future_ts": list[pd.Timestamp] con le barre future
      - "payload": payload del modello
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # 1) Carica payload modello (serve per features_list e standardizzazione)
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {mp}")

    payload = pickle_load_safe(mp)  # deve essere un dict con chiavi: model, features, std_mu, std_sigma, ...
    model = payload.get("model")
    features_list = payload.get("features") or []
    std_mu = payload.get("std_mu") or {}
    std_sigma = payload.get("std_sigma") or {}
    encoder = payload.get("encoder", None)

    if model is None:
        raise RuntimeError("Model payload missing 'model'")
    if not isinstance(features_list, (list, tuple)) or len(features_list) == 0:
        raise RuntimeError("Model payload missing 'features' list")

    # 2) Costruisci feature via pipeline (USA davvero i parametri Advanced se forniti)
    feats_cfg = features_config or {"warmup_bars": 16, "indicators": {}}
    feats_df, _std = pipeline_process(candles_df.copy(), timeframe=timeframe, features_config=feats_cfg)
    if feats_df is None or feats_df.empty:
        raise RuntimeError("pipeline_process produced empty features_df")

    # 3) Ensure: allinea/crea le feature richieste dal modello (stesso ordine)
    if ensure_cfg is not None:
        try:
            feats_df = ensure_features_for_prediction(
                feats_df, timeframe=timeframe, features_list=features_list, adv_cfg=ensure_cfg
            )
        except Exception:
            # non fatale: gestiremo dopo le colonne mancanti
            pass

    # 4) Colonne mancanti → riempi con μ (così dopo z-score ≈ 0)
    missing = [f for f in features_list if f not in feats_df.columns]
    for col in missing:
        try:
            fill_val = float(std_mu.get(col, 0.0))
        except Exception:
            fill_val = 0.0
        feats_df[col] = fill_val

    # 5) Ordine colonne come in training, pulizia e standardizzazione per colonna
    X_df = (
        feats_df[features_list]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    for col in features_list:
        mu_v = std_mu.get(col)
        sig_v = std_sigma.get(col)
        if mu_v is not None and sig_v is not None:
            denom = float(sig_v) if float(sig_v) != 0.0 else 1.0
            X_df[col] = (X_df[col] - float(mu_v)) / denom

    # 6) Costruisci X per la sequenza: ultime 'horizon' righe
    rows_needed = max(int(horizon), 1)
    X_arr = X_df.tail(rows_needed).to_numpy(dtype=float)

    # 7) Inferenza: prova PyTorch poi fallback sklearn
    preds_seq = None
    try:
        import torch
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            t_in = torch.tensor(X_arr, dtype=torch.float32)
            out = model(t_in)
            preds_seq = np.ravel(out.detach().cpu().numpy())
    except Exception:
        if hasattr(model, "predict"):
            preds_seq = np.ravel(model.predict(X_arr))
        else:
            try:
                preds_seq = np.array([float(model)], dtype=float)
            except Exception as e:
                raise RuntimeError(f"Unsupported model type for prediction: {e}")

    if preds_seq.size == 0:
        raise RuntimeError("Model returned empty prediction")

    # 8) Allinea a horizon (trim/pad)
    if preds_seq.size >= horizon:
        seq = preds_seq[-horizon:]
    else:
        pad_n = horizon - preds_seq.size
        seq = np.concatenate([preds_seq, np.repeat(preds_seq[-1], pad_n)])

    # 9) Ricostruzione prezzi e timestamps futuri
    last_close = float(candles_df["close"].iat[-1])
    prices = []
    p = last_close
    for r in seq:
        p = p * (1.0 + float(r))
        prices.append(p)
    pred_prices = np.asarray(prices, dtype=float)

    last_ts = pd.to_datetime(candles_df["ts_utc"].astype("int64"), unit="ms", utc=True).iat[-1]
    delta = timeframe_to_pandas_timedelta(timeframe)
    future_ts = [last_ts + delta * (i + 1) for i in range(horizon)]

    return {
        "pred_returns": np.asarray(seq, dtype=float),
        "pred_prices": pred_prices,
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
