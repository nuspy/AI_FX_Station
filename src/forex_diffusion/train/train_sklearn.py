# src/forex_diffusion/training/train_sklearn.py
# Train a simple sklearn forecaster (ridge/lasso/elasticnet) and save an artifact
# with: model, features (ordered), std_mu, std_sigma, encoder meta, name.
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error

# --- project imports (work with package or local src layout) ---------------
try:
    from forex_diffusion.features.pipeline import pipeline_process, Standardizer
except Exception:  # run as a module inside the repo
    from ..features.pipeline import pipeline_process, Standardizer  # type: ignore

try:
    from forex_diffusion.services.marketdata import MarketDataService
except Exception:
    from ..services.marketdata import MarketDataService  # type: ignore


# ---------------------------- helpers --------------------------------------

def _clean_symbol_for_name(symbol: str) -> str:
    # EUR/USD -> EURUSD, GBP-EUR -> GBPEUR
    return "".join(ch for ch in symbol if ch.isalnum()).upper()


def _fetch_candles_from_db(
    mkt: MarketDataService,
    symbol: str,
    timeframe: str,
    limit: int = 200_000,
) -> pd.DataFrame:
    """
    Pulls recent candles from the project's DB (table: market_data_candles).
    """
    from sqlalchemy import MetaData, select
    meta = MetaData()
    meta.reflect(bind=mkt.engine, only=["market_data_candles"])
    tbl = meta.tables.get("market_data_candles")
    if tbl is None:
        raise RuntimeError("Table 'market_data_candles' not found in DB")

    with mkt.engine.connect() as conn:
        stmt = (
            select(
                tbl.c.ts_utc, tbl.c.open, tbl.c.high, tbl.c.low, tbl.c.close, tbl.c.volume
            )
            .where(tbl.c.symbol == symbol)
            .where(tbl.c.timeframe == timeframe)
            .order_by(tbl.c.ts_utc.desc())
            .limit(limit)
        )
        rows = conn.execute(stmt).fetchall()

    if not rows:
        raise RuntimeError(f"No candles for {symbol} {timeframe}")
    df = pd.DataFrame(rows, columns=["ts_utc", "open", "high", "low", "close", "volume"])
    return df.sort_values("ts_utc").reset_index(drop=True)


def _build_supervised_target(candles: pd.DataFrame, horizon_bars: int) -> pd.Series:
    """
    Next-H simple return as target: y_t = close_{t+H}/close_t - 1
    """
    c = candles["close"].astype(float)
    fut = c.shift(-int(horizon_bars))
    y = (fut / c) - 1.0
    return y


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series


def _time_series_split(X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2) -> Split:
    """
    Simple chronological split: last val_frac as validation.
    """
    n = len(X)
    if n < 100:
        raise RuntimeError("Not enough rows for a train/val split")
    cut = int(n * (1.0 - float(val_frac)))
    return Split(
        X_train=X.iloc[:cut, :].copy(),
        y_train=y.iloc[:cut].copy(),
        X_val=X.iloc[cut:, :].copy(),
        y_val=y.iloc[cut:].copy(),
    )


def _maybe_pca(X_train_z: pd.DataFrame, X_val_z: pd.DataFrame, n_components: int | None):
    """
    Optional PCA encoder. Returns transformed arrays and encoder metadata.
    """
    if not n_components or n_components <= 0:
        return X_train_z.to_numpy(float), X_val_z.to_numpy(float), {"type": "none"}

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    Xt = pca.fit_transform(X_train_z.to_numpy(float))
    Xv = pca.transform(X_val_z.to_numpy(float))
    enc = {"type": "pca", "n_components": int(n_components), "pca": pca}
    return Xt, Xv, enc


def _build_features(
    candles: pd.DataFrame, timeframe: str, warmup_bars: int
) -> pd.DataFrame:
    """
    Calls your pipeline to compute causal features. We pass a NO-OP standardizer
    for training feature creation; we'll fit our own Standardizer on TRAIN only.
    """
    no_std = Standardizer(cols=[], mu={}, sigma={})
    feats_cfg = {
        "warmup_bars": int(warmup_bars),
        # Must be a dict: pipeline expects .get on this field
        "standardization": {"window_bars": 1000},
        "indicators": {
            # Add/adjust indicators as you like; they match common defaults in your repo
            "atr": {"n": 14},
            "rsi": {"n": 14},
            "bollinger": {"n": 20},
            "hurst": {"window": 64},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "donchian": {"n": 20},
        },
    }
    feats_df, _ = pipeline_process(
        candles.copy(), timeframe=timeframe, features_config=feats_cfg, standardizer=no_std
    )
    if feats_df is None or feats_df.empty:
        raise RuntimeError("pipeline_process produced empty features")
    return feats_df


# ---------------------------- main training --------------------------------

def train_and_save(
    symbol: str,
    timeframe: str,
    horizon_bars: int,
    algo: str = "ridge",
    pca_components: int | None = None,
    artifacts_dir: str = "artifacts",
    warmup_bars: int = 64,
    val_frac: float = 0.2,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
) -> Path:
    """
    End-to-end: load candles -> features -> target -> split -> standardize (TRAIN only) ->
    fit sklearn model -> SAVE artifact with model, features, std_mu, std_sigma, name.
    """
    # 1) Load candles from DB
    mkt = MarketDataService()
    candles = _fetch_candles_from_db(mkt, symbol=symbol, timeframe=timeframe)

    # 2) Build features and supervised target
    feats = _build_features(candles, timeframe=timeframe, warmup_bars=warmup_bars)
    y = _build_supervised_target(candles, horizon_bars=horizon_bars)

    # Align and drop NaNs (causal)
    df = feats.join(y.rename("y")).dropna()
    X_full = df.drop(columns=["y"])
    y_full = df["y"].astype(float)

    # 3) Chronological split
    split = _time_series_split(X_full, y_full, val_frac=val_frac)

    # 4) Standardize on TRAIN only
    standardizer = Standardizer(cols=list(split.X_train.columns))
    X_train_z = standardizer.fit_transform(split.X_train.copy())
    X_val_z = standardizer.transform(split.X_val.copy())

    # 5) Optional PCA
    X_train_in, X_val_in, encoder = _maybe_pca(X_train_z, X_val_z, pca_components)

    # 6) Pick and fit model
    algo = (algo or "ridge").lower()
    if algo == "ridge":
        model = Ridge(alpha=alpha, random_state=0)
    elif algo == "lasso":
        model = Lasso(alpha=alpha, random_state=0)
    elif algo in ("elasticnet", "enet", "elastic_net"):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    fitted_model = model.fit(X_train_in, split.y_train.to_numpy(float))

    # 7) Quick validation metrics (optional but useful)
    y_pred_val = fitted_model.predict(X_val_in)
    r2 = r2_score(split.y_val, y_pred_val)
    mae = mean_absolute_error(split.y_val, y_pred_val)
    print(f"[val] R2={r2:.4f}  MAE={mae:.6f}  n_train={len(split.X_train)}  n_val={len(split.X_val)}")

    # 8) SAVE artifact — this is your requested snippet, integrated
    enc = encoder.get("type", "none")
    H = int(horizon_bars)
    sym_clean = _clean_symbol_for_name(symbol)
    name = f"weighted_forecast_{sym_clean}_{timeframe}_h{H}_{algo}_{enc}"

    payload: Dict[str, object] = {
        "model": fitted_model,                         # e.g., Ridge/Lasso/ElasticNet
        "features": list(split.X_train.columns),       # ordered feature names
        "std_mu": dict(standardizer.mu),              # dict[str, float]
        "std_sigma": dict(standardizer.sigma),        # dict[str, float]
        "encoder": encoder,                            # {"type":"none"} or {"type":"pca", ...}
        "name": name,
        "output_type": "returns",
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_bars": H,
    }

    out_dir = Path(artifacts_dir) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.pkl"

    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[saved] {out_path}")
    return out_path


# ----------------------------- CLI -----------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train sklearn forecaster and save artifact (.pkl)")
    p.add_argument("--symbol", required=True, help="e.g., EUR/USD")
    p.add_argument("--timeframe", required=True, help="e.g., 1m, 5m, 15m")
    p.add_argument("--horizon", type=int, required=True, help="H bars ahead (e.g., 5 means t+5)")
    p.add_argument("--algo", choices=["ridge", "lasso", "elasticnet"], default="ridge")
    p.add_argument("--pca", type=int, default=0, help="PCA components (0 disables)")
    p.add_argument("--artifacts_dir", default="artifacts", help="Base artifacts directory")
    p.add_argument("--warmup_bars", type=int, default=64)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--l1_ratio", type=float, default=0.5, help="ElasticNet only")
    return p.parse_args()


def main():
    args = _parse_args()
    train_and_save(
        symbol=args.symbol,
        timeframe=args.timeframe,
        horizon_bars=args.horizon,
        algo=args.algo,
        pca_components=(args.pca if args.pca and args.pca > 0 else None),
        artifacts_dir=args.artifacts_dir,
        warmup_bars=args.warmup_bars,
        val_frac=args.val_frac,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
    )


if __name__ == "__main__":
    main()
