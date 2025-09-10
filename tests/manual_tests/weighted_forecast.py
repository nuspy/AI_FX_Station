#!/usr/bin/env python3
"""
tests/manual_tests/weighted_forecast.py

Supervised + NN hybrid forecasting using all multi-timeframe indicators and time-features.

Features:
- Model/encoder selection:
  --model [ridge|lasso|elasticnet|rf]
  --encoder [none|pca|latents], --encoder_dim (for pca)
- Feature pipeline params:
  --warmup_bars, --atr_n, --rsi_n, --bb_n, --hurst_window, --rv_window, --multi_timeframes
- Clustering/index (optional):
  --build_index, --n_clusters, --index_space, --ef_construction, --M, --ef
- Query (optional, NN forecast):
  --k, --query_vec [last|meanN], --query_N
- Post-process forecast:
  --forecast_method [supervised|nn_mean|nn_median|nn_weighted]
  --horizons "5,10,20"

Usage (PowerShell):
  .\\scripts\\weighted_forecast.ps1 -Symbol "EUR/USD" -Timeframe "1m" -Horizon 5 -Days 7 -Model ridge -Encoder none -ForecastMethod supervised

Outputs:
- Prints metrics and, for linear models, per-feature coefficients (weights).
- For NN methods, prints horizon-wise aggregated returns.
- Saves model/artifacts to artifacts/models.
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import json
import pickle
import math

ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def main():
    p = argparse.ArgumentParser()
    # core
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--horizon", type=int, default=5, help="Forecast horizon in bars (for supervised)")
    p.add_argument("--horizons", default="5,10,20", help="Comma-separated horizons for NN methods (bars)")
    p.add_argument("--days", type=int, default=7, help="Days of history to use for training")
    p.add_argument("--multi_timeframes", nargs="*", default=["5m","10m","20m","30m","45m","1h","2h","4h","1d"])
    # pipeline params
    p.add_argument("--warmup_bars", type=int, default=16)
    p.add_argument("--atr_n", type=int, default=14)
    p.add_argument("--rsi_n", type=int, default=14)
    p.add_argument("--bb_n", type=int, default=20)
    p.add_argument("--hurst_window", type=int, default=64)
    p.add_argument("--rv_window", type=int, default=60)
    # model/encoder
    p.add_argument("--model", default="ridge", choices=["ridge","lasso","elasticnet","rf"])
    p.add_argument("--alpha", type=float, default=1.0, help="Regularization (ridge/lasso/elasticnet)")
    p.add_argument("--l1_ratio", type=float, default=0.5, help="ElasticNet l1_ratio")
    p.add_argument("--n_estimators", type=int, default=200, help="RF trees")
    p.add_argument("--max_depth", type=int, default=None, help="RF max_depth")
    p.add_argument("--test_frac", type=float, default=0.2, help="Fraction for test split (last part)")
    p.add_argument("--weights_file", default=None, help="Optional JSON file with feature multipliers (feature->multiplier)")
    p.add_argument("--encoder", default="none", choices=["none","pca","latents"])
    p.add_argument("--encoder_dim", type=int, default=64, help="Dimensionality for PCA encoder")
    # clustering + index (optional)
    p.add_argument("--build_index", action="store_true", help="Rebuild clusters/index from latents")
    p.add_argument("--n_clusters", type=int, default=8)
    p.add_argument("--index_space", default="l2")
    p.add_argument("--ef_construction", type=int, default=200)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef", type=int, default=50, help="HNSW ef for queries")
    # NN query
    p.add_argument("--k", type=int, default=10, help="Neighbors for NN forecasting")
    p.add_argument("--query_vec", default="last", choices=["last","meanN"])
    p.add_argument("--query_N", type=int, default=5)
    # method
    p.add_argument("--forecast_method", default="supervised", choices=["supervised","nn_mean","nn_median","nn_weighted"])
    args = p.parse_args()

    try:
        from forex_diffusion.services.db_service import DBService
        from forex_diffusion.features.pipeline import pipeline_process
        from forex_diffusion.services.regime_service import RegimeService
        from sqlalchemy import text
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.decomposition import PCA
        from sklearn.metrics import r2_score, mean_absolute_error
    except Exception as e:
        print("Import failed:", e)
        raise SystemExit(1)

    db = DBService()
    eng = db.engine

    def load_candles(limit_rows: int) -> pd.DataFrame:
        """
        Load candles robustly: try to select volume column first; if the column is missing or the DB errors,
        retry without volume. Returns DataFrame with columns present.
        """
        with eng.connect() as conn:
            # Try with volume
            try:
                q1 = text("SELECT ts_utc, open, high, low, close, volume FROM market_data_candles WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc ASC LIMIT :lim")
                rows = conn.execute(q1, {"s": args.symbol, "tf": args.timeframe, "lim": limit_rows}).fetchall()
                if not rows:
                    print("No candles found for", args.symbol, args.timeframe); sys.exit(2)
                out = []
                for r in rows:
                    if hasattr(r, "_mapping"):
                        out.append(dict(r._mapping))
                    else:
                        out.append({
                            "ts_utc": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5] if len(r) > 5 else None
                        })
                return pd.DataFrame(out)
            except Exception:
                # Fallback: without volume
                q2 = text("SELECT ts_utc, open, high, low, close FROM market_data_candles WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc ASC LIMIT :lim")
                rows2 = conn.execute(q2, {"s": args.symbol, "tf": args.timeframe, "lim": limit_rows}).fetchall()
                if not rows2:
                    print("No candles found for", args.symbol, args.timeframe); sys.exit(2)
                out2 = []
                for r in rows2:
                    if hasattr(r, "_mapping"):
                        # mapping may include volume absent; keep mapping keys present
                        d = dict(r._mapping)
                        # ensure keys exist
                        for k in ("ts_utc","open","high","low","close"):
                            if k not in d:
                                # fallback to positional tuple if mapping misses keys
                                pass
                        out2.append(d)
                    else:
                        out2.append({
                            "ts_utc": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4]
                        })
                return pd.DataFrame(out2)

    # rows estimate
    if args.timeframe.endswith("m") or args.timeframe == "1m":
        rows_needed = args.days * 24 * 60 + max(args.horizon, 20) + 200
    else:
        rows_needed = args.days * 24 + max(args.horizon, 20) + 200

    df = load_candles(rows_needed)
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # future returns for supervised
    df["close_future"] = df["close"].shift(-args.horizon)
    df["ret_fut"] = (df["close_future"] - df["close"]) / df["close"]
    df = df.dropna(subset=["ret_fut"]).reset_index(drop=True)

    # pipeline config
    features_config = {
        "warmup_bars": int(args.warmup_bars),
        "indicators": {
            "atr": {"n": int(args.atr_n)},
            "rsi": {"n": int(args.rsi_n)},
            "bollinger": {"n": int(args.bb_n)},
            "hurst": {"window": int(args.hurst_window)},
            "standardization": {},
        },
        "standardization": {"window_bars": int(max(100, args.rv_window))},
    }

    # compute features (NOTE: pipeline_process currently does not accept multi_timeframes param)
    if args.multi_timeframes and args.multi_timeframes != ["5m","10m","20m","30m","45m","1h","2h","4h","1d"]:
        logger = None
        try:
            from loguru import logger as _logger
            logger = _logger
        except Exception:
            import logging as _logging
            logger = _logging.getLogger("weighted_forecast")
        logger.warning("Requested multi_timeframes argument provided but pipeline_process() does not accept it; multi_timeframes will be ignored for this run.")

    feats, std = pipeline_process(df.copy(), timeframe=args.timeframe, features_config=features_config)

    # Ensure ts_utc present in features_df; if missing, infer from raw candles using warmup_bars
    if "ts_utc" not in feats.columns:
        try:
            warmup = int(features_config.get("warmup_bars", 0))
        except Exception:
            warmup = 0
        try:
            df_ts = df.reset_index(drop=True)["ts_utc"]
            aligned = df_ts.iloc[warmup : warmup + len(feats)].reset_index(drop=True)
            if len(aligned) == len(feats):
                feats = feats.reset_index(drop=True)
                feats["ts_utc"] = aligned
                try:
                    logger.info(f"Inferred ts_utc for features_df from raw candles using warmup={warmup}")
                except Exception:
                    pass
            else:
                # fallback: create numeric ts_utc placeholder to avoid KeyError and skip merge
                feats = feats.reset_index(drop=True)
                feats["ts_utc"] = [None] * len(feats)
                try:
                    logger.warning("Could not infer ts_utc for features; merged dataset may be empty")
                except Exception:
                    pass
        except Exception:
            feats = feats.reset_index(drop=True)
            feats["ts_utc"] = [None] * len(feats)

    merged = feats.merge(df[["ts_utc","ret_fut"]], on="ts_utc", how="inner")
    if merged.empty:
        print("No aligned feature+target rows (empty merged). Aborting."); sys.exit(3)

    # build X (all numeric features) and y
    X_full = merged.select_dtypes(include=[float, int]).copy()
    if "ret_fut" in X_full.columns: X_full.drop(columns=["ret_fut"], inplace=True)
    if "ts_utc" in X_full.columns: X_full.drop(columns=["ts_utc"], inplace=True)
    y_full = merged["ret_fut"].astype(float).to_numpy()

    # apply optional feature multipliers (all indicator/time columns included)
    if args.weights_file:
        try:
            with open(args.weights_file, "r", encoding="utf-8") as fh:
                multipliers = json.load(fh)
        except Exception as e:
            print("Could not load weights file:", e)
            multipliers = {}
        for col in X_full.columns:
            m = float(multipliers.get(col, 1.0))
            if m != 1.0:
                X_full[col] = X_full[col] * m

    # drop inf/nan
    X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # encoder
    enc_artifact = None
    if args.encoder == "pca":
        pca = PCA(n_components=min(args.encoder_dim, X_full.shape[1]))
        # split time-wise for fitting to avoid leakage (fit on train)
        n = len(X_full)
        split = int(n * (1.0 - args.test_frac))
        X_train_fit = X_full.iloc[:split].to_numpy(dtype=float)
        X_test_raw = X_full.iloc[split:].to_numpy(dtype=float)
        pca.fit(X_train_fit)
        X_full_enc = np.vstack([pca.transform(X_train_fit), pca.transform(X_test_raw)])
        enc_artifact = {"type": "pca", "components": pca.n_components_}
    elif args.encoder == "latents":
        # Use latents from DB instead of features
        with eng.connect() as conn:
            lat_rows = conn.execute(text("SELECT ts_utc, latent_json FROM latents WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc ASC"),
                                    {"s": args.symbol, "tf": args.timeframe}).fetchall()
        if not lat_rows:
            print("No latents available; use encoder=none or build latents first."); sys.exit(4)
        lat_df = pd.DataFrame([{"ts_utc": (r._mapping["ts_utc"] if hasattr(r, "_mapping") else r[0]),
                                 "vec": json.loads(r._mapping["latent_json"] if hasattr(r, "_mapping") else r[1])} for r in lat_rows])
        # align to merged timestamps
        L = []
        for ts in merged["ts_utc"]:
            rec = lat_df.loc[lat_df["ts_utc"] == int(ts)]
            if len(rec) == 0:
                L.append(None)
            else:
                L.append(np.asarray(rec["vec"].values[0], dtype=float))
        mask = [v is not None for v in L]
        if not any(mask):
            print("Could not align any latent vectors with target timestamps."); sys.exit(5)
        X_full_enc = np.vstack([v for v in L if v is not None])
        y_full = merged.loc[mask, "ret_fut"].astype(float).to_numpy()
        # also restrict features columns for reporting
        X_columns = [f"z{i}" for i in range(X_full_enc.shape[1])]
        X_full = pd.DataFrame(X_full_enc, columns=X_columns)
        enc_artifact = {"type": "latents", "dim": X_full_enc.shape[1]}
    else:
        X_full_enc = X_full.to_numpy(dtype=float)

    # split train/test
    n = len(X_full_enc)
    split = int(n * (1.0 - args.test_frac))
    X_train = X_full_enc[:split]
    y_train = y_full[:split]
    X_test = X_full_enc[split:]
    y_test = y_full[split:]

    # choose model
    if args.model == "ridge":
        model = Ridge(alpha=args.alpha, random_state=42)
    elif args.model == "lasso":
        model = Lasso(alpha=args.alpha, random_state=42, max_iter=10000)
    elif args.model == "elasticnet":
        model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42, max_iter=10000)
    else:
        model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42, n_jobs=-1)

    # if forecast_method is supervised: train/eval and report
    if args.forecast_method == "supervised":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        print(f"[Supervised] model={args.model} encoder={args.encoder} days={args.days} horizon={args.horizon} train={len(X_train)} test={len(X_test)}")
        print(f"R2={r2:.6f} MAE={mae:.6e}")

        # coefficients for linear models as weights
        if hasattr(model, "coef_"):
            cols = X_full.columns.tolist() if args.encoder != "latents" and args.encoder != "pca" else [f"f{i}" for i in range(X_full_enc.shape[1])]
            coefs = dict(zip(cols, model.coef_.tolist()))
            for f, c in sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:40]:
                print(f"  {f}: {c:.6e}")

        # persist
        ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{args.symbol.replace('/','')}_{args.timeframe}_h{args.horizon}_{args.model}_{args.encoder}"
        model_path = ARTIFACTS_DIR / f"weighted_forecast_{name}.pkl"
        # Persist model + metadata + full standardizer params (mu, sigma) for reproducible scoring
        model_payload = {
            "model": model,
            "features": (X_full.columns.tolist() if hasattr(X_full, "columns") else None),
            "encoder": enc_artifact,
            "std_mu": getattr(std, "mu", None),
            "std_sigma": getattr(std, "sigma", None),
        }
        with open(model_path, "wb") as fh:
            pickle.dump(model_payload, fh)
        print("Saved model to", model_path)
        return

    # else NN-based forecast (requires vector for query)
    # build or load index if using latents; else NN in feature space
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    if args.encoder == "latents":
        svc = RegimeService(engine=eng)
        if args.build_index:
            print(f"Building index: n_clusters={args.n_clusters}, space={args.index_space}, ef_construction={args.ef_construction}, M={args.M}")
            svc.fit_clusters_and_index(n_clusters=args.n_clusters, index_space=args.index_space, ef_construction=args.ef_construction, M=args.M)
        else:
            try:
                svc.load_index()
            except Exception:
                print("Index not found; building...")
                svc.fit_clusters_and_index(n_clusters=args.n_clusters, index_space=args.index_space, ef_construction=args.ef_construction, M=args.M)
        try:
            svc.index.set_ef(int(args.ef))
        except Exception:
            pass

        # build query vector (last or mean of last N latents)
        with eng.connect() as conn:
            q = text("SELECT latent_json FROM latents WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc DESC LIMIT :lim")
            rows = conn.execute(q, {"s": args.symbol, "tf": args.timeframe, "lim": (1 if args.query_vec=='last' else args.query_N)}).fetchall()
            if not rows:
                print("No latents available for query."); sys.exit(6)
            vecs = [np.asarray(json.loads((r._mapping["latent_json"] if hasattr(r, "_mapping") else r[0])), dtype=np.float32) for r in rows]
            if args.query_vec == "last":
                qvec = vecs[0]
            else:
                qvec = np.mean(vecs, axis=0).astype(np.float32)

        # find neighbors via index
        res = svc.query_regime(qvec.tolist(), k=args.k)
        neighbor_ids = res.get("neighbor_ids", [])
        # compute future returns for horizons
        with eng.connect() as conn:
            id_ts = conn.execute(text("SELECT id, ts_utc FROM latents WHERE id IN :ids"), {"ids": tuple(neighbor_ids) if neighbor_ids else (0,)}).fetchall()
            id2ts = {r[0]: r[1] for r in id_ts}
            forecasts = {h: [] for h in horizons}
            for nid in neighbor_ids:
                ts = id2ts.get(nid)
                if ts is None: continue
                base = conn.execute(text("SELECT close FROM market_data_candles WHERE symbol=:s AND timeframe=:tf AND ts_utc=:t LIMIT 1"),
                                    {"s": args.symbol, "tf": args.timeframe, "t": ts}).fetchone()
                if not base: continue
                base_close = float(base[0])
                fut_rows = conn.execute(text("SELECT close FROM market_data_candles WHERE symbol=:s AND timeframe=:tf AND ts_utc > :t ORDER BY ts_utc ASC LIMIT :lim"),
                                        {"s": args.symbol, "tf": args.timeframe, "t": ts, "lim": max(horizons)}).fetchall()
                for h in horizons:
                    if len(fut_rows) >= h:
                        fut_close = float(fut_rows[h-1][0])
                        forecasts[h].append((fut_close - base_close)/base_close)
        # aggregate by method
        for h in horizons:
            arr = np.asarray(forecasts[h], dtype=float) if forecasts[h] else np.array([])
            if arr.size == 0:
                print(f"[NN] H={h} no samples"); continue
            if args.forecast_method == "nn_mean":
                val = float(arr.mean())
            elif args.forecast_method == "nn_median":
                val = float(np.median(arr))
            else:
                # distance-weighted using returned distances if available
                # fallback to uniform if distances not exposed
                # Here we use a simple inverse-rank weighting as distances are not returned from query_regime JSON in detail
                ranks = np.arange(1, len(arr)+1, dtype=float)
                w = 1.0 / ranks
                w = w / w.sum()
                val = float((arr * w).sum())
            print(f"[NN {args.forecast_method}] H={h} n={arr.size} forecast_ret={val:.6e}")
    else:
        # NN in feature space (no index)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=args.k, metric="euclidean")
        nn.fit(X_full_enc)
        # query vector
        if args.query_vec == "last":
            qvec = X_full_enc[-1].reshape(1, -1)
        else:
            N = min(args.query_N, X_full_enc.shape[0])
            qvec = np.mean(X_full_enc[-N:], axis=0, keepdims=True)
        dists, idxs = nn.kneighbors(qvec, n_neighbors=args.k, return_distance=True)
        idxs = idxs[0].tolist()
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
        # map from index to ts to future returns using df
        ts_series = merged["ts_utc"].to_numpy()
        close_series = df.set_index("ts_utc")["close"]
        forecasts = {h: [] for h in horizons}
        for ii, rank in zip(idxs, range(1, len(idxs)+1)):
            ts = int(ts_series[ii])
            base_close = float(close_series.loc[ts])
            fut = close_series.loc[close_series.index > ts].iloc[:max(horizons)]
            for h in horizons:
                if len(fut) >= h:
                    fut_close = float(fut.iloc[h-1])
                    forecasts[h].append((fut_close - base_close)/base_close)
        for h in horizons:
            arr = np.asarray(forecasts[h], dtype=float) if forecasts[h] else np.array([])
            if arr.size == 0:
                print(f"[NN(feature)] H={h} no samples"); continue
            if args.forecast_method == "nn_median":
                val = float(np.median(arr))
            elif args.forecast_method == "nn_weighted":
                ranks = np.arange(1, len(arr)+1, dtype=float)
                w = 1.0/ranks; w = w/w.sum()
                val = float((arr*w).sum())
            else:
                val = float(arr.mean())
            print(f"[NN(feature) {args.forecast_method}] H={h} n={arr.size} forecast_ret={val:.6e}")

if __name__ == "__main__":
    main()
