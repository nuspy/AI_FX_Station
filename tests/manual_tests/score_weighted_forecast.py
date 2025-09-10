#!/usr/bin/env python3
"""
tests/manual_tests/score_weighted_forecast.py

Load a saved weighted_forecast pickle (model + features + std_mu/std_sigma + encoder),
apply standardization and encoder (if required) and output predictions for an input CSV
or by pulling features from DB for a given symbol/timeframe.

Usage:
  # predict from CSV of features (columns must match saved feature names)
  python tests/manual_tests/score_weighted_forecast.py --model artifacts/models/weighted_forecast_EURUSD_1m_h5_ridge_none.pkl --input_csv tmp/features_to_score.csv --out_csv tmp/preds.csv

  # or predict latest features for symbol/timeframe (will compute pipeline features and align)
  python tests/manual_tests/score_weighted_forecast.py --model ... --symbol "EUR/USD" --timeframe "1m" --use_pipeline

Outputs:
  - CSV with predictions (ts_utc if available + pred)
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import pickle
import json

ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to pickle model saved by weighted_forecast")
    p.add_argument("--input_csv", default=None, help="Optional CSV with features (columns must match model features)")
    p.add_argument("--out_csv", default="tmp/preds.csv")
    p.add_argument("--symbol", default=None, help="If input_csv not provided, can load features from DB for symbol/timeframe using pipeline")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--use_pipeline", action="store_true", help="When symbol provided, compute features via pipeline")
    args = p.parse_args()

    try:
        import pandas as pd
        import numpy as np
        from forex_diffusion.services.db_service import DBService
        from forex_diffusion.features.pipeline import pipeline_process
    except Exception as e:
        print("Imports failed:", e); sys.exit(1)

    # load model payload
    mp = Path(args.model)
    if not mp.exists():
        print("Model file not found:", mp); sys.exit(2)
    payload = pickle.loads(mp.read_bytes())
    model = payload.get("model")
    features = payload.get("features") or []
    std_mu = payload.get("std_mu") or {}
    std_sigma = payload.get("std_sigma") or {}
    encoder_info = payload.get("encoder")

    # prepare feature dataframe
    if args.input_csv:
        df_in = pd.read_csv(args.input_csv)
        # ensure order of columns according to saved features
        missing = [f for f in features if f not in df_in.columns]
        if missing:
            print("Input CSV missing features:", missing); sys.exit(3)
        X = df_in[features].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ts = df_in["ts_utc"] if "ts_utc" in df_in.columns else None
    else:
        if not args.symbol or not args.use_pipeline:
            print("Either --input_csv or (--symbol and --use_pipeline) must be provided"); sys.exit(4)
        db = DBService()
        eng = db.engine
        # load recent candles, compute features via pipeline_process using default features_config
        with eng.connect() as conn:
            # simple limit
            q = "SELECT ts_utc, open, high, low, close" + (", volume" if True else "") + " FROM market_data_candles WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc ASC"
            rows = conn.execute(q, {"s": args.symbol, "tf": args.timeframe}).fetchall()
            if not rows:
                print("No candles found for", args.symbol, args.timeframe); sys.exit(5)
            df_c = pd.DataFrame([dict(r._mapping) if hasattr(r, "_mapping") else {"ts_utc": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4]} for r in rows])
        feats, _std = pipeline_process(df_c, timeframe=args.timeframe, features_config={"warmup_bars": 16, "indicators": {}})
        if "ts_utc" not in feats.columns:
            feats = feats.reset_index(drop=True)
        # keep only saved features
        missing = [f for f in features if f not in feats.columns]
        if missing:
            print("Pipeline did not produce features required by model (missing):", missing); sys.exit(6)
        X = feats[features].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ts = feats.get("ts_utc", None)

    # standardize using saved mu/sigma (apply feature-wise)
    for col in features:
        mu = std_mu.get(col, None) if isinstance(std_mu, dict) else None
        sigma = std_sigma.get(col, None) if isinstance(std_sigma, dict) else None
        if mu is not None and sigma is not None:
            try:
                X[col] = (X[col].astype(float).fillna(0.0) - float(mu)) / (float(sigma) if float(sigma) != 0.0 else 1.0)
            except Exception:
                X[col] = X[col].astype(float).fillna(0.0)

    # handle encoder if PCA or latents - for simplicity if encoder is PCA we assume X already matches features used for PCA (not stored here)
    # Predict
    preds = model.predict(X.to_numpy(dtype=float))

    out_df = pd.DataFrame({"pred": preds})
    if ts is not None:
        out_df.insert(0, "ts_utc", ts.reset_index(drop=True))
    out_df.to_csv(args.out_csv, index=False)
    print("Saved predictions to", args.out_csv)

if __name__ == "__main__":
    main()
