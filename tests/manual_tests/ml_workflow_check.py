#!/usr/bin/env python3
"""
tests/manual_tests/ml_workflow_check.py

End-to-end check script for ML workflow:
 - verify ingestion (market_data_candles)
 - optional backfill via MarketDataService
 - run pipeline_process to compute features
 - persist latents (latent_json) into latents table
 - run RegimeService.fit_clusters_and_index and report index metrics and files

Usage (PowerShell):
  # with .venv activated
  .\\scripts\\ml_workflow_check.ps1 -Symbol "EUR/USD" -Timeframe "1m" -DaysBackfill 3 -NClusters 8

This script is conservative: uses warmup reduced for testing and logs progress.
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import time
import json
import logging

# ensure project src is importable
ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Use loguru if available, else fallback to logging
try:
    from loguru import logger
except Exception:
    import logging as _logging
    logger = _logging.getLogger("ml_workflow_check")
    logger.setLevel(_logging.INFO)
    ch = _logging.StreamHandler()
    ch.setLevel(_logging.INFO)
    logger.addHandler = lambda h: None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="EUR/USD")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--days_backfill", type=int, default=3)
    p.add_argument("--n_clusters", type=int, default=8)
    p.add_argument("--max_features", type=int, default=2000, help="Max feature rows to process for test")
    args = p.parse_args()

    logger.info("ML workflow check started for %s %s", args.symbol, args.timeframe)

    # Imports that require project on sys.path
    try:
        from forex_diffusion.services.db_service import DBService
        from forex_diffusion.services.marketdata import MarketDataService
        from forex_diffusion.features.pipeline import pipeline_process
        from forex_diffusion.services.regime_service import RegimeService, INDEX_PATH, MAPPING_PATH
        from sqlalchemy import MetaData, text, select
        import pandas as pd
    except Exception as e:
        logger.exception("Failed to import project modules: %s", e)
        raise SystemExit(1)

    # instantiate services
    dbs = DBService()
    ms = None
    try:
        ms = MarketDataService()
    except Exception:
        logger.warning("MarketDataService not available")

    engine = dbs.engine

    # 1) ingestion check: count candles for symbol/timeframe
    try:
        meta = MetaData()
        meta.reflect(bind=engine, only=["market_data_candles", "latents"])
        mkt_tbl = meta.tables.get("market_data_candles")
        lat_tbl = meta.tables.get("latents")
    except Exception as e:
        logger.exception("Failed to reflect tables: %s", e)
        raise SystemExit(2)

    if mkt_tbl is None:
        logger.error("market_data_candles table not found in DB; abort")
        raise SystemExit(3)

    with engine.connect() as conn:
        cnt_q = text("SELECT COUNT(*) FROM market_data_candles WHERE symbol = :sym AND timeframe = :tf")
        cnt = conn.execute(cnt_q, {"sym": args.symbol, "tf": args.timeframe}).scalar() or 0
    logger.info("market_data_candles rows for %s %s = %d", args.symbol, args.timeframe, cnt)

    # 2) backfill if insufficient (heuristic: need at least days_backfill * 24 * 60 rows for 1m)
    required = args.days_backfill * 24 * 60 if args.timeframe == "1m" else args.days_backfill * 24
    if cnt < min(100, required):  # be conservative for test
        if ms is None:
            logger.warning("MarketDataService not available; cannot backfill automatically")
        else:
            logger.info("Triggering backfill (days=%d) via MarketDataService", args.days_backfill)
            try:
                res = ms.backfill_symbol_timeframe(args.symbol, args.timeframe, force_full=False)
                logger.info("Backfill result: %s", res)
                time.sleep(1.0)
                with engine.connect() as conn:
                    cnt = conn.execute(cnt_q, {"sym": args.symbol, "tf": args.timeframe}).scalar() or 0
                logger.info("Post-backfill market_data_candles rows = %d", cnt)
            except Exception as e:
                logger.exception("Backfill failed: %s", e)

    # 3) load candles into DataFrame (limit to max_features)
    with engine.connect() as conn:
        q = text("SELECT ts_utc, open, high, low, close" + (", volume" if "volume" in mkt_tbl.c else "") + " FROM market_data_candles WHERE symbol=:sym AND timeframe=:tf ORDER BY ts_utc ASC LIMIT :lim")
        rows = conn.execute(q, {"sym": args.symbol, "tf": args.timeframe, "lim": args.max_features}).fetchall()
        if not rows:
            logger.error("No candles returned for %s %s; abort", args.symbol, args.timeframe)
            raise SystemExit(4)
        df = pd.DataFrame([dict(r._mapping) if hasattr(r, "_mapping") else {"ts_utc": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5] if len(r)>5 else None} for r in rows])
    logger.info("Loaded %d candle rows into DataFrame", len(df))

    # 4) compute features using pipeline_process (reduced warmup for test)
    features_config = {"warmup_bars": 16, "indicators": {}}
    try:
        feats, std = pipeline_process(df.copy(), timeframe=args.timeframe, features_config=features_config)
    except Exception as e:
        logger.exception("pipeline_process failed: %s", e)
        raise SystemExit(5)
    logger.info("Computed features_df with %d rows and %d columns", len(feats), len(feats.columns))

    # Verify presence of key technical indicators in features_df (multi-timeframe columns handled by prefix)
    indicator_keys = [
        "r", "hl_range", "atr", "rv", "gk_vol", "ema_fast", "ema_slow", "ema_slope",
        "macd", "macd_signal", "macd_hist", "rsi", "bb_pctb", "kelt_upper", "kelt_lower",
        "don_upper", "don_lower", "realized_skew", "realized_kurt", "hurst", "vol_mean"
    ]
    found_indicators = {}
    for key in indicator_keys:
        cols = [c for c in feats.columns if key in c and c != "ts_utc"]
        if cols:
            found_indicators[key] = cols[:5]  # sample up to 5 columns per indicator
    if found_indicators:
        logger.info("Detected indicator columns in features (sample):")
        for k, cols in found_indicators.items():
            logger.info("  %s -> %s", k, ", ".join(cols))
    else:
        logger.warning("No expected technical indicator columns detected in features_df. Check pipeline configuration and multi_timeframes.")

    # persist a sample of features for inspection
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    feats_path = tmp_dir / "features_sample.csv"
    feats.head(200).to_csv(feats_path, index=False)
    logger.info("Saved features sample to %s", feats_path)

    # 5) persist latents derived from features (simple vectorization: use numeric features as latent)
    if lat_tbl is None:
        logger.error("latents table not found; cannot persist latents")
    else:
        # choose numeric columns except ts_utc
        numeric_cols = [c for c in feats.columns if c != "ts_utc" and pd.api.types.is_numeric_dtype(feats[c])]
        logger.info("Selected %d numeric feature columns for latent vectors (example): %s", len(numeric_cols), numeric_cols[:10])
        lat_rows = []
        now_ms = int(time.time() * 1000)
        for _, r in feats.iterrows():
            vec = []
            for c in numeric_cols:
                v = r.get(c, 0.0)
                try:
                    if pd.isna(v):
                        v = 0.0
                except Exception:
                    pass
                try:
                    vec.append(float(v))
                except Exception:
                    vec.append(0.0)
            ts = int(r.get("ts_utc") or now_ms)
            lat_rows.append({
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "ts_utc": int(ts),
                "model_version": "test_v1",
                "latent_json": json.dumps(vec),
                "ts_created_ms": now_ms,
            })
        logger.info("Prepared %d latent rows to insert", len(lat_rows))
        try:
            with engine.begin() as conn:
                # bulk insert
                conn.execute(lat_tbl.insert(), lat_rows)
            logger.info("Inserted %d latents into DB", len(lat_rows))
        except Exception as e:
            logger.exception("Failed to insert latents: %s", e)
            raise SystemExit(6)

        # Verify latent vectors correspond to numeric feature columns
        numeric_cols = [c for c in feats.columns if c != "ts_utc" and pd.api.types.is_numeric_dtype(feats[c])]
        expected_len = len(numeric_cols)
        logger.info("Expect latent vector length == number of numeric feature columns: %d", expected_len)
        try:
            with engine.connect() as conn:
                sample_row = conn.execute(text("SELECT latent_json FROM latents WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc DESC LIMIT 1"), {"s": args.symbol, "tf": args.timeframe}).fetchone()
                if sample_row is not None:
                    # sample_row may be Row or tuple
                    try:
                        lj = sample_row._mapping["latent_json"]
                    except Exception:
                        lj = sample_row[0] if len(sample_row) > 0 else None
                    if lj:
                        vec = json.loads(lj)
                        logger.info("Sample latent vector length: %d", len(vec))
                        if len(vec) != expected_len:
                            logger.warning("Latent vector length (%d) != expected numeric feature count (%d). Check which features were included.", len(vec), expected_len)
                        else:
                            logger.info("Latent vector length matches expected feature count.")
                    else:
                        logger.warning("Sample latent row has no latent_json content.")
                else:
                    logger.warning("No latents found to sample for verification.")
        except Exception as e:
            logger.exception("Failed to verify latent vector lengths: %s", e)

    # 6) run clustering/index build using RegimeService
    try:
        svc = RegimeService(engine=engine)
        logger.info("Starting fit_clusters_and_index with n_clusters=%d", args.n_clusters)
        svc.fit_clusters_and_index(n_clusters=args.n_clusters, limit=None)
        logger.info("fit_clusters_and_index completed")
    except Exception as e:
        logger.exception("RegimeService.fit_clusters_and_index failed: %s", e)
        raise SystemExit(7)

    # 7) verify index metrics and mapping file presence
    try:
        metrics = svc.get_index_metrics()
        logger.info("Index metrics: %s", metrics)
    except Exception as e:
        logger.exception("Failed to get index metrics: %s", e)

    # 8) final checks: count latents and check regimes presence
    try:
        meta = MetaData()
        meta.reflect(bind=engine, only=["latents"])
        lt = meta.tables.get("latents")
        with engine.connect() as conn:
            total_lat = conn.execute(text("SELECT COUNT(*) FROM latents WHERE symbol=:s AND timeframe=:tf"), {"s": args.symbol, "tf": args.timeframe}).scalar() or 0
            logger.info("Total latents for %s %s: %d", args.symbol, args.timeframe, total_lat)
            # sample latents
            sample = conn.execute(text("SELECT id, ts_utc, latent_json FROM latents WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc DESC LIMIT 5"), {"s": args.symbol, "tf": args.timeframe}).fetchall()
            logger.info("Sample latents (most recent 5):")
            for row in sample:
                logger.info(" id=%s ts=%s vec_len=%s", row[0], row[1], len(json.loads(row[2])) if row[2] else 0)
    except Exception as e:
        logger.exception("Final latents check failed: %s", e)

    logger.info("ML workflow check completed successfully.")

if __name__ == "__main__":
    main()
