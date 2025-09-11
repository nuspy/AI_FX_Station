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
from os import truncate
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
        # import pipeline core + specific indicator functions to recompute using identical implementations
        from forex_diffusion.features.pipeline import (
            pipeline_process,
            hurst_feature,
            hurst_aggvar,
            _rs_hurst,
            log_returns,
            atr,
            garman_klass_rolling,
            macd,
            rsi_wilder,
            bollinger,
            realized_volatility,
        )
        from forex_diffusion.services.regime_service import RegimeService, INDEX_PATH, MAPPING_PATH
        from sqlalchemy import MetaData, text, select
        import pandas as pd
        import numpy as np
        import math
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
                logger.info("Backfill result: %s", truncate(20, res))
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

    # Detect available indicator columns (helpers will match by substring)
    indicator_keys = [
        "r", "hl_range", "atr", "rv", "gk_vol", "ema_fast", "ema_slow", "ema_slope",
        "macd", "macd_signal", "macd_hist", "rsi", "bb_pctb", "kelt_upper", "kelt_lower",
        "don_upper", "don_lower", "realized_skew", "realized_kurt", "hurst", "vol_mean"
    ]
    found_indicators = {}
    for key in indicator_keys:
        cols = [c for c in feats.columns if key in c and c != "ts_utc"]
        if cols:
            found_indicators[key] = cols

    if found_indicators:
        logger.info("Detected indicator columns in features (summary):")
        for k, cols in found_indicators.items():
            logger.info("  %s -> %d columns (example: %s)", k, len(cols), ", ".join(cols[:3]))
    else:
        logger.warning("No expected technical indicator columns detected in features_df. Check pipeline configuration and multi_timeframes.")

    # Helper functions to recompute indicators from raw candles (causal, matching pipeline implementations)
    def recompute_basic_indicators(orig_df: pd.DataFrame, upto_ts: int) -> dict:
        """
        Recompute indicators using the exact pipeline functions to ensure parity:
        - uses log_returns, atr, realized_volatility, garman_klass_rolling, macd, rsi_wilder, bollinger, hurst_feature
        """
        # locate index for upto_ts (prefer exact match, fallback to last <=)
        try:
            pos = orig_df.index[orig_df["ts_utc"] == int(upto_ts)][0]
        except Exception:
            valid_idx = orig_df.index[orig_df["ts_utc"] <= int(upto_ts)]
            if len(valid_idx) == 0:
                raise ValueError(f"Cannot find ts_utc {upto_ts} in original DF")
            pos = valid_idx.max()

        window_df = orig_df.iloc[: pos + 1].copy().reset_index(drop=True)
        out = {}

        # use pipeline functions for exact parity
        try:
            # log return last
            lr_df = log_returns(window_df, col="close", out_col="r_temp")
            out["r"] = float(lr_df["r_temp"].iloc[-1]) if "r_temp" in lr_df.columns else float(0.0)
        except Exception:
            out["r"] = 0.0

        try:
            out["hl_range"] = float(window_df["high"].iloc[-1] - window_df["low"].iloc[-1])
        except Exception:
            out["hl_range"] = 0.0

        try:
            atr_df = atr(window_df, n=14, out_col="atr_temp")
            out["atr"] = float(atr_df["atr_temp"].iloc[-1])
        except Exception:
            out["atr"] = 0.0

        try:
            rv_df = realized_volatility(window_df, col="close", window=60, out_col="rv_temp")
            out["rv"] = float(rv_df["rv_temp"].iloc[-1])
        except Exception:
            out["rv"] = 0.0

        try:
            gk_df = garman_klass_rolling(window_df, window=20, out_col="gk_temp")
            out["gk_vol"] = float(gk_df["gk_temp"].iloc[-1])
        except Exception:
            out["gk_vol"] = 0.0

        try:
            mac = macd(window_df, fast=12, slow=26, signal=9)
            out["macd"] = float(mac["macd"].iloc[-1])
            out["macd_signal"] = float(mac["macd_signal"].iloc[-1])
            out["macd_hist"] = float(mac["macd_hist"].iloc[-1])
        except Exception:
            out["macd"] = out["macd_signal"] = out["macd_hist"] = 0.0

        try:
            out["rsi"] = float(rsi_wilder(window_df, n=14, out_col="rsi_temp")["rsi_temp"].iloc[-1])
        except Exception:
            out["rsi"] = 50.0

        try:
            bb = bollinger(window_df, n=20, k=2.0, out_prefix="bb_temp")
            out["bb_pctb"] = float(bb["bb_temp_pctb_20"].iloc[-1]) if "bb_temp_pctb_20" in bb.columns else float(0.5)
        except Exception:
            out["bb_pctb"] = 0.5

        try:
            hur = hurst_feature(window_df, window=64, out_col="hurst_temp")
            out["hurst"] = float(hur["hurst_temp"].iloc[-1]) if "hurst_temp" in hur.columns else float("nan")
        except Exception:
            out["hurst"] = float("nan")

        return out

    # Ensure features_df contains ts_utc; if missing, try to infer from original df using warmup offset
    if "ts_utc" not in feats.columns:
        try:
            warmup = int(features_config.get("warmup_bars", 0))
        except Exception:
            warmup = 0
        try:
            df_ts = df.reset_index(drop=True)["ts_utc"]
            # Align df_ts with feats rows: take slice starting at warmup
            aligned = df_ts.iloc[warmup : warmup + len(feats)].reset_index(drop=True)
            if len(aligned) == len(feats):
                feats = feats.reset_index(drop=True)
                feats["ts_utc"] = aligned
                logger.info("Inferred ts_utc for features_df from raw candles using warmup=%d", warmup)
            else:
                logger.warning("Could not infer ts_utc for features: aligned length %d != feats length %d; skipping indicator validation", len(aligned), len(feats))
                # mark that ts_utc inference failed
                feats["ts_utc"] = [None] * len(feats)
        except Exception as e:
            logger.warning("Failed to infer ts_utc into features_df: %s", e)
            feats["ts_utc"] = [None] * len(feats)

    # choose sample timestamps from feats (most recent 5 rows or fewer)
    sample_n = min(5, len(feats))
    sample_rows = feats.tail(sample_n).reset_index(drop=True)
    logger.info("Validating indicator values for %d sample rows", sample_n)

    tolerance_abs = 1e-6
    tolerance_rel = 1e-3  # 0.1% relative tolerance allowed

    for i, row in sample_rows.iterrows():
        ts_val = row.get("ts_utc", None)
        if ts_val is None or (isinstance(ts_val, float) and pd.isna(ts_val)):
            logger.warning(f"Skipping sample index {i} because ts_utc is missing")
            continue
        try:
            ts = int(ts_val)
        except Exception:
            logger.warning(f"Skipping sample index {i} because ts_utc value invalid: {ts_val}")
            continue

        try:
            recomputed = recompute_basic_indicators(df, ts)
        except Exception as e:
            logger.warning(f"Could not recompute indicators for ts={ts}: {e}")
            continue

        # For each indicator present in features_df, compare (prefer timeframe-prefixed columns and standardize recomputed values)
        comparisons = []
        for key, recom_val in recomputed.items():
            # Build prioritized candidate lists:
            tf_pref = f"{args.timeframe}_{key}"
            exact = [c for c in feats.columns if c == key and c != "ts_utc"]
            pref_tf = [c for c in feats.columns if c == tf_pref and c != "ts_utc"]
            suf = [c for c in feats.columns if c.endswith("_" + key) and c != "ts_utc"]
            contains = [c for c in feats.columns if (key in c) and c != "ts_utc"]

            col_candidates = pref_tf + exact + suf + contains
            if not col_candidates:
                continue
            col = col_candidates[0]  # choose highest priority candidate
            feat_val = row.get(col, None)
            if feat_val is None:
                continue
            try:
                fv = float(feat_val)  # feature value (already standardized by pipeline)
            except Exception:
                continue

            # recomputed raw value
            rv_raw = float(recom_val) if recom_val == recom_val else float("nan")

            # If standardizer is available, standardize recomputed raw value using std.mu/std.sigma for that column
            rv_std = None
            try:
                if 'std' in locals() and std is not None and getattr(std, "mu", None) is not None:
                    mu_map = getattr(std, "mu", {}) or {}
                    sigma_map = getattr(std, "sigma", {}) or {}
                    if col in mu_map and col in sigma_map:
                        mu = float(mu_map[col])
                        sigma = float(sigma_map[col]) if float(sigma_map[col]) != 0.0 else 1.0
                        rv_std = (rv_raw - mu) / sigma if rv_raw == rv_raw else float("nan")
                    else:
                        # try to find a mu/sigma by suffix match (e.g., '1m_r' vs 'r')
                        if key in mu_map and key in sigma_map:
                            mu = float(mu_map[key]); sigma = float(sigma_map[key]) if float(sigma_map[key]) != 0.0 else 1.0
                            rv_std = (rv_raw - mu) / sigma if rv_raw == rv_raw else float("nan")
                else:
                    rv_std = None
            except Exception:
                rv_std = None

            # decide value to compare: if rv_std available, compare fv to rv_std, else compare fv to raw rv_raw
            if rv_std is not None and rv_std == rv_std:
                compare_rhs = rv_std
                compare_note = "std"
            else:
                compare_rhs = rv_raw
                compare_note = "raw"

            abs_diff = abs(fv - compare_rhs) if compare_rhs == compare_rhs else None
            rel_diff = abs_diff / (abs(compare_rhs) + 1e-12) if abs_diff is not None else None
            ok = False
            if abs_diff is None or (abs_diff != abs_diff):
                ok = False
            elif (abs_diff is not None and abs_diff <= tolerance_abs) or (rel_diff is not None and rel_diff <= tolerance_rel):
                ok = True

            comparisons.append((key, col, fv, rv_raw, rv_std, compare_note, abs_diff, rel_diff, ok))

        # log results and details; include standardizer mu/sigma and reconstructed raw feat for mismatches
        for comp in comparisons:
            key, col, fv, rv_raw, rv_std, note, ad, rd, ok = comp
            if ok:
                logger.info(f"PASS ts={ts} {key} ({col}): feat_std={fv} recomputed_raw={rv_raw} recomputed_{note}={rv_std} abs_diff={ad} rel_diff={rd}")
            else:
                # attempt to fetch mu/sigma for this column to reconstruct pipeline raw feature
                mu = None
                sigma = None
                feat_raw = None
                try:
                    if 'std' in locals() and std is not None and getattr(std, "mu", None) is not None:
                        mu_map = getattr(std, "mu", {}) or {}
                        sigma_map = getattr(std, "sigma", {}) or {}
                        if col in mu_map and col in sigma_map:
                            mu = float(mu_map[col]); sigma = float(sigma_map[col]) if float(sigma_map[col]) != 0.0 else 1.0
                        elif key in mu_map and key in sigma_map:
                            mu = float(mu_map[key]); sigma = float(sigma_map[key]) if float(sigma_map[key]) != 0.0 else 1.0
                    if mu is not None and sigma is not None:
                        feat_raw = fv * sigma + mu
                except Exception:
                    mu = sigma = feat_raw = None

                if mu is not None:
                    logger.warning(f"MISMATCH ts={ts} {key} ({col}): feat_std={fv} feat_raw_recon={feat_raw} mu={mu} sigma={sigma} recomputed_raw={rv_raw} recomputed_{note}={rv_std} abs_diff={ad} rel_diff={rd}")
                else:
                    logger.warning(f"MISMATCH ts={ts} {key} ({col}): feat_std={fv} recomputed_raw={rv_raw} recomputed_{note}={rv_std} abs_diff={ad} rel_diff={rd} (no mu/sigma available)")

                # Extra diagnostics for Hurst mismatches: show multiple estimators and the series details
                try:
                    if key == "hurst":
                        # locate position in original df for ts
                        valid_idx = df.index[df["ts_utc"] <= int(ts)]
                        if len(valid_idx) == 0:
                            logger.warning(f"HURST DIAG ts={ts}: no original rows <= ts")
                        else:
                            pos = valid_idx.max()
                            window_df = df.iloc[max(0, pos - 1024 + 1): pos + 1].reset_index(drop=True)  # up to 1024 sample
                            # compute returns series used
                            rseries = np.log(window_df["close"]).diff().dropna()
                            n = len(rseries)
                            logger.info(f"HURST DIAG ts={ts}: series_len={n}, ts_pos={pos}, last_ts={window_df['ts_utc'].iat[-1] if len(window_df)>0 else 'NA'}")
                            # aggregated variance estimator on multiple chunk sizes if possible
                            try:
                                h_agg = hurst_aggvar(rseries) if n >= 32 else float("nan")
                            except Exception as e:
                                h_agg = float("nan")
                                logger.debug("HURST DIAG hurst_aggvar failed: %s", e)
                            # rescaled range on last min-window
                            try:
                                rs_h = _rs_hurst(rseries[-min(n, 256):]) if n >= 20 else float("nan")
                            except Exception as e:
                                rs_h = float("nan")
                                logger.debug("HURST DIAG _rs_hurst failed: %s", e)
                            logger.info(f"HURST DIAG ts={ts}: hurst_aggvar={h_agg} rs_hurst={rs_h} recomputed_raw={rv_raw} recomputed_std={rv_std} feat_raw_recon={feat_raw}")
                except Exception as e:
                    logger.debug("HURST diagnostic failed: %s", e)

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
