#!/usr/bin/env python3
"""
tests/manual_tests/verify_ml_workflow.py

Robust verification helper to confirm that ML clustering + ANN indexing is executed on stored latents.

Behavior:
 - checks for existence and count of latents in DB
 - if none found, prints actionable steps to generate latents from candles (pipeline/backfill)
 - if latents found, runs fit_clusters_and_index and verifies index/mapping and labels

Usage:
  python tests/manual_tests/verify_ml_workflow.py --n_clusters 8 --limit 5000
"""
from __future__ import annotations

import sys
from pathlib import Path
import time
import argparse
import json

# ensure project src is importable
ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of latents to use for fit (None = all)")
    args = parser.parse_args()

    try:
        from forex_diffusion.services.regime_service import RegimeService, INDEX_PATH, MAPPING_PATH
        from forex_diffusion.services.db_service import DBService
    except Exception as e:
        print("Failed to import RegimeService/DBService from project:", e)
        raise SystemExit(1)

    # Create DBService first and verify latents table before creating RegimeService
    try:
        dbs = DBService()
        engine = dbs.engine
        from sqlalchemy import MetaData, select
        meta = MetaData()
        # reflect both latents and market_data_candles (market data used to build latents if missing)
        meta.reflect(bind=engine, only=["latents", "market_data_candles"])
        lat_tbl = meta.tables.get("latents")
        mkt_tbl = meta.tables.get("market_data_candles")

        # If latents table missing, attempt to create via pipeline on recent candles (best-effort)
        if lat_tbl is None or mkt_tbl is None:
            print("Table 'latents' or 'market_data_candles' not found in DB.")
            print("Cannot auto-generate latents without market_data_candles. Please backfill candles first.")
            raise SystemExit(2)

        # count latents
        with engine.connect() as conn:
            cnt = conn.execute(select(lat_tbl.c.id)).fetchall()
            total = len(cnt)
        print(f"Latents table found. Rows (latents) available: {total}")

        # If no latents, try to generate them from the latest candles for a default symbol/timeframe
        if total == 0:
            print("No latent vectors found. Attempting to generate latents from historical candles (EUR/USD 1m) ...")
            try:
                # fetch recent candles for EUR/USD 1m (limit to reasonable window)
                with engine.connect() as conn:
                    stmt = select(mkt_tbl.c.ts_utc, mkt_tbl.c.open, mkt_tbl.c.high, mkt_tbl.c.low, mkt_tbl.c.close, mkt_tbl.c.volume).where(
                        (mkt_tbl.c.symbol == "EUR/USD") & (mkt_tbl.c.timeframe == "1m")
                    ).order_by(mkt_tbl.c.ts_utc.asc())
                    rows = conn.execute(stmt).fetchall()
                if not rows:
                    print("No market_data_candles found for EUR/USD 1m; cannot generate latents automatically.")
                    raise SystemExit(3)
                # build DataFrame
                import pandas as _pd
                df = _pd.DataFrame([{
                    "ts_utc": int(r[0]), "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "volume": r[5]
                } for r in rows])
                # compute features via pipeline_process (catch import/syntax errors and show traceback)
                try:
                    from forex_diffusion.features.pipeline import pipeline_process
                except Exception as e:
                    import traceback, inspect
                    print("Failed to import pipeline_process from forex_diffusion.features.pipeline:")
                    traceback.print_exc()
                    # show file snippet if available
                    try:
                        module_path = None
                        import importlib.util, forex_diffusion.features.pipeline as _mod
                        module_path = inspect.getsourcefile(_mod) or inspect.getfile(_mod)
                    except Exception:
                        module_path = None
                    if module_path:
                        print()
                        print("Check file for syntax errors:", module_path)
                    print()
                    print("Please fix the pipeline module (syntax error) before auto-generating latents.")
                    raise SystemExit(3)
                try:
                    # First attempt: default pipeline settings
                    features_df, std = pipeline_process(df, timeframe="1m")
                except Exception as e:
                    print("Feature pipeline execution failed:", e)
                    import traceback
                    traceback.print_exc()
                    raise SystemExit(3)

                # If pipeline returned empty due to warmup, retry with a small warmup (use only existing data)
                if features_df is None or features_df.empty:
                    print("Feature pipeline returned no rows with default warmup; trying fallback with reduced warmup...")
                    try:
                        features_cfg_fallback = {"warmup_bars": 16, "standardization": {"window_bars": 20}, "indicators": {"atr": {"n": 7}}}
                        features_df, std = pipeline_process(df, timeframe="1m", features_config=features_cfg_fallback)
                        if features_df is None or features_df.empty:
                            print("Fallback pipeline also returned no rows; cannot create latents from available candles.")
                            raise SystemExit(3)
                        else:
                            print(f"Fallback pipeline produced {len(features_df)} feature rows (warmup reduced).")
                    except Exception as e:
                        print("Fallback pipeline execution failed:", e)
                        import traceback
                        traceback.print_exc()
                        raise SystemExit(3)
                # prepare latents rows (latent_json as list of feature floats) - robust to missing ts_utc
                lat_rows = []
                now_ms = int(time.time() * 1000)
                feat_cols = [c for c in features_df.columns if c != "ts_utc"]
                print(f"DEBUG: features_df columns = {list(features_df.columns)}; using feat_cols={feat_cols}")
                for idx, fr in features_df.iterrows():
                    # build vector (replace NaN with 0.0)
                    vec = []
                    for c in feat_cols:
                        try:
                            v = fr[c]
                            # check NaN
                            if v != v:
                                v = 0.0
                        except Exception:
                            v = 0.0
                        try:
                            vec.append(float(v))
                        except Exception:
                            vec.append(0.0)
                    # determine ts_utc: prefer explicit column, fallback to index-based or current time
                    ts_val = None
                    try:
                        if "ts_utc" in features_df.columns:
                            ts_val = fr.get("ts_utc", None)
                    except Exception:
                        ts_val = None
                    if ts_val is None or (isinstance(ts_val, float) and (ts_val != ts_val)):
                        # fallback: try to infer from original df if available and aligned by position
                        try:
                            # features_df was derived from df; use original df index mapping if possible
                            orig_idx = idx
                            if isinstance(orig_idx, int) and orig_idx < len(df):
                                ts_val = int(df.iloc[orig_idx]["ts_utc"])
                        except Exception:
                            ts_val = None
                    if ts_val is None or (isinstance(ts_val, float) and (ts_val != ts_val)):
                        # last resort fallback: use current timestamp (will still allow insertion)
                        ts_ms = now_ms
                        print(f"WARNING: missing ts_utc for feature-row idx={idx}; using fallback ts={ts_ms}")
                    else:
                        try:
                            ts_ms = int(ts_val)
                        except Exception:
                            ts_ms = now_ms
                            print(f"WARNING: invalid ts_utc value for idx={idx}: {ts_val}; using fallback ts={ts_ms}")

                    row = {
                        "symbol": "EUR/USD",
                        "timeframe": "1m",
                        "ts_utc": ts_ms,
                        "model_version": None,
                        "latent_json": json.dumps(vec),
                        "ts_created_ms": now_ms,
                    }
                    lat_rows.append(row)
                # bulk insert into latents table (if any)
                try:
                    if lat_rows:
                        with engine.begin() as conn:
                            conn.execute(lat_tbl.insert(), lat_rows)
                        print(f"Inserted {len(lat_rows)} latents into 'latents' table.")
                    else:
                        print("No latent rows prepared; nothing to insert.")
                except Exception as e:
                    print("Failed to insert latents into DB:", e)
                    raise
                # refresh count
                with engine.connect() as conn:
                    cnt2 = conn.execute(select(lat_tbl.c.id)).fetchall()
                    total = len(cnt2)
                print(f"Latents now available: {total}")
                if total == 0:
                    print("Auto-generation failed to create latents.")
                    raise SystemExit(4)
            except Exception as e:
                print("Auto-generation of latents failed:", e)
                raise SystemExit(4)
    except Exception as e:
        print("Failed to inspect latents table:", e)
        raise SystemExit(4)

    # Now safe to create RegimeService using the same engine
    try:
        svc = RegimeService(engine=engine)
    except Exception as e:
        print("Failed to instantiate RegimeService with existing engine:", e)
        raise SystemExit(5)

    # Run clustering + index build (synchronous)
    print(f"Starting clustering + ANN index build with n_clusters={args.n_clusters} ...")
    t0 = time.time()
    try:
        svc.fit_clusters_and_index(n_clusters=args.n_clusters, limit=args.limit)
    except Exception as e:
        print("fit_clusters_and_index failed:", e)
        raise SystemExit(5)
    dt = time.time() - t0
    print(f"Clustering and index build completed in {dt:.1f}s")

    # Verify index and mapping files
    idx_path = Path(INDEX_PATH)
    map_path = Path(MAPPING_PATH)
    print("Index file exists:", idx_path.exists(), "path=", INDEX_PATH)
    print("Mapping file exists:", map_path.exists(), "path=", MAPPING_PATH)
    if map_path.exists():
        try:
            meta = json.loads(map_path.read_text(encoding="utf-8"))
            last_indexed = meta.get("last_indexed_id", None)
            print("Mapping last_indexed_id:", last_indexed)
            print("Mapping sizes: id_to_idx=", len(meta.get("id_to_idx", {})), " idx_to_id=", len(meta.get("idx_to_id", {})))
        except Exception as e:
            print("Failed to read mapping file:", e)

    # Query DB to count latents with regime_label set
    try:
        with engine.connect() as conn:
            stmt = select(tbl.c.id, tbl.c.regime_label).order_by(tbl.c.ts_utc.desc()).limit(10)
            rows = conn.execute(stmt).fetchall()
            # fetch all labels and count
            stmt_all = select(tbl.c.regime_label)
            all_rows = conn.execute(stmt_all).fetchall()
            total_labeled = sum(1 for r in all_rows if r[0])
            print(f"Latents labeled with regime_label: {total_labeled} / {len(all_rows)}")
            print("Sample recent latents and their regime_label (up to 10):")
            for r in rows:
                print("  id=", r[0], "regime_label=", r[1])
    except Exception as e:
        print("Failed to query latents/regime labels:", e)

    # Index metrics
    try:
        metrics = svc.get_index_metrics()
        print("Index metrics:", json.dumps(metrics, indent=2))
    except Exception as e:
        print("Failed to get index metrics:", e)

    print("Verification completed.")

if __name__ == "__main__":
    main()
