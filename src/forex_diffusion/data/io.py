"""
IO utilities for candle data (CSV/Parquet/DB), validation and causal resampling.

Functions:
- read_file(path) -> pd.DataFrame
- validate_candles_df(df, symbol=None, timeframe=None) -> (df_clean, report)
- resample_candles(df, src_tf, tgt_tf) -> df_resampled
- ensure_candles_table(engine) -> sqlalchemy.Table
- upsert_candles(engine, df, symbol, timeframe, resampled=False) -> qa_report
- get_last_ts_for_symbol_tf(engine, symbol, timeframe) -> Optional[int]
- backfill_from_provider(engine, provider_client, symbol, timeframe, start_ts_ms, end_ts_ms)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    create_engine,
    select,
    insert,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from ..utils.config import get_config

# Column names expected
COLUMNS = ["ts_utc", "open", "high", "low", "close", "volume"]

# Map timeframe labels to pandas offset aliases (used for resampling)
TF_TO_PANDAS = {
    "1m": "1T",
    "2m": "2T",
    "3m": "3T",
    "4m": "4T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "60m": "60T",
    "1h": "60T",
    "1d": "1D",
    "1D": "1D",
    "2h": "120T",
    "4h": "240T",
}


def read_file(path: str) -> pd.DataFrame:
    """
    Read CSV or Parquet into a DataFrame. Normalize column names to expected ones.
    Expects ts column in milliseconds UTC or as ISO datetimes (will be converted).
    """
    p = pd.Path(path) if hasattr(pd, "Path") else None
    path_lower = str(path).lower()
    if path_lower.endswith(".parquet") or path_lower.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # Normalize column names
    df_columns = {c: c.strip() for c in df.columns}
    df.rename(columns=df_columns, inplace=True)
    # Ensure ts_utc exists or infer from common names
    if "ts_utc" not in df.columns:
        for cand in ["ts", "timestamp", "time", "date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "ts_utc"})
                break
    return df


def _to_ms_int_series(series: pd.Series) -> pd.Series:
    """
    Convert a timestamp series to integer milliseconds (UTC).
    Accepts int(ms), int(s), or pandas datetime.
    """
    if pd.api.types.is_integer_dtype(series):
        # Heuristic: if values look like seconds (<= 1e10) convert to ms
        if series.max() < 10_000_000_000:
            return (series.astype("int64") * 1000).astype("int64")
        return series.astype("int64")
    # Try datetime conversion
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return (ts.view("int64") // 1_000_000).astype("Int64")


def validate_candles_df(
    df: pd.DataFrame, symbol: Optional[str] = None, timeframe: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate and sanitize a candles DataFrame.

    Returns (df_clean, report) where report contains:
    - n_rows
    - n_dups
    - n_out_of_order
    - n_invalid_price_relation
    - gaps: list of (prev_ts, next_ts, delta_ms)
    """
    report: Dict = {}
    df2 = df.copy()

    # Normalize columns
    for col in ["open", "high", "low", "close"]:
        if col not in df2.columns:
            raise ValueError(f"Missing required column: {col}")

    # ts_utc normalization
    if "ts_utc" not in df2.columns:
        raise ValueError("ts_utc column is required")
    df2["ts_utc"] = _to_ms_int_series(df2["ts_utc"])
    # Drop rows with NaT/NA ts
    before = len(df2)
    df2 = df2[df2["ts_utc"].notna()]
    after = len(df2)
    report["rows_dropped_invalid_ts"] = before - after

    # Sort by ts_utc
    df2 = df2.sort_values("ts_utc").reset_index(drop=True)

    # Check duplicates
    if symbol is not None and timeframe is not None:
        # duplicates identified by triple, but at this stage we only have ts
        dup_mask = df2.duplicated(subset=["ts_utc"], keep=False)
    else:
        dup_mask = df2.duplicated(subset=["ts_utc"], keep=False)
    n_dups = int(dup_mask.sum())
    report["n_dups"] = n_dups
    if n_dups > 0:
        # Keep first occurrence
        df2 = df2[~df2.duplicated(subset=["ts_utc"], keep="first")].reset_index(drop=True)

    # Price relationships: high >= max(open, close); low <= min(open, close); low <= high
    cond_high = df2["high"] >= df2[["open", "close"]].max(axis=1)
    cond_low = df2["low"] <= df2[["open", "close"]].min(axis=1)
    cond_lr = df2["low"] <= df2["high"]
    valid_prices = cond_high & cond_low & cond_lr
    n_invalid_price_relation = int((~valid_prices).sum())
    report["n_invalid_price_relation"] = n_invalid_price_relation
    if n_invalid_price_relation > 0:
        # Drop invalid rows (could alternatively attempt to fix)
        df2 = df2[valid_prices].reset_index(drop=True)

    # Gaps detection: if timeframe provided, compute expected delta
    gaps = []
    report["n_gaps_flagged"] = 0
    if timeframe and timeframe in TF_TO_PANDAS:
        # convert pandas offset to minutes
        if timeframe.endswith("m") or timeframe.endswith("T"):
            # e.g., '5m' -> 5
            try:
                expected_delta_ms = int(pd.to_timedelta(TF_TO_PANDAS[timeframe]).total_seconds() * 1000)
            except Exception:
                expected_delta_ms = None
        else:
            try:
                expected_delta_ms = int(pd.to_timedelta(TF_TO_PANDAS[timeframe]).total_seconds() * 1000)
            except Exception:
                expected_delta_ms = None
    else:
        expected_delta_ms = None

    if expected_delta_ms:
        ts = df2["ts_utc"].astype("int64").to_numpy()
        if len(ts) >= 2:
            deltas = ts[1:] - ts[:-1]
            idx = np.where(deltas > expected_delta_ms)[0]
            for i in idx:
                gaps.append({"prev_ts": int(ts[i]), "next_ts": int(ts[i + 1]), "delta_ms": int(deltas[i])})
        report["n_gaps_flagged"] = len(gaps)
        report["gaps"] = gaps

    report["n_rows"] = len(df2)
    report["symbol"] = symbol
    report["timeframe"] = timeframe
    return df2, report


def resample_candles(df: pd.DataFrame, src_tf: str, tgt_tf: str) -> pd.DataFrame:
    """
    Causal resampling from src_tf to tgt_tf.
    - open = first open in period
    - close = last close in period
    - high = max(high)
    - low = min(low)
    - volume = sum(volume)
    Input df must contain ts_utc in ms and OHLCV columns.
    """
    if src_tf == tgt_tf:
        return df.copy()

    if tgt_tf not in TF_TO_PANDAS:
        raise ValueError(f"Target timeframe {tgt_tf} not supported for resampling.")

    if "ts_utc" not in df.columns:
        raise ValueError("ts_utc column required for resampling")

    # Convert to datetime index in UTC
    tmp = df.copy()
    tmp["ts_dt"] = pd.to_datetime(tmp["ts_utc"].astype("int64"), unit="ms", utc=True)
    tmp = tmp.set_index("ts_dt")

    rule = TF_TO_PANDAS[tgt_tf]
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in tmp.columns:
        agg["volume"] = "sum"
    res = tmp.resample(rule, origin="start_day", label="right", closed="right").agg(agg)

    # Drop periods without data (NaN open/close)
    res = res.dropna(subset=["open", "close"]).reset_index()
    # ts_utc should be the timestamp of the period end (right label)
    res["ts_utc"] = (res["ts_dt"].view("int64") // 1_000_000).astype("int64")
    res = res[["ts_utc", "open", "high", "low", "close"] + (["volume"] if "volume" in res.columns else [])]
    return res


def _get_engine_from_url(database_url: Optional[str] = None) -> Engine:
    cfg = get_config()
    url = database_url or getattr(cfg.db, "database_url", None)
    if not url:
        raise ValueError("Database URL is not configured")
    # echo disabled by default; caller can enable
    engine = create_engine(url, future=True)
    return engine


def ensure_candles_table(engine: Engine) -> Table:
    """
    Ensure the candles table exists with the required schema.
    Primary key enforced on (symbol, timeframe, ts_utc) via unique index.
    """
    metadata = MetaData()
    candles = Table(
        "market_data_candles",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("symbol", String(64), nullable=False, index=True),
        Column("timeframe", String(16), nullable=False, index=True),
        Column("ts_utc", Integer, nullable=False, index=True),
        Column("open", Float, nullable=False),
        Column("high", Float, nullable=False),
        Column("low", Float, nullable=False),
        Column("close", Float, nullable=False),
        Column("volume", Float, nullable=True),
        Column("resampled", Boolean, default=False),
    )
    metadata.create_all(engine)
    # Create a unique composite index if not present (SQLite supports)
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_market_symbol_tf_ts ON market_data_candles(symbol, timeframe, ts_utc);"
            )
    except Exception:
        logger.debug("Could not create unique index (may already exist or unsupported).")
    return candles


def upsert_candles(engine: Engine, df: pd.DataFrame, symbol: str, timeframe: str, resampled: bool = False) -> Dict:
    """
    Upsert candles into the DB. Uses SQLite 'INSERT OR REPLACE' semantics where available.
    Returns a QA report including rows_inserted, rows_upserted, n_dups_resolved.
    """
    if df.empty:
        return {"rows_inserted": 0, "rows_upserted": 0, "note": "empty_dataframe"}

    # Validate
    dfv, vreport = validate_candles_df(df, symbol=symbol, timeframe=timeframe)
    logger.debug("Validation report pre-upsert: {}", vreport)

    # Prepare engine & table
    tbl = ensure_candles_table(engine)

    rows = []
    for _, r in dfv.iterrows():
        row = {
            "symbol": symbol,
            "timeframe": timeframe,
            "ts_utc": int(r["ts_utc"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r.get("volume", math.nan)) if "volume" in r else None,
            "resampled": bool(resampled),
        }
        rows.append(row)

    inserted = 0
    upserted = 0
    try:
        with engine.begin() as conn:
            # For SQLite, use OR REPLACE with prefix
            dialect_name = conn.dialect.name.lower()
            if dialect_name == "sqlite":
                stmt = insert(tbl).prefix_with("OR REPLACE")
                conn.execute(stmt, rows)
                inserted = len(rows)
                upserted = len(rows)
            else:
                # Generic upsert: try execute insert and ignore conflicts if supported
                stmt = insert(tbl)
                try:
                    conn.execute(stmt, rows)
                    inserted = len(rows)
                except SQLAlchemyError:
                    # Fallback: insert one-by-one with replacement
                    for r in rows:
                        stmt = insert(tbl).values(**r)
                        conn.execute(stmt)
                    inserted = len(rows)
                    upserted = len(rows)
    except Exception as exc:
        logger.exception("Failed to upsert candles: {}", exc)
        raise

    report = {
        "rows_inserted": inserted,
        "rows_upserted": upserted,
        "validation": vreport,
    }
    logger.info("Upserted {} rows for {}/{}", inserted, symbol, timeframe)
    return report


def get_last_ts_for_symbol_tf(engine: Engine, symbol: str, timeframe: str) -> Optional[int]:
    """
    Return last ts_utc for given symbol and timeframe, or None if no rows.
    """
    tbl = ensure_candles_table(engine)
    with engine.connect() as conn:
        stmt = select(tbl.c.ts_utc).where(tbl.c.symbol == symbol).where(tbl.c.timeframe == timeframe).order_by(
            tbl.c.ts_utc.desc()
        ).limit(1)
        r = conn.execute(stmt).first()
        if r:
            return int(r[0])
        return None


def backfill_from_provider(
    engine: Engine,
    provider_client,
    symbol: str,
    timeframe: str,
    start_ts_ms: int,
    end_ts_ms: int,
    resampled: bool = False,
    db_writer=None,
) -> Dict:
    """
    Download from provider_client (must implement get_historical(symbol, timeframe, start_ts_ms, end_ts_ms))
    The provider is expected to return a pandas.DataFrame with columns ts_utc, open, high, low, close, volume(optional).
    The function validates, resamples if needed, and upserts to DB, returning a QA report.

    Optionally compute features via pipeline and persist them (async via db_writer or sync via DBService).
    """
    cfg = get_config()
    logger.info("Backfilling {} {} from {} to {}", symbol, timeframe, start_ts_ms, end_ts_ms)
    df = provider_client.get_historical(symbol=symbol, timeframe=timeframe, start_ts_ms=start_ts_ms, end_ts_ms=end_ts_ms)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Provider client.get_historical must return a pandas.DataFrame")

    # Ensure columns and types
    dfv, vreport = validate_candles_df(df, symbol=symbol, timeframe=timeframe)

    # Upsert
    report = upsert_candles(engine, dfv, symbol, timeframe, resampled=resampled)
    combined = {"provider_rows": len(df), "validation": vreport, "upsert": report}

    # Optionally compute features and persist
    try:
        persist = False
        try:
            persist = bool(getattr(cfg, "persist_features", False) or (isinstance(cfg, dict) and cfg.get("persist_features", False)))
        except Exception:
            persist = False
        if persist:
            # compute features using pipeline
            from ..features.pipeline import pipeline_process
            # compute features for dfv (may require warmup trimming)
            features_df, _ = pipeline_process(dfv, timeframe=timeframe, features_config=getattr(cfg, "features", {}))
            # persist each row using db_writer if available, otherwise use sync DBService
            if db_writer is not None:
                for _, row in features_df.iterrows():
                    enq = db_writer.write_features_async(symbol=symbol, timeframe=timeframe, ts_utc=int(row.get("ts_utc", row.name)), features=row.to_dict(), pipeline_version=getattr(cfg.model, "pipeline_version", "v1") if hasattr(cfg, "model") else "v1")
                    if not enq:
                        logger.warning("backfill: DBWriter queue full while enqueuing features for {} {}", symbol, timeframe)
            else:
                # sync write via DBService
                from ..services.db_service import DBService
                dbs = DBService(engine=engine)
                for _, row in features_df.iterrows():
                    dbs.write_features(symbol=symbol, timeframe=timeframe, ts_utc=int(row.get("ts_utc", row.name)), features=row.to_dict(), pipeline_version=getattr(cfg.model, "pipeline_version", "v1") if hasattr(cfg, "model") else "v1")
            combined["features_persisted"] = True
    except Exception as e:
        logger.exception("Failed to compute/persist features for backfill: {}", e)
        combined["features_persisted"] = False
        combined["features_error"] = str(e)

    logger.info("Backfill report for {}/{}: {}", symbol, timeframe, combined)
    return combined
