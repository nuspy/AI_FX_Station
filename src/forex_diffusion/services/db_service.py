"""
DBService: simple persistence helpers for predictions, calibration records and signals.

 - Ensures tables: predictions, calibration_records (if not present), signals
 - Provides write_prediction, write_calibration_record, write_signal
 - Ensures all required tables exist on initialization.
 - Provides methods to write predictions, signals, ticks, etc.
 """

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine

from ..utils.config import get_config


class DBService:
    """
    DBService: simple persistence helpers for various data models.
    - Ensures all required tables exist on initialization.
    - Provides methods to write predictions, signals, ticks, etc.
    """
    def __init__(self, engine: Optional[Engine] = None):
        cfg = get_config()
        db_url = getattr(cfg.db, "database_url", None) or (
            cfg.db.get("database_url") if isinstance(cfg.db, dict) else None
        )
        if engine is None:
            if not db_url:
                raise ValueError("Database URL not configured")
            self.engine = create_engine(db_url, future=True)
        else:
            self.engine = engine
        self._ensure_tables()
        try:
            self._ensure_latents_columns()
        except Exception as _e:
            logger.debug("Could not ensure latents additional columns: {}", _e)

    # fmt: off
    def _ensure_tables(self):
        meta = MetaData()
        self.pred_tbl = Table(
            "predictions",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False, index=True),
            Column("timeframe", String(16), nullable=False, index=True),
            Column("ts_created_ms", Integer, nullable=False, index=True),
            Column("horizon", String(32), nullable=False),
            Column("q05", Float, nullable=False),
            Column("q50", Float, nullable=False),
            Column("q95", Float, nullable=False),
            Column("meta", Text, nullable=True),
        )
        self.cal_tbl = Table(
            "calibration_records",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False),
            Column("timeframe", String(16), nullable=False),
            Column("ts_created_ms", Integer, nullable=False),
            Column("alpha", Float, nullable=False),
            Column("delta_global", Float, nullable=False),
            Column("details", Text, nullable=True),
        )
        self.sig_tbl = Table(
            "signals",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False),
            Column("timeframe", String(16), nullable=False),
            Column("ts_created_ms", Integer, nullable=False),
            Column("entry_price", Float, nullable=False),
            Column("target_price", Float, nullable=False),
            Column("stop_price", Float, nullable=False),
            Column("metrics", Text, nullable=True),
        )
        # Ensure latents table exists (latent vectors used by ML pipeline)
        self.latents_tbl = Table(
            "latents",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=True, index=True),
            Column("timeframe", String(16), nullable=True, index=True),
            Column("ts_utc", Integer, nullable=False, index=True),
            Column("model_version", String(128), nullable=True),
            Column("latent_json", Text, nullable=False),
            Column("regime_label", String(32), nullable=True, index=True),
            Column("ts_created_ms", Integer, nullable=False),
        )
        self.features_tbl = Table(
            "features",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False, index=True),
            Column("timeframe", String(16), nullable=False, index=True),
            Column("ts_utc", Integer, nullable=False, index=True),
            Column("pipeline_version", String(128), nullable=True),
            Column("features_json", Text, nullable=False),
            Column("ts_created_ms", Integer, nullable=False),
        )
        self.market_data_ticks_tbl = Table(
            "market_data_ticks",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False, index=True),
            Column("timeframe", String(16), nullable=False, index=True),
            Column("ts_utc", Integer, nullable=False, index=True),
            Column("price", Float, nullable=True),
            Column("bid", Float, nullable=True),
            Column("ask", Float, nullable=True),
            Column("volume", Float, nullable=True),
            Column("ts_created_ms", Integer, nullable=False),
        )
        # Optional aggregated ticks table
        self.ticks_tbl = Table(
            "ticks_aggregate",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False, index=True),
            Column("timeframe", String(16), nullable=False, index=True),
            Column("ts_utc", Integer, nullable=False, index=True),
            Column("tick_count", Integer, nullable=False),
            Column("ts_created_ms", Integer, nullable=False),
        )
        # Metrics table for various operational metrics
        self.metrics_tbl = Table(
            "metrics",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("name", String(128), nullable=False, index=True),
            Column("value", Float, nullable=False),
            Column("labels", Text, nullable=True),
            Column("ts_created_ms", Integer, nullable=False, index=True),
        )
        # Create all tables (idempotent)
        meta.create_all(self.engine)

    # fmt: on

    # Ensure additional columns exist on existing DB (e.g. regime_label added later)
    def _ensure_latents_columns(self) -> None:
        """
        Ensure 'regime_label' column exists in latents table for existing DBs.
        """
        try:
            from sqlalchemy import text

            dialect = getattr(self.engine, "dialect", None)
            name = dialect.name.lower() if dialect is not None else ""
            if name == "sqlite":
                with self.engine.connect() as conn:
                    rows = conn.execute(text("PRAGMA table_info(latents)")).fetchall()
                    existing_cols = [r[1] for r in rows]
                    if "regime_label" not in existing_cols:
                        logger.info(
                            "Adding missing column 'regime_label' to latents table."
                        )
                        conn.execute(
                            text("ALTER TABLE latents ADD COLUMN regime_label VARCHAR(32)")
                        )
            else:
                try:
                    with self.engine.begin() as conn:
                        conn.execute(
                            text("ALTER TABLE latents ADD COLUMN regime_label VARCHAR(32)")
                        )
                except Exception:
                    logger.debug(
                        "Non-sqlite DB: attempted to add regime_label column; ignoring errors."
                    )
        except Exception as e:
            logger.debug("Failed to ensure latents columns: {}", e)

    def write_prediction(
            self,
            symbol: str,
            timeframe: str,
            horizon: str,
            q05: float,
            q50: float,
            q95: float,
            meta: Optional[Dict] = None,
    ):
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    self.pred_tbl.insert().values(
                        symbol=symbol,
                        timeframe=timeframe,
                        ts_created_ms=int(time.time() * 1000),
                        horizon=horizon,
                        q05=float(q05),
                        q50=float(q50),
                        q95=float(q95),
                        meta=json.dumps(meta) if meta is not None else None,
                    )
                )
        except Exception as e:
            logger.exception("Failed to write prediction: {}", e)
            raise

    def write_calibration_record(self, record: Dict[str, Any]):
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    self.cal_tbl.insert().values(
                        symbol=record.get("symbol"),
                        timeframe=record.get("timeframe"),
                        ts_created_ms=int(record.get("ts_created_ms", time.time() * 1000)),
                        alpha=float(record.get("alpha", 0.1)),
                        delta_global=float(record.get("delta_global", 0.0)),
                        details=json.dumps(record.get("details", {})),
                    )
                )
        except Exception as e:
            logger.exception("Failed to write calibration record: {}", e)
            raise

    def write_signal(self, payload: Dict[str, Any]):
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    self.sig_tbl.insert().values(
                        symbol=payload.get("symbol"),
                        timeframe=payload.get("timeframe"),
                        ts_created_ms=int(time.time() * 1000),
                        entry_price=float(payload.get("entry_price", 0.0)),
                        target_price=float(payload.get("target_price", 0.0)),
                        stop_price=float(payload.get("stop_price", 0.0)),
                        metrics=json.dumps(payload.get("metrics", {})),
                    )
                )
        except Exception as e:
            logger.exception("Failed to write signal: {}", e)
            raise

    def write_tick(self, payload: Dict[str, Any]) -> bool:
        """
        Insert a raw tick into market_data_ticks table. Handles duplicates gracefully.
        """
        try:
            required = ["symbol", "timeframe", "ts_utc"]
            if not all(k in payload for k in required):
                logger.warning(f"write_tick missing required keys in payload: {payload}")
                return False

            if "ts_created_ms" not in payload:
                payload["ts_created_ms"] = int(time.time() * 1000)

            with self.engine.begin() as conn:
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                dialect = self.engine.dialect.name
                stmt = None
                if dialect == "postgresql":
                    stmt = pg_insert(self.market_data_ticks_tbl).values(payload)
                    stmt = stmt.on_conflict_do_nothing()
                elif dialect == "sqlite":
                    stmt = sqlite_insert(self.market_data_ticks_tbl).values(payload)
                    stmt = stmt.on_conflict_do_nothing()

                if stmt is not None:
                    conn.execute(stmt)
                else:  # Generic fallback
                    try:
                        conn.execute(self.market_data_ticks_tbl.insert().values(payload))
                    except Exception as e:
                        if "UNIQUE constraint" in str(e) or "duplicate key" in str(e):
                            logger.debug(
                                f"Duplicate tick skipped: {payload.get('symbol')} at {payload.get('ts_utc')}"
                            )
                        else:
                            raise
            return True
        except Exception as e:
            logger.exception("write_tick failed: {}", e)
            return False


