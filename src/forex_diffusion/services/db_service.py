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
    UniqueConstraint,
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
        self.features_tbl.append_constraint(
            UniqueConstraint("symbol", "timeframe", "ts_utc", name="uq_features")
        )
        self.market_data_ticks_tbl = Table(
            "market_data_ticks",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False, index=True),
            Column("ts_utc", Integer, nullable=False, index=True),
            Column("price", Float, nullable=True),
            Column("bid", Float, nullable=True),
            Column("ask", Float, nullable=True),
            Column("volume", Float, nullable=True),
            Column("ts_created_ms", Integer, nullable=False),
        )
        self.market_data_ticks_tbl.append_constraint(
            UniqueConstraint("symbol", "ts_utc", "price", name="uq_market_data_ticks")
        )
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
        self.ticks_tbl.append_constraint(
            UniqueConstraint("symbol", "timeframe", "ts_utc", name="uq_ticks_aggregate")
        )
        self.metrics_tbl = Table(
            "metrics",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("name", String(128), nullable=False, index=True),
            Column("value", Float, nullable=False),
            Column("labels", Text, nullable=True),
            Column("ts_created_ms", Integer, nullable=False, index=True),
        )
        meta.create_all(self.engine)

    # fmt: on

    def _ensure_latents_columns(self) -> None:
        try:
            from sqlalchemy import text
            dialect = self.engine.dialect.name.lower()
            if dialect == "sqlite":
                with self.engine.connect() as conn:
                    rows = conn.execute(text("PRAGMA table_info(latents)")).fetchall()
                    if "regime_label" not in [r[1] for r in rows]:
                        conn.execute(text("ALTER TABLE latents ADD COLUMN regime_label VARCHAR(32)"))
        except Exception as e:
            logger.debug(f"Failed to ensure latents columns: {e}")

    def write_prediction(self, payload: Dict[str, Any]):
        try:
            with self.engine.begin() as conn:
                conn.execute(self.pred_tbl.insert().values(**payload))
        except Exception as e:
            logger.exception(f"Failed to write prediction: {e}")

    def write_calibration_record(self, payload: Dict[str, Any]):
        try:
            with self.engine.begin() as conn:
                conn.execute(self.cal_tbl.insert().values(**payload))
        except Exception as e:
            logger.exception(f"Failed to write calibration record: {e}")

    def write_signal(self, payload: Dict[str, Any]):
        try:
            with self.engine.begin() as conn:
                conn.execute(self.sig_tbl.insert().values(**payload))
        except Exception as e:
            logger.exception(f"Failed to write signal: {e}")

    def write_tick(self, payload: Dict[str, Any]) -> bool:
        logger.debug(f"TRACE: write_tick: Received payload: {payload}")
        try:
            if "ts_created_ms" not in payload:
                payload["ts_created_ms"] = int(time.time() * 1000)

            with self.engine.begin() as conn:
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert
                stmt = sqlite_insert(self.market_data_ticks_tbl).values(payload)
                stmt = stmt.on_conflict_do_nothing()
                result = conn.execute(stmt)
                logger.debug(f"TRACE: write_tick: Insert executed. Rowcount: {result.rowcount}")
            return True
        except Exception as e:
            logger.exception(f"TRACE: write_tick failed for payload {payload}: {e}")
            return False
