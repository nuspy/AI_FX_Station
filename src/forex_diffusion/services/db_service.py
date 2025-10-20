"""
DBService: simple persistence helpers for various data models.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from loguru import logger
from sqlalchemy import (
    Column, Float, Integer, MetaData, String, Table, UniqueConstraint, create_engine
)
from sqlalchemy.engine import Engine

from ..utils.config import get_config

class DBService:
    """
    Handles database schema creation and provides methods to write data.
    """
    def __init__(self, engine: Engine | None = None):
        cfg = get_config()
        db_url = getattr(cfg.db, "database_url", None)
        self.engine = engine or create_engine(db_url, future=True)
        self._ensure_tables()

    def _ensure_tables(self):
        meta = MetaData()
        self.market_data_ticks_tbl = Table(
            "market_data_ticks", meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False, index=True),
            # Re-add the timeframe column as it exists in the physical DB
            Column("timeframe", String(16), nullable=False, index=True, server_default="tick"),
            Column("ts_utc", Integer, nullable=False, index=True),
            Column("price", Float, nullable=True),
            Column("bid", Float, nullable=True),
            Column("ask", Float, nullable=True),
            Column("volume", Float, nullable=True),
            # Multi-provider support columns (added for aggregator service)
            Column("tick_volume", Integer, nullable=True),  # Number of ticks/trades in the period
            Column("real_volume", Float, nullable=True),    # Real traded volume (if available from provider)
            Column("provider_source", String(32), nullable=True),  # Data provider identifier (e.g., 'tiingo', 'ctrader')
            Column("ts_created_ms", Integer, nullable=False),
            UniqueConstraint("symbol", "timeframe", "ts_utc", name="uq_market_data_ticks")
        )
        # Autoload other tables to prevent conflicts with existing schema
        try:
            self.pred_tbl = Table("predictions", meta, autoload_with=self.engine, extend_existing=True)
            self.cal_tbl = Table("calibration_records", meta, autoload_with=self.engine, extend_existing=True)
            self.sig_tbl = Table("signals", meta, autoload_with=self.engine, extend_existing=True)
        except Exception as e:
            logger.warning(f"Could not autoload existing tables: {e}")
        meta.create_all(self.engine)

    def write_tick(self, payload: Dict[str, Any]) -> bool:
        """
        Inserts a raw tick, ensuring 'timeframe' is present to satisfy NOT NULL constraint.
        """
        try:
            if "symbol" not in payload or "ts_utc" not in payload:
                logger.warning(f"write_tick missing required keys: {payload}")
                return False

            # Ensure timeframe is set, defaulting to 'tick'
            payload.setdefault("timeframe", "tick")
            payload.setdefault("ts_created_ms", int(time.time() * 1000))

            with self.engine.begin() as conn:
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert
                stmt = sqlite_insert(self.market_data_ticks_tbl).values(payload)
                # Use the correct unique constraint columns for conflict resolution
                stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "timeframe", "ts_utc"])
                result = conn.execute(stmt)
                # if result.rowcount > 0:
                     # logger.debug(f"Persisted tick for {payload['symbol']}")
            return True
        except Exception as e:
            logger.exception(f"write_tick failed for payload {payload}: {e}")
            return False

    # Other write methods remain unchanged
    def write_prediction(self, payload: Dict[str, Any]):
        with self.engine.begin() as conn:
            conn.execute(self.pred_tbl.insert().values(**payload))

    def write_signal(self, payload: Dict[str, Any]):
        with self.engine.begin() as conn:
            conn.execute(self.sig_tbl.insert().values(**payload))

    def write_calibration_record(self, payload: Dict[str, Any]):
        with self.engine.begin() as conn:
            conn.execute(self.cal_tbl.insert().values(**payload))
