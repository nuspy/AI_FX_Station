"""
CalibrationService: performs weighted ICP calibration and persists calibration records.

- Uses postproc.uncertainty.weighted_icp_calibrate for core logic.
- Persists records in SQLite table 'calibration_records' with metadata.
- Provides methods: calibrate_from_history, get_last_calibration.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine

from ..utils.config import get_config
from ..postproc.uncertainty import weighted_icp_calibrate

"""
CalibrationService: performs weighted ICP calibration and persists calibration records.

- Uses postproc.uncertainty.weighted_icp_calibrate for core logic.
- Persists records in SQLite table 'calibration_records' with metadata.
- Provides methods: calibrate_from_history, get_last_calibration.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine

from ..utils.config import get_config
from ..postproc.uncertainty import weighted_icp_calibrate


@dataclass
class CalibrationRecord:
    symbol: str
    timeframe: str
    ts_created_ms: int
    alpha: float
    half_life_days: float
    delta_global: float
    cov_hat: float
    details: Dict[str, Any]


class CalibrationService:
    """
    Service to handle conformal calibration tasks and persistence.
    """
    def __init__(self, engine: Optional[Engine] = None, db_writer: Optional["DBWriter"] = None):
        self.cfg = get_config()
        db_url = getattr(self.cfg.db, "database_url", None) or (self.cfg.db.get("database_url") if isinstance(self.cfg.db, dict) else None)
        if engine is None:
            if not db_url:
                raise ValueError("Database URL not configured for CalibrationService")
            self.engine = create_engine(db_url, future=True)
        else:
            self.engine = engine
        # optional async writer for persistence
        self.db_writer = db_writer
        self._ensure_table()

    def _ensure_table(self) -> None:
        meta = MetaData()
        self.tbl = Table(
            "calibration_records",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False, index=True),
            Column("timeframe", String(16), nullable=False, index=True),
            Column("ts_created_ms", Integer, nullable=False, index=True),
            Column("alpha", Float, nullable=False),
            Column("half_life_days", Float, nullable=False),
            Column("delta_global", Float, nullable=False),
            Column("cov_hat", Float, nullable=False),
            Column("details", Text, nullable=True),
        )
        meta.create_all(self.engine)

    def calibrate_from_history(
        self,
        symbol: str,
        timeframe: str,
        q05_hist: np.ndarray,
        q95_hist: np.ndarray,
        y_hist: np.ndarray,
        ts_hist_ms: np.ndarray,
        alpha: float = 0.10,
        half_life_days: float = 30.0,
        mondrian_buckets: Optional[np.ndarray] = None,
    ) -> CalibrationRecord:
        """
        Compute weighted ICP calibration and persist record.
        Returns CalibrationRecord dataclass.
        """
        logger.info("CalibrationService: calibrating {} {} (alpha={}, half_life_days={})", symbol, timeframe, alpha, half_life_days)
        result = weighted_icp_calibrate(
            q05_hist=q05_hist,
            q95_hist=q95_hist,
            y_hist=y_hist,
            ts_hist_ms=ts_hist_ms,
            t_now_ms=int(time.time() * 1000),
            half_life_days=half_life_days,
            alpha=alpha,
            mondrian_buckets=mondrian_buckets,
        )
        rec = CalibrationRecord(
            symbol=symbol,
            timeframe=timeframe,
            ts_created_ms=int(time.time() * 1000),
            alpha=alpha,
            half_life_days=half_life_days,
            delta_global=float(result.get("delta_global", 0.0)),
            cov_hat=float(result.get("cov_hat", 0.0)),
            details=result,
        )
        # persist
        self._persist_record(rec)
        return rec

    def _persist_record(self, rec: CalibrationRecord) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    self.tbl.insert().values(
                        symbol=rec.symbol,
                        timeframe=rec.timeframe,
                        ts_created_ms=rec.ts_created_ms,
                        alpha=rec.alpha,
                        half_life_days=rec.half_life_days,
                        delta_global=rec.delta_global,
                        cov_hat=rec.cov_hat,
                        details=json.dumps(rec.details),
                    )
                )
            logger.info("Calibration record persisted for {}/{}", rec.symbol, rec.timeframe)
        except Exception as e:
            logger.exception("Failed to persist calibration record: {}", e)
            raise

    def get_last_calibration(self, symbol: str, timeframe: str) -> Optional[CalibrationRecord]:
        with self.engine.connect() as conn:
            stmt = self.tbl.select().where(self.tbl.c.symbol == symbol).where(self.tbl.c.timeframe == timeframe).order_by(self.tbl.c.ts_created_ms.desc()).limit(1)
            r = conn.execute(stmt).first()
            if not r:
                return None
            details = {}
            try:
                details = json.loads(r["details"]) if r["details"] else {}
            except Exception:
                details = {}
            return CalibrationRecord(
                symbol=r["symbol"],
                timeframe=r["timeframe"],
                ts_created_ms=int(r["ts_created_ms"]),
                alpha=float(r["alpha"]),
                half_life_days=float(r["half_life_days"]),
                delta_global=float(r["delta_global"]),
                cov_hat=float(r["cov_hat"]),
                details=details,
            )
@dataclass
class CalibrationRecord:
    symbol: str
    timeframe: str
    ts_created_ms: int
    alpha: float
    half_life_days: float
    delta_global: float
    cov_hat: float
    details: Dict[str, Any]


class CalibrationService:
    """
    Service to handle conformal calibration tasks and persistence.
    """
    def __init__(self, engine: Optional[Engine] = None):
        self.cfg = get_config()
        db_url = getattr(self.cfg.db, "database_url", None) or (self.cfg.db.get("database_url") if isinstance(self.cfg.db, dict) else None)
        if engine is None:
            if not db_url:
                raise ValueError("Database URL not configured for CalibrationService")
            self.engine = create_engine(db_url, future=True)
        else:
            self.engine = engine
        self._ensure_table()

    def _ensure_table(self) -> None:
        meta = MetaData()
        self.tbl = Table(
            "calibration_records",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(64), nullable=False, index=True),
            Column("timeframe", String(16), nullable=False, index=True),
            Column("ts_created_ms", Integer, nullable=False, index=True),
            Column("alpha", Float, nullable=False),
            Column("half_life_days", Float, nullable=False),
            Column("delta_global", Float, nullable=False),
            Column("cov_hat", Float, nullable=False),
            Column("details", Text, nullable=True),
        )
        meta.create_all(self.engine)

    def calibrate_from_history(
        self,
        symbol: str,
        timeframe: str,
        q05_hist: np.ndarray,
        q95_hist: np.ndarray,
        y_hist: np.ndarray,
        ts_hist_ms: np.ndarray,
        alpha: float = 0.10,
        half_life_days: float = 30.0,
        mondrian_buckets: Optional[np.ndarray] = None,
    ) -> CalibrationRecord:
        """
        Compute weighted ICP calibration and persist record.
        Returns CalibrationRecord dataclass.
        """
        logger.info("CalibrationService: calibrating {} {} (alpha={}, half_life_days={})", symbol, timeframe, alpha, half_life_days)
        result = weighted_icp_calibrate(
            q05_hist=q05_hist,
            q95_hist=q95_hist,
            y_hist=y_hist,
            ts_hist_ms=ts_hist_ms,
            t_now_ms=int(time.time() * 1000),
            half_life_days=half_life_days,
            alpha=alpha,
            mondrian_buckets=mondrian_buckets,
        )
        rec = CalibrationRecord(
            symbol=symbol,
            timeframe=timeframe,
            ts_created_ms=int(time.time() * 1000),
            alpha=alpha,
            half_life_days=half_life_days,
            delta_global=float(result.get("delta_global", 0.0)),
            cov_hat=float(result.get("cov_hat", 0.0)),
            details=result,
        )
        # Persist asynchronously if DBWriter provided, otherwise persist synchronously
        try:
            if getattr(self, "db_writer", None) is not None:
                ok = self.db_writer.write_calibration_async(symbol=rec.symbol, timeframe=rec.timeframe, alpha=rec.alpha, delta_global=rec.delta_global, details=rec.details)
                if not ok:
                    logger.warning("CalibrationService: DBWriter queue full, falling back to sync persist")
                    self._persist_record(rec)
            else:
                self._persist_record(rec)
        except Exception as e:
            logger.exception("CalibrationService: failed to persist calibration record: {}", e)
            # still return the record object even if persisting failed
        return rec

    def _persist_record(self, rec: CalibrationRecord) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    self.tbl.insert().values(
                        symbol=rec.symbol,
                        timeframe=rec.timeframe,
                        ts_created_ms=rec.ts_created_ms,
                        alpha=rec.alpha,
                        half_life_days=rec.half_life_days,
                        delta_global=rec.delta_global,
                        cov_hat=rec.cov_hat,
                        details=json.dumps(rec.details),
                    )
                )
            logger.info("Calibration record persisted for {}/{}", rec.symbol, rec.timeframe)
        except Exception as e:
            logger.exception("Failed to persist calibration record: {}", e)
            raise

    def get_last_calibration(self, symbol: str, timeframe: str) -> Optional[CalibrationRecord]:
        with self.engine.connect() as conn:
            stmt = self.tbl.select().where(self.tbl.c.symbol == symbol).where(self.tbl.c.timeframe == timeframe).order_by(self.tbl.c.ts_created_ms.desc()).limit(1)
            r = conn.execute(stmt).first()
            if not r:
                return None
            details = {}
            try:
                details = json.loads(r["details"]) if r["details"] else {}
            except Exception:
                details = {}
            return CalibrationRecord(
                symbol=r["symbol"],
                timeframe=r["timeframe"],
                ts_created_ms=int(r["ts_created_ms"]),
                alpha=float(r["alpha"]),
                half_life_days=float(r["half_life_days"]),
                delta_global=float(r["delta_global"]),
                cov_hat=float(r["cov_hat"]),
                details=details,
            )
