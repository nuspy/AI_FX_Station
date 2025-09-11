"""
DBService: simple persistence helpers for predictions, calibration records and signals.

- Ensures tables: predictions, calibration_records (if not present), signals
- Provides write_prediction, write_calibration_record, write_signal
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

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


class DBService:
    def __init__(self, engine: Optional[Engine] = None):
        cfg = get_config()
        db_url = getattr(cfg.db, "database_url", None) or (cfg.db.get("database_url") if isinstance(cfg.db, dict) else None)
        if engine is None:
            if not db_url:
                raise ValueError("Database URL not configured")
            self.engine = create_engine(db_url, future=True)
        else:
            self.engine = engine
        self._ensure_tables()

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
            Column("ts_created_ms", Integer, nullable=False),
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
"""
DBService: simple persistence helpers for predictions, calibration records and signals.

- Ensures tables: predictions, calibration_records (if not present), signals
- Provides write_prediction, write_calibration_record, write_signal
"""

import json
import time
from typing import Any, Dict, Optional

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


class DBService:
    def __init__(self, engine: Optional[Engine] = None):
        cfg = get_config()
        db_url = getattr(cfg.db, "database_url", None) or (cfg.db.get("database_url") if isinstance(cfg.db, dict) else None)
        if engine is None:
            if not db_url:
                raise ValueError("Database URL not configured")
            self.engine = create_engine(db_url, future=True)
        else:
            self.engine = engine
        self._ensure_tables()

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

        # Ensure additional columns exist on existing DB (e.g. regime_label added later)
        try:
            self._ensure_latents_columns()
        except Exception as _e:
            logger.debug("Could not ensure latents additional columns: {}", _e)

    def _ensure_latents_columns(self) -> None:
        """
        Ensure 'regime_label' column exists in latents table for existing DBs.
        - For SQLite: use PRAGMA table_info to detect column presence and ALTER TABLE to add it.
        - For other DBs: attempt ALTER TABLE and ignore errors (best-effort).
        """
        try:
            from sqlalchemy import text
            dialect = getattr(self.engine, "dialect", None)
            name = dialect.name.lower() if dialect is not None else ""
            if name == "sqlite":
                with self.engine.connect() as conn:
                    rows = conn.execute(text("PRAGMA table_info(latents)")).fetchall()
                    existing_cols = [r[1] for r in rows]  # pragma returns (cid,name,type,notnull,dflt_value,pk)
                    if "regime_label" not in existing_cols:
                        logger.info("Adding missing column 'regime_label' to latents table (sqlite ALTER TABLE).")
                        conn.execute(text("ALTER TABLE latents ADD COLUMN regime_label VARCHAR(32)"))
            else:
                # Best-effort: try to add column; ignore if fails
                try:
                    with self.engine.begin() as conn:
                        conn.execute(text("ALTER TABLE latents ADD COLUMN regime_label VARCHAR(32)"))
                except Exception:
                    # ignore errors (column may already exist or DB not support simple ALTER)
                    logger.debug("Non-sqlite DB: attempted to add regime_label column; ignoring errors if any.")
        except Exception as e:
            logger.debug("Failed to ensure latents columns: {}", e)

    def update_latent_labels_bulk(self, rows: list[dict]) -> None:
        """
        Bulk update regime_label for latents table.
        rows: list of dicts with keys: 'id' (int) and 'regime_label' (str)
        Uses SQLAlchemy executemany for efficiency when supported.
        """
        if not rows:
            return
        try:
            # Prepare payload for executemany: each entry will be bound to its id/regime_label
            payloads = [{"id": int(r["id"]), "regime_label": r.get("regime_label")} for r in rows if "id" in r]
            if not payloads:
                return
            # Use a transaction and update each row (some DBs don't support multi-row update easily)
            with self.engine.begin() as conn:
                # Attempt a single UPDATE per payload using executemany via text clause
                from sqlalchemy import text
                stmt = text("UPDATE latents SET regime_label = :regime_label WHERE id = :id")
                conn.execute(stmt, payloads)
        except Exception as e:
            logger.exception("update_latent_labels_bulk failed, falling back to row-by-row update: {}", e)
            # Fallback: update rows one by one
            try:
                with self.engine.begin() as conn:
                    for r in rows:
                        try:
                            conn.execute(self.latents_tbl.update().where(self.latents_tbl.c.id == int(r["id"])).values(regime_label=r.get("regime_label")))
                        except Exception:
                            continue
            except Exception as ex:
                logger.exception("update_latent_labels_bulk fallback failed: {}", ex)
                raise

    def write_prediction(self, symbol: str, timeframe: str, horizon: str, q05: float, q50: float, q95: float, meta: Optional[Dict] = None):
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

            def write_features_bulk(self, rows: List[Dict[str, Any]]):
                """
                Bulk insert multiple features rows in a single transaction.
                When using Postgres, prefer COPY via STDIN for maximum throughput.
                Requires psycopg[binary] installed.
                rows: list of dicts with keys: symbol, timeframe, ts_utc, features (dict), pipeline_version
                """
                if not rows:
                    return

                dialect = self.engine.dialect.name.lower()
                now_ms = int(time.time() * 1000)

                if dialect == "postgresql":
                    # Use COPY FROM STDIN with CSV in memory for best throughput
                    import io
                    import csv

                    # Prepare CSV buffer
                    buffer = io.StringIO()
                    writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL)
                    for r in rows:
                        symbol = r.get("symbol")
                        timeframe = r.get("timeframe")
                        ts_utc = int(r.get("ts_utc"))
                        pipeline_version = r.get("pipeline_version") or None
                        features_json = json.dumps(r.get("features", {}), ensure_ascii=False)
                        writer.writerow([symbol, timeframe, str(ts_utc), pipeline_version or "", features_json, str(now_ms)])
                    buffer.seek(0)
                    # Execute COPY using raw connection
                    conn = self.engine.raw_connection()
                    try:
                        # psycopg supports copy_expert; use it
                        cur = conn.cursor()
                        copy_sql = "COPY features (symbol, timeframe, ts_utc, pipeline_version, features_json, ts_created_ms) FROM STDIN WITH (FORMAT csv)"
                        cur.copy_expert(copy_sql, buffer)
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        logger.exception("Postgres COPY failed in write_features_bulk, falling back to executemany: {}", e)
                        # fallback to executemany insert
                        payloads = []
                        for r in rows:
                            payloads.append({
                                "symbol": r.get("symbol"),
                                "timeframe": r.get("timeframe"),
                                "ts_utc": int(r.get("ts_utc")),
                                "pipeline_version": r.get("pipeline_version"),
                                "features_json": json.dumps(r.get("features", {})),
                                "ts_created_ms": now_ms,
                            })
                        with self.engine.begin() as conn2:
                            conn2.execute(self.features_tbl.insert(), payloads)
                    finally:
                        try:
                            cur.close()
                        except Exception:
                            pass
                        try:
                            conn.close()
                        except Exception:
                            pass
                else:
                    # Generic bulk insert using executemany via SQLAlchemy
                    payloads = []
                    for r in rows:
                        payloads.append({
                            "symbol": r.get("symbol"),
                            "timeframe": r.get("timeframe"),
                            "ts_utc": int(r.get("ts_utc")),
                            "pipeline_version": r.get("pipeline_version"),
                            "features_json": json.dumps(r.get("features", {})),
                            "ts_created_ms": now_ms,
                        })
                    try:
                        with self.engine.begin() as conn:
                            conn.execute(self.features_tbl.insert(), payloads)
                    except Exception as e:
                        logger.exception("Failed to write_features_bulk (fallback): {}", e)
                        raise

            def compact_features(self, older_than_days: int = 365):
                """
                Compact/retention policy: delete features older than older_than_days (based on ts_created_ms).
                Returns number of rows deleted.
                """
                try:
                    cutoff = int((time.time() - older_than_days * 86400.0) * 1000)
                    with self.engine.begin() as conn:
                        res = conn.execute(self.features_tbl.delete().where(self.features_tbl.c.ts_created_ms < cutoff))
                        # SQLAlchemy core `Result` may not provide rowcount reliably on all DBs; attempt to return int if available.
                        try:
                            deleted = int(res.rowcount) if hasattr(res, "rowcount") and res.rowcount is not None else 0
                        except Exception:
                            deleted = 0
                    logger.info("compact_features: deleted {} rows older than {} days", deleted, older_than_days)
                    return deleted
                except Exception as e:
                    logger.exception("Failed to compact features: {}", e)
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
        Insert a tick into market_data_ticks table.
        Payload expected keys: symbol, timeframe, ts_utc (ms), price, bid, ask, volume, ts_created_ms (optional).
        Returns True if inserted (or attempted), False on non-fatal failure.
        """
        try:
            symbol = payload.get("symbol")
            timeframe = payload.get("timeframe")
            ts_utc = int(payload.get("ts_utc"))
            price = payload.get("price", None)
            bid = payload.get("bid", None)
            ask = payload.get("ask", None)
            volume = payload.get("volume", None)
            created = int(payload.get("ts_created_ms", int(time.time() * 1000)))

            # Try using reflected table if available on the instance
            try:
                tbl = getattr(self, "ticks_tbl", None)
            except Exception:
                tbl = None

            # If ticks table attr not set, reflect metadata lazily
            if tbl is None:
                from sqlalchemy import MetaData
                meta = MetaData()
                try:
                    meta.reflect(bind=self.engine, only=["market_data_ticks"])
                    tbl = meta.tables.get("market_data_ticks")
                except Exception:
                    tbl = None

            if tbl is not None:
                # Prefer dialect-specific INSERT OR IGNORE for sqlite
                dialect = getattr(self.engine, "name", None) or getattr(getattr(self.engine, "url", None), "drivername", None)
                with self.engine.begin() as conn:
                    if dialect and "sqlite" in str(dialect).lower():
                        # sqlite: INSERT OR IGNORE
                        conn.execute(
                            tbl.insert().prefix_with("OR IGNORE"),
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "ts_utc": ts_utc,
                                "price": float(price) if price is not None else None,
                                "bid": float(bid) if bid is not None else None,
                                "ask": float(ask) if ask is not None else None,
                                "volume": float(volume) if volume is not None else None,
                                "ts_created_ms": created,
                            },
                        )
                        return True
                    else:
                        # generic insert; let DB raise if duplicate (caller can ignore)
                        try:
                            conn.execute(
                                tbl.insert(),
                                {
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "ts_utc": ts_utc,
                                    "price": float(price) if price is not None else None,
                                    "bid": float(bid) if bid is not None else None,
                                    "ask": float(ask) if ask is not None else None,
                                    "volume": float(volume) if volume is not None else None,
                                    "ts_created_ms": created,
                                },
                            )
                            return True
                        except Exception:
                            # likely duplicate/constraint; ignore non-fatal
                            return False
            else:
                # fallback: raw SQL INSERT with OR IGNORE for sqlite
                try:
                    from sqlalchemy import text as _text
                    dialect = getattr(self.engine, "name", None) or getattr(getattr(self.engine, "url", None), "drivername", None)
                    with self.engine.begin() as conn:
                        if dialect and "sqlite" in str(dialect).lower():
                            stmt = _text(
                                "INSERT OR IGNORE INTO market_data_ticks (symbol, timeframe, ts_utc, price, bid, ask, volume, ts_created_ms) "
                                "VALUES (:symbol, :timeframe, :ts_utc, :price, :bid, :ask, :volume, :ts_created_ms)"
                            )
                            conn.execute(
                                stmt,
                                {
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "ts_utc": ts_utc,
                                    "price": price,
                                    "bid": bid,
                                    "ask": ask,
                                    "volume": volume,
                                    "ts_created_ms": created,
                                },
                            )
                            return True
                        else:
                            stmt = _text(
                                "INSERT INTO market_data_ticks (symbol, timeframe, ts_utc, price, bid, ask, volume, ts_created_ms) "
                                "VALUES (:symbol, :timeframe, :ts_utc, :price, :bid, :ask, :volume, :ts_created_ms)"
                            )
                            try:
                                conn.execute(
                                    stmt,
                                    {
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "ts_utc": ts_utc,
                                        "price": price,
                                        "bid": bid,
                                        "ask": ask,
                                        "volume": volume,
                                        "ts_created_ms": created,
                                    },
                                )
                                return True
                            except Exception:
                                return False
                except Exception as e:
                    logger.exception("write_tick fallback raw SQL failed: {}", e)
                    return False
        except Exception as e:
            logger.exception("write_tick failed: {}", e)
            return False
    def write_prediction(self, symbol: str, timeframe: str, horizon: str, q05: float, q50: float, q95: float, meta: Optional[Dict] = None):
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
