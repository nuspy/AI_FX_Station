from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.engine import Engine

from ..utils.config import get_config




def fetch_candles(
    engine: Optional[Engine],
    symbol: str,
    timeframe: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch candles for symbol/timeframe from DB.

    No REST calls; uses SQLAlchemy Core with reflected table 'market_data_candles'.
    Returns DataFrame sorted by ts_utc ascending with columns from the table.
    """
    eng = engine or get_engine()
    md = MetaData()
    md.reflect(bind=eng, only=["market_data_candles"])
    tbl = md.tables.get("market_data_candles")
    if tbl is None:
        raise RuntimeError("market_data_candles table not found")
    stmt = select(tbl).where(tbl.c.symbol == symbol).where(tbl.c.timeframe == timeframe)
    if start_ms is not None:
        stmt = stmt.where(tbl.c.ts_utc >= int(start_ms))
    if end_ms is not None:
        stmt = stmt.where(tbl.c.ts_utc <= int(end_ms))
    stmt = stmt.order_by(tbl.c.ts_utc.asc())
    if limit is not None:
        # emulate limit from the end: order desc, limit, then resort asc
        stmt = select(tbl).where(tbl.c.symbol == symbol).where(tbl.c.timeframe == timeframe)
        if start_ms is not None:
            stmt = stmt.where(tbl.c.ts_utc >= int(start_ms))
        if end_ms is not None:
            stmt = stmt.where(tbl.c.ts_utc <= int(end_ms))
        stmt = stmt.order_by(tbl.c.ts_utc.desc()).limit(int(limit))
        with eng.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=rows[0]._mapping.keys())
        return df.sort_values("ts_utc").reset_index(drop=True)
    with eng.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=rows[0]._mapping.keys()).sort_values("ts_utc").reset_index(drop=True)

def get_engine() -> Engine:
    cfg = get_config()
    db_url = getattr(cfg.db, "database_url", None) or (cfg.db.get("database_url") if isinstance(cfg.db, dict) else None)
    if not db_url:
        raise RuntimeError("Database URL not configured")
    return create_engine(db_url, future=True)


