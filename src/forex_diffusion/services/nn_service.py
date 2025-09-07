"""
NearestNeighborService: compute neighborhood on saved latents and return regime proxy.

- Uses sklearn.NearestNeighbors to find k nearest latents to a query latent vector.
- Loads latents from DB (optionally filtered by symbol/timeframe)
- Returns dict: {p_hit:..., regime_score:..., neighbor_ids: [...]} where regime_score can be mean label or distance-weighted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import MetaData, select
from loguru import logger

from .db_service import DBService


class NearestNeighborService:
    def __init__(self, engine=None):
        self.db = DBService(engine=engine)
        self.engine = self.db.engine

    def _load_latents(self, symbol: Optional[str] = None, timeframe: Optional[str] = None, limit: int = 10000):
        meta = MetaData()
        meta.reflect(bind=self.engine, only=["latents"])
        tbl = meta.tables.get("latents")
        if tbl is None:
            return []
        with self.engine.connect() as conn:
            stmt = select(tbl.c.id, tbl.c.latent_json, tbl.c.ts_utc).order_by(tbl.c.ts_utc.desc()).limit(limit)
            if symbol is not None:
                stmt = stmt.where(tbl.c.symbol == symbol)
            if timeframe is not None:
                stmt = stmt.where(tbl.c.timeframe == timeframe)
            rows = conn.execute(stmt).fetchall()
            data = []
            ids = []
            ts = []
            for r in rows:
                ids.append(int(r[0]))
                ts.append(int(r[2]))
                try:
                    vec = np.asarray(json.loads(r[1]), dtype=float)
                except Exception:
                    vec = np.array([])
                data.append(vec)
            # reverse to ascending time
            return ids[::-1], data[::-1], ts[::-1]

    def get_regime(self, query_latent: List[float], symbol: Optional[str] = None, timeframe: Optional[str] = None, k: int = 10) -> Dict[str, Any]:
        """
        Return nearest-neighbor regime info for query_latent.
        """
        ids, data, ts = self._load_latents(symbol=symbol, timeframe=timeframe, limit=50000)
        if len(data) == 0:
            return {"found": False, "reason": "no_latents"}

        # filter out variable-length vectors
        X = [d for d in data if len(d) == len(query_latent)]
        if len(X) == 0:
            return {"found": False, "reason": "dimension_mismatch"}
        X = np.vstack(X)
        q = np.asarray(query_latent, dtype=float).reshape(1, -1)
        try:
            nbrs = NearestNeighbors(n_neighbors=min(k, X.shape[0]), algorithm="auto").fit(X)
            dist, idx = nbrs.kneighbors(q)
            idx = idx.flatten()
            dist = dist.flatten()
            # compute regime proxy: average sign of next-return from neighbor (if stored) - fallback to proximity
            # Here we don't have labels; approximate regime_score = 1/(1+mean(dist))
            regime_score = float(1.0 / (1.0 + float(np.mean(dist))))
            neighbor_ids = idx.tolist()
            return {"found": True, "regime_score": regime_score, "neighbor_idx": neighbor_ids, "distances": dist.tolist()}
        except Exception as e:
            logger.exception("NN service error: {}", e)
            return {"found": False, "reason": str(e)}
