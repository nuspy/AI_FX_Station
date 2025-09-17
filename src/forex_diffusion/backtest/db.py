from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from sqlalchemy import JSON, Boolean, Column, Float, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.engine import Engine

from ..utils.config import get_config


class BacktestDB:
    """Lightweight DB accessor for bt_* tables using SQLAlchemy Core.
    Tables are expected to be created via Alembic migrations; this will reflect them.
    """

    def __init__(self, engine: Optional[Engine] = None):
        cfg = get_config()
        db_url = getattr(cfg.db, "database_url", None) or (cfg.db.get("database_url") if isinstance(cfg.db, dict) else None)
        self.engine = engine or create_engine(db_url, future=True)
        self.meta = MetaData()
        self.meta.reflect(bind=self.engine, only=["bt_job", "bt_config", "bt_result", "bt_trace"])
        self.t_job = self.meta.tables.get("bt_job")
        self.t_cfg = self.meta.tables.get("bt_config")
        self.t_res = self.meta.tables.get("bt_result")
        self.t_trace = self.meta.tables.get("bt_trace")

    # -- Minimal interface used by worker skeleton --
    def create_job(self, status: str = "running", user_id: Optional[str] = None, timezone: Optional[str] = None, market_calendar: Optional[str] = None) -> int:
        with self.engine.begin() as conn:
            ins = self.t_job.insert().values(created_at=self._now_ms(), status=status, user_id=user_id, timezone=timezone, market_calendar=market_calendar)
            rid = conn.execute(ins).inserted_primary_key[0]
            return int(rid)

    def update_job_status(self, job_id: int, status: str):
        from sqlalchemy import update
        with self.engine.begin() as conn:
            conn.execute(update(self.t_job).where(self.t_job.c.id == job_id).values(status=status))

    def upsert_config(self, payload: Dict[str, Any]) -> int:
        """Insert config if fingerprint is new, else return existing id."""
        from sqlalchemy import select
        with self.engine.begin() as conn:
            sel = select(self.t_cfg.c.id).where(self.t_cfg.c.fingerprint == payload["fingerprint"])  # type: ignore[attr-defined]
            row = conn.execute(sel).fetchone()
            if row is not None:
                return int(row[0])
            ins = self.t_cfg.insert().values(**payload)
            rid = conn.execute(ins).inserted_primary_key[0]
            return int(rid)

    def start_config(self, config_id: int):
        from sqlalchemy import update
        with self.engine.begin() as conn:
            stmt = update(self.t_cfg).where(self.t_cfg.c.id == config_id).values(started_at=self._now_ms())
            conn.execute(stmt)

    def end_config(self, config_id: int):
        from sqlalchemy import update
        with self.engine.begin() as conn:
            stmt = update(self.t_cfg).where(self.t_cfg.c.id == config_id).values(ended_at=self._now_ms())
            conn.execute(stmt)

    def log_trace(self, config_id: int, slice_idx: int, metrics: Dict[str, Any]):
        if self.t_trace is None:
            return
        with self.engine.begin() as conn:
            conn.execute(self.t_trace.insert().values(config_id=config_id, slice_idx=slice_idx, payload_json=metrics))

    def mark_dropped(self, config_id: int, reason: str, snapshot: Dict[str, Any]):
        from sqlalchemy import update
        with self.engine.begin() as conn:
            stmt = update(self.t_cfg).where(self.t_cfg.c.id == config_id).values(dropped=True, drop_reason=reason)
            conn.execute(stmt)
            if self.t_trace is not None:
                conn.execute(self.t_trace.insert().values(config_id=config_id, slice_idx=-1, payload_json={"status": "dropped", "reason": reason, "snapshot": snapshot}))

    def save_result(self, config_id: int, result: Dict[str, Any]):
        with self.engine.begin() as conn:
            conn.execute(self.t_res.insert().values(config_id=config_id, **result))

    def results_for_job(self, job_id: int) -> list[Dict[str, Any]]:
        from sqlalchemy import select
        with self.engine.connect() as conn:
            j = self.t_job
            c = self.t_cfg
            r = self.t_res
            stmt = select(c.c.id, c.c.payload_json, r).join(r, r.c.config_id == c.c.id).where(c.c.job_id == job_id)
            out = []
            for row in conn.execute(stmt):
                cfg_id = int(row[0])
                payload = row[1]
                res_map = row[2]._mapping if hasattr(row[2], "_mapping") else row[2]
                res = {k: res_map[k] for k in res_map.keys()}
                res["config_id"] = cfg_id
                res["payload_json"] = payload
                out.append(res)
            return out

    def has_result_for_fingerprint(self, fingerprint: str) -> bool:
        from sqlalchemy import select
        with self.engine.connect() as conn:
            sel = select(self.t_cfg.c.id).where(self.t_cfg.c.fingerprint == fingerprint)
            row = conn.execute(sel).fetchone()
            if not row:
                return False
            cfg_id = int(row[0])
            sel2 = select(self.t_res.c.id).where(self.t_res.c.config_id == cfg_id)
            row2 = conn.execute(sel2).fetchone()
            return row2 is not None

    def result_by_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        from sqlalchemy import select
        with self.engine.connect() as conn:
            sel_cfg = select(self.t_cfg.c.id, self.t_cfg.c.payload_json).where(self.t_cfg.c.fingerprint == fingerprint)
            row = conn.execute(sel_cfg).fetchone()
            if not row:
                return None
            cfg_id = int(row[0])
            payload = row[1]
            sel_res = select(self.t_res).where(self.t_res.c.config_id == cfg_id).order_by(self.t_res.c.id.desc())
            r = conn.execute(sel_res).fetchone()
            if not r:
                return None
            m = r._mapping if hasattr(r, "_mapping") else r
            res = {k: m[k] for k in m.keys()}
            res["config_id"] = cfg_id
            res["payload_json"] = payload
            return res

    def job_status_counts(self, job_id: int) -> Dict[str, int]:
        from sqlalchemy import select
        with self.engine.connect() as conn:
            c = self.t_cfg
            r = self.t_res
            n_cfg = conn.execute(select(c.c.id).where(c.c.job_id == job_id)).fetchall()
            n_cfg_count = len(n_cfg)
            n_res = conn.execute(select(r.c.id).join(c, r.c.config_id == c.c.id).where(c.c.job_id == job_id)).fetchall()
            n_res_count = len(n_res)
            n_drop = conn.execute(select(c.c.id).where(c.c.job_id == job_id).where(c.c.dropped == True)).fetchall()  # noqa: E712
            n_drop_count = len(n_drop)
            return {"n_configs": n_cfg_count, "n_results": n_res_count, "n_dropped": n_drop_count}

    def update_job_for_fingerprints(self, fingerprints: List[str], job_id: int) -> int:
        """Rebind existing configs to a new job_id (dedup binding). Returns updated rows count."""
        from sqlalchemy import update
        if not fingerprints:
            return 0
        with self.engine.begin() as conn:
            stmt = update(self.t_cfg).where(self.t_cfg.c.fingerprint.in_(fingerprints)).values(job_id=job_id)
            res = conn.execute(stmt)
            try:
                return int(res.rowcount or 0)
            except Exception:
                return 0

    def median_at_slice(self, job_id: int, slice_idx: int) -> Optional[float]:
        # simple median over adherence from traces at given slice when payload has 'adherence'
        from sqlalchemy import select
        if self.t_trace is None:
            return None
        with self.engine.connect() as conn:
            sel = select(self.t_trace.c.payload_json).where(self.t_trace.c.slice_idx == slice_idx)  # type: ignore[attr-defined]
            vals = []
            for r in conn.execute(sel):
                p = r[0]
                try:
                    v = float(p.get("adherence"))
                    vals.append(v)
                except Exception:
                    continue
        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2]

    def baseline_value(self, job_id: int) -> float:
        # placeholder baseline adherence for RW
        return 0.55

    # ---- Queue helpers ----
    def next_pending_job(self) -> Optional[Dict[str, Any]]:
        from sqlalchemy import select
        with self.engine.begin() as conn:
            stmt = select(self.t_job).where(self.t_job.c.status == "pending").order_by(self.t_job.c.created_at.asc())
            row = conn.execute(stmt).fetchone()
            if not row:
                return None
            m = row._mapping if hasattr(row, "_mapping") else row
            return {k: m[k] for k in m.keys()}

    def set_job_status(self, job_id: int, status: str):
        self.update_job_status(job_id, status)

    def configs_for_job(self, job_id: int) -> List[Dict[str, Any]]:
        from sqlalchemy import select
        with self.engine.connect() as conn:
            stmt = select(self.t_cfg).where(self.t_cfg.c.job_id == job_id)
            rows = conn.execute(stmt).fetchall()
            out = []
            for r in rows:
                m = r._mapping if hasattr(r, "_mapping") else r
                out.append({k: m[k] for k in m.keys()})
            return out

    def get_job_status(self, job_id: int) -> Optional[str]:
        from sqlalchemy import select
        with self.engine.connect() as conn:
            stmt = select(self.t_job.c.status).where(self.t_job.c.id == job_id)
            row = conn.execute(stmt).fetchone()
            if not row:
                return None
            return str(row[0])

    # ---- Per-config control helpers ----
    def get_config_flags(self, config_id: int) -> Dict[str, Any]:
        from sqlalchemy import select
        with self.engine.connect() as conn:
            stmt = select(self.t_cfg.c.dropped, self.t_cfg.c.drop_reason).where(self.t_cfg.c.id == config_id)
            row = conn.execute(stmt).fetchone()
            if not row:
                return {"dropped": False, "drop_reason": None}
            return {"dropped": bool(row[0]), "drop_reason": row[1]}

    def cancel_config(self, config_id: int, reason: str = "user_cancel") -> None:
        self.mark_dropped(config_id, reason, {"by": "user"})

    def result_for_config(self, config_id: int) -> Optional[Dict[str, Any]]:
        from sqlalchemy import select
        with self.engine.connect() as conn:
            r = conn.execute(select(self.t_res).where(self.t_res.c.config_id == config_id).order_by(self.t_res.c.id.desc())).fetchone()
            if not r:
                return None
            m = r._mapping if hasattr(r, "_mapping") else r
            return {k: m[k] for k in m.keys()}

    @staticmethod
    def _now_ms() -> int:
        import time
        return int(time.time() * 1000)


__all__ = ["BacktestDB"]


