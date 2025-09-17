from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math
import statistics as stats

from loguru import logger

from .schemas import BacktestConfigPayload
from .db import BacktestDB
from .horizons import bars_ahead_for_timeframe


ETA = 3
K_MIN = 4
DELTA = 0.03
EPS_MEDIAN = 0.0


@dataclass
class TrialConfig:
    model_name: str
    prediction_type: str
    timeframe: str
    horizons_sec: List[int]
    samples_range: Tuple[int, int, int]
    indicators: Dict[str, Any]
    interval: Dict[str, Any]
    data_version: str | None
    symbol: str | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = {
            "model": self.model_name,
            "ptype": self.prediction_type,
            "tf": self.timeframe,
            "horiz": self.horizons_sec,
            "samples": self.samples_range,
            "ind": self.indicators,
            "interval": self.interval,
            "data_v": self.data_version,
            "extra": self.extra,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class TrialState:
    config_id: int
    rung: int = 0
    slices_done: int = 0
    perf_history: List[float] = field(default_factory=list)
    n_points_accum: int = 0
    dropped: bool = False
    drop_reason: Optional[str] = None
    slice_coverage: List[float] = field(default_factory=list)
    slice_skill: List[float] = field(default_factory=list)
    slice_band_eff: List[float] = field(default_factory=list)
    horizon_adherence_accum: Dict[str, List[float]] = field(default_factory=dict)
    time_profile: List[Dict[str, Any]] = field(default_factory=list)

    def current_mean(self) -> float:
        return float(sum(self.perf_history) / max(1, len(self.perf_history)))

    def current_std(self) -> float:
        return float(stats.pstdev(self.perf_history)) if len(self.perf_history) > 1 else 0.0


class Worker:
    def __init__(self, db: Optional[BacktestDB] = None):
        self.db = db or BacktestDB()

    # --- Stubs to be implemented with project-specific data/metrics ---
    def make_slices(self, interval_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Implement basic walk-forward based on strings like '90d','7d','7d','0d'
        import datetime as _dt
        import pandas as _pd
        from sqlalchemy import MetaData, select, create_engine
        from ..utils.config import get_config

        # parse durations
        def _parse_dur(s: str) -> _dt.timedelta:
            s = (s or "0d").strip().lower()
            if s.endswith("d"):
                return _dt.timedelta(days=int(s[:-1]))
            if s.endswith("h"):
                return _dt.timedelta(hours=int(s[:-1]))
            if s.endswith("m"):
                return _dt.timedelta(minutes=int(s[:-1]))
            return _dt.timedelta(days=int(s))

        wf = (interval_cfg or {}).get("walkforward", {})
        train_d = _parse_dur(str(wf.get("train", "90d")))
        test_d = _parse_dur(str(wf.get("test", "7d")))
        step_d = _parse_dur(str(wf.get("step", "7d")))
        gap_d = _parse_dur(str(wf.get("gap", "0d")))

        # timeframe inference via recent candles timestamps
        cfg = get_config()
        db_url = getattr(cfg.db, "database_url", None) or (cfg.db.get("database_url") if isinstance(cfg.db, dict) else None)
        engine = create_engine(db_url, future=True)
        md = MetaData()
        md.reflect(bind=engine, only=["market_data_candles"])
        tbl = md.tables.get("market_data_candles")

        # Determine absolute interval bounds
        now = _dt.datetime.now(_dt.timezone.utc)
        itype = (interval_cfg or {}).get("type", "preset")
        start_ts = (interval_cfg or {}).get("start_ts")
        end_ts = (interval_cfg or {}).get("end_ts")
        if itype == "preset":
            preset = (interval_cfg or {}).get("preset", "30d")
            total_d = _parse_dur(str(preset))
            end_dt = now
            start_dt = now - total_d
        elif itype == "absolute" and start_ts and end_ts:
            start_dt = _dt.datetime.fromtimestamp(int(start_ts)/1000.0, tz=_dt.timezone.utc)
            end_dt = _dt.datetime.fromtimestamp(int(end_ts)/1000.0, tz=_dt.timezone.utc)
        else:  # relative
            n = int((interval_cfg or {}).get("relative_n") or 30)
            unit = str((interval_cfg or {}).get("relative_unit") or "days").lower()
            dur = _dt.timedelta(days=n) if unit.startswith("day") else _dt.timedelta(minutes=n)
            end_dt = now
            start_dt = now - dur

        # Convert to ms
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Build slice windows along [start_ms, end_ms]
        slices: List[Dict[str, Any]] = []
        t0 = start_dt
        while True:
            train_start = t0
            train_end = train_start + train_d
            test_start = train_end + gap_d
            test_end = test_start + test_d
            if test_end > end_dt:
                break
            slices.append({
                "train_start_ms": int(train_start.timestamp()*1000),
                "train_end_ms": int(train_end.timestamp()*1000),
                "test_start_ms": int(test_start.timestamp()*1000),
                "test_end_ms": int(test_end.timestamp()*1000),
            })
            t0 = t0 + step_d
        return slices if slices else [{"train_start_ms": start_ms, "train_end_ms": start_ms, "test_start_ms": start_ms, "test_end_ms": end_ms}]

    def run_one_slice(self, cfg: TrialConfig, slice_def: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: compute adherence on slice using merge_asof alignment
        # Placeholder: query last candles for symbol/timeframe and synthesize trivial metrics
        try:
            from sqlalchemy import MetaData, select
            from sqlalchemy import create_engine
            from ..utils.config import get_config
            from .data_access import fetch_candles
            from ..postproc.adherence import adherence_metrics, atr_sigma_from_df
            import pandas as pd

            cfg_env = get_config()
            db_url = getattr(cfg_env.db, "database_url", None) or (cfg_env.db.get("database_url") if isinstance(cfg_env.db, dict) else None)
            engine = create_engine(db_url, future=True)
            sym = cfg.symbol or "EUR/USD"
            # fetch strictly from DB (no REST) using data_access helper
            df = fetch_candles(engine, sym, cfg.timeframe, start_ms=int(slice_def.get("train_start_ms", 0)) or None, end_ms=int(slice_def.get("test_end_ms", 0)) or None, limit=4096)
            if df is None or df.empty:
                return {"adherence": 0.0, "n_points": 0}
            df = df.sort_values("ts_utc").reset_index(drop=True)
            # Build realized y on test slice and predictions via asof align (real per-slice would use model outputs)
            mask = (df["ts_utc"].astype("int64") >= int(slice_def.get("test_start_ms", 0))) & (df["ts_utc"].astype("int64") <= int(slice_def.get("test_end_ms", 2**63-1)))
            dft = df.loc[mask].reset_index(drop=True)
            if dft.empty:
                return {"adherence": 0.0, "n_points": 0}
            # per-horizon metrics using naive RW median and ATR bands scaled by sqrt(h)
            import math as _math
            tf = cfg.timeframe
            base_sigma = atr_sigma_from_df(df, n=14, pre_anchor_only=True, anchor_ts=int(dft["ts_utc"].iloc[0]))

            def _sec_to_label(sec: int) -> str:
                if sec % 3600 == 0:
                    return f"{sec//3600}h"
                if sec % 60 == 0:
                    return f"{sec//60}m"
                return f"{sec}s"

            horizon_profile: Dict[str, Dict[str, float]] = {}
            adh_list: List[float] = []
            cov_list: List[float] = []
            skill_list: List[float] = []
            be_list: List[float] = []
            npoints_total = 0
            for h_sec in (cfg.horizons_sec or [60]):
                try:
                    ba = max(1, int(bars_ahead_for_timeframe(h_sec, tf)))
                except Exception:
                    ba = 1
                close_vals = dft["close"].astype(float).to_numpy()
                ts_vals = dft["ts_utc"].astype("int64").to_numpy()
                if close_vals.size <= ba:
                    continue
                m = close_vals[:-ba]
                fut_ts = ts_vals[ba:]
                y = close_vals[ba:]
                sigma_h = float(base_sigma * _math.sqrt(float(ba)))
                q05 = (m - sigma_h).tolist()
                q95 = (m + sigma_h).tolist()
                m_list = m.tolist()
                fut_ts_list = fut_ts.tolist()
                met = adherence_metrics(
                    fut_ts=fut_ts_list, m=m_list, q05=q05, q95=q95,
                    actual_ts=fut_ts_list, actual_y=y.tolist(),
                    sigma_vol=sigma_h, band_target=0.90,
                )
                lab = _sec_to_label(int(h_sec))
                # compute simple per-horizon quantiles of error magnitude |y-m| for diagnostics
                try:
                    import numpy as _np
                    err = _np.abs(_np.asarray(y, dtype=float) - _np.asarray(m_list, dtype=float))
                    q10 = float(_np.percentile(err, 10))
                    q50 = float(_np.percentile(err, 50))
                    q90 = float(_np.percentile(err, 90))
                except Exception:
                    q10 = q50 = q90 = 0.0
                horizon_profile[lab] = {
                    "adherence": float(met.get("adherence", 0.0)),
                    "coverage": float(met.get("coverage", 0.0) if met.get("coverage") is not None else 0.0),
                    "band_eff": float(met.get("band_eff", 0.0) if met.get("band_eff") is not None else 0.0),
                    "skill_rw": float(met.get("skill_rw", 0.0) if met.get("skill_rw") is not None else 0.0),
                    "n_points": int(met.get("n_points", 0)),
                    "err_q10": q10, "err_q50": q50, "err_q90": q90,
                }
                adh_list.append(float(met.get("adherence", 0.0)))
                cov_list.append(float(met.get("coverage", 0.0) if met.get("coverage") is not None else 0.0))
                be_list.append(float(met.get("band_eff", 0.0) if met.get("band_eff") is not None else 0.0))
                skill_list.append(float(met.get("skill_rw", 0.0) if met.get("skill_rw") is not None else 0.0))
                npoints_total += int(met.get("n_points", 0))

            adh_slice = float(sum(adh_list) / max(1, len(adh_list)))
            cov_slice = float(sum(cov_list) / max(1, len(cov_list))) if cov_list else 0.0
            be_slice = float(sum(be_list) / max(1, len(be_list))) if be_list else 0.0
            skill_slice = float(sum(skill_list) / max(1, len(skill_list))) if skill_list else 0.0

            return {
                "adherence": adh_slice,
                "coverage": cov_slice,
                "band_eff": be_slice,
                "skill_rw": skill_slice,
                "n_points": int(npoints_total),
                "horizon_profile": horizon_profile,
            }
        except Exception:
            return {"adherence": 0.0, "n_points": 0}

    # --- Early stop rules ---
    def _should_stop(self, state: TrialState, baseline: float, median_t: Optional[float]) -> Tuple[bool, Optional[str]]:
        n = state.slices_done
        if n >= K_MIN:
            mu = state.current_mean()
            sd = state.current_std()
            ci_lo = mu - 1.96 * (sd / math.sqrt(max(1, n)))
            if ci_lo < (baseline - DELTA):
                return True, "baseline_fail"
        if median_t is not None and n >= max(3, K_MIN - 1):
            if state.perf_history and state.perf_history[-1] < (median_t - EPS_MEDIAN):
                return True, "median_fail"
        return False, None

    # --- ASHA rung design ---
    @staticmethod
    def _build_rungs(total_slices: int, eta: int = ETA) -> List[int]:
        rungs: List[int] = []
        b = max(1, total_slices // (eta ** 2))
        while b < total_slices:
            rungs.append(b)
            b *= eta
        rungs.append(total_slices)
        return rungs

    # --- Entry point per job ---
    def run_job(self, job_id: int, configs: List[TrialConfig]):
        # Dedup and register configs
        states: List[TrialState] = []
        for cfg in configs:
            payload = {
                "job_id": job_id,
                "fingerprint": cfg.fingerprint(),
                "payload_json": {
                    "model": cfg.model_name,
                    "ptype": cfg.prediction_type,
                    "timeframe": cfg.timeframe,
                    "horizons_sec": cfg.horizons_sec,
                    "samples_range": cfg.samples_range,
                    "indicators": cfg.indicators,
                    "interval": cfg.interval,
                    "data_version": cfg.data_version,
                    "extra": cfg.extra,
                },
            }
            config_id = self.db.upsert_config(payload)
            self.db.start_config(config_id)
            states.append(TrialState(config_id=config_id))

        slices = self.make_slices(configs[0].interval)
        total_slices = len(slices)
        rungs = self._build_rungs(total_slices)
        baseline = self.db.baseline_value(job_id)

        import time as _time
        for r, budget in enumerate(rungs):
            alive_idx = [i for i, st in enumerate(states) if not st.dropped]
            if not alive_idx:
                break
            for i in alive_idx:
                st = states[i]
                cfg = configs[i]
                while st.slices_done < budget and not st.dropped:
                    # pause/cancel guard
                    try:
                        status_now = self.db.get_job_status(job_id) if hasattr(self.db, "get_job_status") else None
                        if status_now == "cancelled":
                            # stop processing further slices
                            st.dropped = True
                            st.drop_reason = "cancelled"
                            break
                        if status_now == "paused":
                            # wait until resumed or cancelled
                            while True:
                                _time.sleep(0.3)
                                status_now = self.db.get_job_status(job_id) if hasattr(self.db, "get_job_status") else None
                                if status_now in (None, "running", "done"):
                                    break
                                if status_now == "cancelled":
                                    st.dropped = True
                                    st.drop_reason = "cancelled"
                                    break
                            if st.dropped:
                                break
                        # per-config cancellation
                        flags = self.db.get_config_flags(st.config_id) if hasattr(self.db, "get_config_flags") else {"dropped": False}
                        if flags.get("dropped"):
                            st.dropped = True
                            st.drop_reason = str(flags.get("drop_reason") or "cancelled")
                            break
                    except Exception:
                        pass
                    s_idx = st.slices_done
                    slice_def = slices[s_idx]
                    m = self.run_one_slice(cfg, slice_def)
                    # include advanced slice metadata in trace
                    try:
                        m.setdefault("slice_window", {
                            "train_start_ms": int(slice_def.get("train_start_ms", 0)),
                            "train_end_ms": int(slice_def.get("train_end_ms", 0)),
                            "test_start_ms": int(slice_def.get("test_start_ms", 0)),
                            "test_end_ms": int(slice_def.get("test_end_ms", 0)),
                        })
                        import datetime as _dt
                        tb = int(slice_def.get("test_end_ms", 0))
                        if tb:
                            m.setdefault("time_bucket", _dt.datetime.utcfromtimestamp(tb/1000.0).strftime("%Y-%m"))
                    except Exception:
                        pass
                    st.perf_history.append(float(m.get("adherence", 0.0)))
                    if m.get("coverage") is not None:
                        st.slice_coverage.append(float(m.get("coverage", 0.0)))
                    if m.get("skill_rw") is not None:
                        st.slice_skill.append(float(m.get("skill_rw", 0.0)))
                    if m.get("band_eff") is not None:
                        st.slice_band_eff.append(float(m.get("band_eff", 0.0)))
                    # accumulate horizon adherence/time buckets
                    try:
                        hp = m.get("horizon_profile") or {}
                        for lab, vals in hp.items():
                            st.horizon_adherence_accum.setdefault(str(lab), []).append(float(vals.get("adherence", 0.0)))
                    except Exception:
                        pass
                    try:
                        import datetime as _dt
                        t_end = int(slice_def.get("test_end_ms", 0))
                        if t_end:
                            ym = _dt.datetime.utcfromtimestamp(t_end/1000.0).strftime("%Y-%m")
                            st.time_profile.append({"bucket": ym, "adherence": float(m.get("adherence", 0.0))})
                    except Exception:
                        pass

                    st.slices_done += 1
                    st.n_points_accum += int(m.get("n_points", 0))
                    self.db.log_trace(st.config_id, s_idx, m)

                    median_t = self.db.median_at_slice(job_id, s_idx)
                    stop, reason = self._should_stop(st, baseline, median_t)
                    if stop:
                        st.dropped = True
                        st.drop_reason = reason
                        self.db.mark_dropped(st.config_id, reason, {
                            "mean": st.current_mean(),
                            "std": st.current_std(),
                            "n": st.slices_done,
                            "baseline": baseline,
                            "delta": DELTA,
                        })
                        break

            # ASHA promotion
            alive_idx = [i for i, st in enumerate(states) if not st.dropped]
            if r < len(rungs) - 1 and alive_idx:
                ranked = sorted(alive_idx, key=lambda i: states[i].current_mean(), reverse=True)
                keep = set(ranked[: max(1, len(ranked) // ETA)])
                for i in alive_idx:
                    if i not in keep and states[i].slices_done >= K_MIN:
                        states[i].dropped = True
                        states[i].drop_reason = "asha_not_promoted"
                        self.db.mark_dropped(states[i].config_id, "asha_not_promoted", {
                            "mean": states[i].current_mean(),
                            "n": states[i].slices_done,
                            "rung": r,
                        })

        # finalize results
        for st, cfg in zip(states, configs):
            if st.dropped and not st.perf_history:
                self.db.end_config(st.config_id)
                continue
            perf = st.perf_history
            perf_sorted = sorted(perf)
            def q(pct: int) -> float:
                if not perf_sorted:
                    return 0.0
                k = max(0, int(len(perf_sorted) * pct / 100) - 1)
                return float(perf_sorted[k])

            win_rate = float(sum(1 for v in perf if v >= (baseline + DELTA)) / max(1, len(perf)))
            # composite score (batch-local normalization can be done externally; here a simple proxy)
            adh_mean = float(sum(perf) / max(1, len(perf)))
            adh_std = float(stats.pstdev(perf)) if len(perf) > 1 else 0.0
            # simple skill proxy equals adh_mean for MVP
            skill_rw = adh_mean
            # coverage/band_eff aggregated from per-slice (averages); error vs target
            cov_vals = [v for v in st.slice_coverage if isinstance(v, (int, float))]
            be_vals = [v for v in st.slice_band_eff if isinstance(v, (int, float))]
            coverage_obs = float(sum(cov_vals) / max(1, len(cov_vals))) if cov_vals else None
            band_efficiency = float(sum(be_vals) / max(1, len(be_vals))) if be_vals else None
            coverage_abs_error = (abs(coverage_obs - 0.90) if coverage_obs is not None else None)
            # RobustnessIndex proxy using CV over time (std/mean of perf) – bounded [0,1]
            try:
                cv_time = float(adh_std / (abs(adh_mean) + 1e-9))
                robustness_index = float(max(0.0, min(1.0, 1.0 - cv_time)))
            except Exception:
                cv_time = None
                robustness_index = None
            complexity_penalty = 0.0
            composite_score = adh_mean - 0.1 * adh_std + (0.05 * (robustness_index or 0.0))
            # optional profiles included in results (aggregated summaries)
            horizon_profile_json = None
            if st.horizon_adherence_accum:
                horizon_profile_json = {lab: {"mean": float(sum(v)/max(1,len(v))), "n": int(len(v))} for lab, v in st.horizon_adherence_accum.items()}
            time_profile_json = None
            if st.time_profile:
                from collections import defaultdict
                import numpy as _np
                acc = defaultdict(list)
                for rec in st.time_profile:
                    acc[str(rec.get("bucket"))].append(float(rec.get("adherence", 0.0)))
                time_profile_json = {}
                for b, v in acc.items():
                    arr = _np.asarray(v, dtype=float)
                    try:
                        q10 = float(_np.percentile(arr, 10))
                        q50 = float(_np.percentile(arr, 50))
                        q90 = float(_np.percentile(arr, 90))
                    except Exception:
                        q10 = q50 = q90 = float(arr.mean() if arr.size else 0.0)
                    time_profile_json[b] = {"mean": float(arr.mean() if arr.size else 0.0), "n": int(arr.size), "q10": q10, "q50": q50, "q90": q90}
            result = {
                "adherence_mean": adh_mean,
                "adherence_std": adh_std,
                "p10": q(10), "p25": q(25), "p50": q(50), "p75": q(75), "p90": q(90),
                "win_rate_delta": win_rate, "delta_used": DELTA,
                "skill_rw": skill_rw,
                "coverage_observed": coverage_obs,
                "coverage_target": 0.90 if coverage_obs is not None else None,
                "coverage_abs_error": coverage_abs_error,
                "band_efficiency": band_efficiency,
                "robustness_index": robustness_index,
                "complexity_penalty": complexity_penalty,
                "composite_score": composite_score,
                "n_points": st.n_points_accum,
                "dropped": st.dropped, "drop_reason": st.drop_reason,
                "horizon_profile_json": horizon_profile_json,
                "time_profile_json": time_profile_json,
            }
            self.db.save_result(st.config_id, result)
            self.db.end_config(st.config_id)


__all__ = ["Worker", "TrialConfig"]


