"""
SignalService: compute first-passage probabilities (p_hit), RR and expectancy using Monte Carlo.

- Uses ModelService for sampling if available, else RW fallback.
- Exposes compute_signal_metrics(symbol, timeframe, entry_price, target_pips, stop_pips, N_samples, max_hold, pip_size)
- Persists signals via DBService if provided.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from .model_service import ModelService
from .db_service import DBService


class SignalService:
    """
    Service to compute trade signals and risk metrics.
    Now supports asynchronous persistence via DBWriter (db_writer) to avoid blocking.
    """
    def __init__(self, engine: Any, model_service: Optional[ModelService] = None, db_service: Optional[DBService] = None, db_writer: Optional["DBWriter"] = None):
        self.model = model_service or ModelService(engine=engine)
        self.db = db_service
        self.db_writer = db_writer  # optional async writer
        self.engine = engine

    def _estimate_sigma(self, symbol: str, timeframe: str) -> float:
        """
        Estimate 1-bar sigma from model_service recent data helper.
        """
        try:
            rec = self.model._get_last_close_and_recent(symbol, timeframe, n=1024)
            recent = rec["recent_closes"]
            return self.model._estimate_sigma(recent, window=100)
        except Exception:
            return 1e-6

    def first_passage_montecarlo(
        self,
        symbol: str,
        timeframe: str,
        entry_price: float,
        target_price: float,
        stop_price: float,
        N_samples: int = 1000,
        max_hold: int = 20,
        pip_size: float = 0.0001,
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation of N sample price paths (per-bar) up to max_hold.
        Returns p_hit, p_stop, empirical distributions and samples (optional).
        """
        sigma_1 = self._estimate_sigma(symbol, timeframe)
        logger.info("SignalService: sigma_1 estimate: {}", sigma_1)

        # attempt to use model sampling to get trajectories; if model available, we sample z->decode and expand to multi-step trajectories
        # MVP: when model present we sample horizon final prices; for first-passage we simulate simple geometric random walk paths
        N = int(N_samples)
        hits = 0
        stops = 0
        hit_times = []
        # simulate per-sample path with GBM increments (log-returns)
        dt = 1.0  # treat dt as 1 bar
        factor = sigma_1 * math.sqrt(dt)
        for i in range(N):
            price = entry_price
            hit = False
            for t in range(1, max_hold + 1):
                # increment log-price
                z = np.random.randn()
                price = price * math.exp(factor * z)
                if price >= target_price:
                    hits += 1
                    hit_times.append(t)
                    hit = True
                    break
                if price <= stop_price:
                    stops += 1
                    hit = True
                    break
            if not hit:
                # neither hit nor stop within max_hold
                pass

        p_hit = hits / N
        p_stop = stops / N
        # p_no_hit = 1 - p_hit - p_stop

        # compute expectancy: gain and loss in pips
        gain_pips = (target_price - entry_price) / pip_size
        loss_pips = (entry_price - stop_price) / pip_size
        # costs in pips: spread + slippage heuristics
        spread_pips = getattr(self.model.cfg, "backtest", {}).get("baseline", {}).get("spread_pips", 0.5) if hasattr(self.model, "cfg") else 0.5
        slippage_pips = getattr(self.model.cfg, "backtest", {}).get("baseline", {}).get("slippage_pips", 0.2) if hasattr(self.model, "cfg") else 0.2
        cost_pips = spread_pips + slippage_pips

        expectancy_pips = p_hit * (gain_pips - cost_pips) - (1 - p_hit) * (loss_pips + cost_pips)
        expectancy = expectancy_pips * pip_size  # convert back to price units

        result = {
            "p_hit": float(p_hit),
            "p_stop": float(p_stop),
            "gain_pips": float(gain_pips),
            "loss_pips": float(loss_pips),
            "cost_pips": float(cost_pips),
            "expectancy_price": float(expectancy),
            "expectancy_pips": float(expectancy_pips),
            "hit_times_mean": float(np.mean(hit_times)) if hit_times else None,
            "N_samples": N,
        }

        # Persist signal: prefer async DBWriter if available, otherwise fallback to DBService sync
        if getattr(self, "db_writer", None) is not None:
            try:
                enq = self.db_writer.write_signal_async(
                    symbol=symbol,
                    timeframe=timeframe,
                    entry_price=float(entry_price),
                    target_price=float(target_price),
                    stop_price=float(stop_price),
                    metrics=result,
                )
                if not enq:
                    logger.warning("SignalService: DBWriter queue full, falling back to sync write")
                    if self.db is not None:
                        self.db.write_signal({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "entry_price": float(entry_price),
                            "target_price": float(target_price),
                            "stop_price": float(stop_price),
                            "metrics": result,
                        })
            except Exception as e:
                logger.exception("SignalService: failed to enqueue signal: {}", e)
        elif self.db is not None:
            try:
                self.db.write_signal({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "entry_price": float(entry_price),
                    "target_price": float(target_price),
                    "stop_price": float(stop_price),
                    "metrics": result,
                })
            except Exception as e:
                logger.warning("SignalService: failed to persist signal: {}", e)

        return result