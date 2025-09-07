"""
Backtest engine for MagicForex.

- Walk-forward splits
- Strategy: entry on median crossing threshold; target/stop from quantiles (optionally conformal adjusted)
- First-passage simulation with max_hold bars
- Baseline RW + sigma
- Metrics: Sharpe (annualized), Max Drawdown, Turnover, Net P&L
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.config import get_config
from ..postproc import uncertainty as unc


@dataclass
class TradeRecord:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    direction: str  # "long" or "short"
    target: float
    stop: float
    pnl: float
    return_pct: float
    hold_bars: int
    reason: str  # target|stop|timeout|none


class BacktestEngine:
    """
    Backtest and walk-forward engine.

    Expects a DataFrame with at least:
        - ts_utc (ms)
        - open, high, low, close
    And a quantiles DataFrame aligned with same index:
        - q05, q50, q95 (columns may be floats or named accordingly)
    """

    def __init__(self, cfg: Optional[object] = None):
        self.cfg = cfg or get_config()
        # default backtest config
        bf = getattr(self.cfg, "backtest", {}) if hasattr(self.cfg, "backtest") else {}
        wf = bf.get("walk_forward", {}) if isinstance(bf, dict) else {}
        self.n_splits = int(wf.get("n_splits", 5))
        self.train_days = int(wf.get("train_window_days", 365 * 2))
        self.val_days = int(wf.get("val_window_days", 90))
        self.test_days = int(wf.get("test_window_days", 90))

        baseline_cfg = bf.get("baseline", {}) if isinstance(bf, dict) else {}
        self.rw_sigma_window = int(baseline_cfg.get("rw_sigma_window", 100))
        self.spread_pips = float(baseline_cfg.get("spread_pips", 0.5))
        self.slippage_pips = float(baseline_cfg.get("slippage_pips", 0.2))

    # ------------------------------
    # Utility metrics
    # ------------------------------
    @staticmethod
    def compute_sharpe(returns: np.ndarray, bars_per_day: float = 24 * 60, annual_days: int = 252) -> float:
        """
        Compute annualized Sharpe ratio (assumes returns are per-bar returns).
        """
        if len(returns) == 0:
            return float("nan")
        mean_r = np.nanmean(returns)
        std_r = np.nanstd(returns, ddof=1)
        if std_r == 0 or np.isnan(std_r):
            return float("nan")
        # annualization factor: sqrt(#bars per year)
        factor = np.sqrt(bars_per_day * annual_days)
        return float((mean_r / std_r) * factor)

    @staticmethod
    def max_drawdown(equity: np.ndarray) -> float:
        """
        Compute maximum drawdown from equity curve (array of cumulative returns or equity levels)
        """
        if len(equity) == 0:
            return 0.0
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / (peak + 1e-12)
        return float(np.max(dd))

    @staticmethod
    def turnover_from_positions(positions: List[float]) -> float:
        """
        Turnover as average absolute change in position size.
        positions: list/array of position sizes over time
        """
        pos = np.asarray(positions)
        if len(pos) < 2:
            return 0.0
        changes = np.abs(np.diff(pos))
        return float(np.nanmean(changes))

    # ------------------------------
    # Baseline: RW + sigma
    # ------------------------------
    def baseline_rw_distribution(self, close_series: pd.Series, h_bars: int = 1) -> Tuple[np.ndarray, float]:
        """
        RW baseline: median = last close, sigma = sqrt(h) * rolling_std (1-bar)
        Returns (median_array, sigma_scalar)
        """
        # rolling std of log returns at 1-bar
        r = np.log(close_series).diff().fillna(0.0)
        sigma_1 = r.rolling(window=self.rw_sigma_window, min_periods=1).std(ddof=1).fillna(method="bfill").iloc[-1]
        median = close_series.iloc[-1]
        sigma_h = float(np.sqrt(h_bars) * sigma_1)
        return np.array([median]), sigma_h

    # ------------------------------
    # Simulation
    # ------------------------------
    def simulate_trades(
        self,
        market_df: pd.DataFrame,
        quantiles_df: pd.DataFrame,
        entry_threshold: float = 0.0,
        max_hold: int = 20,
        target_from: str = "q95",
        stop_from: str = "q05",
        spread_pips: Optional[float] = None,
        slippage_pips: Optional[float] = None,
        pip_size: float = 0.0001,
    ) -> Tuple[List[TradeRecord], Dict]:
        """
        Simulate trades over market_df (aligned with quantiles_df).
        Strategy (MVP):
          - Entry (long) when q50 >= close + entry_threshold (absolute price or fraction? here absolute)
          - Entry price = next bar's open (slippage/spread applied)
          - Target = quantiles_df[target_from] (possible conformal adjusted)
          - Stop = quantiles_df[stop_from]
          - Exit at first bar where high >= target (win) or low <= stop (loss), or at max_hold timeout
        Returns (trades_list, summary_metrics)
        """
        cfg = self.cfg
        spread = self.spread_pips if spread_pips is None else spread_pips
        slip = self.slippage_pips if slippage_pips is None else slippage_pips

        trades: List[TradeRecord] = []
        positions = []  # hold position size over time (0/1)
        in_position = False
        entry_idx = None
        entry_price = None
        direction = "long"

        close = market_df["close"].values
        highs = market_df["high"].values
        lows = market_df["low"].values
        opens = market_df["open"].values

        q50 = quantiles_df.get("q50", quantiles_df.get(0.5, None))
        q05 = quantiles_df.get("q05", quantiles_df.get(0.05, None))
        q95 = quantiles_df.get("q95", quantiles_df.get(0.95, None))

        if q50 is None or q05 is None or q95 is None:
            raise ValueError("Quantiles DataFrame must contain q05,q50,q95 columns (or 0.05/0.5/0.95)")

        q50v = q50.values if isinstance(q50, pd.Series) else np.asarray(q50)
        q05v = q05.values if isinstance(q05, pd.Series) else np.asarray(q05)
        q95v = q95.values if isinstance(q95, pd.Series) else np.asarray(q95)

        n = len(close)
        pos_series = np.zeros(n, dtype=float)
        for t in range(n - 1):
            if not in_position:
                # Check entry condition
                if q50v[t] >= close[t] + entry_threshold:
                    # enter at next open (t+1) to avoid lookahead
                    entry_idx = t + 1
                    if entry_idx >= n:
                        break
                    raw_entry = opens[entry_idx]
                    # apply spread/slippage (assume long: subtract spread/slippage)
                    entry_px = raw_entry + (spread + slip) * pip_size
                    entry_price = entry_px
                    in_position = True
                    entry_bar = entry_idx
                    # define target/stop using quantiles at entry time t (could use t or t+1)
                    target = q95v[t]
                    stop = q05v[t]
                    # record placeholder
                    trade = {
                        "entry_idx": entry_idx,
                        "entry_price": entry_price,
                        "target": float(target),
                        "stop": float(stop),
                        "direction": direction,
                        "exit_idx": None,
                        "exit_price": None,
                        "pnl": None,
                        "return_pct": None,
                        "hold_bars": None,
                        "reason": None,
                    }
                else:
                    pos_series[t] = 0.0
            else:
                # already in position: check target/stop within this bar (we enter at entry_idx)
                # find from entry_idx to entry_idx+max_hold inclusive
                if entry_idx is None:
                    in_position = False
                    continue
                exit_found = False
                end_idx = min(entry_idx + max_hold, n - 1)
                for j in range(entry_idx, end_idx + 1):
                    # check target first
                    if highs[j] >= trade["target"]:
                        # exit at target price (assume fill at target minus half spread/slip)
                        exit_px = trade["target"] - (spread + slip) * (pip_size / 2.0)
                        pnl = exit_px - trade["entry_price"]
                        ret = pnl / trade["entry_price"]
                        trade.update({"exit_idx": j, "exit_price": exit_px, "pnl": float(pnl), "return_pct": float(ret), "hold_bars": j - entry_idx + 1, "reason": "target"})
                        trades.append(self._dict_to_trade(trade))
                        exit_found = True
                        break
                    if lows[j] <= trade["stop"]:
                        exit_px = trade["stop"] + (spread + slip) * (pip_size / 2.0)
                        pnl = exit_px - trade["entry_price"]
                        ret = pnl / trade["entry_price"]
                        trade.update({"exit_idx": j, "exit_price": exit_px, "pnl": float(pnl), "return_pct": float(ret), "hold_bars": j - entry_idx + 1, "reason": "stop"})
                        trades.append(self._dict_to_trade(trade))
                        exit_found = True
                        break
                if not exit_found:
                    # timeout exit at last available close within hold (use close at end_idx)
                    last_idx = end_idx
                    exit_px = closesafe = close[last_idx] - (spread + slip) * (pip_size / 2.0)
                    pnl = exit_px - trade["entry_price"]
                    ret = pnl / trade["entry_price"]
                    trade.update({"exit_idx": last_idx, "exit_price": exit_px, "pnl": float(pnl), "return_pct": float(ret), "hold_bars": last_idx - entry_idx + 1, "reason": "timeout"})
                    trades.append(self._dict_to_trade(trade))
                # reset position after handling
                in_position = False
                entry_idx = None
                entry_price = None

        # summary metrics
        pnl_list = np.array([t.pnl for t in trades]) if len(trades) > 0 else np.array([])
        returns = np.array([t.return_pct for t in trades]) if len(trades) > 0 else np.array([])
        equity_curve = np.cumsum(pnl_list) if pnl_list.size > 0 else np.array([])
        sharpe = self.compute_sharpe(returns)
        mdd = self.max_drawdown(equity_curve) if equity_curve.size > 0 else 0.0
        turnover = self.turnover_from_positions(pos_series.tolist())

        summary = {
            "n_trades": len(trades),
            "net_pnl": float(np.nansum(pnl_list)) if pnl_list.size > 0 else 0.0,
            "avg_return": float(np.nanmean(returns)) if returns.size > 0 else 0.0,
            "sharpe": float(sharpe) if not np.isnan(sharpe) else float("nan"),
            "max_drawdown": float(mdd),
            "turnover": float(turnover),
        }
        return trades, summary

    def _dict_to_trade(self, d: Dict) -> TradeRecord:
        return TradeRecord(
            entry_idx=int(d["entry_idx"]),
            exit_idx=int(d["exit_idx"]),
            entry_price=float(d["entry_price"]),
            exit_price=float(d["exit_price"]),
            direction=str(d.get("direction", "long")),
            target=float(d.get("target", 0.0)),
            stop=float(d.get("stop", 0.0)),
            pnl=float(d.get("pnl", 0.0)),
            return_pct=float(d.get("return_pct", 0.0)),
            hold_bars=int(d.get("hold_bars", 0)),
            reason=str(d.get("reason", "none")),
        )

    # ------------------------------
    # Walk-forward orchestration
    # ------------------------------
    def run_walk_forward(self, market_df: pd.DataFrame, quantiles_df: pd.DataFrame) -> Dict:
        """
        Run walk-forward backtest using configured train/val/test windows.
        Splits are computed sequentially without future leakage.
        Returns aggregated report with per-fold metrics and comparisons vs baseline.
        """
        cfg = self.cfg
        # Align by ts index; ensure both frames sorted and same length
        mdf = market_df.sort_values("ts_utc").reset_index(drop=True)
        qdf = quantiles_df.sort_values("ts_utc").reset_index(drop=True)
        if len(mdf) != len(qdf) or not (mdf["ts_utc"].values == qdf["ts_utc"].values).all():
            logger.warning("Market df and quantiles df not aligned by ts_utc; attempting to align by merge.")
            merged = pd.merge_asof(mdf, qdf, on="ts_utc", direction="backward", suffixes=("", "_q"))
            # split merged into market and quantiles
            mdf = merged[["ts_utc", "open", "high", "low", "close"]]
            qdf = merged[["ts_utc", "q05", "q50", "q95"]]

        reports = []
        # convert days to bars estimation: assume market_df timeframe in minutes; attempt to infer freq
        # fallback: bars per day = 1440 for 1m, else approximate via median diff
        ts = pd.to_datetime(mdf["ts_utc"].astype("int64"), unit="ms", utc=True)
        if len(ts) < 2:
            raise ValueError("Not enough data for walk-forward")
        median_delta = int((ts[1:] - ts[:-1]).median().total_seconds() / 60)
        bars_per_day = int(1440 / median_delta) if median_delta > 0 else 1440

        total_len = len(mdf)
        # compute window sizes in bars
        train_bars = int(self.train_days * bars_per_day)
        val_bars = int(self.val_days * bars_per_day)
        test_bars = int(self.test_days * bars_per_day)

        # sliding windows sequentially
        start = 0
        fold = 0
        while True:
            train_start = start
            train_end = train_start + train_bars
            val_end = train_end + val_bars
            test_end = val_end + test_bars
            if test_end > total_len:
                break
            train_df = mdf.iloc[train_start:train_end].reset_index(drop=True)
            val_df = mdf.iloc[train_end:val_end].reset_index(drop=True)
            test_df = mdf.iloc[val_end:test_end].reset_index(drop=True)
            q_train = qdf.iloc[train_start:train_end].reset_index(drop=True)
            q_val = qdf.iloc[train_end:val_end].reset_index(drop=True)
            q_test = qdf.iloc[val_end:test_end].reset_index(drop=True)

            # Simulate strategy on test_df
            trades, summary = self.simulate_trades(test_df, q_test, entry_threshold=0.0, max_hold=20)
            # Baseline: compute RW baseline CRPS etc placeholder (we compute baseline net_pnl = 0)
            baseline = {"net_pnl": 0.0, "crps": None}

            fold_report = {
                "fold": fold,
                "train_range": (int(train_df["ts_utc"].iloc[0]), int(train_df["ts_utc"].iloc[-1])),
                "val_range": (int(val_df["ts_utc"].iloc[0]), int(val_df["ts_utc"].iloc[-1])),
                "test_range": (int(test_df["ts_utc"].iloc[0]), int(test_df["ts_utc"].iloc[-1])),
                "n_trades": summary.get("n_trades"),
                "net_pnl": summary.get("net_pnl"),
                "sharpe": summary.get("sharpe"),
                "max_drawdown": summary.get("max_drawdown"),
                "turnover": summary.get("turnover"),
                "trades": trades,
                "baseline": baseline,
            }
            reports.append(fold_report)

            fold += 1
            start = start + test_bars  # move forward by test length (non-overlapping)

        agg = {"n_folds": len(reports), "folds": reports}
        return agg
