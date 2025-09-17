"""
Adherence metrics and optional plotting helpers.
- Implements adherence_metrics as specified (magnitude, direction, shape, bands, skill).
- Provides ATR-based sigma estimation and an optional matplotlib overlay for visualization.
"""
from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------- Sigma helpers ------------------------------- #



"""
Adherence metrics and optional plotting helpers.
- Implements adherence_metrics as specified (magnitude, direction, shape, bands, skill).
- Provides ATR-based sigma estimation and an optional matplotlib overlay for visualization.
"""



# ----------------------------- Core Metrics -------------------------------- #


def adherence_metrics( #new
    fut_ts: Iterable[int],
    m: Iterable[float],
    q05: Iterable[float],
    q95: Iterable[float],
    actual_ts: Iterable[int],
    actual_y: Iterable[float],
    sigma_vol: float,
    band_target: float = 0.90,
) -> Dict[str, float]:
    """Compute adherence metrics with exact timestamp alignment and ASOF fallback."""
    import numpy as np
    import pandas as pd

    # 1) Allineamento
    dfp = pd.DataFrame({"ts": list(fut_ts), "m": list(m), "q05": list(q05), "q95": list(q95)})
    dfa = pd.DataFrame({"ts": list(actual_ts), "y": list(actual_y)})

    # Join esatto
    df = dfp.merge(dfa, on="ts", how="inner").dropna()

    # Fallback: asof (nearest) se vuoto
    if df.empty:
        try:
            if not dfp.empty and not dfa.empty:
                dfp2 = dfp.copy()
                dfa2 = dfa.copy()
                dfp2["ts_dt"] = pd.to_datetime(pd.to_numeric(dfp2["ts"], errors="coerce"), unit="ms", utc=True).dt.tz_convert(None)
                dfa2["ts_dt"] = pd.to_datetime(pd.to_numeric(dfa2["ts"], errors="coerce"), unit="ms", utc=True).dt.tz_convert(None)
                fut_ts_arr = pd.to_numeric(dfp2["ts"], errors="coerce").dropna().astype("int64").to_numpy()
                if fut_ts_arr.size >= 2:
                    diffs = np.diff(np.unique(fut_ts_arr))
                    step_ms = float(np.median(diffs)) if diffs.size > 0 else 60_000.0
                else:
                    step_ms = 60_000.0
                tol_ms = max(1_000.0, 0.51 * step_ms)
                df_asof = pd.merge_asof(
                    dfp2.sort_values("ts_dt"),
                    dfa2.sort_values("ts_dt")[["ts_dt", "y"]],
                    on="ts_dt",
                    direction="nearest",
                    tolerance=pd.Timedelta(milliseconds=float(tol_ms)),
                ).dropna()
                if not df_asof.empty:
                    df = pd.DataFrame({
                        "ts": pd.to_numeric(dfp2.loc[df_asof.index, "ts"], errors="coerce").astype("int64").values,
                        "m": dfp2.loc[df_asof.index, "m"].astype(float).values,
                        "q05": dfp2.loc[df_asof.index, "q05"].astype(float).values,
                        "q95": dfp2.loc[df_asof.index, "q95"].astype(float).values,
                        "y": df_asof["y"].astype(float).values,
                    })
        except Exception:
            pass

    if df.empty:
        return {
            "adherence": 0.0,
            "mae_sigma": float("nan"),
            "dir_weighted": float("nan"),
            "shape": float("nan"),
            "coverage": float("nan"),
            "band_eff": float("nan"),
            "skill_rw": float("nan"),
            "n_points": 0,
        }

    # 2) Errori normalizzati
    sv = float(sigma_vol) if abs(float(sigma_vol)) > 0.0 else 1e-9
    e = (df["y"].astype(float) - df["m"].astype(float)) / sv
    mae_sigma = float(np.mean(np.abs(e.to_numpy(dtype=float))))
    S_mag = float(np.exp(-mae_sigma))

    # 3) Direzione pesata dal movimento
    y = df["y"].to_numpy(dtype=float)
    m_ = df["m"].to_numpy(dtype=float)
    r = np.diff(y, prepend=y[0])
    p = np.diff(m_, prepend=m_[0])
    w = np.abs(r) + 1e-12
    S_dir = float((np.sum((np.sign(r) == np.sign(p)) * w)) / np.sum(w))

    # 4) Forma del percorso (correlazione delle variazioni)
    dy = np.diff(y); dm = np.diff(m_)
    if (dy.std() < 1e-12) or (dm.std() < 1e-12):
        S_shape = 0.5
    else:
        rho = float(np.corrcoef(dm, dy)[0, 1])
        S_shape = 0.5 * (rho + 1.0)

    # 5) Coerenza delle bande predittive
    inside = (df["y"] >= df["q05"]) & (df["y"] <= df["q95"])
    C = float(np.mean(inside.to_numpy(dtype=bool)))
    S_cov = max(0.0, 1.0 - abs(C - float(band_target)) / max(1e-9, float(band_target)))

    width = (df["q95"].astype(float) - df["q05"].astype(float)) / sv
    S_eff = float(1.0 / (1.0 + float(np.median(np.abs(width.to_numpy(dtype=float))))))

    # 6) Fusione punteggi (puoi calibrare i pesi)
    S_final = float(0.40 * S_mag + 0.25 * S_dir + 0.15 * S_shape + 0.20 * (0.5*S_cov + 0.5*S_eff))

    # 7) Skill rispetto RW (baseline random-walk appena sensata)
    try:
        rw_mae = float(np.mean(np.abs(np.diff(y, prepend=y[0])))) / (sv + 1e-12)
        Skill = float(1.0 / (1.0 + rw_mae)) if np.isfinite(rw_mae) else float("nan")
    except Exception:
        Skill = float("nan")

    return {
        "adherence": S_final,
        "mae_sigma": mae_sigma,
        "dir_weighted": S_dir,
        "shape": S_shape,
        "coverage": C,
        "band_eff": S_eff,
        "skill_rw": Skill,
        "n_points": int(len(df)),
    }



def atr_sigma_from_df(
    df_candles: pd.DataFrame,
    n: int = 14,
    pre_anchor_only: bool = True,
    anchor_ts: Optional[int] = None,
    robust: bool = True,
) -> float:
    """
    Compute sigma_vol as ATR(n) from candles, optionally restricted to a pre-anchor window.

    Parameters
    ----------
    df_candles : DataFrame with columns ["ts_utc","open","high","low","close"]
    n : int
        ATR period.
    pre_anchor_only : bool
        If True, restrict computation to rows with ts_utc <= anchor_ts.
    anchor_ts : Optional[int]
        Anchor timestamp (ms). Used when pre_anchor_only=True.
    robust : bool
        If True, return median of last n ATR values; else return last ATR value.

    Returns
    -------
    float
        Sigma volatility scale.
    """
    if df_candles is None or df_candles.empty:
        return 1e-9
    df = df_candles.copy()
    if pre_anchor_only and anchor_ts is not None:
        try:
            df = df[df["ts_utc"].astype("int64") <= int(anchor_ts)]
        except Exception:
            pass
    if df.empty:
        return 1e-9

    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]

    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    # Wilder's ATR (simple moving average works fine here)
    atr = pd.Series(tr, dtype=float).rolling(window=int(max(1, n))).mean().to_numpy()
    atr = atr[~np.isnan(atr)]
    if atr.size == 0:
        return 1e-9
    if robust:
        return float(np.median(atr[-int(max(1, n)) :]))
    return float(atr[-1])


# ----------------------------- Plotting helper ----------------------------- #



    return metrics
def plot_adherence_shading(
    ax,
    fut_ts: Iterable[int],
    m: Iterable[float],
    q05: Iterable[float],
    q95: Iterable[float],
    actual_ts: Iterable[int],
    actual_y: Iterable[float],
    sigma_vol: float,
    band_target: float = 0.90,
    draw_band_hatch: bool = True,
    draw_dir_markers: bool = True,
) -> Dict[str, float]:
    """
    Draw a readable overlay to visualize adherence on a Matplotlib Axes.

    - Shades the area between m_t and y_t with a color intensity ~ |e_t|.
    - Adds hatched regions where y_t is outside [q05, q95].
    - Optionally adds green/red markers for directional agreement on a small baseline.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (same axis used to draw price).
    Others: same as adherence_metrics.

    Returns
    -------
    dict
        Metrics dict from adherence_metrics.
    """
    import matplotlib.pyplot as plt  # local import to avoid hard dep if headless

    # Align
    dfp = pd.DataFrame({"ts": list(fut_ts), "m": list(m), "q05": list(q05), "q95": list(q95)})
    dfa = pd.DataFrame({"ts": list(actual_ts), "y": list(actual_y)})
    df = dfp.merge(dfa, on="ts", how="inner").dropna()
    if df.empty:
        return {
            "adherence": 0.0,
            "mae_sigma": float("nan"),
            "dir_weighted": float("nan"),
            "shape": float("nan"),
            "coverage": float("nan"),
            "band_eff": float("nan"),
            "skill_rw": float("nan"),
            "n_points": 0,
        }

    # Metrics first (so caller can use them)
    metrics = adherence_metrics(
        fut_ts=df["ts"].to_list(),
        m=df["m"].to_list(),
        q05=df["q05"].to_list(),
        q95=df["q95"].to_list(),
        actual_ts=df["ts"].to_list(),
        actual_y=df["y"].to_list(),
        sigma_vol=sigma_vol,
        band_target=band_target,
    )

    # Normalize errors
    sv = float(sigma_vol) if abs(float(sigma_vol)) > 0.0 else 1e-9
    e = ((df["y"] - df["m"]) / sv).to_numpy(dtype=float)
    ts = df["ts"].to_numpy(dtype="int64")
    y = df["y"].to_numpy(dtype=float)
    m_ = df["m"].to_numpy(dtype=float)
    q05_ = df["q05"].to_numpy(dtype=float)
    q95_ = df["q95"].to_numpy(dtype=float)

    # Map error magnitude to color (amber->red)
    # Small errors ~ light amber; big errors ~ red
    # We'll cap |e| for color scaling
    cap = np.percentile(np.abs(e), 90) if e.size else 1.0
    cap = float(cap if cap > 1e-6 else 1.0)

    def _color_for_err(val: float) -> Tuple[float, float, float, float]:
        # 0 -> light amber, 1 -> red
        lvl = min(1.0, abs(val) / cap)
        # interpolate between amber (1.0, 0.75, 0.0) and red (0.85, 0.15, 0.15)
        amb = np.array([1.0, 0.75, 0.0])
        red = np.array([0.85, 0.15, 0.15])
        rgb = (1 - lvl) * amb + lvl * red
        alpha = 0.25 + 0.35 * lvl  # more opaque for bigger errors
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))

    # Shade segment by segment to allow variable color
    for i in range(len(ts) - 1):
        x0, x1 = ts[i], ts[i + 1]
        xs = [x0, x1]
        ax.fill_between(xs, [y[i], y[i + 1]], [m_[i], m_[i + 1]], color=_color_for_err(e[i]), linewidth=0.0, zorder=2)

    # Hatch out-of-band segments
    if draw_band_hatch:
        outside_hi = y > q95_
        outside_lo = y < q05_
        # Using step-like polygons per segment
        for i in range(len(ts) - 1):
            xs = [ts[i], ts[i + 1]]
            # Above band
            if outside_hi[i]:
                ax.fill_between(xs, [y[i], y[i + 1]], [q95_[i], q95_[i + 1]], facecolor=(0, 0, 0, 0.0),
                                hatch="///", edgecolor="none", zorder=3)
            # Below band
            if outside_lo[i]:
                ax.fill_between(xs, [q05_[i], q05_[i + 1]], [y[i], y[i + 1]], facecolor=(0, 0, 0, 0.0),
                                hatch="\\\\\\", edgecolor="none", zorder=3)

    # Direction markers (green/red dots near bottom of observed segment)
    if draw_dir_markers and len(y) >= 2:
        r = np.diff(y, prepend=y[0])
        p = np.diff(m_, prepend=m_[0])
        agree = (np.sign(r) == np.sign(p))
        y_min, y_max = np.nanmin(y), np.nanmax(y)
        base = y_min - 0.02 * (y_max - y_min)
        colors = np.where(agree, "#18a558", "#cc3333")
        ax.scatter(ts, np.full_like(ts, base, dtype=float), s=10, c=colors, marker="o", zorder=4)

    # Optional quick band line (for reference)
    ax.plot(ts, q05_, color="#888888", alpha=0.4, linewidth=1.0, zorder=1)
    ax.plot(ts, q95_, color="#888888", alpha=0.4, linewidth=1.0, zorder=1)
    ax.plot(ts, m_, color="#555555", alpha=0.6, linewidth=1.2, zorder=2)

    return metrics
