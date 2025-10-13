"""
In-process training helpers.

Provides `train_sklearn_inproc` which runs a sklearn training flow reusing
helpers from src.forex_diffusion.training.train_sklearn.py while taking
a `fetch_candles` callable provided by the caller (UI), so DB access is
delegated to the application context.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import pandas as pd

# Import trainer helpers (the CLI module contains reusable functions)
try:
    from . import train_sklearn as trainer_mod  # type: ignore
except Exception:
    # fallback path (older layout)
    try:
        from forex_diffusion.training import train_sklearn as trainer_mod  # type: ignore
    except Exception:
        trainer_mod = None  # will error later if missing


def _make_args_from_cfg(cfg: Dict[str, Any]) -> SimpleNamespace:
    """
    Build an args-like object expected by trainer._build_features / main helpers.
    Provide sensible defaults when keys missing.
    """
    a = SimpleNamespace()
    a.symbol = cfg.get("symbol")
    a.timeframe = cfg.get("timeframe")
    a.horizon = int(cfg.get("horizon", cfg.get("horizon_bars", 5)))
    a.algo = cfg.get("algo", cfg.get("model_type", "ridge"))
    a.pca = int(cfg.get("pca", cfg.get("pca_components", 0) or 0))
    a.artifacts_dir = str(
        cfg.get("artifacts_dir", cfg.get("artifacts_dir", "./artifacts"))
    )
    a.warmup_bars = int(
        cfg.get("warmup_bars", cfg.get("advanced_params", {}).get("warmup_bars", 16))
    )
    a.val_frac = float(cfg.get("val_frac", 0.2))
    a.alpha = float(cfg.get("alpha", 0.001))
    a.l1_ratio = float(cfg.get("l1_ratio", 0.5))
    a.random_state = int(cfg.get("random_state", 0))
    a.n_estimators = int(cfg.get("n_estimators", 400))
    a.days_history = int(cfg.get("days_history", cfg.get("days", 60)))
    a.atr_n = int(cfg.get("atr_n", cfg.get("advanced_params", {}).get("atr_n", 14)))
    a.rsi_n = int(cfg.get("rsi_n", cfg.get("advanced_params", {}).get("rsi_n", 14)))
    a.bb_n = int(cfg.get("bb_n", cfg.get("advanced_params", {}).get("bb_n", 20)))
    a.hurst_window = int(
        cfg.get("hurst_window", cfg.get("advanced_params", {}).get("hurst_window", 64))
    )
    a.rv_window = int(
        cfg.get("rv_window", cfg.get("advanced_params", {}).get("rv_window", 60))
    )
    a.min_feature_coverage = float(
        cfg.get(
            "min_feature_coverage",
            cfg.get("advanced_params", {}).get("min_feature_coverage", 0.15),
        )
    )
    raw_tfs = cfg.get("indicator_tfs", {})
    if isinstance(raw_tfs, str):
        a.indicator_tfs = raw_tfs
    else:
        try:
            a.indicator_tfs = json.dumps(raw_tfs or {})
        except Exception:
            a.indicator_tfs = "{}"
    # boolean flags
    a.use_relative_ohlc = bool(cfg.get("use_relative_ohlc", True))
    a.use_temporal_features = bool(cfg.get("use_temporal_features", True))
    return a


def train_sklearn_inproc(
    fetch_candles: Callable[[str, str, int], pd.DataFrame],
    log: Callable[[str], None],
    progress: Optional[Callable[[int], None]] = None,
    **cfg: Any,
) -> Dict[str, Any]:
    """
    Run sklearn training in-process.

    Parameters
    ----------
    fetch_candles : callable(symbol, timeframe, days_history) -> pd.DataFrame
        Provided by the caller (UI) to load candles.
    log : callable(str)
        Logging callback for textual progress.
    progress : callable(int) optional
        Progress callback (0..100) or -1 for indeterminate.

    cfg: arguments similar to CLI/training_tab (symbol, timeframe, horizon, artifacts_dir, ...)

    Returns
    -------
    dict with keys: ok (bool), path (Path or None), val_mae (float), msg (str)
    """
    try:
        if trainer_mod is None:
            raise RuntimeError("Trainer helpers module not available")

        args = _make_args_from_cfg(cfg)

        log(
            f"[train] preparing training for {args.symbol} {args.timeframe} horizon={args.horizon}"
        )
        if progress:
            try:
                progress(-1)
            except Exception:
                pass

        # 1) fetch candles via provided callable
        try:
            candles = fetch_candles(args.symbol, args.timeframe, int(args.days_history))
        except Exception as e:
            raise RuntimeError(f"fetch_candles failed: {e}")

        if not isinstance(candles, pd.DataFrame):
            raise RuntimeError("fetch_candles must return a pandas.DataFrame")

        req = {"ts_utc", "open", "high", "low", "close"}
        if not req.issubset(set(candles.columns)):
            raise RuntimeError(
                f"Candles missing required columns: {req} (got {list(candles.columns)})"
            )

        log("[train] computing features")
        # 2) build features using trainer helper (expects (candles, args))
        X, y, meta = trainer_mod._build_features(candles, args)

        log(f"[train] features shape {X.shape}, labels {y.shape}")
        # 3) standardize and split
        (Xtr, ytr), (Xva, yva), (mu, sigma) = trainer_mod._standardize_train_val(
            X, y, args.val_frac
        )

        if progress:
            try:
                progress(30)
            except Exception:
                pass

        # 4) optional PCA
        pca_model = None
        if int(args.pca) > 0:
            ncomp = min(int(args.pca), Xtr.shape[1], Xtr.shape[0])
            if ncomp > 0:
                from sklearn.decomposition import PCA

                pca_model = PCA(
                    n_components=ncomp, whiten=False, random_state=args.random_state
                )
                Xtr = pca_model.fit_transform(Xtr)
                Xva = pca_model.transform(Xva)

        if progress:
            try:
                progress(55)
            except Exception:
                pass

        # 5) fit model
        log(f"[train] fitting model algo={args.algo}")
        model = trainer_mod._fit_model(args.algo, Xtr, ytr, args)
        model.fit(Xtr, ytr)

        # 6) validate
        val_pred = model.predict(Xva)
        from sklearn.metrics import mean_absolute_error

        mae = float(mean_absolute_error(yva, val_pred))
        log(f"[train] validation MAE={mae:.6f}")

        if progress:
            try:
                progress(80)
            except Exception:
                pass

        # 7) save artifact
        out_dir = Path(args.artifacts_dir) / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{args.symbol.replace('/','')}_{args.timeframe}_d{args.days_history}_h{args.horizon}_{args.algo}{'_pca'+str(args.pca) if int(args.pca)>0 else ''}"
        payload = {
            "model_type": args.algo,
            "model": model,
            "scaler_mu": mu,
            "scaler_sigma": sigma,
            "pca": pca_model,
            "features": meta.get("features"),
            "indicator_tfs": meta.get("indicator_tfs"),
            "params_used": meta.get("args_used"),
            "val_mae": mae,
        }
        out_path = out_dir / f"{run_name}.pkl"
        # use joblib.dump for sklearn objects
        try:
            from joblib import dump

            dump(payload, out_path, compress=3)
        except Exception:
            import pickle

            with open(out_path, "wb") as f:
                pickle.dump(payload, f)

        log(f"[OK] saved model to {out_path} (val_mae={mae:.6f})")
        if progress:
            try:
                progress(100)
            except Exception:
                pass

        return {"ok": True, "path": str(out_path), "val_mae": mae, "msg": "trained"}
    except Exception as e:
        tb = traceback.format_exc()
        try:
            log(f"[error] {e}\n{tb}")
        except Exception:
            pass
        if progress:
            try:
                progress(0)
            except Exception:
                pass
        return {"ok": False, "path": None, "val_mae": None, "msg": str(e)}
