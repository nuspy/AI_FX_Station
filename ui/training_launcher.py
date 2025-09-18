import sys, os, json, subprocess
from pathlib import Path

def ensure_package_installed(repo_root: str):
    try:
        import forex_diffusion  # noqa
        return
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", repo_root])

def start_training(symbol: str, timeframe: str, horizon_bars: int, algo: str,
                   pca_components: int, artifacts_dir: str, days_history: int,
                   warmup_bars: int, val_frac: float, alpha: float, l1_ratio: float,
                   indicator_tfs: dict, atr_n: int, rsi_n: int, bb_n: int,
                   hurst_window: int, rv_window: int, use_relative_ohlc: bool = True,
                   use_temporal_features: bool = True, random_state: int = 0,
                   n_estimators: int = 400, repo_root: str = None):
    repo_root = repo_root or str(Path(__file__).resolve().parents[1])
    ensure_package_installed(repo_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(Path(repo_root) / "src"), env.get("PYTHONPATH","")])

    if algo == "lightning":
        module = "forex_diffusion.training.train"
        cmd = [sys.executable, "-m", module,
            "--symbol", symbol, "--timeframe", timeframe, "--horizon", str(horizon_bars),
            "--artifacts_dir", artifacts_dir, "--epochs", "10"]
    else:
        module = "forex_diffusion.training.train_sklearn"
        cmd = [sys.executable, "-m", module,
            "--symbol", symbol, "--timeframe", timeframe, "--horizon", str(horizon_bars),
            "--algo", algo, "--pca", str(pca_components), "--artifacts_dir", artifacts_dir,
            "--warmup_bars", str(warmup_bars), "--val_frac", str(val_frac),
            "--alpha", str(alpha), "--l1_ratio", str(l1_ratio),
            "--days_history", str(days_history),
            "--atr_n", str(atr_n), "--rsi_n", str(rsi_n), "--bb_n", str(bb_n),
            "--hurst_window", str(hurst_window), "--rv_window", str(rv_window),
            "--indicator_tfs", json.dumps(indicator_tfs),
            "--random_state", str(random_state), "--n_estimators", str(n_estimators)]
        if use_relative_ohlc: cmd.append("--use_relative_ohlc")
        if use_temporal_features: cmd.append("--use_temporal_features")

    print("[DBG] cmd:", " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err
