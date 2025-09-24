from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from loguru import logger

from forex_diffusion.train.loop import ForexDiffusionLit
from .train_sklearn import fetch_candles_from_db, _coerce_indicator_tfs

CHANNEL_ORDER = ["open", "high", "low", "close", "volume", "hour_sin", "hour_cos"]


class CandlePatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, targets: np.ndarray, cond: np.ndarray | None = None):
        super().__init__()
        self.patches = torch.from_numpy(patches.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.cond = torch.from_numpy(cond.astype(np.float32)) if cond is not None else None

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"x": self.patches[idx], "y": self.targets[idx]}
        if self.cond is not None:
            sample["cond"] = self.cond[idx]
        return sample


def _add_time_features(df):
    ts = np.asarray(pd.to_datetime(df["ts_utc"], unit="ms", utc=True))
    hours = np.array([t.hour for t in ts])
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df.sort_values("ts_utc", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _build_arrays(df: pd.DataFrame, patch_len: int, horizon: int, warmup: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = df[CHANNEL_ORDER].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    cond_cols = [col for col in df.columns if col.startswith("cond_")]
    cond_values = df[cond_cols].astype(float).to_numpy() if cond_cols else None

    start_idx = max(0, warmup)
    end_idx = len(df) - horizon
    windows: List[np.ndarray] = []
    targets: List[float] = []
    cond_list: List[np.ndarray] = []
    for start in range(start_idx, end_idx - patch_len + 1):
        stop = start + patch_len
        target_idx = stop + horizon - 1
        if target_idx >= len(df):
            break
        patch = values[start:stop]
        windows.append(patch.T)  # (C, L)
        targets.append(float(closes[target_idx]))
        if cond_values is not None:
            cond_list.append(cond_values[target_idx])
    if not windows:
        raise RuntimeError("Nessun patch generato: riduci warmup o aumenta history.")
    patches = np.stack(windows, axis=0)
    y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    cond = np.stack(cond_list, axis=0) if cond_list else None
    return patches, y, cond


def _standardize_train_val(patches: np.ndarray, val_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = patches.shape[0]
    val_size = max(1, int(n * val_frac))
    train_size = n - val_size
    if train_size <= 1:
        raise RuntimeError("Split training/validation troppo piccolo: riduci val_frac o aumenta dati")
    train = patches[:train_size]
    val = patches[train_size:]
    mu = train.mean(axis=(0, 2), keepdims=True)
    sigma = train.std(axis=(0, 2), keepdims=True)
    sigma[sigma == 0] = 1.0
    train_norm = (train - mu) / sigma
    val_norm = (val - mu) / sigma
    return train_norm, val_norm, mu.squeeze(), sigma.squeeze(), np.array([train_size, val_size])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--days_history", type=int, default=90)
    ap.add_argument("--patch_len", type=int, default=64)
    ap.add_argument("--warmup_bars", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--indicator_tfs", type=str, default="{}")
    ap.add_argument("--atr_n", type=int, default=14)
    ap.add_argument("--rsi_n", type=int, default=14)
    ap.add_argument("--bb_n", type=int, default=20)
    ap.add_argument("--hurst_window", type=int, default=64)
    ap.add_argument("--rv_window", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast_dev_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    logger.info("[train] fetching candles for %s %s", args.symbol, args.timeframe)
    candles = fetch_candles_from_db(args.symbol, args.timeframe, args.days_history)
    candles = _add_time_features(candles)

    patches, targets, cond = _build_arrays(candles, args.patch_len, int(args.horizon), args.warmup_bars)
    train_x, val_x, mu, sigma, split_sizes = _standardize_train_val(patches, args.val_frac)
    train_y = targets[: split_sizes[0]]
    val_y = targets[split_sizes[0] :]
    cond_train = cond[: split_sizes[0]] if cond is not None else None
    cond_val = cond[split_sizes[0] :] if cond is not None else None

    train_ds = CandlePatchDataset(train_x, train_y, cond_train)
    val_ds = CandlePatchDataset(val_x, val_y, cond_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(0, args.num_workers // 2), pin_memory=True)

    model = ForexDiffusionLit()
    model.dataset_stats = {
        "channel_order": CHANNEL_ORDER,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
    }
    try:
        model.save_hyperparameters({
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "horizon": int(args.horizon),
            "patch_len": args.patch_len,
        })
    except Exception:
        pass

    out_dir = Path(args.artifacts_dir).resolve() / "lightning"
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = ModelCheckpoint(dirpath=out_dir, monitor="val/loss", save_top_k=3, mode="min", filename=f"{args.symbol.replace('/', '')}-{args.timeframe}-{{epoch:02d}}-{{val/loss:.4f}}")
    early = EarlyStopping(monitor="val/loss", patience=10, mode="min", verbose=True)
    lr_mon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=str(out_dir),
        callbacks=[monitor, early, lr_mon],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
    )

    logger.info("[train] starting Lightning fit on %s batches (val %s)", len(train_loader), len(val_loader))
    trainer.fit(model, train_loader, val_loader)

    ckpt_path = monitor.best_model_path or (out_dir / "last.ckpt")
    if not monitor.best_model_path:
        trainer.save_checkpoint(ckpt_path)
    logger.info("[train] best checkpoint: %s", ckpt_path)

    sidecar = Path(ckpt_path).with_suffix(Path(ckpt_path).suffix + ".meta.json")
    payload = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "horizon_bars": int(args.horizon),
        "patch_len": args.patch_len,
        "channel_order": CHANNEL_ORDER,
        "mu": model.dataset_stats["mu"],
        "sigma": model.dataset_stats["sigma"],
        "indicator_tfs": _coerce_indicator_tfs(args.indicator_tfs),
        "warmup_bars": args.warmup_bars,
        "rv_window": args.rv_window,
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
