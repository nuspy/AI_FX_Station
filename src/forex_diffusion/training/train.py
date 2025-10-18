from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from loguru import logger

from forex_diffusion.train.loop import ForexDiffusionLit
from ..data.data_loader import fetch_candles_from_db
from ..features.feature_utils import coerce_indicator_tfs

CHANNEL_ORDER = ["open", "high", "low", "close", "volume", "hour_sin", "hour_cos"]


class CandlePatchDataset(Dataset):
    def __init__(
        self, patches: np.ndarray, targets: np.ndarray, cond: np.ndarray | None = None
    ):
        super().__init__()
        self.patches = torch.from_numpy(patches.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.cond = (
            torch.from_numpy(cond.astype(np.float32)) if cond is not None else None
        )

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


def _build_arrays(
    df: pd.DataFrame, patch_len: int, horizon: int | List[int], warmup: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build patches and targets for training.
    
    Args:
        df: OHLCV dataframe with time features
        patch_len: Length of input patch
        horizon: Single horizon (int) or multiple horizons (List[int])
        warmup: Number of warmup bars
        
    Returns:
        Tuple of (patches, targets, conditions)
        - patches: (N, C, L) input sequences
        - targets: (N, 1) or (N, H) targets (single or multi-horizon)
        - conditions: (N, cond_dim) conditioning features
    """
    values = df[CHANNEL_ORDER].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    cond_cols = [col for col in df.columns if col.startswith("cond_")]
    cond_values = df[cond_cols].astype(float).to_numpy() if cond_cols else None

    # Parse horizons
    if isinstance(horizon, int):
        horizons = [horizon]
        is_multi_horizon = False
    else:
        horizons = sorted(horizon)  # Sort for consistency
        is_multi_horizon = True
    
    max_horizon = max(horizons)
    
    start_idx = max(0, warmup)
    end_idx = len(df) - max_horizon
    windows: List[np.ndarray] = []
    targets: List[np.ndarray | float] = []
    cond_list: List[np.ndarray] = []
    
    for start in range(start_idx, end_idx - patch_len + 1):
        stop = start + patch_len
        
        # Collect targets for all horizons
        if is_multi_horizon:
            multi_targets = []
            for h in horizons:
                target_idx = stop + h - 1
                if target_idx >= len(df):
                    break
                multi_targets.append(float(closes[target_idx]))
            
            if len(multi_targets) != len(horizons):
                # Skip if not all horizons available
                continue
            
            patch = values[start:stop]
            windows.append(patch.T)  # (C, L)
            targets.append(multi_targets)  # List of targets
            
            if cond_values is not None:
                # Use conditioning at the furthest horizon
                cond_list.append(cond_values[stop + max_horizon - 1])
        else:
            # Single horizon (backward compatible)
            target_idx = stop + horizons[0] - 1
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
    
    # Stack targets
    if is_multi_horizon:
        y = np.array(targets, dtype=np.float32)  # (N, H)
    else:
        y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)  # (N, 1)
    
    cond = np.stack(cond_list, axis=0) if cond_list else None
    return patches, y, cond


def _standardize_train_val(
    patches: np.ndarray, val_frac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Standardize patches ensuring NO look-ahead bias (2-way split).

    CRITICAL: Computes mean/std ONLY on training set, then applies to validation.
    This prevents information leakage from future data.

    NOTE: This function is kept for backward compatibility. New code should use
    _standardize_train_val_test() for proper 60/20/20 split (PROC-001).

    Returns:
        Tuple of (train_norm, val_norm, mu, sigma, scaler_metadata)
    """
    from scipy import stats

    n = patches.shape[0]
    val_size = max(1, int(n * val_frac))
    train_size = n - val_size

    if train_size <= 1:
        raise RuntimeError(
            "Split training/validation troppo piccolo: riduci val_frac o aumenta dati"
        )

    # Temporal split: train first, val last (NO shuffling)
    train = patches[:train_size]
    val = patches[train_size:]

    # Compute statistics ONLY on training set (NO look-ahead bias)
    mu = train.mean(axis=(0, 2), keepdims=True)
    sigma = train.std(axis=(0, 2), keepdims=True)
    sigma[sigma == 0] = 1.0

    # Apply standardization
    train_norm = (train - mu) / sigma
    val_norm = (val - mu) / sigma

    # VERIFICATION: Statistical test for look-ahead bias detection
    # Test first channel (close price) across samples
    train_flat = train_norm[:, 0, :].flatten()  # Channel 0 (open), all samples
    val_flat = val_norm[:, 0, :].flatten()

    p_value = None
    if len(train_flat) > 20 and len(val_flat) > 20:
        # Kolmogorov-Smirnov test: different distributions should have low p-value
        _, p_value = stats.ks_2samp(train_flat, val_flat)

    # Metadata for debugging
    scaler_metadata = {
        "train_size": int(train_size),
        "val_size": int(val_size),
        "train_mean_shape": list(mu.shape),
        "train_std_shape": list(sigma.shape),
        "ks_test_p_value": float(p_value) if p_value is not None else None,
    }

    # WARNING: If distributions too similar, potential look-ahead bias
    if p_value is not None and p_value > 0.8:
        logger.warning(
            f"⚠️ POTENTIAL LOOK-AHEAD BIAS DETECTED!\n"
            f"Train/Val distributions suspiciously similar (KS p-value={p_value:.3f}).\n"
            f"Expected p < 0.5 for different time periods."
        )

    return train_norm, val_norm, mu.squeeze(), sigma.squeeze(), scaler_metadata


def _standardize_train_val_test(
    patches: np.ndarray, val_frac: float = 0.2, test_frac: float = 0.2
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Standardize patches with 3-way split ensuring NO look-ahead bias.

    CRITICAL: Computes mean/std ONLY on training set, then applies to val/test.
    This prevents information leakage from future data.

    PROC-001: Implements proper 3-way split:
    - Train (60%): For model training
    - Val (20%): For early stopping
    - Test (20%): For final evaluation ONLY (never used during training)

    Args:
        patches: Input patches (N, C, L) where N=samples, C=channels, L=sequence length
        val_frac: Validation fraction (default 0.2)
        test_frac: Test fraction (default 0.2)

    Returns:
        Tuple of (train_norm, val_norm, test_norm, mu, sigma, scaler_metadata)
    """
    from scipy import stats

    n = patches.shape[0]
    test_size = max(1, int(n * test_frac))
    val_size = max(1, int(n * val_frac))
    train_size = n - val_size - test_size

    if train_size <= 1:
        raise RuntimeError(
            "Split training/validation/test troppo piccolo: riduci val_frac/test_frac o aumenta dati"
        )

    # Temporal split: train first, val middle, test last (NO shuffling)
    train = patches[:train_size]
    val = patches[train_size : train_size + val_size]
    test = patches[train_size + val_size :]

    # Compute statistics ONLY on training set (NO look-ahead bias)
    mu = train.mean(axis=(0, 2), keepdims=True)
    sigma = train.std(axis=(0, 2), keepdims=True)
    sigma[sigma == 0] = 1.0

    # Apply standardization to all splits
    train_norm = (train - mu) / sigma
    val_norm = (val - mu) / sigma
    test_norm = (test - mu) / sigma

    # VERIFICATION: Statistical test for look-ahead bias detection
    train_flat = train_norm[:, 0, :].flatten()
    val_flat = val_norm[:, 0, :].flatten()
    test_flat = test_norm[:, 0, :].flatten()

    p_value_val = None
    p_value_test = None

    if len(train_flat) > 20 and len(val_flat) > 20:
        _, p_value_val = stats.ks_2samp(train_flat, val_flat)

    if len(train_flat) > 20 and len(test_flat) > 20:
        _, p_value_test = stats.ks_2samp(train_flat, test_flat)

    # Metadata for debugging
    scaler_metadata = {
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "train_frac": train_size / n,
        "val_frac": val_size / n,
        "test_frac": test_size / n,
        "train_mean_shape": list(mu.shape),
        "train_std_shape": list(sigma.shape),
        "ks_test_p_value_val": float(p_value_val) if p_value_val is not None else None,
        "ks_test_p_value_test": (
            float(p_value_test) if p_value_test is not None else None
        ),
    }

    # WARNING: If distributions too similar, potential look-ahead bias
    if p_value_val is not None and p_value_val > 0.8:
        logger.warning(
            f"⚠️ POTENTIAL LOOK-AHEAD BIAS DETECTED (Train vs Val)!\n"
            f"Train/Val distributions suspiciously similar (KS p-value={p_value_val:.3f}).\n"
            f"Expected p < 0.5 for different time periods."
        )

    if p_value_test is not None and p_value_test > 0.8:
        logger.warning(
            f"⚠️ POTENTIAL LOOK-AHEAD BIAS DETECTED (Train vs Test)!\n"
            f"Train/Test distributions suspiciously similar (KS p-value={p_value_test:.3f}).\n"
            f"Expected p < 0.5 for different time periods."
        )

    return (
        train_norm,
        val_norm,
        test_norm,
        mu.squeeze(),
        sigma.squeeze(),
        scaler_metadata,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument(
        "--horizon", 
        type=str, 
        required=True,
        help="Single horizon (int) or comma-separated list: '15' or '15,60,240'"
    )
    ap.add_argument("--days_history", type=int, default=90)
    ap.add_argument("--patch_len", type=int, default=64)
    ap.add_argument("--warmup_bars", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--indicator_tfs", type=str, default="{}")
    ap.add_argument("--min_feature_coverage", type=float, default=0.15)
    ap.add_argument("--atr_n", type=int, default=14)
    ap.add_argument("--rsi_n", type=int, default=14)
    ap.add_argument("--bb_n", type=int, default=20)
    ap.add_argument("--hurst_window", type=int, default=64)
    ap.add_argument("--rv_window", type=int, default=60)
    ap.add_argument("--returns_window", type=int, default=100)
    ap.add_argument("--session_overlap", type=int, default=30)
    ap.add_argument("--higher_tf", type=str, default="15m")
    ap.add_argument("--vp_bins", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast_dev_run", action="store_true")

    # NVIDIA Optimization Stack parameters
    ap.add_argument(
        "--use_nvidia_opts",
        action="store_true",
        help="Enable NVIDIA optimization stack (AMP, compile, fused optimizers)",
    )
    ap.add_argument(
        "--use_amp", action="store_true", help="Enable Automatic Mixed Precision (AMP)"
    )
    ap.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Training precision (fp16, bf16, or fp32)",
    )
    ap.add_argument(
        "--compile_model",
        action="store_true",
        help="Use torch.compile for model optimization (PyTorch 2.0+)",
    )
    ap.add_argument(
        "--use_fused_optimizer",
        action="store_true",
        help="Use NVIDIA APEX fused optimizer (if available)",
    )
    ap.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention 2 (if available and GPU supports it)",
    )
    ap.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    logger.info("[train] fetching candles for {} {}", args.symbol, args.timeframe)
    candles = fetch_candles_from_db(args.symbol, args.timeframe, args.days_history)
    candles = _add_time_features(candles)

    # Parse horizon(s)
    horizon_str = args.horizon.strip()
    if ',' in horizon_str:
        horizons = [int(h.strip()) for h in horizon_str.split(',')]
        logger.info(f"[Multi-Horizon] Training with horizons: {horizons}")
    else:
        horizons = int(horizon_str)
        logger.info(f"[Single-Horizon] Training with horizon: {horizons}")
    
    patches, targets, cond = _build_arrays(
        candles, args.patch_len, horizons, args.warmup_bars
    )
    train_x, val_x, mu, sigma, scaler_metadata = _standardize_train_val(
        patches, args.val_frac
    )

    # Log scaler metadata for debugging
    logger.info(
        f"[Scaler] Train size: {scaler_metadata['train_size']}, Val size: {scaler_metadata['val_size']}"
    )
    if scaler_metadata.get("ks_test_p_value") is not None:
        logger.info(
            f"[Scaler] KS test p-value: {scaler_metadata['ks_test_p_value']:.4f} (< 0.5 expected for no bias)"
        )

    # Split targets and conditions based on train_size
    train_size = scaler_metadata["train_size"]
    train_y = targets[:train_size]
    val_y = targets[train_size:]
    cond_train = cond[:train_size] if cond is not None else None
    cond_val = cond[train_size:] if cond is not None else None

    train_ds = CandlePatchDataset(train_x, train_y, cond_train)
    val_ds = CandlePatchDataset(val_x, val_y, cond_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True,
    )

    model = ForexDiffusionLit()
    model.dataset_stats = {
        "channel_order": CHANNEL_ORDER,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "scaler_metadata": scaler_metadata,  # NEW: metadata for bias verification
    }
    try:
        # Save horizons info
        if isinstance(horizons, int):
            horizon_meta = horizons
            num_horizons = 1
        else:
            horizon_meta = horizons
            num_horizons = len(horizons)
        
        model.save_hyperparameters(
            {
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "horizon": horizon_meta,  # Can be int or list
                "horizons": horizons if isinstance(horizons, list) else [horizons],
                "num_horizons": num_horizons,
                "patch_len": args.patch_len,
            }
        )
    except Exception:
        pass

    out_dir = Path(args.artifacts_dir).resolve() / "lightning"
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = ModelCheckpoint(
        dirpath=out_dir,
        monitor="val/loss",
        save_top_k=3,
        mode="min",
        filename=f"{args.symbol.replace('/', '')}-{args.timeframe}-{{epoch:02d}}-{{val/loss:.4f}}",
    )
    early = EarlyStopping(monitor="val/loss", patience=10, mode="min", verbose=True)
    lr_mon = LearningRateMonitor(logging_interval="epoch")

    # Build callbacks list
    callbacks = [monitor, early, lr_mon]

    # Add NVIDIA optimization callback if requested
    if args.use_nvidia_opts or args.use_amp or args.compile_model:
        from .optimized_trainer import OptimizedTrainingCallback
        from .optimization_config import (
            OptimizationConfig,
            HardwareInfo,
            PrecisionMode,
            CompileMode,
        )

        # Detect hardware capabilities
        hw_info = HardwareInfo.detect()

        # Map precision string to enum
        precision_map = {
            "fp16": PrecisionMode.FP16,
            "bf16": PrecisionMode.BF16,
            "fp32": PrecisionMode.FP32,
        }

        # Create optimization config
        opt_config = OptimizationConfig(
            hardware_info=hw_info,
            use_amp=args.use_amp or args.use_nvidia_opts,
            precision=precision_map.get(args.precision, PrecisionMode.FP16),
            compile_model=args.compile_model or args.use_nvidia_opts,
            compile_mode=CompileMode.DEFAULT,
            use_fused_optimizer=True,  # Always use APEX fused optimizer if available (5-15% speedup)
            use_flash_attention=args.use_flash_attention,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_channels_last=True,  # Generally beneficial for CNNs
            use_gradient_checkpointing=False,  # Can be enabled for very large models
        )

        # Create and add optimization callback
        opt_callback = OptimizedTrainingCallback(opt_config)
        callbacks.append(opt_callback)

        logger.info("🚀 NVIDIA Optimization Stack ENABLED")
        logger.info(f"   - GPU: {hw_info.gpu_name if hw_info.has_cuda else 'None'}")
        logger.info(f"   - AMP: {opt_config.use_amp} ({opt_config.precision.value})")
        logger.info(f"   - torch.compile: {opt_config.compile_model}")
        logger.info(f"   - Fused optimizer: {opt_config.use_fused_optimizer}")
        logger.info(f"   - Flash Attention: {opt_config.use_flash_attention}")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=str(out_dir),
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
    )

    logger.info(
        "[train] starting Lightning fit on {} batches (val {})",
        len(train_loader),
        len(val_loader),
    )
    trainer.fit(model, train_loader, val_loader)

    ckpt_path = monitor.best_model_path or (out_dir / "last.ckpt")
    if not monitor.best_model_path:
        trainer.save_checkpoint(ckpt_path)
    logger.info("[train] best checkpoint: {}", ckpt_path)

    sidecar = Path(ckpt_path).with_suffix(Path(ckpt_path).suffix + ".meta.json")
    payload = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "horizon_bars": horizons if isinstance(horizons, list) else horizons,
        "horizons": horizons if isinstance(horizons, list) else [horizons],
        "num_horizons": len(horizons) if isinstance(horizons, list) else 1,
        "patch_len": args.patch_len,
        "channel_order": CHANNEL_ORDER,
        "mu": model.dataset_stats["mu"],
        "sigma": model.dataset_stats["sigma"],
        "indicator_tfs": coerce_indicator_tfs(args.indicator_tfs),
        "warmup_bars": args.warmup_bars,
        "rv_window": args.rv_window,
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
