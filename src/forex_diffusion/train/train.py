"""
Training entrypoint for MagicForex.

- Instantiates LightningModule (ForexDiffusionLit) and runs training.
- If no dataset available, uses a synthetic RandomDataset for smoke training.
- Saves checkpoints in artifacts dir configured via get_config().
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from loguru import logger

from ..utils.config import get_config
from .loop import ForexDiffusionLit


class RandomDataset(Dataset):
    """Synthetic dataset emitting random patches and a simple target close value."""
    def __init__(self, num_samples: int = 1024, in_channels: int = 6, patch_len: int = 64):
        super().__init__()
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.patch_len = patch_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.in_channels, self.patch_len).float()
        # synthetic target: last close approximated as normal around 0
        y = torch.randn(1).float()
        return {"x": x, "y": y}


def build_dataloader(batch_size: int = 32, num_workers: int = 4, dataset: Optional[Dataset] = None):
    ds = dataset or RandomDataset(num_samples=2048, in_channels=6, patch_len=64)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use (0 for CPU)")
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--log_dir", default=None)
    return p.parse_args()


def main():
    cfg = get_config()
    args = parse_args()

    # Instantiate model
    model = ForexDiffusionLit(cfg=cfg)
"""
Training entrypoint for MagicForex.

- Instantiates LightningModule (ForexDiffusionLit) and runs training.
- If no dataset available, uses a synthetic RandomDataset for smoke training.
- Saves checkpoints in artifacts dir configured via get_config().
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from loguru import logger

from ..utils.config import get_config
from .loop import ForexDiffusionLit


class RandomDataset(Dataset):
    """Synthetic dataset emitting random patches and a simple target close value."""
    def __init__(self, num_samples: int = 1024, in_channels: int = 6, patch_len: int = 64):
        super().__init__()
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.patch_len = patch_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.in_channels, self.patch_len).float()
        # synthetic target: last close approximated as normal around 0
        y = torch.randn(1).float()
        return {"x": x, "y": y}


def build_dataloader(batch_size: int = 32, num_workers: int = 4, dataset: Optional[Dataset] = None):
    ds = dataset or RandomDataset(num_samples=2048, in_channels=6, patch_len=64)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use (0 for CPU)")
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--log_dir", default=None)
    return p.parse_args()


def main():
    cfg = get_config()
    args = parse_args()

    # Instantiate model
    model = ForexDiffusionLit(cfg=cfg)

    # DataLoaders
    train_dl = build_dataloader(batch_size=args.batch_size, num_workers=int(cfg.training.num_workers if hasattr(cfg, "training") else 4))
    val_dl = build_dataloader(batch_size=args.batch_size, num_workers=0, dataset=None)

    # Callbacks
    artifacts_dir = getattr(cfg.model, "artifacts_dir", "./artifacts/models") if hasattr(cfg, "model") else "./artifacts/models"
    os.makedirs(artifacts_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=artifacts_dir, save_top_k=3, monitor="val/crps", mode="min", filename="magicforex-{epoch:02d}-{val/crps:.4f}")
    early_stop = EarlyStopping(monitor="val/crps", patience=12, mode="min", verbose=True)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=args.log_dir or artifacts_dir,
        callbacks=[ckpt_cb, early_stop],
        accelerator="auto",
        devices=args.gpus if args.gpus > 0 else None,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )

    logger.info("Starting training: epochs=%s, batch_size=%s", args.max_epochs, args.batch_size)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
    # DataLoaders
    train_dl = build_dataloader(batch_size=args.batch_size, num_workers=int(cfg.training.num_workers if hasattr(cfg, "training") else 4))
    val_dl = build_dataloader(batch_size=args.batch_size, num_workers=0, dataset=None)

    # Callbacks
    artifacts_dir = getattr(cfg.model, "artifacts_dir", "./artifacts/models") if hasattr(cfg, "model") else "./artifacts/models"
    os.makedirs(artifacts_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=artifacts_dir, save_top_k=3, monitor="val/crps", mode="min", filename="magicforex-{epoch:02d}-{val/crps:.4f}")
    early_stop = EarlyStopping(monitor="val/crps", patience=12, mode="min", verbose=True)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=args.log_dir or artifacts_dir,
        callbacks=[ckpt_cb, early_stop],
        accelerator="auto",
        devices=args.gpus if args.gpus > 0 else None,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )

    logger.info("Starting training: epochs=%s, batch_size=%s", args.max_epochs, args.batch_size)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
