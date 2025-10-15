"""
SSSD Training Pipeline

Complete training pipeline for SSSD model including:
- Data loading from database
- Feature engineering via UnifiedFeaturePipeline
- Training loop with gradient clipping
- Validation and early stopping
- Checkpointing (best + periodic)
- Logging (TensorBoard/WandB)
- Mixed precision training
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
from tqdm import tqdm

from ..models.sssd import SSSDModel
from ..data.sssd_dataset import SSSDDataModule, collate_fn
from ..config.sssd_config import load_sssd_config, SSSDConfig


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self, patience: int = 15, min_delta: float = 0.0001, mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss if self.mode == "min" else val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


class SSSDTrainer:
    """SSSD Training Manager."""

    def __init__(self, config: SSSDConfig):
        self.config = config
        self.device = torch.device(config.system.device)

        # Initialize model
        self.model = SSSDModel(config).to(self.device)

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.optimizer.learning_rate,
            weight_decay=config.training.optimizer.weight_decay,
            betas=config.training.optimizer.betas,
            eps=config.training.optimizer.eps,
        )

        # Learning rate scheduler
        self.scheduler = None  # Will be initialized after data module

        # Mixed precision
        self.use_amp = config.training.mixed_precision.enabled
        self.scaler = GradScaler() if self.use_amp else None

        # Early stopping
        self.early_stopping = (
            EarlyStopping(
                patience=config.training.early_stopping.patience,
                min_delta=config.training.early_stopping.min_delta,
                mode="min",
            )
            if config.training.early_stopping.enabled
            else None
        )

        # Logging
        self.tensorboard_writer = None
        if config.training.logging.tensorboard:
            log_dir = Path(config.system.tensorboard_dir) / config.model.name
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir)

        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        # Directories
        self.checkpoint_dir = Path(config.system.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized SSSDTrainer: model_params={self.model.get_num_params():,}, "
            f"model_size={self.model.get_model_size_mb():.2f}MB, "
            f"device={self.device}, amp={self.use_amp}"
        )

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            features, targets, horizons = batch
            features = {k: v.to(self.device) for k, v in features.items()}
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                loss = self.model.training_step((features, targets, horizons))

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                if self.config.training.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.training.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip_norm
                    )
                self.optimizer.step()

            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Log to TensorBoard
            if (
                self.tensorboard_writer
                and self.global_step % self.config.training.logging.log_every_n_steps
                == 0
            ):
                self.tensorboard_writer.add_scalar(
                    "train/loss", loss.item(), self.global_step
                )
                self.tensorboard_writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )

        avg_loss = epoch_loss / num_batches
        return {"loss": avg_loss}

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            # Move batch to device
            features, targets, horizons = batch
            features = {k: v.to(self.device) for k, v in features.items()}
            targets = targets.to(self.device)

            # Forward pass
            with autocast(enabled=self.use_amp):
                loss = self.model.training_step((features, targets, horizons))

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        return {"loss": avg_loss}

    def save_checkpoint(self, is_best: bool = False, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_epoch_{self.current_epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save checkpoint
        self.model.save_checkpoint(
            path=checkpoint_path,
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict(),
            metrics=metrics or {},
        )

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            self.model.save_checkpoint(
                path=best_path,
                epoch=self.current_epoch,
                optimizer_state=self.optimizer.state_dict(),
                metrics=metrics or {},
            )
            logger.info(
                f"Saved best model with val_loss={metrics.get('val_loss', 'N/A'):.6f}"
            )

        # Clean up old checkpoints if keep_best_only
        if self.config.training.checkpoint.keep_best_only and not is_best:
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    def train(self, data_module: SSSDDataModule, resume_from: Optional[str] = None):
        """
        Full training loop.

        Args:
            data_module: SSSDDataModule with train/val/test dataloaders
            resume_from: Optional checkpoint path to resume from
        """
        # Resume from checkpoint if provided
        if resume_from is not None:
            self._load_checkpoint(resume_from)

        # Setup data module
        if data_module.train_dataset is None:
            data_module.setup()

        train_loader = data_module.train_dataloader(collate_fn=collate_fn)
        val_loader = data_module.val_dataloader(collate_fn=collate_fn)

        # Initialize scheduler (needs total steps)
        total_steps = self.config.training.epochs * len(train_loader)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.training.scheduler.lr_min,
        )

        logger.info(
            f"Starting training: epochs={self.config.training.epochs}, "
            f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
        )

        # Training loop
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch}: train_loss={train_metrics['loss']:.6f}")

            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f"Epoch {epoch}: val_loss={val_metrics['loss']:.6f}")

            # Log to TensorBoard
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(
                    "epoch/train_loss", train_metrics["loss"], epoch
                )
                self.tensorboard_writer.add_scalar(
                    "epoch/val_loss", val_metrics["loss"], epoch
                )

            # Check for best model
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            # Save checkpoint
            if (
                epoch + 1
            ) % self.config.training.checkpoint.save_every_n_epochs == 0 or is_best:
                metrics = {
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "epoch": epoch,
                }
                self.save_checkpoint(is_best=is_best, metrics=metrics)

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics["loss"]):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            # Empty cache periodically
            if (epoch + 1) % self.config.system.empty_cache_every_n_epochs == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final checkpoint
        final_metrics = {
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
        }
        final_path = self.checkpoint_dir / "final_model.pt"
        self.model.save_checkpoint(
            path=final_path,
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict(),
            metrics=final_metrics,
        )

        logger.info(
            f"Training complete: best_val_loss={self.best_val_loss:.6f}, "
            f"final_epoch={self.current_epoch}"
        )

        if self.tensorboard_writer:
            self.tensorboard_writer.close()

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        model, epoch, optimizer_state, metrics = SSSDModel.load_checkpoint(
            checkpoint_path, config=self.config, map_location=str(self.device)
        )

        self.model = model.to(self.device)
        self.current_epoch = epoch + 1

        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        self.best_val_loss = metrics.get("val_loss", float("inf"))

        logger.info(
            f"Resumed from epoch {epoch}, best_val_loss={self.best_val_loss:.6f}"
        )


def train_sssd_cli(
    config_path: str,
    asset_config_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Main entry point for SSSD training CLI.

    Args:
        config_path: Path to main SSSD config (e.g., configs/sssd/default_config.yaml)
        asset_config_path: Optional asset-specific config
        resume_from: Optional checkpoint path to resume training
        overrides: Optional config overrides (dict)
    """
    # Load configuration
    config = load_sssd_config(config_path, asset_config_path, overrides)

    logger.info(f"Training SSSD for {config.model.asset}")
    logger.info(f"Config: {config.model.name}")

    # Set random seeds
    if config.system.deterministic:
        torch.manual_seed(config.system.seed)
        np.random.seed(config.system.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.system.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Create data module (will load from path specified in config)
    data_module = SSSDDataModule(
        data_path=Path(config.system.checkpoint_dir).parent / "features",
        config=config,
        feature_pipeline=None,
    )

    # Create trainer
    trainer = SSSDTrainer(config)

    # Train
    trainer.train(data_module, resume_from=resume_from)

    logger.info("SSSD training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SSSD model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--asset-config", type=str, help="Path to asset-specific config"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--override", type=str, nargs="+", help="Config overrides (key=value)"
    )

    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    if args.override:
        for override in args.override:
            key, value = override.split("=")
            try:
                value = eval(value)
            except:
                pass
            overrides[key] = value

    # Train
    train_sssd_cli(
        config_path=args.config,
        asset_config_path=args.asset_config,
        resume_from=args.resume,
        overrides=overrides,
    )
