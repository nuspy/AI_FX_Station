"""
Optimized training wrapper with NVIDIA optimization stack.

Integrates all GPU optimizations (AMP, compile, fused optimizers, etc.)
with existing PyTorch Lightning training loop.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from loguru import logger

from .optimization_config import OptimizationConfig


class OptimizedTrainingCallback(Callback):
    """
    PyTorch Lightning callback that applies NVIDIA optimizations.

    Handles:
    - Mixed precision setup (AMP)
    - torch.compile integration
    - Channels last memory format
    - Gradient accumulation
    - Fused optimizer replacement
    """

    def __init__(self, opt_config: OptimizationConfig):
        super().__init__()
        self.opt_config = opt_config
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.compiled_model: Optional[nn.Module] = None

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Setup optimizations before training starts."""
        if stage != "fit":
            return

        logger.info("=== Applying NVIDIA Optimizations ===")

        # 1. Apply cuDNN settings
        if self.opt_config.hardware_info.has_cuda:
            torch.backends.cudnn.benchmark = self.opt_config.cudnn_benchmark
            torch.backends.cudnn.deterministic = self.opt_config.cudnn_deterministic
            logger.info(f"cuDNN benchmark: {self.opt_config.cudnn_benchmark}")
            logger.info(f"cuDNN deterministic: {self.opt_config.cudnn_deterministic}")

        # 2. Setup AMP scaler (if using mixed precision)
        if self.opt_config.use_amp and self.opt_config.hardware_info.has_cuda:
            if self.opt_config.precision.value == "bf16":
                # BF16 doesn't need gradient scaling
                logger.info("Using BF16 mixed precision (no GradScaler needed)")
            else:
                # FP16 requires gradient scaling
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("Using FP16 mixed precision with GradScaler")

        # 3. Apply channels_last memory format
        if self.opt_config.use_channels_last and self.opt_config.hardware_info.has_cuda:
            try:
                # Convert model to channels_last
                # Note: This is primarily beneficial for CNN models
                pl_module = pl_module.to(memory_format=torch.channels_last)
                logger.info("Applied channels_last memory format")
            except Exception as e:
                logger.warning(f"Could not apply channels_last: {e}")

        # 4. Apply gradient checkpointing
        if self.opt_config.use_gradient_checkpointing:
            try:
                # Enable gradient checkpointing for supported modules
                # This trades compute for memory
                if hasattr(pl_module, "vae") and hasattr(pl_module.vae, "encoder"):
                    # Apply to VAE encoder if it has gradient_checkpointing method
                    if hasattr(pl_module.vae.encoder, "gradient_checkpointing_enable"):
                        pl_module.vae.encoder.gradient_checkpointing_enable()
                        logger.info("Enabled gradient checkpointing for VAE encoder")
                if hasattr(pl_module, "diffusion_model"):
                    if hasattr(
                        pl_module.diffusion_model, "gradient_checkpointing_enable"
                    ):
                        pl_module.diffusion_model.gradient_checkpointing_enable()
                        logger.info(
                            "Enabled gradient checkpointing for diffusion model"
                        )
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")

        # 5. Apply torch.compile (PyTorch 2.0+)
        if self.opt_config.compile_model:
            try:
                from torch import _dynamo

                _dynamo.config.suppress_errors = True

                # Compile individual modules for better flexibility
                if hasattr(pl_module, "vae"):
                    pl_module.vae.encoder = torch.compile(
                        pl_module.vae.encoder, mode=self.opt_config.compile_mode.value
                    )
                    pl_module.vae.decoder = torch.compile(
                        pl_module.vae.decoder, mode=self.opt_config.compile_mode.value
                    )
                    logger.info(
                        f"Compiled VAE with mode: {self.opt_config.compile_mode.value}"
                    )

                if hasattr(pl_module, "diffusion_model"):
                    pl_module.diffusion_model = torch.compile(
                        pl_module.diffusion_model,
                        mode=self.opt_config.compile_mode.value,
                    )
                    logger.info(
                        f"Compiled diffusion model with mode: {self.opt_config.compile_mode.value}"
                    )

            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
                logger.warning("Continuing without compilation")

        # 6. Replace optimizer with fused version (APEX)
        if (
            self.opt_config.use_fused_optimizer
            and self.opt_config.hardware_info.has_apex
        ):
            try:
                import apex.optimizers as apex_optim

                # Get current optimizer config
                opt_config = trainer.optimizers[0].defaults
                lr = opt_config.get("lr", 2e-4)
                weight_decay = opt_config.get("weight_decay", 1e-6)

                # Replace with fused AdamW
                fused_opt = apex_optim.FusedAdam(
                    pl_module.parameters(), lr=lr, weight_decay=weight_decay
                )

                # Replace optimizer in trainer
                trainer.optimizers = [fused_opt]
                logger.info("Replaced optimizer with APEX FusedAdam")

            except ImportError:
                logger.warning("APEX not available, using standard optimizer")
            except Exception as e:
                logger.warning(f"Could not apply fused optimizer: {e}")

        logger.info("=== Optimization Setup Complete ===")
        logger.info(
            f"Effective batch size: {self.opt_config.get_effective_batch_size()}"
        )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Handle per-batch optimizations."""
        # Convert batch to channels_last if needed
        if self.opt_config.use_channels_last and self.opt_config.hardware_info.has_cuda:
            if isinstance(batch, dict) and "x" in batch:
                try:
                    batch["x"] = batch["x"].to(memory_format=torch.channels_last)
                except:
                    pass


class OptimizedDataLoader:
    """
    Factory for creating optimized DataLoaders.

    Applies:
    - Optimized num_workers based on CPU cores
    - Pin memory for GPU training
    - Prefetch factor tuning
    """

    @staticmethod
    def create(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        opt_config: OptimizationConfig,
        shuffle: bool = True,
        is_validation: bool = False,
    ) -> DataLoader:
        """
        Create an optimized DataLoader.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            opt_config: Optimization configuration
            shuffle: Whether to shuffle data
            is_validation: If True, use fewer workers and no shuffle

        Returns:
            Optimized DataLoader
        """
        # Determine num_workers
        num_workers = opt_config.num_workers
        if is_validation:
            # Use fewer workers for validation
            num_workers = max(0, num_workers // 2)
            shuffle = False

        # Pin memory for GPU training
        pin_memory = opt_config.pin_memory and opt_config.hardware_info.has_cuda

        # Prefetch factor (only if num_workers > 0)
        prefetch_factor = 2 if num_workers > 0 else None

        # Persistent workers for faster epoch transitions
        persistent_workers = num_workers > 0

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            drop_last=(
                True if shuffle else False
            ),  # Drop last incomplete batch in training
        )

        logger.info(
            f"Created {'validation' if is_validation else 'training'} DataLoader: "
            f"batch_size={batch_size}, workers={num_workers}, "
            f"pin_memory={pin_memory}, persistent={persistent_workers}"
        )

        return loader


def create_optimized_trainer(
    opt_config: OptimizationConfig,
    max_epochs: int,
    output_dir: Path,
    callbacks: list = None,
    **trainer_kwargs,
) -> pl.Trainer:
    """
    Create a PyTorch Lightning Trainer with all optimizations applied.

    Args:
        opt_config: Optimization configuration
        max_epochs: Maximum number of epochs
        output_dir: Directory for checkpoints and logs
        callbacks: Additional callbacks (will be extended with optimization callback)
        **trainer_kwargs: Additional arguments for pl.Trainer

    Returns:
        Configured PyTorch Lightning Trainer
    """
    # Log hardware and configuration
    opt_config.hardware_info.log_hardware_info()
    opt_config.log_config()

    # Create optimization callback
    opt_callback = OptimizedTrainingCallback(opt_config)

    # Combine callbacks
    all_callbacks = [opt_callback]
    if callbacks:
        all_callbacks.extend(callbacks)

    # Determine precision string for Lightning
    if opt_config.use_amp and opt_config.hardware_info.has_cuda:
        if opt_config.precision.value == "bf16":
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
    else:
        precision = "32-true"

    # Determine devices and strategy
    if opt_config.use_ddp and opt_config.num_gpus > 1:
        # Multi-GPU DDP
        devices = opt_config.num_gpus
        strategy = "ddp"
        logger.info(f"Using DDP strategy with {devices} GPUs")
    elif opt_config.hardware_info.has_cuda and opt_config.num_gpus >= 1:
        # Single GPU
        devices = 1
        strategy = "auto"
        logger.info("Using single GPU")
    else:
        # CPU
        devices = "auto"
        strategy = "auto"
        logger.info("Using CPU")

    # Gradient accumulation
    accumulate_grad_batches = opt_config.gradient_accumulation_steps
    if accumulate_grad_batches > 1:
        logger.info(f"Gradient accumulation: {accumulate_grad_batches} steps")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir=str(output_dir),
        callbacks=all_callbacks,
        accelerator="gpu" if opt_config.hardware_info.has_cuda else "cpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        **trainer_kwargs,
    )

    logger.info("=== Trainer Configuration ===")
    logger.info(f"Max epochs: {max_epochs}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Devices: {devices}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Gradient accumulation: {accumulate_grad_batches}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 50)

    return trainer


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    opt_config: OptimizationConfig,
    seconds_per_batch_baseline: float = 0.5,
) -> Dict[str, float]:
    """
    Estimate training time with and without optimizations.

    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        num_epochs: Number of epochs
        opt_config: Optimization configuration
        seconds_per_batch_baseline: Baseline seconds per batch (single GPU, no optimizations)

    Returns:
        Dictionary with time estimates
    """
    # Calculate number of batches
    batches_per_epoch = num_samples // batch_size
    total_batches = batches_per_epoch * num_epochs

    # Baseline time (no optimizations)
    baseline_time_seconds = total_batches * seconds_per_batch_baseline

    # Estimate speedup from optimizations
    speedup = 1.0

    # Mixed precision: ~2.5x speedup for FP16, ~2.0x for BF16
    if opt_config.use_amp:
        if opt_config.precision.value == "bf16":
            speedup *= 2.0
        else:
            speedup *= 2.5

    # torch.compile: ~1.8x speedup
    if opt_config.compile_model:
        speedup *= 1.8

    # DDP multi-GPU: ~(num_gpus * 0.85) speedup (85% efficiency)
    if opt_config.use_ddp and opt_config.num_gpus > 1:
        speedup *= opt_config.num_gpus * 0.85

    # Fused optimizer: ~1.15x speedup
    if opt_config.use_fused_optimizer:
        speedup *= 1.15

    # Flash Attention: ~1.3x speedup (if applicable)
    if opt_config.use_flash_attention:
        speedup *= 1.3

    # DALI DataLoader: ~1.2x speedup
    if opt_config.use_dali:
        speedup *= 1.2

    # Optimized time
    optimized_time_seconds = baseline_time_seconds / speedup

    return {
        "baseline_hours": baseline_time_seconds / 3600,
        "optimized_hours": optimized_time_seconds / 3600,
        "speedup": speedup,
        "time_saved_hours": (baseline_time_seconds - optimized_time_seconds) / 3600,
        "batches_per_epoch": batches_per_epoch,
        "total_batches": total_batches,
    }
