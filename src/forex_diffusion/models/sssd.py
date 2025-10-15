"""
SSSD Model (Structured State Space Diffusion)

Main model class combining:
- Multi-scale encoder (S4 layers for multiple timeframes)
- Diffusion head (noise predictor)
- Horizon embeddings (multi-horizon forecasting)
- Diffusion scheduler (training and sampling)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Literal
from pathlib import Path
from loguru import logger

from .sssd_encoder import MultiScaleEncoder
from .diffusion_head import DiffusionHead
from .diffusion_scheduler import CosineNoiseScheduler
from ..config.sssd_config import SSSDConfig


class SSSDModel(nn.Module):
    """
    SSSD (Structured State Space Diffusion) Model.

    Combines multi-timeframe S4 encoding with diffusion-based forecasting
    for probabilistic multi-horizon predictions.
    """

    def __init__(self, config: SSSDConfig):
        """
        Initialize SSSD model.

        Args:
            config: SSSDConfig object with all model configuration
        """
        super().__init__()

        self.config = config

        # Multi-scale encoder (S4 for each timeframe)
        self.encoder = MultiScaleEncoder(
            timeframes=config.model.encoder.timeframes,
            feature_dim=config.model.encoder.feature_dim,
            s4_state_dim=config.model.s4.state_dim,
            s4_layers=config.model.s4.n_layers,
            s4_dropout=config.model.s4.dropout,
            context_dim=config.model.encoder.context_dim,
            attention_heads=config.model.encoder.attention_heads,
            attention_dropout=config.model.encoder.attention_dropout
        )

        # Horizon embeddings (learnable embeddings for each forecast horizon)
        self.horizon_embeddings = nn.Embedding(
            num_embeddings=len(config.model.horizons.minutes),
            embedding_dim=config.model.horizons.embedding_dim
        )

        # Diffusion head (noise predictor)
        # Conditioning dim = context_dim + horizon_embedding_dim
        conditioning_dim = (
            config.model.encoder.context_dim +
            config.model.horizons.embedding_dim
        )

        self.diffusion_head = DiffusionHead(
            latent_dim=config.model.head.latent_dim,
            timestep_emb_dim=config.model.head.timestep_emb_dim,
            conditioning_dim=conditioning_dim,
            mlp_hidden_dims=config.model.head.mlp_hidden_dims,
            dropout=config.model.head.dropout,
            output_dim=config.model.head.latent_dim
        )

        # Diffusion scheduler
        self.scheduler = CosineNoiseScheduler(
            T=config.model.diffusion.steps_train,
            s=config.model.diffusion.schedule_offset,
            clip_min=config.model.diffusion.clip_min,
            device=config.system.device
        )

        # Target projection (project to 1D for price change prediction)
        self.target_proj = nn.Linear(config.model.head.latent_dim, 1)

        # Horizon mapping (minutes to index)
        self.horizon_to_idx = {
            h: i for i, h in enumerate(config.model.horizons.minutes)
        }

        logger.info(
            f"Initialized SSSDModel: "
            f"horizons={config.model.horizons.minutes}, "
            f"context_dim={config.model.encoder.context_dim}, "
            f"latent_dim={config.model.head.latent_dim}"
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        horizons: List[int],
        targets: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (used during training).

        Args:
            features: Dict of features per timeframe
                {"5m": (batch, seq_len, feature_dim), ...}
            horizons: List of horizon indices to predict
            targets: Optional ground truth targets (batch, num_horizons, 1)
            timesteps: Optional diffusion timesteps (batch,)
                If None, sample random timesteps

        Returns:
            Dict with 'predictions' and 'loss' (if targets provided)
        """
        batch_size = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device

        # Encode multi-scale context
        context = self.encoder(features)  # (batch, context_dim)

        # Process each horizon
        predictions = []
        losses = []

        for h_idx in horizons:
            # Get horizon embedding
            h_tensor = torch.tensor([h_idx], device=device).expand(batch_size)
            h_emb = self.horizon_embeddings(h_tensor)  # (batch, embedding_dim)

            # Conditioning: context + horizon embedding
            conditioning = torch.cat([context, h_emb], dim=-1)  # (batch, conditioning_dim)

            if targets is not None and timesteps is not None:
                # Training mode: predict noise
                # Get target for this horizon
                target = targets[:, horizons.index(h_idx), :]  # (batch, 1)

                # Project target to latent space
                target_latent = target  # For now, assume target is already in latent space
                # In practice, you might want: target_latent = self.target_encoder(target)

                # Add noise to target
                noisy_target, noise = self.scheduler.add_noise(
                    target_latent, timesteps
                )  # (batch, latent_dim), (batch, latent_dim)

                # Predict noise
                predicted_noise = self.diffusion_head(
                    noisy_latent=noisy_target,
                    timesteps=timesteps,
                    conditioning=conditioning
                )  # (batch, latent_dim)

                # Compute loss
                loss = nn.functional.mse_loss(predicted_noise, noise)
                losses.append(loss)

                # For predictions, denoise completely (not used during training)
                pred = None

            else:
                # Inference mode: not implemented in forward()
                # Use inference_forward() instead
                raise NotImplementedError(
                    "Use inference_forward() for inference mode"
                )

        # Return results
        result = {}

        if targets is not None:
            # Weighted loss across horizons
            horizon_weights = torch.tensor(
                self.config.model.horizons.weights,
                device=device
            )

            # Weight losses by horizon importance
            weighted_losses = [
                losses[i] * horizon_weights[h_idx]
                for i, h_idx in enumerate(horizons)
            ]
            total_loss = sum(weighted_losses)

            # Consistency loss (optional)
            if self.config.model.horizons.consistency_weight > 0:
                # TODO: Add consistency loss between horizons
                # Penalize contradictory predictions
                pass

            result['loss'] = total_loss

        return result

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], torch.Tensor, List[int]]
    ) -> torch.Tensor:
        """
        Training step (called by training loop).

        Args:
            batch: Tuple of (features, targets, horizons)
                features: Dict[timeframe -> (batch, seq_len, feature_dim)]
                targets: (batch, num_horizons, 1)
                horizons: List of horizon indices

        Returns:
            loss: Scalar loss tensor
        """
        features, targets, horizons = batch
        batch_size = targets.shape[0]
        device = targets.device

        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.scheduler.T, (batch_size,), device=device
        )

        # Forward pass
        result = self.forward(
            features=features,
            horizons=horizons,
            targets=targets,
            timesteps=timesteps
        )

        return result['loss']

    @torch.no_grad()
    def inference_forward(
        self,
        features: Dict[str, torch.Tensor],
        horizons: List[int],
        num_samples: int = 100,
        sampler: Literal["ddim", "ddpm", "dpmpp"] = "ddim",
        num_steps: Optional[int] = None
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Inference mode: generate samples via diffusion sampling.

        Args:
            features: Dict of features per timeframe
            horizons: List of horizon indices to predict
            num_samples: Number of samples for uncertainty quantification
            sampler: Sampling algorithm (ddim, ddpm, dpmpp)
            num_steps: Number of denoising steps (default from config)

        Returns:
            Dict mapping horizon_idx to statistics:
                {
                    0: {"mean": tensor, "std": tensor, "q05": tensor, "q50": tensor, "q95": tensor},
                    1: {...},
                    ...
                }
        """
        batch_size = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device

        if num_steps is None:
            num_steps = self.config.model.diffusion.steps_inference

        # Get sampling timesteps
        timesteps = self.scheduler.get_sampling_timesteps(num_steps)

        # Encode context (shared across all horizons and samples)
        context = self.encoder(features)  # (batch, context_dim)

        # Generate predictions for each horizon
        predictions = {}

        for h_idx in horizons:
            # Get horizon embedding
            h_tensor = torch.tensor([h_idx], device=device).expand(batch_size)
            h_emb = self.horizon_embeddings(h_tensor)  # (batch, embedding_dim)

            # Conditioning
            conditioning = torch.cat([context, h_emb], dim=-1)

            # Generate multiple samples
            samples = []

            for _ in range(num_samples):
                # Start from pure noise
                z = torch.randn(
                    batch_size, self.config.model.head.latent_dim,
                    device=device
                )

                # Denoise iteratively
                for i in range(len(timesteps) - 1):
                    t_curr = timesteps[i]
                    t_next = timesteps[i + 1]

                    # Current timestep (broadcast to batch)
                    t_curr_batch = torch.full(
                        (batch_size,), t_curr, device=device, dtype=torch.long
                    )

                    # Predict noise
                    predicted_noise = self.diffusion_head(
                        noisy_latent=z,
                        timesteps=t_curr_batch,
                        conditioning=conditioning
                    )

                    # Denoise one step
                    if sampler == "ddim":
                        t_next_batch = torch.full(
                            (batch_size,), t_next, device=device, dtype=torch.long
                        )
                        z = self.scheduler.step_ddim(
                            z, t_curr_batch, t_next_batch, predicted_noise,
                            eta=self.config.inference.ddim_eta
                        )
                    elif sampler == "ddpm":
                        z = self.scheduler.step_ddpm(
                            z, t_curr_batch, predicted_noise
                        )
                    else:
                        raise ValueError(f"Unknown sampler: {sampler}")

                # Project to target space (price change)
                prediction = self.target_proj(z)  # (batch, 1)
                samples.append(prediction)

            # Stack samples
            samples = torch.stack(samples, dim=0)  # (num_samples, batch, 1)

            # Compute statistics
            mean = samples.mean(dim=0)  # (batch, 1)
            std = samples.std(dim=0)    # (batch, 1)
            q05 = torch.quantile(samples, 0.05, dim=0)  # (batch, 1)
            q50 = torch.quantile(samples, 0.50, dim=0)  # (batch, 1)
            q95 = torch.quantile(samples, 0.95, dim=0)  # (batch, 1)

            predictions[h_idx] = {
                "mean": mean,
                "std": std,
                "q05": q05,
                "q50": q50,
                "q95": q95,
                "samples": samples
            }

        return predictions

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        optimizer_state: Optional[Dict] = None,
        metrics: Optional[Dict] = None
    ):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer_state: Optimizer state dict (optional)
            metrics: Training metrics (optional)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics or {}
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        config: Optional[SSSDConfig] = None,
        map_location: Optional[str] = None
    ) -> Tuple[SSSDModel, int, Optional[Dict], Dict]:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            config: Optional config (if None, load from checkpoint)
            map_location: Device to load to (e.g., 'cpu', 'cuda:0')

        Returns:
            (model, epoch, optimizer_state, metrics)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=map_location)

        # Load config
        if config is None:
            from ..config.sssd_config import SSSDConfig
            config = SSSDConfig.from_dict(checkpoint['config'])

        # Create model
        model = cls(config)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Extract metadata
        epoch = checkpoint['epoch']
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        metrics = checkpoint.get('metrics', {})

        logger.info(f"Loaded checkpoint from {path} (epoch {epoch})")

        return model, epoch, optimizer_state, metrics

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
