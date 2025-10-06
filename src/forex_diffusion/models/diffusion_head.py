"""
Diffusion Head for SSSD

Predicts noise (or velocity) given noisy latent, timestep, and conditioning.
Used in the reverse diffusion process to iteratively denoise predictions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Optional
from loguru import logger


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.

    Based on "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, embedding_dim: int, max_timesteps: int = 10000):
        """
        Initialize sinusoidal embedding.

        Args:
            embedding_dim: Embedding dimension (must be even)
            max_timesteps: Maximum number of timesteps to support
        """
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {embedding_dim}")

        self.embedding_dim = embedding_dim

        # Precompute embeddings
        position = torch.arange(max_timesteps).unsqueeze(1)  # (max_timesteps, 1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )  # (embedding_dim/2,)

        embeddings = torch.zeros(max_timesteps, embedding_dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('embeddings', embeddings)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for timesteps.

        Args:
            timesteps: Timestep indices (batch,) or (batch, 1)

        Returns:
            Embeddings (batch, embedding_dim)
        """
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)

        return self.embeddings[timesteps]


class DiffusionHead(nn.Module):
    """
    Diffusion head for noise prediction.

    Takes:
    - Noisy latent (z_t)
    - Diffusion timestep (t)
    - Conditioning (context from encoder + horizon embedding)

    Outputs:
    - Predicted noise (epsilon) or velocity (v)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        timestep_emb_dim: int = 128,
        conditioning_dim: int = 640,
        mlp_hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize diffusion head.

        Args:
            latent_dim: Dimension of noisy latent z_t
            timestep_emb_dim: Timestep embedding dimension
            conditioning_dim: Conditioning vector dimension
            mlp_hidden_dims: Hidden dimensions for MLP
            dropout: Dropout probability
            output_dim: Output dimension (default=latent_dim)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.timestep_emb_dim = timestep_emb_dim
        self.conditioning_dim = conditioning_dim
        self.output_dim = output_dim or latent_dim

        # Timestep embedding
        self.timestep_embedding = SinusoidalPositionalEmbedding(
            embedding_dim=timestep_emb_dim
        )

        # Input projection
        # Concatenate: noisy_latent + timestep_emb + conditioning
        input_dim = latent_dim + timestep_emb_dim + conditioning_dim

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))

        self.mlp = nn.Sequential(*layers)

        logger.info(
            f"Initialized DiffusionHead: latent_dim={latent_dim}, "
            f"timestep_emb_dim={timestep_emb_dim}, "
            f"conditioning_dim={conditioning_dim}, "
            f"output_dim={self.output_dim}, "
            f"mlp_layers={len(mlp_hidden_dims)}"
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through diffusion head.

        Args:
            noisy_latent: Noisy latent z_t (batch, latent_dim)
            timesteps: Diffusion timesteps (batch,) - integers in [0, T]
            conditioning: Conditioning vector (batch, conditioning_dim)

        Returns:
            predicted_noise: Predicted noise epsilon (batch, output_dim)
        """
        # Get timestep embeddings
        t_emb = self.timestep_embedding(timesteps)  # (batch, timestep_emb_dim)

        # Concatenate inputs
        # (batch, latent_dim + timestep_emb_dim + conditioning_dim)
        x = torch.cat([noisy_latent, t_emb, conditioning], dim=-1)

        # Pass through MLP
        predicted_noise = self.mlp(x)  # (batch, output_dim)

        return predicted_noise

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f'latent_dim={self.latent_dim}, '
            f'timestep_emb_dim={self.timestep_emb_dim}, '
            f'conditioning_dim={self.conditioning_dim}, '
            f'output_dim={self.output_dim}'
        )


class TemporalUNetHead(nn.Module):
    """
    Alternative diffusion head using U-Net architecture.

    More powerful than simple MLP but also more complex.
    Can be used for better quality at the cost of inference speed.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        timestep_emb_dim: int = 128,
        conditioning_dim: int = 640,
        base_channels: int = 128,
        channel_multipliers: list[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize U-Net diffusion head.

        Args:
            latent_dim: Dimension of noisy latent
            timestep_emb_dim: Timestep embedding dimension
            conditioning_dim: Conditioning dimension
            base_channels: Base number of channels
            channel_multipliers: Channel multipliers per level
            num_res_blocks: Number of residual blocks per level
            dropout: Dropout probability
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.timestep_emb_dim = timestep_emb_dim
        self.conditioning_dim = conditioning_dim

        # Timestep embedding
        self.timestep_embedding = SinusoidalPositionalEmbedding(
            embedding_dim=timestep_emb_dim
        )

        # Input projection
        input_dim = latent_dim + timestep_emb_dim + conditioning_dim
        self.input_proj = nn.Linear(input_dim, base_channels)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = base_channels

        for mult in channel_multipliers:
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(channels, out_channels, dropout)
                )
                channels = out_channels

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels, channels, dropout),
            ResidualBlock(channels, channels, dropout)
        )

        # Upsampling path
        self.up_blocks = nn.ModuleList()

        for mult in reversed(channel_multipliers):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                # Skip connection from downsampling
                self.up_blocks.append(
                    ResidualBlock(channels * 2, out_channels, dropout)
                )
                channels = out_channels

        # Output projection
        self.output_proj = nn.Linear(channels, latent_dim)

        logger.info(
            f"Initialized TemporalUNetHead: latent_dim={latent_dim}, "
            f"base_channels={base_channels}, levels={len(channel_multipliers)}"
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through U-Net head.

        Args:
            noisy_latent: Noisy latent (batch, latent_dim)
            timesteps: Timesteps (batch,)
            conditioning: Conditioning (batch, conditioning_dim)

        Returns:
            Predicted noise (batch, latent_dim)
        """
        # Get timestep embeddings
        t_emb = self.timestep_embedding(timesteps)

        # Concatenate and project
        x = torch.cat([noisy_latent, t_emb, conditioning], dim=-1)
        x = self.input_proj(x)  # (batch, base_channels)

        # Add sequence dimension for residual blocks
        x = x.unsqueeze(1)  # (batch, 1, channels)

        # Downsampling with skip connections
        skips = []
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling with skip connections
        for block in self.up_blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=-1)  # Concatenate along channel dim
            x = block(x)

        # Remove sequence dimension and project to output
        x = x.squeeze(1)  # (batch, channels)
        predicted_noise = self.output_proj(x)  # (batch, latent_dim)

        return predicted_noise


class ResidualBlock(nn.Module):
    """Residual block for U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels),
            nn.Dropout(dropout),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.Dropout(dropout)
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x: (batch, seq, channels) or (batch, channels)
        if x.dim() == 3:
            # (batch, seq, channels)
            identity = self.residual(x)
            out = self.layers(x)
            return identity + out
        else:
            # (batch, channels) - add dummy seq dimension
            x = x.unsqueeze(1)
            identity = self.residual(x)
            out = self.layers(x)
            result = identity + out
            return result.squeeze(1)
