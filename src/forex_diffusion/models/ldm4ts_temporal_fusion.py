"""
LDM4TS Temporal Fusion Module

Gated fusion mechanism to project reconstructed RGB back to time series.
Combines explicit temporal features with implicit diffusion patterns.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List
from loguru import logger


class TemporalFusionModule(nn.Module):
    """
    Project reconstructed RGB image back to future time series.
    
    Uses gated fusion to combine:
    - Explicit: Direct temporal projection from image features
    - Implicit: Pattern features learned by diffusion model
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 28,
        hidden_dim: int = 256,
        horizons: List[int] = [15, 60, 240],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.horizons = horizons
        self.num_horizons = len(horizons)
        
        # Flatten latent: [B, C, H, W] → [B, C*H*W]
        latent_flat_dim = latent_channels * latent_size * latent_size
        
        # Explicit pathway: Direct projection
        self.explicit_proj = nn.Sequential(
            nn.Linear(latent_flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_horizons)
        )
        
        # Implicit pathway: Pattern features
        self.implicit_proj = nn.Sequential(
            nn.Linear(latent_flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_horizons)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(latent_flat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_horizons),
            nn.Sigmoid()
        )
    
    def forward(self, latent: torch.Tensor, current_price = 1.0) -> torch.Tensor:
        """
        Project latent to future prices.
        
        Args:
            latent: [B, C, H, W] latent features
            current_price: Current price for denormalization (float, list, numpy array, or tensor)
            
        Returns:
            [B, num_horizons] predicted prices
        """
        import torch
        import numpy as np
        
        B = latent.shape[0]
        
        # Flatten latent
        latent_flat = latent.reshape(B, -1)  # [B, C*H*W]
        
        # Explicit and implicit pathways
        explicit_out = self.explicit_proj(latent_flat)  # [B, num_horizons]
        implicit_out = self.implicit_proj(latent_flat)  # [B, num_horizons]
        
        # Gating
        gate_weights = self.gate(latent_flat)  # [B, num_horizons]
        
        # Fused output: gate * explicit + (1-gate) * implicit
        fused = gate_weights * explicit_out + (1 - gate_weights) * implicit_out
        
        # Convert current_price to tensor for broadcasting
        if isinstance(current_price, (int, float)):
            # Single value for all batches
            current_price_tensor = torch.tensor(current_price, dtype=latent.dtype, device=latent.device)
        elif isinstance(current_price, np.ndarray):
            # Numpy array [B] - convert to tensor
            current_price_tensor = torch.from_numpy(current_price).float().to(latent.device)
        elif isinstance(current_price, (list, tuple)):
            # List/tuple - convert to tensor
            current_price_tensor = torch.tensor(current_price, dtype=latent.dtype, device=latent.device)
        elif isinstance(current_price, torch.Tensor):
            # Already tensor
            current_price_tensor = current_price.to(latent.device)
        else:
            # Fallback to 1.0
            current_price_tensor = torch.tensor(1.0, dtype=latent.dtype, device=latent.device)
        
        # Ensure correct shape for broadcasting: [B, 1] or [B]
        if current_price_tensor.ndim == 0:
            # Scalar - broadcast to all batches and horizons
            current_price_tensor = current_price_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1]
        elif current_price_tensor.ndim == 1 and current_price_tensor.shape[0] == B:
            # [B] - add horizon dimension
            current_price_tensor = current_price_tensor.unsqueeze(1)  # [B, 1]
        
        # Convert to prices (add current_price as baseline)
        # Assume fused is normalized price change [-1, 1]
        predictions = current_price_tensor * (1.0 + fused * 0.1)  # Max ±10% change
        
        return predictions


if __name__ == "__main__":
    logger.info("Testing TemporalFusionModule...")
    
    fusion = TemporalFusionModule(
        latent_channels=4,
        latent_size=28,
        hidden_dim=256,
        horizons=[15, 60, 240]
    )
    
    # Test input
    B = 2
    latent = torch.randn(B, 4, 28, 28)
    current_price = 1.05200
    
    predictions = fusion(latent, current_price)
    
    logger.info(f"Latent shape: {latent.shape}")
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Predictions: {predictions}")
    
    assert predictions.shape == (B, 3), f"Expected (2, 3), got {predictions.shape}"
    
    logger.info("✅ TemporalFusionModule test passed!")
