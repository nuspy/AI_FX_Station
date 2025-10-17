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
    
    def forward(self, latent: torch.Tensor, current_price: float = 1.0) -> torch.Tensor:
        """
        Project latent to future prices.
        
        Args:
            latent: [B, C, H, W] latent features
            current_price: Current price for denormalization
            
        Returns:
            [B, num_horizons] predicted prices
        """
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
        
        # Convert to prices (add current_price as baseline)
        # Assume fused is normalized price change [-1, 1]
        predictions = current_price * (1.0 + fused * 0.1)  # Max ±10% change
        
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
