"""
LDM4TS VAE Wrapper

Wraps Stable Diffusion VAE (AutoencoderKL) for time series image encoding/decoding.
Pre-trained on millions of images, provides powerful visual feature extraction.

Reference: Stable Diffusion VAE (stabilityai/sd-vae-ft-mse)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple
from loguru import logger

try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("diffusers not installed. Install with: pip install diffusers>=0.25.0")


class LDM4TSVAE(nn.Module):
    """
    VAE wrapper for LDM4TS.
    
    Encodes RGB images [B, 3, 224, 224] → latent space [B, 4, 28, 28]
    Decodes latent space [B, 4, 28, 28] → RGB images [B, 3, 224, 224]
    
    Uses pre-trained Stable Diffusion VAE for strong visual features.
    """
    
    def __init__(
        self,
        pretrained_model: str = "stabilityai/sd-vae-ft-mse",
        latent_channels: int = 4,
        downsample_factor: int = 8,
        latent_scale_factor: float = 0.18215,
        freeze_vae: bool = True
    ):
        """
        Initialize VAE wrapper.
        
        Args:
            pretrained_model: Hugging Face model ID
            latent_channels: Number of latent channels (4 for SD VAE)
            downsample_factor: Spatial downsampling (8x8 = 224→28)
            latent_scale_factor: Scale factor for latent space
            freeze_vae: If True, freeze VAE weights (faster inference)
        """
        super().__init__()
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers required. Install with: pip install diffusers>=0.25.0")
        
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.latent_scale_factor = latent_scale_factor
        
        # Load pre-trained VAE from local model files
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent.parent
        local_vae = project_root / "models" / "vae"
        
        if not local_vae.exists() or not (local_vae / "config.json").exists():
            raise FileNotFoundError(
                f"VAE model not found at {local_vae}\n"
                f"Please ensure the 'models/vae' directory contains the pre-trained VAE model.\n"
                f"Expected model: {pretrained_model}\n"
                f"The model files should be included in the distribution package."
            )
        
        logger.info(f"Loading VAE from local: {local_vae}")
        self.vae = AutoencoderKL.from_pretrained(
            str(local_vae),
            torch_dtype=torch.float32
        )
        
        # Freeze VAE weights if requested
        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()
            logger.info("VAE weights frozen (eval mode)")
        
        logger.info(f"VAE initialized: latent_channels={latent_channels}, downsample={downsample_factor}x")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image to latent space.
        
        Args:
            x: RGB image [B, 3, H, W] (values in [0, 1])
            
        Returns:
            Latent tensor [B, 4, H/8, W/8]
        """
        # Scale to [-1, 1] (VAE expects this range)
        x = 2.0 * x - 1.0
        
        # Encode
        with torch.no_grad() if not self.training else torch.enable_grad():
            latent_dist = self.vae.encode(x).latent_dist
            latent = latent_dist.sample()  # Sample from distribution
        
        # Scale latent
        latent = latent * self.latent_scale_factor
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent space to RGB image.
        
        Args:
            latent: Latent tensor [B, 4, H/8, W/8]
            
        Returns:
            RGB image [B, 3, H, W] (values in [0, 1])
        """
        # Unscale latent
        latent = latent / self.latent_scale_factor
        
        # Decode
        with torch.no_grad() if not self.training else torch.enable_grad():
            decoded = self.vae.decode(latent).sample
        
        # Scale back to [0, 1]
        decoded = (decoded + 1.0) / 2.0
        decoded = torch.clamp(decoded, 0.0, 1.0)
        
        return decoded
    
    def get_latent_shape(self, image_size: Tuple[int, int]) -> Tuple[int, int, int]:
        """
        Get latent space shape for given image size.
        
        Args:
            image_size: (H, W) tuple
            
        Returns:
            (C, H', W') tuple where H'=H/8, W'=W/8
        """
        h, w = image_size
        return (
            self.latent_channels,
            h // self.downsample_factor,
            w // self.downsample_factor
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (encode + decode for reconstruction loss).
        
        Args:
            x: RGB image [B, 3, H, W]
            
        Returns:
            Tuple of (latent, reconstructed)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed


if __name__ == "__main__":
    # Test VAE
    logger.info("Testing LDM4TSVAE...")
    
    vae = LDM4TSVAE(freeze_vae=True)
    
    # Test encode/decode
    B, C, H, W = 2, 3, 224, 224
    x = torch.randn(B, C, H, W).clamp(0, 1)  # Simulate RGB images
    
    latent = vae.encode(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Latent shape: {latent.shape}")
    
    reconstructed = vae.decode(latent)
    logger.info(f"Reconstructed shape: {reconstructed.shape}")
    
    # Check reconstruction quality
    mse = torch.mean((x - reconstructed) ** 2).item()
    logger.info(f"Reconstruction MSE: {mse:.6f}")
    
    # Expected: latent [2, 4, 28, 28], reconstructed [2, 3, 224, 224]
    assert latent.shape == (B, 4, 28, 28), f"Unexpected latent shape: {latent.shape}"
    assert reconstructed.shape == (B, C, H, W), f"Unexpected output shape: {reconstructed.shape}"
    
    logger.info("✅ VAE test passed!")
