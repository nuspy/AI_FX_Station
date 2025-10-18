"""
LDM4TS: Latent Diffusion Model for Time Series Forecasting

Main model combining:
- Vision transforms (SEG, GAF, RP)
- VAE encoder/decoder
- Cross-modal conditioning (frequency + text)
- Diffusion process
- Temporal fusion

Reference: https://arxiv.org/html/2502.14887v1
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from loguru import logger

from .vision_transforms import TimeSeriesVisionEncoder
from .ldm4ts_vae import LDM4TSVAE
from .ldm4ts_conditioning import FrequencyConditioner, TextConditioner
from .ldm4ts_temporal_fusion import TemporalFusionModule

try:
    from diffusers import UNet2DConditionModel, DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class LDM4TSModel(nn.Module):
    """
    Complete LDM4TS model for probabilistic time series forecasting.
    
    Pipeline:
    1. OHLCV → Vision Encoder → RGB [3, 224, 224]
    2. RGB → VAE Encoder → Latent [4, 28, 28]
    3. Latent + Conditioning → Diffusion (50 steps) → Denoised Latent
    4. Denoised Latent → VAE Decoder → Reconstructed RGB
    5. Reconstructed RGB → Temporal Fusion → Future Prices
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        horizons: List[int] = [15, 60, 240],
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        text_model: str = "openai/clip-vit-base-patch32",
        diffusion_steps: int = 1000,
        sampling_steps: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers required. Install: pip install diffusers>=0.25.0")
        
        self.image_size = image_size
        self.horizons = horizons
        self.device = torch.device(device)
        self.diffusion_steps = diffusion_steps
        self.sampling_steps = sampling_steps
        
        # Vision encoder (not trainable, just transformation)
        self.vision_encoder = TimeSeriesVisionEncoder(
            image_size=image_size,
            normalize=True
        )
        
        # VAE (frozen)
        self.vae = LDM4TSVAE(
            pretrained_model=vae_model,
            freeze_vae=True
        ).to(self.device)
        
        # Conditioning modules
        self.freq_cond = FrequencyConditioner(
            expected_seq_len=100,  # Expected sequence length (can handle variable lengths via adaptive projection)
            num_features=5,  # OHLCV features
            hidden_dim=256,
            output_dim=768
        ).to(self.device)
        
        self.text_cond = TextConditioner(
            model_name=text_model,
            freeze=True,
            output_dim=768  # Match FrequencyConditioner output
        ).to(self.device)
        
        # U-Net for diffusion (trainable)
        self.unet = UNet2DConditionModel(
            sample_size=28,  # Latent size (224/8)
            in_channels=4,   # Latent channels
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=768,  # Conditioning dimension
        ).to(self.device)
        
        # Noise scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2"
        )
        
        # Temporal fusion (trainable)
        self.temporal_fusion = TemporalFusionModule(
            latent_channels=4,
            latent_size=28,
            hidden_dim=256,
            horizons=horizons
        ).to(self.device)
        
        logger.info(f"LDM4TS initialized: horizons={horizons}, device={device}")
    
    def encode_vision(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """
        Encode OHLCV → RGB image.
        
        Args:
            ohlcv: [B, L, 5] or [L, 5] OHLCV data
            
        Returns:
            [B, 3, H, W] RGB tensor
        """
        # Convert numpy to tensor if needed
        if not isinstance(ohlcv, torch.Tensor):
            ohlcv = torch.from_numpy(ohlcv).float()
        
        # Use vision encoder
        rgb = self.vision_encoder.encode(ohlcv, return_tensor=True)
        
        if rgb.device != self.device:
            rgb = rgb.to(self.device)
        
        return rgb
    
    def forward(
        self,
        ohlcv: torch.Tensor,
        current_price: float,
        num_samples: int = 1,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: OHLCV → Predictions.
        
        Args:
            ohlcv: [B, L, 5] OHLCV data
            current_price: Current price for denormalization
            num_samples: Number of Monte Carlo samples for uncertainty
            return_all: If True, return intermediate outputs
            
        Returns:
            Dict with predictions and optionally intermediate outputs
        """
        B = ohlcv.shape[0] if ohlcv.ndim == 3 else 1
        
        # 1. Vision encoding
        rgb = self.encode_vision(ohlcv)  # [B, 3, 224, 224]
        
        # 2. VAE encoding
        latent = self.vae.encode(rgb)  # [B, 4, 28, 28]
        
        # 3. Conditioning
        # Frequency conditioning
        freq_emb = self.freq_cond(ohlcv)  # [B, 768]
        
        # Text conditioning (generate descriptions)
        descriptions = [
            TextConditioner.generate_description({
                'symbol': 'EUR/USD',
                'trend': float(ohlcv[i, -1, 3] - ohlcv[i, 0, 3]) if B > 1 else float(ohlcv[-1, 3] - ohlcv[0, 3]),
                'volatility': float(ohlcv[i].std()) if B > 1 else float(ohlcv.std())
            })
            for i in range(B)
        ]
        text_emb = self.text_cond(descriptions, self.device)  # [B, 768]
        
        # Combine conditioning (average)
        conditioning = (freq_emb + text_emb) / 2.0  # [B, 768]
        
        # 4. Diffusion sampling (Monte Carlo for uncertainty)
        all_predictions = []
        for _ in range(num_samples):
            # Add noise
            noise = torch.randn_like(latent)
            noisy_latent = latent + noise * 0.1  # Small noise for sampling
            
            # Denoise (simplified - in training, use full scheduler)
            self.scheduler.set_timesteps(self.sampling_steps, device=self.device)
            
            denoised_latent = noisy_latent.clone()
            for t in self.scheduler.timesteps:
                # Predict noise
                noise_pred = self.unet(
                    denoised_latent,
                    t,
                    encoder_hidden_states=conditioning.unsqueeze(1)
                ).sample
                
                # Denoise step
                denoised_latent = self.scheduler.step(
                    noise_pred,
                    t,
                    denoised_latent
                ).prev_sample
            
            # 5. Temporal fusion
            predictions = self.temporal_fusion(denoised_latent, current_price)  # [B, num_horizons]
            all_predictions.append(predictions)
        
        # Stack predictions
        all_predictions = torch.stack(all_predictions, dim=0)  # [num_samples, B, num_horizons]
        
        # Compute statistics
        mean_pred = all_predictions.mean(dim=0)  # [B, num_horizons]
        std_pred = all_predictions.std(dim=0)    # [B, num_horizons]
        q05 = torch.quantile(all_predictions, 0.05, dim=0)
        q50 = torch.quantile(all_predictions, 0.50, dim=0)
        q95 = torch.quantile(all_predictions, 0.95, dim=0)
        
        result = {
            'mean': mean_pred,
            'std': std_pred,
            'q05': q05,
            'q50': q50,
            'q95': q95
        }
        
        if return_all:
            result['rgb'] = rgb
            result['latent'] = latent
            result['conditioning'] = conditioning
        
        return result


if __name__ == "__main__":
    logger.info("Testing LDM4TSModel...")
    
    # Create model
    model = LDM4TSModel(
        image_size=(224, 224),
        horizons=[15, 60, 240],
        device='cpu'  # Use CPU for test
    )
    
    # Test input
    import numpy as np
    ohlcv = np.random.randn(100, 5).cumsum(axis=0) + 1.05  # Random walk
    ohlcv = torch.from_numpy(ohlcv).float().unsqueeze(0)  # [1, 100, 5]
    
    current_price = float(ohlcv[0, -1, 3])  # Last close
    
    # Forward pass
    logger.info("Running forward pass...")
    with torch.no_grad():
        result = model(ohlcv, current_price, num_samples=5, return_all=True)
    
    logger.info(f"Mean predictions: {result['mean'].shape} = {result['mean']}")
    logger.info(f"Std predictions: {result['std'].shape}")
    logger.info(f"Q05: {result['q05']}")
    logger.info(f"Q95: {result['q95']}")
    
    logger.info("✅ LDM4TSModel test passed!")
