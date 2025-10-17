"""
LDM4TS Conditioning Modules

Frequency Conditioning: FFT embeddings
Text Conditioning: CLIP embeddings for statistical descriptions
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from loguru import logger

try:
    from transformers import CLIPTextModel, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FrequencyConditioner(nn.Module):
    """FFT-based frequency domain conditioning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] time series
        Returns:
            [B, output_dim] frequency embeddings
        """
        # FFT on each feature
        fft = torch.fft.rfft(x, dim=1)  # [B, L//2+1, D]
        
        # Extract magnitude and phase
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        # Flatten
        freq_features = torch.cat([
            magnitude.reshape(x.shape[0], -1),
            phase.reshape(x.shape[0], -1)
        ], dim=1)  # [B, 2*D*(L//2+1)]
        
        return self.encoder(freq_features)


class TextConditioner(nn.Module):
    """CLIP-based text conditioning for statistical descriptions."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required")
        
        # Load from local cache if available
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        local_cache = project_root / "models" / "huggingface"
        
        if local_cache.exists():
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir=str(local_cache))
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, cache_dir=str(local_cache))
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()
    
    def forward(self, descriptions: list[str], device: torch.device) -> torch.Tensor:
        """
        Args:
            descriptions: List of text descriptions
            device: Target device
        Returns:
            [B, 768] text embeddings
        """
        tokens = self.tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            embeddings = outputs.pooler_output  # [B, 768]
        
        return embeddings
    
    @staticmethod
    def generate_description(stats: dict) -> str:
        """Generate statistical description from price statistics."""
        trend = "increasing" if stats.get('trend', 0) > 0 else "decreasing"
        volatility = "high" if stats.get('volatility', 0) > 0.01 else "low"
        return f"{stats['symbol']} price {trend} with {volatility} volatility"


if __name__ == "__main__":
    logger.info("Testing conditioning modules...")
    
    # Test frequency
    freq_cond = FrequencyConditioner(input_dim=100*5, output_dim=768)
    x = torch.randn(2, 100, 5)  # [B, L, D]
    freq_emb = freq_cond(x)
    logger.info(f"Frequency embedding: {freq_emb.shape}")
    assert freq_emb.shape == (2, 768)
    
    # Test text
    text_cond = TextConditioner()
    descriptions = ["EUR/USD price increasing with low volatility", 
                   "GBP/USD price decreasing with high volatility"]
    text_emb = text_cond(descriptions, device=torch.device('cpu'))
    logger.info(f"Text embedding: {text_emb.shape}")
    assert text_emb.shape == (2, 768)
    
    logger.info("✅ Conditioning tests passed!")
