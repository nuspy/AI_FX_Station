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
    """FFT-based frequency domain conditioning with adaptive input projection."""
    
    def __init__(self, expected_seq_len: int = 100, num_features: int = 5, hidden_dim: int = 256, output_dim: int = 768):
        super().__init__()
        self.expected_seq_len = expected_seq_len
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Calculate expected FFT output size
        expected_fft_size = 2 * num_features * (expected_seq_len // 2 + 1)
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(expected_fft_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Adaptive projection layer (created dynamically if needed)
        self.projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] time series
        Returns:
            [B, output_dim] frequency embeddings
        """
        B, L, D = x.shape
        
        # FFT on each feature
        fft = torch.fft.rfft(x, dim=1)  # [B, L//2+1, D]
        
        # Extract magnitude and phase
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        # Flatten
        freq_features = torch.cat([
            magnitude.reshape(B, -1),
            phase.reshape(B, -1)
        ], dim=1)  # [B, 2*D*(L//2+1)]
        
        # If sequence length differs, use adaptive projection
        expected_fft_size = 2 * self.num_features * (self.expected_seq_len // 2 + 1)
        actual_fft_size = freq_features.shape[1]
        
        if actual_fft_size != expected_fft_size:
            # Create projection layer if not exists or size changed
            if self.projection is None or self.projection.in_features != actual_fft_size:
                self.projection = nn.Linear(actual_fft_size, expected_fft_size).to(x.device)
                # Initialize with small weights
                nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
                nn.init.zeros_(self.projection.bias)
            
            # Project to expected size
            freq_features = self.projection(freq_features)
        
        return self.encoder(freq_features)


class TextConditioner(nn.Module):
    """CLIP-based text conditioning for statistical descriptions."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
        output_dim: int = 768
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required")
        
        self.output_dim = output_dim
        
        # Load from local model files
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        local_clip = project_root / "models" / "clip"
        
        if local_clip.exists() and (local_clip / "config.json").exists():
            self.tokenizer = CLIPTokenizer.from_pretrained(str(local_clip))
            self.text_encoder = CLIPTextModel.from_pretrained(str(local_clip))
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        # Get CLIP embedding dimension (512 for clip-vit-base-patch32)
        clip_dim = self.text_encoder.config.hidden_size
        
        # Add projection layer if output_dim differs from CLIP dimension
        if clip_dim != output_dim:
            self.projection = nn.Linear(clip_dim, output_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        else:
            self.projection = None
        
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
            [B, output_dim] text embeddings (default 768)
        """
        tokens = self.tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            embeddings = outputs.pooler_output  # [B, clip_dim] (512 for base CLIP)
        
        # Project to output_dim if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)  # [B, 512] -> [B, 768]
        
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
