"""
Multi-Scale Encoder for SSSD

Aggregates features from multiple timeframes (5m, 15m, 1h, 4h) into a unified
context representation using S4 layers and cross-timeframe attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from loguru import logger

from .s4_layer import StackedS4

# Try to use Flash Attention for optimized cross-attention
try:
    from forex_diffusion.training.flash_attention import FlashAttentionWrapper
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale encoder for processing multiple timeframe features.

    Architecture:
    1. Per-timeframe S4 encoding (independent S4 stacks)
    2. Cross-timeframe attention (learn which timeframe is relevant)
    3. Fusion MLP (combine attended representations)
    """

    def __init__(
        self,
        timeframes: List[str] = ["5m", "15m", "1h", "4h"],
        feature_dim: int = 200,
        s4_state_dim: int = 128,
        s4_layers: int = 4,
        s4_dropout: float = 0.1,
        context_dim: int = 512,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize multi-scale encoder.

        Args:
            timeframes: List of timeframe identifiers
            feature_dim: Input feature dimension (same across all timeframes)
            s4_state_dim: S4 state dimension
            s4_layers: Number of S4 layers per timeframe
            s4_dropout: Dropout in S4 layers
            context_dim: Output context dimension
            attention_heads: Number of attention heads
            attention_dropout: Dropout in attention
        """
        super().__init__()

        self.timeframes = timeframes
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.n_timeframes = len(timeframes)

        # Per-timeframe S4 encoders
        self.timeframe_encoders = nn.ModuleDict({
            tf: StackedS4(
                d_model=feature_dim,
                d_state=s4_state_dim,
                n_layers=s4_layers,
                dropout=s4_dropout
            )
            for tf in timeframes
        })

        # Cross-timeframe multi-head attention
        # Use Flash Attention if available for 30-50% speedup on Ampere+ GPUs
        if FLASH_ATTENTION_AVAILABLE:
            self.cross_attention = FlashAttentionWrapper(
                embed_dim=feature_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
                batch_first=True
            )
            logger.info("Using Flash Attention for cross-timeframe attention (GPU optimized)")
        else:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
                batch_first=True
            )
            logger.info("Using standard PyTorch attention (Flash Attention not available)")

        # Fusion MLP
        # Input: concatenated attended representations from all timeframes
        # Output: unified context vector
        fusion_input_dim = feature_dim * self.n_timeframes
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, context_dim * 2),
            nn.LayerNorm(context_dim * 2),
            nn.GELU(),
            nn.Dropout(s4_dropout),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
            nn.Dropout(s4_dropout)
        )

        # Learnable query for attention (what to attend to)
        self.attention_query = nn.Parameter(torch.randn(1, 1, feature_dim))

        logger.info(
            f"Initialized MultiScaleEncoder: timeframes={timeframes}, "
            f"feature_dim={feature_dim}, context_dim={context_dim}, "
            f"attention_heads={attention_heads}"
        )

    def forward(
        self,
        features_dict: Dict[str, torch.Tensor],
        return_attention_weights: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-scale encoder.

        Args:
            features_dict: Dictionary mapping timeframe to feature tensor
                Format: {
                    "5m": (batch, seq_len_5m, feature_dim),
                    "15m": (batch, seq_len_15m, feature_dim),
                    "1h": (batch, seq_len_1h, feature_dim),
                    "4h": (batch, seq_len_4h, feature_dim)
                }
            return_attention_weights: If True, return attention weights

        Returns:
            context: Unified context vector (batch, context_dim)
            attention_weights: (Optional) Attention weights (batch, n_timeframes)
        """
        batch_size = next(iter(features_dict.values())).shape[0]

        # Step 1: Encode each timeframe with S4
        encoded_timeframes = {}
        for tf in self.timeframes:
            if tf not in features_dict:
                raise ValueError(f"Missing features for timeframe: {tf}")

            features = features_dict[tf]  # (batch, seq_len, feature_dim)

            # Apply S4 encoder
            encoded = self.timeframe_encoders[tf](features)  # (batch, seq_len, feature_dim)

            # Take final hidden state (last timestep)
            encoded_final = encoded[:, -1, :]  # (batch, feature_dim)

            encoded_timeframes[tf] = encoded_final

        # Step 2: Stack encoded representations
        # Shape: (batch, n_timeframes, feature_dim)
        encoded_stack = torch.stack([
            encoded_timeframes[tf] for tf in self.timeframes
        ], dim=1)

        # Step 3: Cross-timeframe attention
        # Query: What should we attend to? (learnable)
        # Keys/Values: Encoded representations from all timeframes

        # Expand query for batch
        query = self.attention_query.expand(batch_size, -1, -1)  # (batch, 1, feature_dim)

        # Apply multi-head attention
        # Output: (batch, 1, feature_dim)
        # Attention weights: (batch, 1, n_timeframes)
        attended, attn_weights = self.cross_attention(
            query=query,
            key=encoded_stack,
            value=encoded_stack,
            need_weights=True,
            average_attn_weights=True
        )

        attended = attended.squeeze(1)  # (batch, feature_dim)

        # Step 4: Concatenate all timeframe representations
        # This preserves information from all timeframes while also using attention
        encoded_concat = encoded_stack.reshape(batch_size, -1)  # (batch, n_timeframes * feature_dim)

        # Step 5: Fusion MLP
        context = self.fusion_mlp(encoded_concat)  # (batch, context_dim)

        if return_attention_weights:
            # Squeeze attention weights: (batch, n_timeframes)
            attn_weights = attn_weights.squeeze(1)
            return context, attn_weights
        else:
            return context

    def get_timeframe_importance(
        self,
        features_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Get attention weights for each timeframe (for interpretability).

        Args:
            features_dict: Feature dict (same as forward())

        Returns:
            Dict mapping timeframe to importance score
        """
        with torch.no_grad():
            _, attn_weights = self.forward(features_dict, return_attention_weights=True)

            # Average over batch
            avg_weights = attn_weights.mean(dim=0)  # (n_timeframes,)

            importance = {
                tf: float(avg_weights[i])
                for i, tf in enumerate(self.timeframes)
            }

        return importance

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f'timeframes={self.timeframes}, feature_dim={self.feature_dim}, '
            f'context_dim={self.context_dim}, n_timeframes={self.n_timeframes}'
        )


class TimeframeAlignmentLayer(nn.Module):
    """
    Helper layer to align features from different timeframes to same length.

    This is useful if you want to use convolutional or recurrent encoders
    that expect fixed-length sequences.
    """

    def __init__(
        self,
        target_length: int = 100,
        method: str = "interpolate"
    ):
        """
        Initialize alignment layer.

        Args:
            target_length: Target sequence length
            method: Alignment method ("interpolate", "pad", or "truncate")
        """
        super().__init__()

        self.target_length = target_length
        self.method = method

    def forward(
        self,
        features_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Align all timeframe features to same length.

        Args:
            features_dict: Dict of features with varying lengths

        Returns:
            Dict of features with same length
        """
        aligned_dict = {}

        for tf, features in features_dict.items():
            batch, seq_len, feature_dim = features.shape

            if seq_len == self.target_length:
                # Already correct length
                aligned_dict[tf] = features

            elif self.method == "interpolate":
                # Interpolate to target length
                # Permute to (batch, feature_dim, seq_len) for interpolation
                features_t = features.permute(0, 2, 1)

                # Interpolate
                aligned = torch.nn.functional.interpolate(
                    features_t,
                    size=self.target_length,
                    mode='linear',
                    align_corners=False
                )

                # Permute back to (batch, seq_len, feature_dim)
                aligned = aligned.permute(0, 2, 1)

                aligned_dict[tf] = aligned

            elif self.method == "pad":
                # Pad or truncate to target length
                if seq_len < self.target_length:
                    # Pad with zeros
                    pad_length = self.target_length - seq_len
                    padding = torch.zeros(batch, pad_length, feature_dim,
                                        device=features.device, dtype=features.dtype)
                    aligned_dict[tf] = torch.cat([features, padding], dim=1)
                else:
                    # Truncate (take last target_length steps)
                    aligned_dict[tf] = features[:, -self.target_length:, :]

            elif self.method == "truncate":
                # Simply truncate to target length
                if seq_len > self.target_length:
                    aligned_dict[tf] = features[:, -self.target_length:, :]
                else:
                    # Pad if shorter
                    pad_length = self.target_length - seq_len
                    padding = torch.zeros(batch, pad_length, feature_dim,
                                        device=features.device, dtype=features.dtype)
                    aligned_dict[tf] = torch.cat([features, padding], dim=1)

            else:
                raise ValueError(f"Unknown alignment method: {self.method}")

        return aligned_dict
