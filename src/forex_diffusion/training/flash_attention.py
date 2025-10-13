"""
Flash Attention 2 integration for optimized attention mechanisms.

Provides drop-in replacement for standard attention with O(N) memory complexity.
Requires Ampere+ GPU (compute capability >= 8.0) and flash-attn library.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from loguru import logger


class FlashAttentionWrapper(nn.Module):
    """
    Wrapper for Flash Attention 2 with automatic fallback.

    If flash-attn is not available or GPU doesn't support it,
    falls back to standard PyTorch attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        # Check if Flash Attention is available
        self.use_flash_attn = self._check_flash_attention()

        if self.use_flash_attn:
            try:
                from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

                self.flash_attn_func = flash_attn_func
                logger.info("Flash Attention 2 enabled")
            except ImportError:
                self.use_flash_attn = False
                logger.warning("flash-attn import failed, using standard attention")

        # Standard attention fallback
        if not self.use_flash_attn:
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                batch_first=batch_first,
            )
            logger.info("Using standard PyTorch attention")

        # QKV projection (shared for both implementations)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention can be used."""
        # Check CUDA availability
        if not torch.cuda.is_available():
            return False

        # Check compute capability (Ampere+ required)
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        if major < 8:
            logger.warning(
                f"Flash Attention requires Ampere+ GPU (compute capability >= 8.0), "
                f"found {major}.{minor}"
            )
            return False

        # Check if library is installed
        try:
            import flash_attn

            return True
        except ImportError:
            logger.warning("flash-attn library not installed")
            return False

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Flash Attention or standard attention.

        Args:
            query: Query tensor (B, L, D) if batch_first else (L, B, D)
            key: Key tensor (optional, defaults to query for self-attention)
            value: Value tensor (optional, defaults to query for self-attention)
            attn_mask: Attention mask (optional)
            need_weights: Whether to return attention weights (not supported with Flash Attn)

        Returns:
            Tuple of (output, attention_weights)
        """
        # Handle self-attention
        if key is None:
            key = query
        if value is None:
            value = query

        if self.use_flash_attn and not need_weights and attn_mask is None:
            return self._flash_attention_forward(query, key, value)
        else:
            return self._standard_attention_forward(
                query, key, value, attn_mask, need_weights
            )

    def _flash_attention_forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """Flash Attention forward pass."""
        # Ensure batch_first format
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, L, D = query.shape

        # Project to QKV
        qkv = self.qkv_proj(query)  # (B, L, 3*D)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)  # (B, L, 3, H, D_h)
        qkv = qkv.permute(0, 2, 1, 3, 4)  # (B, 3, L, H, D_h)

        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each is (B, L, H, D_h)

        # Flash Attention expects (B, L, H, D_h)
        output = self.flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None,  # Use default 1/sqrt(d_k)
            causal=False,
        )  # (B, L, H, D_h)

        # Reshape output
        output = output.reshape(B, L, self.embed_dim)  # (B, L, D)

        # Output projection
        output = self.out_proj(output)

        # Restore original format if needed
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, None

    def _standard_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        need_weights: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard PyTorch attention forward pass."""
        return self.attn(
            query, key, value, attn_mask=attn_mask, need_weights=need_weights
        )


def replace_attention_with_flash(
    module: nn.Module, recursive: bool = True
) -> nn.Module:
    """
    Replace all MultiheadAttention modules with FlashAttentionWrapper.

    Args:
        module: PyTorch module to modify
        recursive: Whether to replace recursively in child modules

    Returns:
        Modified module
    """
    # Replace direct children
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            # Replace with Flash Attention wrapper
            flash_attn = FlashAttentionWrapper(
                embed_dim=child.embed_dim,
                num_heads=child.num_heads,
                dropout=child.dropout,
                bias=child.in_proj_bias is not None,
                batch_first=child.batch_first,
            )

            # Copy weights if possible
            try:
                if (
                    hasattr(child, "in_proj_weight")
                    and child.in_proj_weight is not None
                ):
                    flash_attn.qkv_proj.weight.data.copy_(child.in_proj_weight)
                if hasattr(child, "in_proj_bias") and child.in_proj_bias is not None:
                    flash_attn.qkv_proj.bias.data.copy_(child.in_proj_bias)
                if hasattr(child, "out_proj"):
                    flash_attn.out_proj.weight.data.copy_(child.out_proj.weight)
                    if child.out_proj.bias is not None:
                        flash_attn.out_proj.bias.data.copy_(child.out_proj.bias)
                logger.info(f"Replaced {name} with Flash Attention (weights copied)")
            except Exception as e:
                logger.warning(f"Could not copy weights for {name}: {e}")

            setattr(module, name, flash_attn)

        elif recursive:
            # Recursively replace in child modules
            replace_attention_with_flash(child, recursive=True)

    return module


class FlashSelfAttention(nn.Module):
    """
    Simplified Flash Self-Attention module for transformer blocks.

    This is a drop-in replacement for standard self-attention in transformers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Check Flash Attention availability
        self.use_flash_attn = self._check_flash_attention()

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Attention dropout (only used in standard attention)
        self.attn_drop = nn.Dropout(attn_drop) if not self.use_flash_attn else None

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_flash_attn:
            from flash_attn import flash_attn_qkvpacked_func

            self.flash_attn_func = flash_attn_qkvpacked_func
            logger.info(
                f"FlashSelfAttention initialized with dim={dim}, heads={num_heads}"
            )

    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention can be used."""
        if not torch.cuda.is_available():
            return False

        device = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            return False

        try:
            import flash_attn

            return True
        except ImportError:
            return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, L, D)

        Returns:
            Output tensor (B, L, D)
        """
        B, L, D = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)

        if self.use_flash_attn:
            # Flash Attention path
            # qkv should be (B, L, 3, H, D_h)
            attn_output = self.flash_attn_func(
                qkv,
                dropout_p=0.0,  # Flash Attn doesn't support attention dropout
                softmax_scale=self.scale,
                causal=False,
            )  # (B, L, H, D_h)

            # Reshape
            attn_output = attn_output.reshape(B, L, D)

        else:
            # Standard attention path
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_h)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, H, L, D_h)

            # Compute attention
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # Apply attention to values
            attn_output = attn @ v  # (B, H, L, D_h)
            attn_output = attn_output.transpose(1, 2).reshape(B, L, D)  # (B, L, D)

        # Output projection
        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x
