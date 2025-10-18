"""
Memory-efficient attention mechanisms for LDM4TS training.

Supports:
- SageAttention 2 (https://github.com/thu-ml/SageAttention)
- FlashAttention 2 (https://github.com/Dao-AILab/flash-attention)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Literal
from loguru import logger

# Try importing attention libraries
try:
    import sageattention
    SAGEATTENTION_AVAILABLE = True
    logger.info("SageAttention 2 available")
except ImportError:
    SAGEATTENTION_AVAILABLE = False
    logger.debug("SageAttention 2 not available")

try:
    from flash_attn import flash_attn_func
    FLASHATTENTION_AVAILABLE = True
    logger.info("FlashAttention 2 available")
except ImportError:
    FLASHATTENTION_AVAILABLE = False
    logger.debug("FlashAttention 2 not available")


AttentionBackend = Literal["default", "sage", "flash", "xformers"]


def patch_unet_attention(
    unet: nn.Module,
    backend: AttentionBackend = "default",
    enable_gradient_checkpointing: bool = True
) -> nn.Module:
    """
    Patch U-Net attention layers with memory-efficient implementations.
    
    Args:
        unet: UNet2DConditionModel to patch
        backend: Attention backend to use
        enable_gradient_checkpointing: Enable gradient checkpointing for further memory savings
        
    Returns:
        Patched U-Net model
    """
    if backend == "sage" and not SAGEATTENTION_AVAILABLE:
        logger.warning("SageAttention requested but not available, falling back to default")
        backend = "default"
    
    if backend == "flash" and not FLASHATTENTION_AVAILABLE:
        logger.warning("FlashAttention requested but not available, falling back to default")
        backend = "default"
    
    if backend == "default":
        logger.info("Using default PyTorch attention (no optimization)")
        if enable_gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled")
        return unet
    
    # Patch attention layers based on backend
    if backend == "sage":
        _patch_sage_attention(unet)
        logger.info("SageAttention 2 enabled - VRAM usage reduced by ~30-40%")
    
    elif backend == "flash":
        _patch_flash_attention(unet)
        logger.info("FlashAttention 2 enabled - VRAM usage reduced by ~40-50%")
    
    # Enable gradient checkpointing for additional savings
    if enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled - further VRAM reduction")
    
    return unet


def _patch_sage_attention(unet: nn.Module) -> None:
    """Replace attention with SageAttention implementation."""
    from diffusers.models.attention_processor import Attention
    
    class SageAttnProcessor:
        """Attention processor using SageAttention."""
        
        def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> torch.Tensor:
            batch_size, sequence_length, _ = hidden_states.shape
            
            # Prepare QKV
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
            value = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
            
            # Reshape for multi-head attention
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            
            # SageAttention expects [batch, heads, seq, dim]
            # Reshape from [batch*heads, seq, dim] to [batch, heads, seq, dim]
            heads = attn.heads
            query = query.view(batch_size, heads, sequence_length, -1)
            key = key.view(batch_size, heads, -1, query.shape[-1])
            value = value.view(batch_size, heads, -1, query.shape[-1])
            
            # Apply SageAttention
            try:
                hidden_states = sageattention.sageattn(
                    query, key, value,
                    is_causal=False,
                    smooth_k=True,  # Enable key smoothing for better accuracy
                    tensor_layout="HND"  # [batch, Heads, seqlen, Dim]
                )
            except Exception as e:
                logger.warning(f"SageAttention failed: {e}, falling back to default")
                # Fallback to default attention
                attention_probs = torch.softmax(
                    torch.matmul(query, key.transpose(-1, -2)) / (query.shape[-1] ** 0.5),
                    dim=-1
                )
                hidden_states = torch.matmul(attention_probs, value)
            
            # Reshape back to [batch*heads, seq, dim]
            hidden_states = hidden_states.reshape(batch_size * heads, sequence_length, -1)
            
            # Linear projection
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
            return hidden_states
    
    # Apply processor to all attention layers
    unet.set_attn_processor(SageAttnProcessor())


def _patch_flash_attention(unet: nn.Module) -> None:
    """Replace attention with FlashAttention implementation."""
    from diffusers.models.attention_processor import Attention
    
    class FlashAttnProcessor:
        """Attention processor using FlashAttention 2."""
        
        def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> torch.Tensor:
            batch_size, sequence_length, _ = hidden_states.shape
            
            # Prepare QKV
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
            value = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
            
            # Reshape for multi-head attention
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            
            # FlashAttention expects [batch, seq, heads, dim]
            heads = attn.heads
            head_dim = query.shape[-1]
            
            query = query.view(batch_size, sequence_length, heads, head_dim)
            key = key.view(batch_size, -1, heads, head_dim)
            value = value.view(batch_size, -1, heads, head_dim)
            
            # Apply FlashAttention
            try:
                hidden_states = flash_attn_func(
                    query, key, value,
                    causal=False,
                    dropout_p=0.0
                )
                # Output shape: [batch, seq, heads, dim]
                hidden_states = hidden_states.reshape(batch_size * heads, sequence_length, head_dim)
            except Exception as e:
                logger.warning(f"FlashAttention failed: {e}, falling back to default")
                # Fallback to default attention
                query = query.reshape(batch_size * heads, sequence_length, head_dim)
                key = key.reshape(batch_size * heads, -1, head_dim)
                value = value.reshape(batch_size * heads, -1, head_dim)
                
                attention_probs = torch.softmax(
                    torch.matmul(query, key.transpose(-1, -2)) / (head_dim ** 0.5),
                    dim=-1
                )
                hidden_states = torch.matmul(attention_probs, value)
            
            # Linear projection
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
            return hidden_states
    
    # Apply processor to all attention layers
    unet.set_attn_processor(FlashAttnProcessor())


def estimate_vram_usage(
    batch_size: int = 4,
    sequence_length: int = 100,
    backend: AttentionBackend = "default",
    gradient_checkpointing: bool = False
) -> dict:
    """
    Estimate VRAM usage for different configurations.
    
    Args:
        batch_size: Training batch size
        sequence_length: Sequence length
        backend: Attention backend
        gradient_checkpointing: Whether gradient checkpointing is enabled
        
    Returns:
        Dict with estimated VRAM usage in GB
    """
    # Base model size
    base_vram = 2.5  # GB (U-Net + VAE + encoders)
    
    # Activation memory (scales with batch size and sequence length)
    activation_memory = (batch_size * sequence_length * 4 * 28 * 28 * 4) / (1024**3)  # Latents
    
    # Attention memory scaling
    attention_factors = {
        "default": 1.0,
        "sage": 0.65,     # ~35% reduction
        "flash": 0.55,    # ~45% reduction
        "xformers": 0.70  # ~30% reduction
    }
    
    attention_memory = activation_memory * attention_factors.get(backend, 1.0)
    
    # Gradient checkpointing reduces activation memory by ~50%
    if gradient_checkpointing:
        attention_memory *= 0.5
    
    total_vram = base_vram + attention_memory
    
    return {
        "base_model_gb": base_vram,
        "activation_memory_gb": attention_memory,
        "total_estimated_gb": total_vram,
        "backend": backend,
        "gradient_checkpointing": gradient_checkpointing,
        "batch_size": batch_size,
        "savings_vs_default_pct": (1 - total_vram / (base_vram + activation_memory / attention_factors.get(backend, 1.0))) * 100
    }


if __name__ == "__main__":
    # Test availability
    logger.info("=== Memory-Efficient Attention Status ===")
    logger.info(f"SageAttention 2: {'✓ Available' if SAGEATTENTION_AVAILABLE else '✗ Not installed'}")
    logger.info(f"FlashAttention 2: {'✓ Available' if FLASHATTENTION_AVAILABLE else '✗ Not installed'}")
    
    # Estimate VRAM usage
    configs = [
        ("Default", "default", False),
        ("Default + GC", "default", True),
        ("SageAttention", "sage", False),
        ("SageAttention + GC", "sage", True),
        ("FlashAttention", "flash", False),
        ("FlashAttention + GC", "flash", True),
    ]
    
    logger.info("\n=== VRAM Usage Estimates (batch_size=4) ===")
    for name, backend, gc in configs:
        estimate = estimate_vram_usage(batch_size=4, backend=backend, gradient_checkpointing=gc)
        logger.info(f"{name}: {estimate['total_estimated_gb']:.2f} GB ({estimate['savings_vs_default_pct']:.1f}% savings)")
