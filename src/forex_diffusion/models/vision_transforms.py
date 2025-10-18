"""
Vision Transformations for Time Series (LDM4TS)

Converts OHLCV time series into multi-view visual representations:
- Segmentation (SEG): Periodic restructuring
- Gramian Angular Field (GAF): Temporal correlations → spatial patterns
- Recurrence Plot (RP): Similarity matrices for cyclical behaviors

References:
- LDM4TS Paper: https://arxiv.org/html/2502.14887v1
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger


class TimeSeriesVisionEncoder:
    """
    Encode time series as RGB images with three complementary views.
    
    Each channel captures different temporal properties:
    - R channel: Segmentation (local patterns)
    - G channel: GAF (long-range correlations)
    - B channel: Recurrence Plot (cyclical behaviors)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        period: Optional[int] = None,
        normalize: bool = True
    ):
        """
        Initialize vision encoder.
        
        Args:
            image_size: Target image size (H, W)
            period: Period for segmentation (auto-detect if None)
            normalize: Whether to normalize each channel independently
        """
        self.image_size = image_size
        self.period = period
        self.normalize = normalize
        
    def encode(
        self,
        X: np.ndarray,
        return_tensor: bool = True
    ) -> np.ndarray | torch.Tensor:
        """
        Encode time series into RGB image.
        
        Args:
            X: Time series [B, L, D] or [L, D] or [L] (numpy array or torch.Tensor)
            return_tensor: Return torch.Tensor if True
            
        Returns:
            RGB image [B, 3, H, W] or [3, H, W]
        """
        # Handle input shapes and convert CUDA tensors to numpy
        if isinstance(X, torch.Tensor):
            # If CUDA tensor, move to CPU first
            if X.is_cuda:
                X = X.cpu().numpy()
            else:
                X = X.numpy()
        else:
            X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # [L] → [L, 1]
        if X.ndim == 2:
            X = X[np.newaxis, ...]  # [L, D] → [1, L, D]
            
        B, L, D = X.shape
        
        # Normalize to [0, 1]
        X_norm = self._normalize_timeseries(X)
        
        # Generate three views
        seg_channel = self._segmentation_transform(X_norm)      # [B, H, W]
        gaf_channel = self._gramian_angular_field(X_norm)       # [B, H, W]
        rp_channel = self._recurrence_plot(X_norm)              # [B, H, W]
        
        # Stack as RGB
        rgb_image = np.stack([seg_channel, gaf_channel, rp_channel], axis=1)  # [B, 3, H, W]
        
        if return_tensor:
            rgb_image = torch.from_numpy(rgb_image).float()
            
        # Remove batch dim if single sample
        if B == 1 and X.ndim == 2:
            rgb_image = rgb_image[0]  # [3, H, W]
            
        return rgb_image
    
    def _normalize_timeseries(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1] with epsilon for stability."""
        X_min = X.min(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        epsilon = 1e-8
        return (X - X_min) / (X_max - X_min + epsilon)
    
    def _segmentation_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Segmentation encoding: reshape into periodic matrix.
        
        Preserves local temporal structures by periodic restructuring.
        
        Args:
            X: [B, L, D] normalized time series
            
        Returns:
            [B, H, W] segmentation image
        """
        B, L, D = X.shape
        H, W = self.image_size
        
        # Detect period if not provided (use FFT)
        if self.period is None:
            period = self._detect_period(X[0, :, 0])  # Use first series
        else:
            period = self.period
            
        # Pad to make divisible by period
        pad_len = (period - L % period) % period
        if pad_len > 0:
            X_padded = np.pad(X, ((0, 0), (0, pad_len), (0, 0)), mode='edge')
        else:
            X_padded = X
            
        L_padded = X_padded.shape[1]
        n_periods = L_padded // period
        
        # Reshape into matrix: [B, n_periods, period, D]
        X_reshaped = X_padded.reshape(B, n_periods, period, D)
        
        # Average over features: [B, n_periods, period]
        seg_matrix = X_reshaped.mean(axis=-1)
        
        # Interpolate to target size
        seg_image = self._bilinear_resize(seg_matrix, H, W)
        
        # Normalize per channel
        if self.normalize:
            seg_image = self._normalize_channel(seg_image)
            
        return seg_image
    
    def _gramian_angular_field(self, X: np.ndarray) -> np.ndarray:
        """
        Gramian Angular Field (GAF) encoding.
        
        Transforms temporal correlations into spatial patterns via polar coordinates.
        Captures long-range dependencies.
        
        Args:
            X: [B, L, D] normalized time series
            
        Returns:
            [B, H, W] GAF image
        """
        B, L, D = X.shape
        H, W = self.image_size
        
        # Average over features: [B, L]
        X_avg = X.mean(axis=-1)
        
        # Convert to polar coordinates
        # x = r*cos(theta), where r=1 and theta = arccos(x)
        # X is already in [0, 1], map to [-1, 1] for arccos
        X_scaled = 2 * X_avg - 1  # [0,1] → [-1,1]
        X_scaled = np.clip(X_scaled, -1, 1)
        
        theta = np.arccos(X_scaled)  # [B, L]
        
        # Compute GAF: cos(theta_i + theta_j) = outer sum
        # For efficiency, use matrix multiplication trick:
        # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        cos_theta = np.cos(theta)  # [B, L]
        sin_theta = np.sin(theta)  # [B, L]
        
        # GAF matrix: [B, L, L]
        gaf_matrix = np.einsum('bi,bj->bij', cos_theta, cos_theta) - \
                     np.einsum('bi,bj->bij', sin_theta, sin_theta)
        
        # Interpolate to target size
        gaf_image = self._bilinear_resize(gaf_matrix, H, W)
        
        # Normalize per channel
        if self.normalize:
            gaf_image = self._normalize_channel(gaf_image)
            
        return gaf_image
    
    def _recurrence_plot(self, X: np.ndarray) -> np.ndarray:
        """
        Recurrence Plot (RP) encoding.
        
        Constructs similarity matrices revealing cyclical behaviors and anomalies.
        
        Args:
            X: [B, L, D] normalized time series
            
        Returns:
            [B, H, W] RP image
        """
        B, L, D = X.shape
        H, W = self.image_size
        
        # Average over features: [B, L]
        X_avg = X.mean(axis=-1)
        
        # Compute pairwise distances: [B, L, L]
        # Use Gaussian kernel: exp(-||x_i - x_j||^2 / 2)
        rp_matrix = np.zeros((B, L, L))
        for b in range(B):
            diffs = X_avg[b, :, np.newaxis] - X_avg[b, np.newaxis, :]  # [L, L]
            rp_matrix[b] = np.exp(-diffs**2 / 2)
        
        # Interpolate to target size
        rp_image = self._bilinear_resize(rp_matrix, H, W)
        
        # Normalize per channel
        if self.normalize:
            rp_image = self._normalize_channel(rp_image)
            
        return rp_image
    
    def _detect_period(self, x: np.ndarray, max_period: int = 100) -> int:
        """
        Auto-detect period using FFT.
        
        Args:
            x: [L] time series
            max_period: Maximum period to consider
            
        Returns:
            Detected period (default to 24 if no clear peak)
        """
        L = len(x)
        if L < 48:
            return min(24, L // 2)
            
        # FFT
        fft = np.fft.fft(x - x.mean())
        power = np.abs(fft[:L//2])**2
        
        # Find dominant frequency (skip DC component)
        freqs = np.fft.fftfreq(L, d=1.0)[:L//2]
        valid = (freqs > 0) & (freqs < 1.0 / 2)  # Nyquist
        
        if valid.sum() == 0:
            return 24
            
        power_valid = power[valid]
        freqs_valid = freqs[valid]
        
        # Peak frequency → period
        peak_idx = np.argmax(power_valid)
        peak_freq = freqs_valid[peak_idx]
        period = int(1.0 / peak_freq) if peak_freq > 0 else 24
        
        # Clamp to reasonable range
        period = np.clip(period, 2, max_period)
        
        return period
    
    def _bilinear_resize(
        self,
        matrix: np.ndarray,
        target_h: int,
        target_w: int
    ) -> np.ndarray:
        """
        Bilinear interpolation to resize matrix.
        
        Args:
            matrix: [B, H_in, W_in] input matrix
            target_h: Target height
            target_w: Target width
            
        Returns:
            [B, target_h, target_w] resized matrix
        """
        matrix_tensor = torch.from_numpy(matrix).float()
        
        # Add channel dim: [B, H, W] → [B, 1, H, W]
        matrix_tensor = matrix_tensor.unsqueeze(1)
        
        # Resize
        resized = F.interpolate(
            matrix_tensor,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Remove channel dim: [B, 1, H, W] → [B, H, W]
        resized = resized.squeeze(1)
        
        return resized.numpy()
    
    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        Normalize each channel independently to [0, 1].
        
        Args:
            channel: [B, H, W] channel
            
        Returns:
            [B, H, W] normalized channel
        """
        B, H, W = channel.shape
        normalized = np.zeros_like(channel)
        
        for b in range(B):
            c_min = channel[b].min()
            c_max = channel[b].max()
            if c_max > c_min:
                normalized[b] = (channel[b] - c_min) / (c_max - c_min)
            else:
                normalized[b] = 0.5  # Constant signal
                
        return normalized


# Utility functions for integration
def ohlcv_to_vision(
    ohlcv: np.ndarray,
    use_close_only: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Convert OHLCV data to vision representation.
    
    Args:
        ohlcv: [L, 5] or [B, L, 5] OHLCV data (open, high, low, close, volume)
        use_close_only: If True, use only close prices
        **kwargs: Additional args for TimeSeriesVisionEncoder
        
    Returns:
        RGB image [B, 3, H, W] or [3, H, W]
    """
    encoder = TimeSeriesVisionEncoder(**kwargs)
    
    if use_close_only:
        # Use close prices only
        if ohlcv.ndim == 2:
            X = ohlcv[:, 3:4]  # [L, 1]
        else:
            X = ohlcv[:, :, 3:4]  # [B, L, 1]
    else:
        # Use all OHLCV
        X = ohlcv
        
    return encoder.encode(X, return_tensor=True)


if __name__ == "__main__":
    # Test
    logger.info("Testing TimeSeriesVisionEncoder...")
    
    # Generate sample OHLCV
    L = 100
    D = 5
    X = np.random.randn(L, D).cumsum(axis=0)  # Random walk
    
    # Encode
    encoder = TimeSeriesVisionEncoder(image_size=(224, 224))
    rgb_image = encoder.encode(X)
    
    logger.info(f"Input shape: {X.shape}")
    logger.info(f"Output shape: {rgb_image.shape}")
    logger.info(f"Output range: [{rgb_image.min():.3f}, {rgb_image.max():.3f}]")
    
    # Visualize (optional)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_image[0].numpy(), cmap='hot')
        axes[0].set_title('Segmentation (R)')
        axes[1].imshow(rgb_image[1].numpy(), cmap='hot')
        axes[1].set_title('GAF (G)')
        axes[2].imshow(rgb_image[2].numpy(), cmap='hot')
        axes[2].set_title('Recurrence Plot (B)')
        plt.tight_layout()
        plt.savefig('vision_transforms_test.png')
        logger.info("Saved visualization to vision_transforms_test.png")
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
