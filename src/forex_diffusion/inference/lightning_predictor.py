"""
Lightning Multi-Horizon Predictor

Handles inference for Lightning models trained with multi-horizon support.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from loguru import logger

from ..utils.horizon_parser import get_model_horizons_from_metadata


class LightningMultiHorizonPredictor:
    """
    Predictor for Lightning models with multi-horizon support.
    
    Features:
    - Loads Lightning checkpoint
    - Extracts horizon metadata
    - Runs diffusion sampling for predictions
    - Returns predictions for all horizons
    - Supports interpolation for missing horizons
    """
    
    def __init__(self, checkpoint_path: str | Path, device: str = 'cpu'):
        """
        Initialize predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to .ckpt file
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.device = device
        
        # Load model
        logger.info(f"Loading Lightning model from {self.checkpoint_path}")
        self._load_model()
        
        # Extract horizons from metadata
        self._extract_horizons()
        
        logger.info(f"Model horizons: {self.horizons} ({'multi' if self.is_multi_horizon else 'single'})")
        logger.info(f"Device: {self.device}")
    
    def _load_model(self):
        """Load Lightning model from checkpoint."""
        try:
            import torch
            from ..train.loop import ForexDiffusionLit
            
            # Load checkpoint
            self.model = ForexDiffusionLit.load_from_checkpoint(
                str(self.checkpoint_path),
                map_location=self.device
            )
            self.model.eval()
            self.model.to(self.device)
            
            logger.info(f"Loaded ForexDiffusionLit model")
            
        except Exception as e:
            logger.error(f"Failed to load Lightning model: {e}")
            raise
    
    def _extract_horizons(self):
        """Extract horizons from model hyperparameters."""
        try:
            hparams = self.model.hparams
            
            # Extract in_channels for patch construction
            self.in_channels = getattr(hparams, 'in_channels', 6)
            logger.debug(f"Model expects {self.in_channels} input channels")
            
            # Try new format first
            if hasattr(hparams, 'horizons'):
                self.horizons = hparams.horizons
            elif hasattr(hparams, 'horizon'):
                horizon = hparams.horizon
                self.horizons = horizon if isinstance(horizon, list) else [horizon]
            else:
                # Fallback
                logger.warning("No horizon info in hparams, using default [60]")
                self.horizons = [60]
            
            self.is_multi_horizon = len(self.horizons) > 1
            self.num_horizons = len(self.horizons)
            
            # Store other useful metadata
            self.patch_len = getattr(hparams, 'patch_len', 64)
            self.symbol = getattr(hparams, 'symbol', 'EUR/USD')
            self.timeframe = getattr(hparams, 'timeframe', '1m')
            
        except Exception as e:
            logger.error(f"Failed to extract horizons: {e}")
            self.horizons = [60]
            self.is_multi_horizon = False
            self.num_horizons = 1
    
    def predict(
        self,
        x: torch.Tensor | np.ndarray,
        num_samples: int = 50,
        return_dict: bool = True,
        return_distribution: bool = False
    ) -> Dict[int, float] | Dict[int, Dict[str, float]]:
        """
        Make predictions for all trained horizons.
        
        Args:
            x: Input patch (B, C, L) or (C, L)
            num_samples: Number of diffusion samples to draw
            return_dict: If True, return dict {horizon: prediction}
            return_distribution: If True, return distribution stats
            
        Returns:
            If return_dict=True and return_distribution=False:
                {15: 0.002, 60: 0.005, 240: 0.012}
            
            If return_dict=True and return_distribution=True:
                {
                    15: {'mean': 0.002, 'std': 0.001, 'q05': 0.001, 'q95': 0.003},
                    60: {'mean': 0.005, 'std': 0.002, 'q05': 0.003, 'q95': 0.007},
                    ...
                }
        """
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Ensure on correct device
        x = x.to(self.device)
        
        # Ensure 3D: (B, C, L)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            # Encode to latent space
            mu, logvar = self.model.vae.encode(x)
            
            # Sample multiple trajectories from posterior
            predictions_all = []  # List of (num_samples, num_horizons)
            
            for _ in range(num_samples):
                # Sample from q(z|x)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                # Multi-horizon prediction
                if self.is_multi_horizon and hasattr(self.model, 'multi_horizon_head') and self.model.multi_horizon_head is not None:
                    # Use dedicated MLP head for proper multi-horizon predictions
                    horizon_preds = self.model.multi_horizon_head(z)  # (B, H)
                    pred_horizons = horizon_preds[0].tolist()  # First batch item, convert to list
                else:
                    # Fallback: decode and extract close from reconstruction
                    x_rec = self.model.vae.decode(z)  # (B, C, L)
                    
                    # Get close channel index
                    close_idx = getattr(self.model.hparams, 'close_channel_idx', None)
                    if close_idx is None:
                        close_idx = 3  # Default: assume 'close' at channel 3
                    close_idx = min(close_idx, x_rec.shape[1] - 1)
                    close_rec = x_rec[0, close_idx, -1]  # Last timestep
                    
                    if self.is_multi_horizon:
                        # Replicate for legacy models without prediction head
                        pred_horizons = [close_rec.item()] * self.num_horizons
                    else:
                        pred_horizons = [close_rec.item()]
                
                predictions_all.append(pred_horizons)
            
            # Convert to numpy for stats
            predictions_all = np.array(predictions_all)  # (num_samples, num_horizons)
        
        # Compute statistics
        if return_dict:
            result = {}
            
            for i, horizon in enumerate(self.horizons):
                samples = predictions_all[:, i]
                
                if return_distribution:
                    result[horizon] = {
                        'mean': float(np.mean(samples)),
                        'std': float(np.std(samples)),
                        'median': float(np.median(samples)),
                        'q05': float(np.percentile(samples, 5)),
                        'q25': float(np.percentile(samples, 25)),
                        'q75': float(np.percentile(samples, 75)),
                        'q95': float(np.percentile(samples, 95)),
                        'samples': samples.tolist() if num_samples <= 100 else None
                    }
                else:
                    result[horizon] = float(np.mean(samples))
            
            return result
        else:
            # Return raw samples array
            return predictions_all
    
    def predict_with_interpolation(
        self,
        x: torch.Tensor | np.ndarray,
        requested_horizons: List[int],
        num_samples: int = 50,
        return_distribution: bool = False
    ) -> Dict[int, float] | Dict[int, Dict[str, float]]:
        """
        Predict for requested horizons, interpolating if needed.
        
        Args:
            x: Input patch
            requested_horizons: Horizons to predict (may include unseen ones)
            num_samples: Number of diffusion samples
            return_distribution: Return distribution stats
            
        Returns:
            Dictionary with predictions for all requested horizons
        """
        # Get predictions for trained horizons
        trained_preds = self.predict(
            x,
            num_samples=num_samples,
            return_dict=True,
            return_distribution=return_distribution
        )
        
        # Check if interpolation needed
        missing_horizons = set(requested_horizons) - set(self.horizons)
        
        if not missing_horizons:
            # All requested horizons are trained - just filter
            result = {h: trained_preds[h] for h in requested_horizons if h in trained_preds}
            return result
        
        # Need interpolation
        logger.warning(f"Interpolating {len(missing_horizons)} horizon(s): {sorted(missing_horizons)}")
        
        result = dict(trained_preds)  # Start with trained predictions
        
        sorted_trained = sorted(self.horizons)
        
        for h in sorted(missing_horizons):
            # Find surrounding trained horizons
            lower = None
            upper = None
            
            for th in sorted_trained:
                if th < h:
                    lower = th
                elif th > h and upper is None:
                    upper = th
                    break
            
            if lower is not None and upper is not None:
                # Linear interpolation
                weight = (h - lower) / (upper - lower)
                
                if return_distribution:
                    # Interpolate each statistic
                    result[h] = {}
                    for key in ['mean', 'std', 'median', 'q05', 'q25', 'q75', 'q95']:
                        lower_val = trained_preds[lower][key]
                        upper_val = trained_preds[upper][key]
                        result[h][key] = lower_val * (1 - weight) + upper_val * weight
                else:
                    # Interpolate mean
                    lower_val = trained_preds[lower]
                    upper_val = trained_preds[upper]
                    result[h] = lower_val * (1 - weight) + upper_val * weight
                
                logger.debug(f"Interpolated horizon {h}: {lower} ({1-weight:.3f}) + {upper} ({weight:.3f})")
            else:
                # Cannot interpolate (outside range)
                logger.error(f"Cannot interpolate horizon {h} - outside trained range {sorted_trained}")
                result[h] = 0.0 if not return_distribution else {'mean': 0.0}
        
        # Filter to requested horizons
        result = {h: result[h] for h in requested_horizons if h in result}
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'checkpoint_path': str(self.checkpoint_path),
            'model_type': 'lightning',
            'horizons': self.horizons,
            'is_multi_horizon': self.is_multi_horizon,
            'num_horizons': self.num_horizons,
            'patch_len': self.patch_len,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'device': str(self.device),
        }
