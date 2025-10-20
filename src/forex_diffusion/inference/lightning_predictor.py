"""
Lightning Multi-Horizon Predictor

Handles inference for Lightning models trained with multi-horizon support.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from loguru import logger



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
    
    def __init__(self, checkpoint_path: str | Path, device: str = 'cpu', metadata: Optional[Dict[str, Any]] = None):
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
        # Normalize metadata to plain dict for simpler access
        if metadata is None:
            self.metadata = {}
        elif isinstance(metadata, dict):
            self.metadata = metadata
        else:
            self.metadata = getattr(metadata, '__dict__', {})

        # Initialize predictor attributes with safe defaults before extraction
        self.channel_order: Optional[List[str]] = None
        self.in_channels: int = 0
        self.channel_mu: Optional[List[float]] = None
        self.channel_sigma: Optional[List[float]] = None
        self.patch_len: Optional[int] = None
        self.symbol: Optional[str] = None
        self.timeframe: Optional[str] = None
        self.horizons: List[int] = []
        self.is_multi_horizon: bool = False
        self.num_horizons: int = 0
        self.price_mu: Optional[float] = None
        self.price_sigma: Optional[float] = None
        
        # Load model
        logger.info(f"Loading Lightning model from {self.checkpoint_path}")
        self._load_model()
        
        # Extract horizons from metadata
        self._extract_horizons()

        # Infer de-normalization stats for price outputs
        if self.channel_mu and self.channel_sigma:
            try:
                mu_idx = self.channel_order.index('close') if self.channel_order else 3
            except ValueError:
                mu_idx = 3
            try:
                self.price_mu = float(self.channel_mu[mu_idx])
                self.price_sigma = float(self.channel_sigma[mu_idx])
            except Exception:
                self.price_mu = None
                self.price_sigma = None

        if hasattr(self, 'model') and self.model is not None:
            setattr(self.model, 'price_mu', self.price_mu)
            setattr(self.model, 'price_sigma', self.price_sigma)
        
        logger.info(f"Model horizons: {self.horizons} ({'multi' if self.is_multi_horizon else 'single'})")
        logger.info(f"Device: {self.device}")
    
    def _load_model(self):
        """Load Lightning model from checkpoint."""
        try:
            from ..train.loop import ForexDiffusionLit
            import torch
            from types import SimpleNamespace

            checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
            state_dict = checkpoint.get('state_dict', {})

            if not state_dict:
                raise RuntimeError("Checkpoint is missing state_dict")

            sanitized_state_dict = {}
            renamed_keys = []
            for key, value in state_dict.items():
                sanitized_key = key
                if '._orig_mod.' in sanitized_key:
                    sanitized_key = sanitized_key.replace('._orig_mod.', '.')
                    renamed_keys.append((key, sanitized_key))
                sanitized_state_dict[sanitized_key] = value

            model = ForexDiffusionLit()

            hyper_parameters = checkpoint.get('hyper_parameters')
            if hyper_parameters:
                try:
                    model.save_hyperparameters(hyper_parameters)
                except Exception:
                    try:
                        model.hparams = SimpleNamespace(**hyper_parameters)
                    except Exception:
                        logger.warning("Failed to attach hyperparameters from checkpoint")

            missing, unexpected = model.load_state_dict(sanitized_state_dict, strict=False)
            if renamed_keys:
                rename_msg = ", ".join([f"{old}->{new}" for old, new in renamed_keys[:5]])
                if len(renamed_keys) > 5:
                    rename_msg += ", ..."
                logger.info(f"Sanitized {len(renamed_keys)} state_dict keys for torch.compile artifacts: {rename_msg}")
            if missing:
                logger.warning(f"Missing keys when loading Lightning model: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading Lightning model: {unexpected}")

            model.eval()
            model.to(self.device)
            self.model = model
            
            logger.info("Loaded ForexDiffusionLit model")
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to load Lightning model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _extract_horizons(self):
        """Extract horizons from model hyperparameters."""
        try:
            hparams = getattr(self.model, 'hparams', None)

            sidecar = self.metadata or {}

            # Channel order / normalization / patch length prefer sidecar
            channel_order = sidecar.get('channel_order') or self.metadata.get('channel_order')
            if channel_order:
                self.channel_order = channel_order
                self.in_channels = len(channel_order)
            elif hparams is not None:
                if hasattr(hparams, 'channel_order'):
                    self.channel_order = list(getattr(hparams, 'channel_order'))
                    self.in_channels = len(self.channel_order)
                elif hasattr(hparams, 'in_channels'):
                    self.in_channels = getattr(hparams, 'in_channels', self.in_channels)
                else:
                    self.in_channels = getattr(hparams, 'in_channels', self.in_channels)

            mu = sidecar.get('mu')
            sigma = sidecar.get('sigma')
            if mu and sigma:
                self.channel_mu = mu
                self.channel_sigma = sigma

            patch_len = sidecar.get('patch_len')
            if patch_len:
                self.patch_len = patch_len
            elif hparams is not None and hasattr(hparams, 'patch_len'):
                self.patch_len = getattr(hparams, 'patch_len', self.patch_len)

            symbol = sidecar.get('symbol')
            if symbol:
                self.symbol = symbol
            elif hparams is not None and hasattr(hparams, 'symbol'):
                self.symbol = getattr(hparams, 'symbol', self.symbol)

            timeframe = sidecar.get('timeframe')
            if timeframe:
                self.timeframe = timeframe
            elif hparams is not None and hasattr(hparams, 'timeframe'):
                self.timeframe = getattr(hparams, 'timeframe', self.timeframe)

            if not self.in_channels:
                if self.channel_order:
                    self.in_channels = len(self.channel_order)
                elif hparams is not None and hasattr(hparams, 'in_channels') and getattr(hparams, 'in_channels'):
                    try:
                        self.in_channels = int(getattr(hparams, 'in_channels'))
                    except (TypeError, ValueError):
                        self.in_channels = 0
                else:
                    fallback_in_channels = getattr(self.model, 'in_channels', None)
                    if isinstance(fallback_in_channels, int) and fallback_in_channels > 0:
                        self.in_channels = fallback_in_channels
                    else:
                        self.in_channels = 6  # Default OHLCV + time encodings

            logger.debug(f"Model expects {self.in_channels} input channels")

            horizons = sidecar.get('horizons')
            if horizons:
                self.horizons = horizons
            elif hparams is not None:
                if hasattr(hparams, 'horizons'):
                    self.horizons = getattr(hparams, 'horizons')
                elif hasattr(hparams, 'horizon'):
                    horizon_val = getattr(hparams, 'horizon')
                    self.horizons = horizon_val if isinstance(horizon_val, list) else [horizon_val]
                else:
                    self.horizons = [60]
            else:
                self.horizons = [60]

            self.is_multi_horizon = len(self.horizons) > 1
            self.num_horizons = len(self.horizons)

            if not self.patch_len:
                fallback_patch_len = getattr(self.model, 'patch_len', None)
                if isinstance(fallback_patch_len, int) and fallback_patch_len > 0:
                    self.patch_len = fallback_patch_len
                else:
                    self.patch_len = 64
            
        except Exception as e:
            logger.error(f"Failed to extract horizons: {e}")
            self.horizons = [60]
            self.is_multi_horizon = False
            self.num_horizons = 1
            if not self.patch_len:
                self.patch_len = 64
    
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

            predictions_all = []  # (num_samples, num_horizons)

            for _ in range(num_samples):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std

                if self.is_multi_horizon and hasattr(self.model, 'multi_horizon_head') and self.model.multi_horizon_head is not None:
                    horizon_preds = self.model.multi_horizon_head(z)[0]  # (H,)
                else:
                    x_rec = self.model.vae.decode(z)
                    close_idx = getattr(self.model.hparams, 'close_channel_idx', None)
                    close_idx = 3 if close_idx is None else min(close_idx, x_rec.shape[1] - 1)
                    close_val = x_rec[0, close_idx, -1]
                    horizon_preds = torch.full((self.num_horizons,), close_val, device=z.device)

                predictions_all.append(horizon_preds.detach().cpu().numpy())

            predictions_all = np.array(predictions_all, dtype=float)
            raw_mean = predictions_all.mean(axis=0)
            raw_std = predictions_all.std(axis=0)
        
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

                logger.debug(
                    "[LightningPredictor] horizon=%s raw_mean=%.6f raw_std=%.6f",
                    horizon,
                    raw_mean[i],
                    raw_std[i]
                )
            
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
