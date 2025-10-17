"""
LDM4TS Inference Service

Real-time inference with caching and performance monitoring.
"""
from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger
import time

from ..models.ldm4ts import LDM4TSModel


@dataclass
class LDM4TSPrediction:
    """Prediction result with uncertainty."""
    
    asset: str
    timestamp: pd.Timestamp
    horizons: List[int]
    
    mean: Dict[int, float]
    std: Dict[int, float]
    q05: Dict[int, float]
    q50: Dict[int, float]
    q95: Dict[int, float]
    
    inference_time_ms: float
    model_name: str = "LDM4TS"
    num_samples: int = 50
    
    def to_dict(self) -> Dict:
        return {
            "asset": self.asset,
            "timestamp": self.timestamp.isoformat(),
            "horizons": self.horizons,
            "predictions": {
                str(h): {
                    "mean": float(self.mean[h]),
                    "std": float(self.std[h]),
                    "q05": float(self.q05[h]),
                    "q50": float(self.q50[h]),
                    "q95": float(self.q95[h])
                }
                for h in self.horizons
            },
            "metadata": {
                "inference_time_ms": self.inference_time_ms,
                "model_name": self.model_name,
                "num_samples": self.num_samples
            }
        }
    
    def to_quantiles_format(self) -> Dict[str, List[float]]:
        """For ForecastService chart overlay."""
        return {
            "q05": [self.q05[h] for h in self.horizons],
            "q50": [self.q50[h] for h in self.horizons],
            "q95": [self.q95[h] for h in self.horizons],
            "std": [self.std[h] for h in self.horizons],
            "future_ts": [
                self.timestamp + pd.Timedelta(minutes=h)
                for h in self.horizons
            ],
            "model_name": self.model_name
        }


class LDM4TSInferenceService:
    """Singleton inference service for LDM4TS."""
    
    _instance = None
    _lock = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.model: Optional[LDM4TSModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False
        logger.info(f"LDM4TSInferenceService created (device={self.device})")
    
    @classmethod
    def get_instance(
        cls,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> "LDM4TSInferenceService":
        """Get or create singleton instance."""
        instance = cls()
        
        if checkpoint_path and not instance._initialized:
            instance.load_model(checkpoint_path, **kwargs)
        
        return instance
    
    def load_model(
        self,
        checkpoint_path: str,
        horizons: List[int] = [15, 60, 240],
        compile_model: bool = False
    ):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            logger.info("Creating fresh model (no pre-trained weights)")
        
        try:
            self.model = LDM4TSModel(
                image_size=(224, 224),
                horizons=horizons,
                device=str(self.device)
            )
            
            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict['model_state_dict'])
                logger.info(f"Loaded checkpoint: {checkpoint_path}")
            
            self.model.eval()
            
            if compile_model and hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(
        self,
        ohlcv: np.ndarray,
        horizons: Optional[List[int]] = None,
        num_samples: int = 50,
        symbol: str = "EUR/USD"
    ) -> LDM4TSPrediction:
        """
        Run inference.
        
        Args:
            ohlcv: [L, 5] OHLCV array
            horizons: Forecast horizons (None = use model default)
            num_samples: Monte Carlo samples
            symbol: Trading symbol
            
        Returns:
            LDM4TSPrediction object
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.perf_counter()
        
        # Get current price
        current_price = float(ohlcv[-1, 3])  # Last close
        
        # Convert to tensor
        ohlcv_tensor = torch.from_numpy(ohlcv).float()
        if ohlcv_tensor.ndim == 2:
            ohlcv_tensor = ohlcv_tensor.unsqueeze(0)  # [1, L, 5]
        
        # Inference
        with torch.no_grad():
            result = self.model(
                ohlcv_tensor,
                current_price=current_price,
                num_samples=num_samples,
                return_all=False
            )
        
        # Extract predictions
        horizons_used = horizons or self.model.horizons
        
        mean_dict = {h: float(result['mean'][0, i]) for i, h in enumerate(horizons_used)}
        std_dict = {h: float(result['std'][0, i]) for i, h in enumerate(horizons_used)}
        q05_dict = {h: float(result['q05'][0, i]) for i, h in enumerate(horizons_used)}
        q50_dict = {h: float(result['q50'][0, i]) for i, h in enumerate(horizons_used)}
        q95_dict = {h: float(result['q95'][0, i]) for i, h in enumerate(horizons_used)}
        
        # Timing
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        prediction = LDM4TSPrediction(
            asset=symbol,
            timestamp=pd.Timestamp.now(),
            horizons=horizons_used,
            mean=mean_dict,
            std=std_dict,
            q05=q05_dict,
            q50=q50_dict,
            q95=q95_dict,
            inference_time_ms=inference_time_ms,
            num_samples=num_samples
        )
        
        logger.debug(
            f"LDM4TS prediction: {symbol}, "
            f"horizons={horizons_used}, "
            f"time={inference_time_ms:.1f}ms"
        )
        
        return prediction


if __name__ == "__main__":
    logger.info("Testing LDM4TSInferenceService...")
    
    # Create service
    service = LDM4TSInferenceService.get_instance()
    service.load_model("dummy_checkpoint.ckpt", horizons=[15, 60, 240])
    
    # Generate test data
    ohlcv = np.random.randn(100, 5).cumsum(axis=0) + 1.05
    
    # Predict
    prediction = service.predict(ohlcv, num_samples=10, symbol="EUR/USD")
    
    logger.info(f"Prediction: {prediction.to_dict()}")
    logger.info(f"Inference time: {prediction.inference_time_ms:.1f}ms")
    
    logger.info("✅ LDM4TSInferenceService test passed!")
