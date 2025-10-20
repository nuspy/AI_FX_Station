"""
Real-Time Inference API Service

Fast API service for model inference with <100ms latency target.
Includes caching, batching, and monitoring.

Reference: Production ML deployment best practices
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from joblib import load
from loguru import logger

# Optional Redis caching
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - caching disabled")


# Request/Response models
class PredictionRequest(BaseModel):
    symbol: str = Field(..., example="EUR/USD")
    timeframe: str = Field(..., example="5m")
    candles: List[Dict[str, Any]] = Field(..., min_items=1)
    model_id: Optional[str] = None


class PredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    prediction: float
    confidence: float
    model_id: str
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    cache_enabled: bool
    uptime_seconds: float


# Inference Service
class InferenceService:
    """
    Real-time inference service with caching and monitoring.
    """

    def __init__(
        self,
        models_dir: Path,
        redis_url: Optional[str] = None,
        cache_ttl: int = 300,  # 5 minutes
        max_batch_size: int = 32,
        max_batch_wait_ms: int = 50,
    ):
        """
        Initialize inference service.

        Args:
            models_dir: Directory containing trained models
            redis_url: Redis connection URL for caching
            cache_ttl: Cache TTL in seconds
            max_batch_size: Maximum batch size for inference
            max_batch_wait_ms: Maximum wait time for batching
        """
        self.models_dir = Path(models_dir)
        self.cache_ttl = cache_ttl
        self.max_batch_size = max_batch_size
        self.max_batch_wait_ms = max_batch_wait_ms

        # Model cache
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}

        # Redis cache
        self.redis_client: Optional[redis.Redis] = None
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                logger.info(f"Redis cache enabled: {redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")

        # Metrics
        self.start_time = datetime.now()
        self.request_count = 0
        self.cache_hits = 0
        self.total_latency_ms = 0.0

        # Batch queue
        self.batch_queue: List[tuple] = []
        self.batch_lock = asyncio.Lock()

    async def load_models(self):
        """Load all models from models directory."""
        model_files = list(self.models_dir.glob("*.pkl"))

        for model_path in model_files:
            try:
                model_id = model_path.stem
                payload = load(model_path)

                self.models[model_id] = payload["model"]
                self.model_metadata[model_id] = {
                    "features": payload.get("features", []),
                    "scaler_mu": payload.get("scaler_mu"),
                    "scaler_sigma": payload.get("scaler_sigma"),
                    "encoder": payload.get("encoder"),
                    "model_type": payload.get("model_type", "unknown"),
                    "val_mae": payload.get("val_mae", 0.0),
                    "loaded_at": datetime.now().isoformat(),
                }

                logger.info(f"Loaded model: {model_id} ({payload.get('model_type')})")

            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")

        logger.info(f"Loaded {len(self.models)} models")

    async def _get_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for request."""
        # Use last candle timestamp + symbol + timeframe
        last_ts = request.candles[-1].get("ts_utc", 0)
        return f"pred:{request.symbol}:{request.timeframe}:{last_ts}"

    async def _get_cached_prediction(self, cache_key: str) -> Optional[Dict]:
        """Get prediction from cache."""
        if not self.redis_client:
            return None

        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                self.cache_hits += 1
                import pickle
                return pickle.loads(cached)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")

        return None

    async def _set_cached_prediction(self, cache_key: str, result: Dict):
        """Cache prediction result."""
        if not self.redis_client:
            return

        try:
            import pickle
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                pickle.dumps(result)
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def _preprocess_features(
        self,
        candles: List[Dict],
        model_id: str,
    ) -> np.ndarray:
        """
        Preprocess candles into feature vector.

        Args:
            candles: List of OHLCV candles
            model_id: Model ID for feature extraction

        Returns:
            Feature vector ready for inference
        """
        metadata = self.model_metadata.get(model_id, {})
        features = metadata.get("features", [])

        if not features:
            raise ValueError(f"No features defined for model {model_id}")

        # Convert to DataFrame
        df = pd.DataFrame(candles)

        # Extract features (simplified - real implementation would use FeatureEngineer)
        X = pd.DataFrame()

        for feat in features:
            if feat in df.columns:
                X[feat] = df[feat]
            else:
                # Missing feature - fill with 0
                X[feat] = 0.0

        # Standardize
        mu = metadata.get("scaler_mu")
        sigma = metadata.get("scaler_sigma")

        if mu is not None and sigma is not None:
            X = (X - mu) / sigma

        # Apply encoder if exists
        encoder = metadata.get("encoder")
        if encoder is not None:
            X = encoder.transform(X.values)

        # Return last row (most recent)
        return X[-1:].values if len(X.shape) > 1 else X.values.reshape(1, -1)

    async def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        """
        Run prediction for single request.

        Args:
            request: Prediction request

        Returns:
            Prediction response with latency tracking
        """
        start_time = datetime.now()

        # Check cache
        cache_key = await self._get_cache_key(request)
        cached = await self._get_cached_prediction(cache_key)

        if cached:
            cached["latency_ms"] = (datetime.now() - start_time).total_seconds() * 1000
            return PredictionResponse(**cached)

        # Select model
        model_id = request.model_id
        if not model_id:
            # Use first available model
            if not self.models:
                raise HTTPException(status_code=503, detail="No models loaded")
            model_id = list(self.models.keys())[0]

        if model_id not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        try:
            # Preprocess
            X = self._preprocess_features(request.candles, model_id)

            # Inference
            model = self.models[model_id]
            prediction = model.predict(X)[0]

            # Calculate confidence (simplified - could use prediction intervals)
            val_mae = self.model_metadata[model_id].get("val_mae", 0.01)
            confidence = max(0.0, min(1.0, 1.0 - abs(prediction) / (val_mae * 3)))

            # Build response
            result = {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "prediction": float(prediction),
                "confidence": float(confidence),
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            await self._set_cached_prediction(cache_key, result)

            # Track metrics
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.request_count += 1
            self.total_latency_ms += latency_ms

            result["latency_ms"] = latency_ms

            return PredictionResponse(**result)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_health(self) -> HealthResponse:
        """Get service health status."""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return HealthResponse(
            status="healthy" if self.models else "no_models",
            models_loaded=len(self.models),
            cache_enabled=self.redis_client is not None,
            uptime_seconds=uptime,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        avg_latency = (
            self.total_latency_ms / self.request_count
            if self.request_count > 0
            else 0.0
        )

        cache_hit_rate = (
            self.cache_hits / self.request_count
            if self.request_count > 0
            else 0.0
        )

        return {
            "request_count": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_latency_ms": avg_latency,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
        }


# FastAPI app
def create_app(
    models_dir: str = "artifacts/models",
    redis_url: Optional[str] = None,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        models_dir: Directory containing trained models
        redis_url: Optional Redis URL for caching

    Returns:
        FastAPI app instance
    """
    app = FastAPI(
        title="ForexGPT Inference API",
        description="Real-time forex prediction inference service",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize service
    service = InferenceService(
        models_dir=Path(models_dir),
        redis_url=redis_url,
    )

    @app.on_event("startup")
    async def startup():
        """Load models on startup."""
        await service.load_models()

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Run prediction."""
        return await service.predict(request)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return service.get_health()

    @app.get("/metrics")
    async def metrics():
        """Get service metrics."""
        return service.get_metrics()

    @app.get("/models")
    async def list_models():
        """List available models."""
        return {
            "models": [
                {
                    "model_id": model_id,
                    "model_type": meta.get("model_type"),
                    "val_mae": meta.get("val_mae"),
                    "features_count": len(meta.get("features", [])),
                    "loaded_at": meta.get("loaded_at"),
                }
                for model_id, meta in service.model_metadata.items()
            ]
        }

    return app


# Entry point
if __name__ == "__main__":
    import uvicorn

    app = create_app(
        models_dir="artifacts/models",
        redis_url="redis://localhost:6379/0",
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
