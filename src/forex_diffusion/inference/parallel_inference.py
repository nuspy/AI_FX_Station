"""
Parallel Model Inference System for efficient multi-model predictions.

Enables running multiple models concurrently to improve performance and
provide ensemble predictions with different models and configurations.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

from ..models.standardized_loader import get_model_loader
from ..models.model_path_resolver import ModelPathResolver
from ..utils.horizon_converter import convert_horizons_for_inference


class ParallelInferenceError(Exception):
    """Raised when parallel inference fails."""
    pass


class ModelExecutor:
    """
    Individual model executor for parallel processing.
    Handles loading and running a single model independently.
    """

    def __init__(self, model_path: str, model_config: Dict[str, Any]):
        self.model_path = model_path
        self.model_config = model_config
        self.model_data = None
        self.is_loaded = False

    def load_model(self) -> None:
        """Load the model for this executor."""
        try:
            loader = get_model_loader()
            self.model_data = loader.load_single_model(self.model_path)
            self.is_loaded = True
            logger.debug(f"Model loaded in executor: {Path(self.model_path).name}")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            raise

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction using the loaded model.

        Args:
            features_df: Prepared features dataframe

        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded or self.model_data is None:
            raise RuntimeError(f"Model not loaded: {self.model_path}")

        try:
            start_time = time.time()

            model = self.model_data['model']
            features_list = self.model_data.get('features', [])

            # Filter features to match model requirements
            if features_list:
                available_features = [f for f in features_list if f in features_df.columns]
                if len(available_features) != len(features_list):
                    missing = set(features_list) - set(available_features)
                    logger.warning(f"Missing features for {Path(self.model_path).name}: {missing}")

                X = features_df[available_features].values
            else:
                X = features_df.values

            # Handle standardization
            scaler = self.model_data.get('scaler') or self.model_data.get('standardizer')
            if scaler:
                std_mu = scaler.get('mu', {}) if isinstance(scaler, dict) else {}
                std_sigma = scaler.get('sigma', {}) if isinstance(scaler, dict) else {}

                if std_mu and std_sigma:
                    # Apply standardization
                    for i, feature in enumerate(available_features if features_list else range(X.shape[1])):
                        if feature in std_mu and feature in std_sigma:
                            mu = std_mu[feature]
                            sigma = std_sigma[feature]
                            if sigma > 0:
                                X[:, i] = (X[:, i] - mu) / sigma

            # Apply dimensionality reduction (PCA/VAE) if present
            encoder = self.model_data.get('pca') or self.model_data.get('encoder')
            if encoder is not None:
                try:
                    X = encoder.transform(X)
                    logger.debug(f"Applied {self.model_data.get('encoder_type', 'encoder')} transform: {X.shape}")
                except Exception as e:
                    logger.warning(f"Failed to apply encoder transform: {e}")

            # Make prediction
            X_last = X[-1:] if len(X) > 0 else np.zeros((1, X.shape[1] if X.ndim > 1 else 1))

            predictions = None
            try:
                # Try PyTorch model
                import torch
                if hasattr(model, 'eval'):
                    model.eval()
                with torch.no_grad():
                    t_in = torch.tensor(X_last, dtype=torch.float32)
                    out = model(t_in)
                    predictions = np.ravel(out.detach().cpu().numpy())
            except Exception:
                # Try sklearn model
                if hasattr(model, "predict"):
                    predictions = np.ravel(model.predict(X_last))

            if predictions is None:
                # NO FALLBACK! Model must have predict method or fail
                raise RuntimeError(f"Model {Path(self.model_path).name} has no predict method")

            execution_time = time.time() - start_time

            return {
                'model_path': self.model_path,
                'model_name': Path(self.model_path).stem,
                'predictions': predictions,
                'features_used': available_features if features_list else list(range(X.shape[1])),
                'execution_time': execution_time,
                'model_type': self.model_data.get('model_type', 'unknown'),
                'success': True
            }

        except Exception as e:
            logger.error(f"Prediction failed for {self.model_path}: {e}")
            return {
                'model_path': self.model_path,
                'model_name': Path(self.model_path).stem,
                'predictions': None,
                'error': str(e),
                'execution_time': 0,
                'success': False
            }


class ParallelInferenceEngine:
    """
    Main parallel inference engine that coordinates multiple model executors.

    Features:
    - Concurrent model loading and execution
    - Configurable thread pool size
    - Error handling and result aggregation
    - Performance monitoring
    """

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, (Path().cwd().parent / "artifacts").exists() and 8 or 2)
        self.path_resolver = ModelPathResolver()

    def run_parallel_inference(
        self,
        settings: Dict[str, Any],
        features_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        horizons: List[str]
    ) -> Dict[str, Any]:
        """
        Run parallel inference with multiple models.

        Args:
            settings: Settings from UI with model paths
            features_df: Prepared features for inference
            symbol: Trading symbol
            timeframe: Data timeframe
            horizons: Prediction horizons

        Returns:
            Dictionary with aggregated results from all models
        """
        # Resolve model paths
        model_paths = self.path_resolver.resolve_model_paths(settings)

        if not model_paths:
            raise ParallelInferenceError("No valid model paths found")

        logger.info(f"Starting parallel inference with {len(model_paths)} models")

        # Convert horizons to proper format
        time_labels, horizon_bars = convert_horizons_for_inference(horizons, timeframe)

        # Create model executors
        executors = []
        for model_path in model_paths:
            config = {
                'symbol': symbol,
                'timeframe': timeframe,
                'horizons': horizons,
                'time_labels': time_labels,
                'horizon_bars': horizon_bars
            }
            executors.append(ModelExecutor(model_path, config))

        # Run parallel inference
        start_time = time.time()
        results = self._execute_parallel(executors, features_df)
        total_time = time.time() - start_time

        # Aggregate results
        aggregated = self._aggregate_results(results, symbol, timeframe, time_labels, horizon_bars)
        aggregated['execution_summary'] = {
            'total_models': len(model_paths),
            'successful_models': len([r for r in results if r['success']]),
            'total_execution_time': total_time,
            'average_model_time': np.mean([r['execution_time'] for r in results if r['success']]),
            'parallel_speedup': sum(r['execution_time'] for r in results if r['success']) / max(total_time, 0.001)
        }

        return aggregated

    def _execute_parallel(self, executors: List[ModelExecutor], features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Execute model predictions in parallel using thread pool."""
        results = []

        def load_and_predict(executor):
            try:
                executor.load_model()
                return executor.predict(features_df)
            except Exception as e:
                return {
                    'model_path': executor.model_path,
                    'model_name': Path(executor.model_path).stem,
                    'predictions': None,
                    'error': str(e),
                    'execution_time': 0,
                    'success': False
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(load_and_predict, model_executor): model_executor
                      for model_executor in executors}

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                model_executor = futures[future]
                try:
                    result = future.result(timeout=30)  # 30-second timeout per model
                    results.append(result)

                except concurrent.futures.TimeoutError:
                    logger.error(f"Model execution timed out: {model_executor.model_path}")
                    results.append({
                        'model_path': model_executor.model_path,
                        'model_name': Path(model_executor.model_path).stem,
                        'predictions': None,
                        'error': 'Execution timeout',
                        'execution_time': 30,
                        'success': False
                    })

                except Exception as e:
                    logger.error(f"Parallel execution failed for {model_executor.model_path}: {e}")
                    results.append({
                        'model_path': model_executor.model_path,
                        'model_name': Path(model_executor.model_path).stem,
                        'predictions': None,
                        'error': str(e),
                        'execution_time': 0,
                        'success': False
                    })

        return results

    def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
        time_labels: List[str],
        horizon_bars: List[int]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple models into ensemble predictions.

        Args:
            results: List of individual model results
            symbol: Trading symbol
            timeframe: Data timeframe
            time_labels: Time labels for horizons
            horizon_bars: Horizon bars for scaling

        Returns:
            Aggregated results with ensemble predictions
        """
        successful_results = [r for r in results if r['success'] and r['predictions'] is not None]

        if not successful_results:
            logger.error("No successful model predictions to aggregate")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'time_labels': time_labels,
                'horizon_bars': horizon_bars,
                'ensemble_predictions': None,
                'individual_results': results,
                'error': 'All models failed'
            }

        # Extract predictions and scale for horizons
        all_predictions = []
        model_weights = []

        for result in successful_results:
            preds = result['predictions']
            if len(preds) == 1 and len(horizon_bars) > 1:
                # Single-step model: replicate prediction for all horizons
                # LINEAR SCALING IS WRONG - model predicts return for horizon H
                # We cannot multiply by bars count - that's mathematically incorrect
                base_pred = preds[0]
                preds = np.full(len(horizon_bars), base_pred)
                logger.debug(f"Replicated single prediction {base_pred:.6f} to {len(horizon_bars)} horizons")
            elif len(preds) < len(horizon_bars):
                # Extend predictions to cover all horizons
                preds = np.pad(preds, (0, len(horizon_bars) - len(preds)), mode='edge')

            all_predictions.append(preds[:len(horizon_bars)])

            # Weight by inverse of execution time (faster models get higher weight)
            weight = 1.0 / max(result['execution_time'], 0.001)
            model_weights.append(weight)

        # Create ensemble predictions
        all_predictions = np.array(all_predictions)
        model_weights = np.array(model_weights)
        model_weights = model_weights / model_weights.sum()  # Normalize

        # Weighted average
        ensemble_mean = np.average(all_predictions, axis=0, weights=model_weights)

        # Prediction uncertainty (weighted standard deviation)
        ensemble_std = np.sqrt(np.average((all_predictions - ensemble_mean)**2, axis=0, weights=model_weights))

        # Create confidence intervals
        confidence_level = 1.645  # 90% confidence
        ensemble_lower = ensemble_mean - confidence_level * ensemble_std
        ensemble_upper = ensemble_mean + confidence_level * ensemble_std

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'time_labels': time_labels,
            'horizon_bars': horizon_bars,
            'ensemble_predictions': {
                'mean': ensemble_mean.tolist(),
                'std': ensemble_std.tolist(),
                'lower': ensemble_lower.tolist(),
                'upper': ensemble_upper.tolist(),
                'individual': all_predictions.tolist()
            },
            'model_weights': model_weights.tolist(),
            'individual_results': results,
            'successful_models': len(successful_results),
            'total_models': len(results)
        }

    async def run_async_inference(
        self,
        settings: Dict[str, Any],
        features_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        horizons: List[str]
    ) -> Dict[str, Any]:
        """
        Async version of parallel inference for integration with async frameworks.

        Args:
            settings: Settings from UI with model paths
            features_df: Prepared features for inference
            symbol: Trading symbol
            timeframe: Data timeframe
            horizons: Prediction horizons

        Returns:
            Dictionary with aggregated results from all models
        """
        loop = asyncio.get_event_loop()

        # Run the synchronous parallel inference in a thread pool
        return await loop.run_in_executor(
            None,
            self.run_parallel_inference,
            settings,
            features_df,
            symbol,
            timeframe,
            horizons
        )


# Global instance for convenience
_global_engine: Optional[ParallelInferenceEngine] = None

def get_parallel_engine(max_workers: Optional[int] = None) -> ParallelInferenceEngine:
    """Get the global parallel inference engine."""
    global _global_engine
    if _global_engine is None or (max_workers is not None and _global_engine.max_workers != max_workers):
        _global_engine = ParallelInferenceEngine(max_workers)
    return _global_engine