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

# Import DeviceManager for GPU support
try:
    from ..utils.device_manager import DeviceManager
except ImportError:
    DeviceManager = None


class ParallelInferenceError(Exception):
    """Raised when parallel inference fails."""
    pass


class ModelExecutor:
    """
    Individual model executor for parallel processing.
    Handles loading and running a single model independently.
    
    CRITICAL-003: Added memory management with proper cleanup.
    """

    def __init__(self, model_path: str, model_config: Dict[str, Any], use_gpu: bool = False):
        self.model_path = model_path
        self.model_config = model_config
        self.model_data = None
        self.is_loaded = False
        self.use_gpu = use_gpu

        # Setup device
        if DeviceManager and use_gpu:
            self.device = DeviceManager.get_device("cuda")
            logger.debug(f"ModelExecutor will use GPU: {self.device}")
        else:
            import torch
            self.device = torch.device("cpu")
    
    def __del__(self):
        """CRITICAL-003: Cleanup when executor is destroyed"""
        self.unload_model()
    
    def unload_model(self):
        """CRITICAL-003: Explicitly unload model from memory"""
        if self.is_loaded and self.model_data is not None:
            try:
                # Move model to CPU first if on GPU
                if self.use_gpu and hasattr(self.model_data.get('model'), 'cpu'):
                    self.model_data['model'].cpu()
                
                # Clear model data
                del self.model_data
                self.model_data = None
                self.is_loaded = False
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache if using GPU
                if self.use_gpu:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                logger.debug(f"Model unloaded: {Path(self.model_path).name}")
                
            except Exception as e:
                logger.warning(f"Error unloading model: {e}")

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

    def predict(self, features_df: pd.DataFrame, candles_df: pd.DataFrame = None, 
                requested_horizons: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Make prediction using the loaded model (multi-horizon aware).

        Args:
            features_df: Prepared features dataframe
            candles_df: Raw OHLCV candles dataframe (optional, required for diffusion models)
            requested_horizons: Specific horizons to predict (None = use model's trained horizons)

        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded or self.model_data is None:
            raise RuntimeError(f"Model not loaded: {self.model_path}")

        try:
            start_time = time.time()

            model = self.model_data['model']
            features_list = self.model_data.get('features', [])
            metadata = self.model_data.get('metadata')

            # Check if model supports multi-horizon
            model_horizons = None
            is_multi_horizon = False
            
            if metadata:
                model_horizons = metadata.get('horizons') or metadata.get('horizon')
                if model_horizons and not isinstance(model_horizons, list):
                    model_horizons = [model_horizons]
                is_multi_horizon = metadata.get('is_multi_horizon', False) or (model_horizons and len(model_horizons) > 1)
            
            # Determine which horizons to predict
            predict_horizons = requested_horizons if requested_horizons is not None else model_horizons

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
            predictions_dict = None  # For multi-horizon results

            # Try UnifiedPredictor for multi-horizon support
            if is_multi_horizon and predict_horizons:
                try:
                    from ..inference.unified_predictor import UnifiedMultiHorizonPredictor
                    
                    predictor = UnifiedMultiHorizonPredictor(
                        self.model_path,
                        device=str(self.device)
                    )
                    
                    predictions_dict = predictor.predict(
                        X_last,
                        horizons=predict_horizons,
                        num_samples=50,
                        return_distribution=False
                    )
                    
                    # Convert dict to array for backward compatibility
                    predictions = np.array([predictions_dict[h] for h in sorted(predictions_dict.keys())])
                    
                    logger.debug(f"Multi-horizon prediction: {len(predictions_dict)} horizons")
                    
                except Exception as e:
                    logger.warning(f"UnifiedPredictor failed, falling back to standard predict: {e}")
                    predictions_dict = None

            # Fallback to standard prediction if multi-horizon failed or not applicable
            if predictions is None:
                # Check if this is a Lightning predictor
                model_type = self.model_data.get('model_type', 'unknown')
                
                if model_type == 'lightning' and hasattr(model, 'predict'):
                    # Lightning predictor expects OHLCV patch (C, L) not features!
                    try:
                        import torch
                        
                        # Build OHLCV patch from candles_df
                        if candles_df is None or candles_df.empty:
                            raise ValueError("Lightning models require candles_df for prediction")
                        
                        # Get patch length from model
                        patch_len = getattr(model, 'patch_len', 64)
                        
                        # Extract last patch_len candles
                        if len(candles_df) < patch_len:
                            logger.warning(f"Not enough candles ({len(candles_df)}) for patch length {patch_len}, padding...")
                            # Pad with first candle repeated
                            padding_needed = patch_len - len(candles_df)
                            first_candle = candles_df.iloc[0:1]
                            padding = pd.concat([first_candle] * padding_needed, ignore_index=True)
                            candles_patch = pd.concat([padding, candles_df], ignore_index=True)
                        else:
                            candles_patch = candles_df.iloc[-patch_len:]
                        
                        # Build patch tensor (C, L) - channels: open, high, low, close, volume
                        patch_data = []
                        for col in ['open', 'high', 'low', 'close']:
                            if col in candles_patch.columns:
                                patch_data.append(candles_patch[col].values)
                        
                        # Optional: add volume if available
                        if 'volume' in candles_patch.columns and len(patch_data) < 5:
                            patch_data.append(candles_patch['volume'].values)
                        
                        # Convert to tensor (C, L)
                        patch_tensor = torch.tensor(patch_data, dtype=torch.float32)
                        
                        logger.debug(f"Built Lightning patch: shape={patch_tensor.shape} (channels={patch_tensor.shape[0]}, length={patch_tensor.shape[1]})")
                        
                        # Lightning predictor returns dict {horizon: value}
                        pred_dict = model.predict(
                            patch_tensor,
                            num_samples=50,
                            return_dict=True,
                            return_distribution=False
                        )
                        
                        # Convert dict to predictions array
                        if pred_dict and isinstance(pred_dict, dict):
                            predictions_dict = pred_dict
                            predictions = np.array(list(pred_dict.values()))
                            logger.debug(f"Lightning prediction: {len(pred_dict)} horizons")
                        
                    except Exception as e:
                        logger.error(f"Lightning predict failed: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        predictions = None
                
                # Standard sklearn/pytorch predict
                elif hasattr(model, "predict"):
                    # Try to pass candles_df if model supports it (for diffusion models)
                    try:
                        import inspect
                        sig = inspect.signature(model.predict)
                        if 'candles_df' in sig.parameters:
                            predictions = np.ravel(model.predict(X_last, candles_df=candles_df))
                        else:
                            predictions = np.ravel(model.predict(X_last))
                    except Exception as e:
                        logger.debug(f"Predict with signature inspection failed: {e}, trying direct call")
                        predictions = np.ravel(model.predict(X_last))

                if predictions is None:
                    try:
                        # Try PyTorch forward pass
                        import torch
                        if hasattr(model, 'eval'):
                            model = model.to(self.device)  # Move model to GPU if available
                            model.eval()
                        with torch.no_grad():
                            t_in = torch.tensor(X_last, dtype=torch.float32).to(self.device)
                            out = model(t_in)
                            predictions = np.ravel(out.detach().cpu().numpy())
                    except Exception:
                        pass

                if predictions is None:
                    # NO FALLBACK! Model must have predict method or fail
                    raise RuntimeError(f"Model {Path(self.model_path).name} has no predict method")

            execution_time = time.time() - start_time

            result = {
                'model_path': self.model_path,
                'model_name': Path(self.model_path).stem,
                'predictions': predictions,
                'predictions_dict': predictions_dict,  # Multi-horizon predictions {horizon: value}
                'horizons': model_horizons,  # Trained horizons
                'is_multi_horizon': is_multi_horizon,
                'features_used': available_features if features_list else list(range(X.shape[1])),
                'execution_time': execution_time,
                'model_type': self.model_data.get('model_type', 'unknown'),
                'success': True
            }
            
            return result

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
        horizons: List[str],
        use_gpu: bool = False,
        candles_df: pd.DataFrame = None,
        aggregation_method: str = 'mean'  # NEW: 'mean', 'median', 'weighted', 'best'
    ) -> Dict[str, Any]:
        """
        Run parallel inference with multiple models.

        Args:
            settings: Settings from UI with model paths
            features_df: Prepared features for inference
            symbol: Trading symbol
            timeframe: Data timeframe
            horizons: Prediction horizons
            use_gpu: Use GPU for inference (default: False)
            candles_df: Raw OHLCV candles dataframe (optional, for diffusion models)

        Returns:
            Dictionary with aggregated results from all models
        """
        # Resolve model paths
        model_paths = self.path_resolver.resolve_model_paths(settings)

        if not model_paths:
            raise ParallelInferenceError("No valid model paths found")

        device_info = "GPU" if use_gpu else "CPU"
        logger.info(f"Starting parallel inference with {len(model_paths)} models on {device_info}")

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
            executors.append(ModelExecutor(model_path, config, use_gpu=use_gpu))

        # Run parallel inference
        start_time = time.time()
        results = self._execute_parallel(executors, features_df, candles_df)
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

    def _execute_parallel(self, executors: List[ModelExecutor], features_df: pd.DataFrame, candles_df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Execute model predictions in parallel using thread pool.
        
        CRITICAL-003: Proper cleanup of model executors
        CRITICAL-004: Timeout protection (30s per model, 60s total)
        """
        results = []

        def load_and_predict(executor):
            try:
                executor.load_model()
                # Extract requested horizons from executor config
                requested_horizons = executor.model_config.get('horizon_bars')
                result = executor.predict(features_df, candles_df, requested_horizons=requested_horizons)
                return result
            except Exception as e:
                return {
                    'model_path': executor.model_path,
                    'model_name': Path(executor.model_path).stem,
                    'predictions': None,
                    'error': str(e),
                    'execution_time': 0,
                    'success': False
                }
            finally:
                # CRITICAL-003: Always cleanup after prediction
                try:
                    executor.unload_model()
                except Exception as e:
                    logger.warning(f"Error unloading model {executor.model_path}: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(load_and_predict, model_executor): model_executor
                      for model_executor in executors}

            # CRITICAL-004: Collect results with timeout
            # Individual timeout: 30s per model
            # Global timeout: 60s for all models
            try:
                for future in concurrent.futures.as_completed(futures, timeout=60):
                    model_executor = futures[future]
                    try:
                        result = future.result(timeout=30)  # 30-second timeout per model
                        results.append(result)

                    except concurrent.futures.TimeoutError:
                        logger.error(f"Model execution timed out (30s): {model_executor.model_path}")
                        results.append({
                            'model_path': model_executor.model_path,
                            'model_name': Path(model_executor.model_path).stem,
                            'predictions': None,
                            'error': 'Execution timeout (30s)',
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
                        
            except concurrent.futures.TimeoutError:
                # Global timeout reached
                logger.error("Global timeout reached (60s) - cancelling remaining models")
                for future in futures:
                    if not future.done():
                        future.cancel()
                        model_executor = futures[future]
                        results.append({
                            'model_path': model_executor.model_path,
                            'model_name': Path(model_executor.model_path).stem,
                            'predictions': None,
                            'error': 'Global timeout (60s)',
                            'execution_time': 60,
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

        # Aggregate based on method
        aggregation_method_lower = aggregation_method.lower()
        
        if aggregation_method_lower == 'mean':
            # Simple average
            ensemble_mean = np.mean(all_predictions, axis=0)
            ensemble_std = np.std(all_predictions, axis=0)
        
        elif aggregation_method_lower == 'median':
            # Median (robust to outliers)
            ensemble_mean = np.median(all_predictions, axis=0)
            # Use MAD (Median Absolute Deviation) for uncertainty
            ensemble_std = np.median(np.abs(all_predictions - ensemble_mean), axis=0) * 1.4826
        
        elif aggregation_method_lower == 'weighted mean' or aggregation_method_lower == 'weighted':
            # Weighted average (by execution time - faster = better)
            ensemble_mean = np.average(all_predictions, axis=0, weights=model_weights)
            ensemble_std = np.sqrt(np.average((all_predictions - ensemble_mean)**2, axis=0, weights=model_weights))
        
        elif aggregation_method_lower == 'best model' or aggregation_method_lower == 'best':
            # Use prediction from best model (fastest execution time as proxy for quality)
            best_idx = np.argmax(model_weights)  # Highest weight = fastest = best
            ensemble_mean = all_predictions[best_idx]
            ensemble_std = np.zeros_like(ensemble_mean)  # No uncertainty from ensemble
            logger.info(f"Using best model (index {best_idx}) for predictions")
        
        else:
            # Default to mean
            logger.warning(f"Unknown aggregation method '{aggregation_method}', defaulting to mean")
            ensemble_mean = np.mean(all_predictions, axis=0)
            ensemble_std = np.std(all_predictions, axis=0)

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
            'aggregation_method': aggregation_method,  # NEW
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