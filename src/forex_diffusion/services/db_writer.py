    def write_prediction_async(self, symbol: str, timeframe: str, horizon: str, q05: float, q50: float, q95: float, meta: Optional[Dict] = None) -> bool:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": horizon,
            "q05": float(q05),
            "q50": float(q50),
            "q95": float(q95),
            "meta": meta or {},
        }
        return self.enqueue_task("prediction", payload)

    def write_signal_async(self, symbol: str, timeframe: str, entry_price: float, target_price: float, stop_price: float, metrics: Optional[Dict] = None) -> bool:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "entry_price": float(entry_price),
            "target_price": float(target_price),
            "stop_price": float(stop_price),
            "metrics": metrics or {},
        }
        return self.enqueue_task("signal", payload)

    def write_calibration_async(self, symbol: str, timeframe: str, alpha: float, delta_global: float, details: Optional[Dict] = None) -> bool:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "alpha": float(alpha),
            "delta_global": float(delta_global),
            "details": details or {},
        }
        return self.enqueue_task("calibration", payload)

    def write_features_async(self, symbol: str, timeframe: str, ts_utc: int, features: Dict[str, Any], pipeline_version: Optional[str] = None) -> bool:
        """
        Enqueue a features row for async persistence.
        """
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "ts_utc": int(ts_utc),
            "features": features,
            "pipeline_version": pipeline_version,
        }
        return self.enqueue_task("features", payload)

    def write_latents_async(self, symbol: Optional[str], timeframe: Optional[str], ts_utc: int, latent: List[float], model_version: Optional[str] = None) -> bool:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "ts_utc": int(ts_utc),
            "latent": latent,
            "model_version": model_version,
        }
        return self.enqueue_task("latents", payload)
