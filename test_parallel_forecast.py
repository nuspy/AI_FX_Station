"""
Automated test for parallel forecast to debug the "index 13 out of bounds" error.
Runs the same forecast code as the UI and logs all details.
"""
import sys
sys.path.insert(0, 'src')

from forex_diffusion.ui.workers.forecast_worker import ForecastWorker
from forex_diffusion.services.marketdata import MarketDataService
from loguru import logger
import json

# Configure detailed logging
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def test_parallel_forecast():
    """Test parallel forecast with exact same settings as UI."""

    # Load settings from file (same as UI)
    with open('configs/prediction_settings.json', 'r') as f:
        settings = json.load(f)

    logger.info(f"Loaded settings: model_paths={settings.get('model_paths')}")

    # Build payload exactly as the UI does
    payload = {
        "symbol": "EUR/USD",
        "timeframe": "1m",
        **settings,
        # Force cache invalidation by slightly modifying config
        "rv_window": settings.get("rv_window", 60) + 0  # No actual change but triggers recalc
    }

    logger.info(f"Payload: {len(payload.get('model_paths', []))} models, horizons={payload.get('horizons')}")

    # Clear feature cache to force recalculation with new indicators
    from forex_diffusion.features.feature_cache import FeatureCache
    cache = FeatureCache()
    cache.clear_cache()
    logger.info("Feature cache cleared to force recalculation")

    # Create worker (same as UI - needs engine_url, payload, market_service, signals)
    # We don't need signals for direct testing, just call _parallel_infer directly
    from forex_diffusion.services.db_service import DBService

    db_service = DBService()
    market_service = MarketDataService(database_url=db_service.engine.url)
    engine_url = "http://127.0.0.1:8000"  # Not used for local inference

    # Create a minimal signals object (not needed for direct _parallel_infer call)
    class DummySignals:
        def __init__(self):
            self.status = type('obj', (object,), {'emit': lambda x: None})()
            self.forecastReady = type('obj', (object,), {'emit': lambda x, y: None})()
            self.error = type('obj', (object,), {'emit': lambda x: None})()

    worker = ForecastWorker(engine_url, payload, market_service, DummySignals())

    try:
        logger.info("Starting parallel inference test...")
        result = worker._parallel_infer()

        if result:
            df_candles, quantiles = result
            logger.success(f"✅ SUCCESS! Got {len(quantiles.get('q50', []))} predictions")
            logger.info(f"Predictions: {quantiles.get('q50', [])[:5]}...")

            # Log ATR columns to debug
            atr_cols = [c for c in df_candles.columns if 'atr' in str(c).lower()]
            logger.info(f"ATR columns in df_candles: {atr_cols}")

            return True
        else:
            logger.error("❌ FAILED: No result returned")
            return False

    except Exception as e:
        logger.error(f"❌ FAILED with exception: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("AUTOMATED PARALLEL FORECAST TEST")
    logger.info("=" * 80)

    success = test_parallel_forecast()

    logger.info("=" * 80)
    if success:
        logger.success("TEST PASSED ✅")
        sys.exit(0)
    else:
        logger.error("TEST FAILED ❌")
        sys.exit(1)
