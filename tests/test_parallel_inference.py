"""
Test script for parallel inference - reproduces exact payload from application
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
import pandas as pd
import json

# Configure logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")

def test_parallel_inference():
    """Test parallel inference with exact payload from application"""

    # Load settings from config file
    config_path = Path("D:/Projects/ForexGPT/configs/prediction_settings.json")
    with open(config_path) as f:
        settings = json.load(f)

    logger.info(f"Loaded settings: {json.dumps(settings, indent=2)}")

    # Generate synthetic candle data (5000 bars for multi-timeframe indicators)
    import numpy as np
    from datetime import datetime, timezone, timedelta

    now = datetime.now(tz=timezone.utc)
    candles_override = []
    base_price = 11000  # Pip format (1.1000 * 10000)

    for i in range(5000, 0, -1):
        ts = int((now - timedelta(minutes=i)).timestamp() * 1000)
        price = base_price + np.random.uniform(-50, 50)
        candles_override.append({
            "ts_utc": ts,
            "open": price + np.random.uniform(-5, 5),
            "high": price + np.random.uniform(5, 15),
            "low": price + np.random.uniform(-15, -5),
            "close": price + np.random.uniform(-5, 5),
            "volume": np.random.randint(100, 1000)
        })

    logger.info(f"Generated {len(candles_override)} synthetic candles for testing")

    # Create payload matching application
    payload = {
        "symbol": "EURUSD",
        "timeframe": "1m",
        "testing_point_ts": None,  # Latest data
        "test_history_bars": 512,  # Fetch 512 bars
        "anchor_price": None,
        "advanced": False,
        "parallel_inference": True,

        # Provide synthetic data to bypass database
        "candles_override": candles_override,

        # From settings file
        "model_paths": settings.get("model_paths", []),
        "horizons": settings.get("horizons", "1-180m"),
        "N_samples": settings.get("n_samples", 99),
        "apply_conformal": True,
        "forecast_step": "auto",

        # Feature configuration
        "use_relative_ohlc": True,
        "use_temporal_features": True,
        "rv_window": settings.get("rv_window", 60),
        "warmup_bars": settings.get("warmup_bars", 16),
        "min_feature_coverage": settings.get("min_feature_coverage", 0.05),

        # Indicator parameters
        "atr_n": 14,
        "rsi_n": 14,
        "bb_n": 20,

        # Parallel settings
        "max_parallel_workers": 4,
        "limit_candles": 5000,  # Match synthetic data length
    }

    logger.info(f"Testing parallel inference with payload:")
    logger.info(f"  Models: {len(payload['model_paths'])}")
    logger.info(f"  Symbol: {payload['symbol']}")
    logger.info(f"  Timeframe: {payload['timeframe']}")
    logger.info(f"  Horizons: {payload['horizons']}")

    # Initialize market service
    from forex_diffusion.services.marketdata import MarketDataService
    from forex_diffusion.utils.config import get_config

    config = get_config()

    # Extract database path from config
    try:
        db_url = config.db.database_url if hasattr(config.db, 'database_url') else "sqlite:///./data/forex_diffusion.db"
    except:
        db_url = "sqlite:///./data/forex_diffusion.db"

    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        if not Path(db_path).is_absolute():
            db_path = str(Path.cwd() / db_path)
    else:
        db_path = "./data/forex_diffusion.db"

    logger.info(f"Using database: {db_path}")

    # MarketDataService only needs database_url
    market_service = MarketDataService(database_url=f"sqlite:///{db_path}")

    # Create forecast worker
    from forex_diffusion.ui.workers.forecast_worker import ForecastWorker
    from PySide6.QtCore import QObject, Signal

    class TestSignals(QObject):
        status = Signal(str)
        forecastReady = Signal(object, object)
        error = Signal(str)

    signals = TestSignals()

    # Track results
    results = {"success": False, "error": None, "quantiles": None}

    def on_forecast_ready(df, quantiles):
        logger.info(f"✓ Forecast ready! DataFrame shape: {df.shape}")
        logger.info(f"✓ Quantiles keys: {list(quantiles.keys())}")
        results["success"] = True
        results["quantiles"] = quantiles

    def on_error(err):
        logger.error(f"✗ Forecast error: {err}")
        results["error"] = err

    def on_status(msg):
        logger.debug(f"Status: {msg}")

    signals.forecastReady.connect(on_forecast_ready)
    signals.error.connect(on_error)
    signals.status.connect(on_status)

    # Run worker
    worker = ForecastWorker(
        engine_url="",
        payload=payload,
        market_service=market_service,
        signals=signals
    )

    logger.info("=" * 80)
    logger.info("Starting parallel inference test...")
    logger.info("=" * 80)

    try:
        worker.run()
    except Exception as e:
        logger.exception(f"Worker failed with exception: {e}")
        results["error"] = str(e)

    logger.info("=" * 80)
    if results["success"]:
        logger.info("✓✓✓ TEST PASSED ✓✓✓")
        return True
    else:
        logger.error(f"✗✗✗ TEST FAILED ✗✗✗")
        logger.error(f"Error: {results['error']}")
        return False

if __name__ == "__main__":
    success = test_parallel_inference()
    sys.exit(0 if success else 1)
