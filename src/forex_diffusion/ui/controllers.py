# ui/controllers.py
# Controller to bind UI menu actions to background workers and services.
from __future__ import annotations

from typing import Optional, Tuple

import httpx
import pandas as pd
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, Slot
from loguru import logger

from ..services.marketdata import MarketDataService
from ..data import io as data_io


class UIControllerSignals(QObject):
    """Signals emitted by the controller to update UI."""
    forecastReady = Signal(object, object)  # (pd.DataFrame, quantiles_dict)
    error = Signal(str)
    status = Signal(str)


class ForecastWorker(QRunnable):
    """
    Worker that calls the inference HTTP endpoint and fetches recent candles from DB,
    then emits the forecastReady signal with (df, quantiles).
    """
    def __init__(self, engine_url: str, payload: dict, market_service: MarketDataService, signals: UIControllerSignals):
        super().__init__()
        self.engine_url = engine_url.rstrip("/")
        self.payload = payload
        self.market_service = market_service
        self.signals = signals

    def run(self):
        try:
            self.signals.status.emit("Forecast: requesting server...")
            url = f"{self.engine_url}/forecast"
            with httpx.Client(timeout=60.0) as client:
                r = client.post(url, json=self.payload)
                r.raise_for_status()
                data = r.json()
            # fetch recent candles to display context (attempt 1000 bars)
            try:
                df = self._fetch_recent_candles(self.market_service.engine, self.payload.get("symbol"), self.payload.get("timeframe"), n_bars=1000)
            except Exception as e:
                logger.warning("Could not fetch recent candles for viewer: {}", e)
                df = pd.DataFrame()
            quantiles = {k: v for k, v in data.get("quantiles", {}).items()}
            self.signals.status.emit("Forecast: ready")
            self.signals.forecastReady.emit(df, quantiles)
        except Exception as e:
            logger.exception("Forecast worker failed: {}", e)
            self.signals.error.emit(str(e))
            self.signals.status.emit("Forecast: failed")

    def _fetch_recent_candles(self, engine, symbol: str, timeframe: str, n_bars: int = 500) -> pd.DataFrame:
        """
        Query market_data_candles for the last n_bars for the given symbol/timeframe.
        Returns DataFrame sorted ascending by ts_utc.

        Uses SQLAlchemy reflection to locate the table safely.
        """
        try:
            from sqlalchemy import MetaData, select
            meta = MetaData()
            meta.reflect(bind=engine, only=["market_data_candles"])
            tbl = meta.tables.get("market_data_candles")
            if tbl is None:
                return pd.DataFrame()
            with engine.connect() as conn:
                stmt = (
                    select(
                        tbl.c.ts_utc,
                        tbl.c.open,
                        tbl.c.high,
                        tbl.c.low,
                        tbl.c.close,
                        tbl.c.volume,
                    )
                    .where(tbl.c.symbol == symbol)
                    .where(tbl.c.timeframe == timeframe)
                    .order_by(tbl.c.ts_utc.desc())
                    .limit(n_bars)
                )
                rows = conn.execute(stmt).fetchall()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, columns=["ts_utc", "open", "high", "low", "close", "volume"])
                df = df.sort_values("ts_utc").reset_index(drop=True)
                return df
        except Exception as e:
            logger.exception("Failed to fetch recent candles: {}", e)
            return pd.DataFrame()


class UIController:
    """
    Binds menu signals to actions, runs background workers, exposes signals for UI updates.
    """
    def __init__(self, main_window, market_service: Optional[MarketDataService] = None, engine_url: str = "http://127.0.0.1:8000"):
        self.main_window = main_window
        self.market_service = market_service or MarketDataService()
        self.engine_url = engine_url
        self.signals = UIControllerSignals()
        self.pool = QThreadPool.globalInstance()

    def bind_menu_signals(self, menu_signals):
        """
        Connect menu signals to controller handlers.
        """
        menu_signals.ingestRequested.connect(self.handle_ingest_requested)
        menu_signals.trainRequested.connect(self.handle_train_requested)
        menu_signals.forecastRequested.connect(self.handle_forecast_requested)
        menu_signals.calibrationRequested.connect(self.handle_calibration_requested)
        menu_signals.backtestRequested.connect(self.handle_backtest_requested)
        menu_signals.realtimeToggled.connect(self.handle_realtime_toggled)
        menu_signals.configRequested.connect(self.handle_config_requested)

    @Slot()
    def handle_ingest_requested(self):
        self.signals.status.emit("Ingest requested: launching backfill...")
        # run backfill in background via MarketDataService.ensure_startup_backfill
        worker = _IngestWorker(self.market_service, self.signals)
        self.pool.start(worker)

    @Slot()
    def handle_train_requested(self):
        self.signals.status.emit("Train requested (not implemented).")

    @Slot()
    def handle_forecast_requested(self):
        # Prepare simple payload using default symbol/timeframe from config if available
        cfg = self.market_service.cfg if hasattr(self.market_service, "cfg") else None
        try:
            symbol = cfg.data.symbols[0] if (cfg and hasattr(cfg, "data") and hasattr(cfg.data, "symbols")) else (cfg.data.get("symbols", [])[0] if isinstance(cfg.data, dict) else "EUR/USD")
            timeframe = (cfg.timeframes.native[0] if (cfg and hasattr(cfg, "timeframes") and hasattr(cfg.timeframes, "native")) else "1m")
        except Exception:
            symbol = "EUR/USD"
            timeframe = "1m"
        payload = {"symbol": symbol, "timeframe": timeframe, "horizons": ["1m", "5m", "15m"], "N_samples": 200, "apply_conformal": True}
        self.signals.status.emit(f"Forecast requested for {symbol} {timeframe}")
        fw = ForecastWorker(engine_url=self.engine_url, payload=payload, market_service=self.market_service, signals=self.signals)
        self.pool.start(fw)

    @Slot()
    def handle_calibration_requested(self):
        self.signals.status.emit("Calibration requested (not implemented).")

    @Slot()
    def handle_backtest_requested(self):
        self.signals.status.emit("Backtest requested (not implemented).")

    @Slot(bool)
    def handle_realtime_toggled(self, enabled: bool):
        self.signals.status.emit("Realtime toggled: {}".format("ON" if enabled else "OFF"))

    @Slot()
    def handle_config_requested(self):
        self.signals.status.emit("Config requested (not implemented).")


class _IngestWorker(QRunnable):
    """Worker that runs MarketDataService.ensure_startup_backfill in background."""
    def __init__(self, market_service: MarketDataService, signals: UIControllerSignals):
        super().__init__()
        self.market_service = market_service
        self.signals = signals

    def run(self):
        try:
            self.signals.status.emit("Backfill: running...")
            reports = self.market_service.ensure_startup_backfill()
            self.signals.status.emit(f"Backfill: completed ({len(reports)} reports)")
        except Exception as e:
            logger.exception("Backfill worker failed: {}", e)
            self.signals.error.emit(str(e))
            self.signals.status.emit("Backfill failed")
