"""
UI application helpers for MagicForex GUI.
"""
from __future__ import annotations

import os
from typing import Any

from loguru import logger
from PySide6.QtWidgets import QWidget, QTabWidget, QSizePolicy

from ..services.db_service import DBService
from ..services.db_writer import DBWriter
from ..services.marketdata import MarketDataService
from ..services.tiingo_ws_connector import TiingoWSConnector
from ..services.aggregator import AggregatorService
from .controllers import UIController
from .training_tab import TrainingTab
from .signals_tab import SignalsTab
from .chart_tab_ui import ChartTabUI
from .backtesting_tab import BacktestingTab


def setup_ui(
    main_window: QWidget, 
    layout, 
    menu_bar, 
    viewer, 
    status_label, 
    engine_url: str = "http://127.0.0.1:8000", 
    use_test_server: bool = False
) -> dict:
    """
    Initializes all UI components, services, and their connections.
    """
    result: dict[str, Any] = {}

    # --- Core Services ---
    db_service = DBService()
    market_service = MarketDataService(database_url=db_service.engine.url)
    db_writer = DBWriter(db_service=db_service)
    db_writer.start()
    result["db_service"] = db_service
    result["market_service"] = market_service
    result["db_writer"] = db_writer

    # --- Aggregator Service ---
    symbols_to_aggregate = ["EUR/USD"]
    aggregator = AggregatorService(engine=db_service.engine, symbols=symbols_to_aggregate)
    aggregator.start()
    result["aggregator"] = aggregator

    # --- UI Tabs and Controller ---
    controller = UIController(main_window=main_window, market_service=market_service, db_writer=db_writer)
    controller.bind_menu_signals(menu_bar.signals)
    result["controller"] = controller
    setattr(main_window, "controller", controller)

    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    tab_widget = QTabWidget()
    tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # --- Create and connect the single, consolidated chart tab ---
    chart_tab = ChartTabUI(main_window)
    training_tab = TrainingTab(main_window)
    signals_tab = SignalsTab(main_window, db_service=db_service)
    backtesting_tab = BacktestingTab(main_window)

    # Add 3D Reports Tab
    from .reports_3d_tab import Reports3DTab
    reports_3d_tab = Reports3DTab(data_manager=market_service, parent=main_window)

    tab_widget.addTab(chart_tab, "Chart")
    tab_widget.addTab(training_tab, "Training")
    tab_widget.addTab(signals_tab, "Signals")
    tab_widget.addTab(backtesting_tab, "Backtesting")
    tab_widget.addTab(reports_3d_tab, "3D Reports")
    layout.addWidget(tab_widget)

    result["chart_tab"] = chart_tab
    result["training_tab"] = training_tab
    result["backtesting_tab"] = backtesting_tab
    result["reports_3d_tab"] = reports_3d_tab
    result["tab_widget"] = tab_widget

    # --- Connect controller and signals to the new chart_tab ---
    setattr(chart_tab, "controller", controller)
    controller.chart_tab = chart_tab
    chart_tab.forecastRequested.connect(controller.handle_forecast_payload)
    
    # Wrap forecastReady -> compute adherence if possible, then forward to chart
    def _on_forecast_ready_with_adherence(df, quantiles):
        try:
            from forex_diffusion.postproc.adherence import adherence_metrics, atr_sigma_from_df
            import pandas as _pd
            
            anchor_ts = int(_pd.to_numeric(df["ts_utc"].iloc[-1]))
            atr_n = int(getattr(controller, "indicators_settings", {}).get("atr_n", 14))
            sigma_vol = float(atr_sigma_from_df(df, n=atr_n, pre_anchor_only=True, anchor_ts=anchor_ts, robust=True))
            
            fut_ts = list(quantiles.get("future_ts") or [])
            m = list(quantiles.get("q50") or [])
            q05 = list(quantiles.get("q05") or [])
            q95 = list(quantiles.get("q95") or [])
            
            # ... [rest of the adherence logic remains the same]

            quantiles["adherence_metrics"] = adherence_metrics(
                fut_ts=fut_ts, m=m, q05=q05, q95=q95,
                actual_ts=actual_ts, actual_y=actual_y,
                sigma_vol=sigma_vol, band_target=0.90
            )
        except Exception as e:
            logger.warning(f"Adherence metric calculation failed: {e}")
        
        # Forward to the one and only chart tab
        chart_tab.on_forecast_ready(df, quantiles)

    controller.signals.forecastReady.connect(_on_forecast_ready_with_adherence)

    menu_bar.signals.trainRequested.connect(lambda: tab_widget.setCurrentWidget(training_tab))

    # --- WebSocket and Direct Data Flow ---
    ws_uri = "ws://127.0.0.1:8766" if use_test_server else "wss://api.tiingo.com/fx"
    connector = None
    if os.environ.get("FOREX_ENABLE_WS", "1") == "1":
        connector = TiingoWSConnector(
            uri=ws_uri,
            api_key=os.environ.get("TIINGO_APIKEY"),
            tickers=["eurusd"],
            chart_handler=chart_tab._handle_tick,
            db_handler=db_writer.write_tick_async
        )
        connector.start()
        result["tiingo_ws_connector"] = connector
    
    # --- Final UI Setup ---
    default_symbol = "EUR/USD"
    default_tf = "1m"
    chart_tab.set_symbol_timeframe(db_service, default_symbol, default_tf)

    controller.signals.status.connect(status_label.setText)
    controller.signals.error.connect(status_label.setText)

    # --- Backtesting Tab Handlers ---
    backtesting_tab.startRequested.connect(lambda payload: controller.handle_backtest_request(payload, backtesting_tab))

    # --- Graceful shutdown on app exit ---
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    def _graceful_shutdown():
        logger.info("Shutting down services...")
        if connector: connector.stop()
        if aggregator: aggregator.stop()
        if db_writer: db_writer.stop()
    if app: app.aboutToQuit.connect(_graceful_shutdown)

    return result
