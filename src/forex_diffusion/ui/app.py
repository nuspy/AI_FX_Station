"""
UI application helpers for MagicForex GUI.

Provides setup_ui(...) to initialize controller, SignalsTab, DBWriter and wire signals
in a safe, well-indented manner. Call setup_ui from your MainWindow or app entrypoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .signals_tab import SignalsTab
from ..services.marketdata import MarketDataService
from .controllers import UIController
from ..services.db_service import DBService
from loguru import logger
from PySide6.QtWidgets import QWidget

def setup_ui(main_window: QWidget, layout, menu_bar, viewer, status_label, engine_url: str = "http://127.0.0.1:8000") -> dict:
    """
    Initialize UI wiring and services.
    """
    logger.critical("--- EXECUTING LATEST APP.PY VERSION ---")
    result: dict[str, Any] = {}
    try:
        market_service = MarketDataService()
        try:
            market_service.set_provider("tiingo")
            logger.info("Forced MarketDataService provider to tiingo at startup")
        except Exception as e:
            logger.warning("Could not force MarketDataService provider to tiingo: {}", e)

        controller = UIController(main_window=main_window, market_service=market_service, engine_url=engine_url)
        try:
            controller.bind_menu_signals(menu_bar.signals)
        except Exception as e:
            logger.warning("Menu signals binding failed: {}", e)
        
        result["controller"] = controller
    except Exception as e:
        logger.exception("Failed to initialize UI controller: {}", e)

    try:
        db_service = DBService()
        result["db_service"] = db_service
    except Exception as e:
        logger.exception("Failed to initialize DBService: {}", e)
        db_service = None

    try:
        from ..services.local_ws import LocalWebsocketServer
        ws_port = 8765
        local_ws = LocalWebsocketServer(host="127.0.0.1", port=ws_port, db_service=db_service)
        if local_ws.start():
            result["local_ws"] = local_ws
            logger.info(f"Started LocalWebsocketServer on port {ws_port}")
    except Exception as e:
        logger.exception(f"Failed to start LocalWebsocketServer: {e}")

    try:
        from ..services.tiingo_ws_connector import TiingoWSConnector
        connector = TiingoWSConnector(api_key=os.environ.get("TIINGO_APIKEY"), tickers=["eurusd"], threshold="5")
        connector.start()
        result["tiingo_ws_connector"] = connector
        logger.info("TiingoWSConnector requested to start")
    except Exception as e:
        logger.debug(f"Failed to start TiingoWSConnector: {e}")

    db_writer = None
    if db_service:
        try:
            from ..services.db_writer import DBWriter
            db_writer = DBWriter(db_service=db_service)
            db_writer.start()
            result["db_writer"] = db_writer
        except Exception as e:
            logger.exception(f"Failed to initialize DBWriter: {e}")

    try:
        from .history_tab import HistoryTab
        from .chart_tab import ChartTab
        from PySide6.QtWidgets import QTabWidget

        tabw = QTabWidget()
        signals_tab = SignalsTab(main_window, db_service=db_service, market_service=market_service)
        history_tab = HistoryTab(main_window, db_service=db_service, market_service=market_service)
        chart_tab = ChartTab(main_window)

        if "controller" in result:
            chart_tab.forecastRequested.connect(result["controller"].handle_forecast_requested)
            result["controller"].signals.forecastReady.connect(chart_tab.update_plot)
            result["controller"].signals.status.connect(lambda s: status_label.setText(f"Status: {s}"))
            result["controller"].signals.error.connect(lambda e: status_label.setText(f"Error: {e}"))
            if db_writer:
                result["controller"].db_writer = db_writer

        try:
            from .event_bridge import EventBridge
            from ..utils.event_bus import subscribe as _eb_subscribe
            bridge = EventBridge(main_window)
            _eb_subscribe("tick", bridge._on_event)
            bridge.tickReceived.connect(chart_tab._handle_tick)
            logger.debug("Connected EventBridge to ChartTab for UI updates.")
            
            if db_writer:
                bridge.tickReceived.connect(db_writer.write_tick_async)
                logger.debug("Connected EventBridge to DBWriter for tick persistence.")

        except Exception as e:
            logger.debug(f"Failed to set up EventBridge: {e}")

        tabw.addTab(signals_tab, "Signals")
        tabw.addTab(history_tab, "History")
        tabw.addTab(chart_tab, "Chart")
        layout.addWidget(tabw)

        history_tab.chart_tab = chart_tab
        default_symbol = "EUR/USD"
        default_tf = "1m"
        history_tab.symbol_combo.setCurrentText(default_symbol)
        history_tab.tf_combo.setCurrentText(default_tf)
        chart_tab.set_symbol_timeframe(db_service, default_symbol, default_tf)
        history_tab.refresh(limit=500)

        result["signals_tab"] = signals_tab
        result["history_tab"] = history_tab
        result["chart_tab"] = chart_tab
        result["market_service"] = market_service

    except Exception as e:
        logger.exception(f"Failed to initialize UI tabs: {e}")

    return result
