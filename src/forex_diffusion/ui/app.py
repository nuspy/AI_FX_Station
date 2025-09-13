"""
UI application helpers for MagicForex GUI.
"""
from __future__ import annotations

import os
from typing import Any

from loguru import logger
from PySide6.QtWidgets import QWidget, QTabWidget

from ..services.db_service import DBService
from ..services.db_writer import DBWriter
from ..services.marketdata import MarketDataService
from ..services.tiingo_ws_connector import TiingoWSConnector
from ..services.local_ws import LocalWebsocketServer
from ..services.aggregator import AggregatorService
from .controllers import UIController
from .history_tab import HistoryTab
from .signals_tab import SignalsTab
from .chart_tab import ChartTab

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
    logger.critical("--- EXECUTING LATEST APP.PY VERSION ---")
    result: dict[str, Any] = {}

    # --- Core Services ---
    db_service = DBService()
    market_service = MarketDataService(database_url=db_service.engine.url)
    db_writer = DBWriter(db_service=db_service)
    db_writer.start()

    result["db_service"] = db_service
    result["market_service"] = market_service
    result["db_writer"] = db_writer

    # --- Start Aggregator Service ---
    symbols_to_aggregate = ["EUR/USD"] # Or load from config
    aggregator = AggregatorService(engine=db_service.engine, symbols=symbols_to_aggregate)
    aggregator.start()
    result["aggregator"] = aggregator
    logger.info("AggregatorService started.")

    # --- UI Tabs and Controller ---
    controller = UIController(main_window=main_window, market_service=market_service, db_writer=db_writer)
    controller.bind_menu_signals(menu_bar.signals)
    result["controller"] = controller

    tab_widget = QTabWidget()
    signals_tab = SignalsTab(main_window, db_service=db_service)
    history_tab = HistoryTab(main_window, db_service=db_service)
    chart_tab = ChartTab(main_window)
    
    tab_widget.addTab(signals_tab, "Signals")
    tab_widget.addTab(history_tab, "History")
    tab_widget.addTab(chart_tab, "Chart")
    layout.addWidget(tab_widget)
    result["chart_tab"] = chart_tab

    # --- WebSocket and Direct Data Flow ---
    ws_uri = "ws://127.0.0.1:8766" if use_test_server else "wss://api.tiingo.com/fx"
    if use_test_server:
        logger.info(f"Redirecting Tiingo WebSocket to test server: {ws_uri}")

    connector = TiingoWSConnector(
        uri=ws_uri,
        api_key=os.environ.get("TIINGO_APIKEY"),
        tickers=["eurusd"],
        chart_handler=chart_tab._handle_tick,
        db_handler=db_writer.write_tick_async
    )
    connector.start()
    result["tiingo_ws_connector"] = connector
    logger.info("TiingoWSConnector started with direct handlers for ChartTab and DBWriter.")

    # --- Final UI Setup ---
    history_tab.chart_tab = chart_tab
    default_symbol = "EUR/USD"
    default_tf = "1m"
    chart_tab.set_symbol_timeframe(db_service, default_symbol, default_tf)
    history_tab.symbol_combo.setCurrentText(default_symbol)
    history_tab.tf_combo.setCurrentText(default_tf)
    history_tab.refresh(limit=500)

    controller.signals.status.connect(status_label.setText)
    controller.signals.error.connect(status_label.setText)

    return result
