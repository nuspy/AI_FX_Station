"""
UI application helpers for MagicForex GUI.

Provides setup_ui(...) to initialize controller, SignalsTab, DBWriter and wire signals
in a safe, well-indented manner. Call setup_ui from your MainWindow or app entrypoint.
"""

from __future__ import annotations

import logging
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

    Args:
      main_window: the main window widget
      layout: the layout to which to add the SignalsTab
      menu_bar: menu bar object with signals attribute for binding
      viewer: chart/view widget exposing update_plot(df, q)
      status_label: QLabel to show status text
      engine_url: URL where FastAPI inference is reachable (for remote admin)
    Returns:
      dict with references to created services (controller, db_service, db_writer, signals_tab)
    """
    result: dict[str, Any] = {}
    try:
        market_service = MarketDataService()
        controller = UIController(main_window=main_window, market_service=market_service, engine_url=engine_url)
        # bind menu signals (best-effort)
        try:
            controller.bind_menu_signals(menu_bar.signals)
        except Exception as e:
            logger.warning("Menu signals binding failed: {}", e)
        # connect controller signals to UI elements
        try:
            controller.signals.forecastReady.connect(lambda df, q: viewer.update_plot(df, q))
            controller.signals.status.connect(lambda s: status_label.setText(f"Status: {s}"))
            controller.signals.error.connect(lambda e: status_label.setText(f"Error: {e}"))
        except Exception as e:
            logger.warning("Failed to connect controller signals to UI: {}", e)
        result["controller"] = controller
    except Exception as e:
        logger.exception("Failed to initialize UI controller: {}", e)
        # still proceed to try create SignalsTab/DBService below

    # Try to create DBService and DBWriter and SignalsTab, robustly
    try:
        db_service = DBService()
        result["db_service"] = db_service
    except Exception as e:
        logger.exception("Failed to initialize DBService: {}", e)
        db_service = None

    try:
        # create and start a DBWriter for UI-originated async writes if DB available
        db_writer = None
        if db_service is not None:
            try:
                from ..services.db_writer import DBWriter
                db_writer = DBWriter(db_service=db_service)
                db_writer.start()
                result["db_writer"] = db_writer
            except Exception as e:
                logger.exception("Failed to initialize DBWriter: {}", e)
                db_writer = None
        # instantiate UI tabs: Signals, History, Chart
        try:
            from .history_tab import HistoryTab
            from .chart_tab import ChartTab
            from PySide6.QtWidgets import QTabWidget

            tabw = QTabWidget()
            signals_tab = SignalsTab(main_window, db_service=db_service, market_service=market_service)
            history_tab = HistoryTab(main_window, db_service=db_service, market_service=market_service)
            chart_tab = ChartTab(main_window)

            tabw.addTab(signals_tab, "Signals")
            tabw.addTab(history_tab, "History")
            tabw.addTab(chart_tab, "Chart")

            layout.addWidget(tabw)

            result["signals_tab"] = signals_tab
            result["history_tab"] = history_tab
            result["chart_tab"] = chart_tab
            result["market_service"] = market_service

            # refresh signals after each successful forecast if controller exists
            if "controller" in result:
                try:
                    result["controller"].signals.forecastReady.connect(lambda df, q: signals_tab.refresh(limit=100))
                except Exception:
                    pass
            # attach writer to controller if present
            if "controller" in result and db_writer is not None:
                try:
                    result["controller"].db_writer = db_writer
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Failed to initialize SignalsTab: {}", e)
    except Exception as e:
        logger.exception("Failed to setup DBWriter/SignalsTab: {}", e)

    return result
