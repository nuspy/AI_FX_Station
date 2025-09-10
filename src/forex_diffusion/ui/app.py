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
        import threading
        market_service = MarketDataService()
        # force provider to tiingo and log (robust attempt)
        try:
            try:
                market_service.set_provider("tiingo")
                logger.info("Forced MarketDataService provider to tiingo at startup")
            except Exception as e:
                # some implementations may expose provider property or method differently; best-effort
                try:
                    setattr(market_service, "provider", "tiingo")
                    logger.info("Set MarketDataService.provider attribute to 'tiingo' as fallback")
                except Exception:
                    logger.warning("Could not force MarketDataService provider to tiingo: {}", e)
        except Exception:
            pass

            # Attempt to start Tiingo WebSocket streamer unconditionally (it will check env/provider for API key)
            try:
                try:
                    market_service.start_ws_streaming(tickers=getattr(market_service, "_realtime_symbols", None), threshold="5")
                    logger.info("Requested MarketDataService Tiingo WS streaming start (background thread)")
                except Exception as e:
                    logger.debug("Failed to request Tiingo WS streaming: {}", e)
            except Exception:
                pass
        # schedule startup backfill in background (non-blocking)
        try:
            threading.Thread(target=market_service.ensure_startup_backfill, daemon=True).start()
        except Exception:
            pass

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

    # start local websocket server for realtime ticks (accept external WS pushes)
    try:
        from ..services.local_ws import LocalWebsocketServer
        ws_port = getattr(market_service, "_ws_port", None) or 8765
        # pass the db_service instance we just created
        local_ws = LocalWebsocketServer(host="127.0.0.1", port=ws_port, db_service=db_service)
        try:
            started = local_ws.start()
            if started:
                result["local_ws"] = local_ws
                logger.info("Started LocalWebsocketServer on port {}", ws_port)
            else:
                logger.warning("LocalWebsocketServer did not start on port {} (check 'websockets' package)", ws_port)
        except Exception as e:
            logger.exception("Failed to start LocalWebsocketServer: {}", e)

        # Start in-process Tiingo WS connector (for direct streaming into event_bus)
        try:
            from ..services.tiingo_ws_connector import TiingoWSConnector
            try:
                connector = TiingoWSConnector(api_key=os.environ.get("TIINGO_APIKEY") or os.environ.get("TIINGO_API_KEY"), tickers=["eurusd"], threshold="5", db_engine=db_service.engine if db_service else None)
                connector.start()
                result["tiingo_ws_connector"] = connector
                logger.info("TiingoWSConnector requested to start")
            except Exception as e:
                logger.debug("Failed to start TiingoWSConnector: {}", e)
        except Exception:
            # best-effort: skip if module not present
            pass
    except Exception as e:
        logger.exception("Failed to start LocalWebsocketServer: {}", e)

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

            # EventBridge: forward event_bus 'tick' messages into Qt main thread via a Signal
            try:
                from .event_bridge import EventBridge
                from ..utils.event_bus import subscribe as _eb_subscribe
                bridge = EventBridge(main_window)
                # subscribe bridge._on_event to event_bus (will be called from connector thread)
                try:
                    _eb_subscribe("tick", bridge._on_event)
                    logger.debug("Registered EventBridge to event_bus 'tick'")
                except Exception as e:
                    logger.debug("Failed to subscribe EventBridge to event_bus: {}", e)
                # log event_bus status snapshot
                try:
                    from ..utils.event_bus import debug_status as _eb_status
                    st = _eb_status()
                    logger.info(f"event_bus status after bridge registration: {st}")
                except Exception:
                    pass
                # connect Qt signal to ChartTab handler (will run in UI thread)
                try:
                    bridge.tickReceived.connect(lambda payload: chart_tab._on_tick_event(payload))
                except Exception as e:
                    logger.debug("Failed to connect EventBridge.tickReceived to chart_tab: {}", e)

                # Self-test: publish a synthetic tick after 2s to verify the full delivery chain (event_bus -> bridge -> ChartTab)
                try:
                    from ..utils.event_bus import publish as _publish
                    import threading as _thr
                    import time as _time

                    def _send_self_test():
                        try:
                            payload = {
                                "symbol": "EUR/USD",
                                "timeframe": "1m",
                                "ts_utc": int(_time.time() * 1000),
                                "price": 1.23456,
                                "bid": 1.23450,
                                "ask": 1.23462
                            }
                            try:
                                logger.info("App self-test: publishing synthetic tick %s", payload)
                            except Exception:
                                pass
                            try:
                                _publish("tick", payload)
                            except Exception as e:
                                try:
                                    logger.debug("App self-test publish failed: {}", e)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    try:
                        _thr.Timer(2.0, _send_self_test).start()
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass

            tabw.addTab(signals_tab, "Signals")
            tabw.addTab(history_tab, "History")
            tabw.addTab(chart_tab, "Chart")

            # link history tab to chart for automatic plotting
            try:
                history_tab.chart_tab = chart_tab
            except Exception:
                pass

            layout.addWidget(tabw)

            # Populate chart at startup: set defaults and refresh history (which will update chart)
            try:
                # choose sensible defaults
                default_symbol = "EUR/USD"
                default_tf = "1m"
                try:
                    history_tab.symbol_combo.setCurrentText(default_symbol)
                except Exception:
                    pass
                try:
                    history_tab.tf_combo.setCurrentText(default_tf)
                except Exception:
                    pass
                # set chart symbol/timeframe so it can accept realtime ticks
                try:
                    chart_tab.set_symbol_timeframe(db_service, default_symbol, default_tf)
                except Exception:
                    pass
                # refresh history (will populate table and chart)
                try:
                    history_tab.refresh(limit=500)
                except Exception:
                    pass
            except Exception:
                pass

            result["signals_tab"] = signals_tab
            result["history_tab"] = history_tab
            result["chart_tab"] = chart_tab
            result["market_service"] = market_service

            # Debug tracer: periodically log event_bus status (queue sizes and subscriber counts)
            # event_bus tracer removed to silence periodic logs

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
