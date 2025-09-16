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
from .training_tab import TrainingTab
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
    symbols_to_aggregate = ["EUR/USD"]  # Or load from config
    aggregator = AggregatorService(engine=db_service.engine, symbols=symbols_to_aggregate)
    aggregator.start()
    result["aggregator"] = aggregator
    logger.info("AggregatorService started.")

    # --- UI Tabs and Controller ---
    controller = UIController(main_window=main_window, market_service=market_service, db_writer=db_writer)
    controller.bind_menu_signals(menu_bar.signals)
    result["controller"] = controller
    try:
        # expose controller on main window so tabs can reach market_service
        setattr(main_window, "controller", controller)
    except Exception:
        pass

    tab_widget = QTabWidget()
    signals_tab = SignalsTab(main_window, db_service=db_service)
    training_tab = TrainingTab(main_window)
    chart_tab = ChartTab(main_window)

    tab_widget.addTab(signals_tab, "Signals")
    tab_widget.addTab(training_tab, "Training")
    tab_widget.addTab(chart_tab, "Chart")
    layout.addWidget(tab_widget)
    result["chart_tab"] = chart_tab
    result["training_tab"] = training_tab
    result["tab_widget"] = tab_widget
    try:
        chart_tab.controller = controller
        controller.chart_tab = chart_tab
        setattr(chart_tab, "controller", controller)

    except Exception:
        pass
    # expose chart_tab on controller for symbol/timeframe discovery
    try:
        controller.chart_tab = chart_tab
        setattr(chart_tab, "controller", controller)
    except Exception:
        pass
    # connect ChartTab forecast requests to controller handler, and results back to the chart
    try:
        chart_tab.forecastRequested.connect(controller.handle_forecast_payload)
    except Exception:
        logger.warning("Failed to connect chart_tab.forecastRequested")
    try:
        controller.signals.forecastReady.connect(chart_tab.on_forecast_ready)
    except Exception:
        logger.warning("Failed to connect controller.forecastReady to chart")

    # bring Training tab to front on menu->Train
    try:
        menu_bar.signals.trainRequested.connect(lambda: tab_widget.setCurrentWidget(training_tab))
    except Exception:
        pass

    # --- WebSocket and Direct Data Flow ---
    ws_uri = "ws://127.0.0.1:8766" if use_test_server else "wss://api.tiingo.com/fx"
    if use_test_server:
        logger.info(f"Redirecting Tiingo WebSocket to test server: {ws_uri}")

    def _ws_status(msg: str):
        try:
            if msg == "ws_down":
                logger.warning("Realtime WS down detected. REST fallback is DISABLED.")
                controller.signals.status.emit("Realtime: WS down (no REST fallback)")
                #controller.signals.status.emit("Realtime: WS down (fallback REST attivo)")

            elif msg == "ws_restored":
                logger.info("Realtime WS restored.")
                controller.signals.status.emit("Realtime: WS restored")
        except Exception:
            pass

    connector = TiingoWSConnector(
        uri=ws_uri,
        api_key=os.environ.get("TIINGO_APIKEY"),
        tickers=["eurusd"],
        chart_handler=chart_tab._handle_tick,
        db_handler=db_writer.write_tick_async
        , status_handler=_ws_status
    )
    connector.start()
    result["tiingo_ws_connector"] = connector
    logger.info("TiingoWSConnector started with direct handlers for ChartTab and DBWriter.")

    # --- Final UI Setup ---
    default_symbol = "EUR/USD"
    default_tf = "1m"
    chart_tab.set_symbol_timeframe(db_service, default_symbol, default_tf)

    # Auto backfill on startup for all supported symbols with existing candles
    try:
        from PySide6.QtCore import QRunnable, QThreadPool, QObject, Signal
        class _BFSignals(QObject):
            progress = Signal(int)
            status = Signal(str)
            done = Signal()

        class _BFJob(QRunnable):
            def __init__(self, market_service, symbols, years, months, signals):
                super().__init__()
                self.ms = market_service
                self.symbols = symbols
                self.years = int(years)
                self.months = int(months)
                self.signals = signals

            def run(self):
                import math, time
                total = 0
                # compute total subranges estimate by summing across symbols later; we update per-symbol progressively
                for sym in self.symbols:
                    try:
                        first_ts = self.ms._get_first_candle_ts(sym)
                        if first_ts is None:
                            continue
                        total += 1
                    except Exception:
                        continue
                done = 0
                for sym in self.symbols:
                    try:
                        first_ts = self.ms._get_first_candle_ts(sym)
                        if first_ts is None:
                            # skip symbols with no candles at all
                            self.signals.status.emit(f"[Backfill] Skip {sym}: no candles in DB")
                            continue
                        # compute start override from UI (0/0 -> full from first), else from now - (years, months)
                        import datetime
                        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
                        if (self.years == 0 and self.months == 0):
                            start_ms = first_ts
                        else:
                            # approximate months as 30 days each
                            days = self.years * 365 + self.months * 30
                            start_ms = max(first_ts, now_ms - days * 24 * 3600 * 1000)
                        self.signals.status.emit(f"[Backfill] {sym} starting range sync...")
                        # nested progress bridge
                        def _cb(p):
                            try:
                                self.signals.status.emit(f"[Backfill] {sym}: {p}%")
                            except Exception:
                                pass
                        # abilita REST solo per la durata di questo backfill
                        try:
                            setattr(self.ms, "rest_enabled", True)
                        except Exception:
                            pass
                        try:
                            # use '1d' to cover all timeframes up to daily
                            self.ms.backfill_symbol_timeframe(sym, "1d", force_full=False, progress_cb=_cb, start_ms_override=start_ms)
                        finally:
                            try:
                                setattr(self.ms, "rest_enabled", False)
                            except Exception:
                                pass
                    except Exception as e:
                        try:
                            self.signals.status.emit(f"[Backfill] {sym} failed: {e}")
                        except Exception:
                            pass
                    finally:
                        done += 1
                        pct = int(min(100, max(0, done / max(1,total) * 100)))
                        try:
                            self.signals.progress.emit(pct)
                        except Exception:
                            pass
                try:
                    self.signals.done.emit()
                except Exception:
                    pass

        bf_signals = _BFSignals()
        bf_signals.status.connect(status_label.setText)
        bf_signals.progress.connect(lambda p: status_label.setText(f"Backfill: {p}%"))
        QThreadPool.globalInstance().start(_BFJob(market_service, chart_tab._symbols_supported, chart_tab.years_combo.currentText(), chart_tab.months_combo.currentText(), bf_signals))
    except Exception as e:
        logger.warning("Auto backfill job not started: {}", e)

    controller.signals.status.connect(status_label.setText)
    controller.signals.error.connect(status_label.setText)

    # --- Graceful shutdown on app exit ---
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        def _graceful_shutdown():
            try:
                logger.info("Shutting down services...")
                try:
                    connector.stop()
                except Exception:
                    pass
                try:
                    aggregator.stop()
                except Exception:
                    pass
                try:
                    db_writer.stop()
                except Exception:
                    pass
            except Exception:
                pass
        if app is not None:
            app.aboutToQuit.connect(_graceful_shutdown)
    except Exception:
        pass

    return result
