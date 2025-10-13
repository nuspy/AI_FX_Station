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
from .forecast_settings_tab import ForecastSettingsTab
from .logs_tab import LogsTab
from .patterns_tab import PatternsTab
from .live_trading_tab import LiveTradingTab
from .portfolio_tab import PortfolioOptimizationTab


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

    # --- Install Log Handler (early for capturing all logs) ---
    from .log_widget import install_log_handler
    install_log_handler()

    # --- Core Services ---
    db_service = DBService()
    market_service = MarketDataService(database_url=db_service.engine.url)
    db_writer = DBWriter(db_service=db_service)
    db_writer.start()
    result["db_service"] = db_service
    result["market_service"] = market_service
    result["db_writer"] = db_writer

    # --- Auto-backfill on startup (Mode A) ---
    # OPTIMIZATION: Only download 1m data from REST API, then derive higher timeframes locally
    # This reduces REST API calls from 8 per symbol to 1 per symbol
    # Higher timeframes (5m, 15m, 30m, 1h, 4h, 1d, 1w) are derived from 1m by AggregatorService
    try:
        logger.info("Auto-backfill: downloading 1m data and deriving higher timeframes...")
        symbols_to_backfill = ["EUR/USD"]  # Add more symbols as needed

        for symbol in symbols_to_backfill:
            try:
                # STEP 1: Download only 1m data from REST API
                logger.info(f"Downloading 1m data for {symbol}...")
                market_service.backfill_symbol_timeframe(
                    symbol=symbol,
                    timeframe="1m",
                    force_full=False,  # Only fill gaps, not full history
                    progress_cb=None,  # No UI progress callback on startup
                    start_ms_override=None  # Auto-detect from last candle
                )
                logger.info(f"1m download completed for {symbol}")

                # STEP 2: Derive higher timeframes from 1m using local database
                # The AggregatorService (started below) will handle this automatically
                # But we can also trigger immediate derivation here if needed
                try:
                    from ..services.aggregator import derive_timeframes_from_base
                    logger.info(f"Deriving higher timeframes from 1m for {symbol}...")
                    # Derive all higher timeframes from 1m using database
                    higher_timeframes = ["5m", "15m", "30m", "1h", "4h", "1d", "1w"]
                    for tf in higher_timeframes:
                        try:
                            derive_timeframes_from_base(
                                engine=db_service.engine,
                                symbol=symbol,
                                base_tf="1m",
                                target_tf=tf
                            )
                            logger.info(f"Derived {tf} from 1m for {symbol}")
                        except Exception as e:
                            logger.warning(f"Failed to derive {tf} from 1m for {symbol}: {e}")
                except ImportError:
                    # If derive function doesn't exist, AggregatorService will handle it later
                    logger.debug("Derivation function not available - AggregatorService will handle it")
                except Exception as e:
                    logger.warning(f"Derivation failed for {symbol}: {e}")

            except Exception as e:
                logger.warning(f"Auto-backfill failed for {symbol}: {e}")

        logger.info("Auto-backfill completed (1m + derived timeframes).")
    except Exception as e:
        logger.exception(f"Auto-backfill error: {e}")

    # --- Aggregator Service ---
    symbols_to_aggregate = ["EUR/USD"]
    aggregator = AggregatorService(engine=db_service.engine, symbols=symbols_to_aggregate)
    aggregator.start()
    result["aggregator"] = aggregator

    # --- DOM Aggregator Service ---
    from ..services.dom_aggregator import DOMAggregatorService
    dom_symbols = ["EURUSD", "GBPUSD", "USDJPY"]  # DOM monitoring symbols
    dom_aggregator = DOMAggregatorService(engine=db_service.engine, symbols=dom_symbols, interval_seconds=2)
    dom_aggregator.start()
    result["dom_aggregator"] = dom_aggregator
    logger.info(f"DOM Aggreg ator Service started for {dom_symbols}")

    # --- Order Flow Analyzer ---
    from ..analysis.order_flow_analyzer import OrderFlowAnalyzer
    order_flow_analyzer = OrderFlowAnalyzer(
        rolling_window=20,
        imbalance_threshold=0.3,
        zscore_threshold=2.0,
        large_order_percentile=0.95
    )
    result["order_flow_analyzer"] = order_flow_analyzer
    logger.info("Order Flow Analyzer initialized")

    # --- UI Tabs and Controller ---
    controller = UIController(main_window=main_window, market_service=market_service, db_writer=db_writer)
    controller.bind_menu_signals(menu_bar.signals)
    result["controller"] = controller
    setattr(main_window, "controller", controller)

    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    tab_widget = QTabWidget()
    tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    tab_widget.setObjectName("level1_tabs")
    # Level 1 tabs: High contrast
    tab_widget.setStyleSheet("""
        QTabWidget#level1_tabs::pane {
            border: 2px solid #555555;
            background: #2b2b2b;
        }
        QTabWidget#level1_tabs QTabBar::tab {
            background: #3a3a3a;
            color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
            font-weight: bold;
            border: 1px solid #555555;
        }
        QTabWidget#level1_tabs QTabBar::tab:selected {
            background: #505050;
            color: #ffffff;
            border-bottom: 3px solid #0078d7;
        }
        QTabWidget#level1_tabs QTabBar::tab:hover {
            background: #454545;
        }
    """)

    # --- Create Chart tab (level_1, no nested tabs) ---
    # Chart content is shown directly without nested tabs
    chart_tab = ChartTabUI(
        main_window,
        dom_service=dom_aggregator,
        order_flow_analyzer=order_flow_analyzer
    )

    # Live Trading will be a separate window, stored as attribute
    live_trading_tab = LiveTradingTab(main_window)

    # --- Create Trading Intelligence tab with nested tabs (level_2) ---
    trading_intelligence_container = QTabWidget()
    trading_intelligence_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    trading_intelligence_container.setObjectName("level2_tabs")
    # Level 2 tabs: Medium contrast
    trading_intelligence_container.setStyleSheet("""
        QTabWidget#level2_tabs::pane {
            border: 1px solid #444444;
            background: #2e2e2e;
        }
        QTabWidget#level2_tabs QTabBar::tab {
            background: #333333;
            color: #c0c0c0;
            padding: 6px 12px;
            margin-right: 1px;
            border: 1px solid #444444;
        }
        QTabWidget#level2_tabs QTabBar::tab:selected {
            background: #404040;
            color: #e0e0e0;
            border-bottom: 2px solid #0078d7;
        }
        QTabWidget#level2_tabs QTabBar::tab:hover {
            background: #3a3a3a;
        }
    """)

    # Create Portfolio and Signals tabs
    portfolio_tab = PortfolioOptimizationTab(main_window)
    from .signals_tab import SignalsTab
    signals_tab = SignalsTab(main_window, db_service=db_service)

    # Add as nested tabs under Trading Intelligence
    trading_intelligence_container.addTab(portfolio_tab, "Portfolio")
    trading_intelligence_container.addTab(signals_tab, "Signals")

    # --- Create Generative Forecast tab with nested tabs (level_2) ---
    uno_tab = QTabWidget()
    uno_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    uno_tab.setObjectName("level2_tabs_alt")
    # Level 2 tabs: Medium contrast (same as Trading Intelligence)
    uno_tab.setStyleSheet("""
        QTabWidget#level2_tabs_alt::pane {
            border: 1px solid #444444;
            background: #2e2e2e;
        }
        QTabWidget#level2_tabs_alt QTabBar::tab {
            background: #333333;
            color: #c0c0c0;
            padding: 6px 12px;
            margin-right: 1px;
            border: 1px solid #444444;
        }
        QTabWidget#level2_tabs_alt QTabBar::tab:selected {
            background: #404040;
            color: #e0e0e0;
            border-bottom: 2px solid #0078d7;
        }
        QTabWidget#level2_tabs_alt QTabBar::tab:hover {
            background: #3a3a3a;
        }
    """)

    # Create tabs for Generative Forecast
    forecast_settings_tab = ForecastSettingsTab(main_window)
    training_tab = TrainingTab(main_window)
    backtesting_tab = BacktestingTab(main_window)

    # Add tabs directly to Generative Forecast (no Training/Backtest container)
    uno_tab.addTab(forecast_settings_tab, "Forecast Settings")
    uno_tab.addTab(training_tab, "Training")
    uno_tab.addTab(backtesting_tab, "Backtesting")

    # Create Tab Patterns (simple widget, pattern controls are in Chart tab)
    patterns_tab = PatternsTab(main_window, chart_tab=chart_tab)

    # Create Logs tab (top-level)
    logs_tab = LogsTab(main_window)

    # Create 3D Reports tab (moved from Chart:Chart:3D Reports to level_1)
    from .reports_3d_tab import Reports3DTab
    reports_3d_tab = Reports3DTab(main_window, db_service=db_service)

    # Add top-level tabs in new order
    tab_widget.addTab(chart_tab, "Chart")  # Chart content shown directly (no nested tabs)
    tab_widget.addTab(trading_intelligence_container, "Trading Intelligence")  # Portfolio + Signals
    tab_widget.addTab(uno_tab, "Generative Forecast")  # Forecast Settings, Training, Backtesting
    tab_widget.addTab(patterns_tab, "Patterns")  # Contains Pattern Training/Backtest
    tab_widget.addTab(logs_tab, "Logs")
    tab_widget.addTab(reports_3d_tab, "3D Reports")
    layout.addWidget(tab_widget)

    result["chart_tab"] = chart_tab  # Chart widget (level_1, no nested tabs)
    result["trading_intelligence_container"] = trading_intelligence_container  # Trading Intelligence level_1 (contains Portfolio, Signals)
    result["portfolio_tab"] = portfolio_tab  # Portfolio tab under Trading Intelligence
    result["signals_tab"] = signals_tab  # Signals tab under Trading Intelligence
    result["training_tab"] = training_tab
    result["forecast_settings_tab"] = forecast_settings_tab
    result["backtesting_tab"] = backtesting_tab
    result["live_trading_tab"] = live_trading_tab  # Stored as window, opened via button
    result["patterns_tab"] = patterns_tab  # Contains Pattern Training/Backtest
    result["logs_tab"] = logs_tab
    result["reports_3d_tab"] = reports_3d_tab  # 3D Reports at level_1
    result["uno_tab"] = uno_tab  # Generative Forecast (now with 3 nested tabs)
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

            # For new forecasts, we don't have actual values yet
            # These will be populated later when comparing past forecasts to actual data
            actual_ts = []
            actual_y = []

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
    # Only start Tiingo WS if it's the primary provider or if explicitly enabled
    connector = None
    if os.environ.get("FOREX_ENABLE_WS", "1") == "1":
        from ..utils.user_settings import get_setting
        primary_provider = get_setting("primary_data_provider", "tiingo").lower()

        # Start Tiingo WS only if Tiingo is the primary provider
        if primary_provider == "tiingo":
            ws_uri = "ws://127.0.0.1:8766" if use_test_server else "wss://api.tiingo.com/fx"
            connector = TiingoWSConnector(
                uri=ws_uri,
                api_key=os.environ.get("TIINGO_APIKEY"),
                tickers=["eurusd"],
                chart_handler=chart_tab._handle_tick,
                db_handler=db_writer.write_tick_async
            )
            connector.start()
            result["tiingo_ws_connector"] = connector
            logger.info("Tiingo WebSocket connector started (primary provider: tiingo)")
        else:
            logger.info(f"Tiingo WebSocket connector NOT started (primary provider: {primary_provider})")

    # --- Connect Logs Tab to Main Window for Data Sources monitoring ---
    main_window.tiingo_ws_connector = connector
    main_window.controller = controller
    main_window.market_service = market_service  # Expose MarketDataService for monitoring
    main_window.live_trading_tab = live_trading_tab  # Expose for Live Trading window
    logs_tab.set_main_window(main_window)

    # --- Check for provider fallback and show warning ---
    if market_service.fallback_occurred:
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtCore import QTimer

        def show_fallback_warning():
            msg = QMessageBox(main_window)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Provider Fallback")
            msg.setText("⚠️ Data Provider Fallback Occurred")
            msg.setInformativeText(
                f"Requested provider: {market_service.requested_provider}\n"
                f"Active provider: {market_service.provider_name}\n\n"
                f"Reason: {market_service.fallback_reason}\n\n"
                "The application will continue using the fallback provider. "
                "Check tab:Logs:Data Sources for details."
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setDetailedText(
                "To resolve this:\n"
                "1. If using cTrader: Ensure broker connection is configured in Settings\n"
                "2. Check that the selected provider is properly configured\n"
                "3. Verify API keys and credentials are correct\n"
                "4. See logs for more detailed error information"
            )
            msg.exec()

        # Show after 2 seconds to let UI fully load
        QTimer.singleShot(2000, show_fallback_warning)

    # --- Final UI Setup ---
    default_symbol = "EUR/USD"
    default_tf = "1m"
    chart_tab.set_symbol_timeframe(db_service, default_symbol, default_tf)

    controller.signals.status.connect(status_label.setText)
    controller.signals.error.connect(status_label.setText)

    # --- Backtesting Tab Handlers ---
    # backtesting_tab.startRequested.connect(lambda payload: controller.handle_backtest_request(payload, backtesting_tab))  # DISABLED: backtesting_tab is placeholder

    # --- Graceful shutdown on app exit ---
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    def _graceful_shutdown():
        logger.info("Shutting down services...")
        if connector: connector.stop()
        if aggregator: aggregator.stop()
        if dom_aggregator: dom_aggregator.stop()
        if db_writer: db_writer.stop()
    if app: app.aboutToQuit.connect(_graceful_shutdown)

    return result
