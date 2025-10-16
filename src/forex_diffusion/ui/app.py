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
    
    # Performance profiling
    import time
    _startup_timer = time.time()
    
    def _log_timing(step_name):
        elapsed = time.time() - _startup_timer
        logger.info(f"⏱️ STARTUP TIMING: {step_name} completed in {elapsed:.2f}s")

    # --- Install Log Handler (early for capturing all logs) ---
    from .log_widget import install_log_handler
    install_log_handler()
    _log_timing("Log handler installed")

    # --- Log Device Info (GPU/CPU) ---
    try:
        from ..utils.device_manager import log_device_info_once
        log_device_info_once()
    except Exception as e:
        logger.debug(f"Device info logging failed: {e}")

    # --- Core Services ---
    db_service = DBService()
    market_service = MarketDataService(database_url=db_service.engine.url)
    db_writer = DBWriter(db_service=db_service)
    db_writer.start()
    result["db_service"] = db_service
    result["market_service"] = market_service
    result["db_writer"] = db_writer
    _log_timing("Database and Market services initialized")

    # --- Auto-backfill on startup (Mode A) ---
    # DISABLED: Auto-backfill slows down startup significantly (600+ days download)
    # Users can manually backfill from UI when needed
    # OPTIMIZATION: Only download 1m data from REST API, then derive higher timeframes locally
    AUTO_BACKFILL_ENABLED = True  # Set to True to enable auto-backfill on startup
    
    if AUTO_BACKFILL_ENABLED:
        try:
            logger.info("Auto-backfill: downloading 1m data and deriving higher timeframes...")
            symbols_to_backfill = ["EUR/USD"]  # Add more symbols as needed

            for symbol in symbols_to_backfill:
                try:
                    # STEP 1: Download only 1m data from REST API
                    logger.info(f"Downloading 1m data for {symbol}...")
                    _backfill_start = time.time()
                    market_service.backfill_symbol_timeframe(
                        symbol=symbol,
                        timeframe="1m",
                        force_full=False,  # Only fill gaps, not full history
                        progress_cb=None,  # No UI progress callback on startup
                        start_ms_override=None  # Auto-detect from last candle
                    )
                    _backfill_elapsed = time.time() - _backfill_start
                    logger.info(f"1m download completed for {symbol} in {_backfill_elapsed:.2f}s")

                    # STEP 2: Derive higher timeframes from 1m using local database
                    try:
                        from ..services.aggregator import derive_timeframes_from_base
                        logger.info(f"Deriving higher timeframes from 1m for {symbol}...")
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
                        logger.debug("Derivation function not available - AggregatorService will handle it")
                    except Exception as e:
                        logger.warning(f"Derivation failed for {symbol}: {e}")

                except Exception as e:
                    logger.warning(f"Auto-backfill failed for {symbol}: {e}")

            logger.info("Auto-backfill completed (1m + derived timeframes).")
            _log_timing("Auto-backfill completed")
        except Exception as e:
            logger.exception(f"Auto-backfill error: {e}")
    else:
        logger.info("Auto-backfill disabled (set AUTO_BACKFILL_ENABLED=True to enable)")
    
    _log_timing("Before Aggregator Service start")

    # --- Aggregator Service ---
    symbols_to_aggregate = ["EUR/USD"]
    aggregator = AggregatorService(engine=db_service.engine, symbols=symbols_to_aggregate)
    aggregator.start()
    result["aggregator"] = aggregator

    # --- DOM Aggregator Service ---
    from ..services.dom_aggregator import DOMAggregatorService
    dom_symbols = ["EURUSD", "GBPUSD", "USDJPY"]  # DOM monitoring symbols
    # Pass provider reference to enable RAM buffer reading (faster than database)
    # IMPORTANT: Get the actual CTraderProvider (not CTraderClient wrapper)
    provider = None
    if hasattr(market_service, 'provider'):
        # Check if it's CTraderClient (which wraps CTraderProvider)
        from ..services.ctrader_client import CTraderClient
        if isinstance(market_service.provider, CTraderClient):
            # Get the underlying CTraderProvider from CTraderClient
            provider = getattr(market_service.provider, '_provider', None)
            if provider is None:
                logger.warning("CTraderClient detected but no underlying CTraderProvider found")
        else:
            provider = market_service.provider
    
    dom_aggregator = DOMAggregatorService(
        engine=db_service.engine, 
        symbols=dom_symbols, 
        interval_seconds=2,
        provider=provider  # Enable RAM buffer access
    )
    dom_aggregator.start()
    result["dom_aggregator"] = dom_aggregator
    provider_info = f"{type(provider).__name__}" if provider else "None"
    logger.info(f"DOM Aggregator Service started for {dom_symbols} (provider: {provider_info})")

    # --- Sentiment Aggregator Service ---
    from ..services.sentiment_aggregator import SentimentAggregatorService
    sentiment_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    sentiment_aggregator = SentimentAggregatorService(
        engine=db_service.engine,
        symbols=sentiment_symbols,
        interval_seconds=30  # Process sentiment every 30 seconds
    )
    sentiment_aggregator.start()
    result["sentiment_aggregator"] = sentiment_aggregator
    logger.info(f"Sentiment Aggregator Service started for {sentiment_symbols}")

    # --- VIX Service ---
    from ..services.vix_service import VIXService
    vix_service = VIXService(
        engine=db_service.engine,
        interval_seconds=300  # Fetch VIX every 5 minutes
    )
    vix_service.start()
    result["vix_service"] = vix_service
    logger.info("VIX Service started (fetch every 5min)")

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
    _log_timing("Before creating ChartTab")
    chart_tab = ChartTabUI(
        main_window,
        dom_service=dom_aggregator,
        order_flow_analyzer=order_flow_analyzer,
        sentiment_service=sentiment_aggregator,
        vix_service=vix_service
    )
    _log_timing("After creating ChartTab")
    logger.info("✓ ChartTab created")
    
    # Connect cTrader spot events to chart (for real-time price updates)
    # Get the actual CTraderProvider (same as dom_aggregator)
    if provider and hasattr(provider, 'on_tick_callback'):
        provider.on_tick_callback = chart_tab._handle_tick
        logger.info(f"✓ {type(provider).__name__} spot events connected to chart")

    # Live Trading will be a separate window, stored as attribute
    # Initialize SmartExecutionOptimizer for pre-trade validation
    from ..execution.smart_execution import SmartExecutionOptimizer
    execution_optimizer = SmartExecutionOptimizer()
    result["execution_optimizer"] = execution_optimizer

    live_trading_tab = LiveTradingTab(
        main_window,
        dom_service=dom_aggregator,
        execution_optimizer=execution_optimizer
    )

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
        # cTrader WebSocket for primary provider
        # STRATEGY: CTraderProvider (historical data) and CTraderWebSocketService (real-time streams)
        # both use Twisted reactor. To avoid conflicts:
        # 1. CTraderProvider starts reactor first (via MarketDataService init)
        # 2. CTraderWebSocketService detects running reactor and reuses it
        # 3. Both connect to same account using same credentials (no auth duplication)
        elif primary_provider == "ctrader":
            logger.info("cTrader provider active - Initializing WebSocket service for real-time market data")
            ctrader_enabled = get_setting("ctrader_enabled", True)  # Default to True when cTrader is primary
            if ctrader_enabled:
                try:
                    from ..services.ctrader_websocket import CTraderWebSocketService
                    
                    # Get cTrader credentials - try both prefixed and non-prefixed keys
                    client_id = (get_setting("provider.ctrader.client_id", "") or
                                get_setting("ctrader_client_id", ""))
                    client_secret = (get_setting("provider.ctrader.client_secret", "") or
                                get_setting("ctrader_client_secret", ""))
                    access_token = (get_setting("provider.ctrader.access_token", "") or
                                   get_setting("ctrader_access_token", ""))
                    environment = (get_setting("provider.ctrader.environment", "demo") or
                                  get_setting("ctrader_environment", "demo"))
                    
                    # IMPORTANT: Get the authenticated account_id from CTraderProvider
                    # CTraderProvider has already connected and fetched the numeric account ID
                    # from cTrader API, so we reuse it instead of creating a new connection
                    account_id = None
                    if hasattr(market_service, 'provider') and hasattr(market_service.provider, '_account_id'):
                        account_id = market_service.provider._account_id
                        logger.info(f"Using authenticated account ID from CTraderProvider: {account_id}")
                    else:
                        # Fallback: try to get from config, but this may be a username
                        account_id_raw = (get_setting("provider.ctrader.account_id", "") or
                                         get_setting("ctrader_account_id", ""))
                        if account_id_raw:
                            account_id = account_id_raw
                            logger.warning(f"Using account_id from config (may need auto-fetch): {account_id}")
                    
                    logger.info(f"cTrader WebSocket initialization: "
                               f"client_id={'set' if client_id else 'missing'}, "
                               f"client_secret={'set' if client_secret else 'missing'}, "
                               f"access_token={'set' if access_token else 'missing'}, "
                               f"account_id={account_id}, "
                               f"environment={environment}")
                    
                    if client_id and client_secret and access_token and account_id:
                        # DISABLED: CTraderProvider already handles WebSocket streaming + DOM
                        # Separate WebSocketService causes auth conflicts (duplicate connections)
                        logger.info(f"✓ cTrader WebSocket handled by CTraderProvider (account: {account_id}, env: {environment})")
                        # ctrader_ws = CTraderWebSocketService(
                        #     client_id=client_id,
                        #     client_secret=client_secret,
                        #     access_token=access_token,
                        #     account_id=account_id,
                        #     db_engine=db_service.engine,
                        #     environment=environment,
                        #     symbols=["EURUSD", "GBPUSD", "USDJPY"]
                        # )
                        # ctrader_ws.start()
                        # result["ctrader_ws"] = ctrader_ws
                        # main_window.ctrader_ws = ctrader_ws
                    else:
                        logger.warning("cTrader credentials incomplete - WebSocket not started")
                        logger.warning(f"Missing credentials: "
                                     f"client_id={'NO' if not client_id else 'OK'}, "
                                     f"client_secret={'NO' if not client_secret else 'OK'}, "
                                     f"access_token={'NO' if not access_token else 'OK'}, "
                                     f"account_id={'NO' if not account_id else 'OK'}")
                except Exception as e:
                    logger.error(f"Failed to start cTrader WebSocket: {e}", exc_info=True)
            else:
                logger.info("cTrader WebSocket not started (ctrader_enabled=False)")
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
        # Stop cTrader WebSocket if it was started
        ctrader_ws = result.get("ctrader_ws")
        if ctrader_ws:
            try:
                ctrader_ws.stop()
                logger.info("cTrader WebSocket stopped")
            except Exception as e:
                logger.error(f"Error stopping cTrader WebSocket: {e}")
        if aggregator: aggregator.stop()
        if dom_aggregator: dom_aggregator.stop()
        if sentiment_aggregator: sentiment_aggregator.stop()
        vix_srv = result.get("vix_service")
        if vix_srv: vix_srv.stop()
        if db_writer: db_writer.stop()
    if app: app.aboutToQuit.connect(_graceful_shutdown)
    
    _log_timing("✅ COMPLETE - UI setup finished")
    return result
