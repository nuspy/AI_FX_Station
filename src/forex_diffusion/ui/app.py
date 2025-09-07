from .signals_tab import SignalsTab
from ..services.marketdata import MarketDataService  # Use UIController to bind menu actions and handle background tasks
        from .controllers import UIController
        # instantiate controller with MarketDataService and bind menu signals
        self.controller = UIController(main_window=self, market_service=MarketDataService(), engine_url="http://127.0.0.1:8000")
        self.controller.bind_menu_signals(self.menu_bar.signals)

        # connect controller signals to UI elements
        self.controller.signals.forecastReady.connect(lambda df, q: self.viewer.update_plot(df, q))
        self.controller.signals.status.connect(lambda s: self.status_label.setText(f"Status: {s}"))
        self.controller.signals.error.connect(lambda e: self.status_label.setText(f"Error: {e}"))

        # Signals tab (shows persisted signals)
        try:
            self.signals_tab = SignalsTab(self, db_service=DBService())
            layout.addWidget(self.signals_tab)
            # refresh signals after each successful forecast
            self.controller.signals.forecastReady.connect(lambda df, q: self.signals_tab.refresh(limit=100))
        except Exception as e:
            logger.exception("Failed to initialize SignalsTab: {}", e)

        # Signals tab (shows persisted signals)
        try:
            self.db_service = DBService()
            # create and start a DBWriter for UI-originated async writes
            from ..services.db_writer import DBWriter
            self.db_writer = DBWriter(db_service=self.db_service)
            self.db_writer.start()

            self.signals_tab = SignalsTab(self, db_service=self.db_service)
            layout.addWidget(self.signals_tab)
            # refresh signals after each successful forecast
            self.controller.signals.forecastReady.connect(lambda df, q: self.signals_tab.refresh(limit=100))
            # also connect controller to use this writer
            self.controller.db_writer = self.db_writer
        except Exception as e:
            logger.exception("Failed to initialize SignalsTab or DBWriter: {}", e)
