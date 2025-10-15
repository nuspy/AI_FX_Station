#!/usr/bin/env python3
"""
Fixed application startup that bypasses the blocking components
"""
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

# Disable WebSocket to avoid blocking
os.environ['FOREX_ENABLE_WS'] = '0'

print("Starting ForexGPT with minimal components...")

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QTabWidget
    app = QApplication(sys.argv)
    print("SUCCESS: QApplication created")

    # Create main window
    main_window = QMainWindow()
    main_window.setWindowTitle("ForexGPT - Fixed Startup")
    main_window.setGeometry(100, 100, 1200, 800)

    # Create central widget with layout
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    main_window.setCentralWidget(central_widget)

    # Create status label
    status_label = QLabel("ForexGPT starting...")
    layout.addWidget(status_label)

    # Create menu bar
    from forex_diffusion.ui.menus import MainMenuBar
    menu_bar = MainMenuBar()
    main_window.setMenuBar(menu_bar)
    print("SUCCESS: Menu bar created")

    # Show window first
    main_window.show()
    print("SUCCESS: Window shown")

    # Create core services without starting them in blocking mode
    from forex_diffusion.services.db_service import DBService
    from forex_diffusion.services.marketdata import MarketDataService
    from forex_diffusion.services.db_writer import DBWriter

    db_service = DBService()
    market_service = MarketDataService(database_url=db_service.engine.url)
    db_writer = DBWriter(db_service=db_service)

    print("SUCCESS: Core services created")

    # Create UI controller
    from forex_diffusion.ui.controllers import UIController
    controller = UIController(main_window=main_window, market_service=market_service, db_writer=db_writer)
    controller.bind_menu_signals(menu_bar.signals)
    print("SUCCESS: UI controller created and signals bound")

    # Create a minimal tab widget instead of the complex tabs
    tab_widget = QTabWidget()
    layout.addWidget(tab_widget)

    # Add a simple placeholder tab instead of the complex ChartTabUI
    placeholder_tab = QWidget()
    placeholder_layout = QVBoxLayout(placeholder_tab)
    placeholder_layout.addWidget(QLabel("Chart functionality temporarily disabled"))
    placeholder_layout.addWidget(QLabel("App started successfully!"))
    tab_widget.addTab(placeholder_tab, "Chart (Minimal)")

    print("SUCCESS: Minimal tabs created")

    # Update status
    status_label.setText("ForexGPT started successfully in minimal mode!")

    print("SUCCESS: Application started successfully!")
    print("Window should be visible. Close it to exit.")

    # Start the event loop
    sys.exit(app.exec())

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)