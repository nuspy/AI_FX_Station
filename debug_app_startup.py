#!/usr/bin/env python3
"""
Debug script for ForexGPT application startup
This will help identify where the application is blocking
"""
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

# Disable WebSocket to avoid blocking
os.environ['FOREX_ENABLE_WS'] = '0'

print("Starting ForexGPT debug session...")

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
    from PySide6.QtCore import QTimer

    # Create QApplication
    app = QApplication(sys.argv)
    print("SUCCESS: QApplication created")

    # Create main window
    main_window = QMainWindow()
    main_window.setWindowTitle("ForexGPT - Debug Mode")
    main_window.setGeometry(100, 100, 1200, 800)

    # Create central widget with layout
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    main_window.setCentralWidget(central_widget)

    # Create status label
    status_label = QLabel("Initializing ForexGPT...")
    layout.addWidget(status_label)
    print("SUCCESS: Main window created")

    # Create a simple viewer placeholder
    viewer = QLabel("Chart Viewer Area")
    layout.addWidget(viewer)

    # Import and create menu bar
    from forex_diffusion.ui.menus import MainMenuBar
    menu_bar = MainMenuBar()
    main_window.setMenuBar(menu_bar)
    print("SUCCESS: Menu bar created")

    # Show window first
    main_window.show()
    print("SUCCESS: Window shown")

    # Now try to set up the UI components
    print("Setting up UI components...")

    # Use a timer to set up components after the window is shown
    def setup_components():
        try:
            status_label.setText("Setting up components...")
            app.processEvents()  # Process events to update UI

            from forex_diffusion.ui.app import setup_ui
            print("About to call setup_ui...")

            # This is where it might block
            result = setup_ui(
                main_window=main_window,
                layout=layout,
                menu_bar=menu_bar,
                viewer=viewer,
                status_label=status_label,
                engine_url="http://127.0.0.1:8000",
                use_test_server=False
            )

            status_label.setText("ForexGPT ready!")
            print("SUCCESS: setup_ui completed successfully")

        except Exception as e:
            error_msg = f"Error in setup_ui: {e}"
            print(f"ERROR: {error_msg}")
            status_label.setText(error_msg)
            import traceback
            traceback.print_exc()

    # Set up components after a short delay
    QTimer.singleShot(1000, setup_components)

    print("Starting event loop...")
    sys.exit(app.exec())

except Exception as e:
    print(f"ERROR: Critical error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)