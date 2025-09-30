#!/usr/bin/env python3
"""
Test calling setup_ui function to find the exact blocking point
"""
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

# Disable WebSocket to avoid blocking
os.environ['FOREX_ENABLE_WS'] = '0'

print("Testing setup_ui function call...")

try:
    # Create Qt application first
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
    app = QApplication(sys.argv)
    print("Step 1: QApplication created")

    # Create main window
    main_window = QMainWindow()
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    main_window.setCentralWidget(central_widget)
    print("Step 2: Main window created")

    # Create menu bar and viewer
    from forex_diffusion.ui.menus import MainMenuBar
    menu_bar = MainMenuBar()
    main_window.setMenuBar(menu_bar)
    print("Step 3: Menu bar created")

    viewer = QLabel("Chart Viewer")
    status_label = QLabel("Status")
    layout.addWidget(viewer)
    layout.addWidget(status_label)
    print("Step 4: UI components created")

    # Import setup_ui
    from forex_diffusion.ui.app import setup_ui
    print("Step 5: setup_ui imported")

    # Show window before calling setup_ui
    main_window.show()
    print("Step 6: Window shown")

    print("Step 7: About to call setup_ui - this may block...")

    # This is where the blocking likely occurs
    result = setup_ui(
        main_window=main_window,
        layout=layout,
        menu_bar=menu_bar,
        viewer=viewer,
        status_label=status_label,
        engine_url="http://127.0.0.1:8000",
        use_test_server=False
    )

    print("Step 8: setup_ui completed successfully!")
    print(f"Result keys: {list(result.keys())}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")