#!/usr/bin/env python3
"""
ForexGPT Application Launcher
This is the main entry point to run the ForexGPT application
"""
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

# Disable WebSocket by default (can be enabled by setting FOREX_ENABLE_WS=1)
os.environ.setdefault('FOREX_ENABLE_WS', '0')

def main():
    """Main application entry point"""
    try:
        from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
        from PySide6.QtCore import QTimer

        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("ForexGPT")
        app.setOrganizationName("ForexGPT")

        # Create main window
        main_window = QMainWindow()
        main_window.setWindowTitle("ForexGPT - Forex Analysis Platform")
        
        # Restore window geometry/state
        from forex_diffusion.utils.user_settings import get_setting, set_setting
        saved_geometry = get_setting('window.geometry')
        saved_state = get_setting('window.state')
        
        if saved_geometry:
            try:
                main_window.restoreGeometry(bytes.fromhex(saved_geometry))
            except Exception:
                main_window.setGeometry(100, 100, 1400, 900)
        else:
            main_window.setGeometry(100, 100, 1400, 900)
        
        if saved_state:
            try:
                main_window.restoreState(bytes.fromhex(saved_state))
            except Exception:
                pass

        # Create central widget
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        main_window.setCentralWidget(central_widget)

        # Create menu bar
        from forex_diffusion.ui.menus import MainMenuBar
        menu_bar = MainMenuBar()
        main_window.setMenuBar(menu_bar)

        # Create viewer and status components
        from PySide6.QtWidgets import QLabel, QStatusBar
        viewer = QLabel("Chart Area")

        status_bar = QStatusBar()
        main_window.setStatusBar(status_bar)
        status_label = QLabel("Initializing ForexGPT...")
        status_bar.addWidget(status_label)

        # Show window
        main_window.show()

        # Set up UI components after window is shown
        def setup_ui_components():
            try:
                status_label.setText("Loading UI components...")
                app.processEvents()

                from forex_diffusion.ui.app import setup_ui

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
                print("ForexGPT application started successfully!")

            except Exception as e:
                error_msg = f"Error starting ForexGPT: {e}"
                print(error_msg)
                status_label.setText(error_msg)
                import traceback
                traceback.print_exc()

        # Use timer to set up components after window is shown
        QTimer.singleShot(100, setup_ui_components)
        
        # Save window state on close
        def save_window_state():
            try:
                set_setting('window.geometry', main_window.saveGeometry().hex())
                set_setting('window.state', main_window.saveState().hex())
            except Exception as e:
                print(f"Error saving window state: {e}")
        
        app.aboutToQuit.connect(save_window_state)

        # Start the application
        print("Starting ForexGPT...")
        sys.exit(app.exec())

    except Exception as e:
        print(f"Critical error starting ForexGPT: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()