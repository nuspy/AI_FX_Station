#!/usr/bin/env python3
"""
Minimal debug script to find exactly where the blocking occurs
"""
import sys
import os

print("1. Script started")

# Add src to path
sys.path.insert(0, 'src')
print("2. Added src to path")

# Disable WebSocket to avoid blocking
os.environ['FOREX_ENABLE_WS'] = '0'
print("3. Disabled WebSocket")

print("4. About to import PySide6...")
try:
    from PySide6.QtWidgets import QApplication, QMainWindow
    print("5. SUCCESS: PySide6 imported")
except Exception as e:
    print(f"5. ERROR importing PySide6: {e}")
    sys.exit(1)

print("6. About to create QApplication...")
try:
    app = QApplication(sys.argv)
    print("7. SUCCESS: QApplication created")
except Exception as e:
    print(f"7. ERROR creating QApplication: {e}")
    sys.exit(1)

print("8. About to create QMainWindow...")
try:
    window = QMainWindow()
    print("9. SUCCESS: QMainWindow created")
except Exception as e:
    print(f"9. ERROR creating QMainWindow: {e}")
    sys.exit(1)

print("10. About to show window...")
try:
    window.show()
    print("11. SUCCESS: Window shown")
except Exception as e:
    print(f"11. ERROR showing window: {e}")
    sys.exit(1)

print("12. About to import menus...")
try:
    from forex_diffusion.ui.menus import MainMenuBar
    print("13. SUCCESS: MainMenuBar imported")
except Exception as e:
    print(f"13. ERROR importing MainMenuBar: {e}")
    # Continue without menu bar

print("14. About to import app module...")
try:
    from forex_diffusion.ui.app import setup_ui
    print("15. SUCCESS: setup_ui imported")
except Exception as e:
    print(f"15. ERROR importing setup_ui: {e}")
    # This is likely where the blocking occurs during imports

print("16. All imports successful - about to exec...")
print("17. Starting Qt event loop...")
sys.exit(app.exec())