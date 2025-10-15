"""
Test Qt performance - minimal app to isolate the issue
"""
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel

app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("Qt Performance Test")
window.resize(800, 600)

label = QLabel("If this window is smooth → Qt works\nIf laggy → System/Qt issue")
label.setStyleSheet("font-size: 20px; padding: 50px;")
window.setCentralWidget(label)

window.show()
print("✓ Qt window created")
print("✓ Try dragging the window - is it smooth?")
print("✓ If yes: problem is in ForexGPT code")
print("✓ If no: problem is Qt/Windows/Graphics driver")

sys.exit(app.exec())
