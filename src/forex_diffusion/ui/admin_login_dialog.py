"""
AdminLoginDialog: simple dialog to enter an admin token for remote admin calls.
Optionally saves token into user settings.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMessageBox, QCheckBox
from ..utils.user_settings import set_setting, get_setting

class AdminLoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Admin Login")
        self.resize(420, 140)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Enter admin token for remote admin actions:"))
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.token_input)
        self.save_chk = QCheckBox("Save token to local settings (insecure)")
        layout.addWidget(self.save_chk)
        btn_h = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        btn_h.addWidget(self.ok_btn)
        btn_h.addWidget(self.cancel_btn)
        layout.addLayout(btn_h)
        self.ok_btn.clicked.connect(self.on_ok)
        self.cancel_btn.clicked.connect(self.reject)
        # load saved default if present
        saved = get_setting("session_admin_token", "")
        self.token_input.setText(saved)

    def on_ok(self):
        tok = self.token_input.text().strip()
        if not tok:
            QMessageBox.warning(self, "No token", "Please enter a token")
            return
        if self.save_chk.isChecked():
            # try to save encrypted if possible
            try:
                from ..utils.user_settings import set_encrypted_setting
                set_encrypted_setting("session_admin_token", tok)
            except Exception:
                from ..utils.user_settings import set_setting
                set_setting("session_admin_token", tok)
        self.accept()

    def token(self) -> str:
        return self.token_input.text().strip()
