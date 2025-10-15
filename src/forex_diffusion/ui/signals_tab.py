# src/forex_diffusion/ui/signals_tab.py
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, 
    QHeaderView, QHBoxLayout, QLabel, QSpinBox
)
from PySide6.QtCore import QTimer
from loguru import logger
from sqlalchemy import text

from ..services.db_service import DBService

class SignalsTab(QWidget):
    """
    A widget to display trading signals from the database.
    """
    def __init__(self, parent=None, db_service: Optional[DBService] = None, **kwargs):
        super().__init__(parent)
        self.db_service = db_service or DBService()
        self._main_window = parent

        self.layout = QVBoxLayout(self)

        # --- Controls ---
        controls_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Signals")
        self.refresh_btn.clicked.connect(self.refresh)
        controls_layout.addWidget(self.refresh_btn)

        self.limit_spinbox = QSpinBox()
        self.limit_spinbox.setRange(10, 1000)
        self.limit_spinbox.setValue(100)
        controls_layout.addWidget(QLabel("Limit:"))
        controls_layout.addWidget(self.limit_spinbox)
        controls_layout.addStretch()
        self.layout.addLayout(controls_layout)

        # --- Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["ID", "Symbol", "Timeframe", "Created At", "Entry Price", "Target Price", "Stop Price"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.table)

        # --- Auto-refresh Timer ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(60000) # Refresh every 60 seconds

        # Don't auto-refresh on init to avoid blocking app startup
        # self.refresh()

    def refresh(self):
        """Refreshes the signals table with the latest data from the database."""
        if not self.db_service:
            self._log("DBService not available. Cannot refresh signals.")
            return
        limit = self.limit_spinbox.value()
        # self._log(f"Refreshing signals (limit={limit})...")
        try:
            with self.db_service.engine.connect() as conn:
                query = text(
                    "SELECT id, symbol, timeframe, ts_created_ms, entry_price, target_price, stop_price "
                    "FROM signals ORDER BY ts_created_ms DESC LIMIT :limit"
                )
                rows = conn.execute(query, {"limit": limit}).fetchall()
                self.table.setRowCount(len(rows))
                for i, row in enumerate(rows):
                    self.table.setItem(i, 0, QTableWidgetItem(str(row.id)))
                    self.table.setItem(i, 1, QTableWidgetItem(row.symbol))
                    self.table.setItem(i, 2, QTableWidgetItem(row.timeframe))
                    self.table.setItem(i, 3, QTableWidgetItem(str(pd.to_datetime(row.ts_created_ms, unit='ms'))))
                    self.table.setItem(i, 4, QTableWidgetItem(f"{row.entry_price:.5f}"))
                    self.table.setItem(i, 5, QTableWidgetItem(f"{row.target_price:.5f}"))
                    self.table.setItem(i, 6, QTableWidgetItem(f"{row.stop_price:.5f}"))
            # self._log("Signals refreshed successfully.")
        except Exception as e:
            self._log(f"Failed to refresh signals: {e}")
            logger.exception(e)

    def _log(self, message: str):
        """Logs a message to the main window's status bar."""
        if self._main_window and hasattr(self._main_window, "statusBar"):
            try:
                self._main_window.statusBar().showMessage(message, 5000)
            except Exception:
                pass # Avoid crashing on logging errors
        logger.info(message)
