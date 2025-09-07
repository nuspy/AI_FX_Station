# ui/signals_tab.py
# Widget to display persisted signals from the DB (simple table with refresh)
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QLabel
from loguru import logger
from sqlalchemy import MetaData, select

from ..services.db_service import DBService


class SignalsTab(QWidget):
    """
    Displays recent signals persisted in the DB.
    Methods:
      - refresh(engine, limit=100): reloads the most recent signals
    """
    def __init__(self, parent=None, db_service: Optional[DBService] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.header = QLabel("Signals")
        self.layout.addWidget(self.header)
        self.refresh_btn = QPushButton("Refresh")
        self.layout.addWidget(self.refresh_btn)
        self.table = QTableWidget()
        self.layout.addWidget(self.table)
        self.refresh_btn.clicked.connect(self.on_refresh_clicked)
        self.db_service = db_service or DBService()
        # initialize empty
        self._populate_empty()

    def _populate_empty(self):
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ts", "symbol", "timeframe", "entry", "target", "stop"])
        self.table.setRowCount(0)

    def on_refresh_clicked(self):
        try:
            self.refresh(limit=100)
        except Exception as e:
            logger.exception("SignalsTab refresh failed: {}", e)

    def refresh(self, limit: int = 100):
        """
        Query DB for most recent signals and populate the table.
        """
        try:
            engine = self.db_service.engine
            meta = MetaData()
            meta.reflect(bind=engine, only=["signals"])
            tbl = meta.tables.get("signals")
            if tbl is None:
                self._populate_empty()
                return
            with engine.connect() as conn:
                stmt = select(tbl.c.ts_created_ms, tbl.c.symbol, tbl.c.timeframe, tbl.c.entry_price, tbl.c.target_price, tbl.c.stop_price).order_by(tbl.c.ts_created_ms.desc()).limit(limit)
                rows = conn.execute(stmt).fetchall()
                if not rows:
                    self._populate_empty()
                    return
                rows = list(rows)
                self.table.setRowCount(len(rows))
                for i, r in enumerate(rows):
                    ts = int(r[0])
                    self.table.setItem(i, 0, QTableWidgetItem(str(ts)))
                    self.table.setItem(i, 1, QTableWidgetItem(str(r[1])))
                    self.table.setItem(i, 2, QTableWidgetItem(str(r[2])))
                    self.table.setItem(i, 3, QTableWidgetItem(str(r[3])))
                    self.table.setItem(i, 4, QTableWidgetItem(str(r[4])))
                    self.table.setItem(i, 5, QTableWidgetItem(str(r[5])))
                self.table.resizeColumnsToContents()
        except Exception as e:
            logger.exception("Failed to load signals: {}", e)
            self._populate_empty()
