# src/forex_diffusion/ui/history_tab.py
from __future__ import annotations

from typing import Optional
import time
import json

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QLabel, QMessageBox, QComboBox
)
from loguru import logger
from sqlalchemy import MetaData, select

from ..services.db_service import DBService
from ..services.marketdata import MarketDataService

class HistoryTab(QWidget):
    """
    Shows historical candles from DB for selected symbol/timeframe.
    Buttons:
      - Refresh: reload latest rows
      - Backfill: trigger market_service.backfill_symbol_timeframe for the symbol/timeframe
    """
    def __init__(self, parent=None, db_service: Optional[DBService] = None, market_service: Optional[MarketDataService] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["EUR/USD","USD/JPY","GBP/USD"])
        top.addWidget(self.symbol_combo)

        top.addWidget(QLabel("Timeframe:"))
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["1m","5m","1d"])
        top.addWidget(self.tf_combo)

        self.refresh_btn = QPushButton("Refresh History")
        self.refresh_btn.clicked.connect(self.on_refresh)
        top.addWidget(self.refresh_btn)

        self.backfill_btn = QPushButton("Backfill")
        self.backfill_btn.clicked.connect(self.on_backfill)
        top.addWidget(self.backfill_btn)

        self.layout.addLayout(top)

        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        self.db_service = db_service or DBService()
        try:
            self.market_service = market_service or MarketDataService()
        except Exception:
            self.market_service = market_service

        self._populate_empty()

    def _populate_empty(self):
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["id","symbol","timeframe","ts_utc","open","high","low","close"])
        self.table.setRowCount(0)

    def on_refresh(self):
        try:
            self.refresh(limit=200)
        except Exception as e:
            logger.exception("HistoryTab refresh failed: {}", e)
            QMessageBox.warning(self, "Refresh failed", str(e))

    def on_backfill(self):
        try:
            sym = self.symbol_combo.currentText()
            tf = self.tf_combo.currentText()
            # run backfill synchronously (quick check) - may be long
            res = {}
            if self.market_service is not None:
                res = self.market_service.backfill_symbol_timeframe(sym, tf, force_full=True)
            self._log(f"Backfill result: {json.dumps(res)}")
            QMessageBox.information(self, "Backfill", "Backfill request completed (see log).")
            self.refresh(limit=200)
        except Exception as e:
            logger.exception("HistoryTab backfill failed: {}", e)
            QMessageBox.warning(self, "Backfill failed", str(e))

    def _log(self, msg: str):
        logger.info(msg)

    def refresh(self, limit: int = 200):
        meta = MetaData()
        meta.reflect(bind=self.db_service.engine, only=["market_data_candles"])
        tbl = meta.tables.get("market_data_candles")
        if tbl is None:
            self._populate_empty()
            return
        with self.db_service.engine.connect() as conn:
            stmt = select(tbl).where(tbl.c.symbol == self.symbol_combo.currentText()).where(tbl.c.timeframe == self.tf_combo.currentText()).order_by(tbl.c.ts_utc.desc()).limit(limit)
            rows = conn.execute(stmt).fetchall()
            if not rows:
                self._populate_empty()
                return
            self.table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                self.table.setItem(i, 0, QTableWidgetItem(str(r["id"] if "id" in r else i)))
                self.table.setItem(i, 1, QTableWidgetItem(str(r["symbol"])))
                self.table.setItem(i, 2, QTableWidgetItem(str(r["timeframe"])))
                self.table.setItem(i, 3, QTableWidgetItem(str(int(r["ts_utc"]))))
                self.table.setItem(i, 4, QTableWidgetItem(str(r["open"])))
                self.table.setItem(i, 5, QTableWidgetItem(str(r["high"])))
                # additional columns can be added if needed
            self.table.resizeColumnsToContents()
