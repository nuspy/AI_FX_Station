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

        # Backfill years selector
        top.addWidget(QLabel("Years:"))
        self.years_combo = QComboBox()
        self.years_combo.addItems([str(x) for x in [1,2,5,10,20]])
        self.years_combo.setCurrentText("20")
        top.addWidget(self.years_combo)

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
            # ensure Tiingo is used as provider for historical data by default
            try:
                self.market_service.set_provider("tiingo")
            except Exception:
                pass
        except Exception:
            self.market_service = market_service

        # linked chart tab (set by app)
        self.chart_tab = None

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
            years = int(self.years_combo.currentText()) if self.years_combo else None
            if self.market_service is not None:
                self._log(f"Using provider '{self.market_service.provider_name()}' for backfill")
                res = self.market_service.backfill_symbol_timeframe(sym, tf, force_full=True, years=years)
            else:
                res = {}
            self._log(f"Backfill result: {json.dumps(res)}")
            QMessageBox.information(self, "Backfill", "Backfill request completed (see log).")
            self.refresh(limit=200)
            # update chart if connected
            if self.chart_tab is not None:
                try:
                    # load data from DB into DataFrame and plot
                    meta = MetaData()
                    meta.reflect(bind=self.db_service.engine, only=["market_data_candles"])
                    tbl = meta.tables.get("market_data_candles")
                    if tbl is not None:
                        with self.db_service.engine.connect() as conn:
                            stmt = select(tbl).where(tbl.c.symbol == sym).where(tbl.c.timeframe == tf).order_by(tbl.c.ts_utc.asc())
                            rows = conn.execute(stmt).fetchall()
                            import pandas as pd
                            if rows:
                                df = pd.DataFrame([dict(r) for r in rows])
                                # let chart_tab know symbol/timeframe and update
                                try:
                                    self.chart_tab.set_symbol_timeframe(self.db_service, sym, tf)
                                except Exception:
                                    pass
                                self.chart_tab.update_plot(df, timeframe=tf)
                except Exception as e:
                    logger.exception("Failed to update chart after backfill: {}", e)
        except Exception as e:
            logger.exception("HistoryTab backfill failed: {}", e)
            QMessageBox.warning(self, "Backfill failed", str(e))

    def _log(self, msg: str):
        logger.info(msg)

    def refresh(self, limit: int = 200):
        # log provider for diagnostics
        try:
            pname = self.market_service.provider_name() if self.market_service is not None else "unknown"
            self._log(f"Refreshing history (provider={pname}) for {self.symbol_combo.currentText()} {self.tf_combo.currentText()}")
        except Exception:
            pass

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
            # show table rows
            self.table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                self.table.setItem(i, 0, QTableWidgetItem(str(r["id"] if "id" in r else i)))
                self.table.setItem(i, 1, QTableWidgetItem(str(r["symbol"])))
                self.table.setItem(i, 2, QTableWidgetItem(str(r["timeframe"])))
                # convert ts to local string for table too
                try:
                    import datetime
                    ts_local = datetime.datetime.fromtimestamp(int(r["ts_utc"]) / 1000, tz=datetime.timezone.utc).astimezone()
                    ts_str = ts_local.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    ts_str = str(int(r["ts_utc"]))
                self.table.setItem(i, 3, QTableWidgetItem(ts_str))
                self.table.setItem(i, 4, QTableWidgetItem(str(r["open"])))
                self.table.setItem(i, 5, QTableWidgetItem(str(r["high"])))
            self.table.resizeColumnsToContents()

        # if chart connected, update chart with fetched rows (ascending order)
        if getattr(self, "chart_tab", None) is not None and rows:
            import pandas as pd
            df = pd.DataFrame([dict(r) for r in rows[::-1]])  # reverse to ascending
            try:
                self.chart_tab.set_symbol_timeframe(self.db_service, self.symbol_combo.currentText(), self.tf_combo.currentText())
            except Exception:
                pass
            self.chart_tab.update_plot(df, timeframe=self.tf_combo.currentText())
