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
# ui/signals_tab.py
# Widget to display persisted signals from the DB with admin controls for regime/index management

from typing import Optional
import threading
import json
import time

import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QLabel, QTextEdit, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt
from loguru import logger
from sqlalchemy import MetaData, select

from ..services.db_service import DBService
from ..services.regime_service import RegimeService
from ..services.scheduler import RegimeScheduler
from ..services.monitor import RegimeMonitor


class SignalsTab(QWidget):
    """
    Displays recent signals persisted in the DB and provides admin controls:
    - Refresh signals
    - Rebuild regime index (async)
    - Incremental update (one batch)
    - Start/Stop scheduler
    - Show index metrics and monitor metrics
    Logs actions into a small console area.
    """
    def __init__(self, parent=None, db_service: Optional[DBService] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.header = QLabel("Signals & Admin")
        self.header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.header)

        # signals table and refresh
        top_h = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Signals")
        self.refresh_btn.clicked.connect(self.on_refresh_clicked)
        top_h.addWidget(self.refresh_btn)

        # Provider selector + poll interval for non-streaming providers
        try:
            from PySide6.QtWidgets import QComboBox, QSpinBox, QLabel
            self.provider_combo = QComboBox()
            # will be populated if market_service provided
            self.provider_combo.addItems(["tiingo", "alpha_vantage"])
            top_h.addWidget(QLabel("Provider:"))
            top_h.addWidget(self.provider_combo)
            self.provider_apply_btn = QPushButton("Apply Provider")
            top_h.addWidget(self.provider_apply_btn)
            self.poll_spin = QSpinBox()
            self.poll_spin.setRange(1, 3600)
            self.poll_spin.setValue(1)
            top_h.addWidget(QLabel("Poll s:"))
            top_h.addWidget(self.poll_spin)
            self.provider_apply_btn.clicked.connect(self.on_apply_provider)
        except Exception:
            self.provider_combo = None
            self.provider_apply_btn = None
            self.poll_spin = None

        self.table = QTableWidget()
        self.layout.addLayout(top_h)
        self.layout.addWidget(self.table)

        # admin controls
        admin_h = QHBoxLayout()
        self.rebuild_btn = QPushButton("Rebuild Regime Index")
        self.rebuild_btn.clicked.connect(self.on_rebuild_clicked)
        admin_h.addWidget(self.rebuild_btn)

        # Settings button
        self.settings_btn = QPushButton("Settings")
        # connect to settings handler (may open settings dialog)
        self.settings_btn.clicked.connect(self.on_settings_clicked)
        admin_h.addWidget(self.settings_btn)

        # Admin login for remote admin actions
        self.login_btn = QPushButton("Login (Admin)")
        # connect to login handler (opens login dialog)
        self.login_btn.clicked.connect(self.on_login_clicked)
        admin_h.addWidget(self.login_btn)

        self.incremental_btn = QPushButton("Incremental Update")
        self.incremental_btn.clicked.connect(self.on_incremental_clicked)
        admin_h.addWidget(self.incremental_btn)

        self.start_sched_btn = QPushButton("Start Scheduler")
        self.start_sched_btn.clicked.connect(self.on_start_scheduler)
        admin_h.addWidget(self.start_sched_btn)

        self.stop_sched_btn = QPushButton("Stop Scheduler")
        self.stop_sched_btn.clicked.connect(self.on_stop_scheduler)
        admin_h.addWidget(self.stop_sched_btn)

        self.metrics_btn = QPushButton("Show Index Metrics")
        self.metrics_btn.clicked.connect(self.on_show_index_metrics)
        admin_h.addWidget(self.metrics_btn)

        self.monitor_btn = QPushButton("Show Monitor Metrics")
        self.monitor_btn.clicked.connect(self.on_show_monitor_metrics)
        admin_h.addWidget(self.monitor_btn)

        self.layout.addLayout(admin_h)

        # log area
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(160)
        self.layout.addWidget(self.log)

    def on_apply_provider(self):
        try:
            svc = getattr(self, "market_service", None)
            if svc is None:
                # try attribute set by setup_ui
                svc = getattr(self, "db_service", None) and getattr(self.db_service, "market_service", None)
            if svc is None:
                # try to import controller market_service from main app (best-effort)
                try:
                    from ..services.marketdata import MarketDataService
                    svc = MarketDataService()
                except Exception:
                    svc = None
            if svc is None:
                self._log("Cannot find MarketDataService to apply provider")
                return
            pname = self.provider_combo.currentText() if self.provider_combo else "tiingo"
            svc.set_provider(pname)
            if self.poll_spin:
                svc.set_poll_interval(self.poll_spin.value())
            self._log(f"Provider set to {pname}, poll_interval={svc.poll_interval()}")
        except Exception as e:
            logger.exception("Failed to apply provider: {}", e)
            self._log(f"Failed to apply provider: {e}")

        # DB and services
        self.db_service = db_service or DBService()
        self.regime_service = RegimeService(engine=self.db_service.engine)
        # local scheduler/monitor (do not conflict with server instances)
        self.scheduler: Optional[RegimeScheduler] = None
        self.monitor: Optional[RegimeMonitor] = None

        # initialize empty table
        self._populate_empty()

    def _log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        logger.info(msg)

    def _populate_empty(self):
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ts", "symbol", "timeframe", "entry", "target", "stop"])
        self.table.setRowCount(0)

    def on_refresh_clicked(self):
        try:
            self.refresh(limit=100)
            self._log("Refreshed signals table")
        except Exception as e:
            logger.exception("SignalsTab refresh failed: {}", e)
            QMessageBox.warning(self, "Refresh failed", str(e))

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
            raise

    def on_rebuild_clicked(self):
        def worker():
            try:
                self._log("Triggering full rebuild (async)...")
                res = self.regime_service.rebuild_async(n_clusters=8, limit=None)
                self._log(f"Rebuild started: {res}")
            except Exception as e:
                self._log(f"Rebuild failed to start: {e}")
        threading.Thread(target=worker, daemon=True).start()

    # --- Added safe handlers for buttons that may be connected by setup_ui ---
    def on_settings_clicked(self):
        """
        Open settings dialog (stub). Real app should show settings_dialog.
        Provide a safe placeholder so UI initialization does not fail.
        """
        try:
            # Try to import real settings dialog if available
            try:
                from .settings_dialog import SettingsDialog  # type: ignore
                dlg = SettingsDialog(parent=self)
                dlg.exec()
                self._log("Settings dialog closed")
            except Exception:
                # Fallback: show a message box or log
                self._log("Settings clicked (no settings dialog available)")
                try:
                    QMessageBox.information(self, "Settings", "Settings dialog not available in this build.")
                except Exception:
                    pass
        except Exception as e:
            logger.exception("on_settings_clicked failed: {}", e)

    def on_login_clicked(self):
        """
        Admin login handler (stub). Real app should open admin_login_dialog.
        """
        try:
            try:
                from .admin_login_dialog import AdminLoginDialog  # type: ignore
                dlg = AdminLoginDialog(parent=self)
                res = dlg.exec()
                self._log(f"Admin login dialog closed (result={res})")
            except Exception:
                self._log("Admin login requested (no dialog available)")
                try:
                    QMessageBox.information(self, "Admin Login", "Admin login dialog not available in this build.")
                except Exception:
                    pass
        except Exception as e:
            logger.exception("on_login_clicked failed: {}", e)

    def on_incremental_clicked(self):
        def worker():
            try:
                self._log("Running incremental update (one batch)...")
                res = self.regime_service.incremental_update(batch_size=500)
                self._log(f"Incremental update result: {res}")
            except Exception as e:
                self._log(f"Incremental update failed: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def on_start_scheduler(self):
        if self.scheduler is None or not getattr(self.scheduler, "_thread", None):
            try:
                self.scheduler = RegimeScheduler(engine=self.db_service.engine, interval_seconds=60, batch_size=500)
                self.scheduler.start()
                self._log("Started local RegimeScheduler")
            except Exception as e:
                self._log(f"Failed to start scheduler: {e}")
        else:
            self._log("Scheduler already running")

    def on_stop_scheduler(self):
        if self.scheduler is not None:
            try:
                self.scheduler.stop()
                self._log("Stopped local RegimeScheduler")
            except Exception as e:
                self._log(f"Failed to stop scheduler: {e}")
        else:
            self._log("No local scheduler to stop")

    def on_show_index_metrics(self):
        try:
            metrics = self.regime_service.get_index_metrics()
            self._log(f"Index metrics: {json.dumps(metrics)}")
            QMessageBox.information(self, "Index Metrics", json.dumps(metrics, indent=2))
        except Exception as e:
            self._log(f"Failed to get index metrics: {e}")
            QMessageBox.warning(self, "Metrics failed", str(e))

    def on_show_monitor_metrics(self):
        try:
            # attempt to connect to a running RegimeMonitor (if server running) else show local metrics
            try:
                rem = RegimeMonitor(engine=self.db_service.engine)
                # get current metrics (may start a monitor thread internally); prefer to read direct metrics
                metrics = rem.get_metrics()
            except Exception:
                metrics = {}
            self._log(f"Monitor metrics: {json.dumps(metrics)}")
            QMessageBox.information(self, "Monitor Metrics", json.dumps(metrics, indent=2))
        except Exception as e:
            self._log(f"Failed to get monitor metrics: {e}")
            QMessageBox.warning(self, "Monitor failed", str(e))
    def _populate_empty(self):
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ts", "symbol", "timeframe", "entry", "target", "stop"])
        self.table.setRowCount(0)

    def on_refresh_clicked(self):
        try:
            self.refresh(limit=100)
            # update UI log area so user sees the refresh happened
            try:
                self._log("Refreshed signals table")
            except Exception:
                # best-effort: do not crash UI if logging widget is unavailable
                pass
        except Exception as e:
            logger.exception("SignalsTab refresh failed: {}", e)
            try:
                QMessageBox.warning(self, "Refresh failed", str(e))
            except Exception:
                pass

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
