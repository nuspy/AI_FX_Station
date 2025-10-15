"""
News and Economic Calendar Tab for multi-provider integration.

Displays news events and economic calendar from providers.
"""
from __future__ import annotations

from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QLabel,
    QPushButton,
    QTabWidget,
    QHeaderView,
)
from loguru import logger


class NewsCalendarTab(QWidget):
    """Tab showing news and economic calendar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = None  # Will be set by main app

        # Setup UI
        layout = QVBoxLayout(self)

        # Tab widget for News and Calendar
        self.tabs = QTabWidget()

        # News tab
        self.news_widget = QWidget()
        news_layout = QVBoxLayout(self.news_widget)

        news_controls = QHBoxLayout()
        news_controls.addWidget(QLabel("Currency:"))
        self.news_currency_filter = QComboBox()
        self.news_currency_filter.addItems(["All", "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"])
        news_controls.addWidget(self.news_currency_filter)

        news_controls.addWidget(QLabel("Impact:"))
        self.news_impact_filter = QComboBox()
        self.news_impact_filter.addItems(["All", "high", "medium", "low"])
        news_controls.addWidget(self.news_impact_filter)

        self.news_refresh_btn = QPushButton("Refresh")
        news_controls.addWidget(self.news_refresh_btn)
        news_controls.addStretch()

        news_layout.addLayout(news_controls)

        # News table
        self.news_table = QTableWidget()
        self.news_table.setColumnCount(6)
        self.news_table.setHorizontalHeaderLabels(["Time", "Currency", "Impact", "Title", "Category", "Provider"])
        self.news_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.news_table.setAlternatingRowColors(True)
        self.news_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.news_table.setSelectionBehavior(QTableWidget.SelectRows)
        news_layout.addWidget(self.news_table)

        self.tabs.addTab(self.news_widget, "News")

        # Calendar tab
        self.calendar_widget = QWidget()
        calendar_layout = QVBoxLayout(self.calendar_widget)

        calendar_controls = QHBoxLayout()
        calendar_controls.addWidget(QLabel("Currency:"))
        self.calendar_currency_filter = QComboBox()
        self.calendar_currency_filter.addItems(["All", "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"])
        calendar_controls.addWidget(self.calendar_currency_filter)

        calendar_controls.addWidget(QLabel("Impact:"))
        self.calendar_impact_filter = QComboBox()
        self.calendar_impact_filter.addItems(["All", "high", "medium", "low"])
        calendar_controls.addWidget(self.calendar_impact_filter)

        self.calendar_refresh_btn = QPushButton("Refresh")
        calendar_controls.addWidget(self.calendar_refresh_btn)
        calendar_controls.addStretch()

        calendar_layout.addLayout(calendar_controls)

        # Calendar table
        self.calendar_table = QTableWidget()
        self.calendar_table.setColumnCount(8)
        self.calendar_table.setHorizontalHeaderLabels([
            "Time", "Event", "Currency", "Impact", "Forecast", "Previous", "Actual", "Provider"
        ])
        self.calendar_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.calendar_table.setAlternatingRowColors(True)
        self.calendar_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.calendar_table.setSelectionBehavior(QTableWidget.SelectRows)
        calendar_layout.addWidget(self.calendar_table)

        self.tabs.addTab(self.calendar_widget, "Economic Calendar")

        layout.addWidget(self.tabs)

        # Connections
        self.news_refresh_btn.clicked.connect(self._refresh_news)
        self.calendar_refresh_btn.clicked.connect(self._refresh_calendar)
        self.news_currency_filter.currentTextChanged.connect(self._refresh_news)
        self.news_impact_filter.currentTextChanged.connect(self._refresh_news)
        self.calendar_currency_filter.currentTextChanged.connect(self._refresh_calendar)
        self.calendar_impact_filter.currentTextChanged.connect(self._refresh_calendar)

        # Auto-refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._auto_refresh)
        self.refresh_timer.start(60000)  # Refresh every minute

    def set_engine(self, engine):
        """Set database engine."""
        self.engine = engine
        self._refresh_news()
        self._refresh_calendar()

    def _refresh_news(self):
        """Refresh news from database."""
        if not self.engine:
            return

        try:
            from sqlalchemy import text

            currency = self.news_currency_filter.currentText()
            impact = self.news_impact_filter.currentText()

            # Build query
            query = "SELECT ts_utc, currency, impact, title, category, provider FROM news_events WHERE 1=1"
            params = {}

            if currency != "All":
                query += " AND currency = :currency"
                params["currency"] = currency

            if impact != "All":
                query += " AND impact = :impact"
                params["impact"] = impact

            # Get last 100 news items
            query += " ORDER BY ts_utc DESC LIMIT 100"

            with self.engine.connect() as conn:
                rows = conn.execute(text(query), params).fetchall()

            # Populate table
            self.news_table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                ts_utc, currency, impact, title, category, provider = row

                # Format timestamp
                dt = datetime.fromtimestamp(ts_utc / 1000, tz=timezone.utc)
                time_str = dt.strftime("%Y-%m-%d %H:%M")

                self.news_table.setItem(i, 0, QTableWidgetItem(time_str))
                self.news_table.setItem(i, 1, QTableWidgetItem(currency or ""))
                self.news_table.setItem(i, 2, QTableWidgetItem(impact or ""))
                self.news_table.setItem(i, 3, QTableWidgetItem(title or ""))
                self.news_table.setItem(i, 4, QTableWidgetItem(category or ""))
                self.news_table.setItem(i, 5, QTableWidgetItem(provider or ""))

                # Color code by impact
                if impact == "high":
                    for col in range(6):
                        item = self.news_table.item(i, col)
                        if item:
                            item.setForeground(Qt.red)
                elif impact == "medium":
                    for col in range(6):
                        item = self.news_table.item(i, col)
                        if item:
                            item.setForeground(Qt.yellow)

            logger.debug(f"Refreshed news: {len(rows)} items")

        except Exception as e:
            logger.error(f"Failed to refresh news: {e}")

    def _refresh_calendar(self):
        """Refresh economic calendar from database."""
        if not self.engine:
            return

        try:
            from sqlalchemy import text

            currency = self.calendar_currency_filter.currentText()
            impact = self.calendar_impact_filter.currentText()

            # Build query - get upcoming events
            now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
            query = "SELECT ts_utc, event_name, currency, impact, forecast, previous, actual, provider " \
                    "FROM economic_calendar WHERE ts_utc >= :now_ts"
            params = {"now_ts": now_ts}

            if currency != "All":
                query += " AND currency = :currency"
                params["currency"] = currency

            if impact != "All":
                query += " AND impact = :impact"
                params["impact"] = impact

            # Get next 50 events
            query += " ORDER BY ts_utc ASC LIMIT 50"

            with self.engine.connect() as conn:
                rows = conn.execute(text(query), params).fetchall()

            # Populate table
            self.calendar_table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                ts_utc, event_name, currency, impact, forecast, previous, actual, provider = row

                # Format timestamp
                dt = datetime.fromtimestamp(ts_utc / 1000, tz=timezone.utc)
                time_str = dt.strftime("%Y-%m-%d %H:%M")

                self.calendar_table.setItem(i, 0, QTableWidgetItem(time_str))
                self.calendar_table.setItem(i, 1, QTableWidgetItem(event_name or ""))
                self.calendar_table.setItem(i, 2, QTableWidgetItem(currency or ""))
                self.calendar_table.setItem(i, 3, QTableWidgetItem(impact or ""))
                self.calendar_table.setItem(i, 4, QTableWidgetItem(forecast or ""))
                self.calendar_table.setItem(i, 5, QTableWidgetItem(previous or ""))
                self.calendar_table.setItem(i, 6, QTableWidgetItem(actual or ""))
                self.calendar_table.setItem(i, 7, QTableWidgetItem(provider or ""))

                # Color code by impact
                if impact == "high":
                    for col in range(8):
                        item = self.calendar_table.item(i, col)
                        if item:
                            item.setForeground(Qt.red)
                elif impact == "medium":
                    for col in range(8):
                        item = self.calendar_table.item(i, col)
                        if item:
                            item.setForeground(Qt.yellow)

            logger.debug(f"Refreshed calendar: {len(rows)} events")

        except Exception as e:
            logger.error(f"Failed to refresh calendar: {e}")

    def _auto_refresh(self):
        """Auto-refresh based on current tab."""
        current_tab = self.tabs.currentWidget()
        if current_tab == self.news_widget:
            self._refresh_news()
        elif current_tab == self.calendar_widget:
            self._refresh_calendar()
