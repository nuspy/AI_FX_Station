"""
Enhanced Status Bar Widget

Extended status bar for main window with trading metrics:
- Active risk profile indicator (clickable)
- Portfolio exposure percentage (color-coded)
- Daily P&L counter (color-coded)
- Drawdown indicator (color-coded)

FASE 9 - Part 4
"""
from __future__ import annotations

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QFrame
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QCursor
from loguru import logger


class EnhancedStatusBar(QWidget):
    """
    Enhanced status bar widget with trading metrics.

    Features:
    - Active risk profile indicator (clickable to open settings)
    - Portfolio exposure percentage (color-coded)
    - Daily P&L counter (with color coding)
    - Drawdown indicator (color-coded: green/yellow/red)
    - Auto-refresh every second
    """

    profile_clicked = Signal()  # Emitted when profile indicator is clicked

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._current_profile = "Moderate"
        self._portfolio_exposure = 0.0
        self._daily_pnl = 0.0
        self._drawdown_pct = 0.0

        self._setup_ui()
        self._setup_timer()

        logger.info("EnhancedStatusBar initialized")

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(15)

        # === Status message (standard) ===
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # === Separator ===
        separator1 = self._create_separator()
        layout.addWidget(separator1)

        # === Active Risk Profile ===
        self.profile_btn = QPushButton()
        self.profile_btn.setFlat(True)
        self.profile_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.profile_btn.clicked.connect(self._on_profile_clicked)
        self.profile_btn.setToolTip("Click to open Risk Profile settings")
        layout.addWidget(self.profile_btn)

        # === Separator ===
        separator2 = self._create_separator()
        layout.addWidget(separator2)

        # === Portfolio Exposure ===
        exposure_label = QLabel("Exposure:")
        font = QFont()
        font.setBold(True)
        exposure_label.setFont(font)
        layout.addWidget(exposure_label)

        self.exposure_value_label = QLabel()
        self.exposure_value_label.setMinimumWidth(60)
        layout.addWidget(self.exposure_value_label)

        # === Separator ===
        separator3 = self._create_separator()
        layout.addWidget(separator3)

        # === Daily P&L ===
        pnl_label = QLabel("Daily P&L:")
        pnl_label.setFont(font)
        layout.addWidget(pnl_label)

        self.pnl_value_label = QLabel()
        self.pnl_value_label.setMinimumWidth(90)
        layout.addWidget(self.pnl_value_label)

        # === Separator ===
        separator4 = self._create_separator()
        layout.addWidget(separator4)

        # === Drawdown Indicator ===
        dd_label = QLabel("Drawdown:")
        dd_label.setFont(font)
        layout.addWidget(dd_label)

        self.drawdown_value_label = QLabel()
        self.drawdown_value_label.setMinimumWidth(60)
        layout.addWidget(self.drawdown_value_label)

        # Initial update
        self._update_display()

    def _create_separator(self) -> QFrame:
        """Create a vertical separator line."""
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator

    def _setup_timer(self):
        """Setup auto-refresh timer (1 second interval)."""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._auto_refresh)
        self.refresh_timer.start(1000)  # 1 second

    def _update_display(self):
        """Update all display elements."""
        # Update profile button
        profile_icon = self._get_profile_icon(self._current_profile)
        profile_color = self._get_profile_color(self._current_profile)
        self.profile_btn.setText(f"{profile_icon} {self._current_profile}")
        self.profile_btn.setStyleSheet(f"color: {profile_color}; font-weight: bold; border: none; padding: 2px 8px;")

        # Update exposure
        exposure_color = self._get_exposure_color(self._portfolio_exposure)
        self.exposure_value_label.setText(f"<b style='color: {exposure_color}'>{self._portfolio_exposure:.1f}%</b>")

        # Update daily P&L
        pnl_color = "#2ecc71" if self._daily_pnl >= 0 else "#e74c3c"
        pnl_sign = "+" if self._daily_pnl > 0 else ""
        self.pnl_value_label.setText(f"<b style='color: {pnl_color}'>{pnl_sign}${self._daily_pnl:,.2f}</b>")

        # Update drawdown
        dd_color = self._get_drawdown_color(self._drawdown_pct)
        self.drawdown_value_label.setText(f"<b style='color: {dd_color}'>{self._drawdown_pct:.1f}%</b>")

    def _get_profile_icon(self, profile: str) -> str:
        """Get icon for profile."""
        icons = {
            'Conservative': '🛡️',
            'Moderate': '⚖️',
            'Aggressive': '⚡',
            'Custom': '🔧'
        }
        return icons.get(profile, '⚙️')

    def _get_profile_color(self, profile: str) -> str:
        """Get color for profile."""
        colors = {
            'Conservative': '#3498db',  # Blue
            'Moderate': '#2ecc71',      # Green
            'Aggressive': '#e74c3c',    # Red
            'Custom': '#f39c12'         # Orange
        }
        return colors.get(profile, '#95a5a6')

    def _get_exposure_color(self, exposure: float) -> str:
        """Get color for portfolio exposure."""
        if exposure <= 30:
            return "#2ecc71"  # Green - safe
        elif exposure <= 50:
            return "#f39c12"  # Orange - moderate
        elif exposure <= 70:
            return "#e67e22"  # Dark orange - caution
        else:
            return "#e74c3c"  # Red - high risk

    def _get_drawdown_color(self, drawdown: float) -> str:
        """Get color for drawdown percentage."""
        if drawdown <= 5:
            return "#2ecc71"  # Green - minimal
        elif drawdown <= 10:
            return "#f39c12"  # Orange - acceptable
        elif drawdown <= 20:
            return "#e67e22"  # Dark orange - concerning
        else:
            return "#e74c3c"  # Red - critical

    def _auto_refresh(self):
        """Auto-refresh callback (called every second)."""
        # In production, this would query live data from trading engine
        # For now, it just updates the display with current values
        self._update_display()

    def _on_profile_clicked(self):
        """Handle profile button click."""
        self.profile_clicked.emit()
        logger.info("Risk profile indicator clicked")

    # === Public API for updating values ===

    def set_status(self, message: str):
        """Set main status message."""
        self.status_label.setText(message)

    def set_risk_profile(self, profile: str):
        """Set active risk profile."""
        self._current_profile = profile
        self._update_display()
        logger.debug(f"Risk profile set to: {profile}")

    def set_portfolio_exposure(self, exposure_pct: float):
        """Set portfolio exposure percentage."""
        self._portfolio_exposure = exposure_pct
        self._update_display()

    def set_daily_pnl(self, pnl: float):
        """Set daily P&L value."""
        self._daily_pnl = pnl
        self._update_display()

    def set_drawdown(self, drawdown_pct: float):
        """Set current drawdown percentage."""
        self._drawdown_pct = drawdown_pct
        self._update_display()

    def update_metrics(self, exposure_pct: float = None, daily_pnl: float = None, drawdown_pct: float = None):
        """Update multiple metrics at once."""
        if exposure_pct is not None:
            self._portfolio_exposure = exposure_pct
        if daily_pnl is not None:
            self._daily_pnl = daily_pnl
        if drawdown_pct is not None:
            self._drawdown_pct = drawdown_pct
        self._update_display()
