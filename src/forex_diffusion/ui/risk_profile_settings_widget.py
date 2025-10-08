"""
Risk Profile Settings Widget

Widget for selecting and configuring risk management profiles.
Can be used as a standalone widget or embedded in settings dialog.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QPushButton, QLabel, QFormLayout, QDoubleSpinBox, QSpinBox,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor
from loguru import logger


class RiskProfileSettingsWidget(QWidget):
    """
    Widget for managing risk profiles.

    Features:
    - ComboBox to select risk profile (Conservative, Moderate, Aggressive, Custom)
    - Display current profile settings (read-only)
    - Activate button to switch profiles
    - Create custom profile button
    - Profile comparison table
    """

    profile_changed = Signal(str)  # Emitted when profile is activated

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._risk_profile_loader = None
        self._current_profile = None

        self._setup_ui()
        self._load_profiles()

        logger.info("RiskProfileSettingsWidget initialized")

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h3>Risk Profile Management</h3>")
        layout.addWidget(title)

        # Profile selector section
        selector_group = QGroupBox("Active Risk Profile")
        selector_layout = QHBoxLayout(selector_group)

        selector_layout.addWidget(QLabel("Profile:"))

        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(200)
        selector_layout.addWidget(self.profile_combo)

        self.activate_btn = QPushButton("Activate")
        self.activate_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        self.activate_btn.clicked.connect(self._on_activate_clicked)
        selector_layout.addWidget(self.activate_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._load_profiles)
        selector_layout.addWidget(self.refresh_btn)

        selector_layout.addStretch()

        layout.addWidget(selector_group)

        # Current profile details
        details_group = QGroupBox("Current Profile Settings")
        details_layout = QFormLayout(details_group)

        self.profile_name_label = QLabel("-")
        self.profile_type_label = QLabel("-")
        self.risk_per_trade_label = QLabel("-")
        self.max_portfolio_risk_label = QLabel("-")
        self.position_sizing_label = QLabel("-")
        self.kelly_fraction_label = QLabel("-")
        self.sl_multiplier_label = QLabel("-")
        self.tp_multiplier_label = QLabel("-")
        self.max_positions_label = QLabel("-")
        self.max_drawdown_label = QLabel("-")

        details_layout.addRow("Profile Name:", self.profile_name_label)
        details_layout.addRow("Profile Type:", self.profile_type_label)
        details_layout.addRow("Risk per Trade:", self.risk_per_trade_label)
        details_layout.addRow("Max Portfolio Risk:", self.max_portfolio_risk_label)
        details_layout.addRow("Position Sizing Method:", self.position_sizing_label)
        details_layout.addRow("Kelly Fraction:", self.kelly_fraction_label)
        details_layout.addRow("Stop Loss (ATR multiplier):", self.sl_multiplier_label)
        details_layout.addRow("Take Profit (ATR multiplier):", self.tp_multiplier_label)
        details_layout.addRow("Max Total Positions:", self.max_positions_label)
        details_layout.addRow("Max Drawdown:", self.max_drawdown_label)

        layout.addWidget(details_group)

        # Profile comparison table
        comparison_group = QGroupBox("Profile Comparison")
        comparison_layout = QVBoxLayout(comparison_group)

        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(4)
        self.comparison_table.setHorizontalHeaderLabels([
            "Setting", "Conservative", "Moderate", "Aggressive"
        ])
        self.comparison_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.comparison_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.comparison_table.setAlternatingRowColors(True)

        header = self.comparison_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        comparison_layout.addWidget(self.comparison_table)

        layout.addWidget(comparison_group)

        # Action buttons
        actions_layout = QHBoxLayout()

        self.create_custom_btn = QPushButton("Create Custom Profile")
        self.create_custom_btn.clicked.connect(self._on_create_custom_clicked)
        actions_layout.addWidget(self.create_custom_btn)

        actions_layout.addStretch()

        layout.addLayout(actions_layout)

    def set_risk_profile_loader(self, loader):
        """Set the risk profile loader service."""
        self._risk_profile_loader = loader
        self._load_profiles()
        logger.info("Risk profile loader set")

    def _load_profiles(self):
        """Load available profiles from database."""
        if not self._risk_profile_loader:
            logger.warning("Risk profile loader not set")
            return

        try:
            # Get all profiles
            from forex_diffusion.services.risk_profile_loader import RiskProfileLoader

            if not isinstance(self._risk_profile_loader, RiskProfileLoader):
                logger.error("Invalid risk profile loader")
                return

            profiles = self._risk_profile_loader.list_all_profiles()

            # Update combo box
            self.profile_combo.clear()
            active_profile_name = None

            for name, info in profiles.items():
                self.profile_combo.addItem(name)
                if info.get('is_active'):
                    active_profile_name = name

            # Set active profile as current
            if active_profile_name:
                index = self.profile_combo.findText(active_profile_name)
                if index >= 0:
                    self.profile_combo.setCurrentIndex(index)
                    self._load_profile_details(active_profile_name)

            # Update comparison table
            self._update_comparison_table(profiles)

            logger.info(f"Loaded {len(profiles)} risk profiles")

        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load risk profiles: {str(e)}"
            )

    def _load_profile_details(self, profile_name: str):
        """Load and display details of a specific profile."""
        if not self._risk_profile_loader:
            return

        try:
            profile = self._risk_profile_loader.load_profile_by_name(profile_name)

            if not profile:
                logger.warning(f"Profile not found: {profile_name}")
                return

            self._current_profile = profile

            # Update labels
            self.profile_name_label.setText(profile.profile_name)
            self.profile_type_label.setText(profile.profile_type.capitalize())
            self.risk_per_trade_label.setText(f"{profile.max_risk_per_trade_pct:.2f}%")
            self.max_portfolio_risk_label.setText(f"{profile.max_portfolio_risk_pct:.2f}%")
            self.position_sizing_label.setText(profile.position_sizing_method.replace('_', ' ').title())
            self.kelly_fraction_label.setText(f"{profile.kelly_fraction:.2f}")
            self.sl_multiplier_label.setText(f"{profile.base_sl_atr_multiplier:.2f}x")
            self.tp_multiplier_label.setText(f"{profile.base_tp_atr_multiplier:.2f}x")
            self.max_positions_label.setText(str(profile.max_total_positions))
            self.max_drawdown_label.setText(f"{profile.max_drawdown_pct:.2f}%")

            # Highlight if active
            if profile.is_active:
                self.profile_name_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            else:
                self.profile_name_label.setStyleSheet("")

            logger.info(f"Loaded profile details: {profile_name}")

        except Exception as e:
            logger.error(f"Error loading profile details: {e}")

    def _update_comparison_table(self, profiles: Dict[str, Dict[str, Any]]):
        """Update the comparison table with profile data."""
        # Define rows
        rows = [
            ("Risk per Trade", "max_risk_per_trade_pct", "%"),
            ("Max Portfolio Risk", "max_portfolio_risk_pct", "%"),
            ("Position Sizing", "position_sizing_method", ""),
            ("Kelly Fraction", "kelly_fraction", ""),
            ("Stop Loss Multiplier", "base_sl_atr_multiplier", "x ATR"),
            ("Take Profit Multiplier", "base_tp_atr_multiplier", "x ATR"),
            ("Max Positions", "max_total_positions", ""),
            ("Max per Symbol", "max_positions_per_symbol", ""),
            ("Max Drawdown", "max_drawdown_pct", "%"),
            ("Daily Loss Limit", "max_daily_loss_pct", "%"),
        ]

        self.comparison_table.setRowCount(len(rows))

        # Get predefined profiles
        conservative = profiles.get('Conservative', {})
        moderate = profiles.get('Moderate', {})
        aggressive = profiles.get('Aggressive', {})

        for row_idx, (label, key, suffix) in enumerate(rows):
            # Setting name
            setting_item = QTableWidgetItem(label)
            self.comparison_table.setItem(row_idx, 0, setting_item)

            # Conservative value
            cons_val = conservative.get(key, '-')
            cons_item = self._format_cell_value(cons_val, suffix)
            self.comparison_table.setItem(row_idx, 1, cons_item)

            # Moderate value
            mod_val = moderate.get(key, '-')
            mod_item = self._format_cell_value(mod_val, suffix)
            self.comparison_table.setItem(row_idx, 2, mod_item)

            # Aggressive value
            agg_val = aggressive.get(key, '-')
            agg_item = self._format_cell_value(agg_val, suffix)
            self.comparison_table.setItem(row_idx, 3, agg_item)

    def _format_cell_value(self, value: Any, suffix: str) -> QTableWidgetItem:
        """Format cell value with suffix."""
        if isinstance(value, (int, float)):
            if suffix == "%":
                text = f"{value:.2f}{suffix}"
            elif suffix:
                text = f"{value} {suffix}"
            else:
                text = str(value)
        elif isinstance(value, str):
            text = value.replace('_', ' ').title()
        else:
            text = str(value)

        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    @Slot()
    def _on_activate_clicked(self):
        """Activate the selected profile."""
        if not self._risk_profile_loader:
            QMessageBox.warning(
                self,
                "Warning",
                "Risk profile loader not configured."
            )
            return

        profile_name = self.profile_combo.currentText()
        if not profile_name:
            return

        reply = QMessageBox.question(
            self,
            "Activate Profile",
            f"Activate risk profile '{profile_name}'?\n\n"
            f"This will deactivate any currently active profile.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                success = self._risk_profile_loader.activate_profile(profile_name)

                if success:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Profile '{profile_name}' activated successfully!"
                    )
                    self._load_profiles()
                    self.profile_changed.emit(profile_name)
                    logger.info(f"Profile activated: {profile_name}")
                else:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"Failed to activate profile '{profile_name}'."
                    )

            except Exception as e:
                logger.error(f"Error activating profile: {e}")
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to activate profile: {str(e)}"
                )

    @Slot()
    def _on_create_custom_clicked(self):
        """Create a custom risk profile."""
        QMessageBox.information(
            self,
            "Create Custom Profile",
            "Custom profile creation dialog coming soon!\n\n"
            "For now, you can create custom profiles by inserting directly into the database:\n\n"
            "INSERT INTO risk_profiles (profile_name, profile_type, ...) VALUES (...);"
        )
