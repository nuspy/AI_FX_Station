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
    QAbstractItemView, QDialog, QLineEdit, QDialogButtonBox
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

        self.create_custom_btn = QPushButton("Create Profile")
        self.create_custom_btn.clicked.connect(self._on_create_custom_clicked)
        self.create_custom_btn.setStyleSheet("background-color: #3498db; color: white;")
        actions_layout.addWidget(self.create_custom_btn)

        self.edit_profile_btn = QPushButton("Edit Profile")
        self.edit_profile_btn.clicked.connect(self._on_edit_profile_clicked)
        self.edit_profile_btn.setStyleSheet("background-color: #f39c12; color: white;")
        actions_layout.addWidget(self.edit_profile_btn)

        self.delete_profile_btn = QPushButton("Delete Profile")
        self.delete_profile_btn.clicked.connect(self._on_delete_profile_clicked)
        self.delete_profile_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        actions_layout.addWidget(self.delete_profile_btn)

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
        if not self._risk_profile_loader:
            QMessageBox.warning(self, "Error", "Risk profile loader not available")
            return

        dialog = RiskProfileEditDialog(self, mode="create")
        if dialog.exec() == QDialog.Accepted:
            profile_data = dialog.get_profile_data()
            try:
                # Save to database via risk profile loader
                self._risk_profile_loader.create_profile(profile_data)
                QMessageBox.information(self, "Success", f"Profile '{profile_data['profile_name']}' created successfully!")
                self._load_profiles()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create profile: {str(e)}")

    def _on_edit_profile_clicked(self):
        """Edit selected risk profile."""
        if not self._risk_profile_loader:
            QMessageBox.warning(self, "Error", "Risk profile loader not available")
            return

        profile_name = self.profile_combo.currentText()
        if not profile_name:
            QMessageBox.warning(self, "Warning", "Please select a profile to edit")
            return

        # Prevent editing built-in profiles
        if profile_name.lower() in ['conservative', 'moderate', 'aggressive']:
            QMessageBox.warning(self, "Warning", "Cannot edit built-in profiles. Create a custom profile instead.")
            return

        try:
            profile = self._risk_profile_loader.load_profile_by_name(profile_name)
            if not profile:
                QMessageBox.warning(self, "Error", f"Profile '{profile_name}' not found")
                return

            dialog = RiskProfileEditDialog(self, mode="edit", profile_data=profile)
            if dialog.exec() == QDialog.Accepted:
                updated_data = dialog.get_profile_data()
                self._risk_profile_loader.update_profile(profile_name, updated_data)
                QMessageBox.information(self, "Success", f"Profile '{profile_name}' updated successfully!")
                self._load_profiles()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to edit profile: {str(e)}")

    def _on_delete_profile_clicked(self):
        """Delete selected risk profile."""
        if not self._risk_profile_loader:
            QMessageBox.warning(self, "Error", "Risk profile loader not available")
            return

        profile_name = self.profile_combo.currentText()
        if not profile_name:
            QMessageBox.warning(self, "Warning", "Please select a profile to delete")
            return

        # Prevent deleting built-in profiles
        if profile_name.lower() in ['conservative', 'moderate', 'aggressive']:
            QMessageBox.warning(self, "Warning", "Cannot delete built-in profiles.")
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the profile '{profile_name}'?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self._risk_profile_loader.delete_profile(profile_name)
                QMessageBox.information(self, "Success", f"Profile '{profile_name}' deleted successfully!")
                self._load_profiles()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete profile: {str(e)}")


class RiskProfileEditDialog(QDialog):
    """Dialog for creating/editing risk profiles."""

    def __init__(self, parent=None, mode="create", profile_data=None):
        super().__init__(parent)
        self.mode = mode
        self.profile_data = profile_data or {}

        self.setWindowTitle("Create Risk Profile" if mode == "create" else "Edit Risk Profile")
        self.setMinimumWidth(500)

        self._setup_ui()
        if profile_data:
            self._load_profile_data(profile_data)

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Form layout
        form_layout = QFormLayout()

        # Profile name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., My Custom Strategy")
        form_layout.addRow("Profile Name:", self.name_input)

        # Profile type
        self.type_input = QComboBox()
        self.type_input.addItems(["custom", "conservative", "moderate", "aggressive"])
        form_layout.addRow("Profile Type:", self.type_input)

        # Risk per trade (%)
        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setRange(0.1, 10.0)
        self.risk_per_trade.setSingleStep(0.1)
        self.risk_per_trade.setSuffix(" %")
        self.risk_per_trade.setValue(1.0)
        form_layout.addRow("Risk per Trade:", self.risk_per_trade)

        # Max portfolio risk (%)
        self.max_portfolio_risk = QDoubleSpinBox()
        self.max_portfolio_risk.setRange(1.0, 50.0)
        self.max_portfolio_risk.setSingleStep(1.0)
        self.max_portfolio_risk.setSuffix(" %")
        self.max_portfolio_risk.setValue(5.0)
        form_layout.addRow("Max Portfolio Risk:", self.max_portfolio_risk)

        # Position sizing method
        self.position_sizing = QComboBox()
        self.position_sizing.addItems(["fixed", "kelly", "volatility", "adaptive"])
        form_layout.addRow("Position Sizing:", self.position_sizing)

        # Kelly fraction
        self.kelly_fraction = QDoubleSpinBox()
        self.kelly_fraction.setRange(0.1, 1.0)
        self.kelly_fraction.setSingleStep(0.05)
        self.kelly_fraction.setValue(0.25)
        form_layout.addRow("Kelly Fraction:", self.kelly_fraction)

        # Stop loss multiplier (ATR)
        self.sl_multiplier = QDoubleSpinBox()
        self.sl_multiplier.setRange(0.5, 5.0)
        self.sl_multiplier.setSingleStep(0.1)
        self.sl_multiplier.setValue(1.5)
        form_layout.addRow("Stop Loss (ATR x):", self.sl_multiplier)

        # Take profit multiplier (ATR)
        self.tp_multiplier = QDoubleSpinBox()
        self.tp_multiplier.setRange(0.5, 10.0)
        self.tp_multiplier.setSingleStep(0.1)
        self.tp_multiplier.setValue(3.0)
        form_layout.addRow("Take Profit (ATR x):", self.tp_multiplier)

        # Max total positions
        self.max_positions = QSpinBox()
        self.max_positions.setRange(1, 20)
        self.max_positions.setValue(5)
        form_layout.addRow("Max Total Positions:", self.max_positions)

        # Max drawdown (%)
        self.max_drawdown = QDoubleSpinBox()
        self.max_drawdown.setRange(5.0, 50.0)
        self.max_drawdown.setSingleStep(1.0)
        self.max_drawdown.setSuffix(" %")
        self.max_drawdown.setValue(15.0)
        form_layout.addRow("Max Drawdown:", self.max_drawdown)

        layout.addLayout(form_layout)

        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _load_profile_data(self, data: Dict[str, Any]):
        """Load profile data into form."""
        self.name_input.setText(data.get('profile_name', ''))

        profile_type = data.get('profile_type', 'custom')
        index = self.type_input.findText(profile_type)
        if index >= 0:
            self.type_input.setCurrentIndex(index)

        self.risk_per_trade.setValue(data.get('risk_per_trade', 1.0))
        self.max_portfolio_risk.setValue(data.get('max_portfolio_risk', 5.0))

        sizing_method = data.get('position_sizing_method', 'fixed')
        index = self.position_sizing.findText(sizing_method)
        if index >= 0:
            self.position_sizing.setCurrentIndex(index)

        self.kelly_fraction.setValue(data.get('kelly_fraction', 0.25))
        self.sl_multiplier.setValue(data.get('stop_loss_atr_multiplier', 1.5))
        self.tp_multiplier.setValue(data.get('take_profit_atr_multiplier', 3.0))
        self.max_positions.setValue(data.get('max_total_positions', 5))
        self.max_drawdown.setValue(data.get('max_drawdown_percent', 15.0))

    def get_profile_data(self) -> Dict[str, Any]:
        """Get profile data from form."""
        return {
            'profile_name': self.name_input.text().strip(),
            'profile_type': self.type_input.currentText(),
            'risk_per_trade': self.risk_per_trade.value(),
            'max_portfolio_risk': self.max_portfolio_risk.value(),
            'position_sizing_method': self.position_sizing.currentText(),
            'kelly_fraction': self.kelly_fraction.value(),
            'stop_loss_atr_multiplier': self.sl_multiplier.value(),
            'take_profit_atr_multiplier': self.tp_multiplier.value(),
            'max_total_positions': self.max_positions.value(),
            'max_drawdown_percent': self.max_drawdown.value()
        }
