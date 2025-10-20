"""
Regime Definition Manager Dialog

Allows users to create, edit, and delete regime definitions via GUI.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QLineEdit,
    QTextEdit, QCheckBox, QAbstractItemView,
    QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt
from typing import Optional, Dict, Any
from loguru import logger
import json

from ..training.training_pipeline.database import (
    session_scope, get_all_regimes, create_regime_definition,
    update_regime_definition, delete_regime_definition
)


class RegimeDefinitionDialog(QDialog):
    """
    Dialog for managing regime definitions.

    Features:
    - List all regimes (active and inactive)
    - Add new regime definitions
    - Edit existing regimes
    - Delete regimes
    - Toggle active/inactive status
    """

    def __init__(self, parent: Optional[QDialog] = None):
        """Initialize regime definition dialog."""
        super().__init__(parent)

        self.setWindowTitle("Regime Definition Manager")
        self.resize(800, 600)

        self.init_ui()
        self.load_regimes()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h2>Regime Definition Manager</h2>")
        layout.addWidget(title)

        info = QLabel(
            "Manage market regime definitions used for model selection and evaluation."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Regimes table
        regimes_group = self.create_regimes_table()
        layout.addWidget(regimes_group)

        # Action buttons
        action_layout = QHBoxLayout()

        self.add_btn = QPushButton("Add New Regime")
        self.add_btn.clicked.connect(self.add_regime)
        action_layout.addWidget(self.add_btn)

        self.edit_btn = QPushButton("Edit Selected")
        self.edit_btn.clicked.connect(self.edit_regime)
        self.edit_btn.setEnabled(False)
        action_layout.addWidget(self.edit_btn)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_regime)
        self.delete_btn.setEnabled(False)
        action_layout.addWidget(self.delete_btn)

        action_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_regimes)
        action_layout.addWidget(self.refresh_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        action_layout.addWidget(self.close_btn)

        layout.addLayout(action_layout)

    def create_regimes_table(self) -> QGroupBox:
        """Create regimes table."""
        group = QGroupBox("Regime Definitions")
        layout = QVBoxLayout(group)

        self.regimes_table = QTableWidget()
        self.regimes_table.setColumnCount(4)
        self.regimes_table.setHorizontalHeaderLabels([
            'Name', 'Description', 'Active', 'Created At'
        ])
        self.regimes_table.horizontalHeader().setStretchLastSection(True)
        self.regimes_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.regimes_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.regimes_table.itemSelectionChanged.connect(self.on_selection_changed)

        layout.addWidget(self.regimes_table)

        return group

    def load_regimes(self):
        """Load all regimes from database."""
        try:
            with session_scope() as session:
                regimes = get_all_regimes(session)

                self.regimes_table.setRowCount(len(regimes))

                for row, regime in enumerate(regimes):
                    # Store regime ID in row
                    self.regimes_table.setItem(row, 0, QTableWidgetItem(regime.regime_name))
                    self.regimes_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, regime.id)

                    # Description
                    self.regimes_table.setItem(row, 1, QTableWidgetItem(regime.description or ''))

                    # Active status
                    active_text = "✓" if regime.is_active else "✗"
                    active_item = QTableWidgetItem(active_text)
                    active_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.regimes_table.setItem(row, 2, active_item)

                    # Created at
                    created_text = regime.created_at.strftime('%Y-%m-%d %H:%M') if regime.created_at else 'N/A'
                    self.regimes_table.setItem(row, 3, QTableWidgetItem(created_text))

                self.regimes_table.resizeColumnsToContents()

                logger.info(f"Loaded {len(regimes)} regime definitions")

        except Exception as e:
            logger.error(f"Failed to load regimes: {e}")
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load regime definitions:\n{e}"
            )

    def on_selection_changed(self):
        """Handle selection change."""
        has_selection = len(self.regimes_table.selectionModel().selectedRows()) > 0
        self.edit_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)

    def add_regime(self):
        """Add new regime definition."""
        dialog = RegimeEditDialog(self, is_new=True)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            regime_data = dialog.get_regime_data()

            try:
                with session_scope() as session:
                    create_regime_definition(
                        session,
                        regime_name=regime_data['name'],
                        description=regime_data['description'],
                        detection_rules=regime_data['detection_rules'],
                        is_active=regime_data['is_active']
                    )

                QMessageBox.information(
                    self,
                    "Success",
                    f"Regime '{regime_data['name']}' created successfully."
                )

                self.load_regimes()

            except Exception as e:
                logger.error(f"Failed to create regime: {e}")
                QMessageBox.critical(
                    self,
                    "Create Error",
                    f"Failed to create regime:\n{e}"
                )

    def edit_regime(self):
        """Edit selected regime."""
        selected_rows = self.regimes_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        row = selected_rows[0].row()
        regime_id = self.regimes_table.item(row, 0).data(Qt.ItemDataRole.UserRole)

        # Load current regime data
        try:
            with session_scope() as session:
                from ..training.training_pipeline.database import get_regime_by_id

                regime = get_regime_by_id(session, regime_id)

                if not regime:
                    QMessageBox.warning(self, "Not Found", "Regime not found")
                    return

                # Open edit dialog
                dialog = RegimeEditDialog(
                    self,
                    is_new=False,
                    regime_name=regime.regime_name,
                    description=regime.description,
                    detection_rules=regime.detection_rules,
                    is_active=regime.is_active
                )

                if dialog.exec() == QDialog.DialogCode.Accepted:
                    regime_data = dialog.get_regime_data()

                    update_regime_definition(
                        session,
                        regime_id=regime_id,
                        description=regime_data['description'],
                        detection_rules=regime_data['detection_rules'],
                        is_active=regime_data['is_active']
                    )

                    QMessageBox.information(
                        self,
                        "Success",
                        f"Regime '{regime_data['name']}' updated successfully."
                    )

                    self.load_regimes()

        except Exception as e:
            logger.error(f"Failed to edit regime: {e}")
            QMessageBox.critical(
                self,
                "Edit Error",
                f"Failed to edit regime:\n{e}"
            )

    def delete_regime(self):
        """Delete selected regime."""
        selected_rows = self.regimes_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        row = selected_rows[0].row()
        regime_id = self.regimes_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        regime_name = self.regimes_table.item(row, 0).text()

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete regime '{regime_name}'?\n\n"
            f"This will also delete all associated best model records.\n"
            f"Training runs will not be affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                with session_scope() as session:
                    delete_regime_definition(session, regime_id)

                QMessageBox.information(
                    self,
                    "Success",
                    f"Regime '{regime_name}' deleted successfully."
                )

                self.load_regimes()

            except Exception as e:
                logger.error(f"Failed to delete regime: {e}")
                QMessageBox.critical(
                    self,
                    "Delete Error",
                    f"Failed to delete regime:\n{e}"
                )


class RegimeEditDialog(QDialog):
    """Dialog for editing/creating a regime definition."""

    def __init__(
        self,
        parent: Optional[QDialog] = None,
        is_new: bool = True,
        regime_name: str = "",
        description: str = "",
        detection_rules: Optional[Dict[str, Any]] = None,
        is_active: bool = True
    ):
        """Initialize regime edit dialog."""
        super().__init__(parent)

        self.is_new = is_new

        self.setWindowTitle("Add New Regime" if is_new else "Edit Regime")
        self.resize(500, 400)

        self.init_ui(regime_name, description, detection_rules, is_active)

    def init_ui(
        self,
        regime_name: str,
        description: str,
        detection_rules: Optional[Dict[str, Any]],
        is_active: bool
    ):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Form layout
        form_group = QGroupBox("Regime Information")
        form_layout = QFormLayout(form_group)

        # Name (only editable for new regimes)
        self.name_edit = QLineEdit(regime_name)
        self.name_edit.setEnabled(self.is_new)
        form_layout.addRow("Regime Name:", self.name_edit)

        # Description
        self.description_edit = QTextEdit(description)
        self.description_edit.setMaximumHeight(80)
        form_layout.addRow("Description:", self.description_edit)

        # Detection rules (JSON format)
        self.rules_edit = QTextEdit()
        if detection_rules:
            self.rules_edit.setPlainText(json.dumps(detection_rules, indent=2))
        else:
            # Default template
            template = {
                "trend_strength": "> 0.7",
                "returns": "> 0",
                "volatility": "< 50th percentile"
            }
            self.rules_edit.setPlainText(json.dumps(template, indent=2))
        form_layout.addRow("Detection Rules (JSON):", self.rules_edit)

        # Active checkbox
        self.active_check = QCheckBox("Active")
        self.active_check.setChecked(is_active)
        form_layout.addRow("", self.active_check)

        layout.addWidget(form_group)

        # Help text
        help_text = QLabel(
            "<b>Detection Rules Format:</b><br>"
            "JSON dictionary with condition expressions.<br>"
            "Example: {\"trend_strength\": \"> 0.7\", \"returns\": \"> 0\"}"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.validate_and_accept)
        button_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def validate_and_accept(self):
        """Validate inputs and accept dialog."""
        # Validate name
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Regime name is required")
            return

        # Validate JSON
        rules_text = self.rules_edit.toPlainText().strip()
        try:
            detection_rules = json.loads(rules_text)
        except json.JSONDecodeError as e:
            QMessageBox.warning(
                self,
                "Validation Error",
                f"Invalid JSON in detection rules:\n{e}"
            )
            return

        self.accept()

    def get_regime_data(self) -> Dict[str, Any]:
        """Get regime data from form."""
        return {
            'name': self.name_edit.text().strip(),
            'description': self.description_edit.toPlainText().strip(),
            'detection_rules': json.loads(self.rules_edit.toPlainText().strip()),
            'is_active': self.active_check.isChecked()
        }
