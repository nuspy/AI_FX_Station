"""
Checkpoint Selector Dialog

Allows users to select and load checkpoints for resuming training queues.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView,
    QAbstractItemView, QGroupBox
)
from PySide6.QtCore import Qt
from typing import Optional, Dict, Any, List
from loguru import logger
from pathlib import Path
import json
from datetime import datetime


class CheckpointSelectorDialog(QDialog):
    """
    Dialog for selecting training queue checkpoints.

    Features:
    - List all available checkpoints
    - Show checkpoint details (queue name, progress, date)
    - Load selected checkpoint
    - Delete old checkpoints
    """

    def __init__(self, parent: Optional[QDialog] = None, checkpoint_dir: Optional[str] = None):
        """
        Initialize checkpoint selector dialog.

        Args:
            parent: Parent widget
            checkpoint_dir: Directory containing checkpoint files
        """
        super().__init__(parent)

        self.setWindowTitle("Select Training Checkpoint")
        self.resize(700, 500)

        self.checkpoint_dir = Path(checkpoint_dir or "./checkpoints/training_pipeline")
        self.selected_checkpoint: Optional[Dict[str, Any]] = None

        self.init_ui()
        self.load_checkpoints()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h2>Training Queue Checkpoints</h2>")
        layout.addWidget(title)

        info = QLabel(
            "Select a checkpoint to resume training from a previous session."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Checkpoints table
        checkpoints_group = self.create_checkpoints_table()
        layout.addWidget(checkpoints_group)

        # Details section
        details_group = self.create_details_section()
        layout.addWidget(details_group)

        # Action buttons
        action_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Selected")
        self.load_btn.clicked.connect(self.load_selected)
        self.load_btn.setEnabled(False)
        action_layout.addWidget(self.load_btn)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setEnabled(False)
        action_layout.addWidget(self.delete_btn)

        action_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_checkpoints)
        action_layout.addWidget(self.refresh_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        action_layout.addWidget(self.cancel_btn)

        layout.addLayout(action_layout)

    def create_checkpoints_table(self) -> QGroupBox:
        """Create checkpoints table."""
        group = QGroupBox("Available Checkpoints")
        layout = QVBoxLayout(group)

        self.checkpoints_table = QTableWidget()
        self.checkpoints_table.setColumnCount(5)
        self.checkpoints_table.setHorizontalHeaderLabels([
            'Queue Name', 'Progress', 'Created', 'Size', 'File'
        ])
        self.checkpoints_table.horizontalHeader().setStretchLastSection(True)
        self.checkpoints_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.checkpoints_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.checkpoints_table.itemSelectionChanged.connect(self.on_selection_changed)
        self.checkpoints_table.doubleClicked.connect(self.load_selected)

        layout.addWidget(self.checkpoints_table)

        return group

    def create_details_section(self) -> QGroupBox:
        """Create checkpoint details section."""
        group = QGroupBox("Checkpoint Details")
        layout = QVBoxLayout(group)

        self.details_label = QLabel("No checkpoint selected")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        return group

    def load_checkpoints(self):
        """Load all available checkpoint files."""
        try:
            # Ensure checkpoint directory exists
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Find all checkpoint files
            checkpoint_files = list(self.checkpoint_dir.glob("queue_*_checkpoint_*.json"))

            self.checkpoints_table.setRowCount(len(checkpoint_files))

            checkpoints_data = []

            for row, checkpoint_file in enumerate(sorted(checkpoint_files, key=lambda f: f.stat().st_mtime, reverse=True)):
                try:
                    # Load checkpoint data
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)

                    # Store file path in row
                    file_item = QTableWidgetItem(checkpoint_file.name)
                    file_item.setData(Qt.ItemDataRole.UserRole, str(checkpoint_file))
                    file_item.setData(Qt.ItemDataRole.UserRole + 1, checkpoint_data)

                    # Queue name
                    queue_name = checkpoint_data.get('queue_name', 'Unknown')
                    self.checkpoints_table.setItem(row, 0, QTableWidgetItem(queue_name))

                    # Progress
                    current = checkpoint_data.get('current_index', 0)
                    total = checkpoint_data.get('total_configs', 0)
                    progress_pct = (current / total * 100) if total > 0 else 0
                    progress_text = f"{current}/{total} ({progress_pct:.1f}%)"
                    self.checkpoints_table.setItem(row, 1, QTableWidgetItem(progress_text))

                    # Created date
                    created = checkpoint_data.get('created_at', 'Unknown')
                    if created != 'Unknown':
                        try:
                            dt = datetime.fromisoformat(created)
                            created = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            pass
                    self.checkpoints_table.setItem(row, 2, QTableWidgetItem(created))

                    # File size
                    size_bytes = checkpoint_file.stat().st_size
                    size_kb = size_bytes / 1024
                    size_text = f"{size_kb:.1f} KB"
                    self.checkpoints_table.setItem(row, 3, QTableWidgetItem(size_text))

                    # Filename
                    self.checkpoints_table.setItem(row, 4, file_item)

                    checkpoints_data.append(checkpoint_data)

                except Exception as e:
                    logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
                    # Add error row
                    self.checkpoints_table.setItem(row, 0, QTableWidgetItem("Error"))
                    self.checkpoints_table.setItem(row, 4, QTableWidgetItem(checkpoint_file.name))

            self.checkpoints_table.resizeColumnsToContents()

            if len(checkpoint_files) == 0:
                self.details_label.setText("No checkpoints found in directory:\n" + str(self.checkpoint_dir))
            else:
                logger.info(f"Loaded {len(checkpoint_files)} checkpoints")

        except Exception as e:
            logger.error(f"Failed to load checkpoints: {e}")
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load checkpoints:\n{e}"
            )

    def on_selection_changed(self):
        """Handle selection change."""
        selected_rows = self.checkpoints_table.selectionModel().selectedRows()

        if not selected_rows:
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.details_label.setText("No checkpoint selected")
            return

        self.load_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)

        # Get checkpoint data
        row = selected_rows[0].row()
        file_item = self.checkpoints_table.item(row, 4)
        checkpoint_data = file_item.data(Qt.ItemDataRole.UserRole + 1)

        if checkpoint_data:
            # Build details text
            details_text = (
                f"<b>Queue Name:</b> {checkpoint_data.get('queue_name', 'N/A')}<br>"
                f"<b>Queue ID:</b> {checkpoint_data.get('queue_id', 'N/A')}<br>"
                f"<b>Status:</b> {checkpoint_data.get('status', 'N/A')}<br><br>"
                f"<b>Progress:</b><br>"
                f"  Current Index: {checkpoint_data.get('current_index', 0)}<br>"
                f"  Total Configs: {checkpoint_data.get('total_configs', 0)}<br>"
                f"  Remaining: {checkpoint_data.get('total_configs', 0) - checkpoint_data.get('current_index', 0)}<br><br>"
                f"<b>Results So Far:</b><br>"
                f"  Models Kept: {checkpoint_data.get('models_kept', 0)}<br>"
                f"  Models Deleted: {checkpoint_data.get('models_deleted', 0)}<br><br>"
                f"<b>Checkpoint Info:</b><br>"
                f"  Created: {checkpoint_data.get('created_at', 'N/A')}<br>"
                f"  Version: {checkpoint_data.get('version', 'N/A')}"
            )

            self.details_label.setText(details_text)

    def load_selected(self):
        """Load the selected checkpoint."""
        selected_rows = self.checkpoints_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        row = selected_rows[0].row()
        file_item = self.checkpoints_table.item(row, 4)
        checkpoint_file = file_item.data(Qt.ItemDataRole.UserRole)
        checkpoint_data = file_item.data(Qt.ItemDataRole.UserRole + 1)

        if not checkpoint_data:
            QMessageBox.warning(self, "Error", "Failed to load checkpoint data")
            return

        # Store selected checkpoint
        self.selected_checkpoint = {
            'file_path': checkpoint_file,
            'data': checkpoint_data
        }

        # Accept dialog
        self.accept()

    def delete_selected(self):
        """Delete the selected checkpoint."""
        selected_rows = self.checkpoints_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        row = selected_rows[0].row()
        file_item = self.checkpoints_table.item(row, 4)
        checkpoint_file = Path(file_item.data(Qt.ItemDataRole.UserRole))
        queue_name = self.checkpoints_table.item(row, 0).text()

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete checkpoint:\n\n{queue_name}\n\n"
            f"File: {checkpoint_file.name}\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                checkpoint_file.unlink()

                QMessageBox.information(
                    self,
                    "Success",
                    f"Checkpoint deleted successfully:\n{checkpoint_file.name}"
                )

                # Reload checkpoints
                self.load_checkpoints()

            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")
                QMessageBox.critical(
                    self,
                    "Delete Error",
                    f"Failed to delete checkpoint:\n{e}"
                )

    def get_selected_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the selected checkpoint data.

        Returns:
            Dictionary with 'file_path' and 'data' keys, or None
        """
        return self.selected_checkpoint
