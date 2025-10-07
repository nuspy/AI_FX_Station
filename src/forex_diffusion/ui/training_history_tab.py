"""
Training History Tab - Browse and search training history

Provides interface for searching, filtering, and viewing historical training runs.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QSpinBox
)
from PySide6.QtCore import Qt
from typing import Optional, List, Dict, Any
from loguru import logger

from ..training.training_pipeline.workers import TrainingHistoryWorker


class TrainingHistoryTab(QWidget):
    """
    Training History Tab for browsing historical training runs.

    Features:
    - Search and filter by symbol, model type, status
    - Paginated results table
    - Detailed view of individual runs
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize Training History Tab."""
        super().__init__(parent)

        self.current_results: List[Dict[str, Any]] = []
        self.init_ui()
        self.load_history()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h2>Training History</h2>")
        layout.addWidget(title)

        # Search/Filter Section
        filter_group = self.create_filter_section()
        layout.addWidget(filter_group)

        # Results Table
        results_group = self.create_results_section()
        layout.addWidget(results_group)

    def create_filter_section(self) -> QGroupBox:
        """Create search and filter controls."""
        group = QGroupBox("Search & Filter")
        layout = QVBoxLayout(group)

        # First row: Symbol, Model Type, Status
        row1 = QHBoxLayout()

        row1.addWidget(QLabel("Symbol:"))
        self.symbol_filter = QComboBox()
        self.symbol_filter.addItems(['All', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'])
        row1.addWidget(self.symbol_filter)

        row1.addWidget(QLabel("Model Type:"))
        self.model_type_filter = QComboBox()
        self.model_type_filter.addItems([
            'All', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm',
            'linear_regression', 'ridge', 'lasso', 'elasticnet'
        ])
        row1.addWidget(self.model_type_filter)

        row1.addWidget(QLabel("Status:"))
        self.status_filter = QComboBox()
        self.status_filter.addItems(['All', 'completed', 'running', 'failed', 'cancelled'])
        row1.addWidget(self.status_filter)

        row1.addStretch()
        layout.addLayout(row1)

        # Second row: Limit and buttons
        row2 = QHBoxLayout()

        row2.addWidget(QLabel("Limit:"))
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(10, 1000)
        self.limit_spin.setValue(100)
        self.limit_spin.setSingleStep(50)
        row2.addWidget(self.limit_spin)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.load_history)
        row2.addWidget(self.search_btn)

        self.clear_btn = QPushButton("Clear Filters")
        self.clear_btn.clicked.connect(self.clear_filters)
        row2.addWidget(self.clear_btn)

        row2.addStretch()
        layout.addLayout(row2)

        return group

    def create_results_section(self) -> QGroupBox:
        """Create results table."""
        group = QGroupBox("Training Runs")
        layout = QVBoxLayout(group)

        # Summary label
        self.summary_label = QLabel("No results loaded")
        layout.addWidget(self.summary_label)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            'ID', 'UUID', 'Model Type', 'Symbol', 'Timeframe',
            'Status', 'Kept', 'Best Regimes', 'Created At'
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.doubleClicked.connect(self.on_row_double_clicked)

        layout.addWidget(self.results_table)

        # Action buttons
        action_layout = QHBoxLayout()

        self.view_details_btn = QPushButton("View Details")
        self.view_details_btn.clicked.connect(self.view_selected_details)
        self.view_details_btn.setEnabled(False)
        action_layout.addWidget(self.view_details_btn)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setEnabled(False)
        action_layout.addWidget(self.delete_btn)

        self.delete_all_failed_btn = QPushButton("Delete All Failed")
        self.delete_all_failed_btn.clicked.connect(self.delete_all_failed)
        self.delete_all_failed_btn.setEnabled(False)
        action_layout.addWidget(self.delete_all_failed_btn)

        action_layout.addStretch()

        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        action_layout.addWidget(self.export_btn)

        layout.addLayout(action_layout)

        return group

    def clear_filters(self):
        """Clear all filter selections."""
        self.symbol_filter.setCurrentIndex(0)
        self.model_type_filter.setCurrentIndex(0)
        self.status_filter.setCurrentIndex(0)
        self.limit_spin.setValue(100)

    def load_history(self):
        """Load training history with current filters."""
        # Get filter values
        symbol = None if self.symbol_filter.currentText() == 'All' else self.symbol_filter.currentText()
        model_type = None if self.model_type_filter.currentText() == 'All' else self.model_type_filter.currentText()
        status = None if self.status_filter.currentText() == 'All' else self.status_filter.currentText()
        limit = self.limit_spin.value()

        # Disable search button
        self.search_btn.setEnabled(False)
        self.summary_label.setText("Loading...")

        # Create and start worker
        worker = TrainingHistoryWorker(
            symbol=symbol,
            model_type=model_type,
            status=status,
            limit=limit
        )
        worker.finished.connect(self.on_history_loaded)
        worker.error.connect(self.on_load_error)
        worker.start()

    def on_history_loaded(self, results: List[Dict[str, Any]]):
        """Handle history load completion."""
        self.current_results = results
        self.search_btn.setEnabled(True)

        # Update summary
        self.summary_label.setText(f"Found {len(results)} training runs")

        # Populate table
        self.results_table.setRowCount(len(results))

        for row, run in enumerate(results):
            # ID
            self.results_table.setItem(row, 0, QTableWidgetItem(str(run.get('id', 'N/A'))))

            # UUID (shortened)
            uuid_short = run.get('run_uuid', 'N/A')[:8]
            self.results_table.setItem(row, 1, QTableWidgetItem(uuid_short))

            # Model Type
            self.results_table.setItem(row, 2, QTableWidgetItem(run.get('model_type', 'N/A')))

            # Symbol
            self.results_table.setItem(row, 3, QTableWidgetItem(run.get('symbol', 'N/A')))

            # Timeframe
            self.results_table.setItem(row, 4, QTableWidgetItem(run.get('base_timeframe', 'N/A')))

            # Status
            status_text = run.get('status', 'N/A')
            status_item = QTableWidgetItem(status_text)

            # Color code by status
            if status_text == 'completed':
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            elif status_text == 'failed':
                status_item.setForeground(Qt.GlobalColor.red)
            elif status_text == 'running':
                status_item.setForeground(Qt.GlobalColor.blue)

            self.results_table.setItem(row, 5, status_item)

            # Kept
            kept = "✓" if run.get('is_model_kept') else "✗"
            kept_item = QTableWidgetItem(kept)
            kept_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, 6, kept_item)

            # Best Regimes
            regimes = run.get('best_regimes', [])
            regimes_text = ', '.join(regimes) if regimes else 'None'
            self.results_table.setItem(row, 7, QTableWidgetItem(regimes_text))

            # Created At
            created_at = run.get('created_at', 'N/A')
            if created_at != 'N/A':
                created_at = created_at[:19]  # Trim to YYYY-MM-DD HH:MM:SS
            self.results_table.setItem(row, 8, QTableWidgetItem(created_at))

        # Resize columns
        self.results_table.resizeColumnsToContents()

        # Enable actions if we have results
        self.export_btn.setEnabled(len(results) > 0)

        # Check if there are failed runs
        failed_count = sum(1 for r in results if r.get('status') == 'failed')
        self.delete_all_failed_btn.setEnabled(failed_count > 0)

        # Enable view details when row selected
        self.results_table.itemSelectionChanged.connect(self.on_selection_changed)

        logger.info(f"Loaded {len(results)} training runs ({failed_count} failed)")

    def on_load_error(self, error_msg: str):
        """Handle history load error."""
        self.search_btn.setEnabled(True)
        self.summary_label.setText("Error loading history")

        logger.error(f"Failed to load training history: {error_msg}")
        QMessageBox.critical(
            self,
            "Error",
            f"Failed to load training history:\n{error_msg}"
        )

    def on_selection_changed(self):
        """Handle table selection change."""
        has_selection = len(self.results_table.selectionModel().selectedRows()) > 0
        self.view_details_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)

    def on_row_double_clicked(self, index):
        """Handle row double-click."""
        self.view_selected_details()

    def view_selected_details(self):
        """View details of selected training run."""
        selected_rows = self.results_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        row = selected_rows[0].row()

        if row < 0 or row >= len(self.current_results):
            return

        run_data = self.current_results[row]

        # Build details text
        details_text = (
            f"Training Run Details:\n\n"
            f"ID: {run_data.get('id', 'N/A')}\n"
            f"UUID: {run_data.get('run_uuid', 'N/A')}\n"
            f"Status: {run_data.get('status', 'N/A')}\n\n"
            f"Configuration:\n"
            f"  Model Type: {run_data.get('model_type', 'N/A')}\n"
            f"  Symbol: {run_data.get('symbol', 'N/A')}\n"
            f"  Base Timeframe: {run_data.get('base_timeframe', 'N/A')}\n\n"
            f"Performance:\n"
            f"  Model Kept: {'Yes' if run_data.get('is_model_kept') else 'No'}\n"
            f"  Best Regimes: {', '.join(run_data.get('best_regimes', [])) or 'None'}\n\n"
            f"Timestamps:\n"
            f"  Created: {run_data.get('created_at', 'N/A')}\n"
        )

        QMessageBox.information(
            self,
            f"Training Run {run_data.get('id', 'N/A')}",
            details_text
        )

    def export_results(self):
        """Export current results to CSV."""
        if not self.current_results:
            QMessageBox.warning(self, "No Data", "No results to export")
            return

        try:
            import csv
            from datetime import datetime
            from pathlib import Path

            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"training_history_{timestamp}.csv"
            filepath = Path.home() / ".forexgpt" / filename

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write CSV
            with open(filepath, 'w', newline='') as f:
                if self.current_results:
                    writer = csv.DictWriter(f, fieldnames=self.current_results[0].keys())
                    writer.writeheader()
                    writer.writerows(self.current_results)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(self.current_results)} runs to:\n{filepath}"
            )

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results:\n{e}"
            )

    def delete_selected(self):
        """Delete selected training runs."""
        selected_rows = self.results_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        selected_ids = []
        selected_names = []

        for row_index in selected_rows:
            row = row_index.row()
            if row < len(self.current_results):
                run_id = self.current_results[row].get('id')
                model_type = self.current_results[row].get('model_type', 'Unknown')
                symbol = self.current_results[row].get('symbol', 'Unknown')
                selected_ids.append(run_id)
                selected_names.append(f"{model_type} ({symbol})")

        if not selected_ids:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {len(selected_ids)} training run(s)?\n\n"
            f"Runs to delete:\n" + "\n".join(f"- {name}" for name in selected_names[:5]) +
            (f"\n... and {len(selected_names) - 5} more" if len(selected_names) > 5 else "") + "\n\n"
            f"This will delete:\n"
            f"- Training run records\n"
            f"- Associated inference backtests\n"
            f"- Model files (if kept)\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                from ..training.training_pipeline.database import session_scope, delete_training_run

                deleted_count = 0
                with session_scope() as session:
                    for run_id in selected_ids:
                        if delete_training_run(session, run_id):
                            deleted_count += 1

                QMessageBox.information(
                    self,
                    "Success",
                    f"Deleted {deleted_count} training run(s) successfully."
                )

                # Reload history
                self.load_history()

            except Exception as e:
                logger.error(f"Failed to delete training runs: {e}")
                QMessageBox.critical(
                    self,
                    "Delete Error",
                    f"Failed to delete training runs:\n{e}"
                )

    def delete_all_failed(self):
        """Delete all failed training runs."""
        if not self.current_results:
            return

        failed_runs = [r for r in self.current_results if r.get('status') == 'failed']

        if not failed_runs:
            QMessageBox.information(
                self,
                "No Failed Runs",
                "There are no failed training runs to delete."
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete All Failed",
            f"Are you sure you want to delete ALL {len(failed_runs)} failed training runs?\n\n"
            f"This will delete:\n"
            f"- Training run records\n"
            f"- Associated inference backtests\n"
            f"- Model files (if any)\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                from ..training.training_pipeline.database import session_scope, delete_training_run

                deleted_count = 0
                with session_scope() as session:
                    for run in failed_runs:
                        run_id = run.get('id')
                        if delete_training_run(session, run_id):
                            deleted_count += 1

                QMessageBox.information(
                    self,
                    "Success",
                    f"Deleted {deleted_count} failed training run(s) successfully."
                )

                # Reload history
                self.load_history()

            except Exception as e:
                logger.error(f"Failed to delete failed runs: {e}")
                QMessageBox.critical(
                    self,
                    "Delete Error",
                    f"Failed to delete failed training runs:\n{e}"
                )
