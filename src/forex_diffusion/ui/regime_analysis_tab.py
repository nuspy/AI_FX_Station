"""
Regime Analysis Tab - View and manage regime performance

Shows best models per regime, performance comparisons, and regime definitions.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QSplitter, QHeaderView,
    QAbstractItemView, QMessageBox
)
from PySide6.QtCore import Qt
from typing import Optional, Dict, Any
from loguru import logger

from ..training.training_pipeline.workers import RegimeSummaryWorker


class RegimeAnalysisTab(QWidget):
    """
    Regime Analysis Tab for viewing regime performance.

    Features:
    - Best model per regime table
    - Performance metrics comparison
    - Regime definitions display
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize Regime Analysis Tab."""
        super().__init__(parent)

        self.regime_data: Dict[str, Any] = {}
        self.init_ui()
        self.load_regime_summary()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h2>Regime Analysis</h2>")
        layout.addWidget(title)

        # Refresh button
        refresh_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.clicked.connect(self.load_regime_summary)
        refresh_layout.addWidget(self.refresh_btn)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)

        # Splitter for regime table and details
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Regime Best Models Table
        regime_table_widget = self.create_regime_table()
        splitter.addWidget(regime_table_widget)

        # Right: Regime Details
        details_widget = self.create_details_section()
        splitter.addWidget(details_widget)

        # Set splitter ratios
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    def create_regime_table(self) -> QWidget:
        """Create regime best models table."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("Best Models per Regime")
        group_layout = QVBoxLayout(group)

        self.regime_table = QTableWidget()
        self.regime_table.setColumnCount(6)
        self.regime_table.setHorizontalHeaderLabels([
            'Regime', 'Has Best Model', 'Sharpe Ratio', 'Max DD',
            'Win Rate', 'Achieved At'
        ])
        self.regime_table.horizontalHeader().setStretchLastSection(True)
        self.regime_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.regime_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.regime_table.itemSelectionChanged.connect(self.on_regime_selected)

        group_layout.addWidget(self.regime_table)
        layout.addWidget(group)

        return widget

    def create_details_section(self) -> QWidget:
        """Create regime details section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Regime Definition
        def_group = QGroupBox("Regime Definition")
        def_layout = QVBoxLayout(def_group)

        self.regime_name_label = QLabel("No regime selected")
        self.regime_name_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        def_layout.addWidget(self.regime_name_label)

        self.regime_desc_label = QLabel("")
        self.regime_desc_label.setWordWrap(True)
        def_layout.addWidget(self.regime_desc_label)

        self.regime_rules_label = QLabel("")
        self.regime_rules_label.setWordWrap(True)
        def_layout.addWidget(self.regime_rules_label)

        layout.addWidget(def_group)

        # Best Model Details
        model_group = QGroupBox("Best Model Details")
        model_layout = QVBoxLayout(model_group)

        self.model_details_label = QLabel("No model selected")
        self.model_details_label.setWordWrap(True)
        model_layout.addWidget(self.model_details_label)

        layout.addWidget(model_group)

        # Performance Metrics
        metrics_group = QGroupBox("Secondary Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_label = QLabel("No metrics available")
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)

        layout.addWidget(metrics_group)

        layout.addStretch()

        return widget

    def load_regime_summary(self):
        """Load regime summary data in background."""
        self.refresh_btn.setEnabled(False)
        self.regime_name_label.setText("Loading...")

        worker = RegimeSummaryWorker()
        worker.finished.connect(self.on_regime_data_loaded)
        worker.error.connect(self.on_load_error)
        worker.start()

    def on_regime_data_loaded(self, data: Dict[str, Any]):
        """Handle regime data load completion."""
        self.regime_data = data
        self.refresh_btn.setEnabled(True)

        # Populate table
        regimes = data.get('regimes', {})
        self.regime_table.setRowCount(len(regimes))

        row = 0
        for regime_name, regime_info in regimes.items():
            # Regime name
            self.regime_table.setItem(row, 0, QTableWidgetItem(regime_name))

            # Has best model
            has_model = "✓" if regime_info.get('has_best_model') else "✗"
            has_model_item = QTableWidgetItem(has_model)
            has_model_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.regime_table.setItem(row, 1, has_model_item)

            # Metrics (if available)
            if regime_info.get('best_model'):
                best_model = regime_info['best_model']
                secondary = best_model.get('secondary_metrics', {})

                sharpe = f"{best_model.get('performance_score', 0):.3f}"
                max_dd = f"{secondary.get('max_drawdown', 0):.3f}"
                win_rate = f"{secondary.get('win_rate', 0):.1%}"
                achieved = best_model.get('achieved_at', 'N/A')[:19]  # Trim timestamp

                self.regime_table.setItem(row, 2, QTableWidgetItem(sharpe))
                self.regime_table.setItem(row, 3, QTableWidgetItem(max_dd))
                self.regime_table.setItem(row, 4, QTableWidgetItem(win_rate))
                self.regime_table.setItem(row, 5, QTableWidgetItem(achieved))
            else:
                for col in range(2, 6):
                    self.regime_table.setItem(row, col, QTableWidgetItem("N/A"))

            row += 1

        # Resize columns
        self.regime_table.resizeColumnsToContents()

        logger.info(f"Loaded {len(regimes)} regimes")

    def on_load_error(self, error_msg: str):
        """Handle regime data load error."""
        self.refresh_btn.setEnabled(True)
        self.regime_name_label.setText("Error loading data")

        logger.error(f"Failed to load regime data: {error_msg}")
        QMessageBox.critical(
            self,
            "Error",
            f"Failed to load regime data:\n{error_msg}"
        )

    def on_regime_selected(self):
        """Handle regime selection in table."""
        selected_rows = self.regime_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        row = selected_rows[0].row()
        regime_name = self.regime_table.item(row, 0).text()

        # Get regime data
        regimes = self.regime_data.get('regimes', {})
        regime_info = regimes.get(regime_name, {})

        # Update regime definition
        self.regime_name_label.setText(regime_name.replace('_', ' ').title())
        self.regime_desc_label.setText(
            f"Description: {regime_info.get('description', 'N/A')}"
        )

        # Show detection rules (if available)
        # Note: detection_rules is stored as JSON in the database
        rules_text = "Detection Rules: Complex rules (see database)"
        self.regime_rules_label.setText(rules_text)

        # Update best model details
        if regime_info.get('has_best_model') and regime_info.get('best_model'):
            best_model = regime_info['best_model']

            model_text = (
                f"Training Run ID: {best_model.get('training_run_id', 'N/A')}\n"
                f"Performance Score: {best_model.get('performance_score', 0):.4f}\n"
                f"Achieved: {best_model.get('achieved_at', 'N/A')[:19]}"
            )
            self.model_details_label.setText(model_text)

            # Update secondary metrics
            secondary = best_model.get('secondary_metrics', {})
            metrics_text = ""
            for metric, value in secondary.items():
                if isinstance(value, (int, float)):
                    if 'rate' in metric.lower() or 'pct' in metric.lower():
                        metrics_text += f"{metric}: {value:.2%}\n"
                    else:
                        metrics_text += f"{metric}: {value:.4f}\n"
                else:
                    metrics_text += f"{metric}: {value}\n"

            self.metrics_label.setText(metrics_text if metrics_text else "No metrics available")
        else:
            self.model_details_label.setText("No best model for this regime yet")
            self.metrics_label.setText("Train models to establish best performers")
