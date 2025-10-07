"""
Training Queue Tab - Grid Training Management UI

Provides interface for creating and managing training queues with
configuration grids, progress monitoring, and results viewing.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QProgressBar, QTextEdit, QSplitter, QListWidget, QMessageBox,
    QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal
from typing import Optional, Dict, Any, List
from loguru import logger

from ..training.training_pipeline.workers import (
    TrainingWorker, QueueCreationWorker, QueueStatusWorker
)
from ..training.training_pipeline import TrainingOrchestrator


class TrainingQueueTab(QWidget):
    """
    Training Queue Tab for grid-based model training.

    Features:
    - Configuration grid builder with multi-select controls
    - Queue creation and management
    - Real-time progress monitoring
    - Results table with metrics
    """

    # Signals
    training_started = Signal(int)  # queue_id
    training_completed = Signal(dict)  # results
    training_error = Signal(str)  # error message

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize Training Queue Tab."""
        super().__init__(parent)

        self.orchestrator = TrainingOrchestrator()
        self.current_worker: Optional[TrainingWorker] = None
        self.status_worker: Optional[QueueStatusWorker] = None
        self.current_queue_id: Optional[int] = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create splitter for top (config) and bottom (progress/results)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: Configuration Builder
        config_widget = self.create_config_builder()
        splitter.addWidget(config_widget)

        # Bottom: Progress and Results
        progress_widget = self.create_progress_section()
        splitter.addWidget(progress_widget)

        # Set splitter ratios
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)

    def create_config_builder(self) -> QWidget:
        """Create configuration grid builder section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("<h2>Training Queue Configuration</h2>")
        layout.addWidget(title)

        # Configuration Grid
        grid_group = QGroupBox("Parameter Grid")
        grid_layout = QVBoxLayout(grid_group)

        # Model Type (multi-select list)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Types:"))
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.model_list.addItems([
            'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm',
            'linear_regression', 'ridge', 'lasso', 'elasticnet'
        ])
        self.model_list.setMaximumHeight(120)
        model_layout.addWidget(self.model_list)
        grid_layout.addLayout(model_layout)

        # Symbol (multi-select list)
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("Symbols:"))
        self.symbol_list = QListWidget()
        self.symbol_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.symbol_list.addItems(['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'])
        self.symbol_list.setMaximumHeight(100)
        symbol_layout.addWidget(self.symbol_list)
        grid_layout.addLayout(symbol_layout)

        # Encoder
        encoder_layout = QHBoxLayout()
        encoder_layout.addWidget(QLabel("Encoder:"))
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(['none', 'vae', 'autoencoder'])
        encoder_layout.addWidget(self.encoder_combo)
        encoder_layout.addStretch()
        grid_layout.addLayout(encoder_layout)

        # Timeframe
        timeframe_layout = QHBoxLayout()
        timeframe_layout.addWidget(QLabel("Base Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'])
        self.timeframe_combo.setCurrentText('H1')
        timeframe_layout.addWidget(self.timeframe_combo)
        timeframe_layout.addStretch()
        grid_layout.addLayout(timeframe_layout)

        # Days History (multi-select via checkboxes)
        days_layout = QHBoxLayout()
        days_layout.addWidget(QLabel("Days History:"))
        self.days_checks = []
        for days in [30, 60, 90, 180]:
            check = QCheckBox(str(days))
            self.days_checks.append(check)
            days_layout.addWidget(check)
        days_layout.addStretch()
        grid_layout.addLayout(days_layout)

        # Horizon
        horizon_layout = QHBoxLayout()
        horizon_layout.addWidget(QLabel("Horizon (bars):"))
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 100)
        self.horizon_spin.setValue(24)
        horizon_layout.addWidget(self.horizon_spin)
        horizon_layout.addStretch()
        grid_layout.addLayout(horizon_layout)

        # Options
        options_layout = QHBoxLayout()
        self.skip_existing_check = QCheckBox("Skip already-trained configurations")
        self.skip_existing_check.setChecked(True)
        options_layout.addWidget(self.skip_existing_check)
        options_layout.addStretch()
        grid_layout.addLayout(options_layout)

        # Config summary
        self.config_summary_label = QLabel("Configurations: 0")
        self.config_summary_label.setStyleSheet("font-weight: bold; color: blue;")
        grid_layout.addWidget(self.config_summary_label)

        layout.addWidget(grid_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.calculate_btn = QPushButton("Calculate Grid")
        self.calculate_btn.clicked.connect(self.calculate_grid_size)
        button_layout.addWidget(self.calculate_btn)

        self.create_queue_btn = QPushButton("Create Queue")
        self.create_queue_btn.clicked.connect(self.create_training_queue)
        self.create_queue_btn.setEnabled(False)
        button_layout.addWidget(self.create_queue_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        return widget

    def create_progress_section(self) -> QWidget:
        """Create progress monitoring and results section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Queue Control
        control_group = QGroupBox("Queue Control")
        control_layout = QHBoxLayout(control_group)

        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_training)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_training)
        self.cancel_btn.setEnabled(False)
        control_layout.addWidget(self.cancel_btn)

        control_layout.addStretch()

        self.queue_status_label = QLabel("Status: No queue")
        control_layout.addWidget(self.queue_status_label)

        layout.addWidget(control_group)

        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%v / %m (%p%)")
        progress_layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        progress_layout.addWidget(self.status_text)

        layout.addWidget(progress_group)

        # Results Table
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            'Model', 'Symbol', 'Status', 'Kept', 'Best Regimes',
            'Sharpe', 'Duration (s)', 'Timestamp'
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        results_layout.addWidget(self.results_table)

        # Results summary
        self.results_summary_label = QLabel("No results yet")
        results_layout.addWidget(self.results_summary_label)

        layout.addWidget(results_group)

        return widget

    def calculate_grid_size(self):
        """Calculate and display grid size."""
        grid_params = self.get_grid_params()

        if not grid_params:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Please select at least one option for each parameter."
            )
            return

        # Calculate size
        from ..training.training_pipeline.config_grid import generate_config_grid

        try:
            configs = generate_config_grid(grid_params)
            count = len(configs)

            self.config_summary_label.setText(
                f"Configurations: {count} models to train"
            )
            self.create_queue_btn.setEnabled(count > 0)

            # Show breakdown
            breakdown = (
                f"Grid: {len(grid_params['model_type'])} models × "
                f"{len(grid_params['symbol'])} symbols × "
                f"{len(grid_params['days_history'])} history lengths = "
                f"{count} total"
            )

            self.log_status(breakdown)

        except Exception as e:
            logger.error(f"Failed to calculate grid: {e}")
            QMessageBox.critical(self, "Error", f"Failed to calculate grid: {e}")

    def get_grid_params(self) -> Optional[Dict[str, List[Any]]]:
        """Get current grid parameters from UI."""
        # Model types (selected items from list)
        model_types = [
            item.text() for item in self.model_list.selectedItems()
        ]

        # Symbols (selected items from list)
        symbols = [
            item.text() for item in self.symbol_list.selectedItems()
        ]

        # Days history (checked boxes)
        days_history = [
            int(check.text()) for check in self.days_checks if check.isChecked()
        ]

        # Validate
        if not model_types or not symbols or not days_history:
            return None

        return {
            'model_type': model_types,
            'symbol': symbols,
            'encoder': [self.encoder_combo.currentText()],
            'base_timeframe': [self.timeframe_combo.currentText()],
            'days_history': days_history,
            'horizon': [self.horizon_spin.value()]
        }

    def create_training_queue(self):
        """Create training queue in background."""
        grid_params = self.get_grid_params()

        if not grid_params:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Please configure the grid parameters first."
            )
            return

        self.log_status("Creating training queue...")
        self.create_queue_btn.setEnabled(False)

        # Create worker
        worker = QueueCreationWorker(
            grid_params=grid_params,
            skip_existing=self.skip_existing_check.isChecked()
        )
        worker.progress.connect(self.log_status)
        worker.finished.connect(self.on_queue_created)
        worker.error.connect(self.on_queue_error)
        worker.start()

    def on_queue_created(self, queue_id: int):
        """Handle queue creation completion."""
        self.current_queue_id = queue_id
        self.log_status(f"✅ Queue {queue_id} created successfully")

        self.queue_status_label.setText(f"Queue ID: {queue_id} (Ready)")
        self.start_btn.setEnabled(True)
        self.create_queue_btn.setEnabled(True)

        QMessageBox.information(
            self,
            "Queue Created",
            f"Training queue {queue_id} created successfully.\n"
            f"Click 'Start Training' to begin."
        )

    def on_queue_error(self, error_msg: str):
        """Handle queue creation error."""
        self.log_status(f"❌ Error: {error_msg}")
        self.create_queue_btn.setEnabled(True)

        QMessageBox.critical(self, "Error", f"Failed to create queue:\n{error_msg}")

    def start_training(self):
        """Start training the current queue."""
        if not self.current_queue_id:
            QMessageBox.warning(self, "No Queue", "Please create a queue first.")
            return

        self.log_status(f"Starting training queue {self.current_queue_id}...")

        # Disable controls
        self.start_btn.setEnabled(False)
        self.create_queue_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)

        # Create and start training worker
        self.current_worker = TrainingWorker(
            queue_id=self.current_queue_id,
            orchestrator=self.orchestrator
        )
        self.current_worker.progress.connect(self.on_training_progress)
        self.current_worker.finished.connect(self.on_training_finished)
        self.current_worker.error.connect(self.on_training_error)
        self.current_worker.cancelled.connect(self.on_training_cancelled)
        self.current_worker.start()

        # Start status monitoring
        self.status_worker = QueueStatusWorker(
            queue_id=self.current_queue_id,
            orchestrator=self.orchestrator
        )
        self.status_worker.status_updated.connect(self.on_status_update)
        self.status_worker.start()

        self.queue_status_label.setText(f"Queue ID: {self.current_queue_id} (Running)")
        self.training_started.emit(self.current_queue_id)

    def on_training_progress(self, current: int, total: int, status: str):
        """Handle training progress update."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.log_status(f"[{current}/{total}] {status}")

    def on_status_update(self, status: Dict[str, Any]):
        """Handle queue status update."""
        progress = status.get('progress_pct', 0)
        completed = status.get('completed_count', 0)
        failed = status.get('failed_count', 0)

        summary = (
            f"Progress: {progress:.1f}% | "
            f"Completed: {completed} | "
            f"Failed: {failed}"
        )
        self.results_summary_label.setText(summary)

    def on_training_finished(self, results: Dict[str, Any]):
        """Handle training completion."""
        self.log_status("✅ Training completed!")

        # Stop status worker
        if self.status_worker:
            self.status_worker.stop()
            self.status_worker = None

        # Re-enable controls
        self.start_btn.setEnabled(False)
        self.create_queue_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

        self.queue_status_label.setText(
            f"Queue ID: {self.current_queue_id} (Completed)"
        )

        # Show summary
        summary = (
            f"Training Complete:\n"
            f"  Completed: {results['completed']}\n"
            f"  Failed: {results['failed']}\n"
            f"  Kept Models: {results['kept_models']}\n"
            f"  Deleted Models: {results['deleted_models']}\n"
            f"  Regime Improvements: {results['regime_improvements']}"
        )
        self.log_status(summary)

        QMessageBox.information(self, "Training Complete", summary)
        self.training_completed.emit(results)

    def on_training_error(self, error_msg: str):
        """Handle training error."""
        self.log_status(f"❌ Training error: {error_msg}")

        # Stop status worker
        if self.status_worker:
            self.status_worker.stop()
            self.status_worker = None

        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

        QMessageBox.critical(self, "Training Error", f"Training failed:\n{error_msg}")
        self.training_error.emit(error_msg)

    def on_training_cancelled(self):
        """Handle training cancellation."""
        self.log_status("⚠️ Training cancelled by user")

        # Stop status worker
        if self.status_worker:
            self.status_worker.stop()
            self.status_worker = None

        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

        self.queue_status_label.setText(
            f"Queue ID: {self.current_queue_id} (Cancelled)"
        )

    def pause_training(self):
        """Pause training."""
        if self.current_worker and self.current_queue_id:
            self.log_status("Pausing training...")
            self.orchestrator.pause_training(self.current_queue_id)

    def cancel_training(self):
        """Cancel training."""
        if self.current_worker:
            reply = QMessageBox.question(
                self,
                "Cancel Training",
                "Are you sure you want to cancel training?\n"
                "Progress will be saved and can be resumed later.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.log_status("Cancelling training...")
                self.current_worker.cancel()

    def log_status(self, message: str):
        """Add status message to log."""
        self.status_text.append(message)
        # Auto-scroll to bottom
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )

    def load_results(self, queue_id: int):
        """Load results for a queue."""
        # TODO: Implement results loading from database
        pass
