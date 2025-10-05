"""
Training Progress Dialog - Real-time Training Monitoring

Provides live visualization of training progress including:
- Loss curves (train/val)
- Metrics dashboard
- GPU utilization
- Estimated time remaining
- Live logs
"""
from __future__ import annotations

import time
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime, timedelta

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QProgressBar, QTextEdit, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QGridLayout, QSplitter
)
from PySide6.QtCore import Qt, Signal, QTimer, Slot
from PySide6.QtGui import QFont
from loguru import logger

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - training progress charts disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrainingProgressDialog(QDialog):
    """
    Real-time training progress monitoring dialog.

    Features:
    - Live loss curves (matplotlib)
    - Metrics table (epoch, train_loss, val_loss, lr, etc.)
    - GPU utilization monitoring
    - ETA calculation
    - Live log streaming
    - Early stopping status
    """

    stop_requested = Signal()  # Signal to request training stop

    def __init__(self, parent=None, total_epochs: int = 100, training_name: str = "Training"):
        super().__init__(parent)

        self.total_epochs = total_epochs
        self.training_name = training_name
        self.start_time = time.time()
        self.current_epoch = 0

        # Data storage for plotting
        self.epochs_history = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.learning_rate_history = []
        self.metrics_history = {}

        # GPU monitoring
        self.gpu_util_history = deque(maxlen=100)
        self.gpu_memory_history = deque(maxlen=100)

        self.setWindowTitle(f"Training Progress: {training_name}")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)
        self.setModal(False)  # Allow interaction with main window

        self._build_ui()
        self._start_monitoring()

    def _build_ui(self):
        """Build the dialog UI"""
        layout = QVBoxLayout(self)

        # Header with overall progress
        header = self._build_header()
        layout.addWidget(header)

        # Main content area with tabs
        tabs = QTabWidget()

        # Tab 1: Loss Curves
        if MATPLOTLIB_AVAILABLE:
            loss_tab = self._build_loss_curves_tab()
            tabs.addTab(loss_tab, "Loss Curves")

        # Tab 2: Metrics Table
        metrics_tab = self._build_metrics_table_tab()
        tabs.addTab(metrics_tab, "Metrics")

        # Tab 3: GPU Monitor
        if TORCH_AVAILABLE:
            gpu_tab = self._build_gpu_monitor_tab()
            tabs.addTab(gpu_tab, "GPU Monitor")

        # Tab 4: Live Logs
        logs_tab = self._build_logs_tab()
        tabs.addTab(logs_tab, "Logs")

        layout.addWidget(tabs)

        # Footer with controls
        footer = self._build_footer()
        layout.addWidget(footer)

    def _build_header(self) -> QWidget:
        """Build header with overall progress"""
        header = QGroupBox("Overall Progress")
        layout = QGridLayout(header)

        # Training name
        layout.addWidget(QLabel("Training:"), 0, 0)
        self.name_label = QLabel(self.training_name)
        self.name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.name_label, 0, 1, 1, 3)

        # Progress bar
        layout.addWidget(QLabel("Progress:"), 1, 0)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.total_epochs)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m epochs (%p%)")
        layout.addWidget(self.progress_bar, 1, 1, 1, 3)

        # Status indicators
        layout.addWidget(QLabel("Current Epoch:"), 2, 0)
        self.epoch_label = QLabel("0 / " + str(self.total_epochs))
        layout.addWidget(self.epoch_label, 2, 1)

        layout.addWidget(QLabel("Elapsed:"), 2, 2)
        self.elapsed_label = QLabel("00:00:00")
        layout.addWidget(self.elapsed_label, 2, 3)

        layout.addWidget(QLabel("ETA:"), 3, 0)
        self.eta_label = QLabel("Calculating...")
        layout.addWidget(self.eta_label, 3, 1)

        layout.addWidget(QLabel("Speed:"), 3, 2)
        self.speed_label = QLabel("-- epoch/s")
        layout.addWidget(self.speed_label, 3, 3)

        # Current metrics
        layout.addWidget(QLabel("Train Loss:"), 4, 0)
        self.train_loss_label = QLabel("--")
        self.train_loss_label.setStyleSheet("color: #3498db; font-weight: bold;")
        layout.addWidget(self.train_loss_label, 4, 1)

        layout.addWidget(QLabel("Val Loss:"), 4, 2)
        self.val_loss_label = QLabel("--")
        self.val_loss_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        layout.addWidget(self.val_loss_label, 4, 3)

        return header

    def _build_loss_curves_tab(self) -> QWidget:
        """Build loss curves visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)

        # Create subplots
        self.ax_loss = self.figure.add_subplot(2, 1, 1)
        self.ax_lr = self.figure.add_subplot(2, 1, 2)

        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Training and Validation Loss')
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend(['Train Loss', 'Val Loss'])

        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_lr.set_title('Learning Rate Schedule')
        self.ax_lr.grid(True, alpha=0.3)

        self.figure.tight_layout()

        layout.addWidget(self.canvas)

        # Controls
        controls = QHBoxLayout()

        self.auto_scale = QPushButton("Auto Scale")
        self.auto_scale.clicked.connect(self._auto_scale_plots)

        self.log_scale = QPushButton("Log Scale")
        self.log_scale.setCheckable(True)
        self.log_scale.clicked.connect(self._toggle_log_scale)

        self.clear_plot = QPushButton("Clear")
        self.clear_plot.clicked.connect(self._clear_plots)

        controls.addWidget(self.auto_scale)
        controls.addWidget(self.log_scale)
        controls.addWidget(self.clear_plot)
        controls.addStretch()

        layout.addLayout(controls)

        return widget

    def _build_metrics_table_tab(self) -> QWidget:
        """Build metrics table tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(7)
        self.metrics_table.setHorizontalHeaderLabels([
            'Epoch', 'Train Loss', 'Val Loss', 'Learning Rate',
            'Duration', 'Best Val', 'Status'
        ])
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectRows)

        # Set column widths
        self.metrics_table.setColumnWidth(0, 60)
        self.metrics_table.setColumnWidth(1, 120)
        self.metrics_table.setColumnWidth(2, 120)
        self.metrics_table.setColumnWidth(3, 100)
        self.metrics_table.setColumnWidth(4, 100)
        self.metrics_table.setColumnWidth(5, 120)
        self.metrics_table.setColumnWidth(6, 150)

        layout.addWidget(self.metrics_table)

        # Summary statistics
        summary = QGroupBox("Summary Statistics")
        summary_layout = QGridLayout(summary)

        summary_layout.addWidget(QLabel("Best Epoch:"), 0, 0)
        self.best_epoch_label = QLabel("--")
        summary_layout.addWidget(self.best_epoch_label, 0, 1)

        summary_layout.addWidget(QLabel("Best Val Loss:"), 0, 2)
        self.best_val_loss_label = QLabel("--")
        summary_layout.addWidget(self.best_val_loss_label, 0, 3)

        summary_layout.addWidget(QLabel("Final Train Loss:"), 1, 0)
        self.final_train_loss_label = QLabel("--")
        summary_layout.addWidget(self.final_train_loss_label, 1, 1)

        summary_layout.addWidget(QLabel("Improvement:"), 1, 2)
        self.improvement_label = QLabel("--")
        summary_layout.addWidget(self.improvement_label, 1, 3)

        layout.addWidget(summary)

        return widget

    def _build_gpu_monitor_tab(self) -> QWidget:
        """Build GPU monitoring tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # GPU info display
        gpu_info_group = QGroupBox("GPU Status")
        gpu_info_layout = QGridLayout(gpu_info_group)

        gpu_info_layout.addWidget(QLabel("GPU Utilization:"), 0, 0)
        self.gpu_util_label = QLabel("--")
        gpu_info_layout.addWidget(self.gpu_util_label, 0, 1)

        gpu_info_layout.addWidget(QLabel("GPU Memory:"), 0, 2)
        self.gpu_memory_label = QLabel("--")
        gpu_info_layout.addWidget(self.gpu_memory_label, 0, 3)

        gpu_info_layout.addWidget(QLabel("GPU Temperature:"), 1, 0)
        self.gpu_temp_label = QLabel("--")
        gpu_info_layout.addWidget(self.gpu_temp_label, 1, 1)

        gpu_info_layout.addWidget(QLabel("GPU Power:"), 1, 2)
        self.gpu_power_label = QLabel("--")
        gpu_info_layout.addWidget(self.gpu_power_label, 1, 3)

        layout.addWidget(gpu_info_group)

        # GPU utilization plot
        if MATPLOTLIB_AVAILABLE:
            self.gpu_figure = Figure(figsize=(8, 4))
            self.gpu_canvas = FigureCanvas(self.gpu_figure)
            self.ax_gpu = self.gpu_figure.add_subplot(1, 1, 1)
            self.ax_gpu.set_xlabel('Time (seconds)')
            self.ax_gpu.set_ylabel('Utilization (%)')
            self.ax_gpu.set_title('GPU Utilization Over Time')
            self.ax_gpu.grid(True, alpha=0.3)
            self.ax_gpu.set_ylim(0, 100)
            layout.addWidget(self.gpu_canvas)

        layout.addStretch()

        return widget

    def _build_logs_tab(self) -> QWidget:
        """Build live logs tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)

        layout.addWidget(self.log_text)

        # Log controls
        controls = QHBoxLayout()

        self.auto_scroll = QPushButton("Auto-scroll: ON")
        self.auto_scroll.setCheckable(True)
        self.auto_scroll.setChecked(True)
        self.auto_scroll.clicked.connect(self._toggle_auto_scroll)

        self.clear_logs = QPushButton("Clear Logs")
        self.clear_logs.clicked.connect(lambda: self.log_text.clear())

        controls.addWidget(self.auto_scroll)
        controls.addWidget(self.clear_logs)
        controls.addStretch()

        layout.addLayout(controls)

        return widget

    def _build_footer(self) -> QWidget:
        """Build footer with control buttons"""
        footer = QWidget()
        layout = QHBoxLayout(footer)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)  # Not implemented yet
        self.pause_btn.setToolTip("Pause/Resume training (not implemented)")

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        self.stop_btn.clicked.connect(self._request_stop)

        self.export_btn = QPushButton("Export Metrics")
        self.export_btn.clicked.connect(self._export_metrics)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setEnabled(False)  # Enable when training finishes

        layout.addWidget(self.pause_btn)
        layout.addStretch()
        layout.addWidget(self.export_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.close_btn)

        return footer

    def _start_monitoring(self):
        """Start monitoring timers"""
        # Update timer for elapsed time and ETA
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_time_displays)
        self.update_timer.start(1000)  # Update every second

        # GPU monitoring timer
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_timer = QTimer()
            self.gpu_timer.timeout.connect(self._update_gpu_stats)
            self.gpu_timer.start(2000)  # Update every 2 seconds

    def _update_time_displays(self):
        """Update elapsed time and ETA displays"""
        elapsed = time.time() - self.start_time
        self.elapsed_label.setText(str(timedelta(seconds=int(elapsed))))

        if self.current_epoch > 0:
            # Calculate ETA
            avg_epoch_time = elapsed / self.current_epoch
            remaining_epochs = self.total_epochs - self.current_epoch
            eta_seconds = avg_epoch_time * remaining_epochs

            self.eta_label.setText(str(timedelta(seconds=int(eta_seconds))))

            # Calculate speed
            epochs_per_sec = self.current_epoch / elapsed
            if epochs_per_sec >= 1:
                self.speed_label.setText(f"{epochs_per_sec:.2f} epoch/s")
            else:
                sec_per_epoch = 1.0 / epochs_per_sec
                self.speed_label.setText(f"{sec_per_epoch:.1f} s/epoch")

    def _update_gpu_stats(self):
        """Update GPU statistics"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return

        try:
            # Get GPU stats
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB

            self.gpu_memory_label.setText(f"{allocated:.1f} / {total:.1f} GB ({allocated/total*100:.1f}%)")

            # Try to get utilization from nvidia-smi (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Watts

                self.gpu_util_label.setText(f"{util.gpu}%")
                self.gpu_temp_label.setText(f"{temp}°C")
                self.gpu_power_label.setText(f"{power:.1f} W")

                # Store for plotting
                self.gpu_util_history.append(util.gpu)
                self.gpu_memory_history.append(allocated / total * 100)

                # Update GPU plot
                if MATPLOTLIB_AVAILABLE and hasattr(self, 'ax_gpu'):
                    self.ax_gpu.clear()
                    self.ax_gpu.plot(list(self.gpu_util_history), label='GPU Util %')
                    self.ax_gpu.plot(list(self.gpu_memory_history), label='Memory %')
                    self.ax_gpu.set_ylim(0, 100)
                    self.ax_gpu.legend()
                    self.ax_gpu.grid(True, alpha=0.3)
                    self.gpu_canvas.draw()

            except ImportError:
                self.gpu_util_label.setText("pynvml not available")

        except Exception as e:
            logger.debug(f"GPU stats update error: {e}")

    @Slot(int, dict)
    def update_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """Update display with new epoch results"""
        self.current_epoch = epoch
        self.progress_bar.setValue(epoch)
        self.epoch_label.setText(f"{epoch} / {self.total_epochs}")

        # Extract metrics
        train_loss = metrics.get('train_loss', metrics.get('train/loss', None))
        val_loss = metrics.get('val_loss', metrics.get('val/loss', None))
        lr = metrics.get('learning_rate', metrics.get('lr', None))

        # Update current loss displays
        if train_loss is not None:
            self.train_loss_label.setText(f"{train_loss:.6f}")
            self.train_loss_history.append(train_loss)

        if val_loss is not None:
            self.val_loss_label.setText(f"{val_loss:.6f}")
            self.val_loss_history.append(val_loss)

        if lr is not None:
            self.learning_rate_history.append(lr)

        self.epochs_history.append(epoch)

        # Update plots
        if MATPLOTLIB_AVAILABLE:
            self._update_loss_plots()

        # Update metrics table
        self._add_metrics_row(epoch, metrics)

        # Update summary
        self._update_summary()

    def _update_loss_plots(self):
        """Update loss curve plots"""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'ax_loss'):
            return

        # Clear and redraw loss plot
        self.ax_loss.clear()

        if self.train_loss_history:
            self.ax_loss.plot(self.epochs_history, self.train_loss_history,
                            'b-', linewidth=2, label='Train Loss', alpha=0.8)

        if self.val_loss_history:
            self.ax_loss.plot(self.epochs_history, self.val_loss_history,
                            'r-', linewidth=2, label='Val Loss', alpha=0.8)

            # Mark best epoch
            if self.val_loss_history:
                best_idx = self.val_loss_history.index(min(self.val_loss_history))
                best_epoch = self.epochs_history[best_idx]
                best_val = self.val_loss_history[best_idx]
                self.ax_loss.plot(best_epoch, best_val, 'g*', markersize=15,
                                label=f'Best (epoch {best_epoch})')

        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Training and Validation Loss')
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend()

        # Update learning rate plot
        self.ax_lr.clear()
        if self.learning_rate_history:
            self.ax_lr.plot(self.epochs_history, self.learning_rate_history,
                          'g-', linewidth=2, alpha=0.8)
            self.ax_lr.set_xlabel('Epoch')
            self.ax_lr.set_ylabel('Learning Rate')
            self.ax_lr.set_title('Learning Rate Schedule')
            self.ax_lr.grid(True, alpha=0.3)

        self.canvas.draw()

    def _add_metrics_row(self, epoch: int, metrics: Dict[str, Any]):
        """Add a row to the metrics table"""
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)

        train_loss = metrics.get('train_loss', metrics.get('train/loss', '--'))
        val_loss = metrics.get('val_loss', metrics.get('val/loss', '--'))
        lr = metrics.get('learning_rate', metrics.get('lr', '--'))
        duration = metrics.get('epoch_duration', '--')

        # Format values
        train_str = f"{train_loss:.6f}" if isinstance(train_loss, (int, float)) else str(train_loss)
        val_str = f"{val_loss:.6f}" if isinstance(val_loss, (int, float)) else str(val_loss)
        lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else str(lr)

        best_val = min(self.val_loss_history) if self.val_loss_history else None
        best_str = f"{best_val:.6f}" if best_val is not None else "--"

        status = "✓" if val_loss == best_val else ""

        self.metrics_table.setItem(row, 0, QTableWidgetItem(str(epoch)))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(train_str))
        self.metrics_table.setItem(row, 2, QTableWidgetItem(val_str))
        self.metrics_table.setItem(row, 3, QTableWidgetItem(lr_str))
        self.metrics_table.setItem(row, 4, QTableWidgetItem(str(duration)))
        self.metrics_table.setItem(row, 5, QTableWidgetItem(best_str))
        self.metrics_table.setItem(row, 6, QTableWidgetItem(status))

        # Scroll to bottom
        self.metrics_table.scrollToBottom()

    def _update_summary(self):
        """Update summary statistics"""
        if not self.val_loss_history:
            return

        best_val = min(self.val_loss_history)
        best_epoch = self.epochs_history[self.val_loss_history.index(best_val)]

        self.best_val_loss_label.setText(f"{best_val:.6f}")
        self.best_epoch_label.setText(str(best_epoch))

        if self.train_loss_history:
            final_train = self.train_loss_history[-1]
            self.final_train_loss_label.setText(f"{final_train:.6f}")

            if len(self.train_loss_history) > 1:
                initial_train = self.train_loss_history[0]
                improvement = ((initial_train - final_train) / initial_train) * 100
                self.improvement_label.setText(f"{improvement:.1f}%")

    @Slot(str)
    def append_log(self, message: str):
        """Append message to log view"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

        if self.auto_scroll.isChecked():
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )

    def _toggle_auto_scroll(self, checked: bool):
        """Toggle auto-scroll for logs"""
        self.auto_scroll.setText(f"Auto-scroll: {'ON' if checked else 'OFF'}")

    def _auto_scale_plots(self):
        """Auto-scale plot axes"""
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'ax_loss'):
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
            self.ax_lr.relim()
            self.ax_lr.autoscale_view()
            self.canvas.draw()

    def _toggle_log_scale(self, checked: bool):
        """Toggle log scale for loss plot"""
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'ax_loss'):
            if checked:
                self.ax_loss.set_yscale('log')
            else:
                self.ax_loss.set_yscale('linear')
            self.canvas.draw()

    def _clear_plots(self):
        """Clear all plot data"""
        self.epochs_history.clear()
        self.train_loss_history.clear()
        self.val_loss_history.clear()
        self.learning_rate_history.clear()

        if MATPLOTLIB_AVAILABLE:
            self._update_loss_plots()

    def _request_stop(self):
        """Request training stop"""
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Stop Training?",
            "Are you sure you want to stop training?\n\nProgress will be saved at the last checkpoint.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping...")
            self.stop_requested.emit()

    def _export_metrics(self):
        """Export metrics to file"""
        from PySide6.QtWidgets import QFileDialog
        import json

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metrics",
            f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )

        if not filename:
            return

        try:
            data = {
                'training_name': self.training_name,
                'total_epochs': self.total_epochs,
                'current_epoch': self.current_epoch,
                'elapsed_time': time.time() - self.start_time,
                'epochs': self.epochs_history,
                'train_loss': self.train_loss_history,
                'val_loss': self.val_loss_history,
                'learning_rate': self.learning_rate_history,
            }

            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            elif filename.endswith('.csv'):
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate'])
                    for i in range(len(self.epochs_history)):
                        writer.writerow([
                            self.epochs_history[i],
                            self.train_loss_history[i] if i < len(self.train_loss_history) else '',
                            self.val_loss_history[i] if i < len(self.val_loss_history) else '',
                            self.learning_rate_history[i] if i < len(self.learning_rate_history) else '',
                        ])

            logger.info(f"Metrics exported to {filename}")

        except Exception as e:
            logger.exception(f"Failed to export metrics: {e}")

    def training_finished(self, success: bool = True):
        """Called when training finishes"""
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)

        if success:
            self.progress_bar.setFormat("Training Complete! (%p%)")
            self.append_log("✅ Training completed successfully")
        else:
            self.progress_bar.setFormat("Training Stopped (%p%)")
            self.append_log("⚠️ Training stopped by user")

        # Stop timers
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        if hasattr(self, 'gpu_timer'):
            self.gpu_timer.stop()

    def closeEvent(self, event):
        """Handle dialog close"""
        # Stop timers
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        if hasattr(self, 'gpu_timer'):
            self.gpu_timer.stop()

        super().closeEvent(event)
