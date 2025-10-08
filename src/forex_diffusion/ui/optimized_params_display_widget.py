"""
Optimized Parameters Display Widget

Widget for displaying optimized parameters after training/backtesting completion.
Shows form_params, action_params, and performance metrics with option to save to database.

FASE 9 - Part 2
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QBrush
from loguru import logger


class OptimizedParamsDisplayWidget(QWidget):
    """
    Widget for displaying optimized parameters after training completion.

    Features:
    - Display form parameters (pattern detection params)
    - Display action parameters (SL/TP/position sizing params)
    - Display performance metrics (Sharpe, Win Rate, etc.)
    - Save to database button
    - Clear display when new training starts
    """

    save_requested = Signal(dict)  # Emitted when user wants to save params

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._current_params = None

        self._setup_ui()

        logger.info("OptimizedParamsDisplayWidget initialized")

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main group box
        main_group = QGroupBox("Optimized Parameters (Last Training)")
        main_layout = QVBoxLayout(main_group)

        # Info label
        self.info_label = QLabel("No optimized parameters available. Run training/backtest to generate.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #888; font-style: italic;")
        main_layout.addWidget(self.info_label)

        # Training metadata section
        self.metadata_frame = QFrame()
        self.metadata_frame.setVisible(False)
        metadata_layout = QVBoxLayout(self.metadata_frame)
        metadata_layout.setContentsMargins(0, 0, 0, 10)

        self.metadata_label = QLabel()
        self.metadata_label.setWordWrap(True)
        font = QFont()
        font.setBold(True)
        self.metadata_label.setFont(font)
        metadata_layout.addWidget(self.metadata_label)

        main_layout.addWidget(self.metadata_frame)

        # Form Parameters Table
        form_group = QGroupBox("Form Parameters (Pattern Detection)")
        form_layout = QVBoxLayout(form_group)

        self.form_table = QTableWidget()
        self.form_table.setColumnCount(2)
        self.form_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.form_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.form_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.form_table.setAlternatingRowColors(True)

        header = self.form_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)

        self.form_table.setVisible(False)
        form_layout.addWidget(self.form_table)

        main_layout.addWidget(form_group)

        # Action Parameters Table
        action_group = QGroupBox("Action Parameters (SL/TP/Position Sizing)")
        action_layout = QVBoxLayout(action_group)

        self.action_table = QTableWidget()
        self.action_table.setColumnCount(2)
        self.action_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.action_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.action_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.action_table.setAlternatingRowColors(True)

        header = self.action_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)

        self.action_table.setVisible(False)
        action_layout.addWidget(self.action_table)

        main_layout.addWidget(action_group)

        # Performance Metrics Table
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setAlternatingRowColors(True)

        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)

        self.metrics_table.setVisible(False)
        metrics_layout.addWidget(self.metrics_table)

        main_layout.addWidget(metrics_group)

        # Action buttons
        actions_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save to Database")
        self.save_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.setVisible(False)
        actions_layout.addWidget(self.save_btn)

        self.clear_btn = QPushButton("Clear Display")
        self.clear_btn.clicked.connect(self.clear)
        self.clear_btn.setVisible(False)
        actions_layout.addWidget(self.clear_btn)

        actions_layout.addStretch()

        main_layout.addLayout(actions_layout)

        layout.addWidget(main_group)

    def set_optimized_params(self, params: Dict[str, Any]):
        """
        Set and display optimized parameters.

        Expected params structure:
        {
            'pattern_type': str,  # e.g., 'head_shoulders', 'double_top'
            'symbol': str,
            'timeframe': str,
            'form_params': dict,  # Pattern detection parameters
            'action_params': dict,  # SL/TP/position sizing parameters
            'performance_metrics': dict,  # Sharpe, win rate, etc.
            'optimization_timestamp': str,
            'data_range': str,  # Optional
        }
        """
        self._current_params = params

        # Hide info label
        self.info_label.setVisible(False)

        # Show metadata
        self.metadata_frame.setVisible(True)
        pattern_type = params.get('pattern_type', 'Unknown')
        symbol = params.get('symbol', 'Unknown')
        timeframe = params.get('timeframe', 'Unknown')
        timestamp = params.get('optimization_timestamp', 'Unknown')

        metadata_text = (
            f"<b>Pattern:</b> {pattern_type} | "
            f"<b>Symbol:</b> {symbol} | "
            f"<b>Timeframe:</b> {timeframe} | "
            f"<b>Optimized:</b> {timestamp}"
        )
        self.metadata_label.setText(metadata_text)

        # Populate form parameters
        form_params = params.get('form_params', {})
        if form_params:
            self._populate_table(self.form_table, form_params)
            self.form_table.setVisible(True)

        # Populate action parameters
        action_params = params.get('action_params', {})
        if action_params:
            self._populate_table(self.action_table, action_params)
            self.action_table.setVisible(True)

        # Populate performance metrics
        metrics = params.get('performance_metrics', {})
        if metrics:
            self._populate_metrics_table(metrics)
            self.metrics_table.setVisible(True)

        # Show buttons
        self.save_btn.setVisible(True)
        self.clear_btn.setVisible(True)

        logger.info(f"Displayed optimized params for {pattern_type} on {symbol} {timeframe}")

    def _populate_table(self, table: QTableWidget, params: Dict[str, Any]):
        """Populate a parameter table."""
        table.setRowCount(0)

        for param_name, value in params.items():
            row = table.rowCount()
            table.insertRow(row)

            # Parameter name
            name_item = QTableWidgetItem(self._format_param_name(param_name))
            table.setItem(row, 0, name_item)

            # Parameter value
            value_item = QTableWidgetItem(self._format_value(value))
            value_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 1, value_item)

    def _populate_metrics_table(self, metrics: Dict[str, Any]):
        """Populate the performance metrics table with color coding."""
        self.metrics_table.setRowCount(0)

        # Define metric display order and formatting
        metric_definitions = [
            ('sharpe_ratio', 'Sharpe Ratio', 2),
            ('sortino_ratio', 'Sortino Ratio', 2),
            ('calmar_ratio', 'Calmar Ratio', 2),
            ('win_rate', 'Win Rate', 1, '%'),
            ('profit_factor', 'Profit Factor', 2),
            ('total_return', 'Total Return', 1, '%'),
            ('max_drawdown', 'Max Drawdown', 1, '%'),
            ('avg_trade_return', 'Avg Trade Return', 2, '%'),
            ('num_trades', 'Number of Trades', 0),
            ('expectancy', 'Expectancy', 2),
        ]

        for metric_info in metric_definitions:
            metric_key = metric_info[0]
            if metric_key not in metrics:
                continue

            metric_label = metric_info[1]
            decimals = metric_info[2] if len(metric_info) > 2 else 2
            suffix = metric_info[3] if len(metric_info) > 3 else ''

            value = metrics[metric_key]

            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)

            # Metric name
            name_item = QTableWidgetItem(metric_label)
            self.metrics_table.setItem(row, 0, name_item)

            # Metric value with formatting
            if isinstance(value, (int, float)):
                if decimals == 0:
                    formatted_value = f"{int(value)}"
                else:
                    formatted_value = f"{value:.{decimals}f}{suffix}"
            else:
                formatted_value = str(value)

            value_item = QTableWidgetItem(formatted_value)
            value_item.setTextAlignment(Qt.AlignCenter)

            # Color coding for specific metrics
            if metric_key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
                if value >= 2.0:
                    value_item.setForeground(QBrush(QColor(0, 200, 0)))  # Dark green
                elif value >= 1.0:
                    value_item.setForeground(QBrush(QColor(144, 238, 144)))  # Light green
                elif value >= 0:
                    value_item.setForeground(QBrush(QColor(255, 165, 0)))  # Orange
                else:
                    value_item.setForeground(QBrush(QColor(255, 0, 0)))  # Red

            elif metric_key == 'win_rate':
                if value >= 60:
                    value_item.setForeground(QBrush(QColor(0, 200, 0)))
                elif value >= 50:
                    value_item.setForeground(QBrush(QColor(144, 238, 144)))
                else:
                    value_item.setForeground(QBrush(QColor(255, 165, 0)))

            elif metric_key == 'max_drawdown':
                if value <= 10:
                    value_item.setForeground(QBrush(QColor(0, 200, 0)))
                elif value <= 20:
                    value_item.setForeground(QBrush(QColor(255, 165, 0)))
                else:
                    value_item.setForeground(QBrush(QColor(255, 0, 0)))

            self.metrics_table.setItem(row, 1, value_item)

    def _format_param_name(self, param_name: str) -> str:
        """Format parameter name for display."""
        # Replace underscores with spaces and title case
        return param_name.replace('_', ' ').title()

    def _format_value(self, value: Any) -> str:
        """Format parameter value for display."""
        if isinstance(value, bool):
            return "Yes" if value else "No"
        elif isinstance(value, float):
            return f"{value:.4f}"
        elif isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        else:
            return str(value)

    def _on_save_clicked(self):
        """Handle save to database button click."""
        if not self._current_params:
            QMessageBox.warning(
                self,
                "No Parameters",
                "No optimized parameters to save."
            )
            return

        reply = QMessageBox.question(
            self,
            "Save Parameters",
            "Save these optimized parameters to the database?\n\n"
            "They will be used for future automated trading.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.save_requested.emit(self._current_params)
            logger.info("User requested to save optimized parameters")

    def clear(self):
        """Clear the display."""
        self._current_params = None

        self.info_label.setVisible(True)
        self.metadata_frame.setVisible(False)
        self.form_table.setVisible(False)
        self.action_table.setVisible(False)
        self.metrics_table.setVisible(False)
        self.save_btn.setVisible(False)
        self.clear_btn.setVisible(False)

        self.form_table.setRowCount(0)
        self.action_table.setRowCount(0)
        self.metrics_table.setRowCount(0)

        logger.info("Optimized params display cleared")

    def has_params(self) -> bool:
        """Check if widget currently has parameters displayed."""
        return self._current_params is not None
