"""
Parameter Adaptation Monitor Tab

Displays parameter adaptation history, current parameters, and performance metrics.
Shows when parameters were adapted, validation results, and deployment status.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal as QtSignal, QTimer
from PySide6.QtGui import QColor
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime


class ParameterAdaptationTab(QWidget):
    """
    Parameter Adaptation Monitor showing:
    - Current active parameters
    - Adaptation history with validation results
    - Performance metrics triggering adaptations
    - Deployment status and rollback capability
    """

    # Signal emitted when rollback is requested
    rollback_requested = QtSignal(str)  # adaptation_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.adaptations_data: List[Dict[str, Any]] = []
        self.current_parameters: Dict[str, float] = {}
        self.performance_metrics: Optional[Dict[str, Any]] = None
        self.init_ui()
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()

    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets"""
        from ..i18n.widget_helper import apply_tooltip
        
        if hasattr(self, 'enable_adaptation_check'):
            apply_tooltip(self.enable_adaptation_check, "enable_adaptation", "parameter_adaptation")
        if hasattr(self, 'adaptation_frequency_combo'):
            apply_tooltip(self.adaptation_frequency_combo, "adaptation_frequency", "parameter_adaptation")
        if hasattr(self, 'adaptation_metric_combo'):
            apply_tooltip(self.adaptation_metric_combo, "adaptation_metric", "parameter_adaptation")
        if hasattr(self, 'adaptation_method_combo'):
            apply_tooltip(self.adaptation_method_combo, "adaptation_method", "parameter_adaptation")
    

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Parameter Adaptation Monitor")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Current Parameters Section
        current_params_group = self._create_current_parameters_section()
        layout.addWidget(current_params_group)

        # Performance Metrics Section
        metrics_group = self._create_performance_metrics_section()
        layout.addWidget(metrics_group)

        # Adaptation History Table
        history_group = self._create_adaptation_history_section()
        layout.addWidget(history_group)

        # Statistics Section
        stats_group = self._create_statistics_section()
        layout.addWidget(stats_group)

        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        self.refresh_timer.start(10000)  # Refresh every 10 seconds

    def _create_current_parameters_section(self) -> QGroupBox:
        """Create current active parameters display"""
        group = QGroupBox("Current Active Parameters")
        layout = QVBoxLayout()

        # Parameters table
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(3)
        self.params_table.setHorizontalHeaderLabels([
            'Parameter', 'Current Value', 'Last Changed'
        ])
        self.params_table.setAlternatingRowColors(True)
        self.params_table.setMaximumHeight(200)

        # Initialize with common parameters
        param_names = [
            'quality_threshold',
            'position_size_multiplier',
            'stop_loss_distance',
            'take_profit_distance',
            'max_signals_per_regime'
        ]

        self.params_table.setRowCount(len(param_names))
        for i, param_name in enumerate(param_names):
            self.params_table.setItem(i, 0, QTableWidgetItem(param_name))
            self.params_table.setItem(i, 1, QTableWidgetItem('--'))
            self.params_table.setItem(i, 2, QTableWidgetItem('Never'))

        layout.addWidget(self.params_table)
        group.setLayout(layout)
        return group

    def _create_performance_metrics_section(self) -> QGroupBox:
        """Create performance metrics display"""
        group = QGroupBox("Recent Performance Metrics")
        layout = QHBoxLayout()

        # Win Rate
        self.win_rate_label = QLabel("Win Rate: --")
        layout.addWidget(self.win_rate_label)

        # Profit Factor
        self.profit_factor_label = QLabel("Profit Factor: --")
        layout.addWidget(self.profit_factor_label)

        # Sharpe Ratio
        self.sharpe_label = QLabel("Sharpe Ratio: --")
        layout.addWidget(self.sharpe_label)

        # Max Drawdown
        self.drawdown_label = QLabel("Max DD: --")
        layout.addWidget(self.drawdown_label)

        # Total Trades
        self.trades_label = QLabel("Trades: 0")
        layout.addWidget(self.trades_label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def _create_adaptation_history_section(self) -> QGroupBox:
        """Create adaptation history table"""
        group = QGroupBox("Adaptation History")
        layout = QVBoxLayout()

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(10)
        self.history_table.setHorizontalHeaderLabels([
            'Timestamp', 'Parameter', 'Old Value', 'New Value',
            'Trigger Reason', 'Validation', 'Improvement', 'Deployed',
            'Regime', 'Actions'
        ])
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setAlternatingRowColors(True)

        layout.addWidget(self.history_table)

        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by:"))

        self.regime_filter = QComboBox()
        self.regime_filter.addItems(['All', 'Trending Up', 'Trending Down', 'Ranging', 'High Volatility'])
        self.regime_filter.currentTextChanged.connect(self.filter_adaptations)
        filter_layout.addWidget(QLabel("Regime:"))
        filter_layout.addWidget(self.regime_filter)

        self.deployed_filter = QComboBox()
        self.deployed_filter.addItems(['All', 'Deployed Only', 'Not Deployed'])
        self.deployed_filter.currentTextChanged.connect(self.filter_adaptations)
        filter_layout.addWidget(QLabel("Status:"))
        filter_layout.addWidget(self.deployed_filter)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        group.setLayout(layout)
        return group

    def _create_statistics_section(self) -> QGroupBox:
        """Create statistics display"""
        group = QGroupBox("Adaptation Statistics")
        layout = QHBoxLayout()

        # Total adaptations
        self.total_adaptations_label = QLabel("Total: 0")
        layout.addWidget(self.total_adaptations_label)

        # Deployed
        self.deployed_label = QLabel("Deployed: 0")
        layout.addWidget(self.deployed_label)

        # Validation pass rate
        self.validation_rate_label = QLabel("Validation Rate: 0%")
        layout.addWidget(self.validation_rate_label)

        # Average improvement
        self.avg_improvement_label = QLabel("Avg Improvement: 0.00")
        layout.addWidget(self.avg_improvement_label)

        # Most adapted parameter
        self.most_adapted_label = QLabel("Most Adapted: N/A")
        layout.addWidget(self.most_adapted_label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def update_current_parameters(self, parameters: Dict[str, float]):
        """
        Update current active parameters display.

        Args:
            parameters: Dictionary of parameter names to values
        """
        self.current_parameters = parameters

        for i in range(self.params_table.rowCount()):
            param_name_item = self.params_table.item(i, 0)
            if param_name_item:
                param_name = param_name_item.text()
                if param_name in parameters:
                    value = parameters[param_name]
                    self.params_table.setItem(i, 1, QTableWidgetItem(f"{value:.3f}"))

    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """
        Update performance metrics display.

        Args:
            metrics: Performance metrics dictionary
        """
        self.performance_metrics = metrics

        self.win_rate_label.setText(f"Win Rate: {metrics.get('win_rate', 0.0) * 100:.1f}%")
        self.profit_factor_label.setText(f"Profit Factor: {metrics.get('profit_factor', 0.0):.2f}")
        self.sharpe_label.setText(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
        self.drawdown_label.setText(f"Max DD: {metrics.get('max_drawdown', 0.0):.2f}")
        self.trades_label.setText(f"Trades: {metrics.get('lookback_trades', 0)}")

        # Color code win rate
        win_rate = metrics.get('win_rate', 0.0)
        if win_rate >= 0.50:
            self.win_rate_label.setStyleSheet("color: green; font-weight: bold;")
        elif win_rate >= 0.45:
            self.win_rate_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.win_rate_label.setStyleSheet("color: red; font-weight: bold;")

        # Color code profit factor
        pf = metrics.get('profit_factor', 0.0)
        if pf >= 1.5:
            self.profit_factor_label.setStyleSheet("color: green; font-weight: bold;")
        elif pf >= 1.2:
            self.profit_factor_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.profit_factor_label.setStyleSheet("color: red; font-weight: bold;")

    def update_adaptations(self, adaptations: List[Dict[str, Any]]):
        """
        Update adaptation history.

        Args:
            adaptations: List of adaptation dictionaries
        """
        self.adaptations_data = adaptations
        self.refresh_display()

    def refresh_display(self):
        """Refresh all displays"""
        self.update_adaptations_table()
        self.update_statistics()

    def update_adaptations_table(self):
        """Update adaptations history table"""
        # Filter adaptations
        filtered = self.get_filtered_adaptations()

        self.history_table.setRowCount(len(filtered))

        for i, adaptation in enumerate(filtered):
            # Timestamp
            timestamp = adaptation.get('timestamp', 0)
            dt = datetime.fromtimestamp(timestamp / 1000)
            self.history_table.setItem(i, 0, QTableWidgetItem(dt.strftime('%Y-%m-%d %H:%M')))

            # Parameter
            self.history_table.setItem(i, 1, QTableWidgetItem(adaptation.get('parameter_name', '')))

            # Old/New values
            old_val = adaptation.get('old_value', 0.0)
            new_val = adaptation.get('new_value', 0.0)
            self.history_table.setItem(i, 2, QTableWidgetItem(f"{old_val:.3f}"))
            self.history_table.setItem(i, 3, QTableWidgetItem(f"{new_val:.3f}"))

            # Trigger reason
            trigger = adaptation.get('trigger_reason', 'unknown')
            self.history_table.setItem(i, 4, QTableWidgetItem(trigger))

            # Validation
            validated = adaptation.get('validation_passed', False)
            val_item = QTableWidgetItem('PASS' if validated else 'FAIL')
            val_item.setForeground(QColor('green') if validated else QColor('red'))
            self.history_table.setItem(i, 5, val_item)

            # Improvement
            improvement = adaptation.get('improvement_expected', 0.0)
            self.history_table.setItem(i, 6, QTableWidgetItem(f"{improvement:.3f}"))

            # Deployed
            deployed = adaptation.get('deployed', False)
            deployed_item = QTableWidgetItem('YES' if deployed else 'NO')
            deployed_item.setForeground(QColor('green') if deployed else QColor('gray'))
            self.history_table.setItem(i, 7, deployed_item)

            # Regime
            regime = adaptation.get('regime', 'N/A')
            self.history_table.setItem(i, 8, QTableWidgetItem(regime or 'Global'))

            # Actions (Rollback button for deployed adaptations)
            if deployed and adaptation.get('rollback_at') is None:
                rollback_btn = QPushButton("Rollback")
                adaptation_id = adaptation.get('adaptation_id', '')
                rollback_btn.clicked.connect(lambda checked, aid=adaptation_id: self.rollback_requested.emit(aid))
                self.history_table.setCellWidget(i, 9, rollback_btn)
            else:
                self.history_table.setItem(i, 9, QTableWidgetItem(''))

        self.history_table.resizeColumnsToContents()

    def update_statistics(self):
        """Update statistics labels"""
        if not self.adaptations_data:
            self.total_adaptations_label.setText("Total: 0")
            self.deployed_label.setText("Deployed: 0")
            self.validation_rate_label.setText("Validation Rate: 0%")
            self.avg_improvement_label.setText("Avg Improvement: 0.00")
            self.most_adapted_label.setText("Most Adapted: N/A")
            return

        total = len(self.adaptations_data)
        deployed = sum(1 for a in self.adaptations_data if a.get('deployed', False))
        validated = sum(1 for a in self.adaptations_data if a.get('validation_passed', False))
        val_rate = (validated / total * 100) if total > 0 else 0.0

        deployed_adaptations = [a for a in self.adaptations_data if a.get('deployed', False)]
        if deployed_adaptations:
            avg_improvement = np.mean([a.get('improvement_expected', 0.0) for a in deployed_adaptations])
        else:
            avg_improvement = 0.0

        # Most adapted parameter
        param_counts = {}
        for a in self.adaptations_data:
            param = a.get('parameter_name', 'unknown')
            param_counts[param] = param_counts.get(param, 0) + 1

        if param_counts:
            most_adapted = max(param_counts, key=param_counts.get)
        else:
            most_adapted = 'N/A'

        self.total_adaptations_label.setText(f"Total: {total}")
        self.deployed_label.setText(f"Deployed: {deployed}")
        self.validation_rate_label.setText(f"Validation Rate: {val_rate:.1f}%")
        self.avg_improvement_label.setText(f"Avg Improvement: {avg_improvement:.3f}")
        self.most_adapted_label.setText(f"Most Adapted: {most_adapted}")

    def get_filtered_adaptations(self) -> List[Dict[str, Any]]:
        """Get filtered adaptations based on current filters"""
        filtered = self.adaptations_data

        # Filter by regime
        regime_filter = self.regime_filter.currentText()
        if regime_filter != 'All':
            filtered = [a for a in filtered if a.get('regime') == regime_filter]

        # Filter by deployment status
        deployed_filter = self.deployed_filter.currentText()
        if deployed_filter == 'Deployed Only':
            filtered = [a for a in filtered if a.get('deployed', False)]
        elif deployed_filter == 'Not Deployed':
            filtered = [a for a in filtered if not a.get('deployed', False)]

        return filtered

    def filter_adaptations(self):
        """Trigger adaptation filtering"""
        self.update_adaptations_table()

    def clear_adaptations(self):
        """Clear all adaptations"""
        self.adaptations_data = []
        self.refresh_display()
