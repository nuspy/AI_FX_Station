"""
Signal Quality Dashboard Tab

Displays real-time signal quality metrics, quality dimensions breakdown,
and historical quality vs outcome tracking.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox
)
from PySide6.QtCore import Qt, Signal as QtSignal, QTimer
from PySide6.QtGui import QColor
from typing import Dict, List, Optional, Any
import numpy as np


class SignalQualityTab(QWidget):
    """
    Signal Quality Dashboard showing:
    - Real-time quality scores per signal
    - Quality dimension breakdown
    - Quality threshold configuration
    - Historical quality vs outcome charts
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals_data: List[Dict[str, Any]] = []
        self.init_ui()
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()

    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets"""
        from ..i18n.widget_helper import apply_tooltip
        
        if hasattr(self, 'min_win_rate_spin'):
            apply_tooltip(self.min_win_rate_spin, "min_win_rate", "signal_quality")
        if hasattr(self, 'min_profit_factor_spin'):
            apply_tooltip(self.min_profit_factor_spin, "min_profit_factor", "signal_quality")
        if hasattr(self, 'min_sharpe_ratio_spin'):
            apply_tooltip(self.min_sharpe_ratio_spin, "min_sharpe_ratio", "signal_quality")
        if hasattr(self, 'track_signal_performance_check'):
            apply_tooltip(self.track_signal_performance_check, "track_signal_performance", "signal_quality")
        if hasattr(self, 'auto_disable_poor_signals_check'):
            apply_tooltip(self.auto_disable_poor_signals_check, "auto_disable_poor_signals", "signal_quality")
    

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Signal Quality Dashboard")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Configuration Section
        config_group = self._create_configuration_section()
        layout.addWidget(config_group)

        # Real-time Signals Table
        signals_group = self._create_signals_table_section()
        layout.addWidget(signals_group)

        # Statistics Section
        stats_group = self._create_statistics_section()
        layout.addWidget(stats_group)

        # Quality Dimensions Chart (placeholder)
        dimensions_group = self._create_dimensions_section()
        layout.addWidget(dimensions_group)

        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds

    def _create_configuration_section(self) -> QGroupBox:
        """Create quality threshold configuration section"""
        group = QGroupBox("Quality Configuration")
        layout = QHBoxLayout()

        # Default threshold
        layout.addWidget(QLabel("Default Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.65)
        self.threshold_spin.setDecimals(2)
        layout.addWidget(self.threshold_spin)

        # Regime selector
        layout.addWidget(QLabel("Regime:"))
        self.regime_combo = QComboBox()
        self.regime_combo.addItems([
            'Global', 'Trending Up', 'Trending Down', 'Ranging',
            'High Volatility', 'Transition', 'Accumulation/Distribution'
        ])
        layout.addWidget(self.regime_combo)

        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_configuration)
        layout.addWidget(apply_btn)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def _create_signals_table_section(self) -> QGroupBox:
        """Create real-time signals table"""
        group = QGroupBox("Active Signals")
        layout = QVBoxLayout()

        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(10)
        self.signals_table.setHorizontalHeaderLabels([
            'Signal ID', 'Source', 'Symbol', 'Direction', 'Strength',
            'Quality Score', 'Pattern', 'MTF', 'Regime', 'Status'
        ])
        self.signals_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.signals_table.setAlternatingRowColors(True)

        layout.addWidget(self.signals_table)

        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Min Quality:"))
        self.min_quality_spin = QDoubleSpinBox()
        self.min_quality_spin.setRange(0.0, 1.0)
        self.min_quality_spin.setSingleStep(0.05)
        self.min_quality_spin.setValue(0.5)
        self.min_quality_spin.valueChanged.connect(self.filter_signals)
        filter_layout.addWidget(self.min_quality_spin)

        filter_layout.addWidget(QLabel("Source:"))
        self.source_filter = QComboBox()
        self.source_filter.addItems(['All', 'Pattern', 'Harmonic', 'Order Flow', 'Correlation', 'Event', 'Ensemble'])
        self.source_filter.currentTextChanged.connect(self.filter_signals)
        filter_layout.addWidget(self.source_filter)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        group.setLayout(layout)
        return group

    def _create_statistics_section(self) -> QGroupBox:
        """Create statistics display"""
        group = QGroupBox("Quality Statistics")
        layout = QHBoxLayout()

        # Total signals
        self.total_signals_label = QLabel("Total Signals: 0")
        layout.addWidget(self.total_signals_label)

        # Pass rate
        self.pass_rate_label = QLabel("Pass Rate: 0%")
        layout.addWidget(self.pass_rate_label)

        # Average quality
        self.avg_quality_label = QLabel("Avg Quality: 0.00")
        layout.addWidget(self.avg_quality_label)

        # Top source
        self.top_source_label = QLabel("Top Source: N/A")
        layout.addWidget(self.top_source_label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def _create_dimensions_section(self) -> QGroupBox:
        """Create quality dimensions breakdown"""
        group = QGroupBox("Quality Dimensions Breakdown")
        layout = QVBoxLayout()

        self.dimensions_table = QTableWidget()
        self.dimensions_table.setColumnCount(7)
        self.dimensions_table.setHorizontalHeaderLabels([
            'Dimension', 'Current', 'Min', 'Max', 'Mean', 'Std', 'Weight'
        ])

        # Add rows for each dimension
        dimensions = [
            'Pattern Strength',
            'MTF Agreement',
            'Regime Confidence',
            'Volume Confirmation',
            'Sentiment Alignment',
            'Correlation Safety'
        ]

        self.dimensions_table.setRowCount(len(dimensions))
        for i, dim in enumerate(dimensions):
            self.dimensions_table.setItem(i, 0, QTableWidgetItem(dim))
            # Initialize with zeros
            for j in range(1, 7):
                self.dimensions_table.setItem(i, j, QTableWidgetItem('0.00'))

        layout.addWidget(self.dimensions_table)
        group.setLayout(layout)
        return group

    def update_signals(self, signals: List[Dict[str, Any]]):
        """
        Update signal list and display.

        Args:
            signals: List of signal dictionaries with quality scores
        """
        self.signals_data = signals
        self.refresh_display()

    def refresh_display(self):
        """Refresh all displays"""
        self.update_signals_table()
        self.update_statistics()
        self.update_dimensions()

    def update_signals_table(self):
        """Update signals table"""
        # Filter signals
        filtered = self.get_filtered_signals()

        self.signals_table.setRowCount(len(filtered))

        for i, signal in enumerate(filtered):
            # Signal ID
            self.signals_table.setItem(i, 0, QTableWidgetItem(str(signal.get('signal_id', 'N/A'))))

            # Source
            self.signals_table.setItem(i, 1, QTableWidgetItem(str(signal.get('source', 'Unknown'))))

            # Symbol
            self.signals_table.setItem(i, 2, QTableWidgetItem(str(signal.get('symbol', 'N/A'))))

            # Direction
            direction_item = QTableWidgetItem(str(signal.get('direction', 'neutral')))
            if signal.get('direction') == 'bull':
                direction_item.setForeground(QColor('green'))
            elif signal.get('direction') == 'bear':
                direction_item.setForeground(QColor('red'))
            self.signals_table.setItem(i, 3, direction_item)

            # Strength
            strength = signal.get('strength', 0.0)
            self.signals_table.setItem(i, 4, QTableWidgetItem(f"{strength:.2f}"))

            # Quality Score
            quality_score = signal.get('quality_score', 0.0)
            quality_item = QTableWidgetItem(f"{quality_score:.2f}")

            # Color code by quality
            if quality_score >= 0.8:
                quality_item.setBackground(QColor(0, 255, 0, 50))  # Green
            elif quality_score >= 0.65:
                quality_item.setBackground(QColor(255, 255, 0, 50))  # Yellow
            else:
                quality_item.setBackground(QColor(255, 0, 0, 50))  # Red

            self.signals_table.setItem(i, 5, quality_item)

            # Quality dimensions (simplified)
            dimensions = signal.get('quality_dimensions', {})
            self.signals_table.setItem(i, 6, QTableWidgetItem(f"{dimensions.get('pattern_strength', 0.0):.2f}"))
            self.signals_table.setItem(i, 7, QTableWidgetItem(f"{dimensions.get('mtf_agreement', 0.0):.2f}"))
            self.signals_table.setItem(i, 8, QTableWidgetItem(f"{dimensions.get('regime_confidence', 0.0):.2f}"))

            # Status
            passed = signal.get('passed', False)
            status_item = QTableWidgetItem('PASS' if passed else 'FAIL')
            status_item.setForeground(QColor('green') if passed else QColor('red'))
            self.signals_table.setItem(i, 9, status_item)

        self.signals_table.resizeColumnsToContents()

    def update_statistics(self):
        """Update statistics labels"""
        if not self.signals_data:
            self.total_signals_label.setText("Total Signals: 0")
            self.pass_rate_label.setText("Pass Rate: 0%")
            self.avg_quality_label.setText("Avg Quality: 0.00")
            self.top_source_label.setText("Top Source: N/A")
            return

        total = len(self.signals_data)
        passed = sum(1 for s in self.signals_data if s.get('passed', False))
        pass_rate = (passed / total * 100) if total > 0 else 0.0

        quality_scores = [s.get('quality_score', 0.0) for s in self.signals_data]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0

        # Find top source
        sources = [s.get('source', 'Unknown') for s in self.signals_data]
        if sources:
            top_source = max(set(sources), key=sources.count)
        else:
            top_source = 'N/A'

        self.total_signals_label.setText(f"Total Signals: {total}")
        self.pass_rate_label.setText(f"Pass Rate: {pass_rate:.1f}%")
        self.avg_quality_label.setText(f"Avg Quality: {avg_quality:.2f}")
        self.top_source_label.setText(f"Top Source: {top_source}")

    def update_dimensions(self):
        """Update dimensions breakdown table"""
        if not self.signals_data:
            return

        dimension_keys = [
            'pattern_strength', 'mtf_agreement', 'regime_confidence',
            'volume_confirmation', 'sentiment_alignment', 'correlation_safety'
        ]

        for i, key in enumerate(dimension_keys):
            values = []
            for signal in self.signals_data:
                dims = signal.get('quality_dimensions', {})
                if key in dims:
                    values.append(dims[key])

            if not values:
                continue

            # Current (last value)
            self.dimensions_table.setItem(i, 1, QTableWidgetItem(f"{values[-1]:.2f}"))

            # Min
            self.dimensions_table.setItem(i, 2, QTableWidgetItem(f"{min(values):.2f}"))

            # Max
            self.dimensions_table.setItem(i, 3, QTableWidgetItem(f"{max(values):.2f}"))

            # Mean
            self.dimensions_table.setItem(i, 4, QTableWidgetItem(f"{np.mean(values):.2f}"))

            # Std
            self.dimensions_table.setItem(i, 5, QTableWidgetItem(f"{np.std(values):.2f}"))

            # Weight (placeholder)
            self.dimensions_table.setItem(i, 6, QTableWidgetItem("0.17"))

    def get_filtered_signals(self) -> List[Dict[str, Any]]:
        """Get filtered signals based on current filters"""
        filtered = self.signals_data

        # Filter by minimum quality
        min_quality = self.min_quality_spin.value()
        filtered = [s for s in filtered if s.get('quality_score', 0.0) >= min_quality]

        # Filter by source
        source_filter = self.source_filter.currentText()
        if source_filter != 'All':
            filtered = [s for s in filtered if s.get('source', '').lower() == source_filter.lower()]

        return filtered

    def filter_signals(self):
        """Trigger signal filtering"""
        self.update_signals_table()

    def apply_configuration(self):
        """Apply configuration changes"""
        threshold = self.threshold_spin.value()
        regime = self.regime_combo.currentText()

        # Emit signal or update configuration
        # This would connect to the signal quality scorer
        print(f"Applied config: threshold={threshold}, regime={regime}")

    def clear_signals(self):
        """Clear all signals"""
        self.signals_data = []
        self.refresh_display()
