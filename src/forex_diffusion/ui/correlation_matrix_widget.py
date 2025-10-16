"""
Correlation Matrix Heatmap Widget

Displays rolling correlation matrix as a heatmap with color-coding,
correlation breakdown alerts, and divergence opportunity detection.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QGroupBox, QPushButton, QComboBox, QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, Signal as QtSignal, QTimer
from PySide6.QtGui import QColor, QBrush, QFont
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class CorrelationMatrixWidget(QWidget):
    """
    Correlation Matrix Heatmap showing:
    - Rolling correlation matrix with color-coded heatmap
    - Correlation breakdown alerts
    - Divergence/convergence opportunities
    - Portfolio correlation risk assessment
    """

    # Signal emitted when breakdown/opportunity detected
    alert_triggered = QtSignal(str, dict)  # alert_type, details

    # Color scheme for correlations (-1 to +1)
    @staticmethod
    def get_correlation_color(correlation: float) -> QColor:
        """
        Get color for correlation value.

        Red: Strong negative correlation (-1.0 to -0.5)
        Orange: Weak negative correlation (-0.5 to 0.0)
        Yellow: No correlation (0.0)
        Light Green: Weak positive correlation (0.0 to 0.5)
        Dark Green: Strong positive correlation (0.5 to 1.0)
        """
        if correlation >= 0.7:
            # Strong positive - Dark green
            intensity = int(34 + (correlation - 0.7) * (100 / 0.3))
            return QColor(intensity, 139 + (100 - intensity), intensity)
        elif correlation >= 0.3:
            # Moderate positive - Light green
            intensity = int(144 + (correlation - 0.3) * (94 / 0.4))
            return QColor(intensity, 238, intensity)
        elif correlation >= -0.3:
            # Near zero - Yellow to white
            intensity = int(255 - abs(correlation) * 85)
            return QColor(255, 255, intensity)
        elif correlation >= -0.7:
            # Moderate negative - Orange
            intensity = int(255 - abs(correlation + 0.3) * (90 / 0.4))
            return QColor(255, intensity, 0)
        else:
            # Strong negative - Red
            intensity = int(220 - abs(correlation + 0.7) * (100 / 0.3))
            return QColor(intensity, 0, 0)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Default asset list
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'XAUUSD']
        self.correlation_matrix: Optional[np.ndarray] = None
        self.expected_correlations: Dict[Tuple[str, str], float] = {}
        self.breakdown_threshold: float = 0.3  # Deviation threshold

        # Known correlations (from CorrelationAnalyzer)
        self._init_expected_correlations()

        self.init_ui()
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()

    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets"""
        from ..i18n.widget_helper import apply_tooltip
        
        if hasattr(self, 'correlation_window_spin'):
            apply_tooltip(self.correlation_window_spin, "correlation_window", "correlation_matrix")
        if hasattr(self, 'correlation_method_combo'):
            apply_tooltip(self.correlation_method_combo, "correlation_method", "correlation_matrix")
        if hasattr(self, 'highlight_threshold_spin'):
            apply_tooltip(self.highlight_threshold_spin, "highlight_threshold", "correlation_matrix")
        if hasattr(self, 'auto_diversify_check'):
            apply_tooltip(self.auto_diversify_check, "auto_diversify", "correlation_matrix")
    

    def _init_expected_correlations(self):
        """Initialize expected correlation patterns"""
        self.expected_correlations = {
            ('EURUSD', 'GBPUSD'): 0.75,
            ('GBPUSD', 'EURUSD'): 0.75,
            ('EURUSD', 'USDCHF'): -0.85,
            ('USDCHF', 'EURUSD'): -0.85,
            ('AUDUSD', 'NZDUSD'): 0.85,
            ('NZDUSD', 'AUDUSD'): 0.85,
            ('AUDUSD', 'XAUUSD'): 0.65,
            ('XAUUSD', 'AUDUSD'): 0.65,
        }

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)

        # Title and controls
        header_layout = QHBoxLayout()
        title = QLabel("Correlation Matrix Heatmap")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title)

        # Window size control
        header_layout.addWidget(QLabel("Window:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(20, 200)
        self.window_spin.setValue(50)
        self.window_spin.setSuffix(" bars")
        self.window_spin.valueChanged.connect(self.on_window_changed)
        header_layout.addWidget(self.window_spin)

        # Show values checkbox
        self.show_values_check = QCheckBox("Show Values")
        self.show_values_check.setChecked(True)
        self.show_values_check.stateChanged.connect(self.refresh_display)
        header_layout.addWidget(self.show_values_check)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_display)
        header_layout.addWidget(refresh_btn)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Correlation matrix heatmap
        matrix_group = self._create_matrix_section()
        layout.addWidget(matrix_group)

        # Statistics and alerts
        stats_layout = QHBoxLayout()

        # Statistics
        stats_group = self._create_statistics_section()
        stats_layout.addWidget(stats_group, 2)

        # Alerts
        alerts_group = self._create_alerts_section()
        stats_layout.addWidget(alerts_group, 3)

        layout.addLayout(stats_layout)

        # Opportunities table
        opps_group = self._create_opportunities_section()
        layout.addWidget(opps_group)

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        self.refresh_timer.start(5000)  # Every 5 seconds

    def _create_matrix_section(self) -> QGroupBox:
        """Create correlation matrix heatmap table"""
        group = QGroupBox("Correlation Heatmap")
        layout = QVBoxLayout()

        self.matrix_table = QTableWidget()
        self.matrix_table.setAlternatingRowColors(False)  # We'll color it ourselves

        # Initialize with default assets
        n = len(self.assets)
        self.matrix_table.setRowCount(n)
        self.matrix_table.setColumnCount(n)
        self.matrix_table.setHorizontalHeaderLabels(self.assets)
        self.matrix_table.setVerticalHeaderLabels(self.assets)

        # Set minimum cell size
        self.matrix_table.setMinimumHeight(300)
        for i in range(n):
            self.matrix_table.setColumnWidth(i, 80)
            self.matrix_table.setRowHeight(i, 40)

        layout.addWidget(self.matrix_table)

        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("Correlation:"))

        # Create color legend
        legend_items = [
            ("Strong -", QColor(139, 0, 0)),
            ("Weak -", QColor(255, 140, 0)),
            ("None", QColor(255, 255, 170)),
            ("Weak +", QColor(144, 238, 144)),
            ("Strong +", QColor(34, 139, 34))
        ]

        for label, color in legend_items:
            color_box = QLabel("  ")
            color_box.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
            legend_layout.addWidget(QLabel(label))
            legend_layout.addWidget(color_box)

        legend_layout.addStretch()
        layout.addLayout(legend_layout)

        group.setLayout(layout)
        return group

    def _create_statistics_section(self) -> QGroupBox:
        """Create statistics display"""
        group = QGroupBox("Correlation Statistics")
        layout = QVBoxLayout()

        self.avg_corr_label = QLabel("Avg Correlation: --")
        layout.addWidget(self.avg_corr_label)

        self.max_corr_label = QLabel("Max Correlation: --")
        layout.addWidget(self.max_corr_label)

        self.min_corr_label = QLabel("Min Correlation: --")
        layout.addWidget(self.min_corr_label)

        self.regime_label = QLabel("Regime: --")
        layout.addWidget(self.regime_label)

        group.setLayout(layout)
        return group

    def _create_alerts_section(self) -> QGroupBox:
        """Create alerts display"""
        group = QGroupBox("Correlation Alerts")
        layout = QVBoxLayout()

        # Breakdown alert
        self.breakdown_alert = QLabel("")
        self.breakdown_alert.setStyleSheet(
            "background-color: #F8D7DA; color: #721C24; "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )
        self.breakdown_alert.setWordWrap(True)
        self.breakdown_alert.hide()
        layout.addWidget(self.breakdown_alert)

        # Divergence opportunity
        self.divergence_alert = QLabel("")
        self.divergence_alert.setStyleSheet(
            "background-color: #D1ECF1; color: #0C5460; "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )
        self.divergence_alert.setWordWrap(True)
        self.divergence_alert.hide()
        layout.addWidget(self.divergence_alert)

        # Portfolio risk
        self.portfolio_risk_label = QLabel("Portfolio Risk: --")
        self.portfolio_risk_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.portfolio_risk_label)

        group.setLayout(layout)
        return group

    def _create_opportunities_section(self) -> QGroupBox:
        """Create opportunities table"""
        group = QGroupBox("Trading Opportunities")
        layout = QVBoxLayout()

        self.opps_table = QTableWidget()
        self.opps_table.setColumnCount(5)
        self.opps_table.setHorizontalHeaderLabels([
            'Type', 'Pair 1', 'Pair 2', 'Current Corr', 'Expected Corr'
        ])
        self.opps_table.setAlternatingRowColors(True)
        self.opps_table.setMaximumHeight(120)

        layout.addWidget(self.opps_table)
        group.setLayout(layout)
        return group

    def update_correlation_matrix(
        self,
        correlation_matrix: np.ndarray,
        assets: List[str],
        window_size: int,
        timestamp: Optional[int] = None
    ):
        """
        Update correlation matrix display.

        Args:
            correlation_matrix: NxN correlation matrix
            assets: List of asset symbols
            window_size: Window size used for calculation
            timestamp: Unix timestamp in milliseconds
        """
        self.correlation_matrix = correlation_matrix
        self.assets = assets

        # Update window display
        self.window_spin.setValue(window_size)

        # Update matrix table
        self._update_matrix_display()

        # Update statistics
        self._update_statistics()

        # Check for breakdowns and opportunities
        self._check_alerts()

    def _update_matrix_display(self):
        """Update matrix heatmap display"""
        if self.correlation_matrix is None:
            return

        n = len(self.assets)
        self.matrix_table.setRowCount(n)
        self.matrix_table.setColumnCount(n)
        self.matrix_table.setHorizontalHeaderLabels(self.assets)
        self.matrix_table.setVerticalHeaderLabels(self.assets)

        show_values = self.show_values_check.isChecked()

        for i in range(n):
            for j in range(n):
                corr = self.correlation_matrix[i, j]

                # Create item
                if show_values:
                    if i == j:
                        item = QTableWidgetItem("1.00")
                    else:
                        item = QTableWidgetItem(f"{corr:.2f}")
                else:
                    item = QTableWidgetItem("")

                # Set background color
                color = self.get_correlation_color(corr)
                item.setBackground(QBrush(color))

                # Set text color (black or white based on background)
                brightness = (color.red() * 299 + color.green() * 587 + color.blue() * 114) / 1000
                text_color = Qt.GlobalColor.white if brightness < 128 else Qt.GlobalColor.black
                item.setForeground(QBrush(text_color))

                # Center align
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Bold for diagonal
                if i == j:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

                self.matrix_table.setItem(i, j, item)

    def _update_statistics(self):
        """Update correlation statistics"""
        if self.correlation_matrix is None:
            return

        # Get upper triangle (excluding diagonal)
        n = self.correlation_matrix.shape[0]
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(self.correlation_matrix[i, j])

        if not upper_triangle:
            return

        avg_corr = np.mean(upper_triangle)
        max_corr = np.max(upper_triangle)
        min_corr = np.min(upper_triangle)

        self.avg_corr_label.setText(f"Avg Correlation: {avg_corr:.2f}")
        self.max_corr_label.setText(f"Max Correlation: {max_corr:.2f}")
        self.min_corr_label.setText(f"Min Correlation: {min_corr:.2f}")

        # Classify correlation regime
        if avg_corr > 0.5:
            regime = "High Correlation"
            color = "red"
        elif avg_corr > 0.2:
            regime = "Moderate Correlation"
            color = "orange"
        else:
            regime = "Low Correlation"
            color = "green"

        self.regime_label.setText(f"Regime: {regime}")
        self.regime_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _check_alerts(self):
        """Check for correlation breakdowns and opportunities"""
        if self.correlation_matrix is None:
            return

        breakdowns = []
        divergences = []

        n = len(self.assets)
        for i in range(n):
            for j in range(i + 1, n):
                pair = (self.assets[i], self.assets[j])
                current_corr = self.correlation_matrix[i, j]

                # Check for expected correlation breakdown
                if pair in self.expected_correlations:
                    expected_corr = self.expected_correlations[pair]
                    deviation = abs(current_corr - expected_corr)

                    if deviation > self.breakdown_threshold:
                        breakdowns.append({
                            'pair1': self.assets[i],
                            'pair2': self.assets[j],
                            'expected': expected_corr,
                            'current': current_corr,
                            'deviation': deviation
                        })

                        # Check if it's a divergence opportunity
                        if abs(current_corr) < abs(expected_corr) * 0.5:
                            divergences.append({
                                'pair1': self.assets[i],
                                'pair2': self.assets[j],
                                'expected': expected_corr,
                                'current': current_corr,
                                'type': 'convergence' if expected_corr > 0 else 'divergence'
                            })

        # Update breakdown alert
        if breakdowns:
            breakdown = breakdowns[0]  # Show first one
            self.breakdown_alert.setText(
                f"⚠️ Correlation Breakdown: {breakdown['pair1']}/{breakdown['pair2']} "
                f"(Expected: {breakdown['expected']:.2f}, Current: {breakdown['current']:.2f})"
            )
            self.breakdown_alert.show()
            self.alert_triggered.emit('breakdown', breakdown)
        else:
            self.breakdown_alert.hide()

        # Update divergence alert
        if divergences:
            div = divergences[0]
            self.divergence_alert.setText(
                f"💡 Trading Opportunity: {div['pair1']}/{div['pair2']} "
                f"{div['type'].upper()} expected (Correlation: {div['current']:.2f})"
            )
            self.divergence_alert.show()
            self.alert_triggered.emit('opportunity', div)
        else:
            self.divergence_alert.hide()

        # Update opportunities table
        self._update_opportunities_table(breakdowns + divergences)

    def _update_opportunities_table(self, opportunities: List[Dict[str, Any]]):
        """Update opportunities table"""
        self.opps_table.setRowCount(len(opportunities))

        for i, opp in enumerate(opportunities):
            # Type
            opp_type = opp.get('type', 'breakdown').upper()
            self.opps_table.setItem(i, 0, QTableWidgetItem(opp_type))

            # Pair 1
            self.opps_table.setItem(i, 1, QTableWidgetItem(opp['pair1']))

            # Pair 2
            self.opps_table.setItem(i, 2, QTableWidgetItem(opp['pair2']))

            # Current correlation
            current = opp['current']
            current_item = QTableWidgetItem(f"{current:.2f}")
            current_item.setForeground(QBrush(self.get_correlation_color(current)))
            self.opps_table.setItem(i, 3, current_item)

            # Expected correlation
            expected = opp['expected']
            expected_item = QTableWidgetItem(f"{expected:.2f}")
            expected_item.setForeground(QBrush(self.get_correlation_color(expected)))
            self.opps_table.setItem(i, 4, expected_item)

        self.opps_table.resizeColumnsToContents()

    def update_portfolio_risk(self, risk_assessment: Dict[str, Any]):
        """
        Update portfolio correlation risk display.

        Args:
            risk_assessment: Risk assessment from CorrelationAnalyzer
        """
        risk_level = risk_assessment.get('risk_level', 'unknown')
        avg_correlation = risk_assessment.get('avg_correlation', 0.0)
        max_correlation = risk_assessment.get('max_correlation', 0.0)

        risk_colors = {
            'low': 'green',
            'medium': 'orange',
            'high': 'red',
            'critical': 'darkred'
        }

        color = risk_colors.get(risk_level, 'gray')

        self.portfolio_risk_label.setText(
            f"Portfolio Risk: {risk_level.upper()} "
            f"(Avg: {avg_correlation:.2f}, Max: {max_correlation:.2f})"
        )
        self.portfolio_risk_label.setStyleSheet(f"color: {color}; font-weight: bold; padding: 5px;")

    def refresh_display(self):
        """Refresh display (called by timer or button)"""
        if self.correlation_matrix is not None:
            self._update_matrix_display()

    def on_window_changed(self, value: int):
        """Handle window size change"""
        # Emit signal or trigger recalculation
        # In production, would request new correlation matrix with new window
        pass

    def clear_data(self):
        """Clear all data"""
        self.correlation_matrix = None
        self.matrix_table.clearContents()
        self.opps_table.setRowCount(0)
        self.breakdown_alert.hide()
        self.divergence_alert.hide()
