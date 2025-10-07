"""
Regime Transition Indicator Widget

Displays current market regime with confidence level, transition state,
and recent regime history. Provides visual alerts for regime changes.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QGroupBox, QTableWidget, QTableWidgetItem, QFrame
)
from PySide6.QtCore import Qt, Signal as QtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QPalette, QFont
from typing import Dict, List, Optional, Any
from datetime import datetime


class RegimeIndicatorWidget(QWidget):
    """
    Compact regime indicator widget showing:
    - Current regime with color coding
    - Confidence level with progress bar
    - Transition state alert
    - Recent regime history
    """

    # Signal emitted when regime changes
    regime_changed = QtSignal(str, float)  # regime_name, confidence

    # Regime color scheme
    REGIME_COLORS = {
        'Trending Up': QColor(34, 139, 34),      # Forest green
        'Trending Down': QColor(220, 20, 60),    # Crimson
        'Ranging': QColor(70, 130, 180),         # Steel blue
        'High Volatility': QColor(255, 140, 0),  # Dark orange
        'Transition': QColor(128, 128, 128),     # Gray
        'Accumulation/Distribution': QColor(138, 43, 226),  # Blue violet
        'Unknown': QColor(169, 169, 169)         # Dark gray
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_regime: str = 'Unknown'
        self.current_confidence: float = 0.0
        self.is_transition: bool = False
        self.regime_history: List[Dict[str, Any]] = []
        self.max_history: int = 10

        # Animation for regime change
        self.flash_animation: Optional[QPropertyAnimation] = None

        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Current Regime Display (Large)
        current_group = self._create_current_regime_section()
        layout.addWidget(current_group)

        # Confidence & Transition Indicators
        indicators_group = self._create_indicators_section()
        layout.addWidget(indicators_group)

        # Recent History
        history_group = self._create_history_section()
        layout.addWidget(history_group)

    def _create_current_regime_section(self) -> QGroupBox:
        """Create current regime display section"""
        group = QGroupBox("Current Market Regime")
        layout = QVBoxLayout()

        # Regime name label (large, bold)
        self.regime_label = QLabel("Unknown")
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.regime_label.setFont(font)
        self.regime_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.regime_label.setStyleSheet("padding: 10px; border-radius: 5px;")
        self._update_regime_label_color('Unknown')

        layout.addWidget(self.regime_label)

        # Transition warning label
        self.transition_label = QLabel("")
        self.transition_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.transition_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
        self.transition_label.hide()
        layout.addWidget(self.transition_label)

        group.setLayout(layout)
        return group

    def _create_indicators_section(self) -> QGroupBox:
        """Create confidence and transition indicators"""
        group = QGroupBox("Regime Metrics")
        layout = QVBoxLayout()

        # Confidence level
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("%p%")
        conf_layout.addWidget(self.confidence_bar)

        self.confidence_value_label = QLabel("0%")
        self.confidence_value_label.setStyleSheet("font-weight: bold;")
        conf_layout.addWidget(self.confidence_value_label)

        layout.addLayout(conf_layout)

        # Entropy indicator (transition detection)
        entropy_layout = QHBoxLayout()
        entropy_layout.addWidget(QLabel("Stability:"))

        self.stability_bar = QProgressBar()
        self.stability_bar.setRange(0, 100)
        self.stability_bar.setValue(100)
        self.stability_bar.setTextVisible(True)
        self.stability_bar.setFormat("%p%")
        entropy_layout.addWidget(self.stability_bar)

        self.stability_label = QLabel("Stable")
        self.stability_label.setStyleSheet("color: green; font-weight: bold;")
        entropy_layout.addWidget(self.stability_label)

        layout.addLayout(entropy_layout)

        # Duration in current regime
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration:"))
        self.duration_label = QLabel("--")
        self.duration_label.setStyleSheet("font-weight: bold;")
        duration_layout.addWidget(self.duration_label)
        duration_layout.addStretch()
        layout.addLayout(duration_layout)

        group.setLayout(layout)
        return group

    def _create_history_section(self) -> QGroupBox:
        """Create regime history table"""
        group = QGroupBox("Recent Regime History")
        layout = QVBoxLayout()

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels([
            'Timestamp', 'Regime', 'Confidence', 'Duration'
        ])
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setMaximumHeight(150)
        self.history_table.verticalHeader().setVisible(False)

        layout.addWidget(self.history_table)
        group.setLayout(layout)
        return group

    def update_regime(
        self,
        regime: str,
        confidence: float,
        is_transition: bool = False,
        entropy: Optional[float] = None,
        duration_bars: Optional[int] = None,
        timestamp: Optional[int] = None
    ):
        """
        Update regime display.

        Args:
            regime: Regime name
            confidence: Confidence level (0-1)
            is_transition: Whether in transition state
            entropy: Probability entropy (higher = more uncertain)
            duration_bars: Bars in current regime
            timestamp: Unix timestamp in milliseconds
        """
        # Detect regime change
        regime_changed = (regime != self.current_regime)

        if regime_changed:
            # Add to history
            self._add_to_history(
                regime=self.current_regime,
                confidence=self.current_confidence,
                timestamp=timestamp or int(datetime.now().timestamp() * 1000)
            )

            # Flash animation on change
            self._flash_regime_change()

            # Emit signal
            self.regime_changed.emit(regime, confidence)

        # Update current state
        self.current_regime = regime
        self.current_confidence = confidence
        self.is_transition = is_transition

        # Update UI
        self.regime_label.setText(regime)
        self._update_regime_label_color(regime)

        # Update confidence
        confidence_pct = int(confidence * 100)
        self.confidence_bar.setValue(confidence_pct)
        self.confidence_value_label.setText(f"{confidence_pct}%")

        # Color code confidence bar
        if confidence >= 0.8:
            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        elif confidence >= 0.6:
            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")

        # Update transition indicator
        if is_transition:
            self.transition_label.setText("⚠️ REGIME TRANSITION IN PROGRESS")
            self.transition_label.show()
        else:
            self.transition_label.hide()

        # Update stability (inverse of entropy)
        if entropy is not None:
            # Normalize entropy to 0-1 (assuming max entropy ~2.5 for 6 states)
            stability = max(0, 1.0 - (entropy / 2.5))
            stability_pct = int(stability * 100)
            self.stability_bar.setValue(stability_pct)

            if stability >= 0.7:
                self.stability_label.setText("Stable")
                self.stability_label.setStyleSheet("color: green; font-weight: bold;")
            elif stability >= 0.4:
                self.stability_label.setText("Uncertain")
                self.stability_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.stability_label.setText("Unstable")
                self.stability_label.setStyleSheet("color: red; font-weight: bold;")

        # Update duration
        if duration_bars is not None:
            self.duration_label.setText(f"{duration_bars} bars")

    def _update_regime_label_color(self, regime: str):
        """Update regime label background color"""
        color = self.REGIME_COLORS.get(regime, self.REGIME_COLORS['Unknown'])

        # Determine text color based on background brightness
        brightness = (color.red() * 299 + color.green() * 587 + color.blue() * 114) / 1000
        text_color = "white" if brightness < 128 else "black"

        self.regime_label.setStyleSheet(
            f"background-color: {color.name()}; "
            f"color: {text_color}; "
            f"padding: 10px; "
            f"border-radius: 5px;"
        )

    def _flash_regime_change(self):
        """Flash animation when regime changes"""
        # Simple opacity flash effect
        self.regime_label.setStyleSheet(
            self.regime_label.styleSheet() + " opacity: 0.5;"
        )

        # Reset after short delay
        QTimer.singleShot(200, lambda: self._update_regime_label_color(self.current_regime))

    def _add_to_history(self, regime: str, confidence: float, timestamp: int):
        """
        Add regime to history.

        Args:
            regime: Regime name
            confidence: Confidence level
            timestamp: Unix timestamp in milliseconds
        """
        if regime == 'Unknown':
            return

        # Calculate duration if we have previous entry
        duration_ms = None
        if self.regime_history:
            last_entry = self.regime_history[-1]
            duration_ms = timestamp - last_entry['timestamp']

        # Add new entry
        entry = {
            'regime': regime,
            'confidence': confidence,
            'timestamp': timestamp,
            'duration_ms': duration_ms
        }
        self.regime_history.append(entry)

        # Limit history size
        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)

        # Update history table
        self._update_history_table()

    def _update_history_table(self):
        """Update history table display"""
        self.history_table.setRowCount(len(self.regime_history))

        for i, entry in enumerate(reversed(self.regime_history)):
            # Timestamp
            dt = datetime.fromtimestamp(entry['timestamp'] / 1000)
            self.history_table.setItem(i, 0, QTableWidgetItem(dt.strftime('%H:%M:%S')))

            # Regime (color-coded)
            regime_item = QTableWidgetItem(entry['regime'])
            color = self.REGIME_COLORS.get(entry['regime'], self.REGIME_COLORS['Unknown'])
            regime_item.setForeground(color)
            regime_item.setFont(QFont('Arial', -1, QFont.Weight.Bold))
            self.history_table.setItem(i, 1, regime_item)

            # Confidence
            conf_pct = int(entry['confidence'] * 100)
            self.history_table.setItem(i, 2, QTableWidgetItem(f"{conf_pct}%"))

            # Duration
            if entry['duration_ms']:
                duration_sec = entry['duration_ms'] / 1000
                if duration_sec < 60:
                    duration_str = f"{duration_sec:.0f}s"
                elif duration_sec < 3600:
                    duration_str = f"{duration_sec/60:.1f}m"
                else:
                    duration_str = f"{duration_sec/3600:.1f}h"
                self.history_table.setItem(i, 3, QTableWidgetItem(duration_str))
            else:
                self.history_table.setItem(i, 3, QTableWidgetItem("--"))

        self.history_table.resizeColumnsToContents()

    def get_current_regime(self) -> Dict[str, Any]:
        """
        Get current regime state.

        Returns:
            Dictionary with regime, confidence, is_transition
        """
        return {
            'regime': self.current_regime,
            'confidence': self.current_confidence,
            'is_transition': self.is_transition
        }

    def clear_history(self):
        """Clear regime history"""
        self.regime_history = []
        self._update_history_table()

    def set_max_history(self, max_entries: int):
        """Set maximum history entries"""
        self.max_history = max_entries
        while len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)
        self._update_history_table()
