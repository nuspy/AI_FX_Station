"""
Multi-Horizon Prediction Display Widget

Shows predictions for multiple horizons in a clean, organized format.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from typing import Dict, Any, List, Optional


class MultiHorizonDisplayWidget(QWidget):
    """
    Widget to display multi-horizon predictions.
    
    Features:
    - Shows each horizon with its prediction
    - Color-coded by direction (up/down)
    - Time conversion (bars → human readable)
    - Percentage change display
    - Distribution stats (if available)
    - Interpolation indicator
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictions = {}
        self.current_price = None
        self.timeframe = '1m'
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("Multi-Horizon Predictions")
        title_font = title.font()
        title_font.setPointSize(10)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Scroll area for predictions
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        self.predictions_container = QWidget()
        self.predictions_layout = QVBoxLayout(self.predictions_container)
        self.predictions_layout.setSpacing(5)
        self.predictions_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll.setWidget(self.predictions_container)
        layout.addWidget(scroll)
        
        self.setLayout(layout)
    
    def set_predictions(
        self,
        predictions: Dict[int, Any],
        current_price: float,
        timeframe: str = '1m',
        trained_horizons: Optional[List[int]] = None
    ):
        """
        Update predictions display.
        
        Args:
            predictions: Dict mapping horizon -> prediction value or stats
            current_price: Current price for percentage calculation
            timeframe: Timeframe for time conversion
            trained_horizons: List of horizons model was trained on (for interpolation indicator)
        """
        self.predictions = predictions
        self.current_price = current_price
        self.timeframe = timeframe
        
        # Clear existing widgets
        while self.predictions_layout.count():
            item = self.predictions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not predictions:
            no_data = QLabel("No predictions available")
            no_data.setStyleSheet("color: #999; font-style: italic;")
            self.predictions_layout.addWidget(no_data)
            return
        
        # Sort horizons
        sorted_horizons = sorted(predictions.keys())
        
        # Create prediction cards
        for horizon in sorted_horizons:
            pred_data = predictions[horizon]
            
            # Check if interpolated
            is_interpolated = (
                trained_horizons is not None and
                horizon not in trained_horizons
            )
            
            card = self._create_prediction_card(
                horizon,
                pred_data,
                current_price,
                timeframe,
                is_interpolated
            )
            
            self.predictions_layout.addWidget(card)
        
        self.predictions_layout.addStretch()
    
    def _create_prediction_card(
        self,
        horizon: int,
        pred_data: Any,
        current_price: float,
        timeframe: str,
        is_interpolated: bool = False
    ) -> QGroupBox:
        """
        Create a prediction card for one horizon.
        
        Args:
            horizon: Horizon in bars
            pred_data: Prediction value (float) or stats (dict)
            current_price: Current price
            timeframe: Timeframe
            is_interpolated: Whether this horizon was interpolated
            
        Returns:
            QGroupBox containing the prediction display
        """
        # Extract prediction value
        if isinstance(pred_data, dict):
            # Distribution stats available
            pred_value = pred_data['mean']
            has_distribution = True
        else:
            # Single value
            pred_value = float(pred_data)
            has_distribution = False
        
        # Calculate percentage change
        if current_price and current_price > 0:
            pct_change = ((pred_value - current_price) / current_price) * 100
        else:
            pct_change = 0.0
        
        # Create card
        card = QGroupBox()
        card_layout = QVBoxLayout()
        card_layout.setSpacing(3)
        
        # Header: Horizon + Time
        header_layout = QHBoxLayout()
        
        horizon_label = QLabel(f"Horizon: {horizon} bars")
        horizon_font = horizon_label.font()
        horizon_font.setBold(True)
        horizon_label.setFont(horizon_font)
        header_layout.addWidget(horizon_label)
        
        time_str = self._bars_to_time(horizon, timeframe)
        time_label = QLabel(f"({time_str})")
        time_label.setStyleSheet("color: #666; font-size: 9pt;")
        header_layout.addWidget(time_label)
        
        if is_interpolated:
            interp_label = QLabel("🔧 Interpolated")
            interp_label.setStyleSheet("color: #ff9800; font-size: 8pt; font-style: italic;")
            interp_label.setToolTip("This horizon was not in training data - value interpolated")
            header_layout.addWidget(interp_label)
        
        header_layout.addStretch()
        card_layout.addLayout(header_layout)
        
        # Prediction value
        value_layout = QHBoxLayout()
        
        value_label = QLabel(f"{pred_value:.5f}")
        value_font = value_label.font()
        value_font.setPointSize(14)
        value_font.setBold(True)
        value_label.setFont(value_font)
        
        # Color code by direction
        if pct_change > 0:
            color = "#4caf50"  # Green
            arrow = "↑"
        elif pct_change < 0:
            color = "#f44336"  # Red
            arrow = "↓"
        else:
            color = "#666"
            arrow = "→"
        
        value_label.setStyleSheet(f"color: {color};")
        value_layout.addWidget(value_label)
        
        pct_label = QLabel(f"{arrow} {abs(pct_change):.2f}%")
        pct_font = pct_label.font()
        pct_font.setPointSize(11)
        pct_label.setFont(pct_font)
        pct_label.setStyleSheet(f"color: {color};")
        value_layout.addWidget(pct_label)
        
        value_layout.addStretch()
        card_layout.addLayout(value_layout)
        
        # Distribution stats (if available)
        if has_distribution and 'std' in pred_data:
            stats_text = (
                f"σ: {pred_data['std']:.5f}  |  "
                f"Q05: {pred_data.get('q05', 0):.5f}  |  "
                f"Q95: {pred_data.get('q95', 0):.5f}"
            )
            stats_label = QLabel(stats_text)
            stats_label.setStyleSheet("color: #888; font-size: 8pt; font-family: monospace;")
            stats_label.setToolTip(
                f"Standard Deviation: {pred_data['std']:.5f}\n"
                f"5th Percentile: {pred_data.get('q05', 0):.5f}\n"
                f"95th Percentile: {pred_data.get('q95', 0):.5f}"
            )
            card_layout.addWidget(stats_label)
        
        card.setLayout(card_layout)
        
        # Style card
        card.setStyleSheet("""
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 5px;
                padding: 8px;
                background-color: #fafafa;
            }
        """)
        
        return card
    
    def _bars_to_time(self, bars: int, timeframe: str) -> str:
        """Convert bars to human-readable time."""
        tf_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        minutes = bars * tf_minutes.get(timeframe, 1)
        
        if minutes < 60:
            return f"{minutes} min"
        elif minutes < 1440:
            hours = minutes / 60
            if hours == int(hours):
                return f"{int(hours)} hour{'s' if hours > 1 else ''}"
            else:
                return f"{hours:.1f} hours"
        else:
            days = minutes / 1440
            if days == int(days):
                return f"{int(days)} day{'s' if days > 1 else ''}"
            else:
                return f"{days:.1f} days"
    
    def clear(self):
        """Clear all predictions."""
        self.set_predictions({}, 0.0)
