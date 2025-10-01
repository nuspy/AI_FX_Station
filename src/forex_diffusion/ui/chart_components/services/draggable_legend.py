"""
Draggable legend overlay for forecast models.
Positioned relative to screen, not graph coordinates.
"""
from __future__ import annotations

from typing import Dict, Optional
from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QColor, QPainter, QFont, QPen

from forex_diffusion.utils.user_settings import get_setting, set_setting


class DraggableLegend(QFrame):
    """
    Draggable overlay legend for forecast models.
    Shows model names with their assigned colors.
    Position is saved between sessions.
    """

    def __init__(self, parent: QWidget, legend_type: str = "forecast"):
        """
        Args:
            parent: Parent widget (typically the plot widget)
            legend_type: Type of legend ("forecast" or "indicator")
        """
        super().__init__(parent)
        self.legend_type = legend_type
        self.dragging = False
        self.drag_offset = QPoint()

        # Style
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 220);
                border: 1px solid #888;
                border-radius: 4px;
                padding: 5px;
            }
        """)

        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 6, 8, 6)
        self.layout.setSpacing(4)

        # Title
        self.title_label = QLabel("Models" if legend_type == "forecast" else "Indicators")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(9)
        self.title_label.setFont(title_font)
        self.layout.addWidget(self.title_label)

        # Model entries container
        self.entries_widget = QWidget()
        self.entries_layout = QVBoxLayout(self.entries_widget)
        self.entries_layout.setContentsMargins(0, 0, 0, 0)
        self.entries_layout.setSpacing(3)
        self.layout.addWidget(self.entries_widget)

        # Store model entries: {model_path: (label_widget, color)}
        self.model_entries: Dict[str, tuple] = {}

        # Load saved position or use default (top-right)
        self._load_position()

        # Make always on top
        self.raise_()

    def add_model(self, model_path: str, model_name: str, color: str):
        """
        Add or update a model in the legend.

        Args:
            model_path: Full path to model (used as unique identifier)
            model_name: Display name for the model
            color: Hex color string (e.g., "#2196F3")
        """
        if model_path in self.model_entries:
            # Update existing entry
            label, _ = self.model_entries[model_path]
            label.setText(model_name)
            self.model_entries[model_path] = (label, color)
            label.update()  # Trigger repaint with new color
        else:
            # Create new entry
            label = ModelLegendEntry(model_name, color)
            self.entries_layout.addWidget(label)
            self.model_entries[model_path] = (label, color)

        self.adjustSize()
        self.update()

    def remove_model(self, model_path: str):
        """Remove a model from the legend."""
        if model_path in self.model_entries:
            label, _ = self.model_entries[model_path]
            self.entries_layout.removeWidget(label)
            label.deleteLater()
            del self.model_entries[model_path]
            self.adjustSize()
            self.update()

    def clear_all(self):
        """Clear all model entries."""
        for model_path in list(self.model_entries.keys()):
            self.remove_model(model_path)

    def mousePressEvent(self, event):
        """Start dragging."""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_offset = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Handle dragging."""
        if self.dragging:
            # Move relative to parent
            new_pos = self.mapToParent(event.pos() - self.drag_offset)
            self.move(new_pos)

    def mouseReleaseEvent(self, event):
        """Stop dragging and save position."""
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
            self._save_position()

    def _load_position(self):
        """Load saved position from settings."""
        setting_key = f'legend_{self.legend_type}_position'
        saved_pos = get_setting(setting_key)

        if saved_pos and isinstance(saved_pos, dict):
            x = saved_pos.get('x', 0)
            y = saved_pos.get('y', 0)
            # Validate position is within parent bounds
            if self.parent():
                parent_width = self.parent().width()
                parent_height = self.parent().height()
                x = max(0, min(x, parent_width - 100))
                y = max(0, min(y, parent_height - 50))
            self.move(x, y)
        else:
            # Default position: top-right corner with 10px margin
            if self.parent():
                parent_width = self.parent().width()
                self.move(parent_width - 210, 10)
            else:
                self.move(10, 10)

    def _save_position(self):
        """Save current position to settings."""
        setting_key = f'legend_{self.legend_type}_position'
        set_setting(setting_key, {
            'x': self.x(),
            'y': self.y()
        })

    def showEvent(self, event):
        """When shown, ensure it's on top and correctly positioned."""
        super().showEvent(event)
        self.raise_()
        self.adjustSize()


class ModelLegendEntry(QWidget):
    """
    Single entry in the legend showing color box and model name.
    """

    def __init__(self, model_name: str, color: str):
        super().__init__()
        self.model_name = model_name
        self.color = color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(f"  {model_name}")
        font = QFont()
        font.setPointSize(8)
        self.label.setFont(font)
        layout.addWidget(self.label)

        self.setFixedHeight(20)

    def paintEvent(self, event):
        """Draw color box before label."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw color box
        color = QColor(self.color)
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#888"), 1))
        painter.drawRect(2, 4, 12, 12)

        painter.end()

    def setText(self, text: str):
        """Update model name."""
        self.model_name = text
        self.label.setText(f"  {text}")
