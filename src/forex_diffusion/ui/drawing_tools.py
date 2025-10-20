"""
Drawing Tools for Chart - Persistent annotations and drawings
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QToolButton,
    QColorDialog, QDialog, QGridLayout, QButtonGroup
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from loguru import logger


@dataclass
class DrawingObject:
    """Represents a drawing on the chart"""
    type: str  # 'line', 'arrow', 'rectangle', 'triangle', 'fib', 'icon', 'freehand', 'gaussian'
    points: List[Tuple[float, float]]  # [(x1, y1), (x2, y2), ...]
    color: str = '#FF0000'  # Line color
    fill_color: str = 'transparent'  # Fill color
    width: float = 2.0
    icon_name: Optional[str] = None  # For icon type
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'type': self.type,
            'points': self.points,
            'color': self.color,
            'fill_color': self.fill_color,
            'width': self.width,
            'icon_name': self.icon_name,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DrawingObject':
        """Create from dictionary"""
        return cls(
            type=data['type'],
            points=data['points'],
            color=data.get('color', '#FF0000'),
            fill_color=data.get('fill_color', 'transparent'),
            width=data.get('width', 2.0),
            icon_name=data.get('icon_name'),
            metadata=data.get('metadata', {})
        )


class IconSelectorDialog(QDialog):
    """Dialog for selecting an icon"""

    ICONS = [
        '⬆️', '⬇️', '➡️', '⬅️',  # Arrows
        '⭐', '❤️', '💰', '⚠️',  # Symbols
        '🔴', '🟢', '🔵', '🟡',  # Colored circles
        '📈', '📉', '💹', '🎯',  # Trading
        '✅', '❌', '❓', '💡'   # Status
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Icon")
        self.selected_icon = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Grid of icons
        grid = QGridLayout()
        self.button_group = QButtonGroup(self)

        row, col = 0, 0
        for i, icon in enumerate(self.ICONS):
            btn = QPushButton(icon)
            btn.setMinimumSize(50, 50)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, ic=icon: self._on_icon_selected(ic))
            self.button_group.addButton(btn, i)
            grid.addWidget(btn, row, col)

            col += 1
            if col >= 4:
                col = 0
                row += 1

        layout.addLayout(grid)

        # OK/Cancel buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _on_icon_selected(self, icon: str):
        self.selected_icon = icon

    @classmethod
    def get_icon(cls, parent=None) -> Optional[str]:
        """Show dialog and return selected icon"""
        dialog = cls(parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_icon
        return None


class DrawingToolbar(QWidget):
    """Toolbar for drawing tools"""

    tool_selected = Signal(str)  # Emits tool name
    color_changed = Signal(str)  # Emits color
    fill_color_changed = Signal(str)  # Emits fill color
    delete_requested = Signal()  # Delete last/selected drawing

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_tool = None
        self.line_color = '#FF0000'
        self.fill_color = 'transparent'
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Drawing tools
        self.tools = {}
        tool_definitions = [
            ('line', 'Line', '📏'),
            ('arrow', 'Arrow', '➡️'),
            ('rectangle', 'Rectangle', '⬜'),
            ('triangle', 'Triangle', '🔺'),
            ('fib', 'Fibonacci', '📊'),
            ('icon', 'Icon', '⭐'),
            ('freehand', 'Freehand', '✏️'),
            ('gaussian', 'Gaussian', '🔔')
        ]

        for tool_id, tooltip, icon in tool_definitions:
            btn = QToolButton()
            btn.setText(icon)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, t=tool_id: self._on_tool_clicked(t))
            layout.addWidget(btn)
            self.tools[tool_id] = btn

        layout.addSpacing(10)

        # Line color
        self.line_color_btn = QPushButton("Line Color")
        self.line_color_btn.clicked.connect(self._select_line_color)
        self._update_color_button(self.line_color_btn, self.line_color)
        layout.addWidget(self.line_color_btn)

        # Fill color
        self.fill_color_btn = QPushButton("Fill Color")
        self.fill_color_btn.clicked.connect(self._select_fill_color)
        self._update_color_button(self.fill_color_btn, self.fill_color)
        layout.addWidget(self.fill_color_btn)

        layout.addSpacing(10)

        # Delete
        delete_btn = QPushButton("🗑️ Delete")
        delete_btn.clicked.connect(self.delete_requested.emit)
        layout.addWidget(delete_btn)

        layout.addStretch()

    def _on_tool_clicked(self, tool_id: str):
        # Deselect other tools
        for tid, btn in self.tools.items():
            if tid != tool_id:
                btn.setChecked(False)

        # Toggle current tool
        if self.current_tool == tool_id:
            self.current_tool = None
            self.tools[tool_id].setChecked(False)
        else:
            self.current_tool = tool_id
            self.tools[tool_id].setChecked(True)

        self.tool_selected.emit(self.current_tool or '')

    def _select_line_color(self):
        color = QColorDialog.getColor(QColor(self.line_color), self)
        if color.isValid():
            self.line_color = color.name()
            self._update_color_button(self.line_color_btn, self.line_color)
            self.color_changed.emit(self.line_color)

    def _select_fill_color(self):
        color = QColorDialog.getColor(QColor(self.fill_color) if self.fill_color != 'transparent' else Qt.GlobalColor.transparent, self)
        if color.isValid():
            self.fill_color = color.name()
            self._update_color_button(self.fill_color_btn, self.fill_color)
            self.fill_color_changed.emit(self.fill_color)

    def _update_color_button(self, button: QPushButton, color: str):
        """Update button background color"""
        if color == 'transparent':
            button.setStyleSheet("")
        else:
            button.setStyleSheet(f"background-color: {color};")


class DrawingManager:
    """Manages drawings on the chart"""

    def __init__(self, chart_tab):
        self.chart_tab = chart_tab
        self.drawings: List[DrawingObject] = []
        self.current_drawing: Optional[DrawingObject] = None
        self.current_tool: Optional[str] = None
        self.line_color = '#FF0000'
        self.fill_color = 'transparent'

        # File for persistence
        self.drawings_file = Path("chart_drawings.json")
        self.load_drawings()

    def set_tool(self, tool: str):
        """Set current drawing tool"""
        self.current_tool = tool
        logger.info(f"Drawing tool set to: {tool}")

    def set_line_color(self, color: str):
        self.line_color = color

    def set_fill_color(self, color: str):
        self.fill_color = color

    def start_drawing(self, x: float, y: float):
        """Start a new drawing"""
        if not self.current_tool:
            return

        # For icon type, save the selected icon
        icon_name = None
        if self.current_tool == 'icon' and hasattr(self, '_selected_icon'):
            icon_name = self._selected_icon

        self.current_drawing = DrawingObject(
            type=self.current_tool,
            points=[(x, y)],
            color=self.line_color,
            fill_color=self.fill_color,
            icon_name=icon_name
        )
        logger.info(f"Started drawing {self.current_tool} at ({x}, {y})")

    def add_point(self, x: float, y: float):
        """Add point to current drawing"""
        if self.current_drawing:
            self.current_drawing.points.append((x, y))

    def finish_drawing(self):
        """Finish current drawing and add to list"""
        if self.current_drawing:
            self.drawings.append(self.current_drawing)
            logger.info(f"Finished drawing {self.current_drawing.type} with {len(self.current_drawing.points)} points")
            self.current_drawing = None
            self.save_drawings()
            self.redraw_all()

    def delete_last(self):
        """Delete last drawing"""
        if self.drawings:
            deleted = self.drawings.pop()
            logger.info(f"Deleted drawing: {deleted.type}")
            self.save_drawings()
            self.redraw_all()

    def clear_all(self):
        """Clear all drawings"""
        count = len(self.drawings)
        self.drawings.clear()
        logger.info(f"Cleared {count} drawings")
        self.save_drawings()
        self.redraw_all()

    def redraw_all(self):
        """Redraw all drawings on the chart using PyQtGraph"""
        try:
            logger.debug(f"redraw_all called with {len(self.drawings)} drawings")

            # Check for PyQtGraph main_plot
            if not hasattr(self.chart_tab, 'main_plot') or self.chart_tab.main_plot is None:
                logger.warning("PyQtGraph main_plot not available for redrawing")
                return

            plot = self.chart_tab.main_plot
            logger.debug(f"Got PyQtGraph plot: {plot}")

            # Remove previous drawing items if they exist
            if hasattr(self, '_drawing_items'):
                logger.debug(f"Removing {len(self._drawing_items)} previous items")
                for item in self._drawing_items:
                    try:
                        plot.removeItem(item)
                    except Exception as e:
                        logger.debug(f"Failed to remove item: {e}")
            self._drawing_items = []

            # Draw each saved drawing
            for i, drawing in enumerate(self.drawings):
                logger.debug(f"Drawing object {i}: type={drawing.type}, points={len(drawing.points)}")
                items = self._draw_object_pyqtgraph(plot, drawing)
                self._drawing_items.extend(items)
                logger.debug(f"Added {len(items)} items for drawing {i}")

            logger.info(f"Redrawn {len(self.drawings)} objects with {len(self._drawing_items)} total items")
        except Exception as e:
            logger.error(f"Failed to redraw drawings: {e}", exc_info=True)

    def _draw_object(self, ax, drawing: DrawingObject) -> list:
        """Draw a single DrawingObject on the axis, return list of artists"""
        from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
        from scipy.stats import norm

        artists = []
        points = drawing.points

        if len(points) < 1:
            return artists

        try:
            if drawing.type == 'line':
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    line, = ax.plot(xs, ys, color=drawing.color, linewidth=drawing.width)
                    artists.append(line)

            elif drawing.type == 'arrow':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[-1]
                    arrow = FancyArrowPatch(
                        (x1, y1), (x2, y2),
                        color=drawing.color,
                        linewidth=drawing.width,
                        arrowstyle='-|>',
                        mutation_scale=20
                    )
                    ax.add_patch(arrow)
                    artists.append(arrow)

            elif drawing.type == 'rectangle':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[-1]
                    width = x2 - x1
                    height = y2 - y1
                    rect = Rectangle(
                        (x1, y1), width, height,
                        edgecolor=drawing.color,
                        facecolor=drawing.fill_color,
                        linewidth=drawing.width,
                        fill=(drawing.fill_color != 'transparent')
                    )
                    ax.add_patch(rect)
                    artists.append(rect)

            elif drawing.type == 'triangle':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[-1]
                    # Create triangle with base at bottom
                    triangle_points = [
                        (x1, y1),
                        (x2, y1),
                        ((x1 + x2) / 2, y2)
                    ]
                    triangle = Polygon(
                        triangle_points,
                        edgecolor=drawing.color,
                        facecolor=drawing.fill_color,
                        linewidth=drawing.width,
                        fill=(drawing.fill_color != 'transparent')
                    )
                    ax.add_patch(triangle)
                    artists.append(triangle)

            elif drawing.type == 'fib':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[-1]
                    # Fibonacci retracement levels
                    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
                    for level in fib_levels:
                        y = y1 + (y2 - y1) * level
                        line, = ax.plot([x1, x2], [y, y],
                                       color=drawing.color,
                                       linewidth=drawing.width,
                                       linestyle='--',
                                       alpha=0.7)
                        artists.append(line)
                        # Add level label
                        text = ax.text(x2, y, f'{level:.3f}',
                                      fontsize=8,
                                      color=drawing.color,
                                      verticalalignment='center')
                        artists.append(text)

            elif drawing.type == 'icon':
                if len(points) >= 1 and drawing.icon_name:
                    x, y = points[0]
                    text = ax.text(x, y, drawing.icon_name,
                                  fontsize=20,
                                  color=drawing.color,
                                  horizontalalignment='center',
                                  verticalalignment='center')
                    artists.append(text)

            elif drawing.type == 'freehand':
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    line, = ax.plot(xs, ys, color=drawing.color, linewidth=drawing.width)
                    artists.append(line)

            elif drawing.type == 'gaussian':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[-1]
                    # Draw gaussian curve
                    mean = (x1 + x2) / 2
                    std = abs(x2 - x1) / 6  # 3 sigma on each side
                    x_range = np.linspace(x1, x2, 100)
                    # Normalize gaussian to fit between y1 and y2
                    y_range = y1 + (y2 - y1) * norm.pdf(x_range, mean, std) / norm.pdf(mean, mean, std)
                    line, = ax.plot(x_range, y_range,
                                   color=drawing.color,
                                   linewidth=drawing.width)
                    artists.append(line)

        except Exception as e:
            logger.error(f"Error drawing {drawing.type}: {e}")

        return artists

    def _draw_object_pyqtgraph(self, plot, drawing: DrawingObject) -> list:
        """Draw a single DrawingObject using PyQtGraph, return list of items"""
        import pyqtgraph as pg
        from PySide6.QtGui import QColor, QPen, QBrush

        items = []
        points = drawing.points

        if len(points) < 1:
            return items

        try:
            # Convert color string to QColor
            color = QColor(drawing.color)
            pen = QPen(color)
            pen.setWidth(int(drawing.width))

            if drawing.type == 'line':
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    line = pg.PlotCurveItem(x=xs, y=ys, pen=pen)
                    plot.addItem(line)
                    items.append(line)

            elif drawing.type == 'arrow':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[-1]
                    # Draw line
                    line = pg.PlotCurveItem(x=[x1, x2], y=[y1, y2], pen=pen)
                    plot.addItem(line)
                    items.append(line)
                    # Draw arrowhead (simplified triangle)
                    dx = x2 - x1
                    dy = y2 - y1
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        # Arrow size proportional to line length
                        arrow_size = min(length * 0.1, 50)
                        angle = np.arctan2(dy, dx)
                        # Arrow tip points
                        arrow_x = [x2,
                                   x2 - arrow_size * np.cos(angle - np.pi/6),
                                   x2 - arrow_size * np.cos(angle + np.pi/6)]
                        arrow_y = [y2,
                                   y2 - arrow_size * np.sin(angle - np.pi/6),
                                   y2 - arrow_size * np.sin(angle + np.pi/6)]
                        arrow_head = pg.PlotCurveItem(x=arrow_x + [arrow_x[0]],
                                                     y=arrow_y + [arrow_y[0]],
                                                     pen=pen,
                                                     brush=color)
                        plot.addItem(arrow_head)
                        items.append(arrow_head)

            elif drawing.type == 'rectangle':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[-1]
                    # Create rectangle outline
                    rect_x = [x1, x2, x2, x1, x1]
                    rect_y = [y1, y1, y2, y2, y1]

                    # Brush for fill
                    brush = None
                    if drawing.fill_color != 'transparent':
                        fill_color = QColor(drawing.fill_color)
                        brush = QBrush(fill_color)

                    rect = pg.PlotCurveItem(x=rect_x, y=rect_y, pen=pen,
                                           brush=brush, fillLevel=None)
                    plot.addItem(rect)
                    items.append(rect)

            elif drawing.type == 'triangle':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[-1]
                    # Create triangle with base at bottom
                    tri_x = [x1, x2, (x1 + x2) / 2, x1]
                    tri_y = [y1, y1, y2, y1]

                    brush = None
                    if drawing.fill_color != 'transparent':
                        fill_color = QColor(drawing.fill_color)
                        brush = QBrush(fill_color)

                    triangle = pg.PlotCurveItem(x=tri_x, y=tri_y, pen=pen,
                                               brush=brush, fillLevel=None)
                    plot.addItem(triangle)
                    items.append(triangle)

            elif drawing.type == 'fib':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[-1]
                    # Fibonacci levels
                    fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
                    y_range = y2 - y1

                    for level in fib_levels:
                        y = y1 + y_range * level
                        fib_line = pg.PlotCurveItem(x=[x1, x2], y=[y, y], pen=pen)
                        plot.addItem(fib_line)
                        items.append(fib_line)

                        # Add text label
                        label = pg.TextItem(text=f"{level:.3f}", anchor=(0, 0.5), color=color)
                        label.setPos(x1, y)
                        plot.addItem(label)
                        items.append(label)

            elif drawing.type == 'icon':
                if len(points) >= 1:
                    x, y = points[0]
                    icon_text = drawing.icon_name if drawing.icon_name else '⭐'
                    text_item = pg.TextItem(text=icon_text, anchor=(0.5, 0.5), color=color)
                    text_item.setPos(x, y)
                    plot.addItem(text_item)
                    items.append(text_item)

            elif drawing.type == 'freehand':
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    freehand = pg.PlotCurveItem(x=xs, y=ys, pen=pen)
                    plot.addItem(freehand)
                    items.append(freehand)

            elif drawing.type == 'gaussian':
                if len(points) >= 2:
                    from scipy.stats import norm
                    x1, y1 = points[0]
                    x2, y2 = points[-1]
                    # Center and width
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    std_x = abs(x2 - x1) / 4
                    std_y = abs(y2 - y1) / 4

                    # Generate gaussian curve
                    x_range = np.linspace(x1, x2, 100)
                    y_range = center_y + std_y * norm.pdf((x_range - center_x) / std_x)

                    gaussian = pg.PlotCurveItem(x=x_range, y=y_range, pen=pen)
                    plot.addItem(gaussian)
                    items.append(gaussian)

        except Exception as e:
            logger.error(f"Error drawing {drawing.type} with PyQtGraph: {e}")

        return items

    def save_drawings(self):
        """Save drawings to file"""
        try:
            data = [d.to_dict() for d in self.drawings]
            with open(self.drawings_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.drawings)} drawings to {self.drawings_file}")
        except Exception as e:
            logger.error(f"Failed to save drawings: {e}")

    def load_drawings(self):
        """Load drawings from file"""
        try:
            if self.drawings_file.exists():
                with open(self.drawings_file, 'r') as f:
                    data = json.load(f)
                self.drawings = [DrawingObject.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.drawings)} drawings from {self.drawings_file}")
                self.redraw_all()
        except Exception as e:
            logger.error(f"Failed to load drawings: {e}")
