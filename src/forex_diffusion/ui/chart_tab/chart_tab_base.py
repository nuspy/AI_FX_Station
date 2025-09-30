"""
Base ChartTab class with core initialization and coordinate methods.
"""
from __future__ import annotations

from typing import Optional, Dict, List
import pandas as pd
from PySide6.QtWidgets import QWidget, QLabel
from PySide6.QtCore import Signal, Qt
from loguru import logger

from ...utils.user_settings import get_setting, set_setting
from ...services.brokers import get_broker_service
from ..chart_components.controllers.chart_controller import ChartTabController

# Mixins for different functionality areas
from .ui_builder import UIBuilderMixin
from .event_handlers import EventHandlersMixin
from .controller_proxy import ControllerProxyMixin
from .patterns_mixin import PatternsMixin
from .overlay_manager import OverlayManagerMixin


class DraggableOverlay(QLabel):
    """Small draggable label overlay used for legend and cursor values."""
    dragStarted = Signal()
    dragEnded = Signal()

    def __init__(self, text: str, parent: QWidget):
        super().__init__(parent)
        self.setText(text)
        self.setStyleSheet("QLabel { background: rgba(0,0,0,160); color: white; border: 1px solid rgba(255,255,255,80); border-radius: 4px; padding: 4px; }")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self._dragging = False
        self._drag_offset = None

    def mousePressEvent(self, event):
        if event and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_offset = event.pos()
            self.dragStarted.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_offset is not None:
            # Move the label keeping the offset
            new_pos = self.mapToParent(event.pos() - self._drag_offset)
            # Keep inside parent bounds
            parent = self.parentWidget()
            if parent:
                x = max(0, min(new_pos.x(), parent.width() - self.width()))
                y = max(0, min(new_pos.y(), parent.height() - self.height()))
                self.move(x, y)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event and event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._drag_offset = None
            self.dragEnded.emit()
        super().mouseReleaseEvent(event)


class ChartTabUI(
    UIBuilderMixin,
    EventHandlersMixin,
    ControllerProxyMixin,
    PatternsMixin,
    OverlayManagerMixin,
    QWidget
):
    """
    The primary chart tab, built entirely programmatically for stability and maintainability.

    This class coordinates multiple mixins that handle different aspects:
    - UIBuilderMixin: UI construction and layout
    - EventHandlersMixin: Event handling and callbacks
    - ControllerProxyMixin: Controller delegation methods
    - PatternsMixin: Pattern detection integration
    - OverlayManagerMixin: Overlay and drawing management
    """
    forecastRequested = Signal(dict)
    tickArrived = Signal(dict)

    def __init__(self, parent=None, viewer=None):
        super().__init__(parent)
        self.setObjectName("chartTabUI")
        self._main_window = parent

        # Core controller setup
        self.controller = getattr(parent, "controller", None)
        if self.controller and hasattr(self.controller, "signals"):
            self.controller.signals.forecastReady.connect(self.on_forecast_ready)

        # Initialize core attributes
        self._drawing_mode: Optional[str] = None
        self._price_mode = "candles"
        self.chart_controller = ChartTabController(self, self.controller)

        # Initialize state variables
        self._initialize_state()

        # Build UI and setup components
        self._build_ui()
        self._setup_chart_components()
        self._initialize_timers_and_connections()

    def _initialize_state(self):
        """Initialize core state variables."""
        # Chart state
        self._last_df = pd.DataFrame()
        self._forecasts: List[Dict] = []
        self.max_forecasts = int(get_setting("max_forecasts", 20))
        self._legend_once = set()

        # Services
        self.broker = get_broker_service()

        # Cache state
        self._current_cache_tf = None
        self._current_cache_range = None

        # Mouse interaction state
        self._rbtn_drag = False
        self._drag_last = None
        self._drag_axis = None

        # Overlay state
        self._overlay_dragging = False
        self._suppress_line_update = False
        self._x_cache_comp = None

        # Pattern state
        self._pattern_artists = []
        self._pattern_cache = {}
        self._last_patterns_scan = 0

    def _setup_chart_components(self):
        """Setup chart components after UI is built."""
        # Check if using finplot or matplotlib
        if hasattr(self, 'use_finplot') and self.use_finplot:
            # Finplot initialization
            # Axes will be created dynamically when plotting
            self.ax = None  # Will be set when first plot is created
            self._osc_ax = None
            self._ind_artists = {}
        else:
            # Matplotlib initialization
            self.ax = self.canvas.figure.subplots()
            self._osc_ax = None
            self._ind_artists = {}

            # Configure figure layout
            self.canvas.figure.set_constrained_layout(False)
            self.canvas.figure.subplots_adjust(left=0.04, right=0.995, top=0.97, bottom=0.08)
            self.ax.margins(x=0.001, y=0.05)

            # Connect xlim change callback
            self._xlim_cid = self.ax.callbacks.connect('xlim_changed', self._on_main_xlim_changed)

    def _initialize_timers_and_connections(self):
        """Setup timers and UI connections."""
        self._init_control_defaults()
        self._connect_ui_signals()
        self._setup_timers()
        self._connect_mouse_events()

        # Apply theme
        if theme_combo := getattr(self, "theme_combo", None):
            self._apply_theme(theme_combo.currentText())

        # Connect main signals
        self.tickArrived.connect(self.chart_controller.on_tick_main)

        # Initialize overlays and grid
        self._init_overlays()
        self._apply_grid_style()

    # Properties for accessing key components
    @property
    def symbol(self) -> str:
        """Current trading symbol."""
        return getattr(self, "_symbol", get_setting("chart.symbol", "EURUSD"))

    @symbol.setter
    def symbol(self, value: str):
        self._symbol = value

    @property
    def timeframe(self) -> str:
        """Current timeframe."""
        return getattr(self, "_timeframe", get_setting("chart.timeframe", "1h"))

    @timeframe.setter
    def timeframe(self, value: str):
        self._timeframe = value

    @property
    def follow_enabled(self) -> bool:
        """Whether chart following is enabled."""
        return getattr(self, "_follow_enabled", get_setting("chart.follow", True))

    @follow_enabled.setter
    def follow_enabled(self, value: bool):
        self._follow_enabled = value