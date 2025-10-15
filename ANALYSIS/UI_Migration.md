# Chart Tab UI Migration Analysis
**Project:** ForexGPT  
**Date:** 2025-10-08  
**Status:** CRITICAL - Architecture Misalignment Detected

## Executive Summary

During recent Claude Code sessions, modifications were incorrectly applied to **File_A** (`chart_tab.py`), an abandoned monolithic implementation, when they should have been applied to **File_B** (the modular `chart_tab/` directory structure). This document provides a component-by-component analysis to migrate missing functionality from File_A to the proper File_B architecture.

---

## File Structure Overview

### File_A (Abandoned - DO NOT MODIFY)
```
src/forex_diffusion/ui/chart_tab.py
├── ChartTab (class) - 2,500+ lines monolithic implementation
└── ChartTabUI (class) - UI loader wrapper
```

### File_B (Current Architecture - TARGET FOR MIGRATION)
```
src/forex_diffusion/ui/chart_tab/
├── __init__.py - Package exports
├── chart_tab_base.py - Core ChartTabUI class with mixins
├── ui_builder.py - UI construction methods
├── event_handlers.py - Event handling methods  
├── controller_proxy.py - Controller delegation methods
├── patterns_mixin.py - Pattern detection integration
└── overlay_manager.py - Overlay and drawing management

Supporting Architecture:
src/forex_diffusion/ui/chart_components/
├── controllers/
│   └── chart_controller.py - Main controller
└── services/
    ├── action_service.py
    ├── data_service.py
    ├── forecast_service.py
    ├── interaction_service.py
    ├── plot_service.py
    ├── patterns/
    │   ├── patterns_service.py
    │   ├── detection_worker.py
    │   └── historical_scan_worker.py
    └── ... (other services)
```

---

## Architecture Comparison

### File_A Architecture (OLD - Monolithic)
- **Single Class:** All functionality in one 2,500+ line ChartTab class
- **No Separation:** UI, logic, services, and state management mixed
- **Matplotlib-based:** Uses matplotlib for charting
- **Direct DB Access:** Methods directly query database
- **Tight Coupling:** Hard to test, maintain, or extend

### File_B Architecture (NEW - Modular)
- **Mixin-Based:** Functionality separated into focused mixins
- **Service Layer:** Dedicated services for data, plotting, patterns, forecasts
- **Controller Pattern:** ChartTabController delegates to services
- **Finplot/PyQtGraph:** High-performance charting library
- **Loose Coupling:** Easy to test, maintain, and extend

---

## Component-by-Component Migration Analysis

### ✅ **1. CORE INITIALIZATION**

#### File_A Implementation
```python
class ChartTab(QWidget):
    def __init__(self, parent=None, viewer=None):
        super().__init__(parent)
        self.setObjectName("chartTab")
        self._main_window = parent
        self.controller = getattr(parent, "controller", None)
        # ... extensive initialization
```

#### File_B Status
✅ **MIGRATED** - Implemented in `chart_tab_base.py`
- Core initialization in `ChartTabUI.__init__()`
- State initialization in `_initialize_state()`
- Component setup in `_setup_chart_components()`

**No Action Required**

---

### ✅ **2. UI CONSTRUCTION**

#### File_A Implementation
```python
def _build_ui(self) -> None:
    # Programmatic UI construction
    # - Creates layouts
    # - Adds matplotlib canvas
    # - Builds toolbar
    # - Creates splitters
    # - Sets up market watch
    # - Creates orders table
```

#### File_B Status
✅ **MIGRATED** - Implemented in `ui_builder.py`
- `UIBuilderMixin._build_ui()` - Main UI construction
- `_create_chart_tab()` - Chart area with finplot
- `_create_drawbar()` - Drawing toolbar
- `_populate_topbar()` - Control bars

**No Action Required**

---

### ⚠️ **3. MATPLOTLIB CANVAS & TOOLBAR**

#### File_A Implementation
```python
self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
self.toolbar = NavigationToolbar(self.canvas, self)
self.ax = self.canvas.figure.subplots()
```

#### File_B Status
❌ **NEEDS MIGRATION** - File_B uses finplot/PyQtGraph instead

**Migration Strategy:**
1. File_B already uses PyQtGraph (higher performance)
2. Keep matplotlib references for compatibility
3. Add fallback support in `ui_builder.py`:

```python
# In ui_builder.py - add matplotlib compatibility layer
def _ensure_matplotlib_canvas(self):
    """Create matplotlib canvas for legacy indicator plotting."""
    if not hasattr(self, 'mpl_canvas'):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        
        self.mpl_canvas = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
        self.mpl_ax = self.mpl_canvas.figure.subplots()
```

**Target File:** `ui_builder.py`  
**Service:** None (UI layer only)

---

### ⚠️ **4. CONTROL DEFAULTS & SETTINGS**

#### File_A Implementation
```python
def _init_control_defaults(self) -> None:
    self._symbols_supported = ["EUR/USD", "GBP/USD", ...]
    self._symbol_row_map: Dict[str, int] = {}
    self._spread_state: Dict[str, Dict[str, float]] = {}
    self._follow_suspend_seconds = float(get_setting(...))
    # ... extensive combo box population
```

#### File_B Status
⚠️ **PARTIALLY MIGRATED** - Basic initialization exists, but missing:
- `_symbol_row_map` dictionary
- `_spread_state` tracking
- `_follow_suspend_seconds` logic
- Full combo box population

**Migration Strategy:**
```python
# In chart_tab_base.py - add to _initialize_state()
def _initialize_state(self):
    # ... existing code ...
    
    # Add missing state tracking
    self._symbols_supported = get_setting('chart.symbols', 
        ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "GBPNZD", "AUDJPY", "GBPEUR", "GBPAUD"])
    self._symbol_row_map: Dict[str, int] = {}
    self._spread_state: Dict[str, Dict[str, float]] = {}
    self._last_bidask: Dict[str, Dict[str, float]] = {}
    
    # Follow behavior settings
    self._follow_suspend_seconds = float(get_setting('chart.follow_suspend_seconds', 30))
    self._follow_enabled = bool(get_setting('chart.follow_enabled', False))
    self._follow_suspend_until = 0.0
```

**Target File:** `chart_tab_base.py` (in `_initialize_state()`)  
**Service:** None (state management)

---

### ⚠️ **5. SIGNAL CONNECTIONS**

#### File_A Implementation
```python
def _connect_ui_signals(self) -> None:
    # Symbol/timeframe changes
    self.symbol_combo.currentTextChanged.connect(self._on_symbol_combo_changed)
    self.tf_combo.currentTextChanged.connect(self._on_timeframe_changed)
    
    # Button clicks
    self.forecast_btn.clicked.connect(self.chart_controller.on_forecast_clicked)
    self.indicators_btn.clicked.connect(self.chart_controller.on_indicators_clicked)
    
    # Pattern toggles
    self.chart_patterns_checkbox.toggled.connect(...)
    
    # Splitter movements
    splitter.splitterMoved.connect(...)
```

#### File_B Status
⚠️ **PARTIALLY MIGRATED** - Some connections exist in `event_handlers.py`, but missing:
- Pattern checkbox toggles
- Splitter persistence
- Some button connections

**Migration Strategy:**
```python
# In event_handlers.py - add to EventHandlersMixin
def _connect_ui_signals(self):
    """Connect all UI signals to their handlers."""
    # ... existing connections ...
    
    # Pattern toggle connections (if checkboxes exist)
    if hasattr(self, 'chart_patterns_checkbox'):
        self.chart_patterns_checkbox.toggled.connect(
            lambda v: self.chart_controller.patterns_service.set_chart_enabled(v)
        )
    
    if hasattr(self, 'candle_patterns_checkbox'):
        self.candle_patterns_checkbox.toggled.connect(
            lambda v: self.chart_controller.patterns_service.set_candle_enabled(v)
        )
    
    if hasattr(self, 'history_patterns_checkbox'):
        self.history_patterns_checkbox.toggled.connect(
            lambda v: self.chart_controller.patterns_service.set_history_enabled(v)
        )
    
    # Splitter persistence
    for key, splitter in [
        ('chart.splitter.main', getattr(self, 'main_splitter', None)),
        ('chart.splitter.right', getattr(self, 'right_splitter', None)),
        ('chart.splitter.chart', getattr(self, '_chart_area', None)),
    ]:
        if splitter:
            splitter.splitterMoved.connect(
                lambda pos, idx, k=key, s=splitter: self._persist_splitter_positions(k, s)
            )
```

**Target File:** `event_handlers.py`  
**Helper Methods:** Add `_persist_splitter_positions()` to event_handlers.py

---

### ⚠️ **6. MARKET WATCH TABLE**

#### File_A Implementation
```python
def _update_market_quote(self, symbol: str, bid: float, ask: float, ts_ms: int):
    # Updates market watch table with bid/ask
    # Color codes based on spread changes
    # Tracks spread state over time
```

#### File_B Status
❌ **NOT MIGRATED** - Market watch exists in UI but no update logic

**Migration Strategy:**
Create dedicated service for market watch updates:

```python
# Create new file: chart_components/services/market_watch_service.py
class MarketWatchService:
    """Service for managing market watch table updates."""
    
    def __init__(self, table_widget):
        self.table = table_widget
        self.symbol_row_map = {}
        self.spread_state = {}
        self.last_bidask = {}
    
    def update_quote(self, symbol: str, bid: float, ask: float, ts_ms: int):
        """Update bid/ask for a symbol with color coding."""
        # Implement spread tracking and coloring logic from File_A
        pass
    
    def add_symbol(self, symbol: str):
        """Add a new symbol to the market watch."""
        pass
```

Then connect in `chart_tab_base.py`:
```python
def _initialize_state(self):
    # ... existing code ...
    from ..chart_components.services.market_watch_service import MarketWatchService
    self.market_watch_service = MarketWatchService(self.market_watch)
```

**Target File:** NEW - `chart_components/services/market_watch_service.py`  
**Integration:** `chart_tab_base.py`, `event_handlers.py`

---

### ❌ **7. HOVER LEGEND / CURSOR INFO**

#### File_A Implementation
```python
def _ensure_hover_legend(self) -> None:
    # Creates draggable legend for cursor position
    # Shows Time/Bid/Ask at cursor location
    
def _update_hover_info(self, event) -> None:
    # Updates legend on mouse move
    
def _reset_hover_info(self) -> None:
    # Resets legend when mouse leaves
```

#### File_B Status
❌ **NOT MIGRATED** - This is a key UX feature that's completely missing

**Migration Strategy:**
Add to `overlay_manager.py`:

```python
# In overlay_manager.py - OverlayManagerMixin
def _ensure_hover_legend(self):
    """Ensure hover legend overlay exists."""
    if hasattr(self, '_hover_overlay') and self._hover_overlay:
        return
    
    from .chart_tab_base import DraggableOverlay
    self._hover_overlay = DraggableOverlay("Time: -\nBid: -\nAsk: -", self.chart_container)
    self._hover_overlay.setVisible(True)
    self._hover_overlay.move(10, 10)  # Initial position

def _update_hover_info(self, event):
    """Update hover legend with current cursor data."""
    if not hasattr(self, '_hover_overlay'):
        self._ensure_hover_legend()
    
    # Extract data from event and format
    # Update overlay text
    pass

def _reset_hover_info(self):
    """Reset hover legend to default state."""
    if hasattr(self, '_hover_overlay'):
        self._hover_overlay.setText("Time: -\nBid: -\nAsk: -")
```

Connect in mouse event handlers:
```python
# In event_handlers.py
def _on_mouse_move(self, event):
    # ... existing code ...
    self._update_hover_info(event)
    
def _on_figure_leave(self, event):
    # ... existing code ...
    self._reset_hover_info()
```

**Target File:** `overlay_manager.py`  
**Integration:** `event_handlers.py` (mouse events)

---

### ⚠️ **8. FOLLOW MODE / AUTO-CENTER**

#### File_A Implementation
```python
def _on_follow_toggled(self, checked: bool):
    # Enable/disable auto-centering on latest data
    
def _suspend_follow(self):
    # Temporarily suspend follow on user interaction
    
def _follow_center_if_needed(self):
    # Center chart on latest price if follow enabled
```

#### File_B Status
⚠️ **PARTIALLY EXISTS** - Basic follow logic may exist, needs enhancement

**Migration Strategy:**
```python
# In event_handlers.py - EventHandlersMixin
def _on_follow_toggled(self, checked: bool):
    """Handle follow mode toggle."""
    self._follow_enabled = checked
    set_setting('chart.follow_enabled', checked)
    if checked:
        self._follow_suspend_until = 0.0
        self._follow_center_if_needed()

def _suspend_follow(self):
    """Temporarily suspend follow mode on user interaction."""
    if self._follow_enabled:
        duration = self._follow_suspend_seconds
        self._follow_suspend_until = time.time() + duration

def _follow_center_if_needed(self):
    """Center chart on latest data if follow mode active."""
    import time
    if not self._follow_enabled:
        return
    if time.time() < self._follow_suspend_until:
        return
    
    # Get latest data point and center view
    if self._last_df is not None and not self._last_df.empty:
        # Center logic here
        pass
```

Integrate suspend calls in interaction handlers:
```python
# In event_handlers.py - all zoom/pan handlers
def _on_mouse_press(self, event):
    if event.button in (1, 3):
        self._suspend_follow()
    # ... rest of handler
```

**Target File:** `event_handlers.py`  
**Service:** None (view management)

---

### ⚠️ **9. PRICE MODE TOGGLE (Candles vs Line)**

#### File_A Implementation
```python
def _on_price_mode_toggled(self, checked: bool):
    self._price_mode = 'candles' if checked else 'line'
    # Updates plot mode
    
def _render_candles(self, df2: pd.DataFrame):
    # Renders OHLC candles
```

#### File_B Status
❌ **NOT MIGRATED** - Finplot handles this differently

**Migration Strategy:**
Delegate to plot service:

```python
# In chart_components/services/plot_service.py
class PlotService:
    # ... existing methods ...
    
    def set_price_mode(self, mode: str):
        """Set chart display mode: 'candles' or 'line'."""
        self.price_mode = mode
        self._redraw_chart()
    
    def _redraw_chart(self):
        """Redraw chart with current mode."""
        if self.price_mode == 'candles':
            self._plot_candles()
        else:
            self._plot_line()
```

Connect in event handlers:
```python
# In event_handlers.py
def _on_mode_toggled(self, checked: bool):
    """Handle candles/line mode toggle."""
    mode = 'candles' if checked else 'line'
    self._price_mode = mode
    set_setting('chart.price_mode', mode)
    self.chart_controller.plot_service.set_price_mode(mode)
```

**Target File:** `chart_components/services/plot_service.py`  
**Integration:** `event_handlers.py`

---

### ⚠️ **10. THEME APPLICATION**

#### File_A Implementation
```python
def _apply_theme(self, theme: str):
    # Applies color scheme to chart
    # Updates matplotlib colors
    
def _get_color(self, key: str, default: str) -> str:
    # Gets theme color from settings
```

#### File_B Status
⚠️ **PARTIALLY MIGRATED** - Basic theme support exists

**Migration Strategy:**
Enhance theme support in plot service:

```python
# In chart_components/services/plot_service.py
def apply_theme(self, theme_name: str):
    """Apply color theme to chart."""
    theme_colors = self._load_theme(theme_name)
    
    # Apply to finplot/pyqtgraph
    if hasattr(self, 'ax') and self.ax:
        # Set colors
        pass
    
    self._redraw_chart()

def _get_color(self, key: str, default: str) -> str:
    """Get color from current theme."""
    return get_setting(f'theme.{self.current_theme}.{key}', default)
```

**Target File:** `chart_components/services/plot_service.py`  
**Service:** PlotService

---

### ❌ **11. MOUSE ZOOM & PAN (Advanced UX)**

#### File_A Implementation
```python
def _on_scroll_zoom(self, event):
    # Wheel zoom centered on cursor
    
def _on_mouse_press(self, event):
    # Left: pan, Right: directional zoom
    
def _on_mouse_move(self, event):
    # Handle dragging for pan/zoom
    
def _on_mouse_release(self, event):
    # End pan/zoom operation
    
def _zoom_axis(self, axis: str, center: float, factor: float):
    # Helper for zooming
```

#### File_B Status
❌ **NOT MIGRATED** - Critical UX feature missing

**Migration Strategy:**
Create dedicated interaction service:

```python
# Create new file: chart_components/services/interaction_service.py
class ChartInteractionService:
    """Service for handling mouse interactions on chart."""
    
    def __init__(self, chart_widget, ax):
        self.chart = chart_widget
        self.ax = ax
        self._pan_state = {'active': False, 'start': None}
        self._zoom_state = {'active': False, 'axis': None}
    
    def on_scroll_zoom(self, event):
        """Handle scroll wheel zoom centered on cursor."""
        if event is None or event.xdata is None:
            return
        
        # Zoom factor based on scroll direction
        factor = 0.9 if event.button == 'up' else 1.1
        
        # Zoom both axes centered on cursor
        self._zoom_axis('x', event.xdata, factor)
        self._zoom_axis('y', event.ydata, factor)
    
    def on_mouse_press(self, event):
        """Handle mouse press for pan/zoom initiation."""
        if event.button == 1:  # Left - pan
            self._pan_state['active'] = True
            self._pan_state['start'] = (event.xdata, event.ydata)
        elif event.button == 3:  # Right - zoom
            self._zoom_state['active'] = True
            self._zoom_state['start'] = (event.x, event.y)
    
    def on_mouse_move(self, event):
        """Handle mouse drag for pan/zoom."""
        if self._pan_state['active']:
            self._handle_pan(event)
        elif self._zoom_state['active']:
            self._handle_zoom_drag(event)
    
    def on_mouse_release(self, event):
        """Handle mouse release - end interaction."""
        self._pan_state['active'] = False
        self._zoom_state['active'] = False
    
    def _zoom_axis(self, axis: str, center: float, factor: float):
        """Zoom an axis around a center point."""
        # Implementation
        pass
    
    def _handle_pan(self, event):
        """Handle panning motion."""
        pass
    
    def _handle_zoom_drag(self, event):
        """Handle zoom drag motion."""
        pass
```

Integrate in chart_tab:
```python
# In chart_tab_base.py - _setup_chart_components()
from ..chart_components.services.interaction_service import ChartInteractionService
self.interaction_service = ChartInteractionService(self, self.ax)

# Connect mouse events in event_handlers.py
def _connect_mouse_events(self):
    """Connect matplotlib mouse events."""
    if hasattr(self, 'canvas') and self.canvas:
        self.canvas.mpl_connect('scroll_event', self.interaction_service.on_scroll_zoom)
        self.canvas.mpl_connect('button_press_event', self.interaction_service.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.interaction_service.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.interaction_service.on_mouse_release)
```

**Target File:** NEW - `chart_components/services/interaction_service.py`  
**Integration:** `chart_tab_base.py`, `event_handlers.py`

---

### ⚠️ **12. DYNAMIC DATA LOADING / VIEW WINDOW**

#### File_A Implementation
```python
def _schedule_view_reload(self):
    # Throttle view window reload
    
def _reload_view_window(self):
    # Load only visible data + buffer
    
def _resolution_for_span(self, ms_span: int) -> str:
    # Pick best timeframe for visible span
```

#### File_B Status
❌ **NOT MIGRATED** - This is an optimization feature

**Migration Strategy:**
Add to data service:

```python
# In chart_components/services/data_service.py
class DataService:
    # ... existing methods ...
    
    def schedule_view_reload(self):
        """Throttle view window reload to avoid excessive queries."""
        if not hasattr(self, '_reload_timer'):
            self._reload_timer = QTimer()
            self._reload_timer.setSingleShot(True)
            self._reload_timer.setInterval(250)
            self._reload_timer.timeout.connect(self.reload_view_window)
        
        self._reload_timer.start()
    
    def reload_view_window(self):
        """Load only data covering visible view + buffer."""
        if not self.chart.ax:
            return
        
        xlim = self.chart.ax.get_xlim()
        # Convert xlim to timestamp range
        # Determine best resolution
        # Load data
        pass
    
    def resolution_for_span(self, ms_span: int) -> str:
        """Pick best timeframe for visible time span."""
        # Resolution selection logic
        if ms_span < 3600000:  # < 1 hour
            return '1m'
        elif ms_span < 86400000:  # < 1 day
            return '5m'
        elif ms_span < 604800000:  # < 1 week
            return '1h'
        else:
            return '1d'
```

**Target File:** `chart_components/services/data_service.py`  
**Service:** DataService

---

### ⚠️ **13. INDICATOR PLOTTING**

#### File_A Implementation
```python
def _ensure_osc_axis(self, need: bool):
    # Create/remove oscillator subplot
    
def _plot_indicators(self, df2: pd.DataFrame, x_dt: pd.Series):
    # Plot SMA, EMA, Bollinger, RSI, MACD, etc.
    
def _sma(self, x: pd.Series, n: int) -> pd.Series:
def _ema(self, x: pd.Series, n: int) -> pd.Series:
def _bollinger(self, x: pd.Series, n: int, k: float):
# ... many indicator calculation methods
```

#### File_B Status
⚠️ **PARTIALLY EXISTS** - Some indicator support, needs enhancement

**Migration Strategy:**
Indicators should be in a dedicated service:

```python
# Check if chart_components/services/indicators_service.py exists
# If not, create it with calculation methods

class IndicatorsService:
    """Service for indicator calculations and plotting."""
    
    def __init__(self, chart_widget):
        self.chart = chart_widget
        self._ind_artists = {}
        self._osc_ax = None
    
    def ensure_osc_axis(self, needed: bool):
        """Create or remove oscillator subplot."""
        # Implementation
        pass
    
    def plot_indicators(self, df: pd.DataFrame, settings: dict):
        """Plot all enabled indicators."""
        # Implementation
        pass
    
    # Calculation methods
    @staticmethod
    def sma(x: pd.Series, n: int) -> pd.Series:
        """Simple Moving Average."""
        return x.rolling(n).mean()
    
    @staticmethod
    def ema(x: pd.Series, n: int) -> pd.Series:
        """Exponential Moving Average."""
        return x.ewm(span=n, adjust=False).mean()
    
    @staticmethod
    def bollinger(x: pd.Series, n: int, k: float):
        """Bollinger Bands."""
        sma = x.rolling(n).mean()
        std = x.rolling(n).std()
        upper = sma + k * std
        lower = sma - k * std
        return sma, upper, lower
    
    # ... other indicators (RSI, MACD, ATR, etc.)
```

**Target File:** `chart_components/services/indicators_service.py` (check if exists, else create)  
**Service:** IndicatorsService

---

### ⚠️ **14. FORECAST OVERLAY**

#### File_A Implementation
```python
def _plot_forecast_overlay(self, quantiles: dict, source: str = "basic"):
    # Plot forecast quantiles on chart
    
def on_forecast_ready(self, df: pd.DataFrame, quantiles: dict):
    # Handle forecast results
    
def clear_all_forecasts(self):
    # Clear all forecast overlays
    
def _trim_forecasts(self):
    # Limit number of overlays
```

#### File_B Status
⚠️ **PARTIALLY MIGRATED** - Forecast service exists but may need enhancement

**Migration Strategy:**
Check and enhance `chart_components/services/forecast_service.py`:

```python
# In forecast_service.py
class ForecastService:
    # ... existing code ...
    
    def plot_forecast_overlay(self, quantiles: dict, source: str = "basic"):
        """Plot forecast quantiles as shaded regions."""
        # Implementation
        pass
    
    def clear_all_forecasts(self):
        """Remove all forecast overlays from chart."""
        for forecast_dict in self._forecasts:
            for artist in forecast_dict.get('artists', []):
                artist.remove()
        self._forecasts.clear()
    
    def trim_forecasts(self):
        """Keep only most recent N forecasts."""
        max_forecasts = get_setting('max_forecasts', 20)
        while len(self._forecasts) > max_forecasts:
            old_forecast = self._forecasts.pop(0)
            for artist in old_forecast.get('artists', []):
                artist.remove()
```

**Target File:** `chart_components/services/forecast_service.py`  
**Service:** ForecastService

---

### ❌ **15. ORDERS TABLE & TRADING**

#### File_A Implementation
```python
def _refresh_orders(self):
    # Query broker for open orders
    # Update orders table
    
def _toggle_orders(self, visible: bool):
    # Show/hide order lines on chart
    
def set_trading_engine(self, engine):
    # Connect trading engine
```

#### File_B Status
❌ **NOT MIGRATED** - Trading integration missing

**Migration Strategy:**
Create trading service:

```python
# Create new file: chart_components/services/trading_service.py
class TradingService:
    """Service for managing trading operations and order display."""
    
    def __init__(self, chart_widget, broker_service):
        self.chart = chart_widget
        self.broker = broker_service
        self._order_artists = []
        self._orders_visible = True
    
    def refresh_orders(self):
        """Query broker and update orders table."""
        if not self.broker:
            return
        
        orders = self.broker.get_open_orders()
        self._update_orders_table(orders)
        if self._orders_visible:
            self._plot_order_lines(orders)
    
    def toggle_orders_visibility(self, visible: bool):
        """Show or hide order lines on chart."""
        self._orders_visible = visible
        for artist in self._order_artists:
            artist.set_visible(visible)
        self.chart.canvas.draw_idle()
    
    def _update_orders_table(self, orders: list):
        """Update the orders table widget."""
        # Implementation
        pass
    
    def _plot_order_lines(self, orders: list):
        """Draw horizontal lines for pending orders."""
        # Clear old lines
        for artist in self._order_artists:
            artist.remove()
        self._order_artists.clear()
        
        # Draw new lines
        for order in orders:
            # Add horizontal line at order price
            pass
```

**Target File:** NEW - `chart_components/services/trading_service.py`  
**Integration:** `chart_tab_base.py`

---

### ⚠️ **16. POSITIONS TABLE INTEGRATION**

#### File_A Implementation
```python
def _on_position_selected(self, position: Dict):
    # Highlight position on chart
    
def _on_close_position_requested(self, position_id: str):
    # Close position via trading engine
    
def _on_modify_sl_requested(self, position_id: str, new_sl: float):
    # Modify stop loss
    
def _on_modify_tp_requested(self, position_id: str, new_tp: float):
    # Modify take profit
```

#### File_B Status
❌ **NOT MIGRATED** - Position table handlers missing

**Migration Strategy:**
Add to trading service (or create positions_service.py):

```python
# In trading_service.py - add position handling
class TradingService:
    # ... existing code ...
    
    def on_position_selected(self, position: dict):
        """Highlight selected position on chart."""
        entry_price = position.get('entry_price', 0.0)
        if self.chart.ax and entry_price > 0:
            # Center view on entry price
            ylim = self.chart.ax.get_ylim()
            y_range = ylim[1] - ylim[0]
            self.chart.ax.set_ylim(entry_price - y_range/2, entry_price + y_range/2)
            self.chart.canvas.draw_idle()
    
    def close_position(self, position_id: str):
        """Request position closure via trading engine."""
        if self.trading_engine:
            self.trading_engine.close_position(position_id)
    
    def modify_stop_loss(self, position_id: str, new_sl: float):
        """Modify stop loss for position."""
        if self.trading_engine:
            self.trading_engine.modify_stop_loss(position_id, new_sl)
    
    def modify_take_profit(self, position_id: str, new_tp: float):
        """Modify take profit for position."""
        if self.trading_engine:
            self.trading_engine.modify_take_profit(position_id, new_tp)
```

Connect signals in event_handlers.py:
```python
# In event_handlers.py
def _connect_ui_signals(self):
    # ... existing code ...
    
    if hasattr(self, 'positions_table'):
        self.positions_table.position_selected.connect(
            self.chart_controller.trading_service.on_position_selected
        )
        self.positions_table.close_position_requested.connect(
            self.chart_controller.trading_service.close_position
        )
        # ... other position signals
```

**Target File:** `chart_components/services/trading_service.py`  
**Integration:** `event_handlers.py`

---

### ⚠️ **17. DRAWING TOOLS**

#### File_A Implementation
```python
def _set_drawing_mode(self, mode: Optional[str]):
    # Set drawing mode (line, trend, fib, etc.)
    
def _on_canvas_click(self, event):
    # Handle click for drawing/testing
    # Supports: h-line, trend line, rectangle, fib retracement
```

#### File_B Status
⚠️ **PARTIALLY EXISTS** - Basic drawing structure exists in overlay_manager.py

**Migration Strategy:**
Enhance overlay_manager.py with full drawing support:

```python
# In overlay_manager.py - OverlayManagerMixin
def _set_drawing_mode(self, mode: Optional[str]):
    """Set current drawing mode."""
    self._drawing_mode = mode
    
    # Update cursor
    if mode:
        self.canvas.setCursor(QCursor(Qt.CrossCursor))
    else:
        self.canvas.setCursor(QCursor(Qt.ArrowCursor))
    
    # Clear pending points
    self._pending_points.clear()

def _on_canvas_click_drawing(self, event):
    """Handle click events for drawing mode."""
    if not self._drawing_mode or event.xdata is None:
        return
    
    point = (event.xdata, event.ydata)
    
    if self._drawing_mode == 'hline':
        self._draw_horizontal_line(point)
        self._drawing_mode = None
    
    elif self._drawing_mode == 'trend':
        self._pending_points.append(point)
        if len(self._pending_points) == 2:
            self._draw_trend_line(self._pending_points)
            self._pending_points.clear()
            self._drawing_mode = None
    
    elif self._drawing_mode == 'rect':
        self._pending_points.append(point)
        if len(self._pending_points) == 2:
            self._draw_rectangle(self._pending_points)
            self._pending_points.clear()
            self._drawing_mode = None
    
    elif self._drawing_mode == 'fib':
        self._pending_points.append(point)
        if len(self._pending_points) == 2:
            self._draw_fibonacci(self._pending_points)
            self._pending_points.clear()
            self._drawing_mode = None

def _draw_horizontal_line(self, point):
    """Draw a horizontal line at y-coordinate."""
    # Implementation
    pass

def _draw_trend_line(self, points):
    """Draw a trend line between two points."""
    # Implementation
    pass

def _draw_rectangle(self, points):
    """Draw a rectangle from two diagonal points."""
    # Implementation
    pass

def _draw_fibonacci(self, points):
    """Draw Fibonacci retracement levels."""
    # Implementation
    pass
```

**Target File:** `overlay_manager.py`  
**Service:** None (UI layer)

---

### ❌ **18. TESTING POINTS (Alt+Click)**

#### File_A Implementation
```python
def _on_canvas_click(self, event):
    # Alt+Click: basic testing point
    # Shift+Alt+Click: advanced testing point
    # Creates markers on chart for analysis
```

#### File_B Status
❌ **NOT MIGRATED** - Testing point feature missing

**Migration Strategy:**
Add to overlay_manager.py or create separate testing_service.py:

```python
# In overlay_manager.py - add testing point support
def _on_canvas_click_testing(self, event):
    """Handle Alt+Click for testing points."""
    if not event or event.xdata is None:
        return
    
    gui_event = getattr(event, 'guiEvent', None)
    if not gui_event:
        return
    
    modifiers = gui_event.modifiers()
    from PySide6.QtCore import Qt
    
    if modifiers & Qt.AltModifier:
        if modifiers & Qt.ShiftModifier:
            self._add_advanced_testing_point(event.xdata, event.ydata)
        else:
            self._add_basic_testing_point(event.xdata, event.ydata)

def _add_basic_testing_point(self, x, y):
    """Add a basic testing marker."""
    # Draw simple marker
    marker = self.ax.plot(x, y, 'ro', markersize=8, zorder=100)[0]
    self._testing_markers.append(marker)
    self.canvas.draw_idle()

def _add_advanced_testing_point(self, x, y):
    """Add an advanced testing marker with analysis."""
    # Draw marker with additional info
    marker = self.ax.plot(x, y, 'bs', markersize=10, zorder=100)[0]
    # Add text annotation
    text = self.ax.text(x, y, f"Test\n{y:.5f}", 
                        fontsize=8, color='blue',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    self._testing_markers.append(marker)
    self._testing_markers.append(text)
    self.canvas.draw_idle()
```

**Target File:** `overlay_manager.py`  
**Service:** None (UI layer)

---

### ⚠️ **19. ADHERENCE BADGES**

#### File_A Implementation
```python
self._adh_badges: list = []

def _update_badge_visibility(self, event):
    # Hide badges when cursor near them
    # Show when cursor away
```

#### File_B Status
❌ **NOT MIGRATED** - Badge system missing

**Migration Strategy:**
Add badge system to overlay_manager.py:

```python
# In overlay_manager.py
def _initialize_badges(self):
    """Initialize adherence badge system."""
    self._adh_badges = []

def add_adherence_badge(self, x, y, text: str, color: str = 'green'):
    """Add an adherence score badge to the chart."""
    badge = self.ax.text(x, y, text,
                        fontsize=9,
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.5',
                                facecolor=color,
                                alpha=0.8,
                                edgecolor='none'),
                        ha='center',
                        va='center',
                        zorder=200)
    self._adh_badges.append(badge)
    return badge

def _update_badge_visibility(self, event):
    """Hide badges when cursor is near them."""
    if not event or event.xdata is None:
        return
    
    for badge in self._adh_badges:
        # Calculate distance from cursor to badge
        x, y = badge.get_position()
        dist_x = abs(event.xdata - x)
        dist_y = abs(event.ydata - y)
        
        # Hide if cursor within 2x badge size
        if dist_x < 0.1 and dist_y < 0.1:  # Adjust thresholds as needed
            badge.set_visible(False)
        else:
            badge.set_visible(True)
    
    self.canvas.draw_idle()
```

Connect in event_handlers.py:
```python
def _on_mouse_move(self, event):
    # ... existing code ...
    self._update_badge_visibility(event)
```

**Target File:** `overlay_manager.py`  
**Service:** None (UI layer)

---

### ⚠️ **20. TIMERS**

#### File_A Implementation
```python
# Auto-forecast timer
self._auto_timer = QTimer(self)
self._auto_timer.timeout.connect(self.chart_controller.auto_forecast_tick)

# Orders refresh timer
self._orders_timer = QTimer(self)
self._orders_timer.timeout.connect(self.chart_controller.refresh_orders)

# Realtime throttling timer
self._rt_timer = QTimer(self)
self._rt_timer.timeout.connect(self.chart_controller.rt_flush)

# View reload timer
self._reload_timer = QTimer(self)
self._reload_timer.timeout.connect(self.chart_controller.reload_view_window)
```

#### File_B Status
⚠️ **PARTIALLY EXISTS** - Some timers may be set up

**Migration Strategy:**
Add comprehensive timer setup in chart_tab_base.py:

```python
# In chart_tab_base.py - add to _initialize_timers_and_connections()
def _setup_timers(self):
    """Initialize all timers."""
    # Auto-forecast timer
    self._auto_timer = QTimer(self)
    self._auto_timer.setInterval(int(get_setting("auto_interval_seconds", 60) * 1000))
    self._auto_timer.timeout.connect(self.chart_controller.auto_forecast_tick)
    
    # Orders refresh timer
    self._orders_timer = QTimer(self)
    self._orders_timer.setInterval(1500)
    self._orders_timer.timeout.connect(self.chart_controller.refresh_orders)
    self._orders_timer.start()
    
    # Realtime update throttling timer
    self._rt_timer = QTimer(self)
    self._rt_timer.setInterval(200)  # 5 FPS
    self._rt_timer.timeout.connect(self.chart_controller.rt_flush)
    self._rt_timer.start()
    
    # View window reload timer (throttled)
    self._reload_timer = QTimer(self)
    self._reload_timer.setSingleShot(True)
    self._reload_timer.setInterval(250)
    self._reload_timer.timeout.connect(self.chart_controller.reload_view_window)
```

**Target File:** `chart_tab_base.py`  
**Service:** None (timer management)

---

### ⚠️ **21. REALTIME TICK HANDLING**

#### File_A Implementation
```python
def _handle_tick(self, payload: dict):
    # Thread-safe tick enqueue
    
def _on_tick_main(self, payload: dict):
    # GUI thread: update market watch and buffer
    
def _rt_flush(self):
    # Throttled redraw preserving zoom/pan
```

#### File_B Status
⚠️ **PARTIALLY EXISTS** - Basic tick handling may exist

**Migration Strategy:**
Enhance data service with tick handling:

```python
# In data_service.py
class DataService:
    # ... existing code ...
    
    def __init__(self, chart_widget):
        # ... existing init ...
        self._rt_buffer = []
        self._rt_dirty = False
    
    def handle_tick(self, payload: dict):
        """Thread-safe tick handler - enqueue for GUI thread."""
        self.chart.tickArrived.emit(payload)
    
    def on_tick_main(self, payload: dict):
        """GUI thread tick handler."""
        # Update market watch
        symbol = payload.get('symbol')
        bid = payload.get('bid')
        ask = payload.get('ask')
        ts_ms = payload.get('ts_ms')
        
        if hasattr(self.chart, 'market_watch_service'):
            self.chart.market_watch_service.update_quote(symbol, bid, ask, ts_ms)
        
        # Add to buffer
        self._rt_buffer.append(payload)
        self._rt_dirty = True
    
    def rt_flush(self):
        """Throttled redraw from buffered ticks."""
        if not self._rt_dirty or not self._rt_buffer:
            return
        
        # Process buffered ticks
        for tick in self._rt_buffer:
            # Update chart data
            pass
        
        self._rt_buffer.clear()
        self._rt_dirty = False
        
        # Redraw preserving view
        self.chart.update_plot(self.chart._last_df, 
                             restore_xlim=self.chart.ax.get_xlim(),
                             restore_ylim=self.chart.ax.get_ylim())
```

**Target File:** `chart_components/services/data_service.py`  
**Service:** DataService

---

### ⚠️ **22. BACKFILL PROGRESS**

#### File_A Implementation
```python
def _on_backfill_missing_clicked(self):
    # Show progress bar
    # Trigger async backfill
    # Update progress
```

#### File_B Status
❌ **NOT MIGRATED** - Progress reporting missing

**Migration Strategy:**
Add to data service with progress callback:

```python
# In data_service.py
class DataService:
    # ... existing code ...
    
    def backfill_data(self, symbol: str, timeframe: str, years: int, months: int, 
                     progress_callback=None):
        """Backfill historical data with progress reporting."""
        # Show progress bar
        if hasattr(self.chart, 'backfill_progress'):
            self.chart.backfill_progress.setVisible(True)
            self.chart.backfill_progress.setValue(0)
        
        # Start async backfill
        # Report progress via callback
        # Hide progress bar when done
        pass
```

Connect in event_handlers.py:
```python
def _on_backfill_clicked(self):
    """Handle backfill button click."""
    years = int(self.years_combo.currentText())
    months = int(self.months_combo.currentText())
    symbol = self.symbol_combo.currentText()
    tf = self.tf_combo.currentText()
    
    def progress_callback(percent):
        if hasattr(self, 'backfill_progress'):
            self.backfill_progress.setValue(percent)
    
    self.chart_controller.data_service.backfill_data(
        symbol, tf, years, months, progress_callback
    )
```

**Target File:** `chart_components/services/data_service.py`  
**Integration:** `event_handlers.py`

---

### ✅ **23. PATTERN DETECTION**

#### File_A Implementation
```python
# Pattern checkboxes
self.chart_patterns_checkbox = QCheckBox("Chart patterns")
self.candle_patterns_checkbox = QCheckBox("Candlestick patterns")
self.history_patterns_checkbox = QCheckBox("Patterns storici")

# Connected to patterns_service
```

#### File_B Status
✅ **MIGRATED** - Pattern service exists in `chart_components/services/patterns/`

**Verification Needed:**
- Ensure checkboxes exist in UI (ui_builder.py)
- Verify connections in event_handlers.py
- Confirm patterns_service integration

**Target File:** `patterns_mixin.py` (check implementation)  
**Service:** `chart_components/services/patterns/patterns_service.py`

---

### ⚠️ **24. SPLITTER PERSISTENCE**

#### File_A Implementation
```python
def _restore_splitters(self) -> None:
    # Load splitter positions from settings
    
def _persist_splitter_positions(self, key: str, splitter: QSplitter):
    # Save splitter positions to settings
```

#### File_B Status
❌ **NOT MIGRATED** - Splitter state not persisted

**Migration Strategy:**
Add to event_handlers.py:

```python
# In event_handlers.py
def _restore_splitters(self):
    """Restore splitter positions from settings."""
    for key, splitter in [
        ('chart.splitter.main', getattr(self, 'main_splitter', None)),
        ('chart.splitter.right', getattr(self, 'right_splitter', None)),
        ('chart.splitter.chart', getattr(self, '_chart_area', None)),
    ]:
        if not splitter:
            continue
        
        sizes = get_setting(key, None)
        if sizes and isinstance(sizes, (list, tuple)):
            try:
                splitter.setSizes([int(s) for s in sizes])
            except Exception:
                pass

def _persist_splitter_positions(self, key: str, splitter):
    """Save splitter positions to settings."""
    try:
        set_setting(key, splitter.sizes())
    except Exception:
        pass
```

Call in initialization:
```python
# In chart_tab_base.py - _initialize_timers_and_connections()
self._restore_splitters()
```

**Target File:** `event_handlers.py`  
**Service:** None (UI state management)

---

### ⚠️ **25. COMBO BOX UTILITIES**

#### File_A Implementation
```python
def _set_combo_with_items(self, combo: Optional[QComboBox], items: List[str], 
                          setting_key: str, default: str) -> str:
    # Populate combo box with items
    # Restore saved selection
    # Block signals during setup
```

#### File_B Status
❌ **NOT MIGRATED** - Helper utility missing

**Migration Strategy:**
Add utility method to ui_builder.py or event_handlers.py:

```python
# In ui_builder.py or event_handlers.py
def _set_combo_with_items(self, combo: Optional[QComboBox], 
                         items: List[str], 
                         setting_key: str, 
                         default: str) -> str:
    """Populate combo box and restore saved value."""
    if combo is None:
        return default
    
    saved = str(get_setting(setting_key, default))
    
    blocker = QSignalBlocker(combo)
    combo.clear()
    combo.addItems(items)
    
    # Add saved value if not in list
    if saved and combo.findText(saved) == -1:
        combo.addItem(saved)
    
    combo.setCurrentText(saved if saved else default)
    del blocker
    
    return saved if saved else default
```

**Target File:** `ui_builder.py`  
**Service:** None (UI utility)

---

### ⚠️ **26. SYMBOL & TIMEFRAME MANAGEMENT**

#### File_A Implementation
```python
def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
    # Update symbol/TF in UI
    # Load data from DB
    # Refresh chart

def _on_symbol_changed(self, new_symbol: str):
    # Handle symbol change
    # Reload chart data
```

#### File_B Status
⚠️ **PARTIALLY EXISTS** - Basic handling present

**Migration Strategy:**
Enhance in controller_proxy.py:

```python
# In controller_proxy.py
def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
    """Set symbol and timeframe, then reload chart."""
    # Update UI controls
    if hasattr(self, 'symbol_combo'):
        blocker = QSignalBlocker(self.symbol_combo)
        if self.symbol_combo.findText(symbol) == -1:
            self.symbol_combo.addItem(symbol)
        self.symbol_combo.setCurrentText(symbol)
        del blocker
    
    if hasattr(self, 'tf_combo'):
        blocker = QSignalBlocker(self.tf_combo)
        if self.tf_combo.findText(timeframe) == -1:
            self.tf_combo.addItem(timeframe)
        self.tf_combo.setCurrentText(timeframe)
        del blocker
    
    # Update state
    self.symbol = symbol
    self.timeframe = timeframe
    set_setting('chart.symbol', symbol)
    set_setting('chart.timeframe', timeframe)
    
    # Reload data
    self.chart_controller.on_symbol_changed(new_symbol=symbol)
```

**Target File:** `controller_proxy.py`  
**Service:** None (state management)

---

## Summary of Migration Tasks

### HIGH PRIORITY (Critical Functionality)
1. ⚠️ **Hover Legend / Cursor Info** - Key UX feature
2. ❌ **Mouse Zoom & Pan** - Critical UX feature
3. ⚠️ **Market Watch Updates** - Real-time data display
4. ⚠️ **Orders Table & Trading** - Trading functionality
5. ⚠️ **Realtime Tick Handling** - Live data updates
6. ⚠️ **Follow Mode** - Auto-centering behavior

### MEDIUM PRIORITY (Important Features)
7. ⚠️ **Price Mode Toggle** - Candles vs Line
8. ⚠️ **Theme Application** - Visual customization
9. ⚠️ **Indicator Plotting** - Technical analysis
10. ⚠️ **Forecast Overlay** - Prediction display
11. ⚠️ **Drawing Tools** - Chart annotations
12. ⚠️ **Dynamic Data Loading** - Performance optimization
13. ⚠️ **Backfill Progress** - User feedback

### LOW PRIORITY (Nice to Have)
14. ⚠️ **Testing Points** - Alt+Click markers
15. ⚠️ **Adherence Badges** - Score display
16. ⚠️ **Splitter Persistence** - UI state
17. ⚠️ **Position Table Integration** - Position management

### ALREADY MIGRATED ✅
- Core initialization
- UI construction (with finplot)
- Basic event handling
- Pattern detection integration
- Controller structure

---

## File-by-File Action Plan

### 1. chart_tab_base.py
**Actions:**
- Add missing state variables (`_symbol_row_map`, `_spread_state`, `_follow_suspend_until`)
- Enhance `_initialize_state()` with comprehensive initialization
- Add timer setup method `_setup_timers()`
- Add matplotlib compatibility layer reference

### 2. ui_builder.py  
**Actions:**
- Add matplotlib canvas creation method `_ensure_matplotlib_canvas()`
- Verify pattern checkboxes exist in topbar
- Add utility method `_set_combo_with_items()`

### 3. event_handlers.py
**Actions:**
- Add pattern checkbox signal connections
- Add splitter persistence methods
- Add follow mode toggle handler
- Add price mode toggle handler
- Enhance mouse event handlers with follow suspension
- Add position table signal connections
- Add splitter restore/persist methods

### 4. overlay_manager.py
**Actions:**
- Add hover legend support (`_ensure_hover_legend`, `_update_hover_info`)
- Enhance drawing tools (trend, rect, fib, testing points)
- Add adherence badge system
- Improve overlay dragging

### 5. controller_proxy.py
**Actions:**
- Add `set_symbol_timeframe()` method
- Ensure all controller methods are proxied correctly

### 6. patterns_mixin.py
**Actions:**
- Verify pattern service integration
- Ensure checkbox toggles work

### 7. NEW FILE: chart_components/services/market_watch_service.py
**Create new service:**
- `MarketWatchService` class
- Methods: `update_quote()`, `add_symbol()`, spread tracking

### 8. NEW FILE: chart_components/services/interaction_service.py
**Create new service:**
- `ChartInteractionService` class
- Methods: `on_scroll_zoom()`, `on_mouse_press()`, pan/zoom handling

### 9. NEW FILE: chart_components/services/trading_service.py
**Create new service:**
- `TradingService` class
- Methods: `refresh_orders()`, `toggle_orders_visibility()`, position handlers

### 10. ENHANCE: chart_components/services/data_service.py
**Add methods:**
- `schedule_view_reload()`
- `reload_view_window()`
- `resolution_for_span()`
- `handle_tick()`, `on_tick_main()`, `rt_flush()`
- `backfill_data()` with progress reporting

### 11. ENHANCE: chart_components/services/plot_service.py
**Add methods:**
- `set_price_mode()` (candles vs line)
- `apply_theme()`
- `_get_color()`

### 12. ENHANCE: chart_components/services/forecast_service.py
**Verify methods:**
- `plot_forecast_overlay()`
- `clear_all_forecasts()`
- `trim_forecasts()`

### 13. CHECK/ENHANCE: chart_components/services/indicators_service.py
**If missing, create; else enhance:**
- Indicator calculation methods (SMA, EMA, Bollinger, RSI, MACD, etc.)
- `ensure_osc_axis()`
- `plot_indicators()`

---

## Migration Workflow

### Phase 1: Critical UX (Week 1)
1. Create `interaction_service.py` - Mouse zoom/pan
2. Add hover legend to `overlay_manager.py`
3. Create `market_watch_service.py` - Real-time quotes
4. Enhance tick handling in `data_service.py`

### Phase 2: Trading Integration (Week 2)
5. Create `trading_service.py` - Orders and positions
6. Add position table handlers
7. Add orders table refresh logic
8. Integrate trading engine

### Phase 3: Chart Features (Week 3)
9. Add price mode toggle to `plot_service.py`
10. Enhance theme support
11. Complete drawing tools in `overlay_manager.py`
12. Add testing points and badges

### Phase 4: Optimization & Polish (Week 4)
13. Add dynamic data loading to `data_service.py`
14. Implement backfill progress reporting
15. Add splitter persistence
16. Complete indicator plotting
17. Verify forecast overlays

---

## Testing Strategy

### Unit Testing
- Test each service independently
- Mock dependencies
- Test edge cases

### Integration Testing
- Test service interactions
- Test UI<->Controller<->Service flow
- Test event propagation

### Manual Testing
- Verify each migrated feature works in UI
- Test with real broker data
- Verify performance with large datasets

---

## Deprecation Plan for File_A

### DO NOT DELETE File_A Yet
File_A should remain as reference until:
1. All features are verified migrated
2. All tests pass
3. Production usage confirms stability

### Mark as Deprecated
Add to top of `chart_tab.py`:
```python
"""
DEPRECATED: This file is the old monolithic implementation.
DO NOT MODIFY THIS FILE.

Use the modular implementation in ui/chart_tab/ directory instead.

This file is kept for reference during migration only and will be
removed once all features are verified migrated.

Deprecation Date: 2025-10-08
Target Removal: 2025-12-01
"""
```

---

## Progress Tracking

Use this checklist to track migration progress:

- [ ] Market Watch Service
- [ ] Interaction Service (Zoom/Pan)
- [ ] Hover Legend
- [ ] Trading Service
- [ ] Tick Handling
- [ ] Follow Mode
- [ ] Price Mode Toggle
- [ ] Theme System
- [ ] Drawing Tools Complete
- [ ] Testing Points
- [ ] Adherence Badges
- [ ] Dynamic Data Loading
- [ ] Backfill Progress
- [ ] Splitter Persistence
- [ ] Position Handlers
- [ ] Indicator Service
- [ ] Forecast Overlays
- [ ] Orders Table Integration

---

## Contact & Support

**Migration Lead:** [Your Name]  
**Start Date:** 2025-10-08  
**Target Completion:** 2025-11-08  
**Status:** In Progress

For questions or issues during migration, document them in:
`D:\Projects\ForexGPT\ANALYSIS\MIGRATION_ISSUES.md`

---

**END OF ANALYSIS**
