# UI Migration Quick Reference

## Critical Missing Features (Priority Order)

### 🔴 MUST MIGRATE IMMEDIATELY
1. **Mouse Zoom & Pan** → `interaction_service.py` (NEW)
2. **Hover Legend** → `overlay_manager.py` 
3. **Market Watch Updates** → `market_watch_service.py` (NEW)
4. **Tick Handling** → `data_service.py` (enhance)

### 🟡 SHOULD MIGRATE SOON
5. **Trading/Orders** → `trading_service.py` (NEW)
6. **Follow Mode** → `event_handlers.py`
7. **Price Mode Toggle** → `plot_service.py`
8. **Drawing Tools** → `overlay_manager.py`

### 🟢 CAN MIGRATE LATER
9. **Testing Points** → `overlay_manager.py`
10. **Adherence Badges** → `overlay_manager.py`
11. **Splitter Persistence** → `event_handlers.py`

---

## New Services to Create

1. **market_watch_service.py** - Real-time quote updates
2. **interaction_service.py** - Mouse zoom/pan/drag
3. **trading_service.py** - Orders, positions, trading engine

---

## Services to Enhance

1. **data_service.py** - Add tick handling, dynamic loading
2. **plot_service.py** - Add price mode, theme support  
3. **forecast_service.py** - Verify overlay methods

---

## Files to Modify

### chart_tab/
- `chart_tab_base.py` - Add state variables
- `ui_builder.py` - Add matplotlib compat
- `event_handlers.py` - Add handlers (10+ methods)
- `overlay_manager.py` - Add overlays (5+ methods)
- `controller_proxy.py` - Add symbol/TF methods

---

## Quick Command Reference

```bash
# Check current structure
tree /F src\forex_diffusion\ui\chart_tab
tree /F src\forex_diffusion\ui\chart_components

# Create new services
mkdir src\forex_diffusion\ui\chart_components\services\trading
type nul > src\forex_diffusion\ui\chart_components\services\market_watch_service.py
type nul > src\forex_diffusion\ui\chart_components\services\interaction_service.py
type nul > src\forex_diffusion\ui\chart_components\services\trading_service.py

# Run tests after migration
pytest tests/ui/test_chart_tab.py -v
```

---

## Progress Checklist

### Week 1 - Critical UX
- [ ] interaction_service.py created
- [ ] Mouse zoom/pan working
- [ ] Hover legend implemented
- [ ] market_watch_service.py created
- [ ] Tick handling enhanced

### Week 2 - Trading
- [ ] trading_service.py created
- [ ] Orders table working
- [ ] Position handlers connected
- [ ] Trading engine integrated

### Week 3 - Features  
- [ ] Price mode toggle
- [ ] Theme system complete
- [ ] Drawing tools complete
- [ ] Testing points added

### Week 4 - Polish
- [ ] Dynamic data loading
- [ ] Backfill progress
- [ ] Splitter persistence
- [ ] All tests passing

---

## Common Patterns

### Adding a New Service
```python
# 1. Create service file
# chart_components/services/my_service.py

class MyService:
    def __init__(self, chart_widget):
        self.chart = chart_widget
    
    def my_method(self):
        pass

# 2. Initialize in chart_tab_base.py
def _initialize_state(self):
    from ..chart_components.services.my_service import MyService
    self.my_service = MyService(self)

# 3. Use in event_handlers.py
def _on_something(self):
    self.my_service.my_method()
```

### Adding Event Handler
```python
# In event_handlers.py
def _on_my_event(self, arg):
    """Handle my event."""
    # Your logic here
    pass

# Connect in _connect_ui_signals()
def _connect_ui_signals(self):
    if hasattr(self, 'my_button'):
        self.my_button.clicked.connect(self._on_my_event)
```

### Adding to Overlay Manager
```python
# In overlay_manager.py
def add_my_overlay(self, x, y, data):
    """Add my overlay."""
    from .chart_tab_base import DraggableOverlay
    overlay = DraggableOverlay(data, self.chart_container)
    overlay.move(x, y)
    self._my_overlays.append(overlay)
```

---

## Architecture Diagram

```
ChartTabUI (chart_tab_base.py)
├── UIBuilderMixin (ui_builder.py)
│   └── Builds UI structure
├── EventHandlersMixin (event_handlers.py)  
│   └── Handles user interactions
├── ControllerProxyMixin (controller_proxy.py)
│   └── Delegates to controller
├── PatternsMixin (patterns_mixin.py)
│   └── Pattern detection
└── OverlayManagerMixin (overlay_manager.py)
    └── Overlays & drawings

ChartTabController (chart_components/controllers/)
├── DataService - Data loading/caching
├── PlotService - Chart rendering
├── ForecastService - Predictions
├── PatternsService - Pattern detection
├── MarketWatchService - Quote updates (NEW)
├── InteractionService - Zoom/pan (NEW)
└── TradingService - Orders/positions (NEW)
```

---

## Key Differences: File_A vs File_B

| Aspect | File_A (Old) | File_B (New) |
|--------|-------------|-------------|
| Structure | Monolithic | Modular Mixins |
| Chart Library | Matplotlib | Finplot/PyQtGraph |
| Data Access | Direct DB | Service Layer |
| Testing | Hard | Easy |
| Lines of Code | 2,500+ | ~500 per file |
| Maintainability | Low | High |

---

## When in Doubt...

1. Check if feature exists in File_A? → See migration doc section
2. Which service? → Check "Services to Create/Enhance" 
3. Which mixin? → Event→EventHandlers, UI→UIBuilder, Draw→OverlayManager
4. Controller vs Service? → Controller delegates, Service implements

---

**Last Updated:** 2025-10-08  
**Full Analysis:** `UI_Migration.md`  
**Issues:** `MIGRATION_ISSUES.md`
