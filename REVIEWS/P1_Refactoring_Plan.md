# P1 Refactoring Plan - Implementation Guide

**Date**: 2025-01-08  
**Priority**: P1 (High - Next 2 Weeks)  
**Estimated Time**: 25 hours  
**Status**: 🔄 **READY TO IMPLEMENT**

---

## P1 TASKS BREAKDOWN

### 1. GUI: Split training_tab.py (6 hours) - **HIGHEST IMPACT**

**Current**: training_tab.py (2,301 lines - MONOLITHIC!)

**Target Structure**:
```
training_tab/
├── __init__.py (20 lines)
├── training_tab_base.py (400 lines) - Main TrainingTab class
├── ui_components.py (600 lines) - UI widgets builder
├── event_handlers.py (500 lines) - Button clicks, signals
├── constants.py (300 lines) - INDICATORS, tooltips, etc.
├── model_manager.py (300 lines) - Model operations
└── utils.py (200 lines) - Helper functions
```

**Total**: ~2,320 lines (modular, maintainable)

**Implementation Steps**:

**Step 1**: Extract constants (1 hour)
```python
# training_tab/constants.py
INDICATORS = ["ATR", "RSI", "MACD", ...]
ADDITIONAL_FEATURES = ["Returns & Volatility", ...]
TIMEFRAMES = ["1m", "5m", "15m", ...]
INDICATOR_TOOLTIPS = {...}
```

**Step 2**: Extract UI builder (2 hours)
```python
# training_tab/ui_components.py
class UIComponents:
    @staticmethod
    def build_model_section(parent) -> QGroupBox:
        \"\"\"Build model selection section.\"\"\"
        group = QGroupBox("Model Configuration", parent)
        # ... build widgets ...
        return group
    
    @staticmethod
    def build_data_section(parent) -> QGroupBox:
        \"\"\"Build data configuration section.\"\"\"
        # ...
        return group
```

**Step 3**: Extract event handlers (2 hours)
```python
# training_tab/event_handlers.py
class TrainingEventHandlers:
    def __init__(self, parent):
        self.parent = parent
    
    def on_train_clicked(self):
        \"\"\"Handle train button click.\"\"\"
        # ...
    
    def on_symbol_changed(self, text):
        \"\"\"Handle symbol combo change.\"\"\"
        # ...
```

**Step 4**: Create base class (1 hour)
```python
# training_tab/training_tab_base.py
from .ui_components import UIComponents
from .event_handlers import TrainingEventHandlers
from .constants import INDICATORS, TIMEFRAMES

class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = UIComponents()
        self.handlers = TrainingEventHandlers(self)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self.ui.build_model_section(self))
        layout.addWidget(self.ui.build_data_section(self))
        # ...
```

**Benefits**:
- 6 files × ~400 lines each (vs 1 file × 2,301 lines)
- Easier testing (mock UIComponents separately)
- Faster load times
- Better maintainability
- Clear separation of concerns

---

### 2. GUI: Split pattern_overlay.py (4 hours)

**Current**: pattern_overlay.py (1,800+ lines)

**Target Structure**:
```
pattern_overlay/
├── __init__.py
├── overlay_renderer.py (600 lines) - Graphics rendering
├── pattern_detector.py (500 lines) - Detection logic
├── pattern_dialog.py (400 lines) - Details dialog
└── pattern_utils.py (300 lines) - Helper functions
```

**Implementation**: Similar to training_tab split

---

### 3. Feature Pipeline: Remove Duplications (3 hours)

**Remaining Duplications**:

**DUP-002**: temporal_features() in pipeline.py
```python
# BEFORE: pipeline.py has time_cyclic_and_session()
# AFTER: Use feature_engineering.temporal_features()

# pipeline.py
def time_cyclic_and_session(df: pd.DataFrame) -> pd.DataFrame:
    from .feature_engineering import temporal_features
    temp_feats = temporal_features(df, use_cyclical=True)
    
    # Add session logic (unique to this function)
    dt = pd.to_datetime(df["ts_utc"], unit="ms", utc=True)
    hours = dt.dt.hour + dt.dt.minute / 60.0
    temp_feats["session_tokyo"] = ((hours >= 0) & (hours < 9)).astype(int)
    temp_feats["session_london"] = ((hours >= 7) & (hours < 16)).astype(int)
    temp_feats["session_ny"] = ((hours >= 13) & (hours < 22)).astype(int)
    
    return temp_feats
```

**DUP-005**: Bollinger bands
```python
# Use consolidated_indicators.bollinger()
# Remove duplicate from pipeline.py
```

**DUP-006**: RSI implementations
```python
# Standardize on consolidated_indicators.rsi()
# Remove duplicates from pipeline.py
```

---

### 4. Training: Implement 60/20/20 Split (4 hours)

**Current**: 2-way split (train/val) in train.py:115-180

**Target**: 3-way split (train/val/test)

**Implementation**:
```python
# train.py (NEW function)
def _standardize_train_val_test(
    patches: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    \"\"\"
    Standardize with proper 60/20/20 split (train/val/test).
    
    CRITICAL: Computes statistics ONLY on training set.
    This is the RECOMMENDED split for production models.
    
    Returns:
        Tuple of (train_norm, val_norm, test_norm, scaler_metadata)
    \"\"\"
    from scipy import stats
    
    n = patches.shape[0]
    
    # Temporal split (no shuffling for time-series!)
    train_size = int(n * train_frac)
    val_size = int(n * val_frac)
    test_size = n - train_size - val_size
    
    if train_size < 100:
        raise ValueError(f"Training set too small: {train_size} samples")
    
    # Split data
    train = patches[:train_size]
    val = patches[train_size:train_size + val_size]
    test = patches[train_size + val_size:]
    
    # Compute statistics ONLY on training set (NO look-ahead bias)
    mu = train.mean(axis=(0, 2), keepdims=True)
    sigma = train.std(axis=(0, 2), keepdims=True)
    sigma[sigma == 0] = 1.0
    
    # Apply standardization
    train_norm = (train - mu) / sigma
    val_norm = (val - mu) / sigma
    test_norm = (test - mu) / sigma
    
    # Statistical verification
    # Check distribution differences (should be different for time-series)
    train_flat = train_norm[:, 0, :].flatten()
    val_flat = val_norm[:, 0, :].flatten()
    test_flat = test_norm[:, 0, :].flatten()
    
    ks_train_val = None
    ks_train_test = None
    if len(train_flat) > 20 and len(val_flat) > 20 and len(test_flat) > 20:
        _, ks_train_val = stats.ks_2samp(train_flat, val_flat)
        _, ks_train_test = stats.ks_2samp(train_flat, test_flat)
    
    scaler_metadata = {
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "ks_train_val_p": float(ks_train_val) if ks_train_val else None,
        "ks_train_test_p": float(ks_train_test) if ks_train_test else None,
    }
    
    return train_norm, val_norm, test_norm, scaler_metadata
```

**Update train.py to use new function**:
```python
# Replace _standardize_train_val() calls with:
train_X, val_X, test_X, scaler_meta = _standardize_train_val_test(patches_X)
train_y, val_y, test_y, _ = _standardize_train_val_test(patches_y)

# Use test set for final evaluation (not val!)
test_loss = model.evaluate(test_X, test_y)
```

**Deprecate old function**:
```python
def _standardize_train_val(patches, val_frac):
    \"\"\"
    DEPRECATED: Use _standardize_train_val_test() for proper 3-way split.
    
    This function is kept for backward compatibility only.
    \"\"\"
    warnings.warn(
        "_standardize_train_val is deprecated. Use _standardize_train_val_test instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing code ...
```

---

### 5. All Systems: Expand Test Coverage (8 hours)

**Current Coverage**: ~15%  
**Target Coverage**: >50%

**Priority Test Files**:

**Feature Pipeline Tests** (3 hours):
```python
# tests/test_feature_engineering.py (NEW)
def test_relative_ohlc():
    df = pd.DataFrame({
        'open': [1.0, 1.1, 1.2],
        'high': [1.1, 1.2, 1.3],
        'low': [0.9, 1.0, 1.1],
        'close': [1.05, 1.15, 1.25]
    })
    result = relative_ohlc(df)
    assert 'r_open' in result.columns
    assert 'r_close' in result.columns
    assert len(result) == len(df)

def test_temporal_features():
    df = pd.DataFrame({'ts_utc': [1609459200000]})  # 2021-01-01
    result = temporal_features(df)
    assert 'hour_sin' in result.columns
    assert 'hour_cos' in result.columns

def test_realized_volatility():
    df = pd.DataFrame({'close': [1.0, 1.1, 1.2, 1.3, 1.4]})
    result = realized_volatility_feature(df, window=3)
    assert f'rv_3' in result.columns
```

**Training Tests** (2 hours):
```python
# tests/test_train_splits.py (NEW)
def test_60_20_20_split():
    patches = np.random.randn(100, 5, 64)
    train, val, test, meta = _standardize_train_val_test(patches)
    
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    assert meta['train_size'] == 60

def test_no_look_ahead_bias():
    \"\"\"Verify statistics computed only on train set.\"\"\"
    patches = np.concatenate([
        np.ones((60, 5, 64)),  # Train: mean=1
        np.zeros((40, 5, 64))  # Val+test: mean=0
    ])
    train, val, test, _ = _standardize_train_val_test(patches)
    
    # Train should be standardized around 0
    assert abs(train.mean()) < 0.1
    
    # Val/test should be negative (since train mean was positive)
    assert val.mean() < -0.5
    assert test.mean() < -0.5
```

**Model Tests** (2 hours):
```python
# tests/test_models.py (NEW)
def test_diffusion_model_forward():
    model = DiffusionModel(z_dim=128)
    z_t = torch.randn(4, 128)
    t = torch.tensor([10, 20, 30, 40])
    v_hat = model(z_t, t)
    assert v_hat.shape == (4, 128)

def test_sssd_model_shapes():
    config = SSSDConfig()
    model = SSSDModel(config)
    # Test forward pass
```

**GUI Tests** (1 hour):
```python
# tests/test_gui_widgets.py (NEW)
import pytest
from pytestqt import qt_compat

def test_training_tab_init(qtbot):
    \"\"\"Test TrainingTab initializes without errors.\"\"\"
    from forex_diffusion.ui.training_tab import TrainingTab
    widget = TrainingTab()
    qtbot.addWidget(widget)
    assert widget is not None

def test_training_tab_buttons(qtbot):
    \"\"\"Test buttons are present.\"\"\"
    widget = TrainingTab()
    qtbot.addWidget(widget)
    # Find train button
    train_btn = widget.findChild(QPushButton, "train_button")
    assert train_btn is not None
```

---

## IMPLEMENTATION TIMELINE

### Week 1 (12 hours):
- **Day 1**: Split training_tab.py (6 hours)
- **Day 2**: Feature Pipeline duplications (3 hours)
- **Day 3**: Training 60/20/20 split (3 hours)

### Week 2 (13 hours):
- **Day 1**: Split pattern_overlay.py (4 hours)
- **Day 2**: Tests - Feature Pipeline (3 hours)
- **Day 3**: Tests - Training (2 hours)
- **Day 4**: Tests - Models (2 hours)
- **Day 5**: Tests - GUI (1 hour), Final testing (1 hour)

**Total**: 25 hours

---

## TESTING CHECKLIST

After implementing each fix:

- [ ] Syntax validation (python -m py_compile)
- [ ] Import tests (python -c "import module")
- [ ] Unit tests (pytest tests/test_*.py)
- [ ] Integration tests (run full pipeline)
- [ ] Performance tests (benchmark before/after)
- [ ] Git commit with detailed message

---

## SUCCESS CRITERIA

**Feature Pipeline**:
- [x] 0 duplications remaining
- [ ] All functions use centralized implementations
- [ ] Test coverage >50%

**Training Infrastructure**:
- [ ] 60/20/20 split implemented
- [ ] Test set used for final evaluation
- [ ] Look-ahead bias verification test passes

**GUI**:
- [ ] training_tab.py split into 6 modules (<400 lines each)
- [ ] pattern_overlay.py split into 4 modules (<600 lines each)
- [ ] All modules have proper __init__.py
- [ ] Original functionality preserved (no regressions)

**Testing**:
- [ ] Coverage >50% (up from ~15%)
- [ ] All critical paths tested
- [ ] pytest runs cleanly

---

## RISKS & MITIGATION

**Risk 1**: Breaking existing functionality during split

**Mitigation**:
- Test after each module extraction
- Keep original file until all tests pass
- Use git branches for each major change

**Risk 2**: Import circular dependencies after split

**Mitigation**:
- Design module hierarchy first
- Base class imports concrete classes (not vice versa)
- Use TYPE_CHECKING for type hints

**Risk 3**: Test coverage goals too ambitious

**Mitigation**:
- Focus on critical paths first
- 50% coverage is minimum (not 80%)
- Prioritize high-value tests

---

## APPENDIX: Example Module Split

**Before** (training_tab.py - single file):
```python
# 2,301 lines in one file
class TrainingTab(QWidget):
    def __init__(self):
        # 100 lines of init
        pass
    
    def _build_ui(self):
        # 500 lines of UI building
        pass
    
    def on_train_clicked(self):
        # 200 lines of training logic
        pass
    
    # ... 50+ more methods ...
```

**After** (modular):
```python
# training_tab/__init__.py
from .training_tab_base import TrainingTab
__all__ = ['TrainingTab']

# training_tab/training_tab_base.py (400 lines)
from .ui_components import UIComponents
from .event_handlers import TrainingEventHandlers

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = UIComponents()
        self.handlers = TrainingEventHandlers(self)
        self.ui.build(self)

# training_tab/ui_components.py (600 lines)
class UIComponents:
    def build(self, parent):
        # UI building logic
        pass

# training_tab/event_handlers.py (500 lines)
class TrainingEventHandlers:
    def on_train_clicked(self):
        # Training logic
        pass
```

---

**READY TO IMPLEMENT!**

**Next Steps**:
1. Start with training_tab.py split (highest impact)
2. Test thoroughly after each module
3. Commit incrementally
4. Move to next P1 task

**Estimated Completion**: 2 weeks (25 hours)
