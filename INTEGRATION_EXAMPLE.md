# i18n Integration Example

## Before (Hardcoded)

```python
lbl_name = QLabel("Model Name:")
lbl_name.setToolTip(
    "Nome del modello salvato.\n"
    "Se impostato: il modello sarà salvato con questo nome.\n"
    "Se vuoto: il nome sarà generato automaticamente dall'elenco delle features.\n"
    "Esempio: 'EUR_USD_1h_ridge_multiTF' o 'my_custom_model_v2'"
)
```

## After (i18n)

```python
from ..i18n.widget_helper import create_label, apply_tooltip

# Method 1: Create label with i18n
lbl_name = create_label("model_name", category="training")

# Method 2: Apply to existing widget
lbl_name = QLabel("Model Name:")
apply_tooltip(lbl_name, "model_name", category="training")

# Method 3: Manual tr() call
from ..i18n import tr
lbl_name = QLabel(tr("training.model_name.label"))
lbl_name.setToolTip(tr("training.model_name.tooltip"))
```

## Full Integration Steps

### 1. Import i18n
```python
from ..i18n import tr
from ..i18n.widget_helper import create_label, apply_tooltip
```

### 2. Replace Labels
```python
# Old
lbl_sym = QLabel("Symbol:")
lbl_sym.setToolTip("Coppia valutaria da usare...")

# New
lbl_sym = create_label("symbol", category="training")
```

### 3. Replace Widget Tooltips
```python
# Old
self.symbol_combo.setToolTip("Seleziona il simbolo...")

# New
apply_tooltip(self.symbol_combo, "symbol", category="training")
```

### 4. Replace Indicator Tooltips
```python
# Old
INDICATOR_TOOLTIPS = {
    "ATR": "Average True Range - Misura la volatilità...",
    # ...
}

# New - Use tr() function
for indicator in INDICATORS:
    tooltip = tr(f"training.indicators.{indicator.lower()}.tooltip")
    self.indicator_checks[indicator].setToolTip(tooltip)
```

## Migration Strategy

1. **Phase 1**: Training Tab (77 tooltips) - Complete example
2. **Phase 2**: Other tabs (178 tooltips) - Use same pattern
3. **Phase 3**: Add language selector
4. **Phase 4**: Add Italian/Spanish translations

## Benefits

- ✅ All UI text in one place (JSON files)
- ✅ Easy to add new languages
- ✅ Consistent tooltips across app
- ✅ Professional multi-language support
- ✅ Centralized maintenance
