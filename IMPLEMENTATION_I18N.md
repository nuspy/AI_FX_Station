# i18n Implementation Guide for ForexGPT

## Sistema Implementato

### Struttura File

```
src/forex_diffusion/i18n/
├── __init__.py                    # Main i18n module
└── translations/
    ├── en_US.json                 # English (default)
    ├── it_IT.json                 # Italian
    ├── es_ES.json                 # Spanish (TODO)
    ├── fr_FR.json                 # French (TODO)
    ├── de_DE.json                 # German (TODO)
    ├── ja_JP.json                 # Japanese (TODO)
    └── zh_CN.json                 # Chinese (TODO)
```

### Funzioni Principali

```python
from forex_diffusion.i18n import tr, set_language, get_available_languages

# Translate text
label = tr("training.symbol.label")  # Returns: "Symbol:"

# Get full tooltip
tooltip = tr("training.symbol.tooltip")  # Returns complete 6-section tooltip

# Change language
set_language("it_IT")  # Switch to Italian

# Get available languages
languages = get_available_languages()  # Returns: ["en_US", "it_IT", ...]
```

## Come Usare nell'UI

### Before (Old Code):

```python
# training_tab.py
lbl = QLabel("Symbol:")
lbl.setToolTip("Select the currency pair to train...")
```

### After (New i18n Code):

```python
# training_tab.py
from forex_diffusion.i18n import tr

lbl = QLabel(tr("training.symbol.label"))
lbl.setToolTip(tr("training.symbol.tooltip"))
```

## Formato Tooltip a 6 Sezioni

Ogni tooltip segue questo schema nel file JSON:

```json
"training.symbol.tooltip": "Symbol Selector\n\n1) WHAT IT IS:\n...\n\n2) HOW AND WHEN TO USE:\n...\n\n3) WHY TO USE IT:\n...\n\n4) EFFECTS:\n\n4.1) LOW VALUE / DISABLED:\n...\n\n4.2) MEDIUM VALUE:\n...\n\n4.3) HIGH VALUE / ENABLED:\n...\n\n5) TYPICAL RANGE / DEFAULT VALUES:\n...\n\n6) ADDITIONAL NOTES / BEST PRACTICES:\n..."
```

## Prossimi Passi

### 1. Aggiornare tutti i file UI

Sostituire stringhe hardcoded con chiamate a `tr()`:

**File da aggiornare** (in ordine priorità):
1. `src/forex_diffusion/ui/training_tab.py` (100+ stringhe)
2. `src/forex_diffusion/ui/chart_tab/ui_builder.py` (50+ stringhe)
3. `src/forex_diffusion/ui/backtesting_tab.py` (50+ stringhe)
4. `src/forex_diffusion/ui/pattern_training_tab.py` (30+ stringhe)
5. `src/forex_diffusion/ui/portfolio_tab.py` (20+ stringhe)
6. `src/forex_diffusion/ui/signals_tab.py` (15+ stringhe)
7. `src/forex_diffusion/ui/live_trading_tab.py` (40+ stringhe)
8. Tutti gli altri file UI

### 2. Completare traduzioni JSON

Aggiungere TUTTI i parametri a `en_US.json` e `it_IT.json`:

**Parametri da aggiungere**:
- Training Tab: ~100 parametri (symbol, timeframe, days, horizon, model, encoder, optimization, indicators, etc.)
- Chart Tab: ~50 parametri (drawing tools, timeframes, symbols, VIX, order books, etc.)
- Backtesting Tab: ~50 parametri
- Pattern Training: ~30 parametri
- Altri tab: ~70 parametri

**Totale stimato**: 300-400 chiavi di traduzione

### 3. Aggiungere Language Selector nell'UI

Creare menu per cambiare lingua:

```python
# In menu bar or settings
def create_language_menu(self):
    lang_menu = self.menuBar().addMenu(tr("menu.language"))
    
    for lang_code in get_available_languages():
        action = lang_menu.addAction(lang_code)
        action.triggered.connect(lambda checked, l=lang_code: self.change_language(l))

def change_language(self, lang_code):
    if set_language(lang_code):
        # Reload UI text (requires restart or manual refresh)
        QMessageBox.information(
            self, 
            tr("common.info"),
            tr("language.changed_restart_required")
        )
```

### 4. Aggiungere altre lingue

Copiare `en_US.json` e tradurre:
- `es_ES.json` (Spanish)
- `fr_FR.json` (French)
- `de_DE.json` (German)
- `ja_JP.json` (Japanese)
- `zh_CN.json` (Chinese Simplified)

## Vantaggi Sistema i18n

1. **Centralizzazione**: Tutte le stringhe in un posto (facile aggiornare)
2. **Tooltip Completi**: Schema a 6 punti garantito per tutti i parametri
3. **Multi-lingua**: Espansione facile a nuove lingue
4. **Manutenzione**: Modifiche rapide senza toccare codice UI
5. **Collaborazione**: Traduttori possono lavorare su JSON senza toccare Python
6. **Qualità**: Review tooltip più facile (tutto in JSON leggibile)

## Esempio Completo

### JSON Translation (en_US.json):

```json
{
  "training.days.label": "Days:",
  "training.days.tooltip": "Days History\n\n1) WHAT IT IS:\nNumber of historical days to include in training dataset...\n\n2) HOW AND WHEN TO USE:\nUse more days (90-365) for production models...\n\n3) WHY TO USE IT:\nMore data improves generalization...\n\n4) EFFECTS:\n\n4.1) LOW VALUE (1-7 days):\nFast training, risk of overfitting...\n\n4.2) MEDIUM VALUE (30-90 days):\nBalanced quality and speed...\n\n4.3) HIGH VALUE (365+ days):\nBest generalization, slow training...\n\n5) TYPICAL RANGE:\n- Beginner: 7-30 days\n- Intermediate: 30-90 days\n- Advanced: 90-365+ days\n- DEFAULT: 7 days\n\n6) ADDITIONAL NOTES:\nWith 1m timeframe, 7 days = ~10K samples..."
}
```

### UI Code:

```python
from forex_diffusion.i18n import tr

# Label
lbl_days = QLabel(tr("training.days.label"))

# Tooltip
self.days_spin.setToolTip(tr("training.days.tooltip"))
```

### Result:
- English user sees: "Days:" label with full English tooltip
- Italian user sees: "Giorni:" label with full Italian tooltip
- No code changes needed to add Spanish/French/etc.

## Migration Plan

**Phase 1** (Week 1):
- ✅ Create i18n module
- ✅ Create en_US.json template
- ✅ Create it_IT.json template
- Add 50 most important keys (Training Tab top parameters)

**Phase 2** (Week 2):
- Migrate Training Tab to use tr()
- Add all Training Tab tooltips (100+ keys)
- Test language switching

**Phase 3** (Week 3):
- Migrate Chart Tab to use tr()
- Add all Chart Tab tooltips (50+ keys)

**Phase 4** (Week 4):
- Migrate remaining tabs
- Complete all 300-400 keys
- Add language selector menu

**Phase 5** (Future):
- Add Spanish, French, German translations
- Community contributions for other languages
