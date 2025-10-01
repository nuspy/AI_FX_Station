"""
Unified Prediction Settings Dialog - combines Base and Advanced settings in tabs.
Includes all indicators, feature weights, regime detection, candlestick patterns, and comprehensive tooltips.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton,
    QFileDialog, QDialogButtonBox, QSpinBox, QCheckBox, QLabel, QTabWidget,
    QGroupBox, QTextEdit, QScrollArea, QWidget, QComboBox, QGridLayout, QMessageBox
)
from PySide6.QtCore import Signal
from loguru import logger

from ..utils.user_settings import get_setting, set_setting

# Configuration file for settings persistence
CONFIG_FILE = Path(__file__).resolve().parents[3] / "configs" / "prediction_settings.json"

# All available indicators with their descriptions
INDICATORS = {
    'atr': 'Average True Range - Misura la volatilità del mercato. Valori alti indicano alta volatilità, valori bassi indicano bassa volatilità.',
    'rsi': 'Relative Strength Index - Oscillatore di momentum (0-100). <30 = ipervenduto, >70 = ipercomprato. Identifica condizioni di eccesso.',
    'macd': 'Moving Average Convergence Divergence - Indicatore di trend e momentum. Crossover della linea segnale indica cambi di trend.',
    'bollinger': 'Bollinger Bands - Bande di volatilità attorno alla media mobile. Prezzo vicino alle bande indica possibile inversione.',
    'stoch': 'Stochastic Oscillator - Oscillatore di momentum (0-100). <20 = ipervenduto, >80 = ipercomprato. Identifica punti di ingresso/uscita.',
    'cci': 'Commodity Channel Index - Misura deviazione dal prezzo medio. >100 = ipercomprato, <-100 = ipervenduto. Identifica cicli di prezzo.',
    'williams_r': "Williams %R - Oscillatore di momentum (-100 a 0). <-80 = ipervenduto, >-20 = ipercomprato. Simile allo Stochastic.",
    'adx': 'Average Directional Index - Misura la forza del trend (0-100). >25 = trend forte, <20 = trend debole. Non indica direzione.',
    'mfi': 'Money Flow Index - RSI pesato per volume (0-100). <20 = ipervenduto, >80 = ipercomprato. Considera pressione di acquisto/vendita.',
    'obv': 'On Balance Volume - Volume cumulativo basato su direzione prezzo. Conferma trend o segnala divergenze.',
    'trix': 'Triple Exponential Average - Oscillatore di momentum che filtra cicli brevi. Crossover con zero indica cambi di trend.',
    'ultimate': 'Ultimate Oscillator - Combina 3 timeframe per momentum (0-100). <30 = ipervenduto, >70 = ipercomprato. Riduce falsi segnali.',
    'donchian': 'Donchian Channels - Massimo/minimo su N periodi. Breakout delle bande indica forte movimento direzionale.',
    'keltner': 'Keltner Channels - Bande basate su ATR. Simile a Bollinger ma usa volatilità vera. Identifica breakout.',
    'ema': 'Exponential Moving Average - Media mobile esponenziale. Dà più peso ai prezzi recenti. Identifica trend e supporto/resistenza.',
    'sma': 'Simple Moving Average - Media mobile semplice. Livella il prezzo per identificare trend. Supporto/resistenza dinamico.',
    'hurst': 'Hurst Exponent - Misura persistenza del trend (0-1). >0.5 = trending, <0.5 = mean-reverting, =0.5 = random walk.',
    'vwap': 'Volume Weighted Average Price - Prezzo medio ponderato per volume. Benchmark intraday per execution quality.',
}

# Additional features
ADDITIONAL_FEATURES = {
    'returns': 'Returns & Volatility - Rendimenti percentuali e volatilità realizzata. Fondamentali per modelli di prezzo.',
    'sessions': 'Trading Sessions - Identifica sessione attiva (Tokyo/London/NY). Volatilità e comportamento variano per sessione.',
    'candlestick': 'Candlestick Patterns - Pattern candlestick su timeframe superiore. Identifica setup di inversione/continuazione.',
    'volume_profile': 'Volume Profile - Distribuzione volume per livello di prezzo. Identifica aree di supporto/resistenza.',
}

# Tooltips for other parameters
TOOLTIPS = {
    'horizons': '''Orizzonti di previsione - Formati supportati:
• Lista semplice: "1m, 5m, 15m, 1h" (separati da virgola, spazi opzionali)
• Range con stessa unità: "1-5m" → espande a 1m, 2m, 3m, 4m, 5m
• Range con unità esplicite: "1m-5m" → stesso risultato
• Range con step: "15-30m/5m" → espande a 15m, 20m, 25m, 30m
• Range tra unità diverse: "30m-2h" → espande con step 30m (30m, 1h, 1h30m, 2h)
• Range tra unità con step: "1h-5h/30m" → espande ogni 30 minuti
• Mix di formati: "1-5m, 10m, 15-30m/5m, 1h-3h/30m"
Orizzonti più brevi = più reattivi ma rumorosi, più lunghi = più stabili ma lenti.''',
    'n_samples': 'Numero di campioni generati per forecast (1-10000). Più campioni = distribuzione più accurata ma più lento.',
    'quantiles': 'Percentili da calcolare per bande di confidenza (es. 0.05, 0.50, 0.95). 0.50 = mediana, 0.05/0.95 = intervallo 90%.',
    'warmup_bars': 'Numero di barre per inizializzare gli indicatori (10-200). Troppo poche = indicatori instabili, troppe = meno dati training.',
    'rv_window': 'Finestra per realized volatility in minuti (30-240). Più lunga = volatilità più stabile, più corta = più reattiva.',
    'returns_window': 'Finestra per calcolo returns e volatilità (10-200 bars). Influenza smoothing e reattività dei returns.',
    'min_coverage': 'Coverage minima features (0.0-1.0). Features con più NaN di questa soglia vengono scartate. Più alto = meno features.',
    'higher_tf': 'Timeframe superiore per candlestick patterns (5m, 15m, 1h, 4h). Patterns su TF alto = segnali più affidabili.',
    'session_overlap': 'Considera overlap tra sessioni trading. Overlap London/NY ha alta volatilità e volume.',
}


class UnifiedPredictionSettingsDialog(QDialog):
    """
    Unified dialog with Base and Advanced settings in tabs.
    Includes all indicators, feature weights, regime detection, and comprehensive configuration.
    """

    # Signal emitted when settings are saved
    settingsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prediction Settings")
        self.resize(900, 700)

        # Internal state for model paths
        if not hasattr(self.__class__, "_last_model_paths"):
            self.__class__._last_model_paths = []
        if not hasattr(self.__class__, "_last_browse_dir"):
            self.__class__._last_browse_dir = str(Path.home())

        self._model_paths = list(self.__class__._last_model_paths)

        # Create main layout
        main_layout = QVBoxLayout(self)

        # Create tab widget
        self.tabs = QTabWidget()

        # Create Base and Advanced tabs
        self.base_tab = self._create_base_tab()
        self.advanced_tab = self._create_advanced_tab()

        self.tabs.addTab(self.base_tab, "Base Settings")
        self.tabs.addTab(self.advanced_tab, "Advanced Settings")

        main_layout.addWidget(self.tabs)

        # Add Save/Load/Reset buttons
        button_layout = QHBoxLayout()

        self.save_config_btn = QPushButton("Save Configuration")
        self.save_config_btn.clicked.connect(self._save_configuration_file)
        self.save_config_btn.setToolTip("Salva configurazione corrente in un file .json")

        self.load_config_btn = QPushButton("Load Configuration")
        self.load_config_btn.clicked.connect(self._load_configuration_file)
        self.load_config_btn.setToolTip("Carica configurazione da file .json")

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        self.reset_btn.setToolTip("Ripristina tutti i valori ai defaults")

        button_layout.addWidget(self.save_config_btn)
        button_layout.addWidget(self.load_config_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        # Dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        # Load saved settings
        self.load_settings()

    def _create_base_tab(self) -> QWidget:
        """Create the Base settings tab"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        # Model Selection
        models_box = QGroupBox("Model Selection")
        models_layout = QVBoxLayout(models_box)

        self.models_edit = QTextEdit()
        self.models_edit.setPlaceholderText("Model paths (one per line)")
        self.models_edit.setMinimumHeight(80)
        self.models_edit.setMaximumHeight(120)
        models_layout.addWidget(self.models_edit)

        model_buttons = QHBoxLayout()
        self.browse_multi_button = QPushButton("Browse Models")
        self.browse_multi_button.clicked.connect(self._browse_model_paths_multi)
        self.browse_multi_button.setToolTip("Seleziona uno o più file di modello (.pkl, .pt, .pth)")

        self.info_button = QPushButton("Model Info")
        self.info_button.clicked.connect(self._show_model_info)
        self.info_button.setToolTip("Mostra informazioni dettagliate sui modelli selezionati")

        self.loadmeta_button = QPushButton("Load Defaults")
        self.loadmeta_button.clicked.connect(self._load_model_defaults)
        self.loadmeta_button.setToolTip("Carica parametri di default dai metadata del modello")

        model_buttons.addWidget(self.browse_multi_button)
        model_buttons.addWidget(self.info_button)
        model_buttons.addWidget(self.loadmeta_button)
        models_layout.addLayout(model_buttons)

        layout.addWidget(models_box)

        # Forecast Types
        types_box = QGroupBox("Forecast Types")
        types_layout = QVBoxLayout(types_box)

        self.type_basic_cb = QCheckBox("Basic Forecast")
        self.type_basic_cb.setChecked(True)
        self.type_basic_cb.setToolTip("Forecast semplice con mediana e bande di confidenza")

        self.type_advanced_cb = QCheckBox("Advanced Forecast")
        self.type_advanced_cb.setToolTip("Forecast avanzato con analisi multi-scenario e distribuzione completa")

        self.type_rw_cb = QCheckBox("Baseline Random Walk")
        self.type_rw_cb.setToolTip("Baseline per confronto - assume random walk (prezzo futuro = prezzo attuale)")

        types_layout.addWidget(self.type_basic_cb)
        types_layout.addWidget(self.type_advanced_cb)
        types_layout.addWidget(self.type_rw_cb)

        layout.addWidget(types_box)

        # Core Settings
        core_box = QGroupBox("Core Prediction Settings")
        core_form = QFormLayout(core_box)

        self.horizons_edit = QLineEdit("1m, 5m, 15m")
        self.horizons_edit.setToolTip(TOOLTIPS['horizons'])
        core_form.addRow("Horizons:", self.horizons_edit)

        self.n_samples_spinbox = QSpinBox()
        self.n_samples_spinbox.setRange(1, 10000)
        self.n_samples_spinbox.setValue(100)
        self.n_samples_spinbox.setToolTip(TOOLTIPS['n_samples'])
        core_form.addRow("N Samples:", self.n_samples_spinbox)

        self.quantiles_edit = QLineEdit("0.05, 0.50, 0.95")
        self.quantiles_edit.setToolTip(TOOLTIPS['quantiles'])
        core_form.addRow("Quantiles:", self.quantiles_edit)

        layout.addWidget(core_box)
        layout.addStretch()

        scroll.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)

        return tab

    def _create_advanced_tab(self) -> QWidget:
        """Create the Advanced settings tab with all indicators and feature weights"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        # Technical Indicators with Enable/Weight controls (4 columns)
        indicators_box = QGroupBox("Technical Indicators (Enable + Weight 0-100)")
        indicators_layout = QGridLayout(indicators_box)

        self.indicator_controls = {}
        row, col = 0, 0

        for indicator, description in INDICATORS.items():
            # Create container for this indicator
            ind_widget = QWidget()
            ind_layout = QVBoxLayout(ind_widget)
            ind_layout.setContentsMargins(5, 5, 5, 5)

            # Checkbox for enable/disable
            cb = QCheckBox(indicator.upper())
            cb.setChecked(True)
            cb.setToolTip(description)

            # ComboBox for weight (0-100, step 10)
            weight_combo = QComboBox()
            weight_combo.addItems([str(i) for i in range(0, 101, 10)])
            weight_combo.setCurrentText("100")
            weight_combo.setToolTip(f"Peso per {indicator} (0=disabilitato, 100=massimo peso)")

            ind_layout.addWidget(cb)
            ind_layout.addWidget(QLabel("Weight:"))
            ind_layout.addWidget(weight_combo)

            self.indicator_controls[indicator] = {'enabled': cb, 'weight': weight_combo}

            indicators_layout.addWidget(ind_widget, row, col)

            col += 1
            if col >= 4:  # 4 columns
                col = 0
                row += 1

        layout.addWidget(indicators_box)

        # Additional Features
        additional_box = QGroupBox("Additional Features")
        additional_layout = QGridLayout(additional_box)

        self.additional_controls = {}
        row = 0

        for feature, description in ADDITIONAL_FEATURES.items():
            cb = QCheckBox(feature.replace('_', ' ').title())
            cb.setChecked(True)
            cb.setToolTip(description)

            weight_combo = QComboBox()
            weight_combo.addItems([str(i) for i in range(0, 101, 10)])
            weight_combo.setCurrentText("100")
            weight_combo.setToolTip(f"Peso per {feature}")

            self.additional_controls[feature] = {'enabled': cb, 'weight': weight_combo}

            additional_layout.addWidget(cb, row, 0)
            additional_layout.addWidget(QLabel("Weight:"), row, 1)
            additional_layout.addWidget(weight_combo, row, 2)
            row += 1

        layout.addWidget(additional_box)

        # Advanced Parameters
        params_box = QGroupBox("Advanced Parameters")
        params_form = QFormLayout(params_box)

        self.warmup_spinbox = QSpinBox()
        self.warmup_spinbox.setRange(10, 200)
        self.warmup_spinbox.setValue(16)
        self.warmup_spinbox.setToolTip(TOOLTIPS['warmup_bars'])
        params_form.addRow("Warmup Bars:", self.warmup_spinbox)

        self.rv_window_spinbox = QSpinBox()
        self.rv_window_spinbox.setRange(30, 240)
        self.rv_window_spinbox.setValue(60)
        self.rv_window_spinbox.setToolTip(TOOLTIPS['rv_window'])
        params_form.addRow("RV Window (min):", self.rv_window_spinbox)

        self.returns_window_spinbox = QSpinBox()
        self.returns_window_spinbox.setRange(10, 200)
        self.returns_window_spinbox.setValue(100)
        self.returns_window_spinbox.setToolTip(TOOLTIPS['returns_window'])
        params_form.addRow("Returns Window:", self.returns_window_spinbox)

        self.min_coverage_spinbox = QSpinBox()
        self.min_coverage_spinbox.setRange(0, 100)
        self.min_coverage_spinbox.setValue(15)
        self.min_coverage_spinbox.setSuffix("%")
        self.min_coverage_spinbox.setToolTip(TOOLTIPS['min_coverage'])
        params_form.addRow("Min Feature Coverage:", self.min_coverage_spinbox)

        layout.addWidget(params_box)

        # Regime Detection Parameters
        regime_box = QGroupBox("Regime Detection (Trading Sessions)")
        regime_layout = QFormLayout(regime_box)

        self.session_overlap_cb = QCheckBox("Consider Session Overlap")
        self.session_overlap_cb.setChecked(True)
        self.session_overlap_cb.setToolTip(TOOLTIPS['session_overlap'])
        regime_layout.addRow(self.session_overlap_cb)

        layout.addWidget(regime_box)

        # Candlestick Pattern Parameters
        candle_box = QGroupBox("Candlestick Patterns (Higher Timeframe)")
        candle_layout = QFormLayout(candle_box)

        self.higher_tf_combo = QComboBox()
        self.higher_tf_combo.addItems(["5m", "15m", "30m", "1h", "4h"])
        self.higher_tf_combo.setCurrentText("15m")
        self.higher_tf_combo.setToolTip(TOOLTIPS['higher_tf'])
        candle_layout.addRow("Higher Timeframe:", self.higher_tf_combo)

        layout.addWidget(candle_box)

        layout.addStretch()

        scroll.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)

        return tab

    def _browse_model_paths_multi(self):
        """Browse and select multiple model files"""
        last_dir = self.__class__._last_browse_dir

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Model Files",
            last_dir,
            "Model Files (*.pkl *.pickle *.pt *.pth);;All Files (*.*)"
        )

        if files:
            # Update last browse directory
            self.__class__._last_browse_dir = str(Path(files[0]).parent)

            # Update model paths
            self._model_paths = files
            self.__class__._last_model_paths = list(files)

            # Update text edit
            self.models_edit.setPlainText("\n".join(files))

            logger.info(f"Selected {len(files)} model file(s)")

    def _show_model_info(self):
        """Show information about selected models"""
        models = self._get_model_paths()

        if not models:
            QMessageBox.warning(self, "No Models", "No model files selected.")
            return

        from ..models.metadata_manager import MetadataManager

        info_parts = []
        manager = MetadataManager()

        for model_path in models:
            try:
                metadata = manager.load_metadata(model_path)
                if metadata:
                    info_parts.append(f"=== {Path(model_path).name} ===")
                    info_parts.append(f"Type: {metadata.model_type}")
                    info_parts.append(f"Symbol: {metadata.symbol}")
                    info_parts.append(f"Timeframe: {metadata.base_timeframe}")
                    info_parts.append(f"Horizon: {metadata.horizon_bars} bars")
                    info_parts.append(f"Features: {metadata.num_features}")
                    info_parts.append("")
                else:
                    info_parts.append(f"=== {Path(model_path).name} ===")
                    info_parts.append("No metadata found (né sidecar, né embedded)")
                    info_parts.append("")
            except Exception as e:
                info_parts.append(f"=== {Path(model_path).name} ===")
                info_parts.append(f"Error loading metadata: {e}")
                info_parts.append("")

        info_text = "\n".join(info_parts)

        # Show in dialog
        msg = QMessageBox(self)
        msg.setWindowTitle("Model Information")
        msg.setText(info_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec()

    def _load_model_defaults(self):
        """Load default settings from model metadata"""
        models = self._get_model_paths()

        if not models:
            QMessageBox.warning(self, "No Models", "No model files selected.")
            return

        from ..models.metadata_manager import MetadataManager

        manager = MetadataManager()
        model_path = models[0]  # Use first model

        try:
            metadata = manager.load_metadata(model_path)

            if not metadata:
                QMessageBox.warning(
                    self,
                    "No Metadata",
                    "Nessun modello trovato (né sidecar, né embedded). Verifica sia creazione che lettura seguano modello attuale."
                )
                return

            # Load settings from metadata
            if metadata.horizon_bars:
                # Convert bars to time format
                horizons = f"{metadata.horizon_bars}m"  # Simplified
                self.horizons_edit.setText(horizons)

            if hasattr(metadata, 'preprocessing_config') and metadata.preprocessing_config:
                config = metadata.preprocessing_config
                if 'warmup_bars' in config:
                    self.warmup_spinbox.setValue(int(config['warmup_bars']))
                if 'rv_window' in config:
                    self.rv_window_spinbox.setValue(int(config['rv_window']))

            QMessageBox.information(self, "Success", f"Loaded defaults from {Path(model_path).name}")

        except Exception as e:
            logger.exception("Failed to load model defaults")
            QMessageBox.warning(self, "Error", f"Failed to load defaults: {e}")

    def _get_model_paths(self) -> List[str]:
        """Get model paths from text edit"""
        text = self.models_edit.toPlainText().strip()
        if not text:
            return []

        paths = [line.strip() for line in text.split('\n') if line.strip()]
        return paths

    def _save_configuration_file(self):
        """Save current configuration to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            str(Path.home() / "prediction_config.json"),
            "JSON Files (*.json);;All Files (*.*)"
        )

        if file_path:
            try:
                config = self.get_settings()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                QMessageBox.information(self, "Success", f"Configuration saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save configuration: {e}")

    def _load_configuration_file(self):
        """Load configuration from a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            str(Path.home()),
            "JSON Files (*.json);;All Files (*.*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.set_settings(config)
                QMessageBox.information(self, "Success", f"Configuration loaded from {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load configuration: {e}")

    def _reset_to_defaults(self):
        """Reset all settings to default values"""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Reset all settings to default values?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.set_settings({})
            QMessageBox.information(self, "Success", "Settings reset to defaults")

    def _on_accept(self):
        """Handle OK button - save settings and emit signal"""
        self.save_settings()
        self.settingsChanged.emit()
        self.accept()

    def get_settings(self) -> Dict[str, Any]:
        """Get all current settings as dictionary"""
        settings = {
            # Model paths
            'model_paths': self._get_model_paths(),

            # Forecast types
            'type_basic': self.type_basic_cb.isChecked(),
            'type_advanced': self.type_advanced_cb.isChecked(),
            'type_rw': self.type_rw_cb.isChecked(),

            # Core settings
            'horizons': self.horizons_edit.text(),
            'n_samples': self.n_samples_spinbox.value(),
            'quantiles': self.quantiles_edit.text(),

            # Indicators
            'indicators': {},

            # Additional features
            'additional_features': {},

            # Advanced parameters
            'warmup_bars': self.warmup_spinbox.value(),
            'rv_window': self.rv_window_spinbox.value(),
            'returns_window': self.returns_window_spinbox.value(),
            'min_coverage': self.min_coverage_spinbox.value() / 100.0,

            # Regime detection
            'session_overlap': self.session_overlap_cb.isChecked(),

            # Candlestick patterns
            'higher_tf': self.higher_tf_combo.currentText(),
        }

        # Collect indicator settings
        for name, controls in self.indicator_controls.items():
            settings['indicators'][name] = {
                'enabled': controls['enabled'].isChecked(),
                'weight': int(controls['weight'].currentText())
            }

        # Collect additional feature settings
        for name, controls in self.additional_controls.items():
            settings['additional_features'][name] = {
                'enabled': controls['enabled'].isChecked(),
                'weight': int(controls['weight'].currentText())
            }

        return settings

    def set_settings(self, settings: Dict[str, Any]):
        """Apply settings from dictionary"""
        # Model paths (handle both old and new formats)
        if 'model_paths' in settings and settings['model_paths']:
            paths = settings['model_paths']
            if isinstance(paths, list):
                self.models_edit.setPlainText("\n".join(paths))
                self._model_paths = paths
            elif isinstance(paths, str):
                self.models_edit.setPlainText(paths)
                self._model_paths = [p.strip() for p in paths.split('\n') if p.strip()]
        elif 'model_path' in settings and settings['model_path']:
            # Legacy single path
            path = settings['model_path']
            self.models_edit.setPlainText(path)
            self._model_paths = [path]

        # Forecast types
        self.type_basic_cb.setChecked(settings.get('type_basic', True))
        self.type_advanced_cb.setChecked(settings.get('type_advanced', False))
        self.type_rw_cb.setChecked(settings.get('type_rw', False))

        # Core settings (handle both string and list formats)
        horizons = settings.get('horizons', '1m, 5m, 15m')
        if isinstance(horizons, list):
            horizons = ', '.join(horizons)
        self.horizons_edit.setText(horizons)

        self.n_samples_spinbox.setValue(settings.get('n_samples', 100))

        quantiles = settings.get('quantiles', '0.05, 0.50, 0.95')
        if isinstance(quantiles, list):
            quantiles = ', '.join(str(q) for q in quantiles)
        self.quantiles_edit.setText(quantiles)

        # Indicators
        if 'indicators' in settings:
            for name, config in settings['indicators'].items():
                if name in self.indicator_controls:
                    self.indicator_controls[name]['enabled'].setChecked(config.get('enabled', True))
                    self.indicator_controls[name]['weight'].setCurrentText(str(config.get('weight', 100)))

        # Additional features
        if 'additional_features' in settings:
            for name, config in settings['additional_features'].items():
                if name in self.additional_controls:
                    self.additional_controls[name]['enabled'].setChecked(config.get('enabled', True))
                    self.additional_controls[name]['weight'].setCurrentText(str(config.get('weight', 100)))

        # Advanced parameters
        self.warmup_spinbox.setValue(settings.get('warmup_bars', 16))
        self.rv_window_spinbox.setValue(settings.get('rv_window', 60))
        self.returns_window_spinbox.setValue(settings.get('returns_window', 100))
        self.min_coverage_spinbox.setValue(int(settings.get('min_coverage', 0.15) * 100))

        # Regime detection
        self.session_overlap_cb.setChecked(settings.get('session_overlap', True))

        # Candlestick patterns
        self.higher_tf_combo.setCurrentText(settings.get('higher_tf', '15m'))

    def save_settings(self):
        """Save settings to persistent storage"""
        try:
            settings = self.get_settings()

            # Save to config file
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)

            # Also save last browse directory
            set_setting('last_model_browse_dir', self.__class__._last_browse_dir)

            logger.info(f"Settings saved to {CONFIG_FILE}")

        except Exception as e:
            logger.exception("Failed to save settings")

    def load_settings(self):
        """Load settings from persistent storage"""
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                self.set_settings(settings)
                logger.info(f"Settings loaded from {CONFIG_FILE}")

            # Load last browse directory
            last_dir = get_setting('last_model_browse_dir')
            if last_dir:
                self.__class__._last_browse_dir = last_dir

        except Exception as e:
            logger.exception("Failed to load settings")

    @staticmethod
    def get_model_paths() -> List[str]:
        """Return last selected model paths (class method for compatibility)"""
        try:
            return list(getattr(UnifiedPredictionSettingsDialog, "_last_model_paths", []) or [])
        except Exception:
            return []

    @staticmethod
    def get_settings() -> Optional[Dict[str, Any]]:
        """Load and return current settings from persistent storage (static method for compatibility)"""
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                # Backward compatibility: add model_path and parallel_inference flag
                if 'model_paths' in settings and settings['model_paths']:
                    paths = settings['model_paths']
                    if isinstance(paths, list):
                        if len(paths) > 0:
                            settings['model_path'] = paths[0]  # First path for backward compatibility
                        if len(paths) > 1:
                            # Multiple models: enable parallel inference
                            settings['parallel_inference'] = True
                            logger.info(f"Parallel inference enabled for {len(paths)} models")
                    elif isinstance(paths, str):
                        settings['model_path'] = paths

                return settings
            return {}
        except Exception as e:
            logger.exception("Failed to load settings statically")
            return {}
