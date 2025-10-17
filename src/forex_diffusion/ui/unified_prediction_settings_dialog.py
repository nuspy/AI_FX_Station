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

        # Create Base, Advanced, and LDM4TS tabs
        self.base_tab = self._create_base_tab()
        self.advanced_tab = self._create_advanced_tab()
        self.ldm4ts_tab = self._create_ldm4ts_tab()

        self.tabs.addTab(self.base_tab, "Base Settings")
        self.tabs.addTab(self.advanced_tab, "Advanced Settings")
        self.tabs.addTab(self.ldm4ts_tab, "LDM4TS (Vision)")

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

        # Initialize GPU info label
        self._update_gpu_info_label()

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

        # Model Combination Settings
        combination_box = QGroupBox("Model Combination")
        combination_layout = QVBoxLayout(combination_box)

        self.combine_models_cb = QCheckBox("Combina modelli (Ensemble)")
        self.combine_models_cb.setChecked(True)  # Default: ensemble attivo
        self.combine_models_cb.setToolTip(
            "Se attivo: combina tutti i modelli in un unico forecast (media ponderata).\n"
            "Se disattivo: genera forecast separati per ogni modello, visualizzati individualmente."
        )
        combination_layout.addWidget(self.combine_models_cb)

        # GPU Inference Checkbox
        self.use_gpu_inference_cb = QCheckBox("Usa GPU per inference")
        self.use_gpu_inference_cb.setChecked(False)  # Default: CPU

        # Check if CUDA is available
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            self.use_gpu_inference_cb.setEnabled(cuda_available)

            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                self.use_gpu_inference_cb.setToolTip(
                    f"Usa GPU per inference dei modelli PyTorch.\n\n"
                    f"GPU rilevata: {gpu_name}\n\n"
                    f"⚠️ LIMITAZIONE GPU:\n"
                    f"• Con GPU attiva: usa SOLO il primo modello\n"
                    f"• Con GPU disattiva: usa TUTTI i modelli in parallelo (CPU)\n\n"
                    f"Speedup singolo modello GPU: ~5x vs CPU\n"
                    f"Ma ensemble multi-modello CPU può essere più accurato\n\n"
                    f"Raccomandazioni:\n"
                    f"• GPU ON: predizioni veloci, singolo modello\n"
                    f"• GPU OFF: predizioni ensemble, più accurato ma lento\n\n"
                    f"NOTA: solo modelli PyTorch/Diffusion usano GPU.\n"
                    f"Modelli sklearn rimangono sempre su CPU."
                )
            else:
                self.use_gpu_inference_cb.setToolTip(
                    "GPU non disponibile.\n\n"
                    "Per usare GPU:\n"
                    "1. Installa CUDA-enabled PyTorch\n"
                    "2. Riavvia l'applicazione"
                )
        except Exception:
            self.use_gpu_inference_cb.setEnabled(False)
            self.use_gpu_inference_cb.setToolTip("PyTorch non disponibile")

        combination_layout.addWidget(self.use_gpu_inference_cb)

        # Add info label for GPU/parallel limitation
        self.gpu_info_label = QLabel()
        self.gpu_info_label.setStyleSheet("color: #888; font-style: italic; padding: 5px;")
        combination_layout.addWidget(self.gpu_info_label)

        # Connect checkbox to update info label
        self.use_gpu_inference_cb.toggled.connect(self._update_gpu_info_label)

        layout.addWidget(combination_box)

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

        # Helper to create parameter row with override checkbox
        def add_param_with_override(label: str, spinbox: QSpinBox, tooltip: str) -> QCheckBox:
            """Add a parameter row with [Override] checkbox + spinbox"""
            row_layout = QHBoxLayout()
            override_cb = QCheckBox("Override")
            override_cb.setChecked(True)  # Default: override enabled (single model behavior)
            override_cb.setToolTip("Se checked: usa valore qui sotto. Se unchecked: usa metadata del modello (solo multi-modello).")
            spinbox.setToolTip(tooltip)
            row_layout.addWidget(override_cb)
            row_layout.addWidget(spinbox)
            row_layout.addStretch()

            # Enable/disable spinbox based on override checkbox
            spinbox.setEnabled(override_cb.isChecked())
            override_cb.toggled.connect(spinbox.setEnabled)

            params_form.addRow(label, row_layout)
            return override_cb

        self.warmup_spinbox = QSpinBox()
        self.warmup_spinbox.setRange(10, 200)
        self.warmup_spinbox.setValue(16)
        self.warmup_override_cb = add_param_with_override("Warmup Bars:", self.warmup_spinbox, TOOLTIPS['warmup_bars'])

        self.rv_window_spinbox = QSpinBox()
        self.rv_window_spinbox.setRange(30, 240)
        self.rv_window_spinbox.setValue(60)
        self.rv_window_override_cb = add_param_with_override("RV Window (min):", self.rv_window_spinbox, TOOLTIPS['rv_window'])

        self.returns_window_spinbox = QSpinBox()
        self.returns_window_spinbox.setRange(10, 200)
        self.returns_window_spinbox.setValue(100)
        self.returns_window_override_cb = add_param_with_override("Returns Window:", self.returns_window_spinbox, TOOLTIPS['returns_window'])

        self.min_coverage_spinbox = QSpinBox()
        self.min_coverage_spinbox.setRange(0, 100)
        self.min_coverage_spinbox.setValue(15)
        self.min_coverage_spinbox.setSuffix("%")
        self.min_coverage_override_cb = add_param_with_override("Min Feature Coverage:", self.min_coverage_spinbox, TOOLTIPS['min_coverage'])

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

    def _create_ldm4ts_tab(self) -> QWidget:
        """Create the LDM4TS (Vision-Enhanced Forecasting) settings tab"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        # Header info
        info_box = QGroupBox("About LDM4TS")
        info_layout = QVBoxLayout(info_box)
        info_label = QLabel(
            "<b>LDM4TS</b> (Latent Diffusion Models for Time Series) - Vision-enhanced forecasting<br><br>"
            "Trasforma OHLCV in immagini RGB (SEG + GAF + RP) e usa Stable Diffusion per predizioni.<br>"
            "Fornisce incertezza quantificata tramite Monte Carlo sampling.<br><br>"
            "<b>Paper:</b> https://arxiv.org/html/2502.14887v1"
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        layout.addWidget(info_box)

        # Enable/Disable
        enable_box = QGroupBox("Enable LDM4TS")
        enable_layout = QVBoxLayout(enable_box)
        
        self.ldm4ts_enabled_cb = QCheckBox("Enable LDM4TS Forecasting")
        self.ldm4ts_enabled_cb.setChecked(False)
        self.ldm4ts_enabled_cb.setToolTip(
            "Abilita LDM4TS per forecasting vision-enhanced.\n"
            "Richiede modello trainato (checkpoint .pt)."
        )
        enable_layout.addWidget(self.ldm4ts_enabled_cb)
        layout.addWidget(enable_box)

        # Model Settings
        model_box = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_box)

        # Checkpoint path
        checkpoint_layout = QHBoxLayout()
        self.ldm4ts_checkpoint_edit = QLineEdit()
        self.ldm4ts_checkpoint_edit.setPlaceholderText("Path to LDM4TS checkpoint (.pt)")
        self.ldm4ts_checkpoint_edit.setToolTip(
            "Percorso al file checkpoint del modello LDM4TS trainato.\n"
            "Generato da: python -m forex_diffusion.training.train_ldm4ts"
        )
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_ldm4ts_checkpoint)
        browse_btn.setToolTip("Seleziona file checkpoint (.pt)")
        
        checkpoint_layout.addWidget(self.ldm4ts_checkpoint_edit)
        checkpoint_layout.addWidget(browse_btn)
        model_layout.addRow("Checkpoint:", checkpoint_layout)

        # Horizons (multi-horizon predictions)
        self.ldm4ts_horizons_edit = QLineEdit("15, 60, 240")
        self.ldm4ts_horizons_edit.setToolTip(
            "Orizzonti di previsione in minuti (separati da virgola).\n"
            "Esempio: 15, 60, 240 = 15min, 1h, 4h\n"
            "Devono corrispondere agli orizzonti usati in training."
        )
        model_layout.addRow("Horizons (minutes):", self.ldm4ts_horizons_edit)

        layout.addWidget(model_box)

        # Inference Settings
        inference_box = QGroupBox("Inference Settings")
        inference_layout = QFormLayout(inference_box)

        # Number of Monte Carlo samples
        self.ldm4ts_num_samples_spinbox = QSpinBox()
        self.ldm4ts_num_samples_spinbox.setRange(10, 200)
        self.ldm4ts_num_samples_spinbox.setValue(50)
        self.ldm4ts_num_samples_spinbox.setToolTip(
            "Numero di campioni Monte Carlo per quantificare incertezza.\n"
            "Più campioni = incertezza più accurata ma più lento.\n"
            "Range: 10-200, Default: 50, Paper: 50"
        )
        inference_layout.addRow("Monte Carlo Samples:", self.ldm4ts_num_samples_spinbox)

        # Window size for vision encoding
        self.ldm4ts_window_size_spinbox = QSpinBox()
        self.ldm4ts_window_size_spinbox.setRange(50, 200)
        self.ldm4ts_window_size_spinbox.setValue(100)
        self.ldm4ts_window_size_spinbox.setToolTip(
            "Numero di candles OHLCV per vision encoding.\n"
            "Deve corrispondere al window_size usato in training.\n"
            "Default: 100 candles"
        )
        inference_layout.addRow("OHLCV Window Size:", self.ldm4ts_window_size_spinbox)

        layout.addWidget(inference_box)

        # Signal Generation Settings
        signal_box = QGroupBox("Signal Generation")
        signal_layout = QFormLayout(signal_box)

        # Uncertainty threshold
        self.ldm4ts_uncertainty_threshold_spinbox = QSpinBox()
        self.ldm4ts_uncertainty_threshold_spinbox.setRange(10, 100)
        self.ldm4ts_uncertainty_threshold_spinbox.setValue(50)
        self.ldm4ts_uncertainty_threshold_spinbox.setSuffix("%")
        self.ldm4ts_uncertainty_threshold_spinbox.setToolTip(
            "Soglia massima di incertezza per accettare segnali.\n"
            "Segnali con incertezza >= threshold vengono rejettati.\n"
            "Range: 10-100%, Default: 50%"
        )
        signal_layout.addRow("Uncertainty Threshold:", self.ldm4ts_uncertainty_threshold_spinbox)

        # Minimum strength
        self.ldm4ts_min_strength_spinbox = QSpinBox()
        self.ldm4ts_min_strength_spinbox.setRange(10, 90)
        self.ldm4ts_min_strength_spinbox.setValue(30)
        self.ldm4ts_min_strength_spinbox.setSuffix("%")
        self.ldm4ts_min_strength_spinbox.setToolTip(
            "Forza minima del segnale per esecuzione.\n"
            "Calcolata come: |price_change| / uncertainty\n"
            "Range: 10-90%, Default: 30%"
        )
        signal_layout.addRow("Min Signal Strength:", self.ldm4ts_min_strength_spinbox)

        # Position scaling
        self.ldm4ts_position_scaling_cb = QCheckBox("Enable Position Scaling")
        self.ldm4ts_position_scaling_cb.setChecked(True)
        self.ldm4ts_position_scaling_cb.setToolTip(
            "Scala la posizione in base all'incertezza.\n"
            "position_size = base_size × (1 - uncertainty / threshold)\n"
            "Alta incertezza → posizione più piccola"
        )
        signal_layout.addRow(self.ldm4ts_position_scaling_cb)

        layout.addWidget(signal_box)

        # Horizon Weights (for combining multi-horizon predictions)
        weights_box = QGroupBox("Horizon Weights (Multi-Horizon Combination)")
        weights_layout = QFormLayout(weights_box)

        self.ldm4ts_horizon_15m_weight_spinbox = QSpinBox()
        self.ldm4ts_horizon_15m_weight_spinbox.setRange(0, 100)
        self.ldm4ts_horizon_15m_weight_spinbox.setValue(30)
        self.ldm4ts_horizon_15m_weight_spinbox.setSuffix("%")
        self.ldm4ts_horizon_15m_weight_spinbox.setToolTip("Peso per previsioni a 15 minuti (reattivo)")
        weights_layout.addRow("15-min Horizon Weight:", self.ldm4ts_horizon_15m_weight_spinbox)

        self.ldm4ts_horizon_1h_weight_spinbox = QSpinBox()
        self.ldm4ts_horizon_1h_weight_spinbox.setRange(0, 100)
        self.ldm4ts_horizon_1h_weight_spinbox.setValue(50)
        self.ldm4ts_horizon_1h_weight_spinbox.setSuffix("%")
        self.ldm4ts_horizon_1h_weight_spinbox.setToolTip("Peso per previsioni a 1 ora (bilanciato)")
        weights_layout.addRow("1-hour Horizon Weight:", self.ldm4ts_horizon_1h_weight_spinbox)

        self.ldm4ts_horizon_4h_weight_spinbox = QSpinBox()
        self.ldm4ts_horizon_4h_weight_spinbox.setRange(0, 100)
        self.ldm4ts_horizon_4h_weight_spinbox.setValue(20)
        self.ldm4ts_horizon_4h_weight_spinbox.setSuffix("%")
        self.ldm4ts_horizon_4h_weight_spinbox.setToolTip("Peso per previsioni a 4 ore (trend)")
        weights_layout.addRow("4-hour Horizon Weight:", self.ldm4ts_horizon_4h_weight_spinbox)

        layout.addWidget(weights_box)

        # Quality Thresholds
        quality_box = QGroupBox("Quality Thresholds")
        quality_layout = QFormLayout(quality_box)

        self.ldm4ts_quality_threshold_spinbox = QSpinBox()
        self.ldm4ts_quality_threshold_spinbox.setRange(30, 95)
        self.ldm4ts_quality_threshold_spinbox.setValue(65)
        self.ldm4ts_quality_threshold_spinbox.setSuffix("%")
        self.ldm4ts_quality_threshold_spinbox.setToolTip(
            "Composite quality score minimo per signal acceptance.\n"
            "Combina 6 dimensioni: pattern, regime, risk/reward, etc."
        )
        quality_layout.addRow("Min Quality Score:", self.ldm4ts_quality_threshold_spinbox)

        self.ldm4ts_directional_confidence_spinbox = QSpinBox()
        self.ldm4ts_directional_confidence_spinbox.setRange(50, 95)
        self.ldm4ts_directional_confidence_spinbox.setValue(60)
        self.ldm4ts_directional_confidence_spinbox.setSuffix("%")
        self.ldm4ts_directional_confidence_spinbox.setToolTip(
            "Confidence minima per segnali direzionali (bull/bear).\n"
            "Basata su % di campioni MC concordi sulla direzione."
        )
        quality_layout.addRow("Directional Confidence:", self.ldm4ts_directional_confidence_spinbox)

        layout.addWidget(quality_box)

        # Status/Info
        status_box = QGroupBox("Status")
        status_layout = QVBoxLayout(status_box)
        
        self.ldm4ts_status_label = QLabel("LDM4TS: Disabled")
        self.ldm4ts_status_label.setStyleSheet("color: #888; font-style: italic;")
        status_layout.addWidget(self.ldm4ts_status_label)
        
        layout.addWidget(status_box)

        # Connect enable checkbox to update status
        self.ldm4ts_enabled_cb.toggled.connect(self._update_ldm4ts_status)
        
        # Connect all LDM4TS widgets to auto-save
        self.ldm4ts_enabled_cb.toggled.connect(self._auto_save_ldm4ts_settings)
        self.ldm4ts_checkpoint_edit.textChanged.connect(self._auto_save_ldm4ts_settings)
        self.ldm4ts_horizons_edit.textChanged.connect(self._auto_save_ldm4ts_settings)
        self.ldm4ts_num_samples_spinbox.valueChanged.connect(self._auto_save_ldm4ts_settings)
        self.ldm4ts_window_size_spinbox.valueChanged.connect(self._auto_save_ldm4ts_settings)
        self.ldm4ts_uncertainty_threshold_spinbox.valueChanged.connect(self._auto_save_ldm4ts_settings)
        self.ldm4ts_min_strength_spinbox.valueChanged.connect(self._auto_save_ldm4ts_settings)

        layout.addStretch()

        scroll.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)

        return tab

    def _browse_ldm4ts_checkpoint(self):
        """Browse for LDM4TS checkpoint file"""
        # Base directory: parent of src (D:\Projects\ForexGPT)
        base_dir = Path(__file__).resolve().parents[3]
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LDM4TS Checkpoint",
            str(base_dir),
            "PyTorch Models (*.pt *.pth);;All Files (*.*)"
        )
        
        if file_path:
            self.ldm4ts_checkpoint_edit.setText(file_path)
            logger.info(f"Selected LDM4TS checkpoint: {file_path}")
            # Auto-save settings
            self._auto_save_ldm4ts_settings()

    def _auto_save_ldm4ts_settings(self):
        """Auto-save LDM4TS settings (inter and intra-session persistence)"""
        try:
            # Get current LDM4TS settings
            ldm4ts_settings = {
                'ldm4ts_enabled': self.ldm4ts_enabled_cb.isChecked(),
                'ldm4ts_checkpoint_path': self.ldm4ts_checkpoint_edit.text().strip(),
                'ldm4ts_horizons': self.ldm4ts_horizons_edit.text().strip(),
                'ldm4ts_num_samples': self.ldm4ts_num_samples_spinbox.value(),
                'ldm4ts_window_size': self.ldm4ts_window_size_spinbox.value(),
                'ldm4ts_uncertainty_threshold': self.ldm4ts_uncertainty_threshold_spinbox.value(),
                'ldm4ts_min_strength': self.ldm4ts_min_strength_spinbox.value(),
            }
            
            # Load existing settings
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            else:
                settings = {}
            
            # Update LDM4TS settings
            settings.update(ldm4ts_settings)
            
            # Save back
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
            
            logger.debug("LDM4TS settings auto-saved")
            
        except Exception as e:
            logger.exception(f"Failed to auto-save LDM4TS settings: {e}")
    
    def _update_ldm4ts_status(self):
        """Update LDM4TS status label"""
        if self.ldm4ts_enabled_cb.isChecked():
            checkpoint = self.ldm4ts_checkpoint_edit.text().strip()
            if checkpoint and Path(checkpoint).exists():
                self.ldm4ts_status_label.setText(f"LDM4TS: ✅ Enabled - Model: {Path(checkpoint).name}")
                self.ldm4ts_status_label.setStyleSheet("color: green;")
            else:
                self.ldm4ts_status_label.setText("LDM4TS: ⚠️ Enabled but no valid checkpoint")
                self.ldm4ts_status_label.setStyleSheet("color: orange;")
        else:
            self.ldm4ts_status_label.setText("LDM4TS: Disabled")
            self.ldm4ts_status_label.setStyleSheet("color: #888; font-style: italic;")

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

            # Update GPU info label
            self._update_gpu_info_label()

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
        logger.warning(f"[GET_MODEL_PATHS] models_edit contains: {paths}")
        logger.warning(f"[GET_MODEL_PATHS] self._model_paths contains: {self._model_paths}")
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
        logger.debug("[GET_SETTINGS] Reading current UI state")
        # Sync self._model_paths with UI content before saving
        ui_paths = self._get_model_paths()
        if ui_paths:
            self._model_paths = ui_paths
            logger.debug(f"Updated model paths from UI: {len(ui_paths)} paths")

        # Read current horizons value from UI
        horizons_value = self.horizons_edit.text()

        settings = {
            # Model paths - use self._model_paths (synced from UI above)
            'model_paths': self._model_paths,

            # Forecast types
            'type_basic': self.type_basic_cb.isChecked(),
            'type_advanced': self.type_advanced_cb.isChecked(),
            'type_rw': self.type_rw_cb.isChecked(),

            # Model combination
            'combine_models': self.combine_models_cb.isChecked(),
            'use_gpu_inference': self.use_gpu_inference_cb.isChecked(),

            # Core settings
            'horizons': horizons_value,
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

            # Parameter override flags (for multi-model metadata usage)
            'override_warmup_bars': self.warmup_override_cb.isChecked(),
            'override_rv_window': self.rv_window_override_cb.isChecked(),
            'override_returns_window': self.returns_window_override_cb.isChecked(),
            'override_min_coverage': self.min_coverage_override_cb.isChecked(),

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
        # IMPORTANT: Check model_paths first (plural), then model_path (singular) for backward compatibility
        if 'model_paths' in settings and settings['model_paths']:
            paths = settings['model_paths']
            if isinstance(paths, list) and len(paths) > 0:
                self.models_edit.setPlainText("\n".join(paths))
                self._model_paths = paths
                logger.debug(f"Loaded {len(paths)} model paths from settings")
            elif isinstance(paths, str):
                self.models_edit.setPlainText(paths)
                self._model_paths = [p.strip() for p in paths.split('\n') if p.strip()]
        elif 'model_path' in settings and settings['model_path']:
            # Legacy single path (only if model_paths not present)
            path = settings['model_path']
            self.models_edit.setPlainText(path)
            self._model_paths = [path]
            logger.debug(f"Loaded 1 legacy model path from settings")

        # Forecast types
        self.type_basic_cb.setChecked(settings.get('type_basic', True))
        self.type_advanced_cb.setChecked(settings.get('type_advanced', False))
        self.type_rw_cb.setChecked(settings.get('type_rw', False))

        # Model combination
        self.combine_models_cb.setChecked(settings.get('combine_models', True))
        self.use_gpu_inference_cb.setChecked(settings.get('use_gpu_inference', False))

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

        # Parameter override flags (default True for backward compatibility)
        self.warmup_override_cb.setChecked(settings.get('override_warmup_bars', True))
        self.rv_window_override_cb.setChecked(settings.get('override_rv_window', True))
        self.returns_window_override_cb.setChecked(settings.get('override_returns_window', True))
        self.min_coverage_override_cb.setChecked(settings.get('override_min_coverage', True))

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

            # Update class variable to persist model paths across instances
            self.__class__._last_model_paths = self._model_paths

            logger.info(f"Settings saved: {len(self._model_paths)} model(s), horizons={settings.get('horizons')}")

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

    def _update_gpu_info_label(self):
        """Update info label based on GPU checkbox state"""
        num_models = len(self._model_paths)
        use_gpu = self.use_gpu_inference_cb.isChecked()

        if num_models == 0:
            self.gpu_info_label.setText("⚠️ Nessun modello selezionato")
            self.gpu_info_label.setStyleSheet("color: #ff6b6b; font-style: italic; padding: 5px;")
        elif use_gpu and num_models > 1:
            self.gpu_info_label.setText(
                f"⚠️ GPU attiva: userà solo il primo modello di {num_models} selezionati. "
                f"Disattiva GPU per usare tutti i modelli in parallelo."
            )
            self.gpu_info_label.setStyleSheet("color: #ffa500; font-style: italic; padding: 5px;")
        elif use_gpu and num_models == 1:
            self.gpu_info_label.setText("✓ GPU attiva: inference veloce con 1 modello")
            self.gpu_info_label.setStyleSheet("color: #51cf66; font-style: italic; padding: 5px;")
        elif not use_gpu and num_models > 1:
            self.gpu_info_label.setText(
                f"✓ CPU parallela: userà tutti i {num_models} modelli in ensemble"
            )
            self.gpu_info_label.setStyleSheet("color: #51cf66; font-style: italic; padding: 5px;")
        elif not use_gpu and num_models == 1:
            self.gpu_info_label.setText("✓ CPU: inference con 1 modello")
            self.gpu_info_label.setStyleSheet("color: #888; font-style: italic; padding: 5px;")

    @staticmethod
    def get_model_paths() -> List[str]:
        """Return last selected model paths (class method for compatibility)"""
        try:
            return list(getattr(UnifiedPredictionSettingsDialog, "_last_model_paths", []) or [])
        except Exception:
            return []

    @staticmethod
    def get_settings_from_file() -> Optional[Dict[str, Any]]:
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
