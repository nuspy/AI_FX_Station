# src/forex_diffusion/ui/training_tab.py
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QLineEdit, QGroupBox, QGridLayout, QMessageBox, QFileDialog, QTextEdit, QProgressBar,
    QCheckBox, QScrollArea
)
from loguru import logger

from ..utils.config import get_config
from ..utils.user_settings import get_setting, set_setting
from ..i18n import tr
from .controllers import TrainingController
from .optimized_params_display_widget import OptimizedParamsDisplayWidget

# Settings file location
TRAINING_SETTINGS_FILE = Path.home() / ".forexgpt" / "training_settings.json"

# All 18 technical indicators + 4 additional features
INDICATORS = [
    "ATR", "RSI", "MACD", "Bollinger", "Stochastic", "CCI", "Williams%R", "ADX",
    "MFI", "OBV", "TRIX", "Ultimate", "Donchian", "Keltner", "EMA", "SMA", "Hurst", "VWAP"
]

ADDITIONAL_FEATURES = [
    "Returns & Volatility",
    "Trading Sessions",
    "Candlestick Patterns",
    "Volume Profile"
]

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Comprehensive tooltips for each indicator
INDICATOR_TOOLTIPS = {
    "ATR": "Average True Range - Misura la volatilità del mercato basata sul range delle candele.\n"
           "Cosa è: indicatore di volatilità che misura l'ampiezza media dei movimenti.\n"
           "Perché è importante: identifica periodi di alta/bassa volatilità per gestire il rischio.\n"
           "Valori bassi: mercato calmo, movimenti piccoli, basso rischio ma poche opportunità.\n"
           "Valori alti: mercato volatile, movimenti ampi, alto rischio ma più opportunità di profitto.",

    "RSI": "Relative Strength Index - Oscillatore di momentum che varia tra 0 e 100.\n"
           "Cosa è: misura la forza relativa dei movimenti rialzisti vs ribassisti.\n"
           "Perché è importante: identifica condizioni di ipercomprato (>70) e ipervenduto (<30).\n"
           "Valori bassi (<30): possibile inversione rialzista, asset ipervenduto.\n"
           "Valori alti (>70): possibile inversione ribassista, asset ipercomprato.",

    "MACD": "Moving Average Convergence Divergence - Indicatore di trend e momentum.\n"
            "Cosa è: differenza tra medie mobili veloci e lente, con linea di segnale.\n"
            "Perché è importante: identifica cambi di trend quando linea MACD incrocia la signal.\n"
            "Valori bassi (negativi): trend ribassista, possibile continuazione o inversione.\n"
            "Valori alti (positivi): trend rialzista, possibile continuazione o inversione.",

    "Bollinger": "Bande di Bollinger - Envelope di volatilità attorno a una media mobile.\n"
                 "Cosa è: banda superiore/inferiore a ±2 deviazioni standard dalla media.\n"
                 "Perché è importante: identifica espansioni/contrazioni di volatilità e livelli estremi.\n"
                 "Bande strette: bassa volatilità, possibile breakout imminente.\n"
                 "Bande larghe: alta volatilità, possibile ritorno verso la media.",

    "Stochastic": "Stochastic Oscillator - Oscillatore di momentum che confronta close con range.\n"
                  "Cosa è: percentuale della posizione del close nel range high-low recente.\n"
                  "Perché è importante: identifica condizioni di ipercomprato/ipervenduto.\n"
                  "Valori bassi (<20): ipervenduto, possibile inversione rialzista.\n"
                  "Valori alti (>80): ipercomprato, possibile inversione ribassista.",

    "CCI": "Commodity Channel Index - Misura la deviazione dal prezzo medio.\n"
           "Cosa è: oscillatore che identifica trend ciclici e condizioni estreme.\n"
           "Perché è importante: identifica quando il prezzo si allontana dalla media.\n"
           "Valori bassi (<-100): ipervenduto, possibile rimbalzo.\n"
           "Valori alti (>+100): ipercomprato, possibile correzione.",

    "Williams%R": "Williams Percent Range - Oscillatore di momentum invertito.\n"
                  "Cosa è: misura la posizione del close rispetto al range high-low.\n"
                  "Perché è importante: identifica ipercomprato/ipervenduto in modo reattivo.\n"
                  "Valori bassi (-100 a -80): ipervenduto, possibile inversione rialzista.\n"
                  "Valori alti (-20 a 0): ipercomprato, possibile inversione ribassista.",

    "ADX": "Average Directional Index - Misura la forza del trend (non la direzione).\n"
           "Cosa è: indicatore che quantifica l'intensità del trend da 0 a 100.\n"
           "Perché è importante: distingue mercati trending da mercati laterali.\n"
           "Valori bassi (<20): mercato laterale, range-bound, evitare strategie trend-following.\n"
           "Valori alti (>40): trend forte, ideale per strategie trend-following.",

    "MFI": "Money Flow Index - RSI ponderato per volume.\n"
           "Cosa è: oscillatore che combina prezzo e volume per misurare pressione acquisto/vendita.\n"
           "Perché è importante: identifica divergenze tra prezzo e volume.\n"
           "Valori bassi (<20): ipervenduto con volumi bassi, possibile rimbalzo.\n"
           "Valori alti (>80): ipercomprato con volumi alti, possibile correzione.",

    "OBV": "On-Balance Volume - Volume cumulativo direzionale.\n"
           "Cosa è: somma cumulativa di volume pesato per direzione del movimento.\n"
           "Perché è importante: conferma trend (divergenze predicono inversioni).\n"
           "Valori crescenti: accumulo, conferma trend rialzista.\n"
           "Valori decrescenti: distribuzione, conferma trend ribassista.",

    "TRIX": "Triple Exponential Average - Rate of change di EMA tripla.\n"
            "Cosa è: indicatore di momentum che filtra movimenti minori.\n"
            "Perché è importante: identifica cambi di trend ignorando rumore.\n"
            "Valori bassi (negativi): momentum ribassista.\n"
            "Valori alti (positivi): momentum rialzista.",

    "Ultimate": "Ultimate Oscillator - Oscillatore multi-timeframe.\n"
                "Cosa è: combina momentum su 3 timeframe diversi (7, 14, 28 periodi).\n"
                "Perché è importante: riduce falsi segnali combinando diversi orizzonti.\n"
                "Valori bassi (<30): ipervenduto, possibile inversione rialzista.\n"
                "Valori alti (>70): ipercomprato, possibile inversione ribassista.",

    "Donchian": "Donchian Channels - Canale basato su high/low di N periodi.\n"
                "Cosa è: banda superiore = max high, banda inferiore = min low.\n"
                "Perché è importante: identifica breakout e supporto/resistenza dinamici.\n"
                "Prezzo vicino banda inferiore: possibile supporto, setup long.\n"
                "Prezzo vicino banda superiore: possibile resistenza, setup short.",

    "Keltner": "Keltner Channels - Canale basato su ATR attorno a EMA.\n"
               "Cosa è: envelope costruito con EMA ± multiplo di ATR.\n"
               "Perché è importante: identifica trend e breakout considerando volatilità.\n"
               "Bande strette: bassa volatilità, possibile breakout.\n"
               "Bande larghe: alta volatilità, possibile ritorno verso media.",

    "EMA": "Exponential Moving Average - Media mobile esponenziale.\n"
           "Cosa è: media mobile che dà più peso ai prezzi recenti.\n"
           "Perché è importante: identifica trend e fornisce supporto/resistenza dinamico.\n"
           "EMA ripida: trend forte, momentum elevato.\n"
           "EMA piatta: mercato laterale, assenza di trend.",

    "SMA": "Simple Moving Average - Media mobile semplice.\n"
           "Cosa è: media aritmetica dei prezzi di N periodi.\n"
           "Perché è importante: livello di supporto/resistenza, smoothing del rumore.\n"
           "Prezzo sopra SMA: trend rialzista, bias long.\n"
           "Prezzo sotto SMA: trend ribassista, bias short.",

    "Hurst": "Hurst Exponent - Misura la persistenza/anti-persistenza delle serie.\n"
             "Cosa è: esponente H che classifica il comportamento della serie (0-1).\n"
             "Perché è importante: identifica se il mercato è trending o mean-reverting.\n"
             "H < 0.5: mean-reverting, inversioni frequenti, strategie contro-trend.\n"
             "H > 0.5: trending/persistente, trend seguono direzione, strategie trend-following.",

    "VWAP": "Volume-Weighted Average Price - Prezzo medio ponderato per volume.\n"
            "Cosa è: media del prezzo pesata per il volume scambiato.\n"
            "Perché è importante: rappresenta il 'fair value' intraday, usato da istituzionali.\n"
            "Prezzo sotto VWAP: possibile sottovalutazione, bias long.\n"
            "Prezzo sopra VWAP: possibile sopravvalutazione, bias short.",
}

# Tooltips for additional features
FEATURE_TOOLTIPS = {
    "Returns & Volatility": "Returns & Volatility - Rendimenti percentuali e volatilità realizzata.\n"
                           "Cosa è: log-returns e rolling standard deviation del prezzo.\n"
                           "Perché è importante: fondamentali per modelli quantitativi, catturano dinamica e rischio.\n"
                           "Volatilità bassa: movimenti prevedibili, range-bound.\n"
                           "Volatilità alta: movimenti imprevedibili, breakout frequenti.",

    "Trading Sessions": "Trading Sessions - Identifica sessione attiva (Tokyo/London/NY).\n"
                       "Cosa è: regime detection basato su orario UTC delle barre.\n"
                       "Perché è importante: volatilità e comportamento variano per sessione.\n"
                       "Tokyo: volatilità bassa, movimenti piccoli.\n"
                       "London/NY overlap: volatilità massima, movimenti ampi.",

    "Candlestick Patterns": "Candlestick Patterns - Pattern candlestick su timeframe superiore.\n"
                           "Cosa è: riconoscimento di pattern (doji, hammer, engulfing, etc.) su TF maggiore.\n"
                           "Perché è importante: identifica setup di inversione/continuazione.\n"
                           "Pattern inversione: possibile cambio di direzione.\n"
                           "Pattern continuazione: conferma del trend in atto.",

    "Volume Profile": "Volume Profile - Distribuzione volume per livello di prezzo.\n"
                     "Cosa è: istogramma del volume scambiato a ciascun livello di prezzo.\n"
                     "Perché è importante: identifica aree di supporto/resistenza basate su activity.\n"
                     "POC (Point of Control): livello con massimo volume, forte supporto/resistenza.\n"
                     "Low volume nodes: zone di transizione rapida, possibili breakout.",
}

# Tooltips for advanced parameters
PARAMETER_TOOLTIPS = {
    "warmup": "Warmup Bars - Barre di warmup per stabilizzare gli indicatori prima del training.\n"
              "Cosa è: numero di barre iniziali scartate per permettere agli indicatori di 'scaldarsi'.\n"
              "Perché è importante: evita valori instabili all'inizio della serie.\n"
              "Valori bassi (0-10): usa quasi tutti i dati, ma primi valori possono essere rumorosi.\n"
              "Valori alti (50+): indicatori più stabili, ma perdi dati di training.",

    "rv_window": "Realized Volatility Window - Finestra per stima di volatilità/standardizzazione.\n"
                 "Cosa è: numero di barre usate per calcolare volatilità realizzata.\n"
                 "Perché è importante: normalizza le feature rispetto alla volatilità recente.\n"
                 "Valori bassi (20-40): reattivo a cambi di volatilità, ma più rumoroso.\n"
                 "Valori alti (100+): stima stabile, ma lenta ad adattarsi.",

    "returns_window": "Returns Window - Finestra per calcolare rendimenti.\n"
                     "Cosa è: numero di barre per calcolare log-returns.\n"
                     "Perché è importante: cattura la dinamica di breve termine.\n"
                     "Valori bassi (1-5): movimenti immediati, alta frequenza.\n"
                     "Valori alti (20+): trend di medio termine, filtra rumore.",

    "min_coverage": "Minimum Feature Coverage - Copertura minima richiesta per includere una feature.\n"
                   "Cosa è: frazione minima di valori non-NaN richiesti (0.0-1.0).\n"
                   "Perché è importante: evita feature con troppi dati mancanti.\n"
                   "Valori bassi (0.1-0.3): include più feature, ma con possibili gap.\n"
                   "Valori alti (0.7-0.9): solo feature complete, ma set più ridotto.",

    "higher_tf": "Higher Timeframe - Timeframe superiore per candlestick patterns.\n"
                "Cosa è: timeframe maggiore del base_tf per pattern recognition.\n"
                "Perché è importante: pattern su TF superiori hanno maggior significato.\n"
                "TF vicini (es. 1m -> 5m): pattern frequenti, più segnali ma meno affidabili.\n"
                "TF distanti (es. 1m -> 1h): pattern rari, meno segnali ma più affidabili.",

    "session_overlap": "Session Overlap Minutes - Minuti di overlap tra sessioni.\n"
                      "Cosa è: finestra di tempo attorno ai cambi di sessione.\n"
                      "Perché è importante: identifica periodi di transizione tra sessioni.\n"
                      "Valori bassi (15-30): transizioni nette, regime change rapidi.\n"
                      "Valori alti (60+): transizioni graduali, considera overlap esteso.",
}

# Default selections for indicators
DEFAULTS = {
    "ATR": ["1m", "5m", "15m", "30m", "1h"],
    "RSI": ["1m", "5m", "15m", "30m", "1h"],
    "Bollinger": ["1m", "5m", "15m", "30m"],
    "MACD": ["5m", "15m", "30m", "1h", "4h", "1d"],
    "Stochastic": ["5m", "15m", "30m", "1h"],
    "CCI": ["15m", "30m", "1h", "4h"],
    "Williams%R": ["5m", "15m", "30m", "1h"],
    "ADX": ["15m", "30m", "1h", "4h"],
    "MFI": ["15m", "30m", "1h"],
    "OBV": ["5m", "15m", "30m", "1h", "4h"],
    "TRIX": ["30m", "1h", "4h"],
    "Ultimate": ["15m", "30m", "1h"],
    "Donchian": ["15m", "30m", "1h", "4h", "1d"],
    "Keltner": ["15m", "30m", "1h", "4h", "1d"],
    "EMA": ["1m", "5m", "15m", "30m", "1h"],
    "SMA": ["5m", "15m", "30m", "1h", "4h"],
    "Hurst": ["30m", "1h", "4h", "1d"],
    "VWAP": ["1m", "5m", "15m"],
}


class TrainingTab(QWidget):
    """
    Training Tab: configure and launch model training.
    - Symbol/timeframe/days/horizon selectors
    - 4-column indicator grid with master checkbox
    - Additional features with enable/parameters
    - Advanced parameters exposed
    - Async training with progress bar and log
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = get_config()

        # Root layout
        self._root = QVBoxLayout(self)

        # Scrollable page
        page = QWidget(self)
        self.layout = QVBoxLayout(page)

        # Controller for async training
        self.controller = TrainingController(self)
        self.controller.signals.log.connect(self._append_log)
        self.controller.signals.progress.connect(self._on_progress)
        self.controller.signals.finished.connect(self._on_finished)

        # Pending metadata for current training run
        self._pending_meta: Optional[Dict] = None
        self._pending_out_dir: Optional[Path] = None

        # Build UI sections
        self._build_top_controls()
        self._build_indicator_grid()
        self._build_additional_features()
        self._build_advanced_params()

        # Install scroll area
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self._root.addWidget(scroll)

        # Output location
        self._build_output_section()

        # Optimized parameters display (FASE 9 - Part 2)
        self._build_optimized_params_section()

        # Log & progress
        self._build_log_section()

        # Actions
        self._build_actions()

        # Load saved settings
        self._load_settings()
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()

    def _save_settings(self):
        """Save all training settings to persistent storage"""
        try:
            settings = {
                # Top controls
                'model_name': self.model_name_edit.text(),
                'symbol': self.symbol_combo.currentText(),
                'timeframe': self.tf_combo.currentText(),
                'days_history': self.days_spin.value(),
                'horizon': self.horizon_spin.text(),
                'model': self.model_combo.currentText(),
                'encoder': self.encoder_combo.currentText(),
                'use_gpu_training': self.use_gpu_training_check.isChecked(),
                'optimization': self.opt_combo.currentText(),
                'gen': self.gen_spin.value(),
                'pop': self.pop_spin.value(),

                # Indicator selections
                'use_indicators': self.use_indicators_check.isChecked(),
                'indicator_tfs': {
                    ind: [tf for tf, cb in self.indicator_checks[ind].items() if cb.isChecked()]
                    for ind in INDICATORS
                },

                # Additional features
                'returns_enabled': self.returns_check.isChecked(),
                'returns_window': self.returns_window.value(),
                'sessions_enabled': self.sessions_check.isChecked(),
                'session_overlap': self.session_overlap.value(),
                'candlestick_enabled': self.candlestick_check.isChecked(),
                'higher_tf': self.higher_tf_combo.currentText(),
                'volume_profile_enabled': self.volume_profile_check.isChecked(),
                'vp_bins': self.vp_bins.value(),
                'vp_window': self.vp_window.value(),
                'use_vsa': self.vsa_check.isChecked(),
                'vsa_volume_ma': self.vsa_volume_ma.value(),
                'vsa_spread_ma': self.vsa_spread_ma.value(),

                # Advanced parameters
                'warmup_bars': self.warmup.value(),
                'rv_window': self.rv_w.value(),
                'min_coverage': self.min_coverage.value(),
                'atr_n': self.atr_n.value(),
                'rsi_n': self.rsi_n.value(),
                'bb_n': self.bb_n.value(),
                'hurst_window': self.hurst_w.value(),
                'light_epochs': self.light_epochs.value(),
                'light_batch': self.light_batch.value(),
                'light_val_frac': self.light_val_frac.value(),
                'patch_len': self.patch_len.value(),
                'latent_dim': self.latent_dim.value(),
                'encoder_epochs': self.encoder_epochs.value(),

                # Diffusion parameters
                'diffusion_timesteps': self.diffusion_timesteps.value(),
                'learning_rate': self.learning_rate.value(),
                'batch_size_dl': self.batch_size_dl.value(),
                'model_channels': self.model_channels.value(),
                'dropout': self.dropout.value(),
                'num_heads': self.num_heads.value(),

                # NVIDIA Optimization Stack
                'nvidia_enable': self.nvidia_enable.isChecked(),
                'use_amp': self.use_amp.isChecked(),
                'precision': self.precision_combo.currentText(),
                'compile_model': self.compile_model.isChecked(),
                'use_fused_optimizer': self.use_fused_optimizer.isChecked(),
                'use_flash_attention': self.use_flash_attention.isChecked(),
                'grad_accumulation_steps': self.grad_accumulation_steps.value(),

                # Output directory
                'output_dir': self.out_dir.text(),
            }

            # Save to file
            TRAINING_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TRAINING_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)

            logger.debug(f"Training settings saved to {TRAINING_SETTINGS_FILE}")

        except Exception as e:
            logger.exception(f"Failed to save training settings: {e}")

    def _load_settings(self):
        """Load training settings from persistent storage"""
        try:
            if not TRAINING_SETTINGS_FILE.exists():
                logger.debug("No saved training settings found")
                return

            with open(TRAINING_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            # Top controls
            if 'model_name' in settings:
                self.model_name_edit.setText(settings['model_name'])
            if 'symbol' in settings:
                self.symbol_combo.setCurrentText(settings['symbol'])
            if 'timeframe' in settings:
                self.tf_combo.setCurrentText(settings['timeframe'])
            if 'days_history' in settings:
                self.days_spin.setValue(settings['days_history'])
            if 'horizon' in settings:
                self.horizon_spin.setText(str(settings['horizon']))
            if 'model' in settings:
                self.model_combo.setCurrentText(settings['model'])
            if 'encoder' in settings:
                self.encoder_combo.setCurrentText(settings['encoder'])
            if 'use_gpu_training' in settings:
                self.use_gpu_training_check.setChecked(settings['use_gpu_training'])
            if 'optimization' in settings:
                self.opt_combo.setCurrentText(settings['optimization'])
            if 'gen' in settings:
                self.gen_spin.setValue(settings['gen'])
            if 'pop' in settings:
                self.pop_spin.setValue(settings['pop'])

            # Indicator selections
            if 'use_indicators' in settings:
                self.use_indicators_check.setChecked(settings['use_indicators'])

            if 'indicator_tfs' in settings:
                for ind, tfs in settings['indicator_tfs'].items():
                    if ind in self.indicator_checks:
                        for tf, cb in self.indicator_checks[ind].items():
                            cb.setChecked(tf in tfs)

            # Additional features
            if 'returns_enabled' in settings:
                self.returns_check.setChecked(settings['returns_enabled'])
            if 'returns_window' in settings:
                self.returns_window.setValue(settings['returns_window'])
            if 'sessions_enabled' in settings:
                self.sessions_check.setChecked(settings['sessions_enabled'])
            if 'session_overlap' in settings:
                self.session_overlap.setValue(settings['session_overlap'])
            if 'candlestick_enabled' in settings:
                self.candlestick_check.setChecked(settings['candlestick_enabled'])
            if 'higher_tf' in settings:
                self.higher_tf_combo.setCurrentText(settings['higher_tf'])
            if 'volume_profile_enabled' in settings:
                self.volume_profile_check.setChecked(settings['volume_profile_enabled'])
            if 'vp_bins' in settings:
                self.vp_bins.setValue(settings['vp_bins'])
            if 'vp_window' in settings:
                self.vp_window.setValue(settings['vp_window'])
            if 'use_vsa' in settings:
                self.vsa_check.setChecked(settings['use_vsa'])
            if 'vsa_volume_ma' in settings:
                self.vsa_volume_ma.setValue(settings['vsa_volume_ma'])
            if 'vsa_spread_ma' in settings:
                self.vsa_spread_ma.setValue(settings['vsa_spread_ma'])

            # Advanced parameters
            if 'warmup_bars' in settings:
                self.warmup.setValue(settings['warmup_bars'])
            if 'rv_window' in settings:
                self.rv_w.setValue(settings['rv_window'])
            if 'min_coverage' in settings:
                self.min_coverage.setValue(settings['min_coverage'])
            if 'atr_n' in settings:
                self.atr_n.setValue(settings['atr_n'])
            if 'rsi_n' in settings:
                self.rsi_n.setValue(settings['rsi_n'])
            if 'bb_n' in settings:
                self.bb_n.setValue(settings['bb_n'])
            if 'hurst_window' in settings:
                self.hurst_w.setValue(settings['hurst_window'])
            if 'light_epochs' in settings:
                self.light_epochs.setValue(settings['light_epochs'])
            if 'light_batch' in settings:
                self.light_batch.setValue(settings['light_batch'])
            if 'light_val_frac' in settings:
                self.light_val_frac.setValue(settings['light_val_frac'])
            if 'patch_len' in settings:
                self.patch_len.setValue(settings['patch_len'])
            if 'latent_dim' in settings:
                self.latent_dim.setValue(settings['latent_dim'])
            if 'encoder_epochs' in settings:
                self.encoder_epochs.setValue(settings['encoder_epochs'])

            # Diffusion parameters
            if 'diffusion_timesteps' in settings:
                self.diffusion_timesteps.setValue(settings['diffusion_timesteps'])
            if 'learning_rate' in settings:
                self.learning_rate.setValue(settings['learning_rate'])
            if 'batch_size_dl' in settings:
                self.batch_size_dl.setValue(settings['batch_size_dl'])
            if 'model_channels' in settings:
                self.model_channels.setValue(settings['model_channels'])
            if 'dropout' in settings:
                self.dropout.setValue(settings['dropout'])
            if 'num_heads' in settings:
                self.num_heads.setValue(settings['num_heads'])

            # NVIDIA Optimization Stack
            if 'nvidia_enable' in settings:
                self.nvidia_enable.setChecked(settings['nvidia_enable'])
            if 'use_amp' in settings:
                self.use_amp.setChecked(settings['use_amp'])
            if 'precision' in settings:
                self.precision_combo.setCurrentText(settings['precision'])
            if 'compile_model' in settings:
                self.compile_model.setChecked(settings['compile_model'])
            if 'use_fused_optimizer' in settings:
                self.use_fused_optimizer.setChecked(settings['use_fused_optimizer'])
            if 'use_flash_attention' in settings:
                self.use_flash_attention.setChecked(settings['use_flash_attention'])
            if 'grad_accumulation_steps' in settings:
                self.grad_accumulation_steps.setValue(settings['grad_accumulation_steps'])

            # Output directory
            if 'output_dir' in settings:
                self.out_dir.setText(settings['output_dir'])

            logger.info(f"Training settings loaded from {TRAINING_SETTINGS_FILE}")

        except Exception as e:
            logger.exception(f"Failed to load training settings: {e}")

    def closeEvent(self, event):
        """Save settings when tab/window is closed"""
        self._save_settings()
        super().closeEvent(event)

    def hideEvent(self, event):
        """Save settings when tab is hidden (user switches tabs)"""
        self._save_settings()
        super().hideEvent(event)
    
    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets"""
        from ..i18n.widget_helper import apply_tooltip
        from ..i18n import tr
        
        logger.info("Applying i18n tooltips to Training Tab widgets...")
        
        # Top controls
        apply_tooltip(self.model_name_edit, "model_name", "training")
        logger.debug(f"Applied tooltip to model_name_edit: {self.model_name_edit.toolTip()[:50]}...")
        apply_tooltip(self.symbol_combo, "symbol", "training")
        apply_tooltip(self.tf_combo, "timeframe", "training")
        apply_tooltip(self.days_spin, "days", "training")
        apply_tooltip(self.horizon_spin, "horizon", "training")
        apply_tooltip(self.model_combo, "model", "training")
        apply_tooltip(self.encoder_combo, "encoder", "training")
        apply_tooltip(self.use_gpu_training_check, "use_gpu_training", "training")
        apply_tooltip(self.opt_combo, "optimization", "training")
        apply_tooltip(self.gen_spin, "gen", "training")
        apply_tooltip(self.pop_spin, "pop", "training")
        
        # Indicator master toggle
        apply_tooltip(self.use_indicators_check, "use_indicators", "training.indicators")
        
        # Feature Engineering
        apply_tooltip(self.returns_check, "returns_volatility", "training.advanced.features")
        apply_tooltip(self.returns_window, "returns_window", "training.advanced.feature_engineering")
        apply_tooltip(self.sessions_check, "trading_sessions", "training.advanced.features")
        apply_tooltip(self.session_overlap, "session_overlap", "training.advanced.feature_engineering")
        apply_tooltip(self.candlestick_check, "candlestick_patterns", "training.advanced.features")
        apply_tooltip(self.higher_tf_combo, "higher_tf", "training.advanced.feature_engineering")
        apply_tooltip(self.volume_profile_check, "volume_profile", "training.advanced.features")
        apply_tooltip(self.vp_bins, "vp_bins", "training.advanced.feature_engineering")
        apply_tooltip(self.vp_window, "vp_window", "training.advanced.feature_engineering")
        apply_tooltip(self.vsa_check, "vsa", "training.advanced.features")
        apply_tooltip(self.vsa_volume_ma, "vsa_volume_ma", "training.advanced.feature_engineering")
        apply_tooltip(self.vsa_spread_ma, "vsa_spread_ma", "training.advanced.feature_engineering")
        
        # Advanced Parameters
        apply_tooltip(self.warmup, "warmup_bars", "training.advanced")
        apply_tooltip(self.rv_w, "rv_window", "training.advanced")
        apply_tooltip(self.min_coverage, "min_coverage", "training.advanced")
        apply_tooltip(self.atr_n, "atr_n", "training.advanced.indicator_periods")
        apply_tooltip(self.rsi_n, "rsi_n", "training.advanced.indicator_periods")
        apply_tooltip(self.bb_n, "bb_n", "training.advanced.indicator_periods")
        apply_tooltip(self.hurst_w, "hurst_window", "training.advanced.indicator_periods")
        
        # LightGBM
        apply_tooltip(self.light_epochs, "epochs", "training.advanced.lightgbm")
        apply_tooltip(self.light_batch, "batch", "training.advanced.lightgbm")
        apply_tooltip(self.light_val_frac, "validation_fraction", "training.advanced.lightgbm")
        
        # Encoder
        apply_tooltip(self.patch_len, "patch_len", "training.advanced.encoder")
        apply_tooltip(self.latent_dim, "latent_dim", "training.advanced.encoder")
        apply_tooltip(self.encoder_epochs, "epochs", "training.advanced.encoder")
        
        # Diffusion (if widgets exist)
        if hasattr(self, 'diffusion_timesteps'):
            apply_tooltip(self.diffusion_timesteps, "timesteps", "training.advanced.diffusion")
        if hasattr(self, 'learning_rate'):
            apply_tooltip(self.learning_rate, "learning_rate", "training.advanced.diffusion")
        if hasattr(self, 'batch_size_dl'):
            apply_tooltip(self.batch_size_dl, "batch_size", "training.advanced.diffusion")
        if hasattr(self, 'model_channels'):
            apply_tooltip(self.model_channels, "model_channels", "training.advanced.diffusion")
        if hasattr(self, 'dropout'):
            apply_tooltip(self.dropout, "dropout", "training.advanced.diffusion")
        if hasattr(self, 'num_heads'):
            apply_tooltip(self.num_heads, "num_heads", "training.advanced.diffusion")
        
        # NVIDIA GPU Optimization
        if hasattr(self, 'nvidia_enable'):
            apply_tooltip(self.nvidia_enable, "enable", "training.advanced.nvidia")
        if hasattr(self, 'use_amp'):
            apply_tooltip(self.use_amp, "use_amp", "training.advanced.nvidia")
        if hasattr(self, 'precision_combo'):
            apply_tooltip(self.precision_combo, "precision", "training.advanced.nvidia")
        if hasattr(self, 'compile_model'):
            apply_tooltip(self.compile_model, "compile_model", "training.advanced.nvidia")
        if hasattr(self, 'use_fused_optimizer'):
            apply_tooltip(self.use_fused_optimizer, "fused_optimizer", "training.advanced.nvidia")
        if hasattr(self, 'use_flash_attention'):
            apply_tooltip(self.use_flash_attention, "flash_attention", "training.advanced.nvidia")
        if hasattr(self, 'grad_accumulation_steps'):
            apply_tooltip(self.grad_accumulation_steps, "grad_accumulation_steps", "training.advanced.nvidia")
        
        # Apply indicator tooltips
        for indicator in INDICATORS:
            ind_key = indicator.lower().replace("%", "").replace("/", "_")
            tooltip_key = f"training.indicators.{ind_key}"
            tooltip_text = tr(f"{tooltip_key}.tooltip", default=None)
            if tooltip_text:
                for tf, cb in self.indicator_checks.get(indicator, {}).items():
                    cb.setToolTip(tooltip_text)

    def _build_top_controls(self):
        """Build top control row: model name, load/save config buttons, symbol, timeframe, days, horizon, model, encoder, optimization"""

        # Row 0: Model Name + Load/Save Config
        row0 = QHBoxLayout()

        lbl_name = QLabel("Model Name:")
        lbl_name.setToolTip(
            "Nome del modello salvato.\n"
            "Se impostato: il modello sarà salvato con questo nome.\n"
            "Se vuoto: il nome sarà generato automaticamente dall'elenco delle features.\n"
            "Esempio: 'EUR_USD_1h_ridge_multiTF' o 'my_custom_model_v2'"
        )
        row0.addWidget(lbl_name)

        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Auto-generate from features")
        self.model_name_edit.setToolTip(
            "Nome personalizzato per il modello.\n"
            "Lascia vuoto per auto-generazione basata su: symbol_timeframe_model_features.\n"
            "Caratteri permessi: lettere, numeri, underscore, trattino."
        )
        row0.addWidget(self.model_name_edit)

        row0.addStretch()

        self.load_config_btn = QPushButton("📂 Load Config")
        self.load_config_btn.setToolTip(
            "Carica una configurazione di training salvata da file JSON.\n"
            "Include tutti i parametri: features, indicatori, hyperparameters.\n"
            "Utile per: riproducibilità, condivisione config, A/B testing."
        )
        self.load_config_btn.clicked.connect(self._on_load_config)
        row0.addWidget(self.load_config_btn)

        self.save_config_btn = QPushButton("💾 Save Config")
        self.save_config_btn.setToolTip(
            "Salva la configurazione attuale in un file JSON.\n"
            "Include: tutte le feature selezionate, parametri, indicatori.\n"
            "Non include: dati o modello addestrato (solo configurazione)."
        )
        self.save_config_btn.clicked.connect(self._on_save_config)
        row0.addWidget(self.save_config_btn)

        self.layout.addLayout(row0)

        # Row 1: Symbol, Timeframe, Days, Horizon, Model, Encoder, Opt, Gen, Pop
        top = QHBoxLayout()

        # Symbol
        lbl_sym = QLabel("Symbol:")
        lbl_sym.setToolTip(
            "Coppia valutaria da usare per il training.\n"
            "Cosa è: asset finanziario su cui addestrare il modello predittivo.\n"
            "Perché è importante: ogni coppia ha caratteristiche uniche (volatilità, correlazioni).\n"
            "Best practice: addestra modelli separati per ciascuna coppia (no mixing)."
        )
        top.addWidget(lbl_sym)
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["EUR/USD", "GBP/USD", "AUX/USD", "GBP/NZD", "AUD/JPY", "GBP/EUR", "GBP/AUD"])
        self.symbol_combo.setToolTip(
            "Seleziona il simbolo per cui addestrare il modello.\n"
            "EUR/USD: coppia più liquida, spread bassi, volatilità media.\n"
            "GBP/USD: alta volatilità, buona per swing trading.\n"
            "Exotic pairs (AUD/JPY, etc.): spread alti, pattern diversi."
        )
        top.addWidget(self.symbol_combo)

        # Base timeframe
        lbl_tf = QLabel("Base TF:")
        lbl_tf.setToolTip(
            "Timeframe base della serie su cui costruire le feature.\n"
            "Cosa è: risoluzione temporale delle candele usate come input.\n"
            "Perché è importante: determina il tipo di trading (scalping vs swing).\n"
            "Valori bassi (1m, 5m): day trading, scalping, necessita molti dati (anni).\n"
            "Valori alti (1h, 4h, 1d): swing/position trading, servono meno dati (mesi).\n"
            "Best practice: usa multi-timeframe analysis (1m base + 5m/15m/1h indicatori)."
        )
        top.addWidget(lbl_tf)
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        self.tf_combo.setCurrentText("1m")
        self.tf_combo.setToolTip(
            "Timeframe delle candele base.\n"
            "1m: ~1400 candele/giorno, ottimo per scalping, richiede 30+ giorni dati.\n"
            "5m: ~280 candele/giorno, day trading, richiede 15+ giorni dati.\n"
            "1h: ~24 candele/giorno, swing trading, richiede 60+ giorni dati.\n"
            "1d: 1 candela/giorno, position trading, richiede 1+ anno dati."
        )
        top.addWidget(self.tf_combo)

        # Days history
        lbl_days = QLabel("Days:")
        lbl_days.setToolTip(
            "Numero di giorni storici da usare per l'addestramento.\n"
            "Cosa è: quanti giorni passati includere nel dataset di training.\n"
            "Perché è importante: più dati = migliore generalizzazione, ma training più lento.\n"
            "Valori bassi (1-7): training veloce, rischio overfitting, buono per test rapidi.\n"
            "Valori medi (30-90): bilanciamento speed/quality, uso standard.\n"
            "Valori alti (365-1825): migliore generalizzazione, cattura cicli stagionali, lento.\n"
            "Best practice: almeno 1000 samples per feature (es. 100 features → 7 giorni su 1m)."
        )
        top.addWidget(lbl_days)
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 3650)
        self.days_spin.setValue(7)
        self.days_spin.setToolTip(
            "Giorni di dati storici per training.\n"
            "1-7: test rapido, pochi dati, rischio overfitting.\n"
            "30-90: standard, bilanciamento qualità/velocità.\n"
            "365+: massima qualità, cattura stagionalità, training lungo (ore).\n"
            "Nota: con TF=1m, 7 giorni ≈ 10K samples. Con TF=1h, 7 giorni ≈ 168 samples."
        )
        top.addWidget(self.days_spin)

        # Horizon
        lbl_h = QLabel("Horizon:")
        lbl_h.setToolTip(
            "Orizzonte di predizione: quante candele future prevedere.\n"
            "Cosa è: numero di step temporali futuri da predire (target).\n"
            "Perché è importante: determina il range temporale della strategia trading.\n"
            "Valori bassi (1-5): predizioni a breve termine, più accurate, scalping.\n"
            "Valori medi (10-50): medio termine, swing trading intraday.\n"
            "Valori alti (100-500): lungo termine, meno accurate, position trading.\n"
            "Best practice: horizon ≤ 20 per supervised models, 50-500 per diffusion models."
        )
        top.addWidget(lbl_h)
        
        # Horizon input with suggestions button
        horizon_row = QHBoxLayout()
        self.horizon_spin = QLineEdit()
        self.horizon_spin.setText("5")
        self.horizon_spin.setPlaceholderText("5 | 1-7/2 | 15,60,240")
        self.horizon_spin.setToolTip(
            "Numero di candele future da prevedere.\n\n"
            "FORMATI SUPPORTATI:\n"
            "  • Singolo: '5'\n"
            "  • Lista: '15,60,240'\n"
            "  • Range: '5-10' → [5,6,7,8,9,10]\n"
            "  • Range con step: '1-7/2' → [1,3,5,7]\n"
            "  • Misto: '1-7/2,60,100-200/50'\n\n"
            "ESEMPI:\n"
            "  • '5' → predici a 5 bars\n"
            "  • '1-10' → predici da 1 a 10 bars (10 orizzonti)\n"
            "  • '1-7/2' → predici a 1,3,5,7 bars\n"
            "  • '15,60,240' → predici a 15min, 1h, 4h\n"
            "  • '10-20/5,50,100-200/25' → [10,15,20,50,100,125,150,175,200]\n\n"
            "RACCOMANDAZIONI:\n"
            "  • 1-5: scalping/day trading\n"
            "  • 10-20: swing intraday\n"
            "  • 50-100: swing multi-day\n"
            "  • 200-500: lungo termine (solo diffusion)\n\n"
            "MULTI-HORIZON:\n"
            "  • UN SOLO modello per tutti gli orizzonti\n"
            "  • Sklearn: Ridge/Lasso/ElasticNet/RF multi-output\n"
            "  • Lightning: Transformer joint distribution\n\n"
            "TF=1m, horizon='5-10' → predici da 5 a 10 minuti\n"
            "TF=1h, horizon='1-24/6,48,72' → predici 1h,7h,13h,19h,48h,72h"
        )
        
        # Add suggest button
        horizon_suggest_btn = QPushButton("💡 Suggest")
        horizon_suggest_btn.setMaximumWidth(80)
        horizon_suggest_btn.setToolTip("Get smart horizon suggestions based on timeframe and style")
        horizon_suggest_btn.clicked.connect(self._suggest_horizons)
        
        horizon_row.addWidget(self.horizon_spin)
        horizon_row.addWidget(horizon_suggest_btn)
        top.addLayout(horizon_row)

        # Model
        lbl_m = QLabel("Model:")
        lbl_m.setToolTip(
            "Tipo di modello supervisionato da addestrare.\n"
            "Cosa è: algoritmo di machine learning per apprendere pattern dai dati.\n"
            "Perché è importante: determina capacità di catturare relazioni complesse.\n"
            "Vedi docs/MODELS_COMPARISON.md per confronto dettagliato supervised vs diffusion."
        )
        top.addWidget(lbl_m)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["ridge", "lasso", "elasticnet", "rf", "lightning", "diffusion-ddpm", "diffusion-ddim", "sssd"])
        self.model_combo.setToolTip(
            "Algoritmi disponibili:\n\n"
            "SUPERVISED (veloce, interpretabile, short-term):\n"
            "• ridge: regressione lineare L2, velocissimo, baseline ottimo.\n"
            "• lasso: regressione lineare L1, feature selection automatica.\n"
            "• elasticnet: combina ridge+lasso, bilanciamento L1/L2.\n"
            "• rf: Random Forest, cattura non-linearità, robusto, lento.\n"
            "• lightning: neural network (MLP/LSTM), molto flessibile, richiede GPU.\n\n"
            "DIFFUSION (lento, generativo, long-term, incertezza):\n"
            "• sssd: Structured State Space Diffusion, multi-timeframe S4+diffusion, richiede GPU.\n"
            "  Previsioni multi-orizzonte [5,15,60,240]min con incertezza quantificata.\n\n"
            "• diffusion-ddpm: Denoising Diffusion Probabilistic Model, alta qualità.\n"
            "• diffusion-ddim: DDIM (deterministic), 10x più veloce di DDPM.\n\n"
            "Raccomandazioni:\n"
            "- Test rapido: ridge (secondi)\n"
            "- Production: ridge/rf (minuti, interpretabile)\n"
            "- Ricerca: lightning/diffusion (ore, GPU consigliata)\n"
            "- Long-term forecast: diffusion-ddim (genera scenari multipli)"
        )
        top.addWidget(self.model_combo)

        # Encoder
        lbl_e = QLabel("Encoder:")
        lbl_e.setToolTip(
            "Preprocessore per ridurre dimensionalità features (opzionale).\n"
            "Cosa è: trasformazione che comprime features mantenendo info utile.\n"
            "Perché è importante: riduce overfitting, accelera training, denoise data.\n"
            "Quando usarlo: se hai >100 features, prova pca/autoencoder.\n"
            "Quando NON usarlo: se hai <50 features, encoder aggiunge complessità inutile."
        )
        top.addWidget(lbl_e)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(["none", "pca", "autoencoder", "vae", "latents"])
        self.encoder_combo.setToolTip(
            "Encoder disponibili:\n\n"
            "• none: nessuna trasformazione, usa features raw.\n"
            "  Pro: semplicità, interpretabilità.\n"
            "  Contro: curse of dimensionality con molte features.\n\n"
            "• pca: Principal Component Analysis (lineare, veloce).\n"
            "  Pro: velocissimo, deterministico, mantiene varianza principale.\n"
            "  Contro: assume linearità, perde info non-lineare.\n"
            "  Quando: 50-200 features, vuoi velocità.\n\n"
            "• autoencoder: Neural autoencoder (non-lineare, lento).\n"
            "  Pro: cattura relazioni non-lineari complesse.\n"
            "  Contro: training lungo, richiede PyTorch, può overfittare.\n"
            "  Quando: >200 features, hai GPU, dati massivi.\n\n"
            "• vae: Variational Autoencoder (probabilistico, regolarizzato).\n"
            "  Pro: versione robusta di autoencoder, meno overfitting.\n"
            "  Contro: training molto lento, complesso.\n"
            "  Quando: >500 features, serve robustezza, hai GPU.\n\n"
            "• latents: usa encoder pre-addestrato salvato.\n"
            "  Pro: skip training encoder, veloce.\n"
            "  Quando: hai già addestrato encoder in run precedente."
        )
        top.addWidget(self.encoder_combo)

        # GPU Training Checkbox
        self.use_gpu_training_check = QCheckBox("Usa GPU")
        self.use_gpu_training_check.setChecked(False)

        # Check if CUDA is available
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            self.use_gpu_training_check.setEnabled(cuda_available)

            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                self.use_gpu_training_check.setToolTip(
                    f"Usa GPU per training encoder (Autoencoder/VAE).\n\n"
                    f"GPU rilevata: {gpu_name}\n\n"
                    f"Speedup atteso:\n"
                    f"• Autoencoder: 10-15x più veloce\n"
                    f"• VAE: 10-15x più veloce\n\n"
                    f"IMPORTANTE: solo encoder usano GPU.\n"
                    f"Ridge/Lasso/ElasticNet/RF rimangono su CPU."
                )
            else:
                self.use_gpu_training_check.setToolTip(
                    "GPU non disponibile.\n\n"
                    "Per usare GPU:\n"
                    "1. Installa CUDA-enabled PyTorch\n"
                    "2. Riavvia l'applicazione"
                )
        except Exception:
            self.use_gpu_training_check.setEnabled(False)
            self.use_gpu_training_check.setToolTip("PyTorch non disponibile")

        top.addWidget(self.use_gpu_training_check)

        # Optimization
        lbl_opt = QLabel("Opt:")
        lbl_opt.setToolTip(
            "Ricerca automatica iperparametri (AutoML).\n"
            "Cosa è: algoritmo evolutivo che trova best hyperparameters.\n"
            "Perché è importante: ottimizza performance senza tuning manuale.\n"
            "Quando usarlo: quando non sai quali parametri scegliere.\n"
            "Quando NON usarlo: per test rapidi (aggiunge ore di training)."
        )
        top.addWidget(lbl_opt)
        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["none", "genetic-basic", "nsga2"])
        self.opt_combo.setToolTip(
            "Metodi di ottimizzazione automatica:\n\n"
            "• none: usa parametri di default, training veloce (1x).\n"
            "  Quando: test rapido, parametri già noti.\n\n"
            "• genetic-basic: algoritmo genetico single-objective.\n"
            "  Cosa ottimizza: solo MAE (errore).\n"
            "  Pro: semplice, funziona bene.\n"
            "  Contro: ignora trade-off (es. accuracy vs complexity).\n"
            "  Tempo: ~5-20x più lento di 'none' (dipende da gen×pop).\n"
            "  Quando: vuoi best accuracy, hai tempo.\n\n"
            "• nsga2: NSGA-II multi-objective optimization.\n"
            "  Cosa ottimizza: MAE + complessità + robustezza.\n"
            "  Pro: trova Pareto front, bilanciamento obiettivi.\n"
            "  Contro: molto lento, complesso interpretare.\n"
            "  Tempo: ~10-50x più lento di 'none'.\n"
            "  Quando: ricerca avanzata, vuoi trade-off espliciti.\n\n"
            "Nota: gen=10, pop=20 → 200 training runs → 200x tempo!"
        )
        top.addWidget(self.opt_combo)

        # Gen
        lbl_gen = QLabel("Gen:")
        lbl_gen.setToolTip(
            "Numero di generazioni dell'algoritmo genetico.\n"
            "Cosa è: iterazioni evolutive per convergere a soluzione ottima.\n"
            "Perché è importante: più generazioni = migliore ottimizzazione, ma più lento.\n"
            "Valori bassi (1-5): esplorazione limitata, veloce, può non convergere.\n"
            "Valori medi (10-20): bilanciamento, raccomandato.\n"
            "Valori alti (30-50): convergenza garantita, molto lento.\n"
            "Best practice: gen × pop ≤ 200 per training ragionevole (<1 giorno)."
        )
        top.addWidget(lbl_gen)
        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(1, 50)
        self.gen_spin.setValue(5)
        self.gen_spin.setToolTip(
            "Iterazioni evolutive.\n"
            "1-5: test rapido, convergenza parziale.\n"
            "10-20: standard, buon bilanciamento.\n"
            "30-50: massima qualità, molto lento (giorni).\n\n"
            "Tempo stimato (gen×pop training runs):\n"
            "- gen=5, pop=8 → 40 runs → ~40min (ridge), ~4h (rf)\n"
            "- gen=20, pop=20 → 400 runs → ~6h (ridge), ~2 giorni (rf)"
        )
        top.addWidget(self.gen_spin)

        # Pop
        lbl_pop = QLabel("Pop:")
        lbl_pop.setToolTip(
            "Dimensione popolazione algoritmo genetico.\n"
            "Cosa è: numero di candidati valutati per ogni generazione.\n"
            "Perché è importante: più popolazione = esplorazione spazio più ampia.\n"
            "Valori bassi (2-8): poca diversità, convergenza prematura, veloce.\n"
            "Valori medi (16-32): bilanciamento diversità/velocità.\n"
            "Valori alti (48-64): massima esplorazione, molto lento.\n"
            "Best practice: pop ≥ 2 × numero hyperparameters da ottimizzare."
        )
        top.addWidget(lbl_pop)
        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(2, 64)
        self.pop_spin.setValue(8)
        self.pop_spin.setToolTip(
            "Candidati per generazione.\n"
            "2-8: veloce, poca esplorazione.\n"
            "16-32: raccomandato, buona diversità.\n"
            "48-64: massima esplorazione, lentissimo.\n\n"
            "Trade-off:\n"
            "- pop bassa + gen alta: convergenza locale (rischio).\n"
            "- pop alta + gen bassa: esplorazione ampia ma non raffina.\n"
            "- bilanciato: pop=20, gen=15 → 300 runs → ~5h (ridge)"
        )
        top.addWidget(self.pop_spin)

        self.layout.addLayout(top)

    def _build_indicator_grid(self):
        """Build 4-column indicator grid with master checkbox"""
        grid_box = QGroupBox("Indicatori Tecnici")
        grid_box.setToolTip("Seleziona gli indicatori da calcolare per ciascun timeframe durante il training.")
        grid_layout = QVBoxLayout(grid_box)

        # Master checkbox at top
        self.use_indicators_check = QCheckBox("Usa indicatori selezionati nel training")
        self.use_indicators_check.setChecked(True)
        self.use_indicators_check.setToolTip("Se disabilitato, il training userà solo feature di base (OHLCV).")
        grid_layout.addWidget(self.use_indicators_check)

        # 4-column grid for indicators
        grid = QGridLayout()

        # Header row: Indicator | TF checkboxes (7 timeframes)
        grid.addWidget(QLabel("Indicator"), 0, 0)
        for j, tf in enumerate(TIMEFRAMES, start=1):
            lbl = QLabel(tf)
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, 0, j)

        # Load saved selections or defaults
        saved = get_setting("training_indicator_tfs", {})
        self.indicator_checks: Dict[str, Dict[str, QCheckBox]] = {}

        # Place indicators in 4 columns
        num_cols = 4
        rows_per_col = (len(INDICATORS) + num_cols - 1) // num_cols  # Ceiling division

        for idx, ind in enumerate(INDICATORS):
            col_group = idx // rows_per_col
            row_in_group = idx % rows_per_col

            # Calculate grid position (each column group takes 9 columns: 1 label + 7 TF + 1 spacing)
            col_offset = col_group * 9
            row_offset = row_in_group + 1  # +1 for header row

            # Indicator label with tooltip
            lbl = QLabel(ind)
            lbl.setToolTip(INDICATOR_TOOLTIPS.get(ind, ""))
            grid.addWidget(lbl, row_offset, col_offset)

            # Timeframe checkboxes
            self.indicator_checks[ind] = {}
            selected = saved.get(ind, DEFAULTS.get(ind, []))
            for j, tf in enumerate(TIMEFRAMES, start=1):
                cb = QCheckBox()
                cb.setChecked(tf in selected)
                cb.setToolTip(f"{ind} on {tf}")
                self.indicator_checks[ind][tf] = cb
                grid.addWidget(cb, row_offset, col_offset + j)

        grid_layout.addLayout(grid)
        self.layout.addWidget(grid_box)

    def _build_additional_features(self):
        """Build additional features section"""
        feat_box = QGroupBox("Feature Aggiuntive")
        feat_box.setToolTip("Feature avanzate oltre agli indicatori tecnici.")
        feat_layout = QGridLayout(feat_box)

        # Returns & Volatility
        self.returns_check = QCheckBox("Returns & Volatility")
        self.returns_check.setChecked(True)
        self.returns_check.setToolTip(FEATURE_TOOLTIPS["Returns & Volatility"])
        feat_layout.addWidget(self.returns_check, 0, 0)

        lbl_ret_win = QLabel("Window:")
        lbl_ret_win.setToolTip(PARAMETER_TOOLTIPS["returns_window"])
        feat_layout.addWidget(lbl_ret_win, 0, 1)
        self.returns_window = QSpinBox()
        self.returns_window.setRange(1, 100)
        self.returns_window.setValue(5)
        self.returns_window.setToolTip(PARAMETER_TOOLTIPS["returns_window"])
        feat_layout.addWidget(self.returns_window, 0, 2)

        # Trading Sessions
        self.sessions_check = QCheckBox("Trading Sessions")
        self.sessions_check.setChecked(True)
        self.sessions_check.setToolTip(FEATURE_TOOLTIPS["Trading Sessions"])
        feat_layout.addWidget(self.sessions_check, 1, 0)

        lbl_sess_overlap = QLabel("Overlap (min):")
        lbl_sess_overlap.setToolTip(PARAMETER_TOOLTIPS["session_overlap"])
        feat_layout.addWidget(lbl_sess_overlap, 1, 1)
        self.session_overlap = QSpinBox()
        self.session_overlap.setRange(0, 120)
        self.session_overlap.setValue(30)
        self.session_overlap.setToolTip(PARAMETER_TOOLTIPS["session_overlap"])
        feat_layout.addWidget(self.session_overlap, 1, 2)

        # Candlestick Patterns
        self.candlestick_check = QCheckBox("Candlestick Patterns")
        self.candlestick_check.setChecked(False)
        self.candlestick_check.setToolTip(FEATURE_TOOLTIPS["Candlestick Patterns"])
        feat_layout.addWidget(self.candlestick_check, 2, 0)

        lbl_higher_tf = QLabel("Higher TF:")
        lbl_higher_tf.setToolTip(PARAMETER_TOOLTIPS["higher_tf"])
        feat_layout.addWidget(lbl_higher_tf, 2, 1)
        self.higher_tf_combo = QComboBox()
        self.higher_tf_combo.addItems(["5m", "15m", "30m", "1h", "4h", "1d"])
        self.higher_tf_combo.setCurrentText("15m")
        self.higher_tf_combo.setToolTip(PARAMETER_TOOLTIPS["higher_tf"])
        feat_layout.addWidget(self.higher_tf_combo, 2, 2)

        # Volume Profile
        self.volume_profile_check = QCheckBox("Volume Profile")
        self.volume_profile_check.setChecked(False)
        self.volume_profile_check.setToolTip(FEATURE_TOOLTIPS["Volume Profile"])
        feat_layout.addWidget(self.volume_profile_check, 3, 0)

        lbl_vp_bins = QLabel("Bins:")
        lbl_vp_bins.setToolTip("Numero di livelli di prezzo per il volume profile.")
        feat_layout.addWidget(lbl_vp_bins, 3, 1)
        self.vp_bins = QSpinBox()
        self.vp_bins.setRange(10, 200)
        self.vp_bins.setValue(50)
        self.vp_bins.setToolTip("Quanti livelli di prezzo considerare per il volume profile.")
        feat_layout.addWidget(self.vp_bins, 3, 2)

        lbl_vp_window = QLabel("Window:")
        lbl_vp_window.setToolTip("Dimensione della finestra per calcolo volume profile (numero di candele).")
        feat_layout.addWidget(lbl_vp_window, 3, 3)
        self.vp_window = QSpinBox()
        self.vp_window.setRange(20, 500)
        self.vp_window.setValue(100)
        self.vp_window.setToolTip("Quante candele considerare per il volume profile (default: 100).")
        feat_layout.addWidget(self.vp_window, 3, 4)

        # VSA (Volume Spread Analysis)
        self.vsa_check = QCheckBox("VSA (Volume Spread Analysis)")
        self.vsa_check.setChecked(False)
        self.vsa_check.setToolTip("Analizza relazione volume/spread per identificare accumulo/distribuzione")
        feat_layout.addWidget(self.vsa_check, 4, 0)

        lbl_vsa_vol_ma = QLabel("Vol MA:")
        lbl_vsa_vol_ma.setToolTip("Periodo moving average per volume VSA")
        feat_layout.addWidget(lbl_vsa_vol_ma, 4, 1)
        self.vsa_volume_ma = QSpinBox()
        self.vsa_volume_ma.setRange(5, 100)
        self.vsa_volume_ma.setValue(20)
        self.vsa_volume_ma.setToolTip("Periodo MA per volume (default: 20)")
        feat_layout.addWidget(self.vsa_volume_ma, 4, 2)

        lbl_vsa_spread_ma = QLabel("Spread MA:")
        lbl_vsa_spread_ma.setToolTip("Periodo moving average per spread VSA")
        feat_layout.addWidget(lbl_vsa_spread_ma, 4, 3)
        self.vsa_spread_ma = QSpinBox()
        self.vsa_spread_ma.setRange(5, 100)
        self.vsa_spread_ma.setValue(20)
        self.vsa_spread_ma.setToolTip("Periodo MA per spread (default: 20)")
        feat_layout.addWidget(self.vsa_spread_ma, 4, 4)

        self.layout.addWidget(feat_box)

    def _build_advanced_params(self):
        """Build advanced parameters section"""
        adv_box = QGroupBox("Parametri Avanzati")
        adv_box.setToolTip("Parametri tecnici per feature engineering e preprocessing.")
        adv = QGridLayout(adv_box)

        row = 0

        # Warmup bars
        lbl_wu = QLabel("Warmup bars:")
        lbl_wu.setToolTip(PARAMETER_TOOLTIPS["warmup"])
        adv.addWidget(lbl_wu, row, 0)
        self.warmup = QSpinBox()
        self.warmup.setRange(0, 5000)
        self.warmup.setValue(16)
        self.warmup.setToolTip(PARAMETER_TOOLTIPS["warmup"])
        adv.addWidget(self.warmup, row, 1)

        # RV window
        lbl_rv = QLabel("RV window:")
        lbl_rv.setToolTip(PARAMETER_TOOLTIPS["rv_window"])
        adv.addWidget(lbl_rv, row, 2)
        self.rv_w = QSpinBox()
        self.rv_w.setRange(1, 10000)
        self.rv_w.setValue(60)
        self.rv_w.setToolTip(PARAMETER_TOOLTIPS["rv_window"])
        adv.addWidget(self.rv_w, row, 3)

        # Min coverage
        lbl_cov = QLabel("Min coverage:")
        lbl_cov.setToolTip(PARAMETER_TOOLTIPS["min_coverage"])
        adv.addWidget(lbl_cov, row, 4)
        self.min_coverage = QDoubleSpinBox()
        self.min_coverage.setRange(0.0, 1.0)
        self.min_coverage.setSingleStep(0.05)
        self.min_coverage.setDecimals(2)
        self.min_coverage.setValue(0.15)
        self.min_coverage.setToolTip(PARAMETER_TOOLTIPS["min_coverage"])
        adv.addWidget(self.min_coverage, row, 5)

        row += 1

        # ATR period
        lbl_atr = QLabel("ATR period:")
        lbl_atr.setToolTip("Numero di periodi per calcolare l'ATR.")
        adv.addWidget(lbl_atr, row, 0)
        self.atr_n = QSpinBox()
        self.atr_n.setRange(1, 500)
        self.atr_n.setValue(14)
        self.atr_n.setToolTip("Finestra dell'ATR (volatilità).")
        adv.addWidget(self.atr_n, row, 1)

        # RSI period
        lbl_rsi = QLabel("RSI period:")
        lbl_rsi.setToolTip("Numero di periodi per calcolare l'RSI.")
        adv.addWidget(lbl_rsi, row, 2)
        self.rsi_n = QSpinBox()
        self.rsi_n.setRange(2, 500)
        self.rsi_n.setValue(14)
        self.rsi_n.setToolTip("Finestra RSI (momento).")
        adv.addWidget(self.rsi_n, row, 3)

        # Bollinger period
        lbl_bb = QLabel("Bollinger period:")
        lbl_bb.setToolTip("Numero di barre per calcolare le bande di Bollinger.")
        adv.addWidget(lbl_bb, row, 4)
        self.bb_n = QSpinBox()
        self.bb_n.setRange(2, 500)
        self.bb_n.setValue(20)
        self.bb_n.setToolTip("Finestra per Bande di Bollinger.")
        adv.addWidget(self.bb_n, row, 5)

        row += 1

        # Hurst window
        lbl_hu = QLabel("Hurst window:")
        lbl_hu.setToolTip("Lunghezza finestra per stimare l'esponente di Hurst.")
        adv.addWidget(lbl_hu, row, 0)
        self.hurst_w = QSpinBox()
        self.hurst_w.setRange(8, 4096)
        self.hurst_w.setValue(64)
        self.hurst_w.setToolTip("Window per l'esponente di Hurst.")
        adv.addWidget(self.hurst_w, row, 1)

        # Lightning epochs (only for lightning model)
        lbl_epochs = QLabel("Lightning epochs:")
        lbl_epochs.setToolTip("Numero di epoche per il trainer Lightning.")
        adv.addWidget(lbl_epochs, row, 2)
        self.light_epochs = QSpinBox()
        self.light_epochs.setRange(1, 1000)
        self.light_epochs.setValue(30)
        self.light_epochs.setToolTip("Epoche per il trainer Lightning.")
        adv.addWidget(self.light_epochs, row, 3)

        # Lightning batch
        lbl_batch = QLabel("Lightning batch:")
        lbl_batch.setToolTip("Batch size per il trainer Lightning.")
        adv.addWidget(lbl_batch, row, 4)
        self.light_batch = QSpinBox()
        self.light_batch.setRange(4, 512)
        self.light_batch.setValue(64)
        self.light_batch.setToolTip("Batch size per il trainer Lightning.")
        adv.addWidget(self.light_batch, row, 5)

        row += 1

        # Lightning val_frac
        lbl_val = QLabel("Lightning val_frac:")
        lbl_val.setToolTip("Frazione dei dati riservata alla validation per Lightning.")
        adv.addWidget(lbl_val, row, 0)
        self.light_val_frac = QDoubleSpinBox()
        self.light_val_frac.setRange(0.05, 0.5)
        self.light_val_frac.setSingleStep(0.05)
        self.light_val_frac.setDecimals(2)
        self.light_val_frac.setValue(0.2)
        self.light_val_frac.setToolTip("Quota di dati usata per la validation (0.05-0.5).")
        adv.addWidget(self.light_val_frac, row, 1)

        # Lightning patch_len
        lbl_patch = QLabel("Lightning patch:")
        lbl_patch.setToolTip("Lunghezza della finestra (patch) per Lightning.")
        adv.addWidget(lbl_patch, row, 2)
        self.patch_len = QSpinBox()
        self.patch_len.setRange(16, 1024)
        self.patch_len.setValue(64)
        self.patch_len.setToolTip("Numero di barre nel patch passato al modello Lightning.")
        adv.addWidget(self.patch_len, row, 3)

        row += 1

        # Encoder latent dimension
        lbl_latent = QLabel("Encoder latent dim:")
        lbl_latent.setToolTip("Dimensione dello spazio latente per autoencoder/VAE.")
        adv.addWidget(lbl_latent, row, 0)
        self.latent_dim = QSpinBox()
        self.latent_dim.setRange(2, 256)
        self.latent_dim.setValue(16)
        self.latent_dim.setToolTip("Numero di dimensioni nello spazio latente compresso (per encoder neurali).")
        adv.addWidget(self.latent_dim, row, 1)

        # Encoder training epochs
        lbl_enc_epochs = QLabel("Encoder epochs:")
        lbl_enc_epochs.setToolTip("Numero di epoche di training per autoencoder/VAE.")
        adv.addWidget(lbl_enc_epochs, row, 2)
        self.encoder_epochs = QSpinBox()
        self.encoder_epochs.setRange(10, 500)
        self.encoder_epochs.setValue(50)
        self.encoder_epochs.setToolTip("Quante epoche addestrare l'encoder neurale.")
        adv.addWidget(self.encoder_epochs, row, 3)

        row += 1

        # === DIFFUSION MODEL PARAMETERS (only used when model=diffusion-*) ===
        lbl_diff_section = QLabel("─── Diffusion Model Parameters ───")
        lbl_diff_section.setStyleSheet("font-weight: bold; color: #2980b9;")
        adv.addWidget(lbl_diff_section, row, 0, 1, 6)
        row += 1

        # Diffusion timesteps
        lbl_timesteps = QLabel("Diffusion timesteps:")
        lbl_timesteps.setToolTip(
            "Numero di timesteps del processo di diffusione.\n"
            "Cosa è: quanti step di denoising usare (T nella formula DDPM).\n"
            "Perché è importante: più steps = migliore qualità, ma inference più lenta.\n"
            "Valori bassi (10-50): veloce, qualità media, buono per test rapidi.\n"
            "Valori medi (100-500): bilanciamento qualità/velocità, raccomandato.\n"
            "Valori alti (1000-5000): massima qualità, molto lento (minuti per sample).\n"
            "Best practice: DDPM usa 1000, DDIM può usare 50-200 (10x più veloce).\n"
            "SOLO per model=diffusion-ddpm o diffusion-ddim."
        )
        adv.addWidget(lbl_timesteps, row, 0)
        self.diffusion_timesteps = QSpinBox()
        self.diffusion_timesteps.setRange(10, 5000)
        self.diffusion_timesteps.setValue(200)
        self.diffusion_timesteps.setToolTip(
            "T timesteps per denoising.\n"
            "10-50: test rapido, bassa qualità.\n"
            "100-500: raccomandato, bilanciato.\n"
            "1000-5000: ricerca, massima qualità.\n"
            "Tempo inference: ~T × 50ms per sample (GPU)."
        )
        adv.addWidget(self.diffusion_timesteps, row, 1)

        # Learning rate
        lbl_lr = QLabel("Learning rate:")
        lbl_lr.setToolTip(
            "Step size per gradient descent durante training.\n"
            "Cosa è: quanto velocemente il modello aggiorna i pesi.\n"
            "Perché è importante: LR troppo alto → divergenza, troppo basso → non converge.\n"
            "Valori bassi (1e-6 - 1e-5): training stabile ma lentissimo, usa per fine-tuning.\n"
            "Valori medi (1e-4 - 5e-4): raccomandato per diffusion/lightning, stabile.\n"
            "Valori alti (1e-3 - 1e-2): training veloce ma instabile, rischio divergenza.\n"
            "Best practice: diffusion → 1e-4, lightning → 1e-3, usa warmup + scheduler.\n"
            "SOLO per model=lightning/diffusion-*."
        )
        adv.addWidget(lbl_lr, row, 2)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(1e-6, 1e-1)
        self.learning_rate.setSingleStep(1e-5)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setValue(1e-4)
        self.learning_rate.setToolTip(
            "Step size ottimizzatore.\n"
            "1e-6 - 1e-5: ultra safe, lentissimo.\n"
            "1e-4 - 5e-4: raccomandato (diffusion/lightning).\n"
            "1e-3 - 1e-2: veloce ma rischioso (divergenza).\n"
            "Monitor loss: se NaN/Inf, riduci LR 10x."
        )
        adv.addWidget(self.learning_rate, row, 3)

        # Batch size (diffusion/lightning)
        lbl_batch_diff = QLabel("Batch size (DL):")
        lbl_batch_diff.setToolTip(
            "Numero di samples per batch durante training deep learning.\n"
            "Cosa è: quanti esempi elaborare in parallelo prima di aggiornare pesi.\n"
            "Perché è importante: batch grande = gradiente stabile, batch piccolo = più noise.\n"
            "Valori bassi (4-16): gradient noisy, generalizza meglio, usa meno RAM.\n"
            "Valori medi (32-128): bilanciamento, raccomandato.\n"
            "Valori alti (256-512): gradient smooth, converge veloce, serve molta RAM/GPU.\n"
            "Best practice: max batch che sta in GPU memory, usa gradient accumulation.\n"
            "SOLO per model=lightning/diffusion-*."
        )
        adv.addWidget(lbl_batch_diff, row, 4)
        self.batch_size_dl = QSpinBox()
        self.batch_size_dl.setRange(4, 512)
        self.batch_size_dl.setValue(64)
        self.batch_size_dl.setToolTip(
            "Samples per batch (deep learning).\n"
            "4-16: bassa RAM, gradient noisy, generalizza.\n"
            "32-128: raccomandato, bilanciato.\n"
            "256-512: serve GPU potente (8+ GB VRAM).\n"
            "Nota: diverso da 'Lightning batch' (supervised)."
        )
        adv.addWidget(self.batch_size_dl, row, 5)

        row += 1

        # Model channels (UNet capacity)
        lbl_channels = QLabel("Model channels:")
        lbl_channels.setToolTip(
            "Numero di canali base per architettura UNet/Transformer.\n"
            "Cosa è: capacità del modello (simile a 'width' di una neural network).\n"
            "Perché è importante: più canali = più parametri = più capacity, ma overfitting.\n"
            "Valori bassi (32-64): modello piccolo, veloce, rischio underfitting.\n"
            "Valori medi (128-192): bilanciamento capacity/overfitting, raccomandato.\n"
            "Valori alti (256-512): massima capacity, solo con dati massivi (>1M samples).\n"
            "Best practice: 128 per dataset medi, 256 per dataset grandi (anni di 1m data).\n"
            "SOLO per model=diffusion-* (parametri UNet = channels² × layers)."
        )
        adv.addWidget(lbl_channels, row, 0)
        self.model_channels = QSpinBox()
        self.model_channels.setRange(32, 512)
        self.model_channels.setValue(128)
        self.model_channels.setToolTip(
            "Canali base UNet.\n"
            "32-64: modello piccolo, <1M params, veloce.\n"
            "128-192: raccomandato, ~5-20M params.\n"
            "256-512: modello enorme, >50M params, serve GPU potente.\n"
            "Parametri totali ≈ channels² × num_layers."
        )
        adv.addWidget(self.model_channels, row, 1)

        # Dropout
        lbl_dropout = QLabel("Dropout:")
        lbl_dropout.setToolTip(
            "Frazione di neuroni disattivati random durante training (regolarizzazione).\n"
            "Cosa è: tecnica per prevenire overfitting disattivando random connections.\n"
            "Perché è importante: riduce overfitting, forza il modello a essere robusto.\n"
            "Valori bassi (0.0-0.1): nessuna/poca regolarizzazione, rischio overfitting.\n"
            "Valori medi (0.1-0.3): bilanciamento, raccomandato per la maggior parte dei casi.\n"
            "Valori alti (0.4-0.6): forte regolarizzazione, rischio underfitting.\n"
            "Best practice: 0.0 per dataset piccoli, 0.1-0.3 per dataset grandi.\n"
            "SOLO per model=lightning/diffusion-* (neural networks)."
        )
        adv.addWidget(lbl_dropout, row, 2)
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.0, 0.6)
        self.dropout.setSingleStep(0.05)
        self.dropout.setDecimals(2)
        self.dropout.setValue(0.1)
        self.dropout.setToolTip(
            "Dropout probability.\n"
            "0.0: nessuna regolarizzazione (overfitting risk).\n"
            "0.1-0.3: raccomandato, bilanciato.\n"
            "0.4-0.6: forte regolarizzazione (underfitting risk).\n"
            "Disabilitato durante inference (automatico)."
        )
        adv.addWidget(self.dropout, row, 3)

        # Num heads (Transformer)
        lbl_heads = QLabel("Attention heads:")
        lbl_heads.setToolTip(
            "Numero di teste di attenzione per architettura Transformer.\n"
            "Cosa è: parallelizzazione del meccanismo di attention (multi-head attention).\n"
            "Perché è importante: più heads = cattura pattern diversi in parallelo.\n"
            "Valori bassi (1-2): attenzione semplice, veloce, capacity limitata.\n"
            "Valori medi (4-8): raccomandato, bilanciamento capacity/complessità.\n"
            "Valori alti (12-16): massima capacity, solo per dataset massivi.\n"
            "Best practice: num_heads deve dividere model_channels (es. 128/8=16).\n"
            "SOLO per model=diffusion-* con architecture=transformer."
        )
        adv.addWidget(lbl_heads, row, 4)
        self.num_heads = QSpinBox()
        self.num_heads.setRange(1, 16)
        self.num_heads.setValue(8)
        self.num_heads.setToolTip(
            "Teste attention (Transformer).\n"
            "1-2: semplice, veloce.\n"
            "4-8: raccomandato (standard).\n"
            "12-16: massima capacity (GPT-like).\n"
            "Vincolo: model_channels % num_heads == 0."
        )
        adv.addWidget(self.num_heads, row, 5)

        self.layout.addWidget(adv_box)

        # NVIDIA Optimization Stack section
        self._build_nvidia_optimizations()

    def _build_nvidia_optimizations(self):
        """Build NVIDIA GPU Optimization Stack section"""
        nvidia_box = QGroupBox("🚀 NVIDIA Optimization Stack (GPU Acceleration)")
        nvidia_box.setToolTip(
            "Ottimizzazioni NVIDIA per accelerare il training su GPU.\n"
            "Richiede GPU NVIDIA con CUDA. Speedup fino a 30x!\n"
            "Automatic Mixed Precision (AMP), torch.compile, fused optimizers, Flash Attention."
        )
        nvidia_layout = QGridLayout(nvidia_box)

        row = 0

        # Master enable checkbox
        self.nvidia_enable = QCheckBox("Abilita NVIDIA Optimization Stack")
        self.nvidia_enable.setToolTip(
            "Abilita tutte le ottimizzazioni NVIDIA disponibili:\n"
            "- Automatic Mixed Precision (AMP)\n"
            "- torch.compile per model optimization\n"
            "- Fused optimizers (APEX)\n"
            "- Channels last memory format\n"
            "Speedup: 2-30x più veloce su GPU NVIDIA"
        )
        self.nvidia_enable.setChecked(False)
        nvidia_layout.addWidget(self.nvidia_enable, row, 0, 1, 6)
        row += 1

        # AMP checkbox
        lbl_amp = QLabel("Mixed Precision (AMP):")
        lbl_amp.setToolTip(
            "Automatic Mixed Precision - usa FP16/BF16 invece di FP32.\n"
            "Speedup: 2-3x più veloce, 50% meno memoria."
        )
        nvidia_layout.addWidget(lbl_amp, row, 0)

        self.use_amp = QCheckBox("Enable")
        self.use_amp.setChecked(True)
        self.use_amp.setToolTip("Abilita AMP (raccomandato)")
        nvidia_layout.addWidget(self.use_amp, row, 1)

        # Precision combo
        lbl_precision = QLabel("Precision:")
        nvidia_layout.addWidget(lbl_precision, row, 2)

        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["fp16", "bf16", "fp32"])
        self.precision_combo.setCurrentText("fp16")
        self.precision_combo.setToolTip(
            "Tipo di precisione:\n"
            "fp16: FP16 (raccomandato, GPU >= GTX 10xx)\n"
            "bf16: BrainFloat16 (GPU Ampere+: RTX 30xx/40xx)\n"
            "fp32: Full precision (no speedup)"
        )
        nvidia_layout.addWidget(self.precision_combo, row, 3)
        row += 1

        # torch.compile checkbox
        lbl_compile = QLabel("torch.compile:")
        lbl_compile.setToolTip(
            "Compila il modello con PyTorch 2.0+ compiler.\n"
            "Speedup: 1.5-2x più veloce.\n"
            "Richiede PyTorch >= 2.0"
        )
        nvidia_layout.addWidget(lbl_compile, row, 0)

        self.compile_model = QCheckBox("Enable")
        self.compile_model.setChecked(True)
        self.compile_model.setToolTip("Abilita torch.compile (PyTorch 2.0+)")
        nvidia_layout.addWidget(self.compile_model, row, 1)
        row += 1

        # Fused optimizer checkbox
        lbl_fused = QLabel("Fused Optimizer:")
        lbl_fused.setToolTip(
            "Usa NVIDIA APEX fused optimizer.\n"
            "Speedup: 1.2-1.5x più veloce.\n"
            "Richiede: pip install apex (vedi NVIDIA_INSTALLATION.md)"
        )
        nvidia_layout.addWidget(lbl_fused, row, 0)

        self.use_fused_optimizer = QCheckBox("Enable (requires APEX)")
        self.use_fused_optimizer.setChecked(False)  # Default off (requires APEX)
        self.use_fused_optimizer.setToolTip("Richiede NVIDIA APEX installato")
        nvidia_layout.addWidget(self.use_fused_optimizer, row, 1)
        row += 1

        # Flash Attention checkbox
        lbl_flash = QLabel("Flash Attention 2:")
        lbl_flash.setToolTip(
            "Flash Attention 2 - attenzione ultra-veloce.\n"
            "Speedup: 2-4x più veloce per transformer.\n"
            "Richiede: GPU Ampere+ (RTX 30xx/40xx, A100)"
        )
        nvidia_layout.addWidget(lbl_flash, row, 0)

        self.use_flash_attention = QCheckBox("Enable (requires Ampere+ GPU)")
        self.use_flash_attention.setChecked(False)  # Default off (requires Ampere+)
        self.use_flash_attention.setToolTip("Solo GPU Ampere+: RTX 30xx/40xx")
        nvidia_layout.addWidget(self.use_flash_attention, row, 1)
        row += 1

        # Gradient accumulation
        lbl_grad_accum = QLabel("Gradient Accumulation:")
        lbl_grad_accum.setToolTip(
            "Accumula gradienti su N steps prima di update.\n"
            "Simula batch size più grandi senza usare più memoria."
        )
        nvidia_layout.addWidget(lbl_grad_accum, row, 0)

        self.grad_accumulation_steps = QSpinBox()
        self.grad_accumulation_steps.setRange(1, 32)
        self.grad_accumulation_steps.setValue(1)
        self.grad_accumulation_steps.setToolTip("1 = no accumulation, >1 = accumulate gradients")
        nvidia_layout.addWidget(self.grad_accumulation_steps, row, 1)
        row += 1

        # Info label
        info_label = QLabel(
            "ℹ️ Per installare APEX e Flash Attention:\n"
            "   python install_nvidia_stack.py --all\n"
            "   Vedi NVIDIA_INSTALLATION.md per dettagli."
        )
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        nvidia_layout.addWidget(info_label, row, 0, 1, 6)

        self.layout.addWidget(nvidia_box)

    def _build_output_section(self):
        """Build output directory section"""
        out_h = QHBoxLayout()

        default_out_dir = None
        try:
            default_out_dir = getattr(getattr(self.cfg, "model", None), "artifacts_dir", None)
        except Exception:
            default_out_dir = None
        default_out_dir = default_out_dir or "./artifacts"

        self.out_dir = QLineEdit(str(Path(default_out_dir)))
        self.browse_btn = QPushButton("Scegli Cartella...")
        self.browse_btn.clicked.connect(self._browse_out)

        out_h.addWidget(QLabel("Output dir:"))
        out_h.addWidget(self.out_dir)
        out_h.addWidget(self.browse_btn)

        self.layout.addLayout(out_h)

    def _build_optimized_params_section(self):
        """Build optimized parameters display section (FASE 9 - Part 2)"""
        self.optimized_params_widget = OptimizedParamsDisplayWidget()
        self.optimized_params_widget.save_requested.connect(self._on_save_optimized_params)
        self.layout.addWidget(self.optimized_params_widget)

    def _build_log_section(self):
        """Build log and progress section"""
        lp = QHBoxLayout()

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(47)  # Reduced to 33% of 140
        self.log_view.setMaximumHeight(100)
        self.log_view.setTextInteractionFlags(
            self.log_view.textInteractionFlags() | Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )

        lp.addWidget(self.progress, 1)
        lp.addWidget(self.log_view, 3)

        self.layout.addLayout(lp)

    def _build_actions(self):
        """Build action buttons"""
        actions = QHBoxLayout()

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self._start_training)

        self.validate_btn = QPushButton("Multi-Horizon Validation")
        self.validate_btn.clicked.connect(self._start_multi_horizon_validation)
        self.validate_btn.setToolTip(
            "Validate trained model across multiple forecast horizons\n"
            "to identify optimal prediction window and assess performance degradation"
        )

        self.grid_training_btn = QPushButton("Grid Training Manager")
        self.grid_training_btn.clicked.connect(self._open_grid_training)
        self.grid_training_btn.setToolTip(
            "Open Grid Training Manager to train multiple model configurations\n"
            "with regime-based selection and automatic performance optimization"
        )
        self.grid_training_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        actions.addWidget(self.train_btn)
        actions.addWidget(self.validate_btn)
        actions.addWidget(self.grid_training_btn)
        self.layout.addLayout(actions)

    def _browse_out(self):
        """Browse for output directory"""
        d = QFileDialog.getExistingDirectory(self, "Scegli cartella output", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _collect_indicator_tfs(self) -> Dict[str, List[str]]:
        """Collect selected indicator timeframes"""
        if not self.use_indicators_check.isChecked():
            return {}

        m: Dict[str, List[str]] = {}
        for ind in INDICATORS:
            tfs = [tf for tf, cb in self.indicator_checks[ind].items() if cb.isChecked()]
            if tfs:
                # Normalize indicator name to lowercase for backend
                ind_key = ind.lower().replace("%", "").replace("&", "").replace(" ", "_")
                m[ind_key] = tfs
        return m

    def _persist_indicator_tfs(self):
        """Persist indicator selections to settings"""
        m = {ind: [tf for tf, cb in self.indicator_checks[ind].items() if cb.isChecked()]
             for ind in INDICATORS}
        set_setting("training_indicator_tfs", m)

    def _suggest_horizons(self):
        """Show horizon suggestions dialog."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QTextEdit, QHBoxLayout
        
        try:
            from ..utils.horizon_suggestions import suggest_horizons, describe_horizons, get_styles
            
            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Horizon Suggestions")
            dialog.resize(500, 400)
            layout = QVBoxLayout(dialog)
            
            # Current timeframe
            tf = self.tf_combo.currentText()
            layout.addWidget(QLabel(f"<b>Timeframe:</b> {tf}"))
            layout.addSpacing(10)
            
            # Style selector
            style_layout = QHBoxLayout()
            style_layout.addWidget(QLabel("Trading Style:"))
            style_combo = QComboBox()
            styles = get_styles()
            style_combo.addItems([s.capitalize() for s in styles])
            style_combo.setCurrentText("Balanced")
            style_layout.addWidget(style_combo)
            style_layout.addStretch()
            layout.addLayout(style_layout)
            
            # Suggestions display
            suggestions_text = QTextEdit()
            suggestions_text.setReadOnly(True)
            suggestions_text.setMaximumHeight(200)
            layout.addWidget(QLabel("<b>Suggestions:</b>"))
            layout.addWidget(suggestions_text)
            
            def update_suggestions():
                style = style_combo.currentText().lower()
                try:
                    horizons = suggest_horizons(tf, style)
                    from ..utils.horizon_parser import format_horizon_spec
                    horizons_str = format_horizon_spec(horizons)
                    description = describe_horizons(horizons, tf)
                    
                    text = f"<b>Recommended Horizons:</b> {horizons_str}\n\n"
                    text += f"<b>Description:</b> {description}\n\n"
                    text += f"<b>Style:</b> {style.capitalize()}\n"
                    text += f"<b>Timeframe:</b> {tf}\n\n"
                    text += f"<i>These horizons are optimized for {style} trading on {tf} timeframe.</i>"
                    
                    suggestions_text.setHtml(text)
                except Exception as e:
                    suggestions_text.setPlainText(f"Error: {e}")
            
            # Update on style change
            style_combo.currentTextChanged.connect(lambda: update_suggestions())
            update_suggestions()
            
            # Buttons
            button_layout = QHBoxLayout()
            apply_btn = QPushButton("✓ Apply")
            cancel_btn = QPushButton("✗ Cancel")
            
            def apply_suggestion():
                style = style_combo.currentText().lower()
                horizons = suggest_horizons(tf, style)
                from ..utils.horizon_parser import format_horizon_spec
                horizons_str = format_horizon_spec(horizons)
                self.horizon_spin.setText(horizons_str)
                dialog.accept()
            
            apply_btn.clicked.connect(apply_suggestion)
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addStretch()
            button_layout.addWidget(apply_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            dialog.exec()
            
        except Exception as e:
            from loguru import logger
            logger.exception(f"Failed to show horizon suggestions: {e}")
            QMessageBox.warning(self, "Error", f"Failed to show suggestions:\n{e}")
    
    def _start_training(self):
        """Start training process"""
        try:
            # Clear previous optimized params display
            self.optimized_params_widget.clear()

            # Save current settings before training
            self._save_settings()

            sym = self.symbol_combo.currentText()
            tf = self.tf_combo.currentText()
            days = int(self.days_spin.value())
            horizon_str = self.horizon_spin.text().strip()
            model = self.model_combo.currentText()
            encoder = self.encoder_combo.currentText()
            
            # Validate horizon format
            if not horizon_str:
                QMessageBox.warning(self, "Invalid Horizon", "Please specify a horizon value.")
                return
            
            try:
                from ..utils.horizon_parser import parse_horizon_spec
                horizons = parse_horizon_spec(horizon_str)
                logger.info(f"Training with horizons: {horizons}")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Invalid Horizon Format",
                    f"Could not parse horizon '{horizon_str}'.\n\n"
                    f"Error: {e}\n\n"
                    f"Supported formats:\n"
                    f"  - Single: 5\n"
                    f"  - List: 15,60,240\n"
                    f"  - Range: 1-7\n"
                    f"  - Range with step: 1-7/2\n"
                    f"  - Mixed: 1-7/2,60,100-200/50"
                )
                return

            # Collect indicators
            ind_tfs = self._collect_indicator_tfs()
            ind_tfs_json = json.dumps(ind_tfs)
            self._persist_indicator_tfs()

            # Generate run name
            tfs_flat = sorted({tf_sel for values in ind_tfs.values() for tf_sel in values})
            tfs_str = '-'.join(tfs_flat) if tfs_flat else 'none'
            name = f"{sym.replace('/', '')}_{tf}_d{days}_h{horizon}_{model}_{encoder}_ind{len(ind_tfs)}_{tfs_str}"

            # Resolve output directory
            out_dir = Path(self.out_dir.text()).resolve()
            artifacts_dir = out_dir if out_dir.name.lower() != 'models' else out_dir.parent
            root = Path(__file__).resolve().parents[3]

            strategy = self.opt_combo.currentText()
            # Optimization is now implemented for both sklearn and lightning models

            from datetime import datetime, timezone

            # Additional features config
            additional_features = {
                'returns': self.returns_check.isChecked(),
                'sessions': self.sessions_check.isChecked(),
                'candlestick': self.candlestick_check.isChecked(),
                'volume_profile': self.volume_profile_check.isChecked(),
            }

            # Lightning and diffusion models use train.py
            if model in ['lightning', 'diffusion-ddpm', 'diffusion-ddim']:
                module = 'src.forex_diffusion.training.train'
                args = [
                    sys.executable, '-m', module,
                    '--symbol', sym,
                    '--timeframe', tf,
                    '--horizon', horizon_str,
                    '--days_history', str(days),
                    '--patch_len', str(int(self.patch_len.value())),
                    '--epochs', str(int(self.light_epochs.value())),
                    '--batch_size', str(int(self.light_batch.value())),
                    '--val_frac', f"{self.light_val_frac.value():.2f}",
                    '--artifacts_dir', str(artifacts_dir),
                    '--indicator_tfs', ind_tfs_json,
                    '--min_feature_coverage', f"{self.min_coverage.value():.2f}",
                    '--warmup_bars', str(int(self.warmup.value())),
                    '--atr_n', str(int(self.atr_n.value())),
                    '--rsi_n', str(int(self.rsi_n.value())),
                    '--bb_n', str(int(self.bb_n.value())),
                    '--hurst_window', str(int(self.hurst_w.value())),
                    '--rv_window', str(int(self.rv_w.value())),
                    '--returns_window', str(int(self.returns_window.value())),
                    '--session_overlap', str(int(self.session_overlap.value())),
                    '--higher_tf', self.higher_tf_combo.currentText(),
                    '--vp_bins', str(int(self.vp_bins.value())),
                    '--vp_window', str(int(self.vp_window.value())),
                ]

                if self.vsa_check.isChecked():
                    args.extend([
                        '--use_vsa',
                        '--vsa_volume_ma', str(int(self.vsa_volume_ma.value())),
                        '--vsa_spread_ma', str(int(self.vsa_spread_ma.value())),
                    ])

                # Add NVIDIA Optimization Stack arguments if enabled
                if self.nvidia_enable.isChecked() or self.use_amp.isChecked():
                    args.append('--use_nvidia_opts')
                if self.use_amp.isChecked():
                    args.append('--use_amp')
                    args.extend(['--precision', self.precision_combo.currentText()])
                if self.compile_model.isChecked():
                    args.append('--compile_model')
                if self.use_fused_optimizer.isChecked():
                    args.append('--use_fused_optimizer')
                if self.use_flash_attention.isChecked():
                    args.append('--use_flash_attention')
                if self.grad_accumulation_steps.value() > 1:
                    args.extend(['--gradient_accumulation_steps', str(self.grad_accumulation_steps.value())])

                meta = {
                    'symbol': sym,
                    'base_timeframe': tf,
                    'days_history': int(days),
                    'horizon_bars': int(horizon),
                    'trainer': 'lightning',
                    'lightning_params': {
                        'epochs': int(self.light_epochs.value()),
                        'batch_size': int(self.light_batch.value()),
                        'val_frac': float(self.light_val_frac.value()),
                        'patch_len': int(self.patch_len.value()),
                    },
                    'nvidia_optimizations': {
                        'enabled': self.nvidia_enable.isChecked(),
                        'use_amp': self.use_amp.isChecked(),
                        'precision': self.precision_combo.currentText(),
                        'compile_model': self.compile_model.isChecked(),
                        'use_fused_optimizer': self.use_fused_optimizer.isChecked(),
                        'use_flash_attention': self.use_flash_attention.isChecked(),
                        'gradient_accumulation_steps': int(self.grad_accumulation_steps.value()),
                    },
                    'indicator_tfs': ind_tfs,
                    'additional_features': additional_features,
                    'advanced_params': {
                        'warmup_bars': int(self.warmup.value()),
                        'atr_n': int(self.atr_n.value()),
                        'rsi_n': int(self.rsi_n.value()),
                        'bb_n': int(self.bb_n.value()),
                        'hurst_window': int(self.hurst_w.value()),
                        'rv_window': int(self.rv_w.value()),
                        'min_feature_coverage': float(self.min_coverage.value()),
                        'returns_window': int(self.returns_window.value()),
                        'session_overlap': int(self.session_overlap.value()),
                        'higher_tf': self.higher_tf_combo.currentText(),
                        'vp_bins': int(self.vp_bins.value()),
                        'vp_window': int(self.vp_window.value()),
                        'use_vsa': self.vsa_check.isChecked(),
                        'vsa_volume_ma': int(self.vsa_volume_ma.value()),
                        'vsa_spread_ma': int(self.vsa_spread_ma.value()),
                    },
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'ui_run_name': name,
                }
                pending_dir = artifacts_dir / 'lightning'
            else:
                module = 'src.forex_diffusion.training.train_sklearn'
                algo = model if model != 'latents' else 'ridge'
                # Use new encoder system instead of legacy pca flag
                args = [
                    sys.executable, '-m', module,
                    '--symbol', sym,
                    '--timeframe', tf,
                    '--horizon', horizon_str,
                    '--algo', algo,
                    '--encoder', encoder,
                    '--latent_dim', str(int(self.latent_dim.value())),
                    '--encoder_epochs', str(int(self.encoder_epochs.value())),
                    '--artifacts_dir', str(artifacts_dir),
                    '--warmup_bars', str(int(self.warmup.value())),
                    '--val_frac', '0.2',
                    '--alpha', '0.001',
                    '--l1_ratio', '0.5',
                    '--days_history', str(days),
                    '--indicator_tfs', ind_tfs_json,
                    '--min_feature_coverage', f"{self.min_coverage.value():.2f}",
                    '--atr_n', str(int(self.atr_n.value())),
                    '--rsi_n', str(int(self.rsi_n.value())),
                    '--bb_n', str(int(self.bb_n.value())),
                    '--hurst_window', str(int(self.hurst_w.value())),
                    '--rv_window', str(int(self.rv_w.value())),
                    '--returns_window', str(int(self.returns_window.value())),
                    '--session_overlap', str(int(self.session_overlap.value())),
                    '--higher_tf', self.higher_tf_combo.currentText(),
                    '--vp_bins', str(int(self.vp_bins.value())),
                    '--optimization', strategy,
                    '--gen', str(int(self.gen_spin.value())),
                    '--pop', str(int(self.pop_spin.value())),
                    '--random_state', '0',
                    '--n_estimators', '400',
                ]

                # Add GPU flag if enabled
                if self.use_gpu_training_check.isChecked():
                    args.append('--use-gpu')
                meta = {
                    'symbol': sym,
                    'base_timeframe': tf,
                    'days_history': int(days),
                    'horizon_bars': int(horizon),
                    'model_type': model,
                    'encoder': encoder,
                    'indicator_tfs': ind_tfs,
                    'additional_features': additional_features,
                    'advanced_params': {
                        'warmup_bars': int(self.warmup.value()),
                        'atr_n': int(self.atr_n.value()),
                        'rsi_n': int(self.rsi_n.value()),
                        'bb_n': int(self.bb_n.value()),
                        'hurst_window': int(self.hurst_w.value()),
                        'rv_window': int(self.rv_w.value()),
                        'min_feature_coverage': float(self.min_coverage.value()),
                        'returns_window': int(self.returns_window.value()),
                        'session_overlap': int(self.session_overlap.value()),
                        'higher_tf': self.higher_tf_combo.currentText(),
                        'vp_bins': int(self.vp_bins.value()),
                        'vp_window': int(self.vp_window.value()),
                        'use_vsa': self.vsa_check.isChecked(),
                        'vsa_volume_ma': int(self.vsa_volume_ma.value()),
                        'vsa_spread_ma': int(self.vsa_spread_ma.value()),
                    },
                    'optimization': strategy,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'ui_run_name': name,
                }
                pending_dir = artifacts_dir / 'models'

            self._pending_meta = meta
            self._pending_out_dir = pending_dir
            
            # Check for unsupported parameters
            from .model_parameter_compatibility import get_model_type_from_model_name, get_unsupported_params
            
            model_type = get_model_type_from_model_name(model)
            
            # Extract parameter names from args
            param_dict = {}
            for i in range(len(args)):
                if args[i].startswith('--'):
                    param_name = args[i][2:].replace('-', '_')
                    param_dict[param_name] = True
            
            unsupported = get_unsupported_params(model_type, param_dict)
            
            if unsupported:
                # Show warning dialog
                unsupported_list = "\n".join([f"• {desc} (--{param})" for param, desc in unsupported])
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Unsupported Parameters")
                msg.setText(f"Some parameters are not supported by the '{model}' model:")
                msg.setInformativeText(
                    f"The following parameters will be ignored:\n\n{unsupported_list}\n\n"
                    f"Training will proceed without these features."
                )
                msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                msg.setDefaultButton(QMessageBox.Ok)
                
                if msg.exec() == QMessageBox.Cancel:
                    self._append_log("[cancelled] Training cancelled by user due to unsupported parameters")
                    return
                
                self._append_log(f"[warning] Proceeding with {len(unsupported)} unsupported parameter(s)")
            
            # Log meta in a more readable format
            self._append_log("[meta] Training parameters prepared:")
            self._append_log(json.dumps(meta, indent=2, default=str))

            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self._append_log(f"\n[command] Starting training process...")
            self._append_log(f"[command] {' '.join(args)}")
            self._append_log(f"[command] Working directory: {root}\n")
            self.controller.start_training(args, cwd=str(root))
            self._append_log("[status] Training process launched, waiting for output...")

        except Exception as e:
            logger.exception("Start training error: {}", e)
            QMessageBox.warning(self, 'Training', str(e))

    def _append_log(self, line: str):
        """Append line to log"""
        try:
            self.log_view.append(line)
        except Exception:
            pass

    def _find_latest_model_file(self, out_dir: Path) -> Optional[Path]:
        """Find the most recently modified model file (.pt/.pth/.pkl/.pickle) in out_dir."""
        try:
            cand = []
            for ext in ("*.pt", "*.pth", "*.pkl", "*.pickle"):
                cand += list(out_dir.glob(ext))
            if not cand:
                return None
            cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cand[0]
        except Exception:
            return None

    def _on_progress(self, value: int):
        """Handle progress update"""
        if value < 0:
            self.progress.setRange(0, 0)  # indeterminate
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(value)

    def _on_finished(self, ok: bool):
        """Handle training completion"""
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if ok else 0)
        self._append_log("[done] ok" if ok else "[done] failed")

        if ok:
            # Attach meta to latest model file using MetadataManager
            try:
                if self._pending_out_dir and self._pending_out_dir.exists() and isinstance(self._pending_meta, dict):
                    latest = self._find_latest_model_file(self._pending_out_dir)
                    if latest:
                        from ..models.metadata_manager import MetadataManager, ModelMetadata

                        # Create ModelMetadata object with correct structure
                        metadata = ModelMetadata()
                        metadata.model_path = str(latest)
                        metadata.file_size = latest.stat().st_size if latest.exists() else 0

                        # Map training metadata to ModelMetadata attributes
                        meta = self._pending_meta
                        metadata.symbol = meta.get('symbol')
                        metadata.base_timeframe = meta.get('base_timeframe')
                        metadata.horizon_bars = meta.get('horizon_bars')
                        metadata.horizon_minutes = None
                        metadata.model_type = meta.get('model_type', 'sklearn')
                        metadata.model_class = meta.get('model_type', 'unknown')
                        metadata.created_at = meta.get('created_at')

                        # Feature configuration - load from model file
                        try:
                            import joblib
                            model_data = joblib.load(latest)
                            if isinstance(model_data, dict):
                                feature_names = model_data.get('features', [])
                                metadata.feature_names = feature_names
                                metadata.num_features = len(feature_names)
                                self._append_log(f"[meta] extracted {len(feature_names)} features from model")
                            else:
                                metadata.feature_names = []
                                metadata.num_features = 0
                        except Exception as e:
                            logger.warning(f"Failed to extract features from model: {e}")
                            metadata.feature_names = []
                            metadata.num_features = 0

                        # Advanced parameters
                        if 'advanced_params' in meta:
                            metadata.preprocessing_config = meta['advanced_params']

                        # Indicator configuration
                        if 'indicator_tfs' in meta:
                            metadata.multi_timeframe_config = {'indicator_tfs': meta['indicator_tfs']}
                            metadata.multi_timeframe_enabled = True

                        # Training parameters
                        metadata.training_params = {
                            'days_history': meta.get('days_history'),
                            'optimization': meta.get('optimization'),
                            'encoder': meta.get('encoder')
                        }

                        # Save using MetadataManager
                        manager = MetadataManager()
                        manager.save_metadata(metadata, str(latest))
                        self._append_log(f"[meta] saved sidecar: {latest}.meta.json")
                    else:
                        self._append_log("[meta] no model file found to attach meta")
            except Exception as e:
                logger.exception("Metadata save error")
                self._append_log(f"[meta] save failed: {e}")

            QMessageBox.information(self, "Training", "Training completato.")
        else:
            QMessageBox.warning(self, "Training", "Training fallito.")

    def _on_save_optimized_params(self, params: Dict[str, Any]):
        """
        Handle save optimized parameters to database.

        This method is called when user clicks "Save to Database" in the
        optimized params widget (FASE 9 - Part 2).

        NOTE: Full implementation requires ParameterLoaderService integration.
        For now, this is a placeholder that shows the save dialog.
        """
        try:
            # TODO: Integrate with ParameterLoaderService from FASE 2
            # from ..services.parameter_loader import ParameterLoaderService
            # loader = ParameterLoaderService(db_path)
            # success = loader.save_optimized_params(params)

            # For now, just log and show success message
            logger.info(f"Optimized params save requested: {params.get('pattern_type')} on {params.get('symbol')} {params.get('timeframe')}")

            QMessageBox.information(
                self,
                "Save Parameters",
                "Optimized parameters saved successfully!\n\n"
                "NOTE: Full database integration pending FASE 2 implementation.\n"
                "Parameters logged for development tracking."
            )

        except Exception as e:
            logger.exception(f"Failed to save optimized params: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save optimized parameters:\n{str(e)}"
            )

    def _on_load_config(self):
        """Load training configuration from JSON file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Training Config",
                str(Path.home()),
                "JSON Files (*.json);;All Files (*.*)"
            )
            if not file_path:
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Top controls
            if 'model_name' in config:
                self.model_name_edit.setText(config['model_name'])
            if 'symbol' in config:
                self.symbol_combo.setCurrentText(config['symbol'])
            if 'timeframe' in config:
                self.tf_combo.setCurrentText(config['timeframe'])
            if 'days_history' in config:
                self.days_spin.setValue(config['days_history'])
            if 'horizon' in config:
                self.horizon_spin.setValue(config['horizon'])
            if 'model' in config:
                self.model_combo.setCurrentText(config['model'])
            if 'encoder' in config:
                self.encoder_combo.setCurrentText(config['encoder'])
            if 'optimization' in config:
                self.opt_combo.setCurrentText(config['optimization'])
            if 'gen' in config:
                self.gen_spin.setValue(config['gen'])
            if 'pop' in config:
                self.pop_spin.setValue(config['pop'])

            # Indicator selections
            if 'use_indicators' in config:
                self.use_indicators_check.setChecked(config['use_indicators'])
            if 'indicator_tfs' in config:
                for ind, tfs in config['indicator_tfs'].items():
                    if ind in self.indicator_checks:
                        for tf, cb in self.indicator_checks[ind].items():
                            cb.setChecked(tf in tfs)

            # Additional features
            if 'returns_enabled' in config:
                self.returns_check.setChecked(config['returns_enabled'])
            if 'returns_window' in config:
                self.returns_window.setValue(config['returns_window'])
            if 'sessions_enabled' in config:
                self.sessions_check.setChecked(config['sessions_enabled'])
            if 'session_overlap' in config:
                self.session_overlap.setValue(config['session_overlap'])
            if 'candlestick_enabled' in config:
                self.candlestick_check.setChecked(config['candlestick_enabled'])
            if 'higher_tf' in config:
                self.higher_tf_combo.setCurrentText(config['higher_tf'])
            if 'volume_profile_enabled' in config:
                self.volume_profile_check.setChecked(config['volume_profile_enabled'])
            if 'vp_bins' in config:
                self.vp_bins.setValue(config['vp_bins'])

            # Advanced parameters
            if 'warmup_bars' in config:
                self.warmup.setValue(config['warmup_bars'])
            if 'rv_window' in config:
                self.rv_w.setValue(config['rv_window'])
            if 'min_coverage' in config:
                self.min_coverage.setValue(config['min_coverage'])
            if 'atr_n' in config:
                self.atr_n.setValue(config['atr_n'])
            if 'rsi_n' in config:
                self.rsi_n.setValue(config['rsi_n'])
            if 'bb_n' in config:
                self.bb_n.setValue(config['bb_n'])
            if 'hurst_window' in config:
                self.hurst_w.setValue(config['hurst_window'])
            if 'light_epochs' in config:
                self.light_epochs.setValue(config['light_epochs'])
            if 'light_batch' in config:
                self.light_batch.setValue(config['light_batch'])
            if 'light_val_frac' in config:
                self.light_val_frac.setValue(config['light_val_frac'])
            if 'patch_len' in config:
                self.patch_len.setValue(config['patch_len'])
            if 'latent_dim' in config:
                self.latent_dim.setValue(config['latent_dim'])
            if 'encoder_epochs' in config:
                self.encoder_epochs.setValue(config['encoder_epochs'])

            # Diffusion parameters
            if 'diffusion_timesteps' in config:
                self.diffusion_timesteps.setValue(config['diffusion_timesteps'])
            if 'learning_rate' in config:
                self.learning_rate.setValue(config['learning_rate'])
            if 'batch_size_dl' in config:
                self.batch_size_dl.setValue(config['batch_size_dl'])
            if 'model_channels' in config:
                self.model_channels.setValue(config['model_channels'])
            if 'dropout' in config:
                self.dropout.setValue(config['dropout'])
            if 'num_heads' in config:
                self.num_heads.setValue(config['num_heads'])

            # Output directory
            if 'output_dir' in config:
                self.out_dir.setText(config['output_dir'])

            QMessageBox.information(self, "Load Config", f"Configurazione caricata da:\n{file_path}")
            logger.info(f"Training configuration loaded from {file_path}")

        except Exception as e:
            logger.exception(f"Failed to load config: {e}")
            QMessageBox.critical(self, "Load Config Error", f"Errore nel caricamento configurazione:\n{e}")

    def _on_save_config(self):
        """Save current training configuration to JSON file"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Training Config",
                str(Path.home() / "training_config.json"),
                "JSON Files (*.json);;All Files (*.*)"
            )
            if not file_path:
                return

            config = {
                # Metadata
                'config_version': '1.0',
                'created_at': datetime.now().isoformat(),

                # Top controls
                'model_name': self.model_name_edit.text(),
                'symbol': self.symbol_combo.currentText(),
                'timeframe': self.tf_combo.currentText(),
                'days_history': self.days_spin.value(),
                'horizon': self.horizon_spin.text(),
                'model': self.model_combo.currentText(),
                'encoder': self.encoder_combo.currentText(),
                'use_gpu_training': self.use_gpu_training_check.isChecked(),
                'optimization': self.opt_combo.currentText(),
                'gen': self.gen_spin.value(),
                'pop': self.pop_spin.value(),

                # Indicator selections
                'use_indicators': self.use_indicators_check.isChecked(),
                'indicator_tfs': {
                    ind: [tf for tf, cb in self.indicator_checks[ind].items() if cb.isChecked()]
                    for ind in INDICATORS
                },

                # Additional features
                'returns_enabled': self.returns_check.isChecked(),
                'returns_window': self.returns_window.value(),
                'sessions_enabled': self.sessions_check.isChecked(),
                'session_overlap': self.session_overlap.value(),
                'candlestick_enabled': self.candlestick_check.isChecked(),
                'higher_tf': self.higher_tf_combo.currentText(),
                'volume_profile_enabled': self.volume_profile_check.isChecked(),
                'vp_bins': self.vp_bins.value(),
                'vp_window': self.vp_window.value(),
                'use_vsa': self.vsa_check.isChecked(),
                'vsa_volume_ma': self.vsa_volume_ma.value(),
                'vsa_spread_ma': self.vsa_spread_ma.value(),

                # Advanced parameters
                'warmup_bars': self.warmup.value(),
                'rv_window': self.rv_w.value(),
                'min_coverage': self.min_coverage.value(),
                'atr_n': self.atr_n.value(),
                'rsi_n': self.rsi_n.value(),
                'bb_n': self.bb_n.value(),
                'hurst_window': self.hurst_w.value(),
                'light_epochs': self.light_epochs.value(),
                'light_batch': self.light_batch.value(),
                'light_val_frac': self.light_val_frac.value(),
                'patch_len': self.patch_len.value(),
                'latent_dim': self.latent_dim.value(),
                'encoder_epochs': self.encoder_epochs.value(),

                # Diffusion parameters
                'diffusion_timesteps': self.diffusion_timesteps.value(),
                'learning_rate': self.learning_rate.value(),
                'batch_size_dl': self.batch_size_dl.value(),
                'model_channels': self.model_channels.value(),
                'dropout': self.dropout.value(),
                'num_heads': self.num_heads.value(),

                # Output directory
                'output_dir': self.out_dir.text(),
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            QMessageBox.information(self, "Save Config", f"Configurazione salvata in:\n{file_path}")
            logger.info(f"Training configuration saved to {file_path}")

        except Exception as e:
            logger.exception(f"Failed to save config: {e}")
            QMessageBox.critical(self, "Save Config Error", f"Errore nel salvataggio configurazione:\n{e}")

    def _start_multi_horizon_validation(self):
        """Start multi-horizon validation on trained model"""
        try:
            from PySide6.QtWidgets import QInputDialog

            # Ask user to select checkpoint file
            checkpoint_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Model Checkpoint",
                str(Path(self.out_dir.text()) / "lightning"),
                "Checkpoint Files (*.ckpt *.pt *.pth);;All Files (*.*)"
            )

            if not checkpoint_path:
                return

            checkpoint_path = Path(checkpoint_path)

            # Get validation parameters from current UI settings
            sym = self.symbol_combo.currentText()
            tf = self.tf_combo.currentText()
            days = int(self.days_spin.value())

            # Ask for horizons to test
            horizons_text, ok = QInputDialog.getText(
                self,
                "Multi-Horizon Validation",
                "Enter forecast horizons to test (comma-separated, in bars):",
                text="1,4,12,24,48"
            )

            if not ok or not horizons_text:
                return

            try:
                horizons = [int(h.strip()) for h in horizons_text.split(',')]
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid comma-separated integers.")
                return

            # Clear log
            self.log_view.clear()
            self._append_log(f"[validation] Starting multi-horizon validation")
            self._append_log(f"[validation] Checkpoint: {checkpoint_path.name}")
            self._append_log(f"[validation] Horizons: {horizons}")
            self._append_log(f"[validation] Symbol: {sym}, Timeframe: {tf}, Days: {days}")

            # Run validation in background thread
            import threading
            from ..validation import validate_model_across_horizons

            def run_validation():
                try:
                    self._append_log("[validation] Loading model...")

                    # Determine device
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._append_log(f"[validation] Using device: {device}")

                    # Run validation
                    results = validate_model_across_horizons(
                        checkpoint_path=checkpoint_path,
                        symbol=sym,
                        timeframe=tf,
                        days_history=days,
                        horizons=horizons,
                        device=device
                    )

                    # Display results
                    self._append_log("\n" + "=" * 60)
                    self._append_log("Multi-Horizon Validation Results")
                    self._append_log("=" * 60)

                    for horizon, result in results.items():
                        self._append_log(f"\nHorizon {horizon}h:")
                        self._append_log(f"  MAE: {result.mae:.6f}")
                        self._append_log(f"  RMSE: {result.rmse:.6f}")
                        self._append_log(f"  MAPE: {result.mape:.2f}%")
                        self._append_log(f"  Directional Accuracy: {result.directional_accuracy:.1f}%")
                        self._append_log(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
                        self._append_log(f"  Max Drawdown: {result.max_drawdown:.1f}%")
                        self._append_log(f"  Coverage (95%): {result.coverage_95:.3f}")
                        self._append_log(f"  Interval Width: {result.interval_width:.6f}")
                        self._append_log(f"  Samples: {result.n_samples}")

                    self._append_log("\n" + "=" * 60)

                    # Export results to CSV
                    output_csv = checkpoint_path.parent / f"validation_{checkpoint_path.stem}_horizons.csv"

                    import pandas as pd
                    rows = []
                    for horizon, result in results.items():
                        rows.append({
                            'horizon': horizon,
                            'mae': result.mae,
                            'rmse': result.rmse,
                            'mape': result.mape,
                            'directional_accuracy': result.directional_accuracy,
                            'sharpe_ratio': result.sharpe_ratio,
                            'max_drawdown': result.max_drawdown,
                            'coverage_95': result.coverage_95,
                            'interval_width': result.interval_width,
                            'n_samples': result.n_samples
                        })

                    df = pd.DataFrame(rows)
                    df.to_csv(output_csv, index=False)
                    self._append_log(f"\n[validation] Results exported to: {output_csv}")

                    # Show completion message
                    QMessageBox.information(
                        self,
                        "Validation Complete",
                        f"Multi-horizon validation completed successfully.\n\nResults saved to:\n{output_csv}"
                    )

                except Exception as e:
                    logger.exception(f"Validation failed: {e}")
                    self._append_log(f"\n[validation] ERROR: {e}")
                    QMessageBox.critical(
                        self,
                        "Validation Error",
                        f"Validation failed:\n{e}"
                    )

            # Start validation thread
            thread = threading.Thread(target=run_validation, daemon=True)
            thread.start()

        except Exception as e:
            logger.exception(f"Failed to start validation: {e}")

    def _open_grid_training(self):
        """Open Grid Training Manager dialog"""
        try:
            from .training_queue_tab import TrainingQueueTab
            from .regime_analysis_tab import RegimeAnalysisTab
            from .training_history_tab import TrainingHistoryTab
            from PySide6.QtWidgets import QDialog, QTabWidget, QVBoxLayout

            # Create dialog with tabs
            dialog = QDialog(self)
            dialog.setWindowTitle("Grid Training Manager")
            dialog.resize(1200, 800)

            layout = QVBoxLayout(dialog)

            # Create tab widget
            tabs = QTabWidget()

            # Add tabs
            queue_tab = TrainingQueueTab(dialog)
            regime_tab = RegimeAnalysisTab(dialog)
            history_tab = TrainingHistoryTab(dialog)

            tabs.addTab(queue_tab, "Training Queue")
            tabs.addTab(regime_tab, "Regime Analysis")
            tabs.addTab(history_tab, "Training History")

            layout.addWidget(tabs)

            dialog.exec()

        except Exception as e:
            logger.exception(f"Failed to open Grid Training Manager: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open Grid Training Manager:\n{e}"
            )
