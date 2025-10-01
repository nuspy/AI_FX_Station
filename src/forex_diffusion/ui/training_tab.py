# src/forex_diffusion/ui/training_tab.py
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QLineEdit, QGroupBox, QGridLayout, QMessageBox, QFileDialog, QTextEdit, QProgressBar,
    QCheckBox, QScrollArea
)
from loguru import logger

from ..utils.config import get_config
from ..utils.user_settings import get_setting, set_setting
from .controllers import TrainingController

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

        # Log & progress
        self._build_log_section()

        # Actions
        self._build_actions()

        # Load saved settings
        self._load_settings()

    def _save_settings(self):
        """Save all training settings to persistent storage"""
        try:
            settings = {
                # Top controls
                'symbol': self.symbol_combo.currentText(),
                'timeframe': self.tf_combo.currentText(),
                'days_history': self.days_spin.value(),
                'horizon': self.horizon_spin.value(),
                'model': self.model_combo.currentText(),
                'encoder': self.encoder_combo.currentText(),
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
            if 'symbol' in settings:
                self.symbol_combo.setCurrentText(settings['symbol'])
            if 'timeframe' in settings:
                self.tf_combo.setCurrentText(settings['timeframe'])
            if 'days_history' in settings:
                self.days_spin.setValue(settings['days_history'])
            if 'horizon' in settings:
                self.horizon_spin.setValue(settings['horizon'])
            if 'model' in settings:
                self.model_combo.setCurrentText(settings['model'])
            if 'encoder' in settings:
                self.encoder_combo.setCurrentText(settings['encoder'])
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

    def _build_top_controls(self):
        """Build top control row: symbol, timeframe, days, horizon, model, encoder, optimization"""
        top = QHBoxLayout()

        # Symbol
        lbl_sym = QLabel("Symbol:")
        lbl_sym.setToolTip("Coppia valutaria da usare per il training.")
        top.addWidget(lbl_sym)
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["EUR/USD", "GBP/USD", "AUX/USD", "GBP/NZD", "AUD/JPY", "GBP/EUR", "GBP/AUD"])
        self.symbol_combo.setToolTip("Seleziona il simbolo per cui addestrare il modello.")
        top.addWidget(self.symbol_combo)

        # Base timeframe
        lbl_tf = QLabel("Base TF:")
        lbl_tf.setToolTip("Timeframe base della serie su cui costruire le feature.")
        top.addWidget(lbl_tf)
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        self.tf_combo.setCurrentText("1m")
        self.tf_combo.setToolTip("Timeframe delle barre usate per il training.")
        top.addWidget(self.tf_combo)

        # Days history
        lbl_days = QLabel("Days:")
        lbl_days.setToolTip("Numero di giorni storici da usare per l'addestramento.")
        top.addWidget(lbl_days)
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 3650)
        self.days_spin.setValue(7)
        self.days_spin.setToolTip("Maggiore è il valore, più dati verranno usati (training più lungo).")
        top.addWidget(self.days_spin)

        # Horizon
        lbl_h = QLabel("Horizon:")
        lbl_h.setToolTip("Orizzonte in barre da prevedere durante il training.")
        top.addWidget(lbl_h)
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 500)
        self.horizon_spin.setValue(5)
        self.horizon_spin.setToolTip("Numero di passi futuri/target usati in fase di training.")
        top.addWidget(self.horizon_spin)

        # Model
        lbl_m = QLabel("Model:")
        lbl_m.setToolTip("Tipo di modello supervisato da addestrare.")
        top.addWidget(lbl_m)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["ridge", "lasso", "elasticnet", "rf", "lightning"])
        self.model_combo.setToolTip("Scegli l'algoritmo di base (regressione lineare regolarizzata o Random Forest).")
        top.addWidget(self.model_combo)

        # Encoder
        lbl_e = QLabel("Encoder:")
        lbl_e.setToolTip("Preprocessore/encoder opzionale per ridurre dimensionalità.")
        top.addWidget(lbl_e)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(["none", "pca", "autoencoder", "vae", "latents"])
        self.encoder_combo.setToolTip(
            "none: nessun encoder\n"
            "pca: PCA (Principal Component Analysis)\n"
            "autoencoder: Neural autoencoder (richiede PyTorch)\n"
            "vae: Variational Autoencoder (richiede PyTorch)\n"
            "latents: usa encoder pre-addestrato"
        )
        top.addWidget(self.encoder_combo)

        # Optimization
        lbl_opt = QLabel("Opt:")
        lbl_opt.setToolTip("Ricerca automatica dell'ipermodello.")
        top.addWidget(lbl_opt)
        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["none", "genetic-basic", "nsga2"])
        self.opt_combo.setToolTip("Seleziona se e come ottimizzare automaticamente gli iperparametri.")
        top.addWidget(self.opt_combo)

        # Gen
        lbl_gen = QLabel("Gen:")
        lbl_gen.setToolTip("Numero di generazioni dell'algoritmo genetico.")
        top.addWidget(lbl_gen)
        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(1, 50)
        self.gen_spin.setValue(5)
        self.gen_spin.setToolTip("Quante iterazioni evolutive eseguire.")
        top.addWidget(self.gen_spin)

        # Pop
        lbl_pop = QLabel("Pop:")
        lbl_pop.setToolTip("Dimensione della popolazione per la ricerca evolutiva.")
        top.addWidget(lbl_pop)
        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(2, 64)
        self.pop_spin.setValue(8)
        self.pop_spin.setToolTip("Quanti candidati valutare per generazione.")
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

        self.layout.addWidget(adv_box)

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

        actions.addWidget(self.train_btn)
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

    def _start_training(self):
        """Start training process"""
        try:
            # Save current settings before training
            self._save_settings()

            sym = self.symbol_combo.currentText()
            tf = self.tf_combo.currentText()
            days = int(self.days_spin.value())
            horizon = int(self.horizon_spin.value())
            model = self.model_combo.currentText()
            encoder = self.encoder_combo.currentText()

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
            if strategy != 'none' and model != 'lightning':
                self._append_log(f"[warn] Optimization '{strategy}' non implementata nel trainer sklearn; eseguo una singola fit.")

            from datetime import datetime, timezone

            # Additional features config
            additional_features = {
                'returns': self.returns_check.isChecked(),
                'sessions': self.sessions_check.isChecked(),
                'candlestick': self.candlestick_check.isChecked(),
                'volume_profile': self.volume_profile_check.isChecked(),
            }

            if model == 'lightning':
                module = 'src.forex_diffusion.training.train'
                args = [
                    sys.executable, '-m', module,
                    '--symbol', sym,
                    '--timeframe', tf,
                    '--horizon', str(horizon),
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
                ]
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
                    '--horizon', str(horizon),
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
                    '--random_state', '0',
                    '--n_estimators', '400',
                ]
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
                    },
                    'optimization': strategy,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'ui_run_name': name,
                }
                pending_dir = artifacts_dir / 'models'

            self._pending_meta = meta
            self._pending_out_dir = pending_dir
            self._append_log(f"[meta] prepared: {meta}")

            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.controller.start_training(args, cwd=str(root))
            self._append_log(f"[start] {' '.join(args)}")

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
