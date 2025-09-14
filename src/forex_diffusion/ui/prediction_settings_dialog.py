"""
Dialog for configuring prediction settings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QFileDialog,
    QDialogButtonBox, QSpinBox, QCheckBox, QHBoxLayout, QLabel
)
from loguru import logger

CONFIG_FILE = Path(__file__).resolve().parents[3] / "configs" / "prediction_settings.json"

class PredictionSettingsDialog(QDialog):
    """
    A dialog window for setting prediction parameters (basic + advanced),
    which are persisted to a JSON file.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prediction Settings")
        self.setMinimumWidth(520)
        # make window compact (scrollable content)
        try:
            self.resize(720, 600)
        except Exception:
            pass

        # Root layout + scroll area with content
        from PySide6.QtWidgets import QScrollArea, QWidget
        self._root_layout = QVBoxLayout(self)
        content = QWidget(self)
        self.layout = QVBoxLayout(content)
        self.form_layout = QFormLayout()

        # Model Path (singolo, per compatibilità)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setToolTip("Percorso del file modello da usare per l'inferenza.\nSupporto tipico: PyTorch (.pt/.pth) o pickle di modelli sklearn.\nSe è valorizzato 'Modelli multipli', questo campo è ignorato.")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setToolTip("Sfoglia e seleziona un file modello da disco.")
        self.browse_button.clicked.connect(self._browse_model_path)
        model_h = QHBoxLayout()
        model_h.addWidget(self.model_path_edit)
        model_h.addWidget(self.browse_button)
        self.form_layout.addRow("Model Path:", model_h)

        # Modelli multipli: uno per riga (opzionale, ha precedenza su Model Path)
        from PySide6.QtWidgets import QTextEdit, QGroupBox, QVBoxLayout as QV
        models_box = QGroupBox("Modelli multipli (uno per riga)")
        models_box.setToolTip("Inserisci uno o più percorsi di modelli (uno per riga). Verrà lanciata una previsione per ciascun modello selezionato.\nSe compilato, ha precedenza su 'Model Path'.")
        box_lay = QV(models_box)
        self.models_edit = QTextEdit()
        self.models_edit.setPlaceholderText("Percorso modello per riga (opzionale). Se valorizzato, verranno eseguite previsioni per ciascun modello.")
        self.models_edit.setToolTip("Ogni riga deve contenere un percorso file valido a un modello. I modelli verranno eseguiti in parallelo.")
        box_lay.addWidget(self.models_edit)
        self.layout.addWidget(models_box)

        # Tipi di previsione (selezione multipla)
        from PySide6.QtWidgets import QCheckBox
        types_box = QGroupBox("Tipi di previsione")
        types_box.setToolTip("Seleziona il tipo di previsione:\n- Basic: pipeline standard con indicatori e standardizzazione.\n- Advanced: come Basic, ma abilita opzioni/feature aggiuntive.\n- Baseline RW: baseline Random Walk/zero-drift (nessun modello richiesto).")
        types_lay = QV(types_box)
        self.type_basic_cb = QCheckBox("Basic"); self.type_basic_cb.setChecked(True); self.type_basic_cb.setToolTip("Basic: usa la pipeline standard e il modello selezionato.")
        self.type_advanced_cb = QCheckBox("Advanced"); self.type_advanced_cb.setToolTip("Advanced: come Basic con opzioni extra (EMA, Hurst, Donchian, Keltner, ecc.).")
        self.type_rw_cb = QCheckBox("Baseline RW"); self.type_rw_cb.setToolTip("Baseline RW: previsione di riferimento a drift nullo. Non richiede un modello.")
        types_lay.addWidget(self.type_basic_cb)
        types_lay.addWidget(self.type_advanced_cb)
        types_lay.addWidget(self.type_rw_cb)
        self.layout.addWidget(types_box)

        # Horizons
        self.horizons_edit = QLineEdit("1m, 5m, 15m")
        self.horizons_edit.setToolTip("Orizzonti temporali di previsione, separati da virgola (es.: 1m, 5m, 15m).\nVengono convertiti in passi rispetto al timeframe corrente.")
        self.form_layout.addRow("Horizons (comma-separated):", self.horizons_edit)

        # N_samples
        self.n_samples_spinbox = QSpinBox()
        self.n_samples_spinbox.setRange(1, 10000)
        self.n_samples_spinbox.setValue(200)
        self.n_samples_spinbox.setToolTip("Numero di campioni/forward-pass per stimare i quantili.\nValori più alti aumentano stabilità ma richiedono più tempo.")
        self.form_layout.addRow("Number of Samples (N_samples):", self.n_samples_spinbox)

        # Conformal Calibration
        self.conformal_checkbox = QCheckBox("Apply Conformal Calibration")
        self.conformal_checkbox.setChecked(True)
        self.conformal_checkbox.setToolTip("Applica calibrazione conformale per intervalli predittivi affidabili.\nSe attiva, i quantili (q05, q95) vengono corretti in base a una stima di errore fuori campione.")
        self.form_layout.addRow(self.conformal_checkbox)

        # Model weight (inference scaling)
        from PySide6.QtWidgets import QComboBox
        self.model_weight_combo = QComboBox()
        for p in range(0, 101, 5):
            self.model_weight_combo.addItem(f"{p} %", p)
        self.model_weight_combo.setCurrentIndex(20)  # default 100%
        self.model_weight_combo.setToolTip("Peso con cui fondere la previsione col prezzo attuale:\n0% = ignora modello (resta ultimo close); 100% = usa la previsione al 100%.")
        self.form_layout.addRow("Model weight (%):", self.model_weight_combo)

        # Indicators × Timeframes selection
        from PySide6.QtWidgets import QGroupBox, QGridLayout, QCheckBox
        self._indicators = ["ATR","RSI","Bollinger","MACD","Donchian","Keltner","Hurst"]
        self._timeframes = ["1m","5m","15m","30m","1h","4h","1d"]
        box = QGroupBox("Indicatori per Timeframe (per training/inferenza)")
        box.setToolTip("Seleziona quali indicatori includere per ciascun timeframe nella pipeline.\nDurante il training/inf. questi flag controllano quali feature vengono calcolate.")
        grid = QGridLayout(box)
        grid.addWidget(QLabel(""), 0, 0)
        for j, tf in enumerate(self._timeframes, start=1):
            grid.addWidget(QLabel(tf), 0, j)
        self.indicator_checks = {}
        for i, ind in enumerate(self._indicators, start=1):
            grid.addWidget(QLabel(ind), i, 0)
            self.indicator_checks[ind] = {}
            for j, tf in enumerate(self._timeframes, start=1):
                cb = QCheckBox()
                self.indicator_checks[ind][tf] = cb
                grid.addWidget(cb, i, j)
        self.layout.addWidget(box)

        # --- Basic indicators (exposed to basic dialog) ---
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(1, 500)
        self.warmup_spin.setValue(16)
        self.warmup_spin.setToolTip("Numero di barre iniziali da scartare/riscaldare per gli indicatori.\nValori maggiori stabilizzano le feature all'inizio della serie.")
        self.form_layout.addRow("Warmup Bars:", self.warmup_spin)

        self.atr_n_spin = QSpinBox()
        self.atr_n_spin.setRange(1, 200)
        self.atr_n_spin.setValue(14)
        self.atr_n_spin.setToolTip("Lunghezza media per l'Average True Range (misura di volatilità).")
        self.form_layout.addRow("ATR n:", self.atr_n_spin)

        self.rsi_n_spin = QSpinBox()
        self.rsi_n_spin.setRange(2, 200)
        self.rsi_n_spin.setValue(14)
        self.rsi_n_spin.setToolTip("Numero di periodi per il Relative Strength Index (momento).")
        self.form_layout.addRow("RSI n:", self.rsi_n_spin)

        self.bb_n_spin = QSpinBox()
        self.bb_n_spin.setRange(2, 200)
        self.bb_n_spin.setValue(20)
        self.bb_n_spin.setToolTip("Finestra per le Bande di Bollinger (deviazioni standard su media mobile).")
        self.form_layout.addRow("Bollinger window n:", self.bb_n_spin)

        self.rv_window_spin = QSpinBox()
        self.rv_window_spin.setRange(1, 1000)
        self.rv_window_spin.setValue(60)
        self.rv_window_spin.setToolTip("Finestra per stimare la Realized Volatility/standardizzazione.\nControlla la scala delle feature in pipeline.")
        self.form_layout.addRow("RV window:", self.rv_window_spin)

        # --- Advanced indicators ---
        self.ema_fast_spin = QSpinBox()
        self.ema_fast_spin.setRange(1, 200)
        self.ema_fast_spin.setValue(12)
        self.ema_fast_spin.setToolTip("Span della EMA veloce per trend/momentum.")
        self.form_layout.addRow("EMA fast span:", self.ema_fast_spin)

        self.ema_slow_spin = QSpinBox()
        self.ema_slow_spin.setRange(1, 400)
        self.ema_slow_spin.setValue(26)
        self.ema_slow_spin.setToolTip("Span della EMA lenta per trend/momentum.")
        self.form_layout.addRow("EMA slow span:", self.ema_slow_spin)

        self.don_n_spin = QSpinBox()
        self.don_n_spin.setRange(1, 400)
        self.don_n_spin.setValue(20)
        self.don_n_spin.setToolTip("Finestra per il canale di Donchian (massimi/minimi su n barre).")
        self.form_layout.addRow("Donchian n:", self.don_n_spin)

        self.hurst_window_spin = QSpinBox()
        self.hurst_window_spin.setRange(1, 1024)
        self.hurst_window_spin.setValue(64)
        self.hurst_window_spin.setToolTip("Window per la stima dell'esponente di Hurst (mean-reversion vs. trending).")
        self.form_layout.addRow("Hurst window:", self.hurst_window_spin)

        self.keltner_k_spin = QSpinBox()
        self.keltner_k_spin.setRange(1, 10)
        self.keltner_k_spin.setValue(1)
        self.keltner_k_spin.setToolTip("Moltiplicatore per le Keltner Channels (ampiezza del canale rispetto all'ATR).")
        self.form_layout.addRow("Keltner multiplier (k):", self.keltner_k_spin)

        # max forecasts
        self.max_forecasts_spin = QSpinBox()
        self.max_forecasts_spin.setRange(1, 100)
        self.max_forecasts_spin.setValue(5)
        self.form_layout.addRow("Max forecasts to display:", self.max_forecasts_spin)

        # Test history bars (for TestingPoint forecasts)
        self.test_history_spin = QSpinBox()
        self.test_history_spin.setRange(1, 10000)
        self.test_history_spin.setValue(128)
        self.form_layout.addRow("Test history bars (N):", self.test_history_spin)

        # auto predict options
        auto_h = QHBoxLayout()
        self.auto_checkbox = QCheckBox("Auto predict")
        self.auto_interval_spin = QSpinBox()
        self.auto_interval_spin.setRange(1, 86400)
        self.auto_interval_spin.setValue(60)
        auto_h.addWidget(self.auto_checkbox)
        auto_h.addWidget(QLabel("Interval (s):"))
        auto_h.addWidget(self.auto_interval_spin)
        self.form_layout.addRow(auto_h)

        self.layout.addLayout(self.form_layout)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Install scroll area
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        # 'content' è il QWidget su cui abbiamo costruito self.layout
        try:
            content = self.layout.parentWidget()
            scroll.setWidget(content)
        except Exception:
            scroll.setWidget(self)
        self._root_layout.addWidget(scroll)
        self._root_layout.addWidget(self.button_box)

        self.load_settings()

    def _browse_model_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Models (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def load_settings(self):
        """Loads settings from the JSON config file."""
        if not CONFIG_FILE.exists():
            logger.debug(f"Prediction settings file not found: {CONFIG_FILE}")
            return
        try:
            with open(CONFIG_FILE, "r") as f:
                settings = json.load(f)
            self.model_path_edit.setText(settings.get("model_path", ""))
            # multi models
            mpaths = settings.get("model_paths", [])
            try:
                self.models_edit.setPlainText("\n".join([str(p) for p in mpaths]) if mpaths else "")
            except Exception:
                self.models_edit.setPlainText("")
            # forecast types
            ftypes = set(settings.get("forecast_types", ["basic"]))
            self.type_basic_cb.setChecked("basic" in ftypes)
            self.type_advanced_cb.setChecked("advanced" in ftypes)
            self.type_rw_cb.setChecked("rw" in ftypes)

            self.horizons_edit.setText(", ".join(settings.get("horizons", ["1m", "5m", "15m"])))
            self.n_samples_spinbox.setValue(settings.get("N_samples", 200))
            self.conformal_checkbox.setChecked(settings.get("apply_conformal", True))
            # model weight
            mw = int(settings.get("model_weight_pct", 100))
            for i in range(self.model_weight_combo.count()):
                if int(self.model_weight_combo.itemData(i)) == mw:
                    self.model_weight_combo.setCurrentIndex(i); break

            # indicators×timeframes
            ind_tfs = settings.get("indicator_tfs", {})
            for ind, tfmap in self.indicator_checks.items():
                lst = ind_tfs.get(ind.lower(), [])
                for tf, cb in tfmap.items():
                    cb.setChecked(tf in lst)

            self.warmup_spin.setValue(int(settings.get("warmup_bars", 16)))
            self.atr_n_spin.setValue(int(settings.get("atr_n", 14)))
            self.rsi_n_spin.setValue(int(settings.get("rsi_n", 14)))
            self.bb_n_spin.setValue(int(settings.get("bb_n", 20)))
            self.rv_window_spin.setValue(int(settings.get("rv_window", 60)))

            self.ema_fast_spin.setValue(int(settings.get("ema_fast", 12)))
            self.ema_slow_spin.setValue(int(settings.get("ema_slow", 26)))
            self.don_n_spin.setValue(int(settings.get("don_n", 20)))
            self.hurst_window_spin.setValue(int(settings.get("hurst_window", 64)))
            self.keltner_k_spin.setValue(int(settings.get("keltner_k", 1)))

            self.max_forecasts_spin.setValue(int(settings.get("max_forecasts", 5)))
            self.test_history_spin.setValue(int(settings.get("test_history_bars", 128)))
            self.auto_checkbox.setChecked(bool(settings.get("auto_predict", False)))
            self.auto_interval_spin.setValue(int(settings.get("auto_interval_seconds", 60)))

            logger.info(f"Loaded prediction settings from {CONFIG_FILE}")
        except Exception as e:
            logger.exception(f"Failed to load prediction settings: {e}")

    def save_settings(self):
        """Saves the current settings to the JSON config file."""
        # parse multi models (one path per line)
        models_list = []
        try:
            txt = self.models_edit.toPlainText().strip()
            if txt:
                models_list = [line.strip() for line in txt.splitlines() if line.strip()]
        except Exception:
            models_list = []

        # forecast types
        ftypes = []
        if self.type_basic_cb.isChecked(): ftypes.append("basic")
        if self.type_advanced_cb.isChecked(): ftypes.append("advanced")
        if self.type_rw_cb.isChecked(): ftypes.append("rw")
        if not ftypes:
            ftypes = ["basic"]

        settings = {
            "model_path": self.model_path_edit.text(),
            "model_paths": models_list,
            "forecast_types": ftypes,
            "horizons": [h.strip() for h in self.horizons_edit.text().split(",") if h.strip()],
            "N_samples": self.n_samples_spinbox.value(),
            "apply_conformal": self.conformal_checkbox.isChecked(),
            "model_weight_pct": int(self.model_weight_combo.currentData()),

            # indicators×timeframes
            "indicator_tfs": {ind.lower(): [tf for tf, cb in tfmap.items() if cb.isChecked()] for ind, tfmap in self.indicator_checks.items()},

            "warmup_bars": int(self.warmup_spin.value()),
            "atr_n": int(self.atr_n_spin.value()),
            "rsi_n": int(self.rsi_n_spin.value()),
            "bb_n": int(self.bb_n_spin.value()),
            "rv_window": int(self.rv_window_spin.value()),

            "ema_fast": int(self.ema_fast_spin.value()),
            "ema_slow": int(self.ema_slow_spin.value()),
            "don_n": int(self.don_n_spin.value()),
            "hurst_window": int(self.hurst_window_spin.value()),
            "keltner_k": int(self.keltner_k_spin.value()),

            "max_forecasts": int(self.max_forecasts_spin.value()),
            "test_history_bars": int(self.test_history_spin.value()),
            "auto_predict": bool(self.auto_checkbox.isChecked()),
            "auto_interval_seconds": int(self.auto_interval_spin.value()),
        }
        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(settings, f, indent=4)
            logger.info(f"Saved prediction settings to {CONFIG_FILE}")
        except Exception as e:
            logger.exception(f"Failed to save prediction settings: {e}")

    def accept(self):
        """Saves settings and closes the dialog."""
        self.save_settings()
        super().accept()

    @staticmethod
    def get_settings() -> Dict[str, Any]:
        """Static method to retrieve the latest saved settings."""
        if not CONFIG_FILE.exists():
            return {}
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
