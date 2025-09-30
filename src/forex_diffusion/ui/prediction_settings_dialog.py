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
            # Load geometry first (will be overridden by load_settings if exists)
            self.resize(720, 600)
        except Exception:
            pass

        # Root layout + scroll area with content
        from PySide6.QtWidgets import QScrollArea, QWidget
        self._root_layout = QVBoxLayout(self)
        content = QWidget(self)
        self.layout = QVBoxLayout(content)
        self.form_layout = QFormLayout()

        # Streamlined Model Selection
        from PySide6.QtWidgets import QTextEdit, QGroupBox, QVBoxLayout as QV
        models_box = QGroupBox("Model Selection")
        box_lay = QV(models_box)

        # Primary method: text area for multiple models
        self.models_edit = QTextEdit()
        self.models_edit.setPlaceholderText("Model paths (one per line)")
        self.models_edit.setMinimumHeight(80)
        box_lay.addWidget(self.models_edit)

        # Secondary method: file browser
        model_h = QHBoxLayout()
        self.browse_multi_button = QPushButton("Browse Models")
        self.browse_multi_button.clicked.connect(self._browse_model_paths_multi)
        self.info_button = QPushButton("Model Info")
        self.info_button.clicked.connect(self._show_model_info)
        self.loadmeta_button = QPushButton("Load Defaults")
        self.loadmeta_button.clicked.connect(self._load_model_defaults)
        model_h.addWidget(self.browse_multi_button)
        model_h.addWidget(self.info_button)
        model_h.addWidget(self.loadmeta_button)
        box_lay.addLayout(model_h)

        self.layout.addWidget(models_box)

        # Legacy single model path (hidden, maintained for compatibility)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setVisible(False)

        # Internal state for multi-selection
        try:
            if not hasattr(self.__class__, "_last_model_paths"):
                self.__class__._last_model_paths = []
            self._model_paths = list(self.__class__._last_model_paths)
        except Exception:
            self._model_paths = []

        # Simplified Forecast Types
        from PySide6.QtWidgets import QCheckBox
        types_box = QGroupBox("Forecast Types")
        types_lay = QV(types_box)
        self.type_basic_cb = QCheckBox("Basic")
        self.type_basic_cb.setChecked(True)
        self.type_advanced_cb = QCheckBox("Advanced")
        self.type_rw_cb = QCheckBox("Baseline RW")
        types_lay.addWidget(self.type_basic_cb)
        types_lay.addWidget(self.type_advanced_cb)
        types_lay.addWidget(self.type_rw_cb)
        self.layout.addWidget(types_box)

        # Core Prediction Settings
        core_box = QGroupBox("Core Settings")
        core_lay = QFormLayout(core_box)

        self.horizons_edit = QLineEdit("1m, 5m, 15m")
        core_lay.addRow("Horizons:", self.horizons_edit)

        self.n_samples_spinbox = QSpinBox()
        self.n_samples_spinbox.setRange(1, 10000)
        self.n_samples_spinbox.setValue(200)
        core_lay.addRow("N Samples:", self.n_samples_spinbox)

        self.conformal_checkbox = QCheckBox("Apply Conformal Calibration")
        self.conformal_checkbox.setChecked(True)
        core_lay.addRow(self.conformal_checkbox)

        from PySide6.QtWidgets import QComboBox
        self.model_weight_combo = QComboBox()
        for p in range(0, 101, 25):  # Simplified: 0%, 25%, 50%, 75%, 100%
            self.model_weight_combo.addItem(f"{p}%", p)
        self.model_weight_combo.setCurrentIndex(4)  # default 100%
        core_lay.addRow("Model Weight:", self.model_weight_combo)

        self.layout.addWidget(core_box)

        # Simplified Indicators
        indicators_box = QGroupBox("Indicators")
        indicators_lay = QVBoxLayout(indicators_box)

        # Common timeframes for quick selection
        tf_row = QHBoxLayout()
        tf_row.addWidget(QLabel("Primary Timeframes:"))
        self.tf_1m_cb = QCheckBox("1m")
        self.tf_5m_cb = QCheckBox("5m")
        self.tf_15m_cb = QCheckBox("15m")
        self.tf_1h_cb = QCheckBox("1h")
        self.tf_1h_cb.setChecked(True)  # Default
        tf_row.addWidget(self.tf_1m_cb)
        tf_row.addWidget(self.tf_5m_cb)
        tf_row.addWidget(self.tf_15m_cb)
        tf_row.addWidget(self.tf_1h_cb)
        indicators_lay.addLayout(tf_row)

        # Standard indicators (enabled by default)
        std_row = QHBoxLayout()
        std_row.addWidget(QLabel("Standard:"))
        self.atr_cb = QCheckBox("ATR")
        self.atr_cb.setChecked(True)
        self.rsi_cb = QCheckBox("RSI")
        self.rsi_cb.setChecked(True)
        self.bb_cb = QCheckBox("Bollinger")
        self.bb_cb.setChecked(True)
        self.macd_cb = QCheckBox("MACD")
        std_row.addWidget(self.atr_cb)
        std_row.addWidget(self.rsi_cb)
        std_row.addWidget(self.bb_cb)
        std_row.addWidget(self.macd_cb)
        indicators_lay.addLayout(std_row)

        # Advanced indicators
        adv_row = QHBoxLayout()
        adv_row.addWidget(QLabel("Advanced:"))
        self.don_cb = QCheckBox("Donchian")
        self.keltner_cb = QCheckBox("Keltner")
        self.hurst_cb = QCheckBox("Hurst")
        adv_row.addWidget(self.don_cb)
        adv_row.addWidget(self.keltner_cb)
        adv_row.addWidget(self.hurst_cb)
        indicators_lay.addLayout(adv_row)

        self.layout.addWidget(indicators_box)

        # Legacy indicator_checks for compatibility
        self._indicators = ["ATR","RSI","Bollinger","MACD","Donchian","Keltner","Hurst"]
        self._timeframes = ["1m","5m","15m","30m","1h","4h","1d"]
        self.indicator_checks = {ind: {tf: QCheckBox() for tf in self._timeframes} for ind in self._indicators}

        # Technical Parameters
        tech_box = QGroupBox("Technical Parameters")
        tech_lay = QFormLayout(tech_box)

        # Essential parameters
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(1, 500)
        self.warmup_spin.setValue(16)
        tech_lay.addRow("Warmup Bars:", self.warmup_spin)

        self.rv_window_spin = QSpinBox()
        self.rv_window_spin.setRange(1, 1000)
        self.rv_window_spin.setValue(60)
        tech_lay.addRow("RV Window:", self.rv_window_spin)

        # Standard indicator periods (in compact grid)
        periods_h = QHBoxLayout()
        periods_h.addWidget(QLabel("Periods - ATR:"))
        self.atr_n_spin = QSpinBox()
        self.atr_n_spin.setRange(1, 200)
        self.atr_n_spin.setValue(14)
        periods_h.addWidget(self.atr_n_spin)

        periods_h.addWidget(QLabel("RSI:"))
        self.rsi_n_spin = QSpinBox()
        self.rsi_n_spin.setRange(2, 200)
        self.rsi_n_spin.setValue(14)
        periods_h.addWidget(self.rsi_n_spin)

        periods_h.addWidget(QLabel("BB:"))
        self.bb_n_spin = QSpinBox()
        self.bb_n_spin.setRange(2, 200)
        self.bb_n_spin.setValue(20)
        periods_h.addWidget(self.bb_n_spin)
        tech_lay.addRow(periods_h)

        # Advanced parameters (compact)
        adv_h = QHBoxLayout()
        adv_h.addWidget(QLabel("EMA:"))
        self.ema_fast_spin = QSpinBox()
        self.ema_fast_spin.setRange(1, 200)
        self.ema_fast_spin.setValue(12)
        adv_h.addWidget(self.ema_fast_spin)
        adv_h.addWidget(QLabel("/"))
        self.ema_slow_spin = QSpinBox()
        self.ema_slow_spin.setRange(1, 400)
        self.ema_slow_spin.setValue(26)
        adv_h.addWidget(self.ema_slow_spin)

        adv_h.addWidget(QLabel(" Don:"))
        self.don_n_spin = QSpinBox()
        self.don_n_spin.setRange(1, 400)
        self.don_n_spin.setValue(20)
        adv_h.addWidget(self.don_n_spin)

        adv_h.addWidget(QLabel(" Hurst:"))
        self.hurst_window_spin = QSpinBox()
        self.hurst_window_spin.setRange(1, 1024)
        self.hurst_window_spin.setValue(64)
        adv_h.addWidget(self.hurst_window_spin)

        adv_h.addWidget(QLabel(" Kelt:"))
        self.keltner_k_spin = QSpinBox()
        self.keltner_k_spin.setRange(1, 10)
        self.keltner_k_spin.setValue(1)
        adv_h.addWidget(self.keltner_k_spin)
        tech_lay.addRow(adv_h)

        self.layout.addWidget(tech_box)

        # ========== MULTI-TIMEFRAME HIERARCHICAL STRATEGY ==========
        mtf_box = QGroupBox("Multi-Timeframe Hierarchical Strategy")
        mtf_box.setToolTip("Sistema gerarchico dove ogni candela ha riferimento alla candela madre.\nImplementa la strategia multi-timeframe con selezione automatica del gruppo di candele.")
        mtf_lay = QVBoxLayout(mtf_box)

        # Enable hierarchical multi-timeframe
        self.hierarchical_cb = QCheckBox("Enable Hierarchical Multi-Timeframe")
        self.hierarchical_cb.setToolTip("Attiva il sistema gerarchico multi-timeframe dove ogni candela ha riferimento alla candela madre.")
        mtf_lay.addWidget(self.hierarchical_cb)

        # Query timeframe selection
        query_h = QHBoxLayout()
        query_h.addWidget(QLabel("Query Timeframe:"))
        self.query_timeframe_combo = QComboBox()
        self.query_timeframe_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        self.query_timeframe_combo.setCurrentText("5m")
        self.query_timeframe_combo.setToolTip("Timeframe di interrogazione per il modello.\nDefinisce il livello gerarchico su cui fare le predizioni.")
        query_h.addWidget(self.query_timeframe_combo)
        mtf_lay.addLayout(query_h)

        # Hierarchical timeframes
        htf_h = QHBoxLayout()
        htf_h.addWidget(QLabel("Hierarchical Timeframes:"))
        self.hierarchical_timeframes_edit = QLineEdit("1m, 5m, 15m, 1h")
        self.hierarchical_timeframes_edit.setToolTip("Timeframes da includere nella gerarchia (comma-separated).\nVengono utilizzati per costruire le relazioni parent-child.")
        htf_h.addWidget(self.hierarchical_timeframes_edit)
        mtf_lay.addLayout(htf_h)

        # Exclude children option
        self.exclude_children_cb = QCheckBox("Exclude Children from Modeling")
        self.exclude_children_cb.setChecked(True)
        self.exclude_children_cb.setToolTip("Esclude le candele child dal modeling, mantenendo solo quelle del query timeframe.\nRiduce il rumore e focalizza il modello sui pattern del timeframe selezionato.")
        mtf_lay.addWidget(self.exclude_children_cb)

        # Parallel inference options
        parallel_h = QHBoxLayout()
        self.parallel_inference_cb = QCheckBox("Use Parallel Model Inference")
        self.parallel_inference_cb.setChecked(True)
        self.parallel_inference_cb.setToolTip("Abilita l'esecuzione parallela di modelli multipli per ensemble predictions.")
        parallel_h.addWidget(self.parallel_inference_cb)

        parallel_h.addWidget(QLabel("Max Workers:"))
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 16)
        self.max_workers_spin.setValue(4)
        self.max_workers_spin.setToolTip("Numero massimo di thread per l'esecuzione parallela dei modelli.")
        parallel_h.addWidget(self.max_workers_spin)
        mtf_lay.addLayout(parallel_h)

        self.layout.addWidget(mtf_box)

        # ========== ENHANCED MULTI-HORIZON SYSTEM ==========
        enhanced_box = QGroupBox("Enhanced Multi-Horizon Predictions")
        enhanced_box.setToolTip("Sistema avanzato per predizioni multi-orizzonte con scaling intelligente e scenario trading.")
        enhanced_lay = QVBoxLayout(enhanced_box)

        # Enable enhanced scaling
        self.enhanced_scaling_cb = QCheckBox("Enable Enhanced Multi-Horizon Scaling")
        self.enhanced_scaling_cb.setToolTip("Attiva il sistema di scaling intelligente per predizioni multi-orizzonte da un singolo modello.")
        self.enhanced_scaling_cb.setChecked(True)
        enhanced_lay.addWidget(self.enhanced_scaling_cb)

        # Scaling mode selection
        scaling_h = QHBoxLayout()
        scaling_h.addWidget(QLabel("Scaling Mode:"))
        from PySide6.QtWidgets import QComboBox
        self.scaling_mode_combo = QComboBox()
        self.scaling_mode_combo.addItems([
            "smart_adaptive",  # Default
            "linear",
            "sqrt",
            "log",
            "volatility_adjusted",
            "regime_aware"
        ])
        self.scaling_mode_combo.setToolTip(
            "Modalità di scaling:\n"
            "• smart_adaptive: Combina volatilità, regime e fattori temporali\n"
            "• linear: Scaling lineare tradizionale\n"
            "• sqrt: Scaling con radice quadrata (decay non-lineare)\n"
            "• log: Scaling logaritmico\n"
            "• volatility_adjusted: Basato su volatilità corrente\n"
            "• regime_aware: Adattivo al regime di mercato"
        )
        scaling_h.addWidget(self.scaling_mode_combo)
        enhanced_lay.addLayout(scaling_h)

        # Trading scenario selection
        scenario_h = QHBoxLayout()
        scenario_h.addWidget(QLabel("Trading Scenario:"))
        self.scenario_combo = QComboBox()

        # Import scenario list from horizon converter
        try:
            from ..utils.horizon_converter import get_trading_scenarios
            scenarios = get_trading_scenarios()
            self.scenario_combo.addItem("Custom (Use Manual Horizons)", "")
            for key, name in scenarios.items():
                self.scenario_combo.addItem(name, key)
        except Exception:
            # Fallback scenarios
            self.scenario_combo.addItems([
                "Custom (Use Manual Horizons)",
                "Scalping (High Frequency)",
                "Intraday 4h",
                "Intraday 8h",
                "Intraday 2 Days",
                "Intraday 3 Days",
                "Intraday 5 Days",
                "Intraday 10 Days",
                "Intraday 15 Days"
            ])

        self.scenario_combo.setToolTip(
            "Scenari di trading predefiniti:\n"
            "• Scalping: 1m-15m, movimenti micro\n"
            "• Intraday 4h: 5m-4h, trend intraday\n"
            "• Intraday 8h: 15m-8h, sessione completa\n"
            "• Intraday 2-15d: Trend a medio-lungo termine\n"
            "• Custom: Usa orizzonti manuali"
        )
        scenario_h.addWidget(self.scenario_combo)
        enhanced_lay.addLayout(scenario_h)

        # Custom horizons for scenarios
        custom_h = QHBoxLayout()
        custom_h.addWidget(QLabel("Custom Horizons:"))
        self.custom_horizons_edit = QLineEdit("10m, 30m, 1h, 4h")
        self.custom_horizons_edit.setToolTip("Orizzonti personalizzati (comma-separated) usati quando scenario è 'Custom'.")
        custom_h.addWidget(self.custom_horizons_edit)
        enhanced_lay.addLayout(custom_h)

        # Performance monitoring
        perf_h = QHBoxLayout()
        self.performance_tracking_cb = QCheckBox("Enable Performance Tracking")
        self.performance_tracking_cb.setToolTip("Traccia le performance delle predizioni in tempo reale per monitoraggio e alerting.")
        self.performance_tracking_cb.setChecked(True)
        perf_h.addWidget(self.performance_tracking_cb)
        enhanced_lay.addLayout(perf_h)

        self.layout.addWidget(enhanced_box)

        # Control Settings
        control_box = QGroupBox("Control Settings")
        control_lay = QFormLayout(control_box)

        self.max_forecasts_spin = QSpinBox()
        self.max_forecasts_spin.setRange(1, 100)
        self.max_forecasts_spin.setValue(5)
        control_lay.addRow("Max Forecasts:", self.max_forecasts_spin)

        self.test_history_spin = QSpinBox()
        self.test_history_spin.setRange(1, 10000)
        self.test_history_spin.setValue(128)
        control_lay.addRow("Test History:", self.test_history_spin)

        auto_h = QHBoxLayout()
        self.auto_checkbox = QCheckBox("Auto Predict")
        self.auto_interval_spin = QSpinBox()
        self.auto_interval_spin.setRange(1, 86400)
        self.auto_interval_spin.setValue(60)
        auto_h.addWidget(self.auto_checkbox)
        auto_h.addWidget(QLabel("Interval (s):"))
        auto_h.addWidget(self.auto_interval_spin)
        control_lay.addRow(auto_h)

        self.layout.addWidget(control_box)

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

    def _sync_indicators_to_legacy(self):
        """Sync simplified indicators to legacy indicator_checks structure."""
        # Get active timeframes
        active_tfs = []
        if self.tf_1m_cb.isChecked(): active_tfs.append("1m")
        if self.tf_5m_cb.isChecked(): active_tfs.append("5m")
        if self.tf_15m_cb.isChecked(): active_tfs.append("15m")
        if self.tf_1h_cb.isChecked(): active_tfs.append("1h")

        # Reset all
        for ind in self._indicators:
            for tf in self._timeframes:
                self.indicator_checks[ind][tf].setChecked(False)

        # Set active combinations
        if self.atr_cb.isChecked():
            for tf in active_tfs:
                if tf in self._timeframes:
                    self.indicator_checks["ATR"][tf].setChecked(True)
        if self.rsi_cb.isChecked():
            for tf in active_tfs:
                if tf in self._timeframes:
                    self.indicator_checks["RSI"][tf].setChecked(True)
        if self.bb_cb.isChecked():
            for tf in active_tfs:
                if tf in self._timeframes:
                    self.indicator_checks["Bollinger"][tf].setChecked(True)
        if self.macd_cb.isChecked():
            for tf in active_tfs:
                if tf in self._timeframes:
                    self.indicator_checks["MACD"][tf].setChecked(True)
        if self.don_cb.isChecked():
            for tf in active_tfs:
                if tf in self._timeframes:
                    self.indicator_checks["Donchian"][tf].setChecked(True)
        if self.keltner_cb.isChecked():
            for tf in active_tfs:
                if tf in self._timeframes:
                    self.indicator_checks["Keltner"][tf].setChecked(True)
        if self.hurst_cb.isChecked():
            for tf in active_tfs:
                if tf in self._timeframes:
                    self.indicator_checks["Hurst"][tf].setChecked(True)

    def _sync_legacy_to_indicators(self):
        """Sync legacy indicator_checks to simplified indicators."""
        # Find which timeframes have any active indicators
        active_tfs = set()
        for ind in self._indicators:
            for tf in self._timeframes:
                if self.indicator_checks[ind][tf].isChecked():
                    active_tfs.add(tf)

        # Set timeframe checkboxes
        self.tf_1m_cb.setChecked("1m" in active_tfs)
        self.tf_5m_cb.setChecked("5m" in active_tfs)
        self.tf_15m_cb.setChecked("15m" in active_tfs)
        self.tf_1h_cb.setChecked("1h" in active_tfs)

        # Set indicator checkboxes based on whether they're active in any timeframe
        self.atr_cb.setChecked(any(self.indicator_checks["ATR"][tf].isChecked() for tf in self._timeframes))
        self.rsi_cb.setChecked(any(self.indicator_checks["RSI"][tf].isChecked() for tf in self._timeframes))
        self.bb_cb.setChecked(any(self.indicator_checks["Bollinger"][tf].isChecked() for tf in self._timeframes))
        self.macd_cb.setChecked(any(self.indicator_checks["MACD"][tf].isChecked() for tf in self._timeframes))
        self.don_cb.setChecked(any(self.indicator_checks["Donchian"][tf].isChecked() for tf in self._timeframes))
        self.keltner_cb.setChecked(any(self.indicator_checks["Keltner"][tf].isChecked() for tf in self._timeframes))
        self.hurst_cb.setChecked(any(self.indicator_checks["Hurst"][tf].isChecked() for tf in self._timeframes))

    def _browse_model_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.pt *.pth *.pkl *.pickle);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def _browse_model_paths_multi(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Model Files", "", "Model Files (*.pt *.pth *.pkl *.pickle);;All Files (*)"
        )
        if not paths:
            return

        # Update text area with new paths
        current_text = self.models_edit.toPlainText().strip()
        existing_paths = [line.strip() for line in current_text.splitlines() if line.strip()] if current_text else []

        # Add new paths to existing ones
        all_paths = list(set(existing_paths + [str(p) for p in paths]))
        self.models_edit.setPlainText('\n'.join(sorted(all_paths)))

        # Update internal state for compatibility
        self._model_paths = all_paths
        self.__class__._last_model_paths = list(all_paths)

    @staticmethod
    def get_model_paths():
        """Return last selected model paths from dialog multi-selection."""
        try:
            return list(getattr(PredictionSettingsDialog, "_last_model_paths", []) or [])
        except Exception:
            return []

    def _load_model_meta(self, path: str) -> Dict[str, Any]:
        """Try to load meta from sidecar (path.meta.json) or from inside model (pickle/torch dict)."""
        from pathlib import Path
        import json
        p = Path(path) if path else None
        if not p or not p.exists():
            return {}
        # 1) sidecar
        side = Path(str(p) + ".meta.json")
        if side.exists():
            try:
                return json.loads(side.read_text(encoding="utf-8"))
            except Exception:
                pass
        # 2) embedded
        try:
            if p.suffix.lower() in (".pkl", ".pickle"):
                import pickle
                with open(p, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict):
                    return obj.get("meta", {})
            elif p.suffix.lower() in (".pt", ".pth"):
                try:
                    import torch  # type: ignore
                except Exception:
                    torch = None
                if torch is not None:
                    ckpt = torch.load(str(p), map_location="cpu")
                    if isinstance(ckpt, dict):
                        return ckpt.get("meta", {})
        except Exception:
            pass
        return {}

    def _show_model_info(self):
        """Open a popup with model meta if available."""
        from PySide6.QtWidgets import QMessageBox
        import json
        path = self.model_path_edit.text().strip()
        meta = self._load_model_meta(path)
        if not meta:
            QMessageBox.information(self, "Model Info", "Nessun metadato trovato (né sidecar né embedded).")
            return
        try:
            text = json.dumps(meta, indent=2, ensure_ascii=False)
        except Exception:
            text = str(meta)
        QMessageBox.information(self, "Model Info", text)

    def _apply_model_defaults_from_meta(self, meta: Dict[str, Any]):
        """Apply common defaults from meta to dialog fields."""
        try:
            # horizons
            hz = meta.get("horizons") or meta.get("default_horizons")
            if isinstance(hz, (list, tuple)):
                self.horizons_edit.setText(", ".join([str(x) for x in hz]))
            # N_samples, conformal, weight
            if "N_samples" in meta:
                self.n_samples_spinbox.setValue(int(meta.get("N_samples", self.n_samples_spinbox.value())))
            if "apply_conformal" in meta:
                self.conformal_checkbox.setChecked(bool(meta.get("apply_conformal", self.conformal_checkbox.isChecked())))
            if "model_weight_pct" in meta:
                # try to match the combo data
                target = int(meta.get("model_weight_pct", 100))
                for i in range(self.model_weight_combo.count()):
                    if int(self.model_weight_combo.itemData(i)) == target:
                        self.model_weight_combo.setCurrentIndex(i); break

            # advanced params
            adv = meta.get("advanced_params", {})
            def _set_spin(spin, key, cast=int):
                try:
                    if key in adv:
                        spin.setValue(cast(adv[key]))
                except Exception:
                    pass
            _set_spin(self.warmup_spin, "warmup_bars")
            _set_spin(self.atr_n_spin, "atr_n")
            _set_spin(self.rsi_n_spin, "rsi_n")
            _set_spin(self.bb_n_spin, "bb_n")
            _set_spin(self.hurst_window_spin, "hurst_window")
            _set_spin(self.rv_window_spin, "rv_window")

            # indicator × timeframes
            ind_tfs = meta.get("indicator_tfs", {})
            if isinstance(ind_tfs, dict) and hasattr(self, "indicator_checks"):
                for ind, tfmap in self.indicator_checks.items():
                    lst = ind_tfs.get(ind.lower()) or ind_tfs.get(ind) or []
                    for tf, cb in tfmap.items():
                        try:
                            cb.setChecked(tf in lst)
                        except Exception:
                            pass
        except Exception:
            pass

    def _load_model_defaults(self):
        """Load meta and apply as dialog defaults."""
        from PySide6.QtWidgets import QMessageBox
        path = self.model_path_edit.text().strip()
        if not path:
            QMessageBox.information(self, "Carica parametri", "Seleziona prima un Model Path.")
            return
        meta = self._load_model_meta(path)
        if not meta:
            QMessageBox.information(self, "Carica parametri", "Nessun metadato trovato per questo modello.")
            return
        self._apply_model_defaults_from_meta(meta)
        QMessageBox.information(self, "Carica parametri", "Parametri base caricati dal modello.")

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

            # Multi-timeframe hierarchical strategy
            self.hierarchical_cb.setChecked(bool(settings.get("use_hierarchical_multitf", False)))
            query_tf = settings.get("query_timeframe", "1h")
            idx = self.query_timeframe_combo.findText(query_tf)
            if idx >= 0:
                self.query_timeframe_combo.setCurrentIndex(idx)

            hierarchical_tfs = settings.get("hierarchical_timeframes", ["1m", "5m", "15m", "1h"])
            if isinstance(hierarchical_tfs, list):
                self.hierarchical_timeframes_edit.setText(", ".join(hierarchical_tfs))
            else:
                self.hierarchical_timeframes_edit.setText("1m, 5m, 15m, 1h")

            self.exclude_children_cb.setChecked(bool(settings.get("exclude_children", True)))
            self.parallel_inference_cb.setChecked(bool(settings.get("use_parallel_inference", True)))
            self.max_workers_spin.setValue(int(settings.get("max_parallel_workers", 4)))

            # Enhanced Multi-Horizon System
            self.enhanced_scaling_cb.setChecked(bool(settings.get("use_enhanced_scaling", True)))

            scaling_mode = settings.get("scaling_mode", "smart_adaptive")
            scaling_index = self.scaling_mode_combo.findText(scaling_mode)
            if scaling_index >= 0:
                self.scaling_mode_combo.setCurrentIndex(scaling_index)

            scenario = settings.get("trading_scenario", "")
            scenario_index = self.scenario_combo.findData(scenario)
            if scenario_index >= 0:
                self.scenario_combo.setCurrentIndex(scenario_index)

            custom_horizons = settings.get("custom_horizons", ["10m", "30m", "1h", "4h"])
            if isinstance(custom_horizons, list):
                self.custom_horizons_edit.setText(", ".join(custom_horizons))
            else:
                self.custom_horizons_edit.setText("10m, 30m, 1h, 4h")

            self.performance_tracking_cb.setChecked(bool(settings.get("enable_performance_tracking", True)))

            # Sync legacy indicator structure to simplified indicators
            self._sync_legacy_to_indicators()

            # Load dialog geometry
            geometry = settings.get("dialog_geometry", None)
            if geometry and len(geometry) == 4:
                self.setGeometry(geometry[0], geometry[1], geometry[2], geometry[3])

            logger.info(f"Loaded prediction settings from {CONFIG_FILE}")
        except Exception as e:
            logger.exception(f"Failed to load prediction settings: {e}")

    def save_settings(self):
        """Saves the current settings to the JSON config file."""
        # Sync simplified indicators to legacy structure before saving
        self._sync_indicators_to_legacy()

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

            # Multi-timeframe hierarchical strategy
            "use_hierarchical_multitf": bool(self.hierarchical_cb.isChecked()),
            "query_timeframe": str(self.query_timeframe_combo.currentText()),
            "hierarchical_timeframes": [h.strip() for h in self.hierarchical_timeframes_edit.text().split(",") if h.strip()],
            "exclude_children": bool(self.exclude_children_cb.isChecked()),
            "use_parallel_inference": bool(self.parallel_inference_cb.isChecked()),
            "max_parallel_workers": int(self.max_workers_spin.value()),

            # Enhanced Multi-Horizon System
            "use_enhanced_scaling": bool(self.enhanced_scaling_cb.isChecked()),
            "scaling_mode": str(self.scaling_mode_combo.currentText()),
            "trading_scenario": str(self.scenario_combo.currentData() or ""),
            "custom_horizons": [h.strip() for h in self.custom_horizons_edit.text().split(",") if h.strip()],
            "enable_performance_tracking": bool(self.performance_tracking_cb.isChecked()),

            # Save dialog geometry
            "dialog_geometry": [self.geometry().x(), self.geometry().y(), self.geometry().width(), self.geometry().height()],
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
