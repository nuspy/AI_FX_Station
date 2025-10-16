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
]

ADDITIONAL_FEATURES = [
]

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Comprehensive tooltips for each indicator
INDICATOR_TOOLTIPS = {












                "Cosa è: banda superiore = max high, banda inferiore = min low.\n"





}

# Tooltips for additional features
FEATURE_TOOLTIPS = {



}

# Tooltips for advanced parameters
PARAMETER_TOOLTIPS = {





}

# Default selections for indicators
DEFAULTS = {
}


class TrainingTab(QWidget):
    Training Tab: configure and launch model training.
    - Symbol/timeframe/days/horizon selectors
    - 4-column indicator grid with master checkbox
    - Additional features with enable/parameters
    - Advanced parameters exposed
    - Async training with progress bar and log

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
        try:
            settings = {
                # Top controls

                # Indicator selections
                    ind: [tf for tf, cb in self.indicator_checks[ind].items() if cb.isChecked()]
                    for ind in INDICATORS
                },

                # Additional features

                # Advanced parameters

                # Diffusion parameters

                # NVIDIA Optimization Stack

                # Output directory
            }

            # Save to file
            TRAINING_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TRAINING_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)

            logger.debug(f"Training settings saved to {TRAINING_SETTINGS_FILE}")

        except Exception as e:
            logger.exception(f"Failed to save training settings: {e}")

    def _load_settings(self):
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
                self.horizon_spin.setValue(settings['horizon'])
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
        self._save_settings()
        super().closeEvent(event)

    def hideEvent(self, event):
        self._save_settings()
        super().hideEvent(event)
    
    def _apply_i18n_tooltips(self):
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


    def _build_top_controls(self):

        # Row 0: Model Name + Load/Save Config
        row0 = QHBoxLayout()

        lbl_name = QLabel("Model Name:")


        )
        row0.addWidget(lbl_name)

        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Auto-generate from features")


        )
        row0.addWidget(self.model_name_edit)

        row0.addStretch()

        self.load_config_btn = QPushButton("📂 Load Config")


        )
        self.load_config_btn.clicked.connect(self._on_load_config)
        row0.addWidget(self.load_config_btn)

        self.save_config_btn = QPushButton("💾 Save Config")


        )
        self.save_config_btn.clicked.connect(self._on_save_config)
        row0.addWidget(self.save_config_btn)

        self.layout.addLayout(row0)

        # Row 1: Symbol, Timeframe, Days, Horizon, Model, Encoder, Opt, Gen, Pop
        top = QHBoxLayout()

        # Symbol
        lbl_sym = QLabel("Symbol:")


        )
        top.addWidget(lbl_sym)
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["EUR/USD", "GBP/USD", "AUX/USD", "GBP/NZD", "AUD/JPY", "GBP/EUR", "GBP/AUD"])


        )
        top.addWidget(self.symbol_combo)

        # Base timeframe
        lbl_tf = QLabel("Base TF:")


        )
        top.addWidget(lbl_tf)
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        self.tf_combo.setCurrentText("1m")


        )
        top.addWidget(self.tf_combo)

        # Days history
        lbl_days = QLabel("Days:")


            "Perché è importante: più dati = migliore generalizzazione, ma training più lento.\n"
        )
        top.addWidget(lbl_days)
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 3650)
        self.days_spin.setValue(7)


            "Nota: con TF=1m, 7 giorni ≈ 10K samples. Con TF=1h, 7 giorni ≈ 168 samples."
        )
        top.addWidget(self.days_spin)

        # Horizon
        lbl_h = QLabel("Horizon:")


        )
        top.addWidget(lbl_h)
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 500)
        self.horizon_spin.setValue(5)


            "Esempio: TF=1m, horizon=5 → predici prossimi 5 minuti."
        )
        top.addWidget(self.horizon_spin)

        # Model
        lbl_m = QLabel("Model:")


        )
        top.addWidget(lbl_m)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["ridge", "lasso", "elasticnet", "rf", "lightning", "diffusion-ddpm", "diffusion-ddim", "sssd"])


        )
        top.addWidget(self.model_combo)

        # Encoder
        lbl_e = QLabel("Encoder:")


        )
        top.addWidget(lbl_e)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(["none", "pca", "autoencoder", "vae", "latents"])


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


                    f"GPU rilevata: {gpu_name}\n\n"
                    f"Speedup atteso:\n"
                    f"• Autoencoder: 10-15x più veloce\n"
                    f"• VAE: 10-15x più veloce\n\n"
                    f"IMPORTANTE: solo encoder usano GPU.\n"
                    f"Ridge/Lasso/ElasticNet/RF rimangono su CPU."
                )
            else:


                )
        except Exception:
            self.use_gpu_training_check.setEnabled(False)


        top.addWidget(self.use_gpu_training_check)

        # Optimization
        lbl_opt = QLabel("Opt:")


        )
        top.addWidget(lbl_opt)
        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["none", "genetic-basic", "nsga2"])


            "Nota: gen=10, pop=20 → 200 training runs → 200x tempo!"
        )
        top.addWidget(self.opt_combo)

        # Gen
        lbl_gen = QLabel("Gen:")


            "Perché è importante: più generazioni = migliore ottimizzazione, ma più lento.\n"
        )
        top.addWidget(lbl_gen)
        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(1, 50)
        self.gen_spin.setValue(5)


            "- gen=5, pop=8 → 40 runs → ~40min (ridge), ~4h (rf)\n"
            "- gen=20, pop=20 → 400 runs → ~6h (ridge), ~2 giorni (rf)"
        )
        top.addWidget(self.gen_spin)

        # Pop
        lbl_pop = QLabel("Pop:")


            "Perché è importante: più popolazione = esplorazione spazio più ampia.\n"
        )
        top.addWidget(lbl_pop)
        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(2, 64)
        self.pop_spin.setValue(8)


            "- bilanciato: pop=20, gen=15 → 300 runs → ~5h (ridge)"
        )
        top.addWidget(self.pop_spin)

        self.layout.addLayout(top)

    def _build_indicator_grid(self):
        grid_box = QGroupBox("Indicatori Tecnici")

        grid_layout = QVBoxLayout(grid_box)

        # Master checkbox at top
        self.use_indicators_check = QCheckBox("Usa indicatori selezionati nel training")
        self.use_indicators_check.setChecked(True)

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

            grid.addWidget(lbl, row_offset, col_offset)

            # Timeframe checkboxes
            self.indicator_checks[ind] = {}
            selected = saved.get(ind, DEFAULTS.get(ind, []))
            for j, tf in enumerate(TIMEFRAMES, start=1):
                cb = QCheckBox()
                cb.setChecked(tf in selected)

                self.indicator_checks[ind][tf] = cb
                grid.addWidget(cb, row_offset, col_offset + j)

        grid_layout.addLayout(grid)
        self.layout.addWidget(grid_box)

    def _build_additional_features(self):
        feat_box = QGroupBox("Feature Aggiuntive")

        feat_layout = QGridLayout(feat_box)

        # Returns & Volatility
        self.returns_check = QCheckBox("Returns & Volatility")
        self.returns_check.setChecked(True)

        feat_layout.addWidget(self.returns_check, 0, 0)

        lbl_ret_win = QLabel("Window:")

        feat_layout.addWidget(lbl_ret_win, 0, 1)
        self.returns_window = QSpinBox()
        self.returns_window.setRange(1, 100)
        self.returns_window.setValue(5)

        feat_layout.addWidget(self.returns_window, 0, 2)

        # Trading Sessions
        self.sessions_check = QCheckBox("Trading Sessions")
        self.sessions_check.setChecked(True)

        feat_layout.addWidget(self.sessions_check, 1, 0)

        lbl_sess_overlap = QLabel("Overlap (min):")

        feat_layout.addWidget(lbl_sess_overlap, 1, 1)
        self.session_overlap = QSpinBox()
        self.session_overlap.setRange(0, 120)
        self.session_overlap.setValue(30)

        feat_layout.addWidget(self.session_overlap, 1, 2)

        # Candlestick Patterns
        self.candlestick_check = QCheckBox("Candlestick Patterns")
        self.candlestick_check.setChecked(False)

        feat_layout.addWidget(self.candlestick_check, 2, 0)

        lbl_higher_tf = QLabel("Higher TF:")

        feat_layout.addWidget(lbl_higher_tf, 2, 1)
        self.higher_tf_combo = QComboBox()
        self.higher_tf_combo.addItems(["5m", "15m", "30m", "1h", "4h", "1d"])
        self.higher_tf_combo.setCurrentText("15m")

        feat_layout.addWidget(self.higher_tf_combo, 2, 2)

        # Volume Profile
        self.volume_profile_check = QCheckBox("Volume Profile")
        self.volume_profile_check.setChecked(False)

        feat_layout.addWidget(self.volume_profile_check, 3, 0)

        lbl_vp_bins = QLabel("Bins:")

        feat_layout.addWidget(lbl_vp_bins, 3, 1)
        self.vp_bins = QSpinBox()
        self.vp_bins.setRange(10, 200)
        self.vp_bins.setValue(50)

        feat_layout.addWidget(self.vp_bins, 3, 2)

        lbl_vp_window = QLabel("Window:")

        feat_layout.addWidget(lbl_vp_window, 3, 3)
        self.vp_window = QSpinBox()
        self.vp_window.setRange(20, 500)
        self.vp_window.setValue(100)

        feat_layout.addWidget(self.vp_window, 3, 4)

        # VSA (Volume Spread Analysis)
        self.vsa_check = QCheckBox("VSA (Volume Spread Analysis)")
        self.vsa_check.setChecked(False)

        feat_layout.addWidget(self.vsa_check, 4, 0)

        lbl_vsa_vol_ma = QLabel("Vol MA:")

        feat_layout.addWidget(lbl_vsa_vol_ma, 4, 1)
        self.vsa_volume_ma = QSpinBox()
        self.vsa_volume_ma.setRange(5, 100)
        self.vsa_volume_ma.setValue(20)

        feat_layout.addWidget(self.vsa_volume_ma, 4, 2)

        lbl_vsa_spread_ma = QLabel("Spread MA:")

        feat_layout.addWidget(lbl_vsa_spread_ma, 4, 3)
        self.vsa_spread_ma = QSpinBox()
        self.vsa_spread_ma.setRange(5, 100)
        self.vsa_spread_ma.setValue(20)

        feat_layout.addWidget(self.vsa_spread_ma, 4, 4)

        self.layout.addWidget(feat_box)

    def _build_advanced_params(self):
        adv_box = QGroupBox("Parametri Avanzati")

        adv = QGridLayout(adv_box)

        row = 0

        # Warmup bars
        lbl_wu = QLabel("Warmup bars:")

        adv.addWidget(lbl_wu, row, 0)
        self.warmup = QSpinBox()
        self.warmup.setRange(0, 5000)
        self.warmup.setValue(16)

        adv.addWidget(self.warmup, row, 1)

        # RV window
        lbl_rv = QLabel("RV window:")

        adv.addWidget(lbl_rv, row, 2)
        self.rv_w = QSpinBox()
        self.rv_w.setRange(1, 10000)
        self.rv_w.setValue(60)

        adv.addWidget(self.rv_w, row, 3)

        # Min coverage
        lbl_cov = QLabel("Min coverage:")

        adv.addWidget(lbl_cov, row, 4)
        self.min_coverage = QDoubleSpinBox()
        self.min_coverage.setRange(0.0, 1.0)
        self.min_coverage.setSingleStep(0.05)
        self.min_coverage.setDecimals(2)
        self.min_coverage.setValue(0.15)

        adv.addWidget(self.min_coverage, row, 5)

        row += 1

        # ATR period
        lbl_atr = QLabel("ATR period:")

        adv.addWidget(lbl_atr, row, 0)
        self.atr_n = QSpinBox()
        self.atr_n.setRange(1, 500)
        self.atr_n.setValue(14)

        adv.addWidget(self.atr_n, row, 1)

        # RSI period
        lbl_rsi = QLabel("RSI period:")

        adv.addWidget(lbl_rsi, row, 2)
        self.rsi_n = QSpinBox()
        self.rsi_n.setRange(2, 500)
        self.rsi_n.setValue(14)

        adv.addWidget(self.rsi_n, row, 3)

        # Bollinger period
        lbl_bb = QLabel("Bollinger period:")

        adv.addWidget(lbl_bb, row, 4)
        self.bb_n = QSpinBox()
        self.bb_n.setRange(2, 500)
        self.bb_n.setValue(20)

        adv.addWidget(self.bb_n, row, 5)

        row += 1

        # Hurst window
        lbl_hu = QLabel("Hurst window:")

        adv.addWidget(lbl_hu, row, 0)
        self.hurst_w = QSpinBox()
        self.hurst_w.setRange(8, 4096)
        self.hurst_w.setValue(64)

        adv.addWidget(self.hurst_w, row, 1)

        # Lightning epochs (only for lightning model)
        lbl_epochs = QLabel("Lightning epochs:")

        adv.addWidget(lbl_epochs, row, 2)
        self.light_epochs = QSpinBox()
        self.light_epochs.setRange(1, 1000)
        self.light_epochs.setValue(30)

        adv.addWidget(self.light_epochs, row, 3)

        # Lightning batch
        lbl_batch = QLabel("Lightning batch:")

        adv.addWidget(lbl_batch, row, 4)
        self.light_batch = QSpinBox()
        self.light_batch.setRange(4, 512)
        self.light_batch.setValue(64)

        adv.addWidget(self.light_batch, row, 5)

        row += 1

        # Lightning val_frac
        lbl_val = QLabel("Lightning val_frac:")

        adv.addWidget(lbl_val, row, 0)
        self.light_val_frac = QDoubleSpinBox()
        self.light_val_frac.setRange(0.05, 0.5)
        self.light_val_frac.setSingleStep(0.05)
        self.light_val_frac.setDecimals(2)
        self.light_val_frac.setValue(0.2)

        adv.addWidget(self.light_val_frac, row, 1)

        # Lightning patch_len
        lbl_patch = QLabel("Lightning patch:")

        adv.addWidget(lbl_patch, row, 2)
        self.patch_len = QSpinBox()
        self.patch_len.setRange(16, 1024)
        self.patch_len.setValue(64)

        adv.addWidget(self.patch_len, row, 3)

        row += 1

        # Encoder latent dimension
        lbl_latent = QLabel("Encoder latent dim:")

        adv.addWidget(lbl_latent, row, 0)
        self.latent_dim = QSpinBox()
        self.latent_dim.setRange(2, 256)
        self.latent_dim.setValue(16)

        adv.addWidget(self.latent_dim, row, 1)

        # Encoder training epochs
        lbl_enc_epochs = QLabel("Encoder epochs:")

        adv.addWidget(lbl_enc_epochs, row, 2)
        self.encoder_epochs = QSpinBox()
        self.encoder_epochs.setRange(10, 500)
        self.encoder_epochs.setValue(50)

        adv.addWidget(self.encoder_epochs, row, 3)

        row += 1

        # === DIFFUSION MODEL PARAMETERS (only used when model=diffusion-*) ===
        lbl_diff_section = QLabel("─── Diffusion Model Parameters ───")
        lbl_diff_section.setStyleSheet("font-weight: bold; color: #2980b9;")
        adv.addWidget(lbl_diff_section, row, 0, 1, 6)
        row += 1

        # Diffusion timesteps
        lbl_timesteps = QLabel("Diffusion timesteps:")


            "Perché è importante: più steps = migliore qualità, ma inference più lenta.\n"
            "SOLO per model=diffusion-ddpm o diffusion-ddim."
        )
        adv.addWidget(lbl_timesteps, row, 0)
        self.diffusion_timesteps = QSpinBox()
        self.diffusion_timesteps.setRange(10, 5000)
        self.diffusion_timesteps.setValue(200)


        )
        adv.addWidget(self.diffusion_timesteps, row, 1)

        # Learning rate
        lbl_lr = QLabel("Learning rate:")


            "SOLO per model=lightning/diffusion-*."
        )
        adv.addWidget(lbl_lr, row, 2)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(1e-6, 1e-1)
        self.learning_rate.setSingleStep(1e-5)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setValue(1e-4)


        )
        adv.addWidget(self.learning_rate, row, 3)

        # Batch size (diffusion/lightning)
        lbl_batch_diff = QLabel("Batch size (DL):")


            "Perché è importante: batch grande = gradiente stabile, batch piccolo = più noise.\n"
            "SOLO per model=lightning/diffusion-*."
        )
        adv.addWidget(lbl_batch_diff, row, 4)
        self.batch_size_dl = QSpinBox()
        self.batch_size_dl.setRange(4, 512)
        self.batch_size_dl.setValue(64)


        )
        adv.addWidget(self.batch_size_dl, row, 5)

        row += 1

        # Model channels (UNet capacity)
        lbl_channels = QLabel("Model channels:")


            "Perché è importante: più canali = più parametri = più capacity, ma overfitting.\n"
            "SOLO per model=diffusion-* (parametri UNet = channels² × layers)."
        )
        adv.addWidget(lbl_channels, row, 0)
        self.model_channels = QSpinBox()
        self.model_channels.setRange(32, 512)
        self.model_channels.setValue(128)


        )
        adv.addWidget(self.model_channels, row, 1)

        # Dropout
        lbl_dropout = QLabel("Dropout:")


            "SOLO per model=lightning/diffusion-* (neural networks)."
        )
        adv.addWidget(lbl_dropout, row, 2)
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.0, 0.6)
        self.dropout.setSingleStep(0.05)
        self.dropout.setDecimals(2)
        self.dropout.setValue(0.1)


        )
        adv.addWidget(self.dropout, row, 3)

        # Num heads (Transformer)
        lbl_heads = QLabel("Attention heads:")


            "Perché è importante: più heads = cattura pattern diversi in parallelo.\n"
            "Best practice: num_heads deve dividere model_channels (es. 128/8=16).\n"
            "SOLO per model=diffusion-* con architecture=transformer."
        )
        adv.addWidget(lbl_heads, row, 4)
        self.num_heads = QSpinBox()
        self.num_heads.setRange(1, 16)
        self.num_heads.setValue(8)


            "Vincolo: model_channels % num_heads == 0."
        )
        adv.addWidget(self.num_heads, row, 5)

        self.layout.addWidget(adv_box)

        # NVIDIA Optimization Stack section
        self._build_nvidia_optimizations()

    def _build_nvidia_optimizations(self):
        nvidia_box = QGroupBox("🚀 NVIDIA Optimization Stack (GPU Acceleration)")


        )
        nvidia_layout = QGridLayout(nvidia_box)

        row = 0

        # Master enable checkbox
        self.nvidia_enable = QCheckBox("Abilita NVIDIA Optimization Stack")


        )
        self.nvidia_enable.setChecked(False)
        nvidia_layout.addWidget(self.nvidia_enable, row, 0, 1, 6)
        row += 1

        # AMP checkbox
        lbl_amp = QLabel("Mixed Precision (AMP):")


        )
        nvidia_layout.addWidget(lbl_amp, row, 0)

        self.use_amp = QCheckBox("Enable")
        self.use_amp.setChecked(True)

        nvidia_layout.addWidget(self.use_amp, row, 1)

        # Precision combo
        lbl_precision = QLabel("Precision:")
        nvidia_layout.addWidget(lbl_precision, row, 2)

        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["fp16", "bf16", "fp32"])
        self.precision_combo.setCurrentText("fp16")


            "fp16: FP16 (raccomandato, GPU >= GTX 10xx)\n"
        )
        nvidia_layout.addWidget(self.precision_combo, row, 3)
        row += 1

        # torch.compile checkbox
        lbl_compile = QLabel("torch.compile:")


            "Richiede PyTorch >= 2.0"
        )
        nvidia_layout.addWidget(lbl_compile, row, 0)

        self.compile_model = QCheckBox("Enable")
        self.compile_model.setChecked(True)

        nvidia_layout.addWidget(self.compile_model, row, 1)
        row += 1

        # Fused optimizer checkbox
        lbl_fused = QLabel("Fused Optimizer:")


        )
        nvidia_layout.addWidget(lbl_fused, row, 0)

        self.use_fused_optimizer = QCheckBox("Enable (requires APEX)")
        self.use_fused_optimizer.setChecked(False)  # Default off (requires APEX)

        nvidia_layout.addWidget(self.use_fused_optimizer, row, 1)
        row += 1

        # Flash Attention checkbox
        lbl_flash = QLabel("Flash Attention 2:")


        )
        nvidia_layout.addWidget(lbl_flash, row, 0)

        self.use_flash_attention = QCheckBox("Enable (requires Ampere+ GPU)")
        self.use_flash_attention.setChecked(False)  # Default off (requires Ampere+)

        nvidia_layout.addWidget(self.use_flash_attention, row, 1)
        row += 1

        # Gradient accumulation
        lbl_grad_accum = QLabel("Gradient Accumulation:")


        )
        nvidia_layout.addWidget(lbl_grad_accum, row, 0)

        self.grad_accumulation_steps = QSpinBox()
        self.grad_accumulation_steps.setRange(1, 32)
        self.grad_accumulation_steps.setValue(1)

        nvidia_layout.addWidget(self.grad_accumulation_steps, row, 1)
        row += 1

        # Info label
        info_label = QLabel(
            "ℹ️ Per installare APEX e Flash Attention:\n"
        )
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        nvidia_layout.addWidget(info_label, row, 0, 1, 6)

        self.layout.addWidget(nvidia_box)

    def _build_output_section(self):
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
        self.optimized_params_widget = OptimizedParamsDisplayWidget()
        self.optimized_params_widget.save_requested.connect(self._on_save_optimized_params)
        self.layout.addWidget(self.optimized_params_widget)

    def _build_log_section(self):
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
        actions = QHBoxLayout()

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self._start_training)

        self.validate_btn = QPushButton("Multi-Horizon Validation")
        self.validate_btn.clicked.connect(self._start_multi_horizon_validation)


        )

        self.grid_training_btn = QPushButton("Grid Training Manager")
        self.grid_training_btn.clicked.connect(self._open_grid_training)


        )
        self.grid_training_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        actions.addWidget(self.train_btn)
        actions.addWidget(self.validate_btn)
        actions.addWidget(self.grid_training_btn)
        self.layout.addLayout(actions)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Scegli cartella output", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _collect_indicator_tfs(self) -> Dict[str, List[str]]:
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
        m = {ind: [tf for tf, cb in self.indicator_checks[ind].items() if cb.isChecked()]
             for ind in INDICATORS}
        set_setting("training_indicator_tfs", m)

    def _start_training(self):
        try:
            # Clear previous optimized params display
            self.optimized_params_widget.clear()

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
            # Optimization is now implemented for both sklearn and lightning models

            from datetime import datetime, timezone

            # Additional features config
            additional_features = {
            }

            # Lightning and diffusion models use train.py
            if model in ['lightning', 'diffusion-ddpm', 'diffusion-ddim']:
                module = 'src.forex_diffusion.training.train'
                args = [
                    sys.executable, '-m', module,
                ]

                if self.vsa_check.isChecked():
                    args.extend([
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
                    },
                    },
                    },
                }
                pending_dir = artifacts_dir / 'lightning'
            else:
                module = 'src.forex_diffusion.training.train_sklearn'
                algo = model if model != 'latents' else 'ridge'
                # Use new encoder system instead of legacy pca flag
                args = [
                    sys.executable, '-m', module,
                ]

                # Add GPU flag if enabled
                if self.use_gpu_training_check.isChecked():
                    args.append('--use-gpu')
                meta = {
                    },
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
        try:
            self.log_view.append(line)
        except Exception:
            pass

    def _find_latest_model_file(self, out_dir: Path) -> Optional[Path]:
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
        if value < 0:
            self.progress.setRange(0, 0)  # indeterminate
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(value)

    def _on_finished(self, ok: bool):
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
        Handle save optimized parameters to database.

        This method is called when user clicks "Save to Database" in the
        optimized params widget (FASE 9 - Part 2).

        NOTE: Full implementation requires ParameterLoaderService integration.
        For now, this is a placeholder that shows the save dialog.
        try:
            # TODO: Integrate with ParameterLoaderService from FASE 2
            # from ..services.parameter_loader import ParameterLoaderService
            # loader = ParameterLoaderService(db_path)
            # success = loader.save_optimized_params(params)

            # For now, just log and show success message
            logger.info(f"Optimized params save requested: {params.get('pattern_type')} on {params.get('symbol')} {params.get('timeframe')}")

            QMessageBox.information(
                self,
            )

        except Exception as e:
            logger.exception(f"Failed to save optimized params: {e}")
            QMessageBox.critical(
                self,
                f"Failed to save optimized parameters:\n{str(e)}"
            )

    def _on_load_config(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                str(Path.home()),
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
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                str(Path.home() / "training_config.json"),
            )
            if not file_path:
                return

            config = {
                # Metadata

                # Top controls

                # Indicator selections
                    ind: [tf for tf, cb in self.indicator_checks[ind].items() if cb.isChecked()]
                    for ind in INDICATORS
                },

                # Additional features

                # Advanced parameters

                # Diffusion parameters

                # Output directory
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            QMessageBox.information(self, "Save Config", f"Configurazione salvata in:\n{file_path}")
            logger.info(f"Training configuration saved to {file_path}")

        except Exception as e:
            logger.exception(f"Failed to save config: {e}")
            QMessageBox.critical(self, "Save Config Error", f"Errore nel salvataggio configurazione:\n{e}")

    def _start_multi_horizon_validation(self):
        try:
            from PySide6.QtWidgets import QInputDialog

            # Ask user to select checkpoint file
            checkpoint_path, _ = QFileDialog.getOpenFileName(
                self,
                str(Path(self.out_dir.text()) / "lightning"),
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
                        })

                    df = pd.DataFrame(rows)
                    df.to_csv(output_csv, index=False)
                    self._append_log(f"\n[validation] Results exported to: {output_csv}")

                    # Show completion message
                    QMessageBox.information(
                        self,
                        f"Multi-horizon validation completed successfully.\n\nResults saved to:\n{output_csv}"
                    )

                except Exception as e:
                    logger.exception(f"Validation failed: {e}")
                    self._append_log(f"\n[validation] ERROR: {e}")
                    QMessageBox.critical(
                        self,
                        f"Validation failed:\n{e}"
                    )

            # Start validation thread
            thread = threading.Thread(target=run_validation, daemon=True)
            thread.start()

        except Exception as e:
            logger.exception(f"Failed to start validation: {e}")

    def _open_grid_training(self):
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
                f"Failed to open Grid Training Manager:\n{e}"
            )
