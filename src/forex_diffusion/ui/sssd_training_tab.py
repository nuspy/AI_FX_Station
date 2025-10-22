"""
SSSD Training Tab

Standalone tab for SSSD model training in Generative Forecast → Training.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QLabel, QScrollArea,
    QCheckBox, QFileDialog, QProgressBar, QTextEdit, QMessageBox
)
from PySide6.QtCore import Signal
from loguru import logger
from pathlib import Path
import json


class SSSDTrainingTab(QWidget):
    """
    SSSD Training Tab - Train SSSD multi-timeframe diffusion models.
    """
    
    trainingStarted = Signal(dict)
    trainingFinished = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_settings()
        
        logger.info("SSSD Training Tab initialized")
    
    def _build_ui(self):
        """Build SSSD training UI"""
        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Header
        header_box = QGroupBox("SSSD Training")
        header_layout = QVBoxLayout(header_box)
        header_label = QLabel(
            "<b>Train SSSD multi-timeframe diffusion forecasting model</b><br><br>"
            "SSSD combina S4 layers (State Space Models) per encoding multi-timeframe + "
            "Diffusion per forecasting probabilistico.<br><br>"
            "<b>Architettura:</b> Multi-timeframe S4 Encoder → Horizon Embeddings → Diffusion Head"
        )
        header_label.setWordWrap(True)
        header_layout.addWidget(header_label)
        layout.addWidget(header_box)
        
        # Data Settings
        data_box = QGroupBox("Data Settings")
        data_layout = QFormLayout(data_box)
        
        # Symbol
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"])
        self.symbol_combo.setToolTip("Symbol to train on")
        data_layout.addRow("Symbol:", self.symbol_combo)
        
        # Timeframe
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h"])
        self.timeframe_combo.setCurrentText("1m")
        self.timeframe_combo.setToolTip("Base timeframe for training")
        data_layout.addRow("Timeframe:", self.timeframe_combo)
        
        # Horizons (formato unificato x-ym/zm)
        self.horizons_edit = QLineEdit("1-10m/2m, 15m, 30m, 1h")
        self.horizons_edit.setToolTip(
            "Orizzonti di previsione - Formati supportati:\n"
            "• Lista semplice: '1m, 5m, 15m'\n"
            "• Range con step: '1-10m/2m' → 1m, 3m, 5m, 7m, 9m\n"
            "• Mix: '1-10m/2m, 15m, 30m, 1h'"
        )
        data_layout.addRow("Horizons:", self.horizons_edit)
        
        # Date range
        self.train_start_edit = QLineEdit("2019-01-01")
        self.train_start_edit.setToolTip("Training start date (YYYY-MM-DD)")
        data_layout.addRow("Train Start:", self.train_start_edit)
        
        self.train_end_edit = QLineEdit("2023-06-30")
        self.train_end_edit.setToolTip("Training end date (YYYY-MM-DD)")
        data_layout.addRow("Train End:", self.train_end_edit)
        
        self.val_start_edit = QLineEdit("2023-07-01")
        self.val_start_edit.setToolTip("Validation start date (YYYY-MM-DD)")
        data_layout.addRow("Val Start:", self.val_start_edit)
        
        self.val_end_edit = QLineEdit("2023-12-31")
        self.val_end_edit.setToolTip("Validation end date (YYYY-MM-DD)")
        data_layout.addRow("Val End:", self.val_end_edit)
        
        layout.addWidget(data_box)
        
        # Model Architecture
        model_box = QGroupBox("Model Architecture")
        model_layout = QFormLayout(model_box)
        
        # Encoder timeframes
        self.encoder_timeframes_edit = QLineEdit("1m, 5m, 15m")
        self.encoder_timeframes_edit.setToolTip(
            "Multi-scale encoder timeframes (comma-separated).\n"
            "SSSD usa S4 layers per catturare pattern su timeframe diversi.\n"
            "Default 1m: usa [1m, 5m, 15m] per intraday."
        )
        model_layout.addRow("Encoder Timeframes:", self.encoder_timeframes_edit)
        
        # S4 state dimension
        self.s4_state_dim_spinbox = QSpinBox()
        self.s4_state_dim_spinbox.setRange(32, 512)
        self.s4_state_dim_spinbox.setValue(64)
        self.s4_state_dim_spinbox.setToolTip("S4 state dimension (default: 64 per 1m)")
        model_layout.addRow("S4 State Dim:", self.s4_state_dim_spinbox)
        
        # S4 layers
        self.s4_layers_spinbox = QSpinBox()
        self.s4_layers_spinbox.setRange(1, 10)
        self.s4_layers_spinbox.setValue(3)
        self.s4_layers_spinbox.setToolTip("Number of S4 layers (default: 3 per 1m)")
        model_layout.addRow("S4 Layers:", self.s4_layers_spinbox)
        
        # S4 dropout
        self.s4_dropout_spinbox = QDoubleSpinBox()
        self.s4_dropout_spinbox.setRange(0.0, 0.5)
        self.s4_dropout_spinbox.setSingleStep(0.05)
        self.s4_dropout_spinbox.setValue(0.1)
        self.s4_dropout_spinbox.setToolTip("S4 dropout rate")
        model_layout.addRow("S4 Dropout:", self.s4_dropout_spinbox)
        
        # Diffusion steps (training)
        self.diffusion_steps_train_spinbox = QSpinBox()
        self.diffusion_steps_train_spinbox.setRange(100, 2000)
        self.diffusion_steps_train_spinbox.setValue(1000)
        self.diffusion_steps_train_spinbox.setToolTip("Diffusion steps during training (default: 1000)")
        model_layout.addRow("Diffusion Steps (Train):", self.diffusion_steps_train_spinbox)
        
        # Diffusion schedule
        self.diffusion_schedule_combo = QComboBox()
        self.diffusion_schedule_combo.addItems(["cosine", "linear", "quadratic"])
        self.diffusion_schedule_combo.setCurrentText("cosine")
        self.diffusion_schedule_combo.setToolTip("Noise schedule (cosine recommended)")
        model_layout.addRow("Diffusion Schedule:", self.diffusion_schedule_combo)
        
        layout.addWidget(model_box)
        
        # Training Settings
        training_box = QGroupBox("Training Settings")
        training_layout = QFormLayout(training_box)
        
        # Epochs
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 500)
        self.epochs_spinbox.setValue(50)
        self.epochs_spinbox.setToolTip("Number of training epochs (default: 50 per 1m)")
        training_layout.addRow("Epochs:", self.epochs_spinbox)
        
        # Batch size
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(8, 512)
        self.batch_size_spinbox.setValue(128)
        self.batch_size_spinbox.setToolTip("Batch size (default: 128 per 1m con GPU)")
        training_layout.addRow("Batch Size:", self.batch_size_spinbox)
        
        # Learning rate
        self.learning_rate_spinbox = QDoubleSpinBox()
        self.learning_rate_spinbox.setRange(0.00001, 0.01)
        self.learning_rate_spinbox.setSingleStep(0.00001)
        self.learning_rate_spinbox.setValue(0.0002)
        self.learning_rate_spinbox.setDecimals(5)
        self.learning_rate_spinbox.setToolTip("Learning rate (default: 0.0002 per 1m)")
        training_layout.addRow("Learning Rate:", self.learning_rate_spinbox)
        
        # Weight decay
        self.weight_decay_spinbox = QDoubleSpinBox()
        self.weight_decay_spinbox.setRange(0.0, 0.1)
        self.weight_decay_spinbox.setSingleStep(0.001)
        self.weight_decay_spinbox.setValue(0.01)
        self.weight_decay_spinbox.setDecimals(3)
        self.weight_decay_spinbox.setToolTip("Weight decay for AdamW optimizer")
        training_layout.addRow("Weight Decay:", self.weight_decay_spinbox)
        
        # Gradient clip norm
        self.grad_clip_spinbox = QDoubleSpinBox()
        self.grad_clip_spinbox.setRange(0.0, 10.0)
        self.grad_clip_spinbox.setSingleStep(0.1)
        self.grad_clip_spinbox.setValue(1.0)
        self.grad_clip_spinbox.setToolTip("Gradient clipping norm (0 = disabled)")
        training_layout.addRow("Gradient Clip:", self.grad_clip_spinbox)
        
        # Early stopping
        self.early_stopping_cb = QCheckBox("Enable Early Stopping")
        self.early_stopping_cb.setChecked(True)
        self.early_stopping_cb.setToolTip("Stop training when validation loss stops improving")
        training_layout.addRow(self.early_stopping_cb)
        
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setRange(1, 100)
        self.patience_spinbox.setValue(15)
        self.patience_spinbox.setToolTip("Patience for early stopping (epochs)")
        training_layout.addRow("Patience:", self.patience_spinbox)
        
        layout.addWidget(training_box)
        
        # NVIDIA Optimizations
        nvidia_box = QGroupBox("NVIDIA Optimizations")
        nvidia_layout = QFormLayout(nvidia_box)
        
        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cuda:0", "cuda:1", "cpu"])
        self.device_combo.setCurrentText("cuda")
        self.device_combo.setToolTip("Device for training")
        nvidia_layout.addRow("Device:", self.device_combo)
        
        # Mixed precision
        self.mixed_precision_cb = QCheckBox("Enable Mixed Precision (AMP)")
        self.mixed_precision_cb.setChecked(True)
        self.mixed_precision_cb.setToolTip(
            "Automatic Mixed Precision per training veloce.\n"
            "Usa FP16 dove possibile per ridurre VRAM e accelerare.\n"
            "Richiede GPU NVIDIA con Tensor Cores (RTX/V100+)"
        )
        nvidia_layout.addRow(self.mixed_precision_cb)
        
        layout.addWidget(nvidia_box)
        
        # Output Settings
        output_box = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_box)
        
        # Output directory
        output_path_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit("artifacts/sssd/checkpoints")
        self.output_dir_edit.setToolTip("Output directory for checkpoints")
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)
        
        output_path_layout.addWidget(self.output_dir_edit)
        output_path_layout.addWidget(browse_btn)
        output_layout.addRow("Output Directory:", output_path_layout)
        
        layout.addWidget(output_box)
        
        # Control Buttons
        button_box = QGroupBox("Control")
        button_layout = QHBoxLayout(button_box)
        
        self.start_training_btn = QPushButton("🚀 Start Training")
        self.start_training_btn.clicked.connect(self._start_training)
        
        self.stop_training_btn = QPushButton("⏹️ Stop")
        self.stop_training_btn.clicked.connect(self._stop_training)
        self.stop_training_btn.setEnabled(False)
        
        self.save_settings_btn = QPushButton("💾 Save Settings")
        self.save_settings_btn.clicked.connect(self._save_settings)
        self.save_settings_btn.setToolTip("Save settings to configs/sssd_settings.json")
        
        button_layout.addWidget(self.start_training_btn)
        button_layout.addWidget(self.stop_training_btn)
        button_layout.addWidget(self.save_settings_btn)
        button_layout.addStretch()
        
        layout.addWidget(button_box)
        
        # Progress
        progress_box = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_box)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_box)
        
        # Log output
        log_box = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_box)
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        log_layout.addWidget(self.log_view)
        
        layout.addWidget(log_box)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        root.addWidget(scroll)
    
    def _load_settings(self):
        """Load SSSD training settings from configs/sssd_settings.json"""
        try:
            sssd_config_file = Path(__file__).resolve().parents[3] / "configs" / "sssd_settings.json"
            
            if not sssd_config_file.exists():
                return
            
            with open(sssd_config_file, 'r', encoding='utf-8') as f:
                sssd_config = json.load(f)
            
            training_settings = sssd_config.get("training", {})
            
            # Load settings into widgets
            if "horizons" in training_settings:
                self.horizons_edit.setText(training_settings["horizons"])
            if "encoder_timeframes" in training_settings:
                self.encoder_timeframes_edit.setText(", ".join(training_settings["encoder_timeframes"]))
            if "s4_state_dim" in training_settings:
                self.s4_state_dim_spinbox.setValue(training_settings["s4_state_dim"])
            if "s4_n_layers" in training_settings:
                self.s4_layers_spinbox.setValue(training_settings["s4_n_layers"])
            if "s4_dropout" in training_settings:
                self.s4_dropout_spinbox.setValue(training_settings["s4_dropout"])
            if "steps_train" in training_settings:
                self.diffusion_steps_train_spinbox.setValue(training_settings["steps_train"])
            if "schedule" in training_settings:
                self.diffusion_schedule_combo.setCurrentText(training_settings["schedule"])
            if "epochs" in training_settings:
                self.epochs_spinbox.setValue(training_settings["epochs"])
            if "batch_size" in training_settings:
                self.batch_size_spinbox.setValue(training_settings["batch_size"])
            if "learning_rate" in training_settings:
                self.learning_rate_spinbox.setValue(training_settings["learning_rate"])
            if "weight_decay" in training_settings:
                self.weight_decay_spinbox.setValue(training_settings["weight_decay"])
            if "gradient_clip_norm" in training_settings:
                self.grad_clip_spinbox.setValue(training_settings["gradient_clip_norm"])
            if "early_stopping_enabled" in training_settings:
                self.early_stopping_cb.setChecked(training_settings["early_stopping_enabled"])
            if "patience" in training_settings:
                self.patience_spinbox.setValue(training_settings["patience"])
            if "device" in training_settings:
                self.device_combo.setCurrentText(training_settings["device"])
            if "mixed_precision_enabled" in training_settings:
                self.mixed_precision_cb.setChecked(training_settings["mixed_precision_enabled"])
            if "output_dir" in training_settings:
                self.output_dir_edit.setText(training_settings["output_dir"])
                
            logger.info(f"SSSD training settings loaded from {sssd_config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load SSSD training settings: {e}")
    
    def _save_settings(self):
        """Save SSSD training settings to configs/sssd_settings.json"""
        try:
            sssd_config_file = Path(__file__).resolve().parents[3] / "configs" / "sssd_settings.json"
            
            # Load existing or create new
            if sssd_config_file.exists():
                with open(sssd_config_file, 'r', encoding='utf-8') as f:
                    sssd_config = json.load(f)
            else:
                sssd_config = {"inference": {}, "training": {}, "backtesting": {}}
            
            # Parse encoder timeframes
            encoder_tfs = [tf.strip() for tf in self.encoder_timeframes_edit.text().split(',')]
            
            # Update training settings
            sssd_config["training"].update({
                "horizons": self.horizons_edit.text().strip(),
                "encoder_timeframes": encoder_tfs,
                "s4_state_dim": self.s4_state_dim_spinbox.value(),
                "s4_n_layers": self.s4_layers_spinbox.value(),
                "s4_dropout": self.s4_dropout_spinbox.value(),
                "steps_train": self.diffusion_steps_train_spinbox.value(),
                "schedule": self.diffusion_schedule_combo.currentText(),
                "epochs": self.epochs_spinbox.value(),
                "batch_size": self.batch_size_spinbox.value(),
                "learning_rate": self.learning_rate_spinbox.value(),
                "weight_decay": self.weight_decay_spinbox.value(),
                "gradient_clip_norm": self.grad_clip_spinbox.value(),
                "early_stopping_enabled": self.early_stopping_cb.isChecked(),
                "patience": self.patience_spinbox.value(),
                "device": self.device_combo.currentText(),
                "mixed_precision_enabled": self.mixed_precision_cb.isChecked(),
                "output_dir": self.output_dir_edit.text().strip(),
            })
            
            # Save to file
            sssd_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sssd_config_file, 'w', encoding='utf-8') as f:
                json.dump(sssd_config, f, indent=2)
            
            logger.info(f"SSSD training settings saved to {sssd_config_file}")
            QMessageBox.information(
                self,
                "Settings Saved",
                f"SSSD training settings saved to:\n{sssd_config_file}"
            )
            
        except Exception as e:
            logger.exception(f"Failed to save SSSD training settings: {e}")
            QMessageBox.warning(
                self,
                "Save Failed",
                f"Failed to save SSSD training settings:\n{str(e)}"
            )
    
    def _browse_output_dir(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir_edit.text()
        )
        
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def _start_training(self):
        """Start SSSD training"""
        try:
            # Validate horizons
            horizons_str = self.horizons_edit.text().strip()
            if not horizons_str:
                QMessageBox.warning(self, "Invalid Horizons", "Please specify forecast horizons.")
                return
            
            # Collect parameters
            params = {
                'model_type': 'sssd',
                'symbol': self.symbol_combo.currentText(),
                'timeframe': self.timeframe_combo.currentText(),
                'horizons': horizons_str,
                'train_start': self.train_start_edit.text().strip(),
                'train_end': self.train_end_edit.text().strip(),
                'val_start': self.val_start_edit.text().strip(),
                'val_end': self.val_end_edit.text().strip(),
                'encoder_timeframes': [tf.strip() for tf in self.encoder_timeframes_edit.text().split(',')],
                's4_state_dim': self.s4_state_dim_spinbox.value(),
                's4_n_layers': self.s4_layers_spinbox.value(),
                's4_dropout': self.s4_dropout_spinbox.value(),
                'diffusion_steps_train': self.diffusion_steps_train_spinbox.value(),
                'diffusion_schedule': self.diffusion_schedule_combo.currentText(),
                'epochs': self.epochs_spinbox.value(),
                'batch_size': self.batch_size_spinbox.value(),
                'learning_rate': self.learning_rate_spinbox.value(),
                'weight_decay': self.weight_decay_spinbox.value(),
                'gradient_clip_norm': self.grad_clip_spinbox.value(),
                'early_stopping_enabled': self.early_stopping_cb.isChecked(),
                'patience': self.patience_spinbox.value(),
                'device': self.device_combo.currentText(),
                'mixed_precision_enabled': self.mixed_precision_cb.isChecked(),
                'output_dir': self.output_dir_edit.text().strip(),
            }
            
            # Build command
            import sys
            cmd = [
                sys.executable, '-m', 'forex_diffusion.training.train_sssd',
                '--symbol', params['symbol'].replace('/', ''),  # EURUSD
                '--timeframe', params['timeframe'],
                '--horizons', params['horizons'],
                '--train_start', params['train_start'],
                '--train_end', params['train_end'],
                '--val_start', params['val_start'],
                '--val_end', params['val_end'],
                '--encoder_timeframes', ','.join(params['encoder_timeframes']),
                '--s4_state_dim', str(params['s4_state_dim']),
                '--s4_n_layers', str(params['s4_n_layers']),
                '--s4_dropout', str(params['s4_dropout']),
                '--diffusion_steps_train', str(params['diffusion_steps_train']),
                '--diffusion_schedule', params['diffusion_schedule'],
                '--epochs', str(params['epochs']),
                '--batch_size', str(params['batch_size']),
                '--learning_rate', str(params['learning_rate']),
                '--weight_decay', str(params['weight_decay']),
                '--gradient_clip_norm', str(params['gradient_clip_norm']),
                '--patience', str(params['patience']),
                '--device', params['device'],
                '--output_dir', params['output_dir'],
            ]
            
            if params['early_stopping_enabled']:
                cmd.append('--early_stopping')
            
            if params['mixed_precision_enabled']:
                cmd.append('--mixed_precision')
            
            # Log command
            self.log_view.clear()
            self.log_view.append(f"[INFO] Starting SSSD training")
            self.log_view.append(f"[CMD] {' '.join(cmd)}\n")
            
            # Disable start, enable stop
            self.start_training_btn.setEnabled(False)
            self.stop_training_btn.setEnabled(True)
            self.status_label.setText("Training started...")
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # TODO: Launch training process (needs worker implementation)
            self.log_view.append(
                "[WARNING] SSSD training worker not yet implemented.\n"
                "[INFO] Please run the command manually in terminal for now.\n"
            )
            
            # Emit signal
            logger.info(f"SSSD training requested: {params}")
            self.trainingStarted.emit(params)
            
            # Auto re-enable for now (until worker is implemented)
            QMessageBox.information(
                self,
                "Training Command Ready",
                "SSSD training command generated.\n\n"
                "Please copy the command from the log and run manually:\n\n"
                f"{' '.join(cmd[:5])} ..."
            )
            self.start_training_btn.setEnabled(True)
            self.stop_training_btn.setEnabled(False)
            
        except Exception as e:
            logger.exception(f"Failed to start SSSD training: {e}")
            QMessageBox.critical(self, "Training Error", f"Failed to start training:\n{str(e)}")
            self.start_training_btn.setEnabled(True)
            self.stop_training_btn.setEnabled(False)
    
    def _stop_training(self):
        """Stop SSSD training"""
        self.status_label.setText("Stopping training...")
        self.stop_training_btn.setEnabled(False)
        # TODO: Implement training stop logic
        logger.info("SSSD training stop requested")
