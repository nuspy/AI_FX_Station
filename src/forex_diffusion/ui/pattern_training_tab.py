"""
Pattern Training/Backtest Tab - Genetic Algorithm Optimization for Chart and Candlestick Patterns

This tab provides a comprehensive interface for training and optimizing pattern detection parameters
using genetic algorithms. It supports multi-objective optimization (profit vs risk) and handles
both chart patterns and candlestick patterns.

Restored from commit 11d3627: Training parametri patterns
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QGroupBox, QScrollArea, QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QRadioButton, QButtonGroup, QComboBox, QProgressBar, QTabWidget,
    QDateEdit, QFrame, QSlider, QMessageBox, QGridLayout
)
from PySide6.QtCore import Qt, QDate, QTimer
from PySide6.QtGui import QTextCursor
from loguru import logger


class PatternTrainingTab(QWidget):
    """
    Standalone tab for pattern training and optimization.

    Features:
    - Chart pattern vs Candlestick pattern training selection
    - Multi-objective optimization (D1: profit, D2: risk)
    - Genetic algorithm with customizable parameters
    - Real-time progress tracking and time estimates
    - Parameter promotion and rollback
    - Comprehensive results analysis
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._setup_timers()
        self._init_state()

    def _init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create scrollable area for all the controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)

        # === Study Setup Section ===
        self._create_study_setup_section(scroll_layout)

        # === Dataset Configuration Section ===
        self._create_dataset_config_section(scroll_layout)

        # === Parameter Space Section ===
        self._create_parameter_space_section(scroll_layout)

        # === Optimization Configuration Section ===
        self._create_optimization_config_section(scroll_layout)

        # === Execution Control Section ===
        self._create_execution_control_section(scroll_layout)

        # === Results and Status Section ===
        self._create_results_section(scroll_layout)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        logger.info("PatternTrainingTab initialized")

    def _init_state(self):
        """Initialize internal state variables"""
        self.is_training = False
        self.is_paused = False
        self.current_task = None
        self.training_start_time = None
        self.best_params_cache = {}

    def _setup_timers(self):
        """Setup auto-refresh timer for progress updates"""
        self.progress_refresh_timer = QTimer()
        self.progress_refresh_timer.timeout.connect(self._refresh_progress)
        # Timer will be started when training begins

    # === UI CREATION METHODS (from commit 11d3627) ===

    def _create_study_setup_section(self, layout: QVBoxLayout) -> None:
        """Create study setup section with separate chart/candlestick pattern training"""
        group = QGroupBox("Training Setup")
        group_layout = QVBoxLayout(group)

        # Training Type Selection
        training_type_frame = QFrame()
        training_type_layout = QHBoxLayout(training_type_frame)

        self.training_type_group = QButtonGroup()
        self.chart_patterns_radio = QRadioButton("Train Chart Patterns")
        self.candlestick_patterns_radio = QRadioButton("Train Candlestick Patterns")
        self.chart_patterns_radio.setChecked(True)

        self.training_type_group.addButton(self.chart_patterns_radio, 0)
        self.training_type_group.addButton(self.candlestick_patterns_radio, 1)

        training_type_layout.addWidget(self.chart_patterns_radio)
        training_type_layout.addWidget(self.candlestick_patterns_radio)
        training_type_layout.addStretch()

        group_layout.addWidget(QLabel("Training Type:"))
        group_layout.addWidget(training_type_frame)

        # Training Period
        period_frame = QFrame()
        period_frame.setFrameStyle(QFrame.StyledPanel)
        period_layout = QFormLayout(period_frame)

        self.training_start_date = QDateEdit()
        self.training_start_date.setDate(QDate(2020, 1, 1))
        self.training_start_date.setCalendarPopup(True)
        self.training_start_date.setToolTip("Inizio periodo di addestramento")
        period_layout.addRow("Training Start Date:", self.training_start_date)

        self.training_end_date = QDateEdit()
        self.training_end_date.setDate(QDate(2024, 12, 31))
        self.training_end_date.setCalendarPopup(True)
        self.training_end_date.setToolTip("Fine periodo di addestramento")
        period_layout.addRow("Training End Date:", self.training_end_date)

        group_layout.addWidget(QLabel("Training Data Period:"))
        group_layout.addWidget(period_frame)

        # Asset selection
        self.assets_edit = QTextEdit()
        self.assets_edit.setMaximumHeight(60)
        self.assets_edit.setPlainText("EUR/USD, GBP/USD, USD/JPY")
        self.assets_edit.setToolTip("Assets to train (comma separated)")
        group_layout.addWidget(QLabel("Assets (comma separated):"))
        group_layout.addWidget(self.assets_edit)

        # Timeframe selection
        self.timeframes_edit = QTextEdit()
        self.timeframes_edit.setMaximumHeight(60)
        self.timeframes_edit.setPlainText("1h, 4h, 1d")
        self.timeframes_edit.setToolTip("Timeframes to optimize")
        group_layout.addWidget(QLabel("Timeframes (comma separated):"))
        group_layout.addWidget(self.timeframes_edit)

        # Info
        info_label = QLabel(
            "Training will automatically include:\n"
            "• Both BULL and BEAR directions\n"
            "• ALL regime filter combinations\n"
            "• ALL patterns in selected family\n"
            "• Genetic algorithm optimization"
        )
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 10px; background: #f0f0f0; border-radius: 5px;")
        info_label.setWordWrap(True)
        group_layout.addWidget(info_label)

        layout.addWidget(group)

    def _create_dataset_config_section(self, layout: QVBoxLayout) -> None:
        """Create optimization targets configuration section"""
        group = QGroupBox("Optimization Targets (D1/D2)")
        group_layout = QVBoxLayout(group)

        # Placeholder
        placeholder = QLabel("Multi-Objective Optimization Configuration - Implementation in progress...")
        placeholder.setStyleSheet("color: #666; font-style: italic;")
        group_layout.addWidget(placeholder)

        layout.addWidget(group)

    def _create_parameter_space_section(self, layout: QVBoxLayout) -> None:
        """Create parameter space configuration section with detailed tooltips"""
        group = QGroupBox("Parameter Space Configuration")
        group_layout = QVBoxLayout(group)

        # Placeholder
        placeholder = QLabel("Parameter Space - Implementation in progress...")
        placeholder.setStyleSheet("color: #666; font-style: italic;")
        group_layout.addWidget(placeholder)

        layout.addWidget(group)

    def _create_optimization_config_section(self, layout: QVBoxLayout) -> None:
        """Create genetic algorithm configuration section"""
        group = QGroupBox("Genetic Algorithm Configuration")
        group_layout = QVBoxLayout(group)

        # Placeholder
        placeholder = QLabel("GA Configuration - Implementation in progress...")
        placeholder.setStyleSheet("color: #666; font-style: italic;")
        group_layout.addWidget(placeholder)

        layout.addWidget(group)

    def _create_execution_control_section(self, layout: QVBoxLayout) -> None:
        """Create execution control section with start/pause/resume/stop buttons"""
        group = QGroupBox("Training Execution")
        group_layout = QVBoxLayout(group)

        # Control buttons
        control_buttons_layout = QHBoxLayout()

        self.start_training_btn = QPushButton("▶️ Start Training")
        self.start_training_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        self.start_training_btn.clicked.connect(self._start_optimization)
        control_buttons_layout.addWidget(self.start_training_btn)

        self.train_vae_btn = QPushButton("🧠 Train Pattern VAE")
        self.train_vae_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 10px; }")
        self.train_vae_btn.clicked.connect(self._train_pattern_vae)
        self.train_vae_btn.setToolTip("Train ML-based pattern detection (Variational Autoencoder)")
        control_buttons_layout.addWidget(self.train_vae_btn)

        self.pause_training_btn = QPushButton("⏸️ Pause")
        self.pause_training_btn.setEnabled(False)
        self.pause_training_btn.clicked.connect(self._pause_optimization)
        control_buttons_layout.addWidget(self.pause_training_btn)

        self.resume_training_btn = QPushButton("▶️ Resume")
        self.resume_training_btn.setEnabled(False)
        self.resume_training_btn.clicked.connect(self._resume_optimization)
        control_buttons_layout.addWidget(self.resume_training_btn)

        self.stop_optimization_btn = QPushButton("⏹️ Stop")
        self.stop_optimization_btn.setEnabled(False)
        self.stop_optimization_btn.clicked.connect(self._stop_optimization)
        control_buttons_layout.addWidget(self.stop_optimization_btn)

        control_buttons_layout.addStretch()
        group_layout.addLayout(control_buttons_layout)

        # Status display
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_layout = QFormLayout(status_frame)

        self.optimization_status_label = QLabel("Status: Ready")
        self.optimization_status_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; }")
        status_layout.addRow("Current Status:", self.optimization_status_label)

        self.current_pattern_label = QLabel("-")
        status_layout.addRow("Current Pattern:", self.current_pattern_label)

        self.current_asset_label = QLabel("-")
        status_layout.addRow("Current Asset/TF:", self.current_asset_label)

        group_layout.addWidget(status_frame)

        # Progress bars (initially hidden)
        progress_frame = QFrame()
        progress_frame.setFrameStyle(QFrame.StyledPanel)
        progress_layout = QVBoxLayout(progress_frame)

        self.overall_progress = QProgressBar()
        self.overall_progress.setFormat("Overall: %p% (%v/%m patterns)")
        progress_layout.addWidget(self.overall_progress)

        self.pattern_progress = QProgressBar()
        self.pattern_progress.setFormat("Current Pattern: %p% (%v/%m trials)")
        progress_layout.addWidget(self.pattern_progress)

        progress_frame.setVisible(False)
        self.progress_frame = progress_frame
        group_layout.addWidget(progress_frame)

        layout.addWidget(group)

    def _create_results_section(self, layout: QVBoxLayout) -> None:
        """Create results and status section"""
        group = QGroupBox("Results and Analysis")
        group_layout = QVBoxLayout(group)

        # Results tabs
        self.results_tabs = QTabWidget()

        # Status tab
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)

        self.study_status_text = QTextEdit()
        self.study_status_text.setMaximumHeight(150)
        self.study_status_text.setReadOnly(True)
        self.study_status_text.setPlainText("No optimization running...")
        status_layout.addWidget(self.study_status_text)

        self.refresh_status_btn = QPushButton("Refresh Status")
        self.refresh_status_btn.clicked.connect(self._refresh_status)
        status_layout.addWidget(self.refresh_status_btn)

        self.results_tabs.addTab(status_tab, "Status")

        # Best parameters tab
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)

        self.best_params_text = QTextEdit()
        self.best_params_text.setReadOnly(True)
        self.best_params_text.setPlainText("No results yet...")
        params_layout.addWidget(self.best_params_text)

        promote_layout = QHBoxLayout()
        self.promote_params_btn = QPushButton("Promote Parameters")
        self.promote_params_btn.clicked.connect(self._promote_parameters)
        self.promote_params_btn.setEnabled(False)
        promote_layout.addWidget(self.promote_params_btn)

        self.rollback_params_btn = QPushButton("Rollback Parameters")
        self.rollback_params_btn.clicked.connect(self._rollback_parameters)
        promote_layout.addWidget(self.rollback_params_btn)

        promote_layout.addStretch()
        params_layout.addLayout(promote_layout)

        self.results_tabs.addTab(params_tab, "Best Parameters")

        # Performance breakdown tab
        perf_tab = QWidget()
        perf_layout = QVBoxLayout(perf_tab)

        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        perf_layout.addWidget(self.performance_text)

        self.results_tabs.addTab(perf_tab, "Performance")

        group_layout.addWidget(self.results_tabs)

        layout.addWidget(group)

    # === TRAINING EXECUTION METHODS (placeholders - to be implemented) ===

    def _start_optimization(self):
        """Start pattern training optimization"""
        logger.info("Start training requested")
        QMessageBox.information(
            self,
            "Training",
            "Pattern training implementation will be restored from commit 11d3627.\n\n"
            "This will include:\n"
            "• Full UI controls for all parameters\n"
            "• Genetic algorithm optimization\n"
            "• Multi-objective optimization (D1/D2)\n"
            "• Real-time progress tracking\n"
            "• Results analysis and parameter management"
        )

    def _pause_optimization(self):
        """Pause ongoing optimization"""
        logger.info("Pause training requested")
        self.is_paused = True

    def _resume_optimization(self):
        """Resume paused optimization"""
        logger.info("Resume training requested")
        self.is_paused = False

    def _stop_optimization(self):
        """Stop optimization completely"""
        logger.info("Stop training requested")
        self.is_training = False
        self.is_paused = False

    def _refresh_progress(self):
        """Refresh progress display"""
        # TODO: Implement progress refresh
        pass

    def _refresh_status(self):
        """Refresh status display"""
        logger.info("Status refresh requested")
        self.study_status_text.append(f"Status refreshed at {QDate.currentDate().toString()}")

    def _promote_parameters(self):
        """Promote best parameters to production"""
        logger.info("Promote parameters requested")
        QMessageBox.information(self, "Promote", "Parameter promotion will save best parameters to production config")

    def _rollback_parameters(self):
        """Rollback to previous parameters"""
        logger.info("Rollback parameters requested")
        QMessageBox.information(self, "Rollback", "Parameter rollback will restore previous production config")

    def _train_pattern_vae(self):
        """Train ML-based pattern detection using VAE"""
        from PySide6.QtWidgets import QInputDialog, QFileDialog
        from pathlib import Path
        import threading

        try:
            # Get dataset parameters
            symbol_text = self.dataset_symbols_edit.text()
            if not symbol_text:
                QMessageBox.warning(self, "Missing Input", "Please specify trading symbols")
                return

            symbols = [s.strip() for s in symbol_text.split(',')]

            timeframe = self.dataset_timeframe_combo.currentText()
            days_history = self.dataset_days_spin.value()

            # Ask for VAE training parameters
            latent_dim, ok1 = QInputDialog.getInt(
                self,
                "VAE Configuration",
                "Latent dimension (size of pattern encoding):",
                32, 8, 128, 8
            )
            if not ok1:
                return

            epochs, ok2 = QInputDialog.getInt(
                self,
                "VAE Configuration",
                "Training epochs:",
                50, 10, 200, 10
            )
            if not ok2:
                return

            batch_size, ok3 = QInputDialog.getInt(
                self,
                "VAE Configuration",
                "Batch size:",
                128, 32, 512, 32
            )
            if not ok3:
                return

            # Ask where to save model
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Pattern VAE Model",
                str(Path.home() / "pattern_vae.pt"),
                "PyTorch Models (*.pt *.pth);;All Files (*.*)"
            )

            if not save_path:
                return

            save_path = Path(save_path)

            # Clear status text
            self.study_status_text.clear()
            self.study_status_text.append("[VAE] Starting pattern VAE training...")
            self.study_status_text.append(f"[VAE] Symbols: {symbols}")
            self.study_status_text.append(f"[VAE] Timeframe: {timeframe}, Days: {days_history}")
            self.study_status_text.append(f"[VAE] Latent dim: {latent_dim}, Epochs: {epochs}, Batch size: {batch_size}")

            def run_vae_training():
                try:
                    import torch
                    import numpy as np
                    from ..data.data_loader import fetch_candles_from_db
                    from ..training.train import _add_time_features, CHANNEL_ORDER
                    from ..models.pattern_autoencoder import train_pattern_vae, PatternDetector

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.study_status_text.append(f"[VAE] Using device: {device}")

                    # Collect data from all symbols
                    all_sequences = []

                    for symbol in symbols:
                        self.study_status_text.append(f"[VAE] Loading data for {symbol}...")

                        df = fetch_candles_from_db(symbol, timeframe, days_history)
                        df = _add_time_features(df)

                        # Extract features (normalized)
                        values = df[CHANNEL_ORDER].values  # (N, C)

                        # Calculate normalization stats
                        mu = values.mean(axis=0)
                        sigma = values.std(axis=0)

                        # Create sliding windows (64-bar sequences)
                        sequence_length = 64
                        for i in range(sequence_length, len(values)):
                            seq = values[i - sequence_length:i].T  # (C, L)
                            seq_norm = (seq - mu[:, None]) / (sigma[:, None] + 1e-8)
                            all_sequences.append(seq_norm)

                    if not all_sequences:
                        self.study_status_text.append("[VAE] ERROR: No data collected")
                        return

                    # Convert to tensor
                    data = torch.from_numpy(np.array(all_sequences, dtype=np.float32))
                    self.study_status_text.append(f"[VAE] Collected {len(data)} sequences")

                    # Split train/val
                    val_size = int(len(data) * 0.2)
                    train_data = data[:-val_size]
                    val_data = data[-val_size:]

                    self.study_status_text.append(f"[VAE] Training: {len(train_data)}, Validation: {len(val_data)}")

                    # Train model
                    self.study_status_text.append("[VAE] Starting training...")

                    model = train_pattern_vae(
                        train_data=train_data,
                        val_data=val_data,
                        input_channels=len(CHANNEL_ORDER),
                        sequence_length=64,
                        latent_dim=latent_dim,
                        epochs=epochs,
                        batch_size=batch_size,
                        device=device
                    )

                    # Save model
                    torch.save(model.state_dict(), save_path)
                    self.study_status_text.append(f"[VAE] Model saved to: {save_path}")

                    # Calibrate detector on validation data
                    self.study_status_text.append("[VAE] Calibrating anomaly detector...")
                    detector = PatternDetector(model, device=device)
                    detector.calibrate(val_data[:500])  # Use first 500 validation samples

                    # Test pattern detection
                    self.study_status_text.append("[VAE] Testing pattern detection...")
                    result = detector.detect_patterns(val_data[500:600])

                    self.study_status_text.append(f"[VAE] Detected {result.n_anomalies} anomalies in 100 test samples")
                    self.study_status_text.append(f"[VAE] Anomaly rate: {result.n_anomalies/100*100:.1f}%")

                    # Cluster patterns
                    self.study_status_text.append("[VAE] Clustering patterns...")
                    labels = detector.cluster_patterns(val_data[:200], n_clusters=10)

                    cluster_counts = {}
                    for label in labels:
                        cluster_counts[label] = cluster_counts.get(label, 0) + 1

                    self.study_status_text.append(f"[VAE] Found {len(cluster_counts)} pattern clusters")
                    for cluster_id, count in sorted(cluster_counts.items()):
                        self.study_status_text.append(f"[VAE]   Cluster {cluster_id}: {count} patterns")

                    self.study_status_text.append("\n[VAE] ✅ Training complete!")

                    QMessageBox.information(
                        self,
                        "VAE Training Complete",
                        f"Pattern VAE training completed successfully!\n\n"
                        f"Model saved to:\n{save_path}\n\n"
                        f"Detected {result.n_anomalies} anomalies in test set\n"
                        f"Identified {len(cluster_counts)} distinct pattern clusters"
                    )

                except Exception as e:
                    logger.exception(f"VAE training failed: {e}")
                    self.study_status_text.append(f"\n[VAE] ❌ ERROR: {e}")
                    QMessageBox.critical(
                        self,
                        "VAE Training Error",
                        f"Training failed:\n{e}"
                    )

            # Start training thread
            thread = threading.Thread(target=run_vae_training, daemon=True)
            thread.start()

        except Exception as e:
            logger.exception(f"Failed to start VAE training: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start training:\n{e}")
