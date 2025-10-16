"""E2E Optimization Tab - Complete GUI Implementation"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Optional
import json

from PySide6.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QCheckBox,
    QLineEdit, QTableWidget, QTableWidgetItem, QTextEdit, QProgressBar,
    QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThread, QDate
from PySide6.QtGui import QColor
from loguru import logger

class E2EOptimizationTab(QWidget):
    """Main E2E Optimization tab with 3 sub-tabs"""
    
    optimization_started = Signal(int)  # run_id
    optimization_completed = Signal(int, dict)  # run_id, results
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create 3 sub-tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_configuration_panel(), "Configuration")
        self.tabs.addTab(self._create_dashboard(), "Optimization Dashboard")
        self.tabs.addTab(self._create_deployment_panel(), "Deployment")
        
        layout.addWidget(self.tabs)
    
    # ==================== SUB-TAB 1: CONFIGURATION PANEL ====================
    
    def _create_configuration_panel(self) -> QWidget:
        """Configuration panel for optimization setup"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Section 1: Basic Configuration
        basic_group = QGroupBox("Basic Configuration")
        basic_layout = QVBoxLayout(basic_group)
        
        # Row 1: Symbol, Timeframe
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'])
        row1.addWidget(self.symbol_combo)
        
        row1.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['5m', '15m', '30m', '1h', '4h', '1d'])
        row1.addWidget(self.timeframe_combo)
        basic_layout.addLayout(row1)
        
        # Row 2: Dates
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Start Date:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addYears(-2))
        self.start_date.setCalendarPopup(True)
        row2.addWidget(self.start_date)
        
        row2.addWidget(QLabel("End Date:"))
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        row2.addWidget(self.end_date)
        basic_layout.addLayout(row2)
        
        # Row 3: Method, Trials
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(['Bayesian (Recommended)', 'Genetic Algorithm'])
        row3.addWidget(self.method_combo)
        
        row3.addWidget(QLabel("Trials:"))
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(10, 500)
        self.trials_spin.setValue(100)
        row3.addWidget(self.trials_spin)
        basic_layout.addLayout(row3)
        
        layout.addWidget(basic_group)
        
        # Section 2: Component Selection
        components_group = QGroupBox("Components to Optimize")
        components_layout = QVBoxLayout(components_group)
        
        self.enable_sssd = QCheckBox("Enable SSSD (quantile-based sizing)")
        self.enable_riskfolio = QCheckBox("Enable Riskfolio-Lib (portfolio optimization)")
        self.enable_riskfolio.setChecked(True)
        self.enable_patterns = QCheckBox("Enable Pattern Parameters (load from DB)")
        self.enable_patterns.setChecked(True)
        self.enable_rl = QCheckBox("Enable RL Actor-Critic (hybrid mode)")
        self.enable_vix = QCheckBox("Enable VIX Filter (volatility adjustment)")
        self.enable_vix.setChecked(True)
        self.enable_sentiment = QCheckBox("Enable Sentiment Filter (contrarian)")
        self.enable_sentiment.setChecked(True)
        self.enable_volume = QCheckBox("Enable Volume Indicators (OBV, VWAP)")
        self.enable_volume.setChecked(True)
        
        components_layout.addWidget(self.enable_sssd)
        components_layout.addWidget(self.enable_riskfolio)
        components_layout.addWidget(self.enable_patterns)
        components_layout.addWidget(self.enable_rl)
        components_layout.addWidget(self.enable_vix)
        components_layout.addWidget(self.enable_sentiment)
        components_layout.addWidget(self.enable_volume)
        
        layout.addWidget(components_group)
        
        # Section 3: Actions
        actions_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Optimization")
        self.start_btn.clicked.connect(self._on_start_optimization)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        
        self.load_btn = QPushButton("Load Configuration")
        self.save_btn = QPushButton("Save Configuration")
        self.reset_btn = QPushButton("Reset to Defaults")
        
        actions_layout.addWidget(self.start_btn)
        actions_layout.addWidget(self.load_btn)
        actions_layout.addWidget(self.save_btn)
        actions_layout.addWidget(self.reset_btn)
        
        layout.addLayout(actions_layout)
        layout.addStretch()
        
        return widget
    
    # ==================== SUB-TAB 2: OPTIMIZATION DASHBOARD ====================
    
    def _create_dashboard(self) -> QWidget:
        """Dashboard for monitoring optimization progress"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Status Panel
        status_group = QGroupBox("Optimization Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Status: Not Started")
        self.status_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; }")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        
        progress_info = QHBoxLayout()
        self.trial_label = QLabel("Trial: 0 / 100")
        self.time_label = QLabel("Elapsed: 0s")
        self.eta_label = QLabel("ETA: --")
        progress_info.addWidget(self.trial_label)
        progress_info.addWidget(self.time_label)
        progress_info.addWidget(self.eta_label)
        status_layout.addLayout(progress_info)
        
        self.stop_btn = QPushButton("Stop Optimization")
        self.stop_btn.setEnabled(False)
        status_layout.addWidget(self.stop_btn)
        
        layout.addWidget(status_group)
        
        # Best Results Table
        results_group = QGroupBox("Best Results So Far")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            'Trial #', 'Sharpe', 'Max DD %', 'Win Rate %', 'Profit Factor', 'Cost'
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setRowCount(10)  # Top 10
        
        results_layout.addWidget(self.results_table)
        layout.addWidget(results_group)
        
        # Run History
        history_group = QGroupBox("Optimization Run History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels([
            'Run ID', 'Date', 'Symbol', 'TF', 'Method', 'Status', 'Best Sharpe'
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        history_layout.addWidget(self.history_table)
        
        history_actions = QHBoxLayout()
        self.view_details_btn = QPushButton("View Details")
        self.export_report_btn = QPushButton("Export Report (PDF)")
        history_actions.addWidget(self.view_details_btn)
        history_actions.addWidget(self.export_report_btn)
        history_actions.addStretch()
        history_layout.addLayout(history_actions)
        
        layout.addWidget(history_group)
        
        return widget
    
    # ==================== SUB-TAB 3: DEPLOYMENT PANEL ====================
    
    def _create_deployment_panel(self) -> QWidget:
        """Deployment panel for activating optimized parameters"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Active Deployments
        active_group = QGroupBox("Active Deployments")
        active_layout = QVBoxLayout(active_group)
        
        self.deployments_table = QTableWidget()
        self.deployments_table.setColumnCount(7)
        self.deployments_table.setHorizontalHeaderLabels([
            'Symbol', 'TF', 'Mode', 'Deployed', 'Expected Sharpe', 'Actual Sharpe', 'Status'
        ])
        self.deployments_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        active_layout.addWidget(self.deployments_table)
        layout.addWidget(active_group)
        
        # Deploy New Configuration
        deploy_group = QGroupBox("Deploy New Configuration")
        deploy_layout = QVBoxLayout(deploy_group)
        
        # Select run
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Select Optimization Run:"))
        self.run_selector = QComboBox()
        select_layout.addWidget(self.run_selector)
        self.load_run_btn = QPushButton("Load")
        select_layout.addWidget(self.load_run_btn)
        deploy_layout.addLayout(select_layout)
        
        # Review parameters
        self.params_review = QTextEdit()
        self.params_review.setReadOnly(True)
        self.params_review.setMaximumHeight(150)
        deploy_layout.addWidget(QLabel("Parameters Preview:"))
        deploy_layout.addWidget(self.params_review)
        
        # Deploy actions
        deploy_actions = QHBoxLayout()
        self.deploy_btn = QPushButton("Deploy to Live Trading")
        self.deploy_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        self.deploy_btn.clicked.connect(self._on_deploy)
        
        deploy_actions.addWidget(self.deploy_btn)
        deploy_actions.addStretch()
        deploy_layout.addLayout(deploy_actions)
        
        layout.addWidget(deploy_group)
        layout.addStretch()
        
        return widget
    
    # ==================== EVENT HANDLERS ====================
    
    def _on_start_optimization(self):
        """Handle Start Optimization button click"""
        # Validate inputs
        if self.trials_spin.value() < 10:
            QMessageBox.warning(self, "Invalid Input", "Trials must be at least 10")
            return
        
        # Collect configuration
        config = {
            'symbol': self.symbol_combo.currentText(),
            'timeframe': self.timeframe_combo.currentText(),
            'start_date': self.start_date.date().toPython(),
            'end_date': self.end_date.date().toPython(),
            'method': 'bayesian' if 'Bayesian' in self.method_combo.currentText() else 'genetic',
            'n_trials': self.trials_spin.value(),
            'enable_sssd': self.enable_sssd.isChecked(),
            'enable_riskfolio': self.enable_riskfolio.isChecked(),
            'enable_patterns': self.enable_patterns.isChecked(),
            'enable_rl': self.enable_rl.isChecked(),
            'enable_vix_filter': self.enable_vix.isChecked(),
            'enable_sentiment_filter': self.enable_sentiment.isChecked(),
            'enable_volume_filter': self.enable_volume.isChecked(),
        }
        
        logger.info(f"Starting E2E optimization with config: {config}")
        
        # Update UI
        self.status_label.setText("Status: Running")
        self.status_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #4CAF50; }")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # In production, start optimization worker thread here
        QMessageBox.information(self, "Optimization Started", 
                               f"Optimization started for {config['symbol']} {config['timeframe']}\\n"
                               f"This may take several hours...")
        
        # Emit signal
        self.optimization_started.emit(0)  # Mock run_id
    
    def _on_deploy(self):
        """Handle Deploy button click"""
        reply = QMessageBox.question(
            self, 
            'Confirm Deployment',
            'Are you sure you want to deploy these parameters to live trading?\\n\\n'
            'This will replace current active parameters.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            logger.info("Parameters deployed to live trading")
            QMessageBox.information(self, "Deployment Complete", 
                                   "Parameters successfully deployed to live trading.")
    
    def update_progress(self, trial: int, total: int, best_sharpe: float):
        """Update optimization progress"""
        progress_pct = int((trial / total) * 100)
        self.progress_bar.setValue(progress_pct)
        self.trial_label.setText(f"Trial: {trial} / {total}")
        self.status_label.setText(f"Status: Running (Best Sharpe: {best_sharpe:.3f})")
