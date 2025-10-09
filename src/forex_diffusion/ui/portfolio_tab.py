"""
Portfolio Optimization Tab for GUI.

Provides interface for configuring and visualizing portfolio optimization
using Riskfolio-Lib.
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QTableWidget,
    QTableWidgetItem, QCheckBox, QTabWidget, QTextEdit,
)
from PySide6.QtCore import Qt, Signal
from loguru import logger

try:
    from ..portfolio.optimizer import PortfolioOptimizer
    from ..portfolio.position_sizer import AdaptivePositionSizer
    _HAS_PORTFOLIO = True
except ImportError:
    _HAS_PORTFOLIO = False
    logger.warning("Portfolio optimization modules not available")

try:
    from .portfolio_viz import EfficientFrontierWidget
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False
    logger.warning("Portfolio visualization not available")


class PortfolioOptimizationTab(QWidget):
    """
    Portfolio Optimization configuration and visualization tab.

    Features:
    - Optimizer configuration (risk measure, objective, constraints)
    - Position sizing settings
    - Real-time portfolio stats display
    - Efficient frontier visualization
    - Risk metrics dashboard
    """

    settings_changed = Signal(dict)  # Emitted when settings change

    def __init__(self, parent=None):
        super().__init__(parent)

        self.optimizer: Optional[PortfolioOptimizer] = None
        self.position_sizer: Optional[AdaptivePositionSizer] = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Create tab widget for different sections
        tabs = QTabWidget()

        # Tab 1: Optimizer Settings
        tabs.addTab(self._create_optimizer_settings(), "Optimizer Settings")

        # Tab 2: Position Sizing
        tabs.addTab(self._create_position_sizing_settings(), "Position Sizing")

        # Tab 3: Portfolio Stats
        tabs.addTab(self._create_portfolio_stats(), "Portfolio Stats")

        # Tab 4: Risk Dashboard
        tabs.addTab(self._create_risk_dashboard(), "Risk Metrics")

        # Tab 5: Efficient Frontier Visualization
        if _HAS_VISUALIZATION:
            tabs.addTab(self._create_frontier_visualization(), "Efficient Frontier")

        layout.addWidget(tabs)

        # Control buttons at bottom
        controls = self._create_controls()
        layout.addLayout(controls)

    def _create_optimizer_settings(self) -> QWidget:
        """Create optimizer configuration panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Risk Measure Group
        risk_group = QGroupBox("Risk Measure")
        risk_layout = QVBoxLayout(risk_group)

        self.risk_measure_combo = QComboBox()
        self.risk_measure_combo.addItems([
            "MV - Mean-Variance (Standard Deviation)",
            "CVaR - Conditional Value at Risk (Expected Shortfall)",
            "CDaR - Conditional Drawdown at Risk",
            "EVaR - Entropic Value at Risk",
            "WR - Worst Realization (Worst Case)",
            "MDD - Maximum Drawdown",
        ])
        self.risk_measure_combo.setCurrentText("CVaR - Conditional Value at Risk (Expected Shortfall)")
        risk_layout.addWidget(QLabel("Select Risk Measure:"))
        risk_layout.addWidget(self.risk_measure_combo)

        layout.addWidget(risk_group)

        # Objective Group
        obj_group = QGroupBox("Optimization Objective")
        obj_layout = QVBoxLayout(obj_group)

        self.objective_combo = QComboBox()
        self.objective_combo.addItems([
            "Sharpe - Maximum Sharpe Ratio",
            "MinRisk - Minimum Risk",
            "Utility - Maximum Utility Function",
            "MaxRet - Maximum Return",
        ])
        obj_layout.addWidget(QLabel("Select Objective:"))
        obj_layout.addWidget(self.objective_combo)

        layout.addWidget(obj_group)

        # Parameters Group
        params_group = QGroupBox("Optimization Parameters")
        params_layout = QVBoxLayout(params_group)

        # Risk-free rate
        rf_layout = QHBoxLayout()
        rf_layout.addWidget(QLabel("Risk-Free Rate (%):"))
        self.risk_free_rate_spin = QDoubleSpinBox()
        self.risk_free_rate_spin.setRange(0.0, 10.0)
        self.risk_free_rate_spin.setValue(0.0)
        self.risk_free_rate_spin.setSingleStep(0.1)
        self.risk_free_rate_spin.setDecimals(2)
        rf_layout.addWidget(self.risk_free_rate_spin)
        rf_layout.addStretch()
        params_layout.addLayout(rf_layout)

        # Risk aversion
        ra_layout = QHBoxLayout()
        ra_layout.addWidget(QLabel("Risk Aversion (λ):"))
        self.risk_aversion_spin = QDoubleSpinBox()
        self.risk_aversion_spin.setRange(0.1, 10.0)
        self.risk_aversion_spin.setValue(1.0)
        self.risk_aversion_spin.setSingleStep(0.1)
        self.risk_aversion_spin.setDecimals(1)
        ra_layout.addWidget(self.risk_aversion_spin)
        ra_layout.addStretch()
        params_layout.addLayout(ra_layout)

        layout.addWidget(params_group)

        # Constraints Group
        constraints_group = QGroupBox("Portfolio Constraints")
        constraints_layout = QVBoxLayout(constraints_group)

        # Max position size
        max_weight_layout = QHBoxLayout()
        max_weight_layout.addWidget(QLabel("Max Weight per Asset (%):"))
        self.max_weight_spin = QDoubleSpinBox()
        self.max_weight_spin.setRange(1.0, 100.0)
        self.max_weight_spin.setValue(25.0)
        self.max_weight_spin.setSingleStep(1.0)
        self.max_weight_spin.setDecimals(1)
        max_weight_layout.addWidget(self.max_weight_spin)
        max_weight_layout.addStretch()
        constraints_layout.addLayout(max_weight_layout)

        # Min position size
        min_weight_layout = QHBoxLayout()
        min_weight_layout.addWidget(QLabel("Min Weight per Asset (%):"))
        self.min_weight_spin = QDoubleSpinBox()
        self.min_weight_spin.setRange(0.0, 50.0)
        self.min_weight_spin.setValue(1.0)
        self.min_weight_spin.setSingleStep(0.5)
        self.min_weight_spin.setDecimals(1)
        min_weight_layout.addWidget(self.min_weight_spin)
        min_weight_layout.addStretch()
        constraints_layout.addLayout(min_weight_layout)

        layout.addWidget(constraints_group)

        layout.addStretch()
        return widget

    def _create_position_sizing_settings(self) -> QWidget:
        """Create position sizing configuration panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Enable adaptive sizing
        self.adaptive_sizing_checkbox = QCheckBox("Enable Adaptive Position Sizing")
        self.adaptive_sizing_checkbox.setChecked(True)
        layout.addWidget(self.adaptive_sizing_checkbox)

        # Settings Group
        settings_group = QGroupBox("Position Sizing Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Lookback period
        lookback_layout = QHBoxLayout()
        lookback_layout.addWidget(QLabel("Lookback Period (days):"))
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(10, 252)
        self.lookback_spin.setValue(60)
        self.lookback_spin.setSingleStep(5)
        lookback_layout.addWidget(self.lookback_spin)
        lookback_layout.addStretch()
        settings_layout.addLayout(lookback_layout)

        # Rebalance frequency
        rebalance_layout = QHBoxLayout()
        rebalance_layout.addWidget(QLabel("Rebalance Frequency (days):"))
        self.rebalance_spin = QSpinBox()
        self.rebalance_spin.setRange(1, 30)
        self.rebalance_spin.setValue(5)
        self.rebalance_spin.setSingleStep(1)
        rebalance_layout.addWidget(self.rebalance_spin)
        rebalance_layout.addStretch()
        settings_layout.addLayout(rebalance_layout)

        layout.addWidget(settings_group)

        # Risk Parity Group
        rp_group = QGroupBox("Risk Parity Options")
        rp_layout = QVBoxLayout(rp_group)

        self.risk_parity_checkbox = QCheckBox("Use Risk Parity Allocation")
        self.risk_parity_checkbox.setChecked(False)
        rp_layout.addWidget(self.risk_parity_checkbox)

        rp_layout.addWidget(QLabel("Risk Parity ensures equal risk contribution from each asset."))

        layout.addWidget(rp_group)

        # Volatility Targeting Group
        vol_group = QGroupBox("Volatility Targeting")
        vol_layout = QVBoxLayout(vol_group)

        self.vol_targeting_checkbox = QCheckBox("Enable Volatility Targeting")
        self.vol_targeting_checkbox.setChecked(False)
        vol_layout.addWidget(self.vol_targeting_checkbox)

        target_vol_layout = QHBoxLayout()
        target_vol_layout.addWidget(QLabel("Target Annual Volatility (%):"))
        self.target_vol_spin = QDoubleSpinBox()
        self.target_vol_spin.setRange(1.0, 100.0)
        self.target_vol_spin.setValue(15.0)
        self.target_vol_spin.setSingleStep(1.0)
        self.target_vol_spin.setDecimals(1)
        target_vol_layout.addWidget(self.target_vol_spin)
        target_vol_layout.addStretch()
        vol_layout.addLayout(target_vol_layout)

        layout.addWidget(vol_group)

        layout.addStretch()
        return widget

    def _create_portfolio_stats(self) -> QWidget:
        """Create portfolio statistics display."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Current Portfolio Info
        info_group = QGroupBox("Current Portfolio")
        info_layout = QVBoxLayout(info_group)

        self.portfolio_info_text = QTextEdit()
        self.portfolio_info_text.setReadOnly(True)
        self.portfolio_info_text.setMaximumHeight(100)
        self.portfolio_info_text.setText("No portfolio optimized yet. Click 'Optimize Portfolio' to begin.")
        info_layout.addWidget(self.portfolio_info_text)

        layout.addWidget(info_group)

        # Weights Table
        weights_group = QGroupBox("Asset Weights")
        weights_layout = QVBoxLayout(weights_group)

        self.weights_table = QTableWidget()
        self.weights_table.setColumnCount(3)
        self.weights_table.setHorizontalHeaderLabels(["Asset", "Weight (%)", "Position Size"])
        weights_layout.addWidget(self.weights_table)

        layout.addWidget(weights_group)

        return widget

    def _create_risk_dashboard(self) -> QWidget:
        """Create risk metrics dashboard."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Risk Metrics Table
        metrics_group = QGroupBox("Risk Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.risk_metrics_table = QTableWidget()
        self.risk_metrics_table.setColumnCount(2)
        self.risk_metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.risk_metrics_table.setRowCount(10)

        # Initialize metric labels
        metrics = [
            "Expected Return (Annual)",
            "Volatility (Annual)",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Maximum Drawdown",
            "CVaR 95%",
            "CVaR 99%",
            "Skewness",
            "Kurtosis",
            "Concentration (HHI)",
        ]

        for i, metric in enumerate(metrics):
            self.risk_metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.risk_metrics_table.setItem(i, 1, QTableWidgetItem("N/A"))

        metrics_layout.addWidget(self.risk_metrics_table)
        layout.addWidget(metrics_group)

        return widget

    def _create_frontier_visualization(self) -> QWidget:
        """Create efficient frontier visualization tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        if not _HAS_VISUALIZATION:
            error_label = QLabel("Visualization not available. Install matplotlib.")
            layout.addWidget(error_label)
            return widget

        # Create visualization widget
        self.frontier_viz = EfficientFrontierWidget()
        layout.addWidget(self.frontier_viz)

        return widget

    def _create_controls(self) -> QHBoxLayout:
        """Create control buttons."""
        layout = QHBoxLayout()

        self.optimize_btn = QPushButton("Optimize Portfolio")
        self.optimize_btn.clicked.connect(self._on_optimize)
        layout.addWidget(self.optimize_btn)

        self.calculate_frontier_btn = QPushButton("Calculate Efficient Frontier")
        self.calculate_frontier_btn.clicked.connect(self._on_calculate_frontier)
        layout.addWidget(self.calculate_frontier_btn)

        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.clicked.connect(self._on_apply_settings)
        layout.addWidget(self.apply_btn)

        layout.addStretch()

        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)

        return layout

    def _load_settings(self):
        """Load settings from configuration."""
        # TODO: Load from config file
        logger.debug("Portfolio settings loaded")

    def _on_optimize(self):
        """Handle optimize button click."""
        if not _HAS_PORTFOLIO:
            self.status_label.setText("Status: Portfolio optimization not available")
            return

        try:
            # Get settings
            settings = self._get_settings()

            # Create optimizer with current settings
            self.optimizer = PortfolioOptimizer(
                risk_measure=settings["risk_measure"],
                objective=settings["objective"],
                risk_free_rate=settings["risk_free_rate"] / 100.0,
                risk_aversion=settings["risk_aversion"],
            )

            self.status_label.setText("Status: Optimizer configured. Ready to calculate weights.")
            logger.info("Portfolio optimizer configured successfully")

        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            logger.error(f"Failed to configure optimizer: {e}")

    def _on_calculate_frontier(self):
        """Handle calculate frontier button click."""
        if not _HAS_PORTFOLIO:
            self.status_label.setText("Status: Portfolio optimization not available")
            return

        if not _HAS_VISUALIZATION:
            self.status_label.setText("Status: Visualization not available - install matplotlib")
            return

        if not hasattr(self, 'frontier_viz'):
            self.status_label.setText("Status: Frontier visualization widget not initialized")
            return

        # TODO: This needs real historical data from the system
        # For now, we show a message
        self.status_label.setText("Status: Connect to data source to calculate efficient frontier")
        logger.info("Efficient frontier calculation requested - needs historical data integration")

    def _on_apply_settings(self):
        """Handle apply settings button click."""
        settings = self._get_settings()
        self.settings_changed.emit(settings)
        self.status_label.setText("Status: Settings applied")
        logger.info("Portfolio optimization settings applied")

    def _get_settings(self) -> dict:
        """Get current settings from UI."""
        # Extract risk measure code from combo box text
        risk_measure_text = self.risk_measure_combo.currentText()
        risk_measure = risk_measure_text.split(" - ")[0]

        # Extract objective code from combo box text
        objective_text = self.objective_combo.currentText()
        objective = objective_text.split(" - ")[0]

        return {
            "risk_measure": risk_measure,
            "objective": objective,
            "risk_free_rate": self.risk_free_rate_spin.value(),
            "risk_aversion": self.risk_aversion_spin.value(),
            "max_weight": self.max_weight_spin.value() / 100.0,
            "min_weight": self.min_weight_spin.value() / 100.0,
            "adaptive_sizing_enabled": self.adaptive_sizing_checkbox.isChecked(),
            "lookback_period": self.lookback_spin.value(),
            "rebalance_frequency": self.rebalance_spin.value(),
            "risk_parity_enabled": self.risk_parity_checkbox.isChecked(),
            "vol_targeting_enabled": self.vol_targeting_checkbox.isChecked(),
            "target_volatility": self.target_vol_spin.value() / 100.0,
        }

    def update_portfolio_display(self, weights, stats):
        """Update portfolio display with new weights and stats."""
        # Update weights table
        self.weights_table.setRowCount(len(weights))
        for i, (asset, weight) in enumerate(weights.items()):
            self.weights_table.setItem(i, 0, QTableWidgetItem(asset))
            self.weights_table.setItem(i, 1, QTableWidgetItem(f"{weight * 100:.2f}%"))
            # Position size would be calculated with actual capital
            self.weights_table.setItem(i, 2, QTableWidgetItem("N/A"))

        # Update risk metrics
        if stats:
            metrics_map = {
                0: ("expected_return", "{:.2%}"),
                1: ("volatility", "{:.2%}"),
                2: ("sharpe_ratio", "{:.2f}"),
                3: ("sortino_ratio", "{:.2f}"),
                4: ("max_drawdown", "{:.2%}"),
                5: ("cvar_95", "{:.2%}"),
                6: ("cvar_99", "{:.2%}"),
                7: ("skewness", "{:.2f}"),
                8: ("kurtosis", "{:.2f}"),
                9: ("concentration", "{:.2%}"),
            }

            for row, (key, fmt) in metrics_map.items():
                value = stats.get(key, 0.0)
                self.risk_metrics_table.setItem(row, 1, QTableWidgetItem(fmt.format(value)))

    def update_efficient_frontier(self, frontier_data, current_portfolio=None, asset_data=None):
        """
        Update efficient frontier visualization.

        Args:
            frontier_data: DataFrame with efficient frontier points
            current_portfolio: Dict with current portfolio stats
            asset_data: DataFrame with individual asset stats
        """
        if not _HAS_VISUALIZATION or not hasattr(self, 'frontier_viz'):
            logger.warning("Frontier visualization not available")
            return

        try:
            self.frontier_viz.plot_efficient_frontier(
                frontier_data=frontier_data,
                current_portfolio=current_portfolio,
                asset_data=asset_data
            )
            logger.info("Efficient frontier visualization updated")
        except Exception as e:
            logger.error(f"Failed to update frontier visualization: {e}")
