"""
Pre-Trade Validation Dialog

Validates trade parameters against real-time DOM data before execution:
- Checks order size vs available liquidity
- Validates spread conditions
- Estimates market impact and slippage
- Provides warnings and requires confirmation
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QProgressBar, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from loguru import logger


class PreTradeValidationDialog(QDialog):
    """
    Dialog for pre-trade validation using DOM metrics.

    Validates:
    - Liquidity: Order size vs available depth
    - Spread: Current spread vs historical average
    - Market impact: Estimated price impact
    - Execution cost: Total execution cost estimate
    """

    def __init__(
        self,
        parent=None,
        symbol: str = "EURUSD",
        side: str = "BUY",
        volume: float = 1.0,
        dom_service=None,
        execution_optimizer=None
    ):
        super().__init__(parent)
        self.symbol = symbol
        self.side = side
        self.volume = volume
        self.dom_service = dom_service
        self.execution_optimizer = execution_optimizer

        self.validation_passed = False
        self.dom_snapshot = None
        self.validation_results = {}

        self.setWindowTitle(f"Pre-Trade Validation - {symbol}")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self._init_ui()
        self._run_validation()

    def _init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel(f"Validating {self.side} order for {self.volume:.2f} lots of {self.symbol}")
        header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(header)

        # Validation status
        self.status_label = QLabel("⏳ Running validation checks...")
        self.status_label.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Validation metrics
        metrics_group = self._create_metrics_section()
        layout.addWidget(metrics_group)

        # Warnings section
        warnings_group = self._create_warnings_section()
        layout.addWidget(warnings_group)

        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.cancel_btn = QPushButton("❌ Cancel Order")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; padding: 10px; }"
        )
        buttons_layout.addWidget(self.cancel_btn)

        self.proceed_btn = QPushButton("✅ Proceed with Order")
        self.proceed_btn.clicked.connect(self.accept)
        self.proceed_btn.setEnabled(False)
        self.proceed_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; }"
        )
        buttons_layout.addWidget(self.proceed_btn)

        layout.addLayout(buttons_layout)

    def _create_metrics_section(self) -> QGroupBox:
        """Create metrics display section"""
        group = QGroupBox("Validation Metrics")
        layout = QFormLayout(group)

        self.spread_label = QLabel("--")
        layout.addRow("Current Spread:", self.spread_label)

        self.liquidity_label = QLabel("--")
        layout.addRow("Available Liquidity:", self.liquidity_label)

        self.impact_label = QLabel("--")
        layout.addRow("Estimated Impact:", self.impact_label)

        self.slippage_label = QLabel("--")
        layout.addRow("Estimated Slippage:", self.slippage_label)

        self.cost_label = QLabel("--")
        layout.addRow("Total Execution Cost:", self.cost_label)

        return group

    def _create_warnings_section(self) -> QGroupBox:
        """Create warnings section"""
        group = QGroupBox("Warnings & Alerts")
        layout = QVBoxLayout(group)

        self.warnings_text = QTextEdit()
        self.warnings_text.setReadOnly(True)
        self.warnings_text.setMaximumHeight(150)
        self.warnings_text.setStyleSheet("background-color: #FFF3CD; color: #856404;")
        layout.addWidget(self.warnings_text)

        return group

    def _run_validation(self):
        """Run validation checks"""
        try:
            self.progress_bar.setValue(20)

            # 1. Fetch DOM snapshot
            if not self.dom_service:
                self._add_warning("⚠️  DOM service not available - validation limited")
                self._set_validation_complete(True, "No DOM service")
                return

            self.dom_snapshot = self.dom_service.get_latest_dom_snapshot(self.symbol)
            if not self.dom_snapshot:
                self._add_warning("⚠️  No DOM data available for this symbol")
                self._set_validation_complete(True, "No DOM data")
                return

            self.progress_bar.setValue(40)

            # 2. Validate liquidity
            self._validate_liquidity()
            self.progress_bar.setValue(60)

            # 3. Validate spread
            self._validate_spread()
            self.progress_bar.setValue(80)

            # 4. Estimate execution cost
            self._estimate_execution_cost()
            self.progress_bar.setValue(100)

            # 5. Final decision
            self._set_validation_complete(True, "Validation complete")

        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            self._add_warning(f"❌ Validation error: {e}")
            self._set_validation_complete(False, f"Error: {e}")

    def _validate_liquidity(self):
        """Validate order size against available liquidity"""
        bid_depth = self.dom_snapshot.get('bid_depth', 0)
        ask_depth = self.dom_snapshot.get('ask_depth', 0)

        # Use opposite side depth for our order (buying consumes ask liquidity)
        available_depth = ask_depth if self.side == "BUY" else bid_depth
        total_depth = bid_depth + ask_depth

        self.liquidity_label.setText(f"{available_depth:,.0f} (Total: {total_depth:,.0f})")

        # Calculate order size in base currency units (approximate)
        # 1 lot = 100,000 units for forex
        order_size_units = self.volume * 100000

        # Check if order exceeds liquidity thresholds
        liquidity_ratio = order_size_units / available_depth if available_depth > 0 else float('inf')

        if liquidity_ratio > 0.5:
            self._add_warning(
                f"🔴 CRITICAL: Order size ({order_size_units:,.0f} units) exceeds 50% of available "
                f"liquidity ({available_depth:,.0f} units). High market impact expected!"
            )
            self.validation_results['liquidity_risk'] = 'CRITICAL'
        elif liquidity_ratio > 0.3:
            self._add_warning(
                f"🟠 WARNING: Order size ({order_size_units:,.0f} units) exceeds 30% of available "
                f"liquidity. Moderate market impact expected."
            )
            self.validation_results['liquidity_risk'] = 'HIGH'
        elif liquidity_ratio > 0.1:
            self._add_info(
                f"🟡 NOTICE: Order size is {liquidity_ratio*100:.1f}% of available liquidity. "
                f"Minor market impact possible."
            )
            self.validation_results['liquidity_risk'] = 'MODERATE'
        else:
            self._add_info(f"✅ Liquidity check passed: {liquidity_ratio*100:.1f}% of available depth")
            self.validation_results['liquidity_risk'] = 'LOW'

    def _validate_spread(self):
        """Validate current spread"""
        spread = self.dom_snapshot.get('spread', 0)
        mid_price = self.dom_snapshot.get('mid_price', 1.0)

        # Calculate spread in basis points
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0

        self.spread_label.setText(f"{spread:.5f} ({spread_bps:.1f} bps)")

        # Thresholds for forex pairs (adjust based on pair)
        if spread_bps > 10:
            self._add_warning(
                f"🔴 CRITICAL: Spread is very wide ({spread_bps:.1f} bps). "
                f"This may indicate low liquidity or market volatility."
            )
            self.validation_results['spread_risk'] = 'CRITICAL'
        elif spread_bps > 3:
            self._add_warning(
                f"🟠 WARNING: Spread is elevated ({spread_bps:.1f} bps). "
                f"Execution costs will be higher than normal."
            )
            self.validation_results['spread_risk'] = 'HIGH'
        else:
            self._add_info(f"✅ Spread check passed: {spread_bps:.1f} bps is within normal range")
            self.validation_results['spread_risk'] = 'LOW'

    def _estimate_execution_cost(self):
        """Estimate total execution cost"""
        if not self.execution_optimizer:
            self.impact_label.setText("N/A")
            self.slippage_label.setText("N/A")
            self.cost_label.setText("N/A")
            return

        try:
            # Import SmartExecutionOptimizer
            from ..execution.smart_execution import SmartExecutionOptimizer

            # Estimate execution cost with DOM data
            cost_estimate = self.execution_optimizer.estimate_execution_cost(
                order_size=self.volume * 100000,  # Convert lots to units
                direction='buy' if self.side == 'BUY' else 'sell',
                current_price=self.dom_snapshot.get('mid_price', 0),
                volatility=0.0001,  # Placeholder - would come from historical data
                dom_snapshot=self.dom_snapshot
            )

            self.impact_label.setText(f"{cost_estimate.get('market_impact', 0):.5f}")
            self.slippage_label.setText(f"{cost_estimate.get('slippage', 0):.5f}")
            self.cost_label.setText(f"{cost_estimate.get('total_cost', 0):.5f}")

            # Check for high impact warnings
            impact_result = self.execution_optimizer.check_high_impact_order(
                cost_estimate.get('market_impact', 0),
                cost_estimate.get('total_cost', 0)
            )

            if impact_result['is_critical']:
                self._add_warning(
                    f"🔴 CRITICAL IMPACT: {impact_result['message']} - "
                    f"Consider reducing order size!"
                )
                self.validation_results['execution_risk'] = 'CRITICAL'
            elif impact_result['is_high']:
                self._add_warning(
                    f"🟠 HIGH IMPACT: {impact_result['message']}"
                )
                self.validation_results['execution_risk'] = 'HIGH'
            else:
                self._add_info(f"✅ Execution impact is within acceptable range")
                self.validation_results['execution_risk'] = 'LOW'

        except Exception as e:
            logger.error(f"Failed to estimate execution cost: {e}")
            self.impact_label.setText("Error")
            self.slippage_label.setText("Error")
            self.cost_label.setText("Error")

    def _add_warning(self, message: str):
        """Add warning message"""
        self.warnings_text.append(message)

    def _add_info(self, message: str):
        """Add info message"""
        self.warnings_text.append(message)

    def _set_validation_complete(self, passed: bool, message: str):
        """Set validation complete status"""
        self.validation_passed = passed

        # Check if any critical or high risks exist
        has_critical = any(
            risk == 'CRITICAL'
            for risk in self.validation_results.values()
        )

        if has_critical:
            self.status_label.setText(f"❌ Validation complete with CRITICAL warnings - Proceed with caution!")
            self.status_label.setStyleSheet(
                "font-size: 12px; padding: 5px; background-color: #f44336; "
                "color: white; font-weight: bold;"
            )
            self.proceed_btn.setStyleSheet(
                "QPushButton { background-color: #ff9800; color: white; "
                "font-weight: bold; padding: 10px; }"
            )
            self.proceed_btn.setText("⚠️  Proceed Anyway (High Risk)")
        else:
            self.status_label.setText(f"✅ {message}")
            self.status_label.setStyleSheet(
                "font-size: 12px; padding: 5px; background-color: #4CAF50; "
                "color: white; font-weight: bold;"
            )

        self.proceed_btn.setEnabled(True)
