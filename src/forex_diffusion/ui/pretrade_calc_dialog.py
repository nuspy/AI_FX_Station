"""
Pre-Trade Calculations Dialog

Dialog showing detailed calculations before trade execution.
Displays signal info, position sizing, SL/TP, margin requirements with
editable fields and real-time recalculation.

FASE 9 - Part 3
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QFormLayout, QDoubleSpinBox, QSpinBox, QLineEdit,
    QFrame, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QPalette
from loguru import logger


class PreTradeCalcDialog(QDialog):
    """
    Pre-trade calculations dialog.

    Shows comprehensive breakdown of trade calculations before execution:
    - Signal information (symbol, direction, pattern, quality)
    - Position size calculation (method, suggested size, risk)
    - Stop Loss / Take Profit (prices, R:R ratio)
    - Margin & Exposure (required margin, utilization, portfolio exposure)
    - Editable fields with automatic recalculation
    - Execute / Modify / Cancel buttons
    """

    execute_requested = Signal(dict)  # Emitted when user confirms trade

    def __init__(self, signal_data: Dict[str, Any], parent: Optional[QDialog] = None):
        super().__init__(parent)

        self.signal_data = signal_data
        self.calculated_params = {}

        self._setup_ui()
        self._populate_from_signal()
        self._connect_signals()

        logger.info("PreTradeCalcDialog initialized")
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()

    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets"""
        from ..i18n.widget_helper import apply_tooltip
        
        if hasattr(self, 'enable_validation_check'):
            apply_tooltip(self.enable_validation_check, "enable_validation", "pretrade_validation")
        if hasattr(self, 'max_trade_size_spin'):
            apply_tooltip(self.max_trade_size_spin, "max_trade_size", "pretrade_validation")
        if hasattr(self, 'require_confirmation_check'):
            apply_tooltip(self.require_confirmation_check, "require_confirmation", "pretrade_validation")
        if hasattr(self, 'check_spread_check'):
            apply_tooltip(self.check_spread_check, "check_spread", "pretrade_validation")
        if hasattr(self, 'check_margin_check'):
            apply_tooltip(self.check_margin_check, "check_margin", "pretrade_validation")
    

    def _setup_ui(self):
        """Setup the UI components."""
        self.setWindowTitle("Pre-Trade Calculations")
        self.setMinimumWidth(600)
        self.setMinimumHeight(700)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h2>Trade Execution Preview</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # === Signal Information Section ===
        signal_group = QGroupBox("Signal Information")
        signal_layout = QFormLayout(signal_group)

        self.symbol_label = QLabel()
        self.direction_label = QLabel()
        self.pattern_type_label = QLabel()
        self.quality_score_label = QLabel()
        self.entry_price_label = QLabel()

        signal_layout.addRow("Symbol:", self.symbol_label)
        signal_layout.addRow("Direction:", self.direction_label)
        signal_layout.addRow("Pattern Type:", self.pattern_type_label)
        signal_layout.addRow("Quality Score:", self.quality_score_label)
        signal_layout.addRow("Entry Price:", self.entry_price_label)

        layout.addWidget(signal_group)

        # === Position Sizing Section ===
        sizing_group = QGroupBox("Position Sizing")
        sizing_layout = QFormLayout(sizing_group)

        self.sizing_method_label = QLabel()
        self.account_balance_label = QLabel()

        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setDecimals(2)
        self.position_size_spin.setRange(0.01, 100.0)
        self.position_size_spin.setSingleStep(0.01)
        self.position_size_spin.setSuffix(" lots")

        self.risk_amount_label = QLabel()
        self.risk_pct_label = QLabel()

        sizing_layout.addRow("Sizing Method:", self.sizing_method_label)
        sizing_layout.addRow("Account Balance:", self.account_balance_label)
        sizing_layout.addRow("Position Size:", self.position_size_spin)
        sizing_layout.addRow("Risk Amount:", self.risk_amount_label)
        sizing_layout.addRow("Risk Percentage:", self.risk_pct_label)

        layout.addWidget(sizing_group)

        # === Stop Loss / Take Profit Section ===
        sltp_group = QGroupBox("Stop Loss / Take Profit")
        sltp_layout = QFormLayout(sltp_group)

        self.sl_price_spin = QDoubleSpinBox()
        self.sl_price_spin.setDecimals(5)
        self.sl_price_spin.setRange(0.00001, 100000.0)
        self.sl_price_spin.setSingleStep(0.0001)

        self.tp_price_spin = QDoubleSpinBox()
        self.tp_price_spin.setDecimals(5)
        self.tp_price_spin.setRange(0.00001, 100000.0)
        self.tp_price_spin.setSingleStep(0.0001)

        self.sl_distance_label = QLabel()
        self.tp_distance_label = QLabel()
        self.rr_ratio_label = QLabel()

        sltp_layout.addRow("Stop Loss Price:", self.sl_price_spin)
        sltp_layout.addRow("SL Distance (pips):", self.sl_distance_label)
        sltp_layout.addRow("Take Profit Price:", self.tp_price_spin)
        sltp_layout.addRow("TP Distance (pips):", self.tp_distance_label)
        sltp_layout.addRow("Risk:Reward Ratio:", self.rr_ratio_label)

        layout.addWidget(sltp_group)

        # === Margin & Exposure Section ===
        margin_group = QGroupBox("Margin & Exposure")
        margin_layout = QFormLayout(margin_group)

        self.required_margin_label = QLabel()
        self.margin_utilization_label = QLabel()
        self.portfolio_exposure_label = QLabel()
        self.available_margin_label = QLabel()

        margin_layout.addRow("Required Margin:", self.required_margin_label)
        margin_layout.addRow("Margin Utilization:", self.margin_utilization_label)
        margin_layout.addRow("Portfolio Exposure:", self.portfolio_exposure_label)
        margin_layout.addRow("Available After Trade:", self.available_margin_label)

        layout.addWidget(margin_group)

        # === Warnings Section ===
        self.warnings_label = QLabel()
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet("color: orange; font-weight: bold;")
        self.warnings_label.setVisible(False)
        layout.addWidget(self.warnings_label)

        # === Action Buttons ===
        actions_layout = QHBoxLayout()

        self.execute_btn = QPushButton("Execute Trade")
        self.execute_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 10px;")
        self.execute_btn.clicked.connect(self._on_execute_clicked)

        self.modify_btn = QPushButton("Recalculate")
        self.modify_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 10px;")
        self.modify_btn.clicked.connect(self._recalculate)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("padding: 10px;")
        self.cancel_btn.clicked.connect(self.reject)

        actions_layout.addWidget(self.execute_btn)
        actions_layout.addWidget(self.modify_btn)
        actions_layout.addWidget(self.cancel_btn)

        layout.addLayout(actions_layout)

    def _populate_from_signal(self):
        """Populate fields from signal data."""
        # Signal information
        self.symbol_label.setText(self.signal_data.get('symbol', 'N/A'))

        direction = self.signal_data.get('direction', 'LONG')
        direction_color = "#2ecc71" if direction.upper() == "LONG" else "#e74c3c"
        self.direction_label.setText(f"<b style='color: {direction_color}'>{direction.upper()}</b>")

        self.pattern_type_label.setText(self.signal_data.get('pattern_type', 'N/A'))

        quality_score = self.signal_data.get('quality_score', 0.0)
        quality_color = self._get_quality_color(quality_score)
        self.quality_score_label.setText(f"<b style='color: {quality_color}'>{quality_score:.2f}</b>")

        entry_price = self.signal_data.get('entry_price', 0.0)
        self.entry_price_label.setText(f"{entry_price:.5f}")

        # Position sizing
        sizing_method = self.signal_data.get('sizing_method', 'Fixed Fractional')
        self.sizing_method_label.setText(sizing_method)

        account_balance = self.signal_data.get('account_balance', 10000.0)
        self.account_balance_label.setText(f"${account_balance:,.2f}")

        position_size = self.signal_data.get('position_size', 0.1)
        self.position_size_spin.setValue(position_size)

        # Stop Loss / Take Profit
        sl_price = self.signal_data.get('sl_price', 0.0)
        tp_price = self.signal_data.get('tp_price', 0.0)

        self.sl_price_spin.setValue(sl_price)
        self.tp_price_spin.setValue(tp_price)

        # Trigger initial calculation
        self._recalculate()

    def _connect_signals(self):
        """Connect spin box signals for automatic recalculation."""
        self.position_size_spin.valueChanged.connect(self._recalculate)
        self.sl_price_spin.valueChanged.connect(self._recalculate)
        self.tp_price_spin.valueChanged.connect(self._recalculate)

    def _recalculate(self):
        """Recalculate all derived values based on current inputs."""
        try:
            # Get current values
            entry_price = self.signal_data.get('entry_price', 0.0)
            position_size = self.position_size_spin.value()
            sl_price = self.sl_price_spin.value()
            tp_price = self.tp_price_spin.value()
            account_balance = self.signal_data.get('account_balance', 10000.0)
            direction = self.signal_data.get('direction', 'LONG').upper()

            # Calculate SL/TP distances in pips
            pip_multiplier = 10000 if 'JPY' not in self.signal_data.get('symbol', '') else 100

            if direction == 'LONG':
                sl_distance_pips = (entry_price - sl_price) * pip_multiplier
                tp_distance_pips = (tp_price - entry_price) * pip_multiplier
            else:  # SHORT
                sl_distance_pips = (sl_price - entry_price) * pip_multiplier
                tp_distance_pips = (entry_price - tp_price) * pip_multiplier

            self.sl_distance_label.setText(f"{abs(sl_distance_pips):.1f} pips")
            self.tp_distance_label.setText(f"{abs(tp_distance_pips):.1f} pips")

            # Calculate R:R ratio
            if sl_distance_pips > 0:
                rr_ratio = tp_distance_pips / sl_distance_pips
                rr_color = "#2ecc71" if rr_ratio >= 2.0 else "#e74c3c" if rr_ratio < 1.0 else "#f39c12"
                self.rr_ratio_label.setText(f"<b style='color: {rr_color}'>1:{rr_ratio:.2f}</b>")
            else:
                self.rr_ratio_label.setText("N/A")

            # Calculate risk amount
            pip_value = 10.0  # Simplified: $10 per pip per lot (varies by instrument)
            risk_amount = abs(sl_distance_pips) * pip_value * position_size
            risk_pct = (risk_amount / account_balance) * 100

            self.risk_amount_label.setText(f"${risk_amount:,.2f}")

            risk_color = "#2ecc71" if risk_pct <= 1.0 else "#f39c12" if risk_pct <= 2.0 else "#e74c3c"
            self.risk_pct_label.setText(f"<b style='color: {risk_color}'>{risk_pct:.2f}%</b>")

            # Calculate margin requirements (simplified)
            leverage = self.signal_data.get('leverage', 100)
            contract_size = 100000  # Standard lot
            required_margin = (position_size * contract_size * entry_price) / leverage

            self.required_margin_label.setText(f"${required_margin:,.2f}")

            # Calculate margin utilization
            total_equity = account_balance  # Simplified
            margin_utilization = (required_margin / total_equity) * 100
            margin_color = "#2ecc71" if margin_utilization <= 30 else "#f39c12" if margin_utilization <= 50 else "#e74c3c"
            self.margin_utilization_label.setText(f"<b style='color: {margin_color}'>{margin_utilization:.2f}%</b>")

            # Portfolio exposure
            exposure_value = position_size * contract_size * entry_price
            portfolio_exposure = (exposure_value / total_equity) * 100
            self.portfolio_exposure_label.setText(f"{portfolio_exposure:.2f}%")

            # Available margin after trade
            available_margin = total_equity - required_margin
            self.available_margin_label.setText(f"${available_margin:,.2f}")

            # Check for warnings
            warnings = []
            if risk_pct > 2.0:
                warnings.append("⚠ Risk exceeds 2% of account balance")
            if margin_utilization > 50:
                warnings.append("⚠ High margin utilization (>50%)")
            if rr_ratio < 1.5:
                warnings.append("⚠ Risk:Reward ratio below 1.5")

            if warnings:
                self.warnings_label.setText("\n".join(warnings))
                self.warnings_label.setVisible(True)
            else:
                self.warnings_label.setVisible(False)

            # Store calculated parameters
            self.calculated_params = {
                'position_size': position_size,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'risk_amount': risk_amount,
                'risk_pct': risk_pct,
                'required_margin': required_margin,
                'margin_utilization': margin_utilization,
                'rr_ratio': rr_ratio,
                'sl_distance_pips': abs(sl_distance_pips),
                'tp_distance_pips': abs(tp_distance_pips),
            }

            logger.debug(f"Recalculated trade params: {self.calculated_params}")

        except Exception as e:
            logger.exception(f"Error recalculating trade params: {e}")
            QMessageBox.warning(
                self,
                "Calculation Error",
                f"Failed to recalculate trade parameters:\n{str(e)}"
            )

    def _get_quality_color(self, score: float) -> str:
        """Get color for quality score."""
        if score >= 0.8:
            return "#2ecc71"  # Green
        elif score >= 0.6:
            return "#f39c12"  # Orange
        else:
            return "#e74c3c"  # Red

    def _on_execute_clicked(self):
        """Handle execute trade button click."""
        # Final validation
        risk_pct = self.calculated_params.get('risk_pct', 0.0)
        margin_util = self.calculated_params.get('margin_utilization', 0.0)

        if risk_pct > 5.0:
            reply = QMessageBox.warning(
                self,
                "High Risk Warning",
                f"This trade risks {risk_pct:.2f}% of your account.\n\n"
                "Are you sure you want to proceed?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        if margin_util > 80:
            QMessageBox.critical(
                self,
                "Insufficient Margin",
                "Margin utilization would exceed 80%.\n\n"
                "Trade execution cancelled for safety."
            )
            return

        # Emit signal with final parameters
        execution_params = {
            **self.signal_data,
            **self.calculated_params,
        }

        self.execute_requested.emit(execution_params)
        self.accept()

        logger.info(f"Trade execution confirmed: {execution_params.get('symbol')} {execution_params.get('direction')}")

    def get_execution_params(self) -> Dict[str, Any]:
        """Get the final execution parameters."""
        return {
            **self.signal_data,
            **self.calculated_params,
        }
