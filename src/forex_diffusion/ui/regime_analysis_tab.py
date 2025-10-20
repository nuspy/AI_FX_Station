"""
Regime Analysis Tab - View and manage regime performance

Shows best models per regime, performance comparisons, and regime definitions.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QSplitter, QAbstractItemView, QMessageBox
)
from PySide6.QtCore import Qt
from typing import Optional, Dict, Any
from loguru import logger

from ..training.training_pipeline.workers import RegimeSummaryWorker


class RegimeAnalysisTab(QWidget):
    """
    Regime Analysis Tab for viewing regime performance.

    Features:
    - Best model per regime table
    - Performance metrics comparison
    - Regime definitions display
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize Regime Analysis Tab."""
        super().__init__(parent)

        self.regime_data: Dict[str, Any] = {}
        self.init_ui()
        self.load_regime_summary()
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()

    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets"""
        from ..i18n.widget_helper import apply_tooltip
        
        if hasattr(self, 'regime_window_spin'):
            apply_tooltip(self.regime_window_spin, "regime_window", "regime_analysis")
        if hasattr(self, 'trend_threshold_spin'):
            apply_tooltip(self.trend_threshold_spin, "trend_threshold", "regime_analysis")
        if hasattr(self, 'volatility_threshold_spin'):
            apply_tooltip(self.volatility_threshold_spin, "volatility_threshold", "regime_analysis")
        if hasattr(self, 'regime_stability_spin'):
            apply_tooltip(self.regime_stability_spin, "regime_stability", "regime_analysis")
        if hasattr(self, 'adaptive_strategy_check'):
            apply_tooltip(self.adaptive_strategy_check, "adaptive_strategy", "regime_analysis")
    

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h2>Regime Analysis</h2>")
        layout.addWidget(title)

        # Refresh button and charts button
        refresh_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.clicked.connect(self.load_regime_summary)
        refresh_layout.addWidget(self.refresh_btn)

        self.charts_btn = QPushButton("View Performance Charts")
        self.charts_btn.clicked.connect(self.show_performance_charts)
        self.charts_btn.setEnabled(False)
        refresh_layout.addWidget(self.charts_btn)

        self.manage_btn = QPushButton("Manage Regimes")
        self.manage_btn.clicked.connect(self.manage_regimes)
        refresh_layout.addWidget(self.manage_btn)

        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)

        # Splitter for regime table and details
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Regime Best Models Table
        regime_table_widget = self.create_regime_table()
        splitter.addWidget(regime_table_widget)

        # Right: Regime Details
        details_widget = self.create_details_section()
        splitter.addWidget(details_widget)

        # Set splitter ratios
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    def create_regime_table(self) -> QWidget:
        """Create regime best models table."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("Best Models per Regime")
        group_layout = QVBoxLayout(group)

        self.regime_table = QTableWidget()
        self.regime_table.setColumnCount(6)
        self.regime_table.setHorizontalHeaderLabels([
            'Regime', 'Has Best Model', 'Sharpe Ratio', 'Max DD',
            'Win Rate', 'Achieved At'
        ])
        self.regime_table.horizontalHeader().setStretchLastSection(True)
        self.regime_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.regime_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.regime_table.itemSelectionChanged.connect(self.on_regime_selected)

        group_layout.addWidget(self.regime_table)
        layout.addWidget(group)

        return widget

    def create_details_section(self) -> QWidget:
        """Create regime details section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Regime Definition
        def_group = QGroupBox("Regime Definition")
        def_layout = QVBoxLayout(def_group)

        self.regime_name_label = QLabel("No regime selected")
        self.regime_name_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        def_layout.addWidget(self.regime_name_label)

        self.regime_desc_label = QLabel("")
        self.regime_desc_label.setWordWrap(True)
        def_layout.addWidget(self.regime_desc_label)

        self.regime_rules_label = QLabel("")
        self.regime_rules_label.setWordWrap(True)
        def_layout.addWidget(self.regime_rules_label)

        layout.addWidget(def_group)

        # Best Model Details
        model_group = QGroupBox("Best Model Details")
        model_layout = QVBoxLayout(model_group)

        self.model_details_label = QLabel("No model selected")
        self.model_details_label.setWordWrap(True)
        model_layout.addWidget(self.model_details_label)

        layout.addWidget(model_group)

        # Performance Metrics
        metrics_group = QGroupBox("Secondary Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_label = QLabel("No metrics available")
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)

        layout.addWidget(metrics_group)

        layout.addStretch()

        return widget

    def load_regime_summary(self):
        """Load regime summary data in background."""
        self.refresh_btn.setEnabled(False)
        self.regime_name_label.setText("Loading...")

        worker = RegimeSummaryWorker()
        worker.finished.connect(self.on_regime_data_loaded)
        worker.error.connect(self.on_load_error)
        worker.start()

    def on_regime_data_loaded(self, data: Dict[str, Any]):
        """Handle regime data load completion."""
        self.regime_data = data
        self.refresh_btn.setEnabled(True)

        # Populate table
        regimes = data.get('regimes', {})
        self.regime_table.setRowCount(len(regimes))

        row = 0
        for regime_name, regime_info in regimes.items():
            # Regime name
            self.regime_table.setItem(row, 0, QTableWidgetItem(regime_name))

            # Has best model
            has_model = "✓" if regime_info.get('has_best_model') else "✗"
            has_model_item = QTableWidgetItem(has_model)
            has_model_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.regime_table.setItem(row, 1, has_model_item)

            # Metrics (if available)
            if regime_info.get('best_model'):
                best_model = regime_info['best_model']
                secondary = best_model.get('secondary_metrics', {})

                sharpe = f"{best_model.get('performance_score', 0):.3f}"
                max_dd = f"{secondary.get('max_drawdown', 0):.3f}"
                win_rate = f"{secondary.get('win_rate', 0):.1%}"
                achieved = best_model.get('achieved_at', 'N/A')[:19]  # Trim timestamp

                self.regime_table.setItem(row, 2, QTableWidgetItem(sharpe))
                self.regime_table.setItem(row, 3, QTableWidgetItem(max_dd))
                self.regime_table.setItem(row, 4, QTableWidgetItem(win_rate))
                self.regime_table.setItem(row, 5, QTableWidgetItem(achieved))
            else:
                for col in range(2, 6):
                    self.regime_table.setItem(row, col, QTableWidgetItem("N/A"))

            row += 1

        # Resize columns
        self.regime_table.resizeColumnsToContents()

        # Enable charts button if we have data
        self.charts_btn.setEnabled(len(regimes) > 0)

        logger.info(f"Loaded {len(regimes)} regimes")

    def on_load_error(self, error_msg: str):
        """Handle regime data load error."""
        self.refresh_btn.setEnabled(True)
        self.regime_name_label.setText("Error loading data")

        logger.error(f"Failed to load regime data: {error_msg}")
        QMessageBox.critical(
            self,
            "Error",
            f"Failed to load regime data:\n{error_msg}"
        )

    def on_regime_selected(self):
        """Handle regime selection in table."""
        selected_rows = self.regime_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        row = selected_rows[0].row()
        regime_name = self.regime_table.item(row, 0).text()

        # Get regime data
        regimes = self.regime_data.get('regimes', {})
        regime_info = regimes.get(regime_name, {})

        # Update regime definition
        self.regime_name_label.setText(regime_name.replace('_', ' ').title())
        self.regime_desc_label.setText(
            f"Description: {regime_info.get('description', 'N/A')}"
        )

        # Show detection rules (if available)
        # Note: detection_rules is stored as JSON in the database
        rules_text = "Detection Rules: Complex rules (see database)"
        self.regime_rules_label.setText(rules_text)

        # Update best model details
        if regime_info.get('has_best_model') and regime_info.get('best_model'):
            best_model = regime_info['best_model']

            model_text = (
                f"Training Run ID: {best_model.get('training_run_id', 'N/A')}\n"
                f"Performance Score: {best_model.get('performance_score', 0):.4f}\n"
                f"Achieved: {best_model.get('achieved_at', 'N/A')[:19]}"
            )
            self.model_details_label.setText(model_text)

            # Update secondary metrics
            secondary = best_model.get('secondary_metrics', {})
            metrics_text = ""
            for metric, value in secondary.items():
                if isinstance(value, (int, float)):
                    if 'rate' in metric.lower() or 'pct' in metric.lower():
                        metrics_text += f"{metric}: {value:.2%}\n"
                    else:
                        metrics_text += f"{metric}: {value:.4f}\n"
                else:
                    metrics_text += f"{metric}: {value}\n"

            self.metrics_label.setText(metrics_text if metrics_text else "No metrics available")
        else:
            self.model_details_label.setText("No best model for this regime yet")
            self.metrics_label.setText("Train models to establish best performers")

    def show_performance_charts(self):
        """Show performance charts dialog."""
        if not self.regime_data or not self.regime_data.get('regimes'):
            QMessageBox.warning(
                self,
                "No Data",
                "No regime data available for charting.\nLoad data first."
            )
            return

        try:
            from PySide6.QtWidgets import QDialog
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            dialog = QDialog(self)
            dialog.setWindowTitle("Regime Performance Charts")
            dialog.resize(1000, 700)

            layout = QVBoxLayout(dialog)

            # Create matplotlib figure with subplots
            fig = Figure(figsize=(10, 8))

            regimes = self.regime_data.get('regimes', {})
            regime_names = list(regimes.keys())
            n_regimes = len(regime_names)

            # Extract metrics for each regime
            sharpe_ratios = []
            max_drawdowns = []
            win_rates = []
            has_model = []

            for regime_name in regime_names:
                regime_info = regimes[regime_name]

                if regime_info.get('has_best_model') and regime_info.get('best_model'):
                    best_model = regime_info['best_model']
                    secondary = best_model.get('secondary_metrics', {})

                    sharpe_ratios.append(best_model.get('performance_score', 0))
                    max_drawdowns.append(abs(secondary.get('max_drawdown', 0)))
                    win_rates.append(secondary.get('win_rate', 0) * 100)  # Convert to percentage
                    has_model.append(True)
                else:
                    sharpe_ratios.append(0)
                    max_drawdowns.append(0)
                    win_rates.append(0)
                    has_model.append(False)

            # Clean regime names for display
            display_names = [name.replace('_', ' ').title() for name in regime_names]

            # Chart 1: Sharpe Ratio Comparison
            ax1 = fig.add_subplot(2, 2, 1)
            colors = ['green' if has else 'lightgray' for has in has_model]
            bars1 = ax1.bar(display_names, sharpe_ratios, color=colors)
            ax1.set_title('Sharpe Ratio by Regime', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Sharpe Ratio')
            ax1.set_xlabel('Market Regime')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.grid(axis='y', alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Chart 2: Max Drawdown Comparison
            ax2 = fig.add_subplot(2, 2, 2)
            bars2 = ax2.bar(display_names, max_drawdowns, color=colors)
            ax2.set_title('Max Drawdown by Regime', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Max Drawdown (abs)')
            ax2.set_xlabel('Market Regime')
            ax2.grid(axis='y', alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Chart 3: Win Rate Comparison
            ax3 = fig.add_subplot(2, 2, 3)
            bars3 = ax3.bar(display_names, win_rates, color=colors)
            ax3.set_title('Win Rate by Regime', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Win Rate (%)')
            ax3.set_xlabel('Market Regime')
            ax3.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% Baseline')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Chart 4: Performance Summary (Radar Chart)
            ax4 = fig.add_subplot(2, 2, 4, projection='polar')

            # Select regime with best model or first regime
            selected_regime_idx = 0
            for i, has in enumerate(has_model):
                if has:
                    selected_regime_idx = i
                    break

            if has_model[selected_regime_idx]:
                # Normalize metrics to 0-1 scale for radar chart
                norm_sharpe = min(max(sharpe_ratios[selected_regime_idx] / 3.0, 0), 1)  # Assume 3.0 is excellent
                norm_dd = 1 - min(max_drawdowns[selected_regime_idx], 1)  # Lower is better, so invert
                norm_winrate = win_rates[selected_regime_idx] / 100.0

                categories = ['Sharpe\nRatio', 'Drawdown\nControl', 'Win\nRate']
                values = [norm_sharpe, norm_dd, norm_winrate]

                # Repeat first value to close the circle
                values += values[:1]

                angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
                angles += angles[:1]

                ax4.plot(angles, values, 'o-', linewidth=2, color='blue', label=display_names[selected_regime_idx])
                ax4.fill(angles, values, alpha=0.25, color='blue')
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels(categories)
                ax4.set_ylim(0, 1)
                ax4.set_title('Best Model Performance Profile', fontweight='bold', fontsize=12, pad=20)
                ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                ax4.grid(True)
            else:
                ax4.text(0.5, 0.5, 'No model data\navailable',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax4.transAxes,
                        fontsize=14)
                ax4.set_xticks([])
                ax4.set_yticks([])

            fig.tight_layout()

            # Create canvas and add to dialog
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)

            # Add close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)

            dialog.exec()

        except Exception as e:
            logger.error(f"Failed to create charts: {e}")
            QMessageBox.critical(
                self,
                "Chart Error",
                f"Failed to create performance charts:\n{e}"
            )

    def manage_regimes(self):
        """Open regime definition manager."""
        try:
            from .regime_definition_dialog import RegimeDefinitionDialog

            dialog = RegimeDefinitionDialog(self)
            dialog.exec()

            # Reload regime data after dialog closes
            self.load_regime_summary()

        except Exception as e:
            logger.error(f"Failed to open regime manager: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open regime manager:\n{e}"
            )
