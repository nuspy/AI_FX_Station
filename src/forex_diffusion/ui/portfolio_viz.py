"""
Efficient Frontier Visualization Widget for Portfolio Optimization.

Displays risk-return tradeoff with interactive plotting.
"""

from typing import Optional, Dict
import pandas as pd
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal
from loguru import logger

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not available for portfolio visualization")


class EfficientFrontierWidget(QWidget):
    """
    Widget for visualizing efficient frontier and portfolio statistics.

    Features:
    - Interactive efficient frontier plot
    - Current portfolio marker
    - Risk-return scatter for individual assets
    - Sharpe ratio isoclines
    """

    point_clicked = Signal(dict)  # Emitted when a point on frontier is clicked

    def __init__(self, parent=None):
        super().__init__(parent)

        self.frontier_data: Optional[pd.DataFrame] = None
        self.current_portfolio: Optional[Dict] = None
        self.asset_data: Optional[pd.DataFrame] = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        if not _HAS_MATPLOTLIB:
            error_label = QLabel("Matplotlib not installed. Install with: pip install matplotlib")
            layout.addWidget(error_label)
            return

        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout.addWidget(self.canvas)

        # Control buttons
        controls = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh Plot")
        self.refresh_btn.clicked.connect(self._refresh_plot)
        controls.addWidget(self.refresh_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_plot)
        controls.addWidget(self.clear_btn)

        controls.addStretch()

        self.status_label = QLabel("Ready")
        controls.addWidget(self.status_label)

        layout.addLayout(controls)

    def plot_efficient_frontier(
        self,
        frontier_data: pd.DataFrame,
        current_portfolio: Optional[Dict] = None,
        asset_data: Optional[pd.DataFrame] = None,
    ):
        """
        Plot efficient frontier with optional current portfolio marker.

        Args:
            frontier_data: DataFrame with columns [return, risk, sharpe]
            current_portfolio: Dict with keys {return, risk, sharpe, weights}
            asset_data: DataFrame with columns [return, risk] for individual assets
        """
        if not _HAS_MATPLOTLIB:
            return

        self.frontier_data = frontier_data
        self.current_portfolio = current_portfolio
        self.asset_data = asset_data

        self._refresh_plot()

    def _refresh_plot(self):
        """Refresh the plot with current data."""
        if not _HAS_MATPLOTLIB or self.frontier_data is None:
            return

        try:
            self.ax.clear()

            # Plot efficient frontier
            returns = self.frontier_data.get("return", self.frontier_data.get("Return"))
            risks = self.frontier_data.get("risk", self.frontier_data.get("Volatility"))

            if returns is None or risks is None:
                logger.warning("Frontier data missing return/risk columns")
                return

            # Convert to percentage for better readability
            returns_pct = returns * 100
            risks_pct = risks * 100

            self.ax.plot(
                risks_pct, returns_pct,
                'b-', linewidth=2, label='Efficient Frontier'
            )

            # Plot individual assets if available
            if self.asset_data is not None:
                asset_returns = self.asset_data.get("return", self.asset_data.get("Return"))
                asset_risks = self.asset_data.get("risk", self.asset_data.get("Volatility"))

                if asset_returns is not None and asset_risks is not None:
                    self.ax.scatter(
                        asset_risks * 100, asset_returns * 100,
                        c='gray', alpha=0.6, s=50, label='Individual Assets'
                    )

                    # Add asset labels
                    for idx, (risk, ret) in enumerate(zip(asset_risks * 100, asset_returns * 100)):
                        asset_name = self.asset_data.index[idx] if hasattr(self.asset_data, 'index') else f"Asset {idx}"
                        self.ax.annotate(
                            asset_name, (risk, ret),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.7
                        )

            # Plot current portfolio if available
            if self.current_portfolio:
                port_return = self.current_portfolio.get("return", self.current_portfolio.get("expected_return", 0))
                port_risk = self.current_portfolio.get("risk", self.current_portfolio.get("volatility", 0))

                self.ax.scatter(
                    port_risk * 100, port_return * 100,
                    c='red', marker='*', s=300, label='Current Portfolio',
                    edgecolors='black', linewidths=1.5, zorder=5
                )

                # Add Sharpe ratio annotation
                sharpe = self.current_portfolio.get("sharpe_ratio", 0)
                self.ax.annotate(
                    f'Sharpe: {sharpe:.2f}',
                    (port_risk * 100, port_return * 100),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
                )

            # Plot Sharpe ratio isoclines (lines of constant Sharpe ratio)
            if "sharpe" in self.frontier_data.columns or "Sharpe" in self.frontier_data.columns:
                sharpe_col = "sharpe" if "sharpe" in self.frontier_data.columns else "Sharpe"
                max_sharpe_idx = self.frontier_data[sharpe_col].idxmax()
                max_sharpe_point = self.frontier_data.loc[max_sharpe_idx]

                max_sharpe_return = max_sharpe_point.get("return", max_sharpe_point.get("Return")) * 100
                max_sharpe_risk = max_sharpe_point.get("risk", max_sharpe_point.get("Volatility")) * 100

                self.ax.scatter(
                    max_sharpe_risk, max_sharpe_return,
                    c='green', marker='D', s=150, label='Max Sharpe',
                    edgecolors='black', linewidths=1, zorder=4
                )

            # Formatting
            self.ax.set_xlabel('Risk (Annualized Volatility %)', fontsize=12)
            self.ax.set_ylabel('Expected Return (Annual %)', fontsize=12)
            self.ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
            self.ax.legend(loc='best', fontsize=10)
            self.ax.grid(True, alpha=0.3)

            # Set axis limits with some padding
            if len(risks_pct) > 0 and len(returns_pct) > 0:
                risk_range = risks_pct.max() - risks_pct.min()
                return_range = returns_pct.max() - returns_pct.min()

                self.ax.set_xlim(
                    risks_pct.min() - 0.1 * risk_range,
                    risks_pct.max() + 0.1 * risk_range
                )
                self.ax.set_ylim(
                    returns_pct.min() - 0.1 * return_range,
                    returns_pct.max() + 0.1 * return_range
                )

            self.canvas.draw()
            self.status_label.setText(f"Frontier plotted: {len(self.frontier_data)} points")
            logger.info("Efficient frontier plot updated")

        except Exception as e:
            logger.error(f"Failed to plot efficient frontier: {e}")
            self.status_label.setText(f"Plot error: {str(e)}")

    def clear_plot(self):
        """Clear the plot."""
        if not _HAS_MATPLOTLIB:
            return

        self.ax.clear()
        self.ax.set_xlabel('Risk (Annualized Volatility %)')
        self.ax.set_ylabel('Expected Return (Annual %)')
        self.ax.set_title('Efficient Frontier')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

        self.frontier_data = None
        self.current_portfolio = None
        self.asset_data = None
        self.status_label.setText("Plot cleared")

    def export_plot(self, filename: str):
        """
        Export plot to file.

        Args:
            filename: Output filename (PNG, PDF, SVG supported)
        """
        if not _HAS_MATPLOTLIB:
            return

        try:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Plot exported to {filename}")
            self.status_label.setText(f"Exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export plot: {e}")
            self.status_label.setText(f"Export error: {str(e)}")
