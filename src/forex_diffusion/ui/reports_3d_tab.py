"""
3D Reports Tab for ForexGPT
Displays interactive 3D visualization reports with file management and descriptions
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QPushButton, QLabel,
    QTextEdit, QGroupBox, QComboBox, QSpinBox,
    QCheckBox, QProgressBar, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, Slot, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QIcon, QFont

import pandas as pd
import numpy as np

# Import our 3D visualization system
try:
    from ..visualization.advanced.visualization_3d import Advanced3DVisualizer
except ImportError:
    Advanced3DVisualizer = None

import logging
logger = logging.getLogger(__name__)


class ReportGeneratorThread(QThread):
    """Thread for generating 3D reports without blocking UI"""
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, report_type: str, data: Dict[str, pd.DataFrame], params: Dict):
        super().__init__()
        self.report_type = report_type
        self.data = data
        self.params = params
        self.visualizer = Advanced3DVisualizer() if Advanced3DVisualizer else None

    def run(self):
        """Generate the report in background"""
        try:
            if not self.visualizer:
                self.error.emit("3D Visualizer not available. Please check installation.")
                return

            self.status.emit(f"Generating {self.report_type} report...")
            self.progress.emit(20)

            result = None
            output_dir = Path("reports_3d")
            output_dir.mkdir(exist_ok=True)

            if self.report_type == "market_surface":
                pairs = self.params.get('pairs', list(self.data.keys())[:5])
                result = self.visualizer.create_3d_market_surface(pairs, self.data)

            elif self.report_type == "correlation_sphere":
                # Prepare correlation matrix
                combined_data = pd.concat([
                    self.data[pair][['close']].rename(columns={'close': pair})
                    for pair in list(self.data.keys())[:10]
                ], axis=1)
                corr_matrix = combined_data.corr()
                result = self.visualizer.create_correlation_sphere(corr_matrix)

            elif self.report_type == "volatility_landscape":
                # Calculate volatility
                vol_data = pd.concat([
                    self.data[pair][['close']].rename(columns={'close': pair})
                    .pct_change().rolling(20).std()
                    for pair in list(self.data.keys())[:8]
                ], axis=1).fillna(0)
                result = self.visualizer.create_volatility_landscape(vol_data)

            elif self.report_type == "heat_map":
                analysis_type = self.params.get('analysis_type', 'correlation')
                combined_data = pd.concat([
                    self.data[pair][['close']].rename(columns={'close': pair})
                    for pair in list(self.data.keys())[:10]
                ], axis=1)
                result = self.visualizer.create_heat_map_analytics(combined_data, analysis_type)

            self.progress.emit(80)

            if result and result.get('success'):
                # Move generated HTML to reports directory
                if 'html_file' in result:
                    source_file = Path(result['html_file'])
                    if source_file.exists():
                        dest_file = output_dir / source_file.name
                        source_file.rename(dest_file)
                        result['html_file'] = str(dest_file)

                self.status.emit(f"Report generated successfully!")
                self.progress.emit(100)
                self.finished.emit(result)
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'Generation failed'
                self.error.emit(f"Failed to generate report: {error_msg}")

        except Exception as e:
            self.error.emit(f"Error generating report: {str(e)}")
            logger.error(f"Report generation error: {e}", exc_info=True)


class Reports3DTab(QWidget):
    """3D Reports Tab with file management and visualization"""

    def __init__(self, data_manager=None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.current_report = None
        self.report_threads = []
        self.reports_dir = Path("reports_3d")
        self.reports_dir.mkdir(exist_ok=True)

        self.report_descriptions = {
            "3d_market_surface": {
                "title": "3D Market Surface",
                "description": "Visualizes price movements across multiple currency pairs over time in a 3D surface plot.",
                "how_to_read": """
                • X-Axis: Time periods (hours/days)
                • Y-Axis: Different currency pairs
                • Z-Axis: Price levels
                • Colors: Price intensity (green=high, red=low)

                INSIGHTS:
                - Identify synchronized movements across pairs
                - Spot divergences and convergences
                - Detect market regime changes
                - Find arbitrage opportunities
                """,
                "use_cases": "Portfolio analysis, trend identification, market structure analysis"
            },
            "correlation_sphere": {
                "title": "Correlation Sphere",
                "description": "Shows correlations between currency pairs in an interactive 3D sphere.",
                "how_to_read": """
                • Points: Each currency pair positioned on sphere
                • Red Lines: Positive correlation (move together)
                • Blue Lines: Negative correlation (move opposite)
                • Line Thickness: Correlation strength

                INSIGHTS:
                - Build diversified portfolios
                - Identify hedging opportunities
                - Understand market relationships
                - Risk management optimization
                """,
                "use_cases": "Portfolio construction, risk hedging, pairs trading"
            },
            "volatility_landscape": {
                "title": "Volatility Landscape",
                "description": "3D terrain map showing volatility patterns across pairs and time.",
                "how_to_read": """
                • Peaks: High volatility periods
                • Valleys: Low volatility periods
                • Colors: Red=extreme volatility, Blue=calm
                • Contours: Volatility levels

                INSIGHTS:
                - Identify risk periods
                - Time entry/exit points
                - Detect market events
                - Volatility clustering patterns
                """,
                "use_cases": "Risk assessment, position sizing, event detection"
            },
            "correlation": {
                "title": "Correlation Heat Map",
                "description": "Matrix showing pairwise correlations between all currency pairs.",
                "how_to_read": """
                • Red: Strong positive correlation (+0.7 to +1.0)
                • Blue: Strong negative correlation (-0.7 to -1.0)
                • White: No correlation (near 0)
                • Numbers: Exact correlation coefficients

                INSIGHTS:
                - Quick correlation overview
                - Portfolio correlation check
                - Relationship strength assessment
                - Diversification opportunities
                """,
                "use_cases": "Quick reference, portfolio review, correlation monitoring"
            }
        }

        self.init_ui()
        self.load_existing_reports()

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)

        # Top controls
        controls_layout = QHBoxLayout()

        # Report generation controls
        gen_group = QGroupBox("Generate New Report")
        gen_layout = QHBoxLayout()

        self.report_type_combo = QComboBox()
        self.report_type_combo.addItems([
            "3D Market Surface",
            "Correlation Sphere",
            "Volatility Landscape",
            "Correlation Heat Map"
        ])
        self.report_type_combo.currentTextChanged.connect(self.on_report_type_changed)

        self.generate_btn = QPushButton("Generate Report")
        self.generate_btn.clicked.connect(self.generate_report)

        self.auto_refresh_cb = QCheckBox("Auto-refresh (5 min)")
        self.auto_refresh_cb.toggled.connect(self.toggle_auto_refresh)

        gen_layout.addWidget(QLabel("Report Type:"))
        gen_layout.addWidget(self.report_type_combo)
        gen_layout.addWidget(self.generate_btn)
        gen_layout.addWidget(self.auto_refresh_cb)
        gen_layout.addStretch()

        gen_group.setLayout(gen_layout)
        controls_layout.addWidget(gen_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("")

        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

        main_layout.addLayout(controls_layout)

        # Main content area - Splitter with file list and viewer
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Report file list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("Generated Reports"))

        self.report_list = QListWidget()
        self.report_list.itemClicked.connect(self.on_report_selected)
        left_layout.addWidget(self.report_list)

        # File management buttons
        file_btns_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh List")
        self.refresh_btn.clicked.connect(self.load_existing_reports)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected_report)

        self.export_btn = QPushButton("Export...")
        self.export_btn.clicked.connect(self.export_report)

        file_btns_layout.addWidget(self.refresh_btn)
        file_btns_layout.addWidget(self.delete_btn)
        file_btns_layout.addWidget(self.export_btn)

        left_layout.addLayout(file_btns_layout)

        # Right panel - Report viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # HTML viewer for reports
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(400)
        right_layout.addWidget(self.web_view)

        # Report description panel
        desc_group = QGroupBox("Report Description")
        desc_layout = QVBoxLayout()

        self.desc_title = QLabel("Select a report to view")
        self.desc_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        desc_layout.addWidget(self.desc_title)

        self.desc_text = QTextEdit()
        self.desc_text.setReadOnly(True)
        self.desc_text.setMaximumHeight(200)
        desc_layout.addWidget(self.desc_text)

        desc_group.setLayout(desc_layout)
        right_layout.addWidget(desc_group)

        # Add panels to splitter
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([300, 700])  # Initial sizes

        main_layout.addWidget(content_splitter)

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_generate_report)

    def load_existing_reports(self):
        """Load list of existing report files"""
        self.report_list.clear()

        if not self.reports_dir.exists():
            return

        # Find all HTML files in reports directory
        html_files = sorted(self.reports_dir.glob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True)

        for file_path in html_files:
            # Parse filename to get report type and timestamp
            filename = file_path.name

            # Determine report type from filename
            report_type = "Unknown"
            icon = "📊"

            if "market_surface" in filename:
                report_type = "3D Market Surface"
                icon = "📈"
            elif "correlation_sphere" in filename:
                report_type = "Correlation Sphere"
                icon = "🌐"
            elif "volatility_landscape" in filename:
                report_type = "Volatility Landscape"
                icon = "⛰️"
            elif "heatmap" in filename or "heat_map" in filename:
                report_type = "Heat Map"
                icon = "🔥"

            # Extract timestamp from filename
            try:
                # Assuming format: report_type_YYYYMMDD_HHMMSS.html
                parts = filename.replace(".html", "").split("_")
                if len(parts) >= 2:
                    date_str = parts[-2]  # YYYYMMDD
                    time_str = parts[-1]  # HHMMSS
                    timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                    time_display = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    time_display = "Unknown time"
            except:
                time_display = file_path.stat().st_mtime
                time_display = datetime.fromtimestamp(time_display).strftime("%Y-%m-%d %H:%M:%S")

            # Add to list
            item_text = f"{icon} {report_type} - {time_display}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, str(file_path))
            self.report_list.addItem(item)

    def on_report_selected(self, item: QListWidgetItem):
        """Handle report selection from list"""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path and Path(file_path).exists():
            self.display_report(file_path)

    def display_report(self, file_path: str):
        """Display selected report in viewer"""
        try:
            path = Path(file_path)
            if not path.exists():
                QMessageBox.warning(self, "File Not Found", f"Report file not found: {file_path}")
                return

            # Load HTML in web view
            url = QUrl.fromLocalFile(str(path.absolute()))
            self.web_view.load(url)

            # Update description based on report type
            filename = path.name.lower()

            # Find matching description
            desc_info = None
            if "market_surface" in filename:
                desc_info = self.report_descriptions["3d_market_surface"]
            elif "correlation_sphere" in filename:
                desc_info = self.report_descriptions["correlation_sphere"]
            elif "volatility_landscape" in filename:
                desc_info = self.report_descriptions["volatility_landscape"]
            elif "correlation" in filename or "heatmap" in filename:
                desc_info = self.report_descriptions["correlation"]

            if desc_info:
                self.desc_title.setText(desc_info["title"])

                desc_text = f"""
<b>Description:</b> {desc_info["description"]}

<b>How to Read:</b>
{desc_info["how_to_read"]}

<b>Use Cases:</b> {desc_info["use_cases"]}

<b>File:</b> {path.name}
<b>Generated:</b> {datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")}
<b>Size:</b> {path.stat().st_size / 1024:.1f} KB
                """

                self.desc_text.setHtml(desc_text.replace('\n', '<br>'))

            self.current_report = file_path

        except Exception as e:
            logger.error(f"Error displaying report: {e}")
            QMessageBox.critical(self, "Error", f"Failed to display report: {str(e)}")

    def generate_report(self):
        """Generate new 3D report with current data"""
        try:
            # Check if data is available
            if not self.data_manager:
                QMessageBox.warning(self, "No Data", "Data manager not available. Please load market data first.")
                return

            # Get current market data
            # This assumes data_manager has a method to get current data
            # You may need to adjust based on your actual data manager implementation
            market_data = self.get_current_market_data()

            if not market_data:
                QMessageBox.warning(self, "No Data", "No market data available. Please ensure data is loaded.")
                return

            # Map combo box text to report types
            report_type_map = {
                "3D Market Surface": "market_surface",
                "Correlation Sphere": "correlation_sphere",
                "Volatility Landscape": "volatility_landscape",
                "Correlation Heat Map": "heat_map"
            }

            report_type = report_type_map.get(self.report_type_combo.currentText(), "market_surface")

            # Prepare parameters
            params = {
                'pairs': list(market_data.keys())[:10],  # Limit to 10 pairs
                'analysis_type': 'correlation'
            }

            # Create and start generator thread
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            thread = ReportGeneratorThread(report_type, market_data, params)
            thread.progress.connect(self.progress_bar.setValue)
            thread.status.connect(self.status_label.setText)
            thread.finished.connect(self.on_report_generated)
            thread.error.connect(self.on_generation_error)
            thread.start()

            self.report_threads.append(thread)
            self.generate_btn.setEnabled(False)

        except Exception as e:
            logger.error(f"Error starting report generation: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate report: {str(e)}")

    def get_current_market_data(self) -> Dict[str, pd.DataFrame]:
        """Get current market data from data manager"""
        try:
            if not self.data_manager:
                return {}

            # This is a placeholder - adjust based on your actual data manager
            # For testing, generate sample data
            pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'NZDUSD', 'USDCAD', 'EURJPY']
            dates = pd.date_range(end=datetime.now(), periods=500, freq='h')

            market_data = {}
            for i, pair in enumerate(pairs):
                np.random.seed(42 + i)
                base_price = 1.1000 + i * 0.1
                returns = np.random.normal(0.0001, 0.01, len(dates))
                prices = base_price * np.exp(np.cumsum(returns))

                market_data[pair] = pd.DataFrame({
                    'open': prices,
                    'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
                    'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
                    'close': prices * (1 + np.random.normal(0, 0.001, len(dates))),
                    'volume': np.random.uniform(10000, 100000, len(dates))
                }, index=dates)

            return market_data

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}

    @Slot(dict)
    def on_report_generated(self, result: Dict):
        """Handle successful report generation"""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Report generated successfully!")

        # Refresh file list
        self.load_existing_reports()

        # Auto-select and display the new report
        if 'html_file' in result and Path(result['html_file']).exists():
            self.display_report(result['html_file'])

        # Show success message
        QMessageBox.information(self, "Success", "3D report generated successfully!")

    @Slot(str)
    def on_generation_error(self, error: str):
        """Handle report generation error"""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_label.setText("")

        QMessageBox.critical(self, "Generation Error", error)

    def delete_selected_report(self):
        """Delete selected report file"""
        current_item = self.report_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a report to delete.")
            return

        file_path = current_item.data(Qt.ItemDataRole.UserRole)
        if not file_path:
            return

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete this report?\n{Path(file_path).name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                Path(file_path).unlink()
                self.load_existing_reports()
                self.web_view.setHtml("")
                self.desc_title.setText("Select a report to view")
                self.desc_text.clear()
                QMessageBox.information(self, "Deleted", "Report deleted successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete report: {str(e)}")

    def export_report(self):
        """Export selected report to chosen location"""
        current_item = self.report_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a report to export.")
            return

        file_path = current_item.data(Qt.ItemDataRole.UserRole)
        if not file_path or not Path(file_path).exists():
            return

        # Get save location from user
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report",
            Path(file_path).name,
            "HTML Files (*.html);;All Files (*.*)"
        )

        if save_path:
            try:
                import shutil
                shutil.copy2(file_path, save_path)
                QMessageBox.information(self, "Exported", f"Report exported to:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export report: {str(e)}")

    def toggle_auto_refresh(self, checked: bool):
        """Toggle automatic report generation"""
        if checked:
            # Start timer for 5 minutes
            self.refresh_timer.start(5 * 60 * 1000)  # 5 minutes in milliseconds
            self.status_label.setText("Auto-refresh enabled (5 min)")
        else:
            self.refresh_timer.stop()
            self.status_label.setText("Auto-refresh disabled")

    def auto_generate_report(self):
        """Automatically generate report on timer"""
        if self.generate_btn.isEnabled():  # Only if not already generating
            self.generate_report()

    def on_report_type_changed(self, report_type: str):
        """Update UI when report type changes"""
        # Could add specific options for each report type here
        pass

    def cleanup(self):
        """Clean up resources when closing"""
        # Stop timer
        self.refresh_timer.stop()

        # Wait for any running threads
        for thread in self.report_threads:
            if thread.isRunning():
                thread.quit()
                thread.wait(1000)