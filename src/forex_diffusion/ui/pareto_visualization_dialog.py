"""
Pareto Visualization Dialog - Multi-Objective Optimization Results

Visualizes Pareto frontier for hyperparameter optimization results,
showing trade-offs between competing objectives (e.g., accuracy vs complexity,
train time vs performance, etc.)
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import json

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QWidget, QSplitter, QCheckBox, QFileDialog,
    QMessageBox, QHeaderView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from loguru import logger

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - Pareto visualization disabled")


class ParetoVisualizationDialog(QDialog):
    """
    Dialog for visualizing multi-objective optimization results.

    Features:
    - Interactive Pareto frontier plot
    - 2D and 3D visualization options
    - Hyperparameter comparison table
    - Solution selection and export
    - Trade-off analysis
    """

    solution_selected = Signal(dict)  # Emits selected solution configuration

    def __init__(self, parent=None, optimization_results: Optional[List[Dict[str, Any]]] = None):
        super().__init__(parent)

        self.results = optimization_results or []
        self.pareto_solutions = []
        self.selected_solution = None

        self.setWindowTitle("Multi-Objective Optimization Results - Pareto Frontier")
        self.setMinimumWidth(1200)
        self.setMinimumHeight(800)

        if self.results:
            self._compute_pareto_frontier()

        self._build_ui()

        if self.results:
            self._update_visualizations()

    def _build_ui(self):
        """Build the dialog UI"""
        layout = QVBoxLayout(self)

        # Header with summary
        header = self._build_header()
        layout.addWidget(header)

        # Main content: split between plot and table
        splitter = QSplitter(Qt.Vertical)

        # Top: Visualization tabs
        viz_tabs = QTabWidget()

        if MATPLOTLIB_AVAILABLE:
            # Tab 1: 2D Pareto Plot
            pareto_2d_tab = self._build_pareto_2d_tab()
            viz_tabs.addTab(pareto_2d_tab, "Pareto Frontier (2D)")

            # Tab 2: 3D Pareto Plot
            pareto_3d_tab = self._build_pareto_3d_tab()
            viz_tabs.addTab(pareto_3d_tab, "Pareto Frontier (3D)")

            # Tab 3: Hyperparameter Parallel Coordinates
            parallel_tab = self._build_parallel_coordinates_tab()
            viz_tabs.addTab(parallel_tab, "Hyperparameter Analysis")

        splitter.addWidget(viz_tabs)

        # Bottom: Solutions table
        table_widget = self._build_solutions_table()
        splitter.addWidget(table_widget)

        splitter.setSizes([500, 300])

        layout.addWidget(splitter)

        # Footer with actions
        footer = self._build_footer()
        layout.addWidget(footer)

    def _build_header(self) -> QWidget:
        """Build header with summary statistics"""
        header = QGroupBox("Optimization Summary")
        layout = QHBoxLayout(header)

        # Total solutions
        total_label = QLabel(f"Total Solutions: {len(self.results)}")
        total_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(total_label)

        layout.addWidget(QLabel("|"))

        # Pareto optimal solutions
        pareto_label = QLabel(f"Pareto Optimal: {len(self.pareto_solutions)}")
        pareto_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #27ae60;")
        layout.addWidget(pareto_label)

        layout.addWidget(QLabel("|"))

        # Best solution (by primary objective)
        if self.results:
            best_idx = self._find_best_solution()
            best_label = QLabel(f"Best Solution: #{best_idx}")
            best_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #3498db;")
            layout.addWidget(best_label)

        layout.addStretch()

        return header

    def _build_pareto_2d_tab(self) -> QWidget:
        """Build 2D Pareto frontier visualization"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Objective selection
        controls = QHBoxLayout()

        controls.addWidget(QLabel("X-axis:"))
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItems(self._get_objective_names())
        self.x_axis_combo.currentTextChanged.connect(self._update_2d_plot)
        controls.addWidget(self.x_axis_combo)

        controls.addWidget(QLabel("Y-axis:"))
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.addItems(self._get_objective_names())
        if len(self._get_objective_names()) > 1:
            self.y_axis_combo.setCurrentIndex(1)
        self.y_axis_combo.currentTextChanged.connect(self._update_2d_plot)
        controls.addWidget(self.y_axis_combo)

        self.show_dominated = QCheckBox("Show Dominated Solutions")
        self.show_dominated.setChecked(True)
        self.show_dominated.toggled.connect(self._update_2d_plot)
        controls.addWidget(self.show_dominated)

        controls.addStretch()
        layout.addLayout(controls)

        # Matplotlib canvas
        self.figure_2d = Figure(figsize=(10, 6))
        self.canvas_2d = FigureCanvas(self.figure_2d)
        self.ax_2d = self.figure_2d.add_subplot(1, 1, 1)

        # Add matplotlib toolbar
        toolbar = NavigationToolbar(self.canvas_2d, widget)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas_2d)

        return widget

    def _build_pareto_3d_tab(self) -> QWidget:
        """Build 3D Pareto frontier visualization"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Objective selection
        controls = QHBoxLayout()

        controls.addWidget(QLabel("X-axis:"))
        self.x_axis_3d_combo = QComboBox()
        self.x_axis_3d_combo.addItems(self._get_objective_names())
        self.x_axis_3d_combo.currentTextChanged.connect(self._update_3d_plot)
        controls.addWidget(self.x_axis_3d_combo)

        controls.addWidget(QLabel("Y-axis:"))
        self.y_axis_3d_combo = QComboBox()
        self.y_axis_3d_combo.addItems(self._get_objective_names())
        if len(self._get_objective_names()) > 1:
            self.y_axis_3d_combo.setCurrentIndex(1)
        self.y_axis_3d_combo.currentTextChanged.connect(self._update_3d_plot)
        controls.addWidget(self.y_axis_3d_combo)

        controls.addWidget(QLabel("Z-axis:"))
        self.z_axis_3d_combo = QComboBox()
        self.z_axis_3d_combo.addItems(self._get_objective_names())
        if len(self._get_objective_names()) > 2:
            self.z_axis_3d_combo.setCurrentIndex(2)
        self.z_axis_3d_combo.currentTextChanged.connect(self._update_3d_plot)
        controls.addWidget(self.z_axis_3d_combo)

        controls.addStretch()
        layout.addLayout(controls)

        # Matplotlib 3D canvas
        self.figure_3d = Figure(figsize=(10, 6))
        self.canvas_3d = FigureCanvas(self.figure_3d)
        self.ax_3d = self.figure_3d.add_subplot(1, 1, 1, projection='3d')

        toolbar = NavigationToolbar(self.canvas_3d, widget)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas_3d)

        return widget

    def _build_parallel_coordinates_tab(self) -> QWidget:
        """Build parallel coordinates plot for hyperparameters"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Controls
        controls = QHBoxLayout()

        self.normalize_params = QCheckBox("Normalize Parameters")
        self.normalize_params.setChecked(True)
        self.normalize_params.toggled.connect(self._update_parallel_plot)
        controls.addWidget(self.normalize_params)

        self.highlight_pareto = QCheckBox("Highlight Pareto Solutions")
        self.highlight_pareto.setChecked(True)
        self.highlight_pareto.toggled.connect(self._update_parallel_plot)
        controls.addWidget(self.highlight_pareto)

        controls.addStretch()
        layout.addLayout(controls)

        # Matplotlib canvas
        self.figure_parallel = Figure(figsize=(12, 6))
        self.canvas_parallel = FigureCanvas(self.figure_parallel)
        self.ax_parallel = self.figure_parallel.add_subplot(1, 1, 1)

        toolbar = NavigationToolbar(self.canvas_parallel, widget)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas_parallel)

        return widget

    def _build_solutions_table(self) -> QWidget:
        """Build solutions comparison table"""
        widget = QGroupBox("Solutions Comparison")
        layout = QVBoxLayout(widget)

        self.solutions_table = QTableWidget()
        self.solutions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.solutions_table.setSelectionMode(QTableWidget.SingleSelection)
        self.solutions_table.setAlternatingRowColors(True)
        self.solutions_table.itemSelectionChanged.connect(self._on_solution_selected)

        # Build table headers
        self._populate_solutions_table()

        layout.addWidget(self.solutions_table)

        return widget

    def _build_footer(self) -> QWidget:
        """Build footer with action buttons"""
        footer = QWidget()
        layout = QHBoxLayout(footer)

        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)

        self.export_plot_btn = QPushButton("Export Plot")
        self.export_plot_btn.clicked.connect(self._export_plot)

        self.apply_btn = QPushButton("Apply Selected Solution")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_solution)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)

        layout.addWidget(self.export_btn)
        layout.addWidget(self.export_plot_btn)
        layout.addStretch()
        layout.addWidget(self.apply_btn)
        layout.addWidget(self.close_btn)

        return footer

    def _get_objective_names(self) -> List[str]:
        """Get list of objective names from results"""
        if not self.results:
            return ["val_loss", "train_time", "model_size"]

        # Extract objectives from first result
        objectives = []
        for key in self.results[0].keys():
            if key not in ['params', 'hyperparameters', 'config', 'is_pareto']:
                objectives.append(key)

        return objectives if objectives else ["val_loss", "train_time"]

    def _compute_pareto_frontier(self):
        """Compute Pareto optimal solutions"""
        if not self.results:
            return

        # Assume minimization for all objectives
        objectives = self._get_objective_names()

        # Extract objective values
        obj_values = []
        for result in self.results:
            values = [result.get(obj, float('inf')) for obj in objectives]
            obj_values.append(values)

        obj_array = np.array(obj_values)

        # Find Pareto optimal solutions
        is_pareto = np.ones(len(obj_array), dtype=bool)

        for i in range(len(obj_array)):
            if is_pareto[i]:
                # Check if any other solution dominates this one
                is_pareto[is_pareto] = np.any(obj_array[is_pareto] < obj_array[i], axis=1) | \
                                       ~np.all(obj_array[is_pareto] <= obj_array[i], axis=1)
                is_pareto[i] = True

        # Mark Pareto solutions
        for i, result in enumerate(self.results):
            result['is_pareto'] = bool(is_pareto[i])

        self.pareto_solutions = [r for r in self.results if r.get('is_pareto', False)]

        logger.info(f"Found {len(self.pareto_solutions)} Pareto optimal solutions out of {len(self.results)}")

    def _find_best_solution(self) -> int:
        """Find best solution by primary objective (first objective)"""
        if not self.results:
            return -1

        objectives = self._get_objective_names()
        if not objectives:
            return 0

        primary_obj = objectives[0]
        best_idx = 0
        best_val = float('inf')

        for i, result in enumerate(self.results):
            val = result.get(primary_obj, float('inf'))
            if val < best_val:
                best_val = val
                best_idx = i

        return best_idx

    def _update_visualizations(self):
        """Update all visualizations"""
        if MATPLOTLIB_AVAILABLE:
            self._update_2d_plot()
            self._update_3d_plot()
            self._update_parallel_plot()

    def _update_2d_plot(self):
        """Update 2D Pareto frontier plot"""
        if not MATPLOTLIB_AVAILABLE or not self.results:
            return

        self.ax_2d.clear()

        x_obj = self.x_axis_combo.currentText()
        y_obj = self.y_axis_combo.currentText()

        # Extract data
        x_data = [r.get(x_obj, 0) for r in self.results]
        y_data = [r.get(y_obj, 0) for r in self.results]
        is_pareto = [r.get('is_pareto', False) for r in self.results]

        # Plot dominated solutions
        if self.show_dominated.isChecked():
            dominated_x = [x for x, p in zip(x_data, is_pareto) if not p]
            dominated_y = [y for y, p in zip(y_data, is_pareto) if not p]
            self.ax_2d.scatter(dominated_x, dominated_y, c='gray', alpha=0.3,
                             s=50, label='Dominated Solutions')

        # Plot Pareto solutions
        pareto_x = [x for x, p in zip(x_data, is_pareto) if p]
        pareto_y = [y for y, p in zip(y_data, is_pareto) if p]

        if pareto_x:
            self.ax_2d.scatter(pareto_x, pareto_y, c='red', s=100,
                             marker='*', label='Pareto Optimal', zorder=5)

            # Draw Pareto frontier line
            sorted_indices = np.argsort(pareto_x)
            sorted_x = [pareto_x[i] for i in sorted_indices]
            sorted_y = [pareto_y[i] for i in sorted_indices]
            self.ax_2d.plot(sorted_x, sorted_y, 'r--', alpha=0.5, linewidth=2)

        self.ax_2d.set_xlabel(x_obj)
        self.ax_2d.set_ylabel(y_obj)
        self.ax_2d.set_title('Pareto Frontier')
        self.ax_2d.legend()
        self.ax_2d.grid(True, alpha=0.3)

        self.canvas_2d.draw()

    def _update_3d_plot(self):
        """Update 3D Pareto frontier plot"""
        if not MATPLOTLIB_AVAILABLE or not self.results:
            return

        self.ax_3d.clear()

        x_obj = self.x_axis_3d_combo.currentText()
        y_obj = self.y_axis_3d_combo.currentText()
        z_obj = self.z_axis_3d_combo.currentText()

        # Extract data
        x_data = [r.get(x_obj, 0) for r in self.results]
        y_data = [r.get(y_obj, 0) for r in self.results]
        z_data = [r.get(z_obj, 0) for r in self.results]
        is_pareto = [r.get('is_pareto', False) for r in self.results]

        # Plot dominated solutions
        dominated_x = [x for x, p in zip(x_data, is_pareto) if not p]
        dominated_y = [y for y, p in zip(y_data, is_pareto) if not p]
        dominated_z = [z for z, p in zip(z_data, is_pareto) if not p]

        self.ax_3d.scatter(dominated_x, dominated_y, dominated_z,
                          c='gray', alpha=0.3, s=30, label='Dominated Solutions')

        # Plot Pareto solutions
        pareto_x = [x for x, p in zip(x_data, is_pareto) if p]
        pareto_y = [y for y, p in zip(y_data, is_pareto) if p]
        pareto_z = [z for z, p in zip(z_data, is_pareto) if p]

        self.ax_3d.scatter(pareto_x, pareto_y, pareto_z,
                          c='red', s=100, marker='*', label='Pareto Optimal')

        self.ax_3d.set_xlabel(x_obj)
        self.ax_3d.set_ylabel(y_obj)
        self.ax_3d.set_zlabel(z_obj)
        self.ax_3d.set_title('3D Pareto Frontier')
        self.ax_3d.legend()

        self.canvas_3d.draw()

    def _update_parallel_plot(self):
        """Update parallel coordinates plot"""
        if not MATPLOTLIB_AVAILABLE or not self.results:
            return

        self.ax_parallel.clear()

        # Get hyperparameter names
        if not self.results or 'params' not in self.results[0]:
            return

        param_names = list(self.results[0]['params'].keys())
        if not param_names:
            return

        # Extract parameter values
        data = []
        colors = []

        for result in self.results:
            params = result.get('params', {})
            values = [params.get(name, 0) for name in param_names]
            data.append(values)
            colors.append('red' if result.get('is_pareto', False) else 'gray')

        data = np.array(data)

        # Normalize if requested
        if self.normalize_params.isChecked():
            data_min = data.min(axis=0)
            data_max = data.max(axis=0)
            data_range = data_max - data_min
            data_range[data_range == 0] = 1
            data = (data - data_min) / data_range

        # Plot
        x = np.arange(len(param_names))

        for i, (row, color) in enumerate(zip(data, colors)):
            alpha = 0.8 if color == 'red' and self.highlight_pareto.isChecked() else 0.2
            linewidth = 2 if color == 'red' and self.highlight_pareto.isChecked() else 1
            self.ax_parallel.plot(x, row, color=color, alpha=alpha, linewidth=linewidth)

        self.ax_parallel.set_xticks(x)
        self.ax_parallel.set_xticklabels(param_names, rotation=45, ha='right')
        self.ax_parallel.set_ylabel('Normalized Value' if self.normalize_params.isChecked() else 'Value')
        self.ax_parallel.set_title('Hyperparameter Parallel Coordinates')
        self.ax_parallel.grid(True, alpha=0.3, axis='y')

        self.figure_parallel.tight_layout()
        self.canvas_parallel.draw()

    def _populate_solutions_table(self):
        """Populate solutions table with results"""
        if not self.results:
            return

        # Get column names
        objectives = self._get_objective_names()
        param_names = list(self.results[0].get('params', {}).keys()) if 'params' in self.results[0] else []

        columns = ['ID', 'Pareto'] + objectives + param_names
        self.solutions_table.setColumnCount(len(columns))
        self.solutions_table.setHorizontalHeaderLabels(columns)

        # Populate rows
        self.solutions_table.setRowCount(len(self.results))

        for i, result in enumerate(self.results):
            # ID
            self.solutions_table.setItem(i, 0, QTableWidgetItem(str(i)))

            # Pareto optimal marker
            pareto_item = QTableWidgetItem("✓" if result.get('is_pareto', False) else "")
            if result.get('is_pareto', False):
                pareto_item.setBackground(QColor(39, 174, 96, 100))
            self.solutions_table.setItem(i, 1, pareto_item)

            # Objectives
            for j, obj in enumerate(objectives):
                value = result.get(obj, 0)
                item = QTableWidgetItem(f"{value:.6f}" if isinstance(value, float) else str(value))
                self.solutions_table.setItem(i, 2 + j, item)

            # Parameters
            params = result.get('params', {})
            for j, param in enumerate(param_names):
                value = params.get(param, '--')
                item = QTableWidgetItem(str(value))
                self.solutions_table.setItem(i, 2 + len(objectives) + j, item)

        # Resize columns
        self.solutions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _on_solution_selected(self):
        """Handle solution selection in table"""
        selected_rows = self.solutions_table.selectedItems()
        if not selected_rows:
            self.apply_btn.setEnabled(False)
            return

        row = selected_rows[0].row()
        self.selected_solution = self.results[row]
        self.apply_btn.setEnabled(True)

    def _apply_solution(self):
        """Apply selected solution"""
        if self.selected_solution:
            self.solution_selected.emit(self.selected_solution)
            QMessageBox.information(self, "Solution Applied",
                                  "Selected solution has been applied to training configuration.")

    def _export_results(self):
        """Export optimization results to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Optimization Results",
            "optimization_results.json",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )

        if not filename:
            return

        try:
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(self.results, f, indent=2)
            elif filename.endswith('.csv'):
                import csv
                objectives = self._get_objective_names()
                param_names = list(self.results[0].get('params', {}).keys())

                with open(filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['id', 'is_pareto'] + objectives + param_names)
                    writer.writeheader()

                    for i, result in enumerate(self.results):
                        row = {'id': i, 'is_pareto': result.get('is_pareto', False)}
                        row.update({obj: result.get(obj, '') for obj in objectives})
                        row.update(result.get('params', {}))
                        writer.writerow(row)

            logger.info(f"Results exported to {filename}")
            QMessageBox.information(self, "Export Successful", f"Results exported to:\n{filename}")

        except Exception as e:
            logger.exception(f"Failed to export results: {e}")
            QMessageBox.critical(self, "Export Failed", f"Error exporting results:\n{str(e)}")

    def _export_plot(self):
        """Export current plot to image file"""
        if not MATPLOTLIB_AVAILABLE:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "pareto_plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )

        if not filename:
            return

        try:
            self.figure_2d.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Plot exported to {filename}")
            QMessageBox.information(self, "Export Successful", f"Plot exported to:\n{filename}")

        except Exception as e:
            logger.exception(f"Failed to export plot: {e}")
            QMessageBox.critical(self, "Export Failed", f"Error exporting plot:\n{str(e)}")

    def set_results(self, results: List[Dict[str, Any]]):
        """Update dialog with new optimization results"""
        self.results = results
        self._compute_pareto_frontier()
        self._populate_solutions_table()
        self._update_visualizations()
