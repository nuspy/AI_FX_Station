"""
Enhanced Indicators Dialog with Complete VectorBT Pro + bta-lib Integration
Supports 200+ indicators with advanced categorization, scrollable interface, and subplot recommendations
"""
from __future__ import annotations

import json
from typing import Dict, List

from loguru import logger
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QWidget,
    QComboBox, QLineEdit, QMessageBox, QColorDialog, QScrollArea,
    QTreeWidget, QTreeWidgetItem, QSplitter, QTextEdit
)
from PySide6.QtGui import QColor, QFont
from PySide6.QtCore import Qt, Signal

from ..features.indicators_talib import (
    TALibIndicators as BTALibIndicators, IndicatorConfig, IndicatorCategories
)
from ..features.indicator_ranges import (
    indicator_range_classifier
)

# Persistenza settings
try:
    from ..utils.user_settings import get_setting, set_setting
except Exception:
    def get_setting(key: str, default=None):
        return default
    def set_setting(key: str, value):
        pass


class SubplotRecommendationWidget(QWidget):
    """Widget showing subplot recommendations for indicators"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Subplot Recommendations")
        title.setFont(QFont("", -1, QFont.Bold))
        layout.addWidget(title)

        # Recommendations text
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setMaximumHeight(120)
        self.recommendations_text.setReadOnly(True)
        layout.addWidget(self.recommendations_text)

        self.update_recommendations()

    def update_recommendations(self):
        """Update recommendations text"""
        recommendations = {
            "main_chart": "Price overlay indicators (Moving Averages, Bollinger Bands, SAR)",
            "normalized_subplot": "Normalized indicators 0-100 range (RSI, Stochastic, Williams %R)",
            "volume_subplot": "Volume-based indicators (OBV, A/D Line)",
            "custom_subplot": "Custom range indicators (MACD, CCI, Momentum)"
        }

        text = ""
        for subplot_type, description in recommendations.items():
            indicators = indicator_range_classifier.get_indicators_by_subplot(subplot_type)
            count = len(indicators)
            text += f"<b>{subplot_type.replace('_', ' ').title()}</b> ({count} indicators)<br>"
            text += f"<i>{description}</i><br><br>"

        self.recommendations_text.setHtml(text)


class EnhancedIndicatorTreeWidget(QTreeWidget):
    """Enhanced tree widget with search, filtering, and subplot categorization"""

    indicatorToggled = Signal(str, bool)  # indicator_name, enabled

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setHeaderLabels(["Indicator", "Range", "Subplot", "Color", "Status"])
        self.setColumnWidth(0, 250)
        self.setColumnWidth(1, 120)
        self.setColumnWidth(2, 120)
        self.setColumnWidth(3, 60)  # Color column
        self.setColumnWidth(4, 80)  # Status column

        # Allow sorting
        self.setSortingEnabled(True)

        # Enable double-click to change color
        self.itemDoubleClicked.connect(self.on_item_double_clicked)

        # Connect checkbox changes to update config
        self.itemChanged.connect(self.on_item_changed)

        self.indicators_system = None
        self.category_items = {}
        self.indicator_items = {}
        self.range_classifier = indicator_range_classifier

    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double click on item to change color"""
        # Only handle indicator items (not category items)
        if not isinstance(item, EnhancedIndicatorTreeWidgetItem):
            return

        # Only if clicked on color column
        if column == 3:
            self._change_indicator_color(item)

    def _change_indicator_color(self, item: EnhancedIndicatorTreeWidgetItem):
        """Open color picker dialog for indicator"""
        current_color = QColor(item.config.color)
        color = QColorDialog.getColor(current_color, self, f"Choose Color for {item.config.display_name}")

        if color.isValid():
            item.config.color = color.name()
            item._update_color_display()
            # Signal that color changed
            self.indicatorToggled.emit(item.name, item.config.enabled)

    def on_item_changed(self, item: QTreeWidgetItem, column: int):
        """Handle checkbox state changes"""
        # Only handle indicator items (not category items) and only column 0 (checkbox)
        if not isinstance(item, EnhancedIndicatorTreeWidgetItem) or column != 0:
            return

        # Update config.enabled based on checkbox state
        is_checked = item.checkState(0) == Qt.CheckState.Checked
        item.config.enabled = is_checked

        # Update status column
        item.setText(4, "Enabled" if is_checked else "Disabled")

        # Emit signal for external listeners
        self.indicatorToggled.emit(item.name, is_checked)

    def setup_indicators(self, indicators_system: BTALibIndicators):
        """Setup tree with indicators from the system"""
        self.indicators_system = indicators_system
        self.clear()
        self.category_items.clear()
        self.indicator_items.clear()

        # Create category items with enhanced information
        categories = [
            IndicatorCategories.OVERLAP,
            IndicatorCategories.MOMENTUM,
            IndicatorCategories.VOLATILITY,
            IndicatorCategories.TREND,
            IndicatorCategories.VOLUME,
            IndicatorCategories.PRICE_TRANSFORM,
            IndicatorCategories.STATISTICS,
            IndicatorCategories.CYCLE,
            IndicatorCategories.PATTERN
        ]

        for category in categories:
            category_item = QTreeWidgetItem(self, [category, "", "", ""])
            category_item.setFont(0, QFont("", -1, QFont.Bold))
            category_item.setExpanded(True)
            self.category_items[category] = category_item

            # Add indicators for this category
            category_indicators = indicators_system.get_indicators_by_category(category)

            for name, config in category_indicators.items():
                # Get range information
                range_info = self.range_classifier.get_range_info(name)
                range_text = range_info.typical_range if range_info else "Unknown"
                subplot_rec = range_info.subplot_recommendation if range_info else "custom_subplot"

                # Create indicator item
                indicator_item = EnhancedIndicatorTreeWidgetItem(
                    category_item, name, config, range_text, subplot_rec
                )
                self.indicator_items[name] = indicator_item

        # Update counts
        self.update_category_counts()

    def update_category_counts(self):
        """Update category headers with counts"""
        for category, item in self.category_items.items():
            total = item.childCount()
            enabled = sum(1 for i in range(total)
                         if item.child(i).config.enabled)
            item.setText(0, f"{category} ({enabled}/{total})")

    def filter_by_subplot(self, subplot_type: str):
        """Filter indicators by subplot recommendation"""
        for name, item in self.indicator_items.items():
            if subplot_type == "all":
                # Show all indicators
                item.setHidden(False)
            else:
                # Filter by subplot type
                range_info = self.range_classifier.get_range_info(name)
                if range_info:
                    should_show = (range_info.subplot_recommendation == subplot_type)
                    item.setHidden(not should_show)
                else:
                    # If no range info, hide by default when filtering
                    item.setHidden(True)

        # Hide empty categories
        for category_item in self.category_items.values():
            has_visible = any(not category_item.child(i).isHidden()
                            for i in range(category_item.childCount()))
            category_item.setHidden(not has_visible)

    def search_indicators(self, search_text: str):
        """Search indicators by name or description"""
        search_text = search_text.lower()
        for name, item in self.indicator_items.items():
            matches = (search_text in name.lower() or
                      search_text in item.config.display_name.lower() or
                      search_text in item.config.description.lower())
            item.setHidden(not matches)


class EnhancedIndicatorTreeWidgetItem(QTreeWidgetItem):
    """Enhanced tree widget item for indicators with range and subplot info"""

    def __init__(self, parent: QTreeWidgetItem, name: str, config: IndicatorConfig,
                 range_text: str, subplot_rec: str):
        super().__init__(parent, [config.display_name, range_text,
                                 subplot_rec.replace('_', ' ').title(), "", ""])
        self.name = name
        self.config = config
        self.range_text = range_text
        self.subplot_recommendation = subplot_rec

        # Initialize color if not set
        if not hasattr(config, 'color') or not config.color:
            config.color = self._get_default_color(subplot_rec)

        # Set up checkbox for enabling/disabling
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.setCheckState(0, Qt.CheckState.Checked if config.enabled else Qt.CheckState.Unchecked)

        # Set initial status text
        self.setText(4, "Enabled" if config.enabled else "Disabled")

        # Show color in the color column
        self._update_color_display()

    def _get_default_color(self, subplot_rec: str) -> str:
        """Get default color based on subplot type"""
        default_colors = {
            "main_chart": "#6495ED",         # Cornflower Blue
            "normalized_subplot": "#3CB371", # Medium Sea Green
            "volume_subplot": "#FFA500",     # Orange
            "custom_subplot": "#9370DB"      # Medium Purple
        }
        return default_colors.get(subplot_rec, "#6495ED")

    def _update_color_display(self):
        """Update color display in the tree"""
        color = QColor(self.config.color)
        self.setBackground(3, color)  # Color column
        # Set text color to contrast
        luminance = (0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()) / 255
        text_color = QColor(0, 0, 0) if luminance > 0.5 else QColor(255, 255, 255)
        self.setForeground(3, text_color)
        self.setText(3, "■")  # Color indicator

        # Color coding by subplot type
        subplot_colors = {
            "main_chart": QColor(100, 149, 237),      # Cornflower Blue
            "normalized_subplot": QColor(60, 179, 113), # Medium Sea Green
            "volume_subplot": QColor(255, 165, 0),     # Orange
            "custom_subplot": QColor(147, 112, 219)    # Medium Purple
        }

        if self.subplot_recommendation in subplot_colors:
            bg_color = subplot_colors[self.subplot_recommendation]
            self.setBackground(2, bg_color)
            self.setForeground(2, QColor(255, 255, 255))  # White text

        # Tooltip with description
        tooltip = f"<b>{self.config.display_name}</b><br>"
        tooltip += f"Range: {self.range_text}<br>"
        tooltip += f"Subplot: {self.subplot_recommendation.replace('_', ' ').title()}<br>"
        tooltip += f"Data Requirement: {self.config.data_requirement.value}<br>"
        tooltip += f"Description: {self.config.description}"
        self.setToolTip(0, tooltip)


class EnhancedIndicatorsDialog(QDialog):
    """
    Enhanced indicators dialog with complete VectorBT Pro + bta-lib integration
    Features:
    - 200+ indicators with categorization
    - Scrollable interface with search
    - Subplot recommendations
    - Range-based filtering
    - Bulk enable/disable operations
    """

    def __init__(self, parent=None, available_data: List[str] = None):
        super().__init__(parent)
        self.available_data = available_data or ['open', 'high', 'low', 'close']
        self.indicators_system = BTALibIndicators(self.available_data)

        self.setWindowTitle("Enhanced Indicators Configuration - 200+ Professional Indicators")
        self.setModal(True)

        # Load geometry from settings or use defaults
        geometry = get_setting("indicators.dialog.geometry", None)
        if geometry and len(geometry) == 4:
            self.setGeometry(geometry[0], geometry[1], geometry[2], geometry[3])
        else:
            self.resize(1000, 700)

        self.initUI()
        self.load_settings()

    def initUI(self):
        """Initialize the enhanced UI"""
        main_layout = QVBoxLayout(self)

        # Top controls
        controls_layout = QHBoxLayout()

        # Search box
        search_label = QLabel("Search:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search indicators by name or description...")
        self.search_box.textChanged.connect(self.on_search_changed)
        controls_layout.addWidget(search_label)
        controls_layout.addWidget(self.search_box)

        # Filter by subplot
        filter_label = QLabel("Filter by Subplot:")
        self.subplot_filter = QComboBox()
        self.subplot_filter.addItems([
            "all", "main_chart", "normalized_subplot", "volume_subplot", "custom_subplot"
        ])
        self.subplot_filter.currentTextChanged.connect(self.on_filter_changed)
        controls_layout.addWidget(filter_label)
        controls_layout.addWidget(self.subplot_filter)

        # Bulk operations
        self.enable_all_btn = QPushButton("Enable All")
        self.disable_all_btn = QPushButton("Disable All")
        self.enable_all_btn.clicked.connect(self.enable_all_indicators)
        self.disable_all_btn.clicked.connect(self.disable_all_indicators)
        controls_layout.addWidget(self.enable_all_btn)
        controls_layout.addWidget(self.disable_all_btn)

        main_layout.addLayout(controls_layout)

        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left side: Indicators tree (scrollable)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Data availability info
        data_info = QLabel(f"Available Data: {', '.join(self.available_data)}")
        data_info.setStyleSheet("QLabel { background: #f0f0f0; padding: 5px; border-radius: 3px; }")
        left_layout.addWidget(data_info)

        # Scrollable indicators tree
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(500)

        self.indicators_tree = EnhancedIndicatorTreeWidget()
        self.indicators_tree.setup_indicators(self.indicators_system)
        scroll_area.setWidget(self.indicators_tree)
        left_layout.addWidget(scroll_area)

        # Statistics
        self.stats_label = QLabel()
        self.update_statistics()
        left_layout.addWidget(self.stats_label)

        splitter.addWidget(left_widget)

        # Right side: Configuration and recommendations
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Subplot recommendations
        self.subplot_widget = SubplotRecommendationWidget()
        right_layout.addWidget(self.subplot_widget)

        # Performance impact info
        perf_info = QGroupBox("Performance Impact")
        perf_layout = QVBoxLayout(perf_info)
        perf_text = QLabel(
            "• Normalized indicators (0-100 range): Low impact\n"
            "• Price overlay indicators: Low impact\n"
            "• Volume indicators: Medium impact\n"
            "• Custom range indicators: Medium impact\n"
            "• Total recommended limit: 50 active indicators"
        )
        perf_layout.addWidget(perf_text)
        right_layout.addWidget(perf_info)

        # Settings export/import
        settings_group = QGroupBox("Settings Management")
        settings_layout = QVBoxLayout(settings_group)

        export_btn = QPushButton("Export Configuration")
        import_btn = QPushButton("Import Configuration")
        reset_btn = QPushButton("Reset to Defaults")

        export_btn.clicked.connect(self.export_configuration)
        import_btn.clicked.connect(self.import_configuration)
        reset_btn.clicked.connect(self.reset_to_defaults)

        settings_layout.addWidget(export_btn)
        settings_layout.addWidget(import_btn)
        settings_layout.addWidget(reset_btn)
        right_layout.addWidget(settings_group)

        right_layout.addStretch()
        splitter.addWidget(right_widget)

        # Bottom buttons
        buttons_layout = QHBoxLayout()

        self.ok_button = QPushButton("Apply & Close")
        self.cancel_button = QPushButton("Cancel")
        self.apply_button = QPushButton("Apply")

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.apply_settings)

        buttons_layout.addStretch()
        buttons_layout.addWidget(self.apply_button)
        buttons_layout.addWidget(self.ok_button)
        buttons_layout.addWidget(self.cancel_button)

        main_layout.addLayout(buttons_layout)

    def on_search_changed(self, text: str):
        """Handle search text changes"""
        self.indicators_tree.search_indicators(text)
        self.update_statistics()

    def on_filter_changed(self, subplot_type: str):
        """Handle subplot filter changes"""
        self.indicators_tree.filter_by_subplot(subplot_type)
        self.update_statistics()

    def enable_all_indicators(self):
        """Enable all visible indicators"""
        for name, item in self.indicators_tree.indicator_items.items():
            if not item.isHidden():
                item.setCheckState(0, Qt.CheckState.Checked)
                item.config.enabled = True
        self.update_statistics()

    def disable_all_indicators(self):
        """Disable all visible indicators"""
        for name, item in self.indicators_tree.indicator_items.items():
            if not item.isHidden():
                item.setCheckState(0, Qt.CheckState.Unchecked)
                item.config.enabled = False
        self.update_statistics()

    def update_statistics(self):
        """Update statistics display"""
        total_indicators = len(self.indicators_tree.indicator_items)
        enabled_indicators = sum(1 for item in self.indicators_tree.indicator_items.values()
                               if item.config.enabled)
        visible_indicators = sum(1 for item in self.indicators_tree.indicator_items.values()
                               if not item.isHidden())

        stats_text = f"Total: {total_indicators} | Enabled: {enabled_indicators} | Visible: {visible_indicators}"
        self.stats_label.setText(stats_text)

    def export_configuration(self):
        """Export current configuration to JSON"""
        config = {}
        for name, item in self.indicators_tree.indicator_items.items():
            config[name] = {
                'enabled': item.config.enabled,
                'parameters': item.config.parameters
            }

        # Save to file (simplified implementation)
        try:
            with open('indicator_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Export", "Configuration exported to indicator_config.json")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export: {e}")

    def import_configuration(self):
        """Import configuration from JSON"""
        try:
            with open('indicator_config.json', 'r') as f:
                config = json.load(f)

            for name, settings in config.items():
                if name in self.indicators_tree.indicator_items:
                    item = self.indicators_tree.indicator_items[name]
                    item.config.enabled = settings.get('enabled', False)
                    item.config.parameters = settings.get('parameters', {})
                    item.setCheckState(0, Qt.CheckState.Checked if item.config.enabled
                                     else Qt.CheckState.Unchecked)

            self.update_statistics()
            QMessageBox.information(self, "Import", "Configuration imported successfully")
        except Exception as e:
            QMessageBox.warning(self, "Import Error", f"Failed to import: {e}")

    def reset_to_defaults(self):
        """Reset all indicators to default configuration"""
        reply = QMessageBox.question(self, "Reset", "Reset all indicators to default settings?")
        if reply == QMessageBox.StandardButton.Yes:
            self.indicators_system = BTALibIndicators(self.available_data)
            self.indicators_tree.setup_indicators(self.indicators_system)
            self.update_statistics()

    def load_settings(self):
        """Load settings from persistent storage"""
        try:
            # Load indicator colors from settings
            colors_dict = get_setting("indicators.colors", {})
            if isinstance(colors_dict, dict):
                for name, color in colors_dict.items():
                    if name in self.indicators_tree.indicator_items:
                        item = self.indicators_tree.indicator_items[name]
                        item.config.color = color
                        item._update_color_display()

            # Load enabled indicators list and update checkboxes
            enabled_list = get_setting("indicators.enabled_list", [])
            if isinstance(enabled_list, list):
                # Temporarily block signals to avoid triggering itemChanged
                self.indicators_tree.blockSignals(True)

                # First disable all indicators
                for name, item in self.indicators_tree.indicator_items.items():
                    item.config.enabled = False
                    item.setCheckState(0, Qt.CheckState.Unchecked)
                    item.setText(4, "Disabled")

                # Then enable only the saved ones
                for name in enabled_list:
                    if name in self.indicators_tree.indicator_items:
                        item = self.indicators_tree.indicator_items[name]
                        item.config.enabled = True
                        item.setCheckState(0, Qt.CheckState.Checked)
                        item.setText(4, "Enabled")

                # Re-enable signals
                self.indicators_tree.blockSignals(False)

                # Update category counts and statistics
                self.indicators_tree.update_category_counts()
                self.update_statistics()
        except Exception as e:
            logger.warning(f"Failed to load indicator settings: {e}")

    def save_settings(self):
        """Save settings to persistent storage"""
        try:
            # Save indicator colors
            colors_dict = {}
            # Save enabled indicators list
            enabled_list = []

            for name, item in self.indicators_tree.indicator_items.items():
                if hasattr(item.config, 'color') and item.config.color:
                    colors_dict[name] = item.config.color
                if item.config.enabled:
                    enabled_list.append(name)

            set_setting("indicators.colors", colors_dict)
            set_setting("indicators.enabled_list", enabled_list)

            # Save dialog geometry
            geometry = self.geometry()
            set_setting("indicators.dialog.geometry", [geometry.x(), geometry.y(), geometry.width(), geometry.height()])
        except Exception as e:
            logger.warning(f"Failed to save indicator settings: {e}")

    def apply_settings(self):
        """Apply current settings without closing dialog"""
        self.save_settings()
        # Signal that settings have been applied
        # You can emit a signal here for the main application to update

    def accept(self):
        """Accept and close dialog"""
        self.apply_settings()
        super().accept()

    def get_enabled_indicators(self) -> Dict[str, IndicatorConfig]:
        """Get dictionary of enabled indicators"""
        return {
            name: item.config
            for name, item in self.indicators_tree.indicator_items.items()
            if item.config.enabled
        }