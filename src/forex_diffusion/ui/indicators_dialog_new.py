# src/forex_diffusion/ui/indicators_dialog_new.py
"""
Modern indicators dialog for bta-lib integration
Supports 80+ indicators with data requirement filtering and configuration management
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QLabel, QPushButton, QWidget,
    QComboBox, QLineEdit, QMessageBox, QTreeWidget, QTreeWidgetItem, QSplitter, QSlider
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from ..features.indicators_btalib import (
    BTALibIndicators, IndicatorConfig, IndicatorCategories, DataRequirement
)

# Persistenza (uguale a quella usata altrove nella UI)
try:
    from ..utils.user_settings import get_setting, set_setting
except Exception:  # fallback no-op se non disponibile
    def get_setting(key: str, default=None):
        return default
    def set_setting(key: str, value):
        pass


class DataAvailabilityWidget(QGroupBox):
    """Widget per configurare i dati disponibili"""

    def __init__(self, parent=None):
        super().__init__("Available Data Types", parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Always available
        self.ohlc_check = QCheckBox("OHLC Data (Open, High, Low, Close)")
        self.ohlc_check.setChecked(True)
        self.ohlc_check.setEnabled(False)  # Always required
        layout.addWidget(self.ohlc_check)

        # Optional data types
        self.volume_check = QCheckBox("Volume Data")
        self.volume_check.setToolTip("Enable if you have volume data available")
        layout.addWidget(self.volume_check)

        self.book_check = QCheckBox("Order Book Data (Bid/Ask/Spread)")
        self.book_check.setToolTip("Enable if you have order book data available")
        layout.addWidget(self.book_check)

        self.tick_check = QCheckBox("Tick Data")
        self.tick_check.setToolTip("Enable if you have tick-level data available")
        layout.addWidget(self.tick_check)

        # Info label
        info_label = QLabel("💡 Indicators requiring unavailable data will be automatically disabled")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)

    def get_available_data(self) -> List[str]:
        """Get list of available data types"""
        data = ['open', 'high', 'low', 'close']

        if self.volume_check.isChecked():
            data.append('volume')

        if self.book_check.isChecked():
            data.extend(['bid', 'ask', 'spread'])

        if self.tick_check.isChecked():
            data.extend(['tick_volume', 'tick_count'])

        return data

    def set_available_data(self, data_list: List[str]):
        """Set available data types"""
        self.volume_check.setChecked('volume' in data_list)
        self.book_check.setChecked(any(col in data_list for col in ['bid', 'ask', 'spread']))
        self.tick_check.setChecked(any(col in data_list for col in ['tick_volume', 'tick_count']))


class IndicatorTreeWidget(QTreeWidget):
    """Tree widget for organizing indicators by category"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Indicator", "Status", "Weight", "Data Req."])
        self.setColumnWidth(0, 250)
        self.setColumnWidth(1, 80)
        self.setColumnWidth(2, 80)
        self.setColumnWidth(3, 100)

        self.indicators_system = None
        self.category_items = {}
        self.indicator_items = {}

    def setup_indicators(self, indicators_system: BTALibIndicators):
        """Setup tree with indicators from the system"""
        self.indicators_system = indicators_system
        self.clear()
        self.category_items.clear()
        self.indicator_items.clear()

        # Create category items
        categories = [
            IndicatorCategories.OVERLAP,
            IndicatorCategories.MOMENTUM,
            IndicatorCategories.VOLATILITY,
            IndicatorCategories.TREND,
            IndicatorCategories.VOLUME,
            IndicatorCategories.PRICE_TRANSFORM,
            IndicatorCategories.STATISTICS,
            IndicatorCategories.CYCLE
        ]

        for category in categories:
            category_item = QTreeWidgetItem(self, [category])
            category_item.setFont(0, QFont("", -1, QFont.Bold))
            category_item.setExpanded(True)
            self.category_items[category] = category_item

            # Add indicators for this category
            category_indicators = indicators_system.get_indicators_by_category(category)

            for name, config in category_indicators.items():
                indicator_item = IndicatorTreeWidgetItem(category_item, name, config)
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

    def get_indicator_configs(self) -> Dict[str, IndicatorConfig]:
        """Get all indicator configurations"""
        configs = {}
        for name, item in self.indicator_items.items():
            configs[name] = item.config
        return configs

    def update_from_data_availability(self, available_data: List[str]):
        """Update indicators based on data availability"""
        if not self.indicators_system:
            return

        # Update system with new data availability
        self.indicators_system.available_data = available_data
        self.indicators_system.has_volume = 'volume' in available_data
        self.indicators_system.has_book_data = any(col in available_data for col in ['bid', 'ask', 'spread'])
        self.indicators_system.has_tick_data = any(col in available_data for col in ['tick_volume', 'tick_count'])

        # Update indicator items
        for name, item in self.indicator_items.items():
            item.update_availability()

        # Update category counts
        self.update_category_counts()


class IndicatorTreeWidgetItem(QTreeWidgetItem):
    """Custom tree widget item for indicators"""

    def __init__(self, parent, name: str, config: IndicatorConfig):
        super().__init__(parent)
        self.name = name
        self.config = config

        # Setup UI elements
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI elements for this indicator"""
        # Column 0: Indicator name and description
        self.setText(0, self.config.display_name)
        self.setToolTip(0, f"{self.config.description}\nParameters: {self.config.parameters}")

        # Column 1: Status (enabled/disabled checkbox)
        self.setFlags(self.flags() | Qt.ItemIsUserCheckable)
        self.setCheckState(1, Qt.Checked if self.config.enabled else Qt.Unchecked)

        # Column 2: Weight (will be set by parent widget)
        self.setText(2, f"{self.config.weight:.1f}")

        # Column 3: Data requirement
        req_text = {
            DataRequirement.OHLC_ONLY: "OHLC",
            DataRequirement.VOLUME_REQUIRED: "Volume",
            DataRequirement.BOOK_REQUIRED: "Book",
            DataRequirement.TICK_REQUIRED: "Tick"
        }.get(self.config.data_requirement, "Unknown")
        self.setText(3, req_text)

        self.update_availability()

    def update_availability(self):
        """Update availability based on data requirements"""
        # Check if data requirements are met
        parent_tree = self.treeWidget()
        if hasattr(parent_tree, 'indicators_system') and parent_tree.indicators_system:
            system = parent_tree.indicators_system

            can_enable = False
            if self.config.data_requirement == DataRequirement.OHLC_ONLY:
                can_enable = True
            elif self.config.data_requirement == DataRequirement.VOLUME_REQUIRED:
                can_enable = system.has_volume
            elif self.config.data_requirement == DataRequirement.BOOK_REQUIRED:
                can_enable = system.has_book_data
            elif self.config.data_requirement == DataRequirement.TICK_REQUIRED:
                can_enable = system.has_tick_data

            # Update UI
            if can_enable:
                self.setDisabled(False)
                self.setText(1, "✅" if self.config.enabled else "⭕")
                self.setToolTip(1, "Available - data requirements met")
            else:
                self.setDisabled(True)
                self.setText(1, "❌")
                self.setToolTip(1, f"Unavailable - missing {req_text} data")
                self.config.enabled = False
                self.setCheckState(1, Qt.Unchecked)

    def data(self, column: int, role: int):
        """Override data to handle checkbox changes"""
        if column == 1 and role == Qt.CheckStateRole:
            return Qt.Checked if self.config.enabled else Qt.Unchecked
        return super().data(column, role)

    def setData(self, column: int, role: int, value):
        """Override setData to handle checkbox changes"""
        if column == 1 and role == Qt.CheckStateRole:
            self.config.enabled = (value == Qt.Checked)
            self.update_availability()
            # Update category counts
            if self.treeWidget():
                self.treeWidget().update_category_counts()
        else:
            super().setData(column, role, value)


class WeightSliderWidget(QWidget):
    """Widget for adjusting indicator weights"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)  # 0.0 to 10.0 in 0.1 steps
        self.slider.setValue(10)  # Default 1.0
        self.slider.valueChanged.connect(self.update_label)

        self.label = QLabel("1.0")
        self.label.setMinimumWidth(30)

        layout.addWidget(self.slider)
        layout.addWidget(self.label)

    def update_label(self):
        value = self.slider.value() / 10.0
        self.label.setText(f"{value:.1f}")

    def get_weight(self) -> float:
        return self.slider.value() / 10.0

    def set_weight(self, weight: float):
        self.slider.setValue(int(weight * 10))
        self.update_label()


class ModernIndicatorsDialog(QDialog):
    """
    Modern indicators configuration dialog for bta-lib system
    Features 80+ indicators with smart data filtering and configuration management
    """

    def __init__(self, parent: Optional[QWidget] = None, initial_config: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Technical Indicators Configuration - bta-lib")
        self.setMinimumSize(1000, 700)

        self.initial_config = initial_config or {}
        self.indicators_system = None

        self.initUI()
        self.load_initial_config()

    def initUI(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)

        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: Data availability and session management
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Data availability section
        self.data_widget = DataAvailabilityWidget()
        self.data_widget.volume_check.toggled.connect(self.on_data_availability_changed)
        self.data_widget.book_check.toggled.connect(self.on_data_availability_changed)
        self.data_widget.tick_check.toggled.connect(self.on_data_availability_changed)
        left_layout.addWidget(self.data_widget)

        # Session management
        session_group = QGroupBox("Configuration Management")
        session_layout = QVBoxLayout(session_group)

        # Session selector
        session_row1 = QHBoxLayout()
        session_row1.addWidget(QLabel("Session:"))
        self.session_combo = QComboBox()
        self.session_combo.currentTextChanged.connect(self.load_session)
        session_row1.addWidget(self.session_combo)
        session_layout.addLayout(session_row1)

        # New session
        session_row2 = QHBoxLayout()
        self.session_name_edit = QLineEdit()
        self.session_name_edit.setPlaceholderText("New session name...")
        self.btn_session_save = QPushButton("Save")
        self.btn_session_save.clicked.connect(self.save_session)
        self.btn_session_delete = QPushButton("Delete")
        self.btn_session_delete.clicked.connect(self.delete_session)

        session_row2.addWidget(self.session_name_edit)
        session_row2.addWidget(self.btn_session_save)
        session_row2.addWidget(self.btn_session_delete)
        session_layout.addLayout(session_row2)

        # Preset buttons
        preset_row = QHBoxLayout()
        self.btn_default = QPushButton("Load Defaults")
        self.btn_default.clicked.connect(self.load_defaults)
        self.btn_enable_all = QPushButton("Enable All Available")
        self.btn_enable_all.clicked.connect(self.enable_all_available)
        self.btn_disable_all = QPushButton("Disable All")
        self.btn_disable_all.clicked.connect(self.disable_all)

        preset_row.addWidget(self.btn_default)
        preset_row.addWidget(self.btn_enable_all)
        preset_row.addWidget(self.btn_disable_all)
        session_layout.addLayout(preset_row)

        left_layout.addWidget(session_group)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_total_label = QLabel("Total Indicators: 0")
        self.stats_enabled_label = QLabel("Enabled: 0")
        self.stats_available_label = QLabel("Available: 0")
        self.stats_disabled_label = QLabel("Disabled (missing data): 0")

        stats_layout.addWidget(self.stats_total_label)
        stats_layout.addWidget(self.stats_enabled_label)
        stats_layout.addWidget(self.stats_available_label)
        stats_layout.addWidget(self.stats_disabled_label)

        left_layout.addWidget(stats_group)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Right panel: Indicators tree
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Search and filter
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Search indicators...")
        self.filter_edit.textChanged.connect(self.filter_indicators)
        filter_row.addWidget(self.filter_edit)

        self.category_filter = QComboBox()
        self.category_filter.addItem("All Categories")
        for category in [IndicatorCategories.OVERLAP, IndicatorCategories.MOMENTUM,
                        IndicatorCategories.VOLATILITY, IndicatorCategories.TREND,
                        IndicatorCategories.VOLUME, IndicatorCategories.PRICE_TRANSFORM,
                        IndicatorCategories.STATISTICS, IndicatorCategories.CYCLE]:
            self.category_filter.addItem(category)
        self.category_filter.currentTextChanged.connect(self.filter_indicators)
        filter_row.addWidget(self.category_filter)

        right_layout.addLayout(filter_row)

        # Indicators tree
        self.indicators_tree = IndicatorTreeWidget()
        right_layout.addWidget(self.indicators_tree)

        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])  # Give more space to indicators tree

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.btn_test = QPushButton("Test Configuration")
        self.btn_test.clicked.connect(self.test_configuration)
        button_layout.addWidget(self.btn_test)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_ok.setDefault(True)
        button_layout.addWidget(self.btn_ok)

        main_layout.addLayout(button_layout)

        # Initialize indicators system
        self.initialize_indicators_system()
        self.load_sessions()

    def initialize_indicators_system(self):
        """Initialize the indicators system"""
        available_data = self.data_widget.get_available_data()
        self.indicators_system = BTALibIndicators(available_data)
        self.indicators_tree.setup_indicators(self.indicators_system)
        self.update_statistics()

    def on_data_availability_changed(self):
        """Handle data availability changes"""
        if self.indicators_system:
            available_data = self.data_widget.get_available_data()
            self.indicators_tree.update_from_data_availability(available_data)
            self.update_statistics()

    def update_statistics(self):
        """Update statistics display"""
        if not self.indicators_system:
            return

        total = len(self.indicators_system.indicators_config)
        enabled = len(self.indicators_system.get_enabled_indicators())

        # Count available (can be enabled based on data)
        available = 0
        disabled_missing_data = 0

        for name, config in self.indicators_system.indicators_config.items():
            if config.data_requirement == DataRequirement.OHLC_ONLY:
                available += 1
            elif config.data_requirement == DataRequirement.VOLUME_REQUIRED and self.indicators_system.has_volume:
                available += 1
            elif config.data_requirement == DataRequirement.BOOK_REQUIRED and self.indicators_system.has_book_data:
                available += 1
            elif config.data_requirement == DataRequirement.TICK_REQUIRED and self.indicators_system.has_tick_data:
                available += 1
            else:
                disabled_missing_data += 1

        self.stats_total_label.setText(f"Total Indicators: {total}")
        self.stats_enabled_label.setText(f"Enabled: {enabled}")
        self.stats_available_label.setText(f"Available: {available}")
        self.stats_disabled_label.setText(f"Disabled (missing data): {disabled_missing_data}")

    def filter_indicators(self):
        """Filter indicators based on search text and category"""
        search_text = self.filter_edit.text().lower()
        category_filter = self.category_filter.currentText()

        for category, category_item in self.indicators_tree.category_items.items():
            category_visible = False

            # Check if category matches filter
            if category_filter == "All Categories" or category_filter == category:
                # Show/hide indicators based on search
                for i in range(category_item.childCount()):
                    indicator_item = category_item.child(i)
                    indicator_name = indicator_item.config.display_name.lower()
                    indicator_desc = indicator_item.config.description.lower()

                    visible = (search_text == "" or
                              search_text in indicator_name or
                              search_text in indicator_desc or
                              search_text in indicator_item.name.lower())

                    indicator_item.setHidden(not visible)
                    if visible:
                        category_visible = True
            else:
                # Hide all indicators in this category
                for i in range(category_item.childCount()):
                    category_item.child(i).setHidden(True)

            category_item.setHidden(not category_visible)

    def load_defaults(self):
        """Load default configuration"""
        if self.indicators_system:
            # Enable commonly used indicators
            common_indicators = [
                'sma', 'ema', 'rsi', 'macd', 'bbands', 'atr', 'stoch',
                'cci', 'willr', 'adx', 'aroon', 'roc', 'mom'
            ]

            for name, config in self.indicators_system.indicators_config.items():
                if name in common_indicators:
                    self.indicators_system.enable_indicator(name, 1.0)
                else:
                    self.indicators_system.disable_indicator(name)

            self.indicators_tree.setup_indicators(self.indicators_system)
            self.update_statistics()

    def enable_all_available(self):
        """Enable all available indicators"""
        if self.indicators_system:
            available = self.indicators_system.get_available_indicators()
            for name, config in available.items():
                if (config.data_requirement == DataRequirement.OHLC_ONLY or
                    (config.data_requirement == DataRequirement.VOLUME_REQUIRED and self.indicators_system.has_volume) or
                    (config.data_requirement == DataRequirement.BOOK_REQUIRED and self.indicators_system.has_book_data) or
                    (config.data_requirement == DataRequirement.TICK_REQUIRED and self.indicators_system.has_tick_data)):
                    self.indicators_system.enable_indicator(name, 1.0)

            self.indicators_tree.setup_indicators(self.indicators_system)
            self.update_statistics()

    def disable_all(self):
        """Disable all indicators"""
        if self.indicators_system:
            for name in self.indicators_system.indicators_config:
                self.indicators_system.disable_indicator(name)

            self.indicators_tree.setup_indicators(self.indicators_system)
            self.update_statistics()

    def test_configuration(self):
        """Test the current configuration"""
        if not self.indicators_system:
            QMessageBox.warning(self, "Warning", "No indicators system available")
            return

        enabled = self.indicators_system.get_enabled_indicators()
        if not enabled:
            QMessageBox.warning(self, "Warning", "No indicators enabled")
            return

        # Create a simple test
        msg = "Configuration Test Results:\n\n"
        msg += f"✅ Data Available: {', '.join(self.indicators_system.available_data)}\n"
        msg += f"✅ Enabled Indicators: {len(enabled)}\n\n"

        msg += "Categories with enabled indicators:\n"
        categories = {}
        for name, config in enabled.items():
            if config.category not in categories:
                categories[config.category] = 0
            categories[config.category] += 1

        for category, count in categories.items():
            msg += f"  • {category}: {count} indicators\n"

        msg += "\nConfiguration appears valid! ✅"

        QMessageBox.information(self, "Test Results", msg)

    def save_session(self):
        """Save current configuration as a session"""
        session_name = self.session_name_edit.text().strip()
        if not session_name:
            QMessageBox.warning(self, "Warning", "Please enter a session name")
            return

        if self.indicators_system:
            config = self.get_configuration()
            sessions = get_setting("indicators.sessions", {})
            sessions[session_name] = config
            set_setting("indicators.sessions", sessions)

            # Update combo box
            if session_name not in [self.session_combo.itemText(i) for i in range(self.session_combo.count())]:
                self.session_combo.addItem(session_name)

            self.session_combo.setCurrentText(session_name)
            self.session_name_edit.clear()

            QMessageBox.information(self, "Success", f"Session '{session_name}' saved successfully")

    def load_session(self, session_name: str):
        """Load a saved session"""
        if not session_name:
            return

        sessions = get_setting("indicators.sessions", {})
        if session_name in sessions:
            config = sessions[session_name]
            self.apply_configuration(config)

    def delete_session(self):
        """Delete the selected session"""
        session_name = self.session_combo.currentText()
        if not session_name:
            return

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete session '{session_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            sessions = get_setting("indicators.sessions", {})
            if session_name in sessions:
                del sessions[session_name]
                set_setting("indicators.sessions", sessions)

                # Update combo box
                index = self.session_combo.findText(session_name)
                if index >= 0:
                    self.session_combo.removeItem(index)

                QMessageBox.information(self, "Success", f"Session '{session_name}' deleted")

    def load_sessions(self):
        """Load saved sessions into combo box"""
        sessions = get_setting("indicators.sessions", {})
        self.session_combo.clear()
        self.session_combo.addItems(list(sessions.keys()))

    def load_initial_config(self):
        """Load initial configuration"""
        if self.initial_config:
            self.apply_configuration(self.initial_config)

    def apply_configuration(self, config: Dict[str, Any]):
        """Apply a configuration"""
        # Apply data availability
        if 'available_data' in config:
            self.data_widget.set_available_data(config['available_data'])
            self.on_data_availability_changed()

        # Apply indicator settings
        if 'indicators' in config and self.indicators_system:
            self.indicators_system.load_config_dict(config['indicators'])
            self.indicators_tree.setup_indicators(self.indicators_system)
            self.update_statistics()

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        config = {
            'available_data': self.data_widget.get_available_data(),
            'indicators': self.indicators_system.get_config_dict() if self.indicators_system else {}
        }
        return config

    @staticmethod
    def edit_indicators(parent=None, initial_config=None) -> Optional[Dict[str, Any]]:
        """Static method to show dialog and return configuration"""
        dialog = ModernIndicatorsDialog(parent, initial_config)
        if dialog.exec() == QDialog.Accepted:
            return dialog.get_configuration()
        return None


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Test the dialog
    initial_config = {
        'available_data': ['open', 'high', 'low', 'close'],
        'indicators': {}
    }

    result = ModernIndicatorsDialog.edit_indicators(None, initial_config)
    if result:
        print("Configuration saved:")
        print(f"Available data: {result['available_data']}")
        enabled_count = sum(1 for config in result['indicators'].values() if config.get('enabled', False))
        print(f"Enabled indicators: {enabled_count}")
    else:
        print("Dialog cancelled")

    sys.exit(0)