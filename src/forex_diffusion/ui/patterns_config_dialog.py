
from __future__ import annotations
from typing import List, Dict, Any
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QTabWidget, QWidget, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLabel, QComboBox, QLineEdit, QSplitter, QScrollArea, QFrame, QApplication)
from PySide6.QtCore import Qt
import yaml, os
import inspect
from ...patterns.registry import PatternRegistry

class PatternsConfigDialog(QDialog):
    def __init__(self, parent=None, yaml_path:str="configs/patterns.yaml", patterns_service=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configura Patterns")

        # Resize to 80% of monitor height
        screen = QApplication.primaryScreen().geometry()
        dialog_height = int(screen.height() * 0.8)
        dialog_width = min(1200, int(screen.width() * 0.7))
        self.resize(dialog_width, dialog_height)

        self.yaml_path = yaml_path
        self.patterns_service = patterns_service

        # Initialize pattern registry to get available patterns
        self.registry = PatternRegistry()
        self.available_patterns = self._get_available_patterns()
        with open(self.yaml_path, 'r', encoding='utf-8') as fh:
            self.cfg = yaml.safe_load(fh) or {}
        self.patterns = self.cfg.get('patterns', {})
        self.historical_settings = self.cfg.get('historical_patterns', {
            'enabled': False,
            'start_time': '30d',
            'end_time': '7d'
        })
        self._build_ui()

    def _get_available_patterns(self) -> Dict[str, List[Dict]]:
        """Get available patterns and their parameters from registry"""
        patterns = {'chart_patterns': [], 'candle_patterns': []}

        try:
            # Get all detectors
            all_detectors = self.registry.detectors()

            for detector in all_detectors:
                # Get detector information
                detector_info = {
                    'key': getattr(detector, 'key', detector.__class__.__name__),
                    'kind': getattr(detector, 'kind', 'unknown'),
                    'name': getattr(detector, 'key', detector.__class__.__name__).replace('_', ' ').title(),
                    'parameters': self._extract_detector_parameters(detector)
                }

                # Add to appropriate category
                if detector_info['kind'] == 'chart':
                    patterns['chart_patterns'].append(detector_info)
                elif detector_info['kind'] == 'candle':
                    patterns['candle_patterns'].append(detector_info)

        except Exception as e:
            print(f"Error getting available patterns: {e}")

        return patterns

    def _extract_detector_parameters(self, detector) -> Dict[str, Any]:
        """Extract parameters from detector constructor"""
        parameters = {}
        try:
            # Get constructor signature
            sig = inspect.signature(detector.__class__.__init__)

            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'key', 'mode']:
                    continue

                # Get current value from detector instance
                current_value = getattr(detector, param_name, param.default)

                param_info = {
                    'name': param_name,
                    'type': type(current_value).__name__ if current_value != param.default else 'int',
                    'default': param.default if param.default != inspect.Parameter.empty else 0,
                    'current': current_value,
                    'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                }

                parameters[param_name] = param_info

        except Exception as e:
            print(f"Error extracting parameters for {detector.__class__.__name__}: {e}")

        return parameters

    def _build_ui(self):
        lay = QVBoxLayout(self)
        tabs = QTabWidget(self); lay.addWidget(tabs)

        self.chart_tab = self._make_pattern_tab(kind='chart_patterns')
        self.candle_tab = self._make_pattern_tab(kind='candle_patterns')
        self.common_tab = self._make_common_settings_tab()

        tabs.addTab(self.chart_tab, "Patterns a Chart")
        tabs.addTab(self.candle_tab, "Patterns a Candela")
        tabs.addTab(self.common_tab, "Common Settings")

        btns = QHBoxLayout()
        self.btn_scan_historical = QPushButton("Scan Historical")
        self.btn_save = QPushButton("Salva")
        self.btn_cancel = QPushButton("Chiudi")
        btns.addWidget(self.btn_scan_historical)
        btns.addStretch(1); btns.addWidget(self.btn_save); btns.addWidget(self.btn_cancel)
        lay.addLayout(btns)
        self.btn_scan_historical.clicked.connect(self._on_scan_historical_clicked)
        self.btn_save.clicked.connect(self._save)
        self.btn_cancel.clicked.connect(self.reject)

    def _make_pattern_tab(self, kind: str) -> QWidget:
        """Create new pattern tab with list on left and parameters on right"""
        container = QWidget()
        layout = QVBoxLayout(container)

        # Create splitter with pattern list on left and parameters on right
        splitter = QSplitter(Qt.Horizontal)

        # Left side: Pattern list (scrollable)
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.addWidget(QLabel(f"Available {kind.replace('_', ' ').title()}:"))

        pattern_list = QListWidget()
        pattern_list.setMaximumWidth(300)
        pattern_list.setMinimumWidth(200)

        # Add patterns to list
        patterns = self.available_patterns.get(kind, [])
        for pattern in patterns:
            item = QListWidgetItem(pattern['name'])
            item.setData(Qt.UserRole, pattern)
            pattern_list.addItem(item)

        left_layout.addWidget(pattern_list)

        # Right side: Parameters (scrollable)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.parameters_widget = QWidget()
        self.parameters_layout = QFormLayout(self.parameters_widget)
        self.parameters_layout.addRow(QLabel("Select a pattern to configure its parameters"))

        right_scroll.setWidget(self.parameters_widget)

        # Add to splitter
        splitter.addWidget(left_frame)
        splitter.addWidget(right_scroll)
        splitter.setSizes([250, 750])  # 25% for list, 75% for parameters

        layout.addWidget(splitter)

        # Connect pattern selection to parameter display
        pattern_list.currentItemChanged.connect(
            lambda current, previous: self._on_pattern_selected(current, kind)
        )

        # Store references for later use
        setattr(container, f'pattern_list_{kind}', pattern_list)
        setattr(container, f'parameters_scroll_{kind}', right_scroll)

        return container

    def _on_pattern_selected(self, current_item, kind: str):
        """Display parameters for selected pattern"""
        if not current_item:
            return

        pattern = current_item.data(Qt.UserRole)
        if not pattern:
            return

        # Clear current parameters widget
        while self.parameters_layout.count():
            item = self.parameters_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add pattern name header
        self.parameters_layout.addRow(QLabel(f"Parameters for: {pattern['name']}"))

        # Add enable checkbox for the pattern
        enable_cb = QCheckBox("Enable Pattern")
        enable_cb.setChecked(True)  # Default enabled
        self.parameters_layout.addRow("Status:", enable_cb)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        self.parameters_layout.addRow(separator)

        # Add parameters with override checkboxes
        for param_name, param_info in pattern['parameters'].items():
            param_row = QWidget()
            param_layout = QHBoxLayout(param_row)

            # Override checkbox
            override_cb = QCheckBox("Override")
            override_cb.setChecked(False)  # Default no override

            # Parameter widget (spinbox for numbers, lineedit for strings)
            if param_info['type'] in ['int', 'float']:
                if param_info['type'] == 'int':
                    param_widget = QSpinBox()
                    param_widget.setRange(0, 10000)
                    param_widget.setValue(int(param_info['current']))
                else:
                    param_widget = QDoubleSpinBox()
                    param_widget.setRange(0.0, 1000.0)
                    param_widget.setDecimals(3)
                    param_widget.setValue(float(param_info['current']))
            else:
                param_widget = QLineEdit()
                param_widget.setText(str(param_info['current']))

            # Initially disable parameter widget (no override)
            param_widget.setEnabled(False)

            # Connect override checkbox to enable/disable parameter widget
            override_cb.toggled.connect(param_widget.setEnabled)

            # Add to layout
            param_layout.addWidget(override_cb)
            param_layout.addWidget(param_widget)
            param_layout.addWidget(QLabel(f"(default: {param_info['default']})"))
            param_layout.addStretch()

            # Add to form
            param_label = param_name.replace('_', ' ').title()
            self.parameters_layout.addRow(f"{param_label}:", param_row)

            # Store widgets for saving later
            if not hasattr(self, '_parameter_widgets'):
                self._parameter_widgets = {}
            if kind not in self._parameter_widgets:
                self._parameter_widgets[kind] = {}
            if pattern['key'] not in self._parameter_widgets[kind]:
                self._parameter_widgets[kind][pattern['key']] = {}

            self._parameter_widgets[kind][pattern['key']][param_name] = {
                'override': override_cb,
                'widget': param_widget,
                'default': param_info['default']
            }

        # Store enable checkbox
        if not hasattr(self, '_enable_widgets'):
            self._enable_widgets = {}
        if kind not in self._enable_widgets:
            self._enable_widgets[kind] = {}
        self._enable_widgets[kind][pattern['key']] = enable_cb

        # Load saved settings for this pattern
        self._load_pattern_settings(pattern, kind, enable_cb)

    def _load_pattern_settings(self, pattern, kind, enable_cb):
        """Load saved settings for a pattern"""
        try:
            pattern_key = pattern['key']

            # Load enabled status
            enabled_patterns = self.patterns.get(kind, {}).get('keys_enabled', [])
            enable_cb.setChecked(pattern_key in enabled_patterns)

            # Load parameter overrides
            overrides = self.patterns.get(kind, {}).get('parameter_overrides', {}).get(pattern_key, {})

            if pattern_key in self._parameter_widgets.get(kind, {}):
                param_widgets = self._parameter_widgets[kind][pattern_key]

                for param_name, widgets in param_widgets.items():
                    override_cb = widgets['override']
                    param_widget = widgets['widget']

                    if param_name in overrides:
                        # Enable override and set value
                        override_cb.setChecked(True)
                        param_widget.setEnabled(True)

                        override_value = overrides[param_name]
                        if isinstance(param_widget, QSpinBox):
                            param_widget.setValue(int(override_value))
                        elif isinstance(param_widget, QDoubleSpinBox):
                            param_widget.setValue(float(override_value))
                        elif isinstance(param_widget, QLineEdit):
                            param_widget.setText(str(override_value))

        except Exception as e:
            print(f"Error loading pattern settings for {pattern.get('key', 'unknown')}: {e}")

    def _save(self):
        # Save enabled patterns and their override parameters
        if hasattr(self, '_enable_widgets') and hasattr(self, '_parameter_widgets'):
            for kind in ('chart_patterns', 'candle_patterns'):
                if kind not in self.patterns:
                    self.patterns[kind] = {}

                enabled_patterns = []
                pattern_overrides = {}

                enable_widgets = self._enable_widgets.get(kind, {})
                param_widgets = self._parameter_widgets.get(kind, {})

                for pattern_key, enable_widget in enable_widgets.items():
                    if enable_widget.isChecked():
                        enabled_patterns.append(pattern_key)

                    # Save parameter overrides
                    if pattern_key in param_widgets:
                        pattern_overrides[pattern_key] = {}
                        for param_name, widgets in param_widgets[pattern_key].items():
                            override_cb = widgets['override']
                            param_widget = widgets['widget']

                            if override_cb.isChecked():
                                # Save override value
                                if isinstance(param_widget, QSpinBox):
                                    pattern_overrides[pattern_key][param_name] = param_widget.value()
                                elif isinstance(param_widget, QDoubleSpinBox):
                                    pattern_overrides[pattern_key][param_name] = param_widget.value()
                                elif isinstance(param_widget, QLineEdit):
                                    pattern_overrides[pattern_key][param_name] = param_widget.text()

                self.patterns[kind]['keys_enabled'] = enabled_patterns
                self.patterns[kind]['parameter_overrides'] = pattern_overrides

        # Save dataset strategy
        strategy_text = self.cb_dataset_strategy.currentText()
        strategy_key = strategy_text.split(' - ')[0]  # Extract key (profittabilità, rischio, bilanciato)
        self.cfg['dataset_strategy'] = strategy_key

        # Save historical patterns settings
        historical_settings = {
            'enabled': self.cb_historical_enabled.isChecked(),
            'start_time': self.le_start_time.text().strip(),
            'end_time': self.le_end_time.text().strip()
        }
        self.cfg['historical_patterns'] = historical_settings

        self.cfg['patterns'] = self.patterns
        with open(self.yaml_path, 'w', encoding='utf-8') as fh:
            yaml.safe_dump(self.cfg, fh, allow_unicode=True, sort_keys=False)
        self.accept()

    def _make_common_settings_tab(self) -> QWidget:
        box = QWidget()
        main_layout = QVBoxLayout(box)

        # Create scroll area for the content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Dataset Strategy Selection
        strategy_group = QWidget()
        strategy_layout = QFormLayout(strategy_group)
        strategy_layout.addRow(QLabel("Dataset Strategy Configuration"))

        self.cb_dataset_strategy = QComboBox()
        self.cb_dataset_strategy.addItems([
            "profittabilità - Maximize profits, accept higher risk",
            "rischio - Minimize losses, conservative approach",
            "bilanciato - Balanced risk/reward profile"
        ])
        # Load saved dataset strategy
        saved_strategy = self.cfg.get('dataset_strategy', 'bilanciato')
        strategy_map = {
            'profittabilità': "profittabilità - Maximize profits, accept higher risk",
            'rischio': "rischio - Minimize losses, conservative approach",
            'bilanciato': "bilanciato - Balanced risk/reward profile"
        }
        if saved_strategy in strategy_map:
            self.cb_dataset_strategy.setCurrentText(strategy_map[saved_strategy])
        else:
            self.cb_dataset_strategy.setCurrentText("bilanciato - Balanced risk/reward profile")
        strategy_layout.addRow(QLabel("Dataset Strategy:"), self.cb_dataset_strategy)

        layout.addWidget(strategy_group)

        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        layout.addWidget(separator1)

        # Historical Patterns Settings
        self.cb_historical_enabled = QCheckBox("Abilita Patterns Storici")
        self.cb_historical_enabled.setChecked(self.historical_settings.get('enabled', False))
        layout.addWidget(self.cb_historical_enabled)

        # Gruppo per le impostazioni temporali (inizialmente nascosto)
        self.historical_settings_group = QWidget()
        settings_layout = QFormLayout(self.historical_settings_group)

        # Tempo inizio patterns storici
        self.le_start_time = QLineEdit()
        self.le_start_time.setPlaceholderText("es. 30d, 24h, 120m")
        self.le_start_time.setText(self.historical_settings.get('start_time', '30d'))
        settings_layout.addRow(QLabel("Tempo inizio patterns storici:"), self.le_start_time)

        # Tempo fine patterns storici
        self.le_end_time = QLineEdit()
        self.le_end_time.setPlaceholderText("es. 7d, 12h, 60m")
        self.le_end_time.setText(self.historical_settings.get('end_time', '7d'))
        settings_layout.addRow(QLabel("Tempo fine patterns storici:"), self.le_end_time)

        # Visibilità basata sulle impostazioni salvate
        enabled = self.historical_settings.get('enabled', False)
        self.historical_settings_group.setVisible(enabled)

        layout.addWidget(self.historical_settings_group)
        layout.addStretch()

        # Set content widget to scroll area
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # Connetti il checkbox per mostrare/nascondere le impostazioni
        self.cb_historical_enabled.toggled.connect(self._on_historical_enabled_changed)

        return box

    def _on_historical_enabled_changed(self, enabled: bool):
        """Mostra/nasconde le impostazioni quando historical patterns è abilitato"""
        self.historical_settings_group.setVisible(enabled)

    @staticmethod
    def parse_time_string(time_str: str) -> int:
        """Converte stringa tempo formato 'xm', 'xh', 'xd' in minuti"""
        if not time_str:
            return 0

        time_str = time_str.strip().lower()
        try:
            if time_str.endswith('m'):
                return int(time_str[:-1])
            elif time_str.endswith('h'):
                return int(time_str[:-1]) * 60
            elif time_str.endswith('d'):
                return int(time_str[:-1]) * 60 * 24
            else:
                # Assume minuti se non specificato
                return int(time_str)
        except ValueError:
            return 0

    def get_historical_time_range_minutes(self) -> tuple[int, int]:
        """Restituisce il range temporale in minuti (inizio, fine)"""
        start_minutes = self.parse_time_string(self.le_start_time.text())
        end_minutes = self.parse_time_string(self.le_end_time.text())
        return start_minutes, end_minutes

    def _on_scan_historical_clicked(self):
        """Handler per il bottone di scansione patterns storici"""
        try:
            if not self.patterns_service:
                print("Patterns service non disponibile")
                return

            # Salva prima le impostazioni attuali
            self._save()

            # Avvia la scansione storica
            self.patterns_service.start_historical_scan_with_range()

            print(f"Scansione storica avviata: {self.le_start_time.text()} - {self.le_end_time.text()}")

        except Exception as e:
            print(f"Errore durante la scansione storica: {e}")
