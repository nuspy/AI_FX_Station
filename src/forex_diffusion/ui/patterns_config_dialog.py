
from __future__ import annotations
from typing import List, Dict, Any
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QTabWidget, QWidget, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLabel, QComboBox, QLineEdit, QSplitter, QScrollArea, QFrame, QApplication,
    QGroupBox, QGridLayout)
from PySide6.QtCore import Qt
import yaml, os
import inspect
from ..patterns.registry import PatternRegistry
from ..patterns.boundary_config import get_boundary_config

class PatternsConfigDialog(QDialog):
    def __init__(self, parent=None, yaml_path:str="configs/patterns.yaml", patterns_service=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configura Patterns")

        # Resize to 80% of monitor height (with safety check)
        try:
            screen = QApplication.primaryScreen().geometry()
            dialog_height = int(screen.height() * 0.8)
            dialog_width = min(1200, int(screen.width() * 0.7))
            self.resize(dialog_width, dialog_height)
        except Exception:
            # Fallback to default size if screen detection fails
            self.resize(1200, 800)

        self.yaml_path = yaml_path
        self.patterns_service = patterns_service

        # Initialize pattern registry to get available patterns
        self.registry = PatternRegistry()
        self.available_patterns = self._get_available_patterns()

        # Initialize boundary configuration
        self.boundary_config = get_boundary_config()

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
                default_value = param.default if param.default != inspect.Parameter.empty else 0

                # Determine type from annotation, default, or current value
                param_type = 'int'  # Default type
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = 'int'
                    elif param.annotation == float:
                        param_type = 'float'
                    elif param.annotation == str:
                        param_type = 'str'
                elif current_value is not None:
                    param_type = type(current_value).__name__
                elif default_value is not None:
                    param_type = type(default_value).__name__

                param_info = {
                    'name': param_name,
                    'type': param_type,
                    'default': default_value,
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

        # Now trigger initial pattern selection for both tabs
        self._trigger_initial_pattern_selection()

    def _trigger_initial_pattern_selection(self):
        """Trigger initial pattern selection for both tabs to show parameters"""
        try:
            for kind, tab in [('chart_patterns', self.chart_tab), ('candle_patterns', self.candle_tab)]:
                pattern_list = getattr(tab, f'pattern_list_{kind}', None)
                if pattern_list and pattern_list.count() > 0:
                    first_item = pattern_list.item(0)
                    if first_item:
                        self._on_pattern_selected(first_item, kind)
        except AttributeError:
            # Tabs not yet fully initialized, skip initial selection
            pass

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

        # Create tab-specific parameters widget
        parameters_widget = QWidget()
        parameters_layout = QFormLayout(parameters_widget)
        parameters_layout.addRow(QLabel("Select a pattern to configure its parameters"))

        right_scroll.setWidget(parameters_widget)

        # Add to splitter
        splitter.addWidget(left_frame)
        splitter.addWidget(right_scroll)
        splitter.setSizes([250, 750])  # 25% for list, 75% for parameters

        layout.addWidget(splitter)

        # Store references for later use - including layout for this specific tab
        setattr(container, f'pattern_list_{kind}', pattern_list)
        setattr(container, f'parameters_scroll_{kind}', right_scroll)
        setattr(container, f'parameters_widget_{kind}', parameters_widget)
        setattr(container, f'parameters_layout_{kind}', parameters_layout)

        # Connect pattern selection to parameter display
        pattern_list.currentItemChanged.connect(
            lambda current, previous: self._on_pattern_selected(current, kind)
        )

        # Auto-select first pattern but don't trigger yet (will be triggered after full initialization)
        if patterns and pattern_list.count() > 0:
            pattern_list.setCurrentRow(0)

        # Initialize dictionaries for this tab if they don't exist
        if not hasattr(self, '_parameter_widgets'):
            self._parameter_widgets = {}
        if not hasattr(self, '_enable_widgets'):
            self._enable_widgets = {}
        if not hasattr(self, '_pattern_states'):
            self._pattern_states = {}  # For storing temporary states

        if kind not in self._parameter_widgets:
            self._parameter_widgets[kind] = {}
        if kind not in self._enable_widgets:
            self._enable_widgets[kind] = {}
        if kind not in self._pattern_states:
            self._pattern_states[kind] = {}

        return container

    def _on_pattern_selected(self, current_item, kind: str):
        """Display parameters for selected pattern"""
        if not current_item:
            return

        pattern = current_item.data(Qt.UserRole)
        if not pattern:
            return

        # Get the correct layout for this tab
        try:
            current_tab = self.chart_tab if kind == 'chart_patterns' else self.candle_tab
            parameters_layout = getattr(current_tab, f'parameters_layout_{kind}')
            parameters_widget = getattr(current_tab, f'parameters_widget_{kind}')
        except AttributeError as e:
            print(f"Error: Could not find layout for {kind}: {e}")
            return

        # Save current pattern state before switching (if any)
        self._save_current_pattern_state(kind)

        # Save the state of currently visible widgets before deleting them
        if hasattr(self, '_current_pattern') and kind in self._current_pattern:
            current_key = self._current_pattern[kind]
            if current_key in self._enable_widgets.get(kind, {}):
                current_enable_widget = self._enable_widgets[kind][current_key]
                try:
                    if current_enable_widget and hasattr(current_enable_widget, 'isChecked'):
                        current_state = current_enable_widget.isChecked()
                        if kind not in self._pattern_states:
                            self._pattern_states[kind] = {}
                        self._pattern_states[kind][current_key] = {
                            'enabled': current_state,
                            'parameters': self._pattern_states[kind].get(current_key, {}).get('parameters', {})
                        }
                except:
                    pass

        # Clear current parameters widget
        while parameters_layout.count():
            item = parameters_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add pattern name header
        parameters_layout.addRow(QLabel(f"Parameters for: {pattern['name']}"))

        # Add enable checkbox for the pattern
        enable_cb = QCheckBox("Enable Pattern")
        parameters_layout.addRow("Status:", enable_cb)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        parameters_layout.addRow(separator)

        # Add parameters with override checkboxes
        for param_name, param_info in pattern['parameters'].items():
            param_row = QWidget()
            param_layout = QHBoxLayout(param_row)

            # Override checkbox
            override_cb = QCheckBox("Override")

            # Parameter widget (spinbox for numbers, lineedit for strings)
            # Use default value if current is None
            current_value = param_info['current'] if param_info['current'] is not None else param_info['default']

            if param_info['type'] in ['int', 'float']:
                if param_info['type'] == 'int':
                    param_widget = QSpinBox()
                    param_widget.setRange(0, 10000)
                    try:
                        param_widget.setValue(int(current_value) if current_value is not None else 0)
                    except (ValueError, TypeError):
                        param_widget.setValue(0)
                else:
                    param_widget = QDoubleSpinBox()
                    param_widget.setRange(0.0, 1000.0)
                    param_widget.setDecimals(3)
                    try:
                        param_widget.setValue(float(current_value) if current_value is not None else 0.0)
                    except (ValueError, TypeError):
                        param_widget.setValue(0.0)
            else:
                param_widget = QLineEdit()
                param_widget.setText(str(current_value) if current_value is not None else "")

            # Initially disable parameter widget (no override)
            param_widget.setEnabled(False)
            override_cb.setChecked(False)

            # Connect override checkbox to enable/disable parameter widget
            override_cb.toggled.connect(param_widget.setEnabled)

            # Add to layout
            param_layout.addWidget(override_cb)
            param_layout.addWidget(param_widget)
            param_layout.addWidget(QLabel(f"(default: {param_info['default']})"))
            param_layout.addStretch()

            # Add to form
            param_label = param_name.replace('_', ' ').title()
            parameters_layout.addRow(f"{param_label}:", param_row)

            # Store widgets for this tab/pattern
            if pattern['key'] not in self._parameter_widgets[kind]:
                self._parameter_widgets[kind][pattern['key']] = {}

            self._parameter_widgets[kind][pattern['key']][param_name] = {
                'override': override_cb,
                'widget': param_widget,
                'default': param_info['default']
            }

        # Store enable checkbox
        self._enable_widgets[kind][pattern['key']] = enable_cb

        # Add boundaries section
        self._add_boundary_section(parameters_layout, pattern['key'], kind)

        # Load saved settings for this pattern
        self._load_pattern_settings(pattern, kind, enable_cb)

        # Set current pattern for this tab
        if not hasattr(self, '_current_pattern'):
            self._current_pattern = {}
        self._current_pattern[kind] = pattern['key']

    def _add_boundary_section(self, layout: QFormLayout, pattern_key: str, kind: str):
        """Add historical boundaries configuration section"""
        try:
            # Add separator before boundaries
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            layout.addRow(separator)

            # Boundaries section header
            boundaries_label = QLabel("Historical Boundaries (Candles from Present)")
            boundaries_label.setStyleSheet("font-weight: bold; color: #0066cc;")
            layout.addRow(boundaries_label)

            # Helper text
            help_text = QLabel("Define how many candles back to consider as 'current' vs 'historical'")
            help_text.setStyleSheet("font-size: 10px; color: #666;")
            help_text.setWordWrap(True)
            layout.addRow(help_text)

            # Create boundaries grid widget
            boundaries_widget = QGroupBox("Boundaries by Timeframe")
            boundaries_layout = QGridLayout(boundaries_widget)

            # Get available timeframes
            timeframes = self.boundary_config.get_timeframes()

            # Initialize boundary widgets storage if needed
            if not hasattr(self, '_boundary_widgets'):
                self._boundary_widgets = {}
            if kind not in self._boundary_widgets:
                self._boundary_widgets[kind] = {}
            if pattern_key not in self._boundary_widgets[kind]:
                self._boundary_widgets[kind][pattern_key] = {}

            # Add header row
            boundaries_layout.addWidget(QLabel("Timeframe"), 0, 0)
            boundaries_layout.addWidget(QLabel("Candles"), 0, 1)
            boundaries_layout.addWidget(QLabel("Default"), 0, 2)

            # Add boundary controls for each timeframe
            for row, timeframe in enumerate(timeframes, start=1):
                # Timeframe label
                tf_label = QLabel(timeframe.upper())
                boundaries_layout.addWidget(tf_label, row, 0)

                # Boundary spinbox
                boundary_spinbox = QSpinBox()
                boundary_spinbox.setRange(1, 10000)
                current_boundary = self.boundary_config.get_boundary(pattern_key, timeframe)
                boundary_spinbox.setValue(current_boundary)
                boundary_spinbox.setToolTip(f"Number of candles from present to consider as 'current' for {timeframe}")
                boundaries_layout.addWidget(boundary_spinbox, row, 1)

                # Default value label
                default_boundary = self.boundary_config._get_default_boundaries().get(pattern_key, {}).get(timeframe, 50)
                default_label = QLabel(f"({default_boundary})")
                default_label.setStyleSheet("color: #666; font-size: 10px;")
                boundaries_layout.addWidget(default_label, row, 2)

                # Store widget reference
                self._boundary_widgets[kind][pattern_key][timeframe] = boundary_spinbox

            # Reset to defaults button
            reset_btn = QPushButton("Reset to Defaults")
            reset_btn.clicked.connect(lambda: self._reset_boundaries_to_default(pattern_key, kind))
            boundaries_layout.addWidget(reset_btn, len(timeframes) + 1, 0, 1, 3)

            layout.addRow(boundaries_widget)

        except Exception as e:
            print(f"Error adding boundary section: {e}")

    def _reset_boundaries_to_default(self, pattern_key: str, kind: str):
        """Reset boundaries for a pattern to default values"""
        try:
            # Reset in config
            self.boundary_config.reset_to_defaults(pattern_key)

            # Update UI widgets
            if (kind in self._boundary_widgets and
                pattern_key in self._boundary_widgets[kind]):

                for timeframe, widget in self._boundary_widgets[kind][pattern_key].items():
                    default_value = self.boundary_config.get_boundary(pattern_key, timeframe)
                    widget.setValue(default_value)

        except Exception as e:
            print(f"Error resetting boundaries: {e}")

    def _save_current_pattern_state(self, kind: str):
        """Save current pattern state before switching to another pattern"""
        if not hasattr(self, '_current_pattern') or kind not in self._current_pattern:
            return

        current_pattern_key = self._current_pattern[kind]

        # Always save enable state if it exists (even if no parameter widgets)
        if current_pattern_key in self._enable_widgets.get(kind, {}):
            enable_widget = self._enable_widgets[kind][current_pattern_key]
            try:
                if enable_widget and enable_widget.isVisible():
                    state = enable_widget.isChecked()
                    self._pattern_states[kind][current_pattern_key] = {
                        'enabled': state,
                        'parameters': {}
                    }
            except RuntimeError:
                # Widget was deleted, skip
                pass

        # Save parameter states only if parameter widgets exist
        if current_pattern_key not in self._parameter_widgets.get(kind, {}):
            return  # No parameter widgets, but enable state was already saved above

        param_widgets = self._parameter_widgets[kind][current_pattern_key]
        for param_name, widgets in param_widgets.items():
            override_cb = widgets['override']
            param_widget = widgets['widget']

            try:
                if override_cb and param_widget and override_cb.isVisible() and param_widget.isVisible():
                    param_state = {
                        'override': override_cb.isChecked(),
                        'value': None
                    }

                    if override_cb.isChecked():  # Only save value if override is enabled
                        if isinstance(param_widget, QSpinBox):
                            param_state['value'] = param_widget.value()
                        elif isinstance(param_widget, QDoubleSpinBox):
                            param_state['value'] = param_widget.value()
                        elif isinstance(param_widget, QLineEdit):
                            param_state['value'] = param_widget.text()

                    if current_pattern_key not in self._pattern_states[kind]:
                        self._pattern_states[kind][current_pattern_key] = {'enabled': True, 'parameters': {}}
                    self._pattern_states[kind][current_pattern_key]['parameters'][param_name] = param_state
            except RuntimeError:
                # Widget was deleted, skip
                continue

    def _load_pattern_settings(self, pattern, kind, enable_cb):
        """Load saved settings for a pattern"""
        try:
            pattern_key = pattern['key']

            # First check if we have temporary states from navigation
            if (hasattr(self, '_pattern_states') and
                kind in self._pattern_states and
                pattern_key in self._pattern_states[kind]):

                temp_state = self._pattern_states[kind][pattern_key]

                # Load temporary enabled status
                enable_cb.setChecked(temp_state.get('enabled', True))

                # Load temporary parameter states
                if pattern_key in self._parameter_widgets.get(kind, {}):
                    param_widgets = self._parameter_widgets[kind][pattern_key]
                    temp_params = temp_state.get('parameters', {})

                    for param_name, widgets in param_widgets.items():
                        override_cb = widgets['override']
                        param_widget = widgets['widget']

                        if param_name in temp_params:
                            temp_param = temp_params[param_name]

                            # Set override state
                            override_cb.setChecked(temp_param.get('override', False))
                            param_widget.setEnabled(temp_param.get('override', False))

                            # Set parameter value if override is enabled
                            if temp_param.get('override', False) and temp_param.get('value') is not None:
                                try:
                                    if isinstance(param_widget, QSpinBox):
                                        param_widget.setValue(int(temp_param['value']))
                                    elif isinstance(param_widget, QDoubleSpinBox):
                                        param_widget.setValue(float(temp_param['value']))
                                    elif isinstance(param_widget, QLineEdit):
                                        param_widget.setText(str(temp_param['value']))
                                except (ValueError, TypeError):
                                    print(f"Warning: Could not set temporary value for {param_name}")

                return  # Don't load from config if we have temporary state

            # Load from configuration if no temporary state exists
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
                        try:
                            if isinstance(param_widget, QSpinBox):
                                param_widget.setValue(int(override_value))
                            elif isinstance(param_widget, QDoubleSpinBox):
                                param_widget.setValue(float(override_value))
                            elif isinstance(param_widget, QLineEdit):
                                param_widget.setText(str(override_value))
                        except (ValueError, TypeError):
                            print(f"Warning: Could not set override value for {param_name}")

        except Exception as e:
            print(f"Error loading pattern settings for {pattern.get('key', 'unknown')}: {e}")

    def _save(self):
        # Save current pattern state before saving to config
        for kind in ('chart_patterns', 'candle_patterns'):
            self._save_current_pattern_state(kind)

        # Save enabled patterns and their override parameters
        if hasattr(self, '_enable_widgets') and hasattr(self, '_parameter_widgets'):
            for kind in ('chart_patterns', 'candle_patterns'):
                if kind not in self.patterns:
                    self.patterns[kind] = {}

                enabled_patterns = []
                pattern_overrides = {}

                enable_widgets = self._enable_widgets.get(kind, {})
                param_widgets = self._parameter_widgets.get(kind, {})

                # Use temporary states if available, otherwise use current widget values
                for pattern_key in enable_widgets.keys():
                    # Check if we have temporary state for this pattern
                    if (hasattr(self, '_pattern_states') and
                        kind in self._pattern_states and
                        pattern_key in self._pattern_states[kind]):

                        temp_state = self._pattern_states[kind][pattern_key]

                        # Use temporary enabled state
                        if temp_state.get('enabled', False):
                            enabled_patterns.append(pattern_key)

                        # Use temporary parameter overrides
                        pattern_overrides[pattern_key] = {}
                        temp_params = temp_state.get('parameters', {})
                        for param_name, param_state in temp_params.items():
                            if param_state.get('override', False) and param_state.get('value') is not None:
                                pattern_overrides[pattern_key][param_name] = param_state['value']

                    else:
                        # Use current widget values
                        enable_widget = enable_widgets[pattern_key]
                        try:
                            if enable_widget and enable_widget.isVisible() and enable_widget.isChecked():
                                enabled_patterns.append(pattern_key)
                        except RuntimeError:
                            # Widget was deleted, skip
                            pass

                        # Save parameter overrides from current widgets
                        if pattern_key in param_widgets:
                            pattern_overrides[pattern_key] = {}
                            for param_name, widgets in param_widgets[pattern_key].items():
                                override_cb = widgets['override']
                                param_widget = widgets['widget']

                                try:
                                    if (override_cb and param_widget and
                                        override_cb.isVisible() and param_widget.isVisible() and
                                        override_cb.isChecked()):
                                        # Save override value
                                        if isinstance(param_widget, QSpinBox):
                                            pattern_overrides[pattern_key][param_name] = param_widget.value()
                                        elif isinstance(param_widget, QDoubleSpinBox):
                                            pattern_overrides[pattern_key][param_name] = param_widget.value()
                                        elif isinstance(param_widget, QLineEdit):
                                            pattern_overrides[pattern_key][param_name] = param_widget.text()
                                except RuntimeError:
                                    # Widget was deleted, skip
                                    continue

                self.patterns[kind]['keys_enabled'] = enabled_patterns
                self.patterns[kind]['parameter_overrides'] = pattern_overrides

        # Save dataset strategy
        strategy_text = self.cb_dataset_strategy.currentText()
        strategy_key = strategy_text.split(' - ')[0]  # Extract key (profittabilità, rischio, bilanciato)
        self.cfg['dataset_strategy'] = strategy_key

        # Save historical patterns settings
        historical_settings = {
            'enabled': self.cb_historical_enabled.isChecked(),
            'start_time': self.le_start_time.text().strip()
            # end_time removed - now defined per pattern via boundaries
        }
        self.cfg['historical_patterns'] = historical_settings

        # Save boundary configurations
        self._save_boundary_configurations()

        # Save performance settings
        self._save_performance_settings()

        self.cfg['patterns'] = self.patterns
        with open(self.yaml_path, 'w', encoding='utf-8') as fh:
            yaml.safe_dump(self.cfg, fh, allow_unicode=True, sort_keys=False)
        self.accept()

    def _save_boundary_configurations(self):
        """Save boundary configurations from UI to boundary config"""
        try:
            if not hasattr(self, '_boundary_widgets'):
                return

            # Iterate through all boundary widgets and save values
            for kind in self._boundary_widgets:
                for pattern_key in self._boundary_widgets[kind]:
                    for timeframe, widget in self._boundary_widgets[kind][pattern_key].items():
                        try:
                            if widget and hasattr(widget, 'value'):
                                boundary_value = widget.value()
                                self.boundary_config.set_boundary(pattern_key, timeframe, boundary_value)
                        except RuntimeError:
                            # Widget was deleted
                            continue

            # Save boundary config to file
            self.boundary_config.save_config()

        except Exception as e:
            print(f"Error saving boundary configurations: {e}")

    def _save_performance_settings(self):
        """Save performance settings to config"""
        try:
            if not hasattr(self, 'thread_count_spinbox'):
                return

            # Extract performance mode
            mode_text = self.performance_mode_combo.currentText()
            mode_key = mode_text.split(' - ')[0]  # Extract key (balanced, max_speed, low_cpu)

            performance_settings = {
                'detection_threads': self.thread_count_spinbox.value(),
                'detection_mode': mode_key,
                'parallel_historical': self.parallel_historical_cb.isChecked(),
                'parallel_realtime': self.parallel_realtime_cb.isChecked()
            }

            # Save to config
            if 'performance' not in self.cfg:
                self.cfg['performance'] = {}

            self.cfg['performance'].update(performance_settings)

        except Exception as e:
            print(f"Error saving performance settings: {e}")

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

        # Note about historical boundaries
        note_label = QLabel("Nota: Il confine tra patterns 'attuali' e 'storici' è ora configurato per singolo pattern nei tab sopra (sezione 'Historical Boundaries')")
        note_label.setStyleSheet("color: #0066cc; font-style: italic; font-size: 10px;")
        note_label.setWordWrap(True)
        settings_layout.addRow(note_label)

        # Visibilità basata sulle impostazioni salvate
        enabled = self.historical_settings.get('enabled', False)
        self.historical_settings_group.setVisible(enabled)

        layout.addWidget(self.historical_settings_group)

        # === MULTITHREAD PERFORMANCE SETTINGS ===
        performance_group = QGroupBox("Performance Settings")
        performance_layout = QFormLayout(performance_group)

        # Thread count setting
        self.thread_count_spinbox = QSpinBox()
        self.thread_count_spinbox.setRange(1, 64)  # Support up to 64 threads

        # Auto-detect optimal thread count (use 75% of available cores)
        import os
        cpu_count = os.cpu_count() or 8
        optimal_threads = max(1, min(32, int(cpu_count * 0.75)))

        # Load saved thread count or use optimal
        saved_threads = self.cfg.get('performance', {}).get('detection_threads', optimal_threads)
        self.thread_count_spinbox.setValue(saved_threads)

        self.thread_count_spinbox.setToolTip(f"Number of parallel threads for pattern detection. System has {cpu_count} logical cores. Recommended: {optimal_threads}")

        thread_label = QLabel(f"Detection Threads (Recommended: {optimal_threads}):")
        performance_layout.addRow(thread_label, self.thread_count_spinbox)

        # Performance mode selection
        self.performance_mode_combo = QComboBox()
        self.performance_mode_combo.addItems([
            "balanced - Balance speed and CPU usage",
            "max_speed - Maximum speed, high CPU usage",
            "low_cpu - Lower CPU usage, slower detection"
        ])

        # Load saved mode
        saved_mode = self.cfg.get('performance', {}).get('detection_mode', 'balanced')
        mode_items = {
            'balanced': "balanced - Balance speed and CPU usage",
            'max_speed': "max_speed - Maximum speed, high CPU usage",
            'low_cpu': "low_cpu - Lower CPU usage, slower detection"
        }
        if saved_mode in mode_items:
            self.performance_mode_combo.setCurrentText(mode_items[saved_mode])

        performance_layout.addRow(QLabel("Performance Mode:"), self.performance_mode_combo)

        # System info display
        info_label = QLabel(f"System: {cpu_count} logical cores detected")
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        performance_layout.addRow(info_label)

        # Parallel scanning options
        self.parallel_historical_cb = QCheckBox("Enable parallel historical scanning")
        self.parallel_historical_cb.setChecked(self.cfg.get('performance', {}).get('parallel_historical', True))
        self.parallel_historical_cb.setToolTip("Use multithread detection for historical pattern scanning")
        performance_layout.addRow(self.parallel_historical_cb)

        self.parallel_realtime_cb = QCheckBox("Enable parallel real-time scanning")
        self.parallel_realtime_cb.setChecked(self.cfg.get('performance', {}).get('parallel_realtime', True))
        self.parallel_realtime_cb.setToolTip("Use multithread detection for real-time pattern scanning")
        performance_layout.addRow(self.parallel_realtime_cb)

        layout.addWidget(performance_group)
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
