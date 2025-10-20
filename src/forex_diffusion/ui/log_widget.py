"""
Log Widget - Display application logs with filtering and search capabilities.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Dict
import re

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QComboBox,
    QLineEdit, QPushButton, QLabel, QCheckBox
)
from PySide6.QtCore import Signal, QTimer, Slot
from PySide6.QtGui import QTextCursor, QColor, QTextCharFormat
from loguru import logger


class LogWidget(QWidget):
    """
    Widget to display application logs with filtering and search.

    Features:
    - Real-time log display
    - Level filtering (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Text search
    - Auto-scroll toggle
    - Color-coded log levels
    - Log statistics
    """

    logReceived = Signal(dict)  # Signal when new log is received (thread-safe)

    # Class-level storage for logs (shared across instances)
    _log_buffer: Deque[Dict] = deque(maxlen=10000)  # Last 10k logs
    _instances = []

    def __init__(self, parent=None):
        super().__init__(parent)

        # Register this instance
        LogWidget._instances.append(self)

        # State
        self._auto_scroll = True
        self._paused = False

        # Statistics
        self._stats = {
            'debug': 0,
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }

        self._init_ui()
        self._load_existing_logs()

        # Connect signal for thread-safe log updates
        self.logReceived.connect(self._on_log_received_gui_thread)

        # Update timer for periodic refresh
        # IMPORTANT: QTimer must be created on GUI thread
        # Use QTimer.singleShot to defer creation to GUI thread event loop
        self._update_timer = None
        from PySide6.QtCore import QMetaObject, Qt as QtCore
        QMetaObject.invokeMethod(self, "_init_timer", QtCore.ConnectionType.QueuedConnection)

    @Slot()
    def _init_timer(self):
        """Initialize update timer on GUI thread."""
        if self._update_timer is None:
            self._update_timer = QTimer(self)
            self._update_timer.timeout.connect(self._update_display)
            self._update_timer.start(1000)  # Update every second

    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Controls row 1: Level filter + search
        controls1 = QHBoxLayout()

        # Level filter
        controls1.addWidget(QLabel("Level:"))
        self.level_filter = QComboBox()
        self.level_filter.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_filter.setCurrentText("INFO")
        self.level_filter.currentTextChanged.connect(self._apply_filters)
        controls1.addWidget(self.level_filter)

        controls1.addSpacing(10)

        # Qt messages only filter
        self.q_messages_only = QCheckBox("Qt messages only")
        self.q_messages_only.setChecked(True)  # Default: enabled
        self.q_messages_only.setToolTip("Show only Qt messages (qDebug, qWarning, qCritical, qFatal)")
        self.q_messages_only.toggled.connect(self._apply_filters)
        controls1.addWidget(self.q_messages_only)

        controls1.addSpacing(10)

        # Search
        controls1.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter text to search...")
        self.search_input.textChanged.connect(self._apply_filters)
        controls1.addWidget(self.search_input)

        # Search options
        self.case_sensitive = QCheckBox("Case Sensitive")
        controls1.addWidget(self.case_sensitive)
        self.case_sensitive.toggled.connect(self._apply_filters)

        self.regex_search = QCheckBox("Regex")
        controls1.addWidget(self.regex_search)
        self.regex_search.toggled.connect(self._apply_filters)

        layout.addLayout(controls1)

        # Controls row 2: Actions
        controls2 = QHBoxLayout()

        # Auto-scroll toggle
        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.toggled.connect(self._toggle_auto_scroll)
        controls2.addWidget(self.auto_scroll_cb)

        # Pause toggle
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._toggle_pause)
        controls2.addWidget(self.pause_btn)

        # Clear button
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self._clear_logs)
        controls2.addWidget(clear_btn)

        # Export button
        export_btn = QPushButton("💾 Export")
        export_btn.clicked.connect(self._export_logs)
        controls2.addWidget(export_btn)

        controls2.addStretch()

        # Statistics
        self.stats_label = QLabel("Entries: 0 | 🔵 0 | 🟢 0 | 🟡 0 | 🔴 0 | 🔥 0")
        controls2.addWidget(self.stats_label)

        layout.addLayout(controls2)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        # Monospace font for logs
        font = self.log_display.font()
        font.setFamily("Consolas, Monaco, Courier New, monospace")
        font.setPointSize(9)
        self.log_display.setFont(font)

        # Dark background for logs
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
            }
        """)

        layout.addWidget(self.log_display)

    def _load_existing_logs(self):
        """Load logs that were already captured before widget creation."""
        # Create snapshot to avoid "deque mutated during iteration" error
        buffer_snapshot = list(LogWidget._log_buffer)
        for log_entry in buffer_snapshot:
            self._process_log(log_entry, update_display=False)
        self._update_display()

    def _on_log_received_gui_thread(self, log_entry: Dict):
        """Handle log received on GUI thread (called via signal)."""
        if self._paused:
            return

        # Update statistics
        level = log_entry.get('level', 'INFO').upper()
        level_key = level.lower()
        if level_key in self._stats:
            self._stats[level_key] += 1

    def _process_log(self, log_entry: Dict, update_display: bool = True):
        """Process a log entry and emit signal for thread-safe update."""
        # Emit signal instead of directly updating (thread-safe)
        try:
            self.logReceived.emit(log_entry)
        except RuntimeError:
            # Widget might be deleted
            pass

    def _update_display(self):
        """Update the log display with filtered logs."""
        if self._paused:
            return

        # Get filter settings
        level_filter = self.level_filter.currentText()
        search_text = self.search_input.text()
        case_sensitive = self.case_sensitive.isChecked()
        use_regex = self.regex_search.isChecked()
        q_messages_only = self.q_messages_only.isChecked()

        # Clear display
        self.log_display.clear()

        # Prepare search pattern
        search_pattern = None
        if search_text:
            if use_regex:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    search_pattern = re.compile(search_text, flags)
                except re.error:
                    search_pattern = None
            else:
                search_text_lower = search_text if case_sensitive else search_text.lower()

        # Filter and display logs
        # Create snapshot to avoid "deque mutated during iteration" error
        buffer_snapshot = list(LogWidget._log_buffer)
        displayed = 0
        for log_entry in buffer_snapshot:
            level = log_entry.get('level', 'INFO').upper()
            message = log_entry.get('message', '')
            timestamp = log_entry.get('timestamp', '')
            module = log_entry.get('module', '')
            function = log_entry.get('function', '')
            line = log_entry.get('line', '')

            # Qt messages only filter
            if q_messages_only:
                # Check if message contains '[Qt]' (Qt library messages)
                if '[Qt]' not in message and 'qt' not in module.lower():
                    continue

            # Level filter
            if level_filter != "ALL":
                # Get level priority
                level_priority = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
                filter_priority = level_priority.get(level_filter, 1)
                current_priority = level_priority.get(level, 1)
                if current_priority < filter_priority:
                    continue

            # Search filter
            if search_text:
                full_text = f"{timestamp} {level} {module} {function} {message}"

                if use_regex and search_pattern:
                    if not search_pattern.search(full_text):
                        continue
                else:
                    search_in = full_text if case_sensitive else full_text.lower()
                    if search_text_lower not in search_in:
                        continue

            # Format and append log
            self._append_log(log_entry)
            displayed += 1

        # Auto-scroll to bottom
        if self._auto_scroll:
            self.log_display.moveCursor(QTextCursor.MoveOperation.End)

        # Update statistics
        total = sum(self._stats.values())
        self.stats_label.setText(
            f"Entries: {total} (Showing: {displayed}) | "
            f"🔵 {self._stats['debug']} | "
            f"🟢 {self._stats['info']} | "
            f"🟡 {self._stats['warning']} | "
            f"🔴 {self._stats['error']} | "
            f"🔥 {self._stats['critical']}"
        )

    def _append_log(self, log_entry: Dict):
        """Append a formatted log entry to the display."""
        level = log_entry.get('level', 'INFO').upper()
        message = log_entry.get('message', '')
        timestamp = log_entry.get('timestamp', '')
        module = log_entry.get('module', '')
        function = log_entry.get('function', '')
        line = log_entry.get('line', '')

        # Color based on level
        color_map = {
            'DEBUG': '#569cd6',      # Blue
            'INFO': '#4ec9b0',       # Cyan/Green
            'WARNING': '#dcdcaa',    # Yellow
            'ERROR': '#f48771',      # Orange/Red
            'CRITICAL': '#f14c4c',   # Bright Red
        }
        color = color_map.get(level, '#d4d4d4')

        # Level icon
        icon_map = {
            'DEBUG': '🔵',
            'INFO': '🟢',
            'WARNING': '🟡',
            'ERROR': '🔴',
            'CRITICAL': '🔥',
        }
        icon = icon_map.get(level, '⚪')

        # Format: [timestamp] LEVEL icon | module:function:line | message
        location = f"{module}:{function}:{line}" if module else ""
        formatted = f"[{timestamp}] {level:8} {icon} | {location:40} | {message}\n"

        # Apply color
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(formatted)

    def _apply_filters(self):
        """Apply current filters to the log display."""
        self._update_display()

    def _toggle_auto_scroll(self, checked: bool):
        """Toggle auto-scroll."""
        self._auto_scroll = checked

    def _toggle_pause(self, checked: bool):
        """Toggle pause."""
        self._paused = checked
        if checked:
            self.pause_btn.setText("▶️ Resume")
            if self._update_timer:
                self._update_timer.stop()
        else:
            self.pause_btn.setText("⏸ Pause")
            if self._update_timer:
                self._update_timer.start(1000)
            self._update_display()

    def _clear_logs(self):
        """Clear all logs."""
        LogWidget._log_buffer.clear()
        self.log_display.clear()
        self._stats = {
            'debug': 0,
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }
        self._update_display()
        logger.info("Logs cleared")

    def _export_logs(self):
        """Export logs to a file."""
        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Logs",
            f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                # Create snapshot to avoid "deque mutated during iteration" error
                buffer_snapshot = list(LogWidget._log_buffer)
                with open(file_path, 'w', encoding='utf-8') as f:
                    for log_entry in buffer_snapshot:
                        level = log_entry.get('level', 'INFO')
                        message = log_entry.get('message', '')
                        timestamp = log_entry.get('timestamp', '')
                        module = log_entry.get('module', '')
                        function = log_entry.get('function', '')
                        line = log_entry.get('line', '')

                        f.write(f"[{timestamp}] {level:8} | {module}:{function}:{line} | {message}\n")

                logger.info(f"Logs exported to {file_path}")
            except Exception as e:
                logger.error(f"Failed to export logs: {e}")

    @classmethod
    def capture_log(cls, log_entry: Dict):
        """
        Capture a log entry (class method called by log handler).

        Args:
            log_entry: Dict with keys: level, message, timestamp, module, function, line
        """
        # Add to buffer
        cls._log_buffer.append(log_entry)

        # Notify all instances
        for instance in cls._instances:
            if instance and not instance._paused:
                instance._process_log(log_entry)

    def closeEvent(self, event):
        """Handle widget close."""
        # Unregister instance
        if self in LogWidget._instances:
            LogWidget._instances.remove(self)
        event.accept()


# Custom log handler for loguru
class QtLogHandler:
    """Custom log handler that captures logs and sends to LogWidget."""

    def write(self, message):
        """Write log message."""
        # loguru sends formatted messages, we need to parse them
        # Format: "2025-10-09 18:30:55.596 | INFO     | module:function:line - message"
        try:
            # Simple parsing (loguru already formats it)
            import re

            # Pattern: timestamp | LEVEL | location - message
            pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) \| ([A-Z]+)\s+\| ([^:]+):([^:]+):(\d+) - (.+)'
            match = re.match(pattern, message.strip())

            if match:
                timestamp, level, module, function, line, msg = match.groups()
                log_entry = {
                    'timestamp': timestamp,
                    'level': level.strip(),
                    'module': module.strip(),
                    'function': function.strip(),
                    'line': line.strip(),
                    'message': msg.strip()
                }
            else:
                # Fallback: just capture the message
                log_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'level': 'INFO',
                    'module': 'unknown',
                    'function': 'unknown',
                    'line': '0',
                    'message': message.strip()
                }

            # Send to LogWidget
            LogWidget.capture_log(log_entry)

        except Exception:
            # Don't break logging if parsing fails
            pass

    def flush(self):
        """Flush (no-op for our use case)."""
        pass


# Qt message handler
def qt_message_handler(mode, context, message):
    """Handle Qt log messages (qDebug, qWarning, qCritical, qFatal)."""
    from PySide6.QtCore import QtMsgType

    # Map Qt message types to log levels
    level_map = {
        QtMsgType.QtDebugMsg: 'DEBUG',
        QtMsgType.QtInfoMsg: 'INFO',
        QtMsgType.QtWarningMsg: 'WARNING',
        QtMsgType.QtCriticalMsg: 'ERROR',
        QtMsgType.QtFatalMsg: 'CRITICAL',
    }

    level = level_map.get(mode, 'INFO')

    # Extract context information
    file_name = context.file if context.file else 'qt'
    function = context.function if context.function else 'unknown'
    line = str(context.line) if context.line else '0'

    # Create log entry
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        'level': level,
        'module': file_name,
        'function': function,
        'line': line,
        'message': message
    }

    # Send to LogWidget
    LogWidget.capture_log(log_entry)

    # Also send to loguru for file logging
    if level == 'DEBUG':
        logger.debug(f"[Qt] {message}")
    elif level == 'INFO':
        logger.info(f"[Qt] {message}")
    elif level == 'WARNING':
        logger.warning(f"[Qt] {message}")
    elif level == 'ERROR':
        logger.error(f"[Qt] {message}")
    elif level == 'CRITICAL':
        logger.critical(f"[Qt] {message}")


# Install log handler
def install_log_handler():
    """Install the Qt log handler for loguru and Qt messages."""
    from PySide6.QtCore import qInstallMessageHandler

    # Install loguru handler
    handler = QtLogHandler()
    logger.add(handler, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:8} | {name}:{function}:{line} - {message}")

    # Install Qt message handler
    qInstallMessageHandler(qt_message_handler)

    logger.info("Qt log handler installed (loguru + Qt messages)")
