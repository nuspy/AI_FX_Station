
from __future__ import annotations
from typing import List, Dict, Any
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QTabWidget, QWidget, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLabel, QComboBox, QLineEdit)
from PySide6.QtCore import Qt
import yaml, os

class PatternsConfigDialog(QDialog):
    def __init__(self, parent=None, yaml_path:str="configs/patterns.yaml", patterns_service=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configura Patterns")
        self.resize(820, 560)
        self.yaml_path = yaml_path
        self.patterns_service = patterns_service
        with open(self.yaml_path, 'r', encoding='utf-8') as fh:
            self.cfg = yaml.safe_load(fh) or {}
        self.patterns = self.cfg.get('patterns', {})
        self.historical_settings = self.cfg.get('historical_patterns', {
            'enabled': False,
            'start_time': '30d',
            'end_time': '7d'
        })
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        tabs = QTabWidget(self); lay.addWidget(tabs)

        self.chart_tab = self._make_tab(kind='chart_patterns')
        self.candle_tab = self._make_tab(kind='candle_patterns')
        self.historical_tab = self._make_historical_tab()

        tabs.addTab(self.chart_tab, "Patterns a Chart")
        tabs.addTab(self.candle_tab, "Patterns a Candela")
        tabs.addTab(self.historical_tab, "Patterns Storici")

        btns = QHBoxLayout()
        self.btn_save = QPushButton("Salva")
        self.btn_cancel = QPushButton("Chiudi")
        btns.addStretch(1); btns.addWidget(self.btn_save); btns.addWidget(self.btn_cancel)
        lay.addLayout(btns)
        self.btn_save.clicked.connect(self._save)
        self.btn_cancel.clicked.connect(self.reject)

    def _make_tab(self, kind: str) -> QWidget:
        box = QWidget(); hl = QHBoxLayout(box)
        left = QListWidget(); right = QWidget(); form = QFormLayout(right)
        data = self.patterns.get(kind, {})
        keys = list(data.get('keys_enabled', []))
        self._widgets: Dict[str, Dict[str, Any]] = getattr(self, '_widgets', {})
        self._widgets[kind] = {}

        for k in keys:
            QListWidgetItem(k, left)
        # per-pattern configurable controls
        # thresholds / confidence / targets
        for k in keys:
            row = QWidget(); row_l = QHBoxLayout(row)
            cb_en = QCheckBox("Attivo"); cb_en.setChecked(True)
            sp_conf = QDoubleSpinBox(); sp_conf.setRange(0.0, 1.0); sp_conf.setSingleStep(0.05); sp_conf.setValue(0.5)
            sp_th = QDoubleSpinBox(); sp_th.setRange(0.0, 10.0); sp_th.setSingleStep(0.1); sp_th.setValue(1.0)
            dd_ts = QComboBox(); dd_ts.addItems(["auto","height","flag_pole","atr_multiple"])
            sp_ts = QDoubleSpinBox(); sp_ts.setRange(0.0, 10.0); sp_ts.setSingleStep(0.1); sp_ts.setValue(1.5)
            row_l.addWidget(QLabel(k)); row_l.addWidget(cb_en); row_l.addWidget(QLabel("Conf")); row_l.addWidget(sp_conf)
            row_l.addWidget(QLabel("Soglia")); row_l.addWidget(sp_th)
            row_l.addWidget(QLabel("Target/Stop")); row_l.addWidget(dd_ts); row_l.addWidget(sp_ts)
            form.addRow(row)
            self._widgets[kind][k] = {"enabled":cb_en, "conf":sp_conf, "th":sp_th, "ts_mode":dd_ts, "ts_val":sp_ts}

        hl.addWidget(left, 2); hl.addWidget(right, 5)
        return box

    def _save(self):
        # Persist 'enabled' flags back under keys_enabled and store per-key params
        for kind in ('chart_patterns','candle_patterns'):
            if kind not in self.patterns: self.patterns[kind] = {}
            keys = list(self.patterns[kind].get('keys_enabled', []))
            w = self._widgets.get(kind, {})
            kept = []
            per_key = self.patterns[kind].setdefault('params', {})
            for k in keys:
                wi = w.get(k, {})
                en = wi.get('enabled')
                if en is None or en.isChecked():
                    kept.append(k)
                per_key[k] = {
                    'confidence': float(wi.get('conf').value()) if wi.get('conf') else 0.5,
                    'threshold': float(wi.get('th').value()) if wi.get('th') else 1.0,
                    'target_stop': {'mode': wi.get('ts_mode').currentText() if wi.get('ts_mode') else 'auto',
                                    'value': float(wi.get('ts_val').value()) if wi.get('ts_val') else 1.5}
                }
            self.patterns[kind]['keys_enabled'] = kept

        # Salva le impostazioni dei patterns storici
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

    def _make_historical_tab(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)

        # Checkbox per abilitare patterns storici
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

        # Bottone scansiona patterns storici
        self.btn_scan_historical = QPushButton("Scansiona Patterns Storici")
        self.btn_scan_historical.setEnabled(self.historical_settings.get('enabled', False))
        settings_layout.addRow(self.btn_scan_historical)

        # Visibilità basata sulle impostazioni salvate
        enabled = self.historical_settings.get('enabled', False)
        self.historical_settings_group.setVisible(enabled)

        layout.addWidget(self.historical_settings_group)
        layout.addStretch()

        # Connetti il checkbox per mostrare/nascondere le impostazioni
        self.cb_historical_enabled.toggled.connect(self._on_historical_enabled_changed)

        # Connetti il bottone di scansione storica
        self.btn_scan_historical.clicked.connect(self._on_scan_historical_clicked)

        return box

    def _on_historical_enabled_changed(self, enabled: bool):
        """Mostra/nasconde le impostazioni quando historical patterns è abilitato"""
        self.historical_settings_group.setVisible(enabled)
        self.btn_scan_historical.setEnabled(enabled)

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
