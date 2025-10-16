from __future__ import annotations

from typing import Any, Dict, List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QCheckBox,
    QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QGroupBox, QFileDialog, QGridLayout,
    QScrollArea
)

from .prediction_settings_dialog import PredictionSettingsDialog
from ..utils.user_settings import get_setting, set_setting


INDICATORS = ["ATR", "RSI", "Bollinger", "MACD", "Donchian", "Keltner", "Hurst"]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
DEFAULT_INDICATOR_TFS = {
    "ATR": ["1m", "5m", "15m", "30m", "1h"],
    "RSI": ["1m", "5m", "15m", "30m", "1h"],
    "Bollinger": ["5m", "15m", "30m"],
    "MACD": ["15m", "30m", "1h"],
    "Donchian": ["30m", "1h", "4h"],
    "Keltner": ["15m", "30m", "1h"],
    "Hurst": ["30m", "1h", "4h", "1d"],
}
NUMERIC_PARAM_BOUNDS = {
    "N_samples": (50, 5000, 50, 200),
    "model_weight_pct": (0, 100, 10, 100),
    "warmup_bars": (0, 5000, 16, 16),
    "atr_n": (1, 500, 1, 14),
    "rsi_n": (2, 500, 1, 14),
    "bb_n": (2, 500, 1, 20),
    "rv_window": (5, 1000, 5, 60),
    "ema_fast": (1, 100, 1, 12),
    "ema_slow": (1, 200, 5, 26),
    "don_n": (5, 500, 5, 20),
    "hurst_window": (8, 4096, 8, 64),
    "keltner_k": (1, 10, 1, 1),
    "max_forecasts": (1, 100, 1, 5),
    "test_history_bars": (10, 10000, 10, 128),
    "auto_interval_seconds": (5, 86400, 5, 60),
}
BOOLEAN_PARAMS = ["apply_conformal", "auto_predict"]


class BacktestingTab(QWidget):
    startRequested = Signal(dict)
    pauseRequested = Signal()
    stopRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_prediction_defaults()
        self._load_persisted()
        # track shown results to append only new rows during polling
        self._shown_config_ids: set[int] = set()
        self._row_by_config_id: dict[int,int] = {}
        # local polling timer for job status (DB-only)
        try:
            from PySide6.QtCore import QTimer
            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(500)
            self._poll_timer.timeout.connect(self._poll_job_status)
        except Exception:
            self._poll_timer = None
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()

    def _build_ui(self):
        # Scrollable content root
        root = QVBoxLayout(self)
        content = QWidget()
        lay = QVBoxLayout(content)

        # Prediction types
        grp_types = QGroupBox("Tipo di previsione")
        lt = QHBoxLayout(grp_types)
        self.chk_basic = QCheckBox("Basic"); self.chk_basic.setChecked(True)
        self.chk_adv = QCheckBox("Advanced")
        self.chk_rw = QCheckBox("Baseline (RW)")
        lt.addWidget(self.chk_basic); lt.addWidget(self.chk_adv); lt.addWidget(self.chk_rw)
        lay.addWidget(grp_types)

        # Timeframe + Horizons
        grp_h = QGroupBox("Indicatori × Timeframe / Orizzonti")
        lh = QVBoxLayout(grp_h)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Timeframe"))
        self.cmb_tf = QComboBox(); self.cmb_tf.addItems(["1m","5m","15m","30m","1h","4h","1d"])
        row1.addWidget(self.cmb_tf)
        row1.addWidget(QLabel("Orizzonti"))
        self.ed_h = QLineEdit("30s, 1m, (5-15)m, 30m, 1h, 2h")
        row1.addWidget(self.ed_h)
        lh.addLayout(row1)
        lay.addWidget(grp_h)

        # Indicator selection grid with per-indicator params
        ind_group = QGroupBox("Indicatori tecnici per TimeFrame")
        ind_grid = QGridLayout(ind_group)
        # headers
        ind_grid.addWidget(QLabel("Indicatore"), 0, 0)
        ind_grid.addWidget(QLabel("Parametri (da / a / passo)"), 0, 1)
        for j, tf in enumerate(TIMEFRAMES, start=2):
            ind_grid.addWidget(QLabel(tf), 0, j)
        self.indicator_checks: Dict[str, Dict[str, QCheckBox]] = {}
        # store per-indicator param editors
        self.indicator_param_ranges: Dict[str, Dict[str, Tuple[QSpinBox, QSpinBox, QSpinBox]]] = {}
        for i, ind in enumerate(INDICATORS, start=1):
            # label + default button
            row_box = QHBoxLayout()
            lbl = QLabel(ind)
            btn_default = QPushButton("Default")
            btn_default.setFixedWidth(64)
            btn_default.clicked.connect(lambda _, name=ind: self._reset_indicator_row(name))
            wrap0 = QWidget(); _hb = QHBoxLayout(wrap0); _hb.setContentsMargins(0,0,0,0)
            _hb.addWidget(lbl); _hb.addWidget(btn_default); _hb.addStretch(1)
            ind_grid.addWidget(wrap0, i, 0)

            # parameter editors per indicator
            params_wrap = QWidget()
            params_lay = QHBoxLayout(params_wrap); params_lay.setContentsMargins(0,0,0,0)
            param_editors: Dict[str, Tuple[QSpinBox, QSpinBox, QSpinBox]] = {}
            def _mk_triplet(vmin: int, vmax: int, step: int, default: int) -> Tuple[QSpinBox, QSpinBox, QSpinBox]:
                sp_from = QSpinBox(); sp_from.setRange(vmin, vmax); sp_from.setValue(default)
                sp_to = QSpinBox(); sp_to.setRange(vmin, vmax); sp_to.setValue(default)
                sp_step = QSpinBox(); sp_step.setRange(1, max(1, vmax - vmin)); sp_step.setValue(max(1, step))
                return sp_from, sp_to, sp_step
            # map indicator -> param keys from NUMERIC_PARAM_BOUNDS
            if ind == "ATR":
                vmin, vmax, step, default = NUMERIC_PARAM_BOUNDS["atr_n"]
                spf, spt, sps = _mk_triplet(vmin, vmax, step, default)
                params_lay.addWidget(QLabel("n:")); params_lay.addWidget(spf); params_lay.addWidget(spt); params_lay.addWidget(sps)
                param_editors["atr_n"] = (spf, spt, sps)
            elif ind == "RSI":
                vmin, vmax, step, default = NUMERIC_PARAM_BOUNDS["rsi_n"]
                spf, spt, sps = _mk_triplet(vmin, vmax, step, default)
                params_lay.addWidget(QLabel("n:")); params_lay.addWidget(spf); params_lay.addWidget(spt); params_lay.addWidget(sps)
                param_editors["rsi_n"] = (spf, spt, sps)
            elif ind == "Bollinger":
                vmin, vmax, step, default = NUMERIC_PARAM_BOUNDS["bb_n"]
                spf, spt, sps = _mk_triplet(vmin, vmax, step, default)
                params_lay.addWidget(QLabel("n:")); params_lay.addWidget(spf); params_lay.addWidget(spt); params_lay.addWidget(sps)
                param_editors["bb_n"] = (spf, spt, sps)
            elif ind == "MACD":
                vminf, vmaxf, stepf, deff = NUMERIC_PARAM_BOUNDS["ema_fast"]
                spff, sptf, spsf = _mk_triplet(vminf, vmaxf, stepf, deff)
                vmins, vmaxs, steps, defs = NUMERIC_PARAM_BOUNDS["ema_slow"]
                spfs, spts, spss = _mk_triplet(vmins, vmaxs, steps, defs)
                params_lay.addWidget(QLabel("fast:")); params_lay.addWidget(spff); params_lay.addWidget(sptf); params_lay.addWidget(spsf)
                params_lay.addWidget(QLabel("slow:")); params_lay.addWidget(spfs); params_lay.addWidget(spts); params_lay.addWidget(spss)
                param_editors["ema_fast"] = (spff, sptf, spsf)
                param_editors["ema_slow"] = (spfs, spts, spss)
            elif ind == "Donchian":
                vmin, vmax, step, default = NUMERIC_PARAM_BOUNDS["don_n"]
                spf, spt, sps = _mk_triplet(vmin, vmax, step, default)
                params_lay.addWidget(QLabel("n:")); params_lay.addWidget(spf); params_lay.addWidget(spt); params_lay.addWidget(sps)
                param_editors["don_n"] = (spf, spt, sps)
            elif ind == "Keltner":
                vmin, vmax, step, default = NUMERIC_PARAM_BOUNDS["keltner_k"]
                spf, spt, sps = _mk_triplet(vmin, vmax, step, default)
                params_lay.addWidget(QLabel("k:")); params_lay.addWidget(spf); params_lay.addWidget(spt); params_lay.addWidget(sps)
                param_editors["keltner_k"] = (spf, spt, sps)
            elif ind == "Hurst":
                vmin, vmax, step, default = NUMERIC_PARAM_BOUNDS["hurst_window"]
                spf, spt, sps = _mk_triplet(vmin, vmax, step, default)
                params_lay.addWidget(QLabel("window:")); params_lay.addWidget(spf); params_lay.addWidget(spt); params_lay.addWidget(sps)
                param_editors["hurst_window"] = (spf, spt, sps)
            ind_grid.addWidget(params_wrap, i, 1)
            self.indicator_param_ranges[ind] = param_editors

            # timeframe checkboxes
            self.indicator_checks[ind] = {}
            defaults = DEFAULT_INDICATOR_TFS.get(ind, [])
            for j, tf in enumerate(TIMEFRAMES, start=2):
                cb = QCheckBox()
                cb.setChecked(tf in defaults)
                self.indicator_checks[ind][tf] = cb
                ind_grid.addWidget(cb, i, j)
        lay.addWidget(ind_group)

        # Forecast parameter ranges (non-indicator)
        params_group = QGroupBox("Forecast: Parametri (range da / a / passo)")
        params_layout = QGridLayout(params_group)
        params_layout.addWidget(QLabel("Parametro"), 0, 0)
        params_layout.addWidget(QLabel("Da"), 0, 1)
        params_layout.addWidget(QLabel("A"), 0, 2)
        params_layout.addWidget(QLabel("Passo"), 0, 3)
        self.range_fields: Dict[str, Tuple[QSpinBox, QSpinBox, QSpinBox]] = {}
        # omit indicator-specific keys (handled in indicator grid)
        skip_keys = {"atr_n","rsi_n","bb_n","ema_fast","ema_slow","don_n","hurst_window","keltner_k"}
        r = 1
        for param, (vmin, vmax, step, default) in NUMERIC_PARAM_BOUNDS.items():
            if param in skip_keys:
                continue
            row_idx = r; r += 1
            lbl = QLabel(param)
            sp_from = QSpinBox(); sp_from.setRange(vmin, vmax); sp_from.setValue(default)
            sp_to = QSpinBox(); sp_to.setRange(vmin, vmax); sp_to.setValue(default)
            sp_step = QSpinBox(); sp_step.setRange(1, max(1, vmax - vmin)); sp_step.setValue(max(1, step))
            params_layout.addWidget(lbl, row_idx, 0)
            params_layout.addWidget(sp_from, row_idx, 1)
            params_layout.addWidget(sp_to, row_idx, 2)
            params_layout.addWidget(sp_step, row_idx, 3)
            self.range_fields[param] = (sp_from, sp_to, sp_step)
        lay.addWidget(params_group)

        bool_group = QGroupBox("Parametri booleani")
        from PySide6.QtWidgets import QGridLayout as _QGrid
        bool_layout = _QGrid(bool_group)
        self.boolean_fields: Dict[str, Tuple[QCheckBox, QCheckBox]] = {}
        for idx, param in enumerate(BOOLEAN_PARAMS):
            lbl = QLabel(param)
            cb_true = QCheckBox("True"); cb_true.setChecked(True)
            cb_false = QCheckBox("False")
            bool_layout.addWidget(lbl, idx, 0)
            bool_layout.addWidget(cb_true, idx, 1)
            bool_layout.addWidget(cb_false, idx, 2)
            self.boolean_fields[param] = (cb_true, cb_false)
        lay.addWidget(bool_group)

        # Candle parameters (flags)
        flags_group = QGroupBox("Parametri candele (flags)")
        fl = QHBoxLayout(flags_group)
        self.cb_use_hours = QCheckBox("Usa ore")
        self.cb_use_day = QCheckBox("Usa giorno")
        fl.addWidget(self.cb_use_hours)
        fl.addWidget(self.cb_use_day)
        fl.addStretch(1)
        lay.addWidget(flags_group)

# Models
        grp_m = QGroupBox("Modelli")
        lm = QHBoxLayout(grp_m)
        self.ed_models = QLineEdit("")
        btn_browse = QPushButton("Sfoglia Modelli")
        btn_browse.clicked.connect(self._browse_models)
        lm.addWidget(self.ed_models); lm.addWidget(btn_browse)
        lay.addWidget(grp_m)

        # Samples
        grp_s = QGroupBox("Samples (Da/A/Passo)")
        ls = QHBoxLayout(grp_s)
        self.sp_s = QSpinBox(); self.sp_s.setRange(1, 100000); self.sp_s.setValue(200)
        self.sp_e = QSpinBox(); self.sp_e.setRange(1, 100000); self.sp_e.setValue(1500)
        self.sp_p = QSpinBox(); self.sp_p.setRange(1, 10000); self.sp_p.setValue(200)
        ls.addWidget(QLabel("Da")); ls.addWidget(self.sp_s)
        ls.addWidget(QLabel("A")); ls.addWidget(self.sp_e)
        ls.addWidget(QLabel("Passo")); ls.addWidget(self.sp_p)
        lay.addWidget(grp_s)

        # Interval presets
        grp_i = QGroupBox("Intervallo")
        li = QHBoxLayout(grp_i)
        self.cmb_preset = QComboBox(); self.cmb_preset.addItems(["7d","30d","90d","180d","YTD","1Y","3Y"]) 
        li.addWidget(QLabel("Preset")); li.addWidget(self.cmb_preset)
        lay.addWidget(grp_i)

        # Exec
        row_exec = QHBoxLayout()
        self.chk_cache = QCheckBox("Evita ricalcoli (cache DB)"); self.chk_cache.setChecked(True)
        self.btn_start = QPushButton("Avvia"); self.btn_pause = QPushButton("Pausa"); self.btn_stop = QPushButton("Ferma")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(self.pauseRequested.emit)
        self.btn_stop.clicked.connect(self.stopRequested.emit)
        row_exec.addWidget(self.chk_cache); row_exec.addStretch(1)
        row_exec.addWidget(self.btn_start); row_exec.addWidget(self.btn_pause); row_exec.addWidget(self.btn_stop)
        lay.addLayout(row_exec)

        # Results table (expanded) + actions
        self.tbl = QTableWidget(0, 10)
        self.tbl.setHorizontalHeaderLabels(["ConfigId","Modello","Tipo","TF","Adh_mean","p50","Win@Δ","Coverage","BandEff","Score"]) 
        lay.addWidget(self.tbl)
        act_row = QHBoxLayout()
        self.btn_apply_basic = QPushButton("Applica a Basic")
        self.btn_apply_adv = QPushButton("Applica a Advanced")
        self.btn_pause_job = QPushButton("Pausa Job")
        self.btn_cancel_job = QPushButton("Cancella Job")
        self.btn_resume_job = QPushButton("Resume Job")
        self.btn_cancel_config = QPushButton("Cancel Config")
        self.btn_apply_basic.clicked.connect(lambda: self._apply_selected("Basic"))
        self.btn_apply_adv.clicked.connect(lambda: self._apply_selected("Advanced"))
        self.btn_pause_job.clicked.connect(self._pause_job)
        self.btn_cancel_job.clicked.connect(self._cancel_job)
        self.btn_resume_job.clicked.connect(self._resume_job)
        self.btn_cancel_config.clicked.connect(self._cancel_selected_config)
        act_row.addWidget(self.btn_apply_basic)
        act_row.addWidget(self.btn_apply_adv)
        act_row.addWidget(self.btn_pause_job)
        act_row.addWidget(self.btn_cancel_job)
        act_row.addWidget(self.btn_resume_job)
        act_row.addWidget(self.btn_cancel_config)
        lay.addLayout(act_row)

        # Profiles plot (pyqtgraph minimal)
        try:
            import pyqtgraph as pg
            self.plot = pg.PlotWidget()
            self.plot.setBackground('w')
            self.plot.showGrid(x=True, y=True, alpha=0.3)
            try:
                self.plot.setMinimumHeight(360)
            except Exception:
                pass
            try:
                self.legend = self.plot.addLegend()
            except Exception:
                self.legend = None
            # Save button under plot
            save_row = QHBoxLayout()
            self.btn_save_plot = QPushButton("Salva Grafico…")
            self.btn_save_plot.clicked.connect(self._save_plot)
            save_row.addWidget(self.btn_save_plot)
            lay.addLayout(save_row)
            lay.addWidget(self.plot)
        except Exception:
            self.plot = None

        # Status
        self.lbl_status = QLabel("")
        lay.addWidget(self.lbl_status)

        # finalize scroll container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        root.addWidget(scroll)

    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets"""
        from ..i18n.widget_helper import apply_tooltip
        
        # Backtesting parameters (10 tooltips from backtesting category)
        if hasattr(self, 'initial_balance_spin'):
            apply_tooltip(self.initial_balance_spin, "initial_balance", "backtesting")
        if hasattr(self, 'risk_per_trade_spin'):
            apply_tooltip(self.risk_per_trade_spin, "risk_per_trade", "backtesting")
        if hasattr(self, 'max_positions_spin'):
            apply_tooltip(self.max_positions_spin, "max_positions", "backtesting")
        if hasattr(self, 'commission_spin'):
            apply_tooltip(self.commission_spin, "commission", "backtesting")
        if hasattr(self, 'slippage_spin'):
            apply_tooltip(self.slippage_spin, "slippage_pips", "backtesting")
        if hasattr(self, 'stop_loss_atr_spin'):
            apply_tooltip(self.stop_loss_atr_spin, "stop_loss_atr", "backtesting")
        if hasattr(self, 'take_profit_atr_spin'):
            apply_tooltip(self.take_profit_atr_spin, "take_profit_atr", "backtesting")
        if hasattr(self, 'trailing_stop_check'):
            apply_tooltip(self.trailing_stop_check, "trailing_stop", "backtesting")
        if hasattr(self, 'walk_forward_check'):
            apply_tooltip(self.walk_forward_check, "walk_forward", "backtesting")
        if hasattr(self, 'optimization_metric_combo'):
            apply_tooltip(self.optimization_metric_combo, "optimization_metric", "backtesting")
    
    def _poll_job_status(self):
        try:
            from ..backtest.db import BacktestDB
            job_id = int(getattr(self, "last_job_id", 0) or 0)
            if not job_id:
                return
            db = BacktestDB()
            counts = db.job_status_counts(job_id)
            ncfg = max(1, int(counts.get("n_configs", 0)))
            prog = float(min(1.0, (counts.get("n_results", 0) + counts.get("n_dropped", 0)) / ncfg)) if ncfg else 0.0
            try:
                self.lbl_status.setText(f"Backtesting: in corso ({int(prog*100)}%)")
            except Exception:
                pass
            # append new result rows as they become available
            try:
                rows = db.results_for_job(job_id) or []
                # order by composite_score desc for stable view
                def _rk(x: dict):
                    try:
                        return (float(x.get("composite_score", 0.0) or 0.0), float(x.get("adherence_mean", 0.0) or 0.0))
                    except Exception:
                        return (0.0, 0.0)
                rows = sorted(rows, key=_rk, reverse=True)
                for r in rows:
                    try:
                        cfg_id = int(r.get("config_id") or 0)
                    except Exception:
                        cfg_id = 0
                    if cfg_id <= 0:
                        continue
                    if cfg_id in self._row_by_config_id:
                        # update in-place
                        self._update_row_metrics(self._row_by_config_id[cfg_id], r)
                    else:
                        # create new row
                        p = r.get("payload_json") or {}
                        model = (p.get("model") or p.get("model_name") or "?")
                        ptype = p.get("ptype") or p.get("prediction_type") or "?"
                        tf = p.get("timeframe") or "?"
                        self.add_result_row(
                            model, ptype, tf,
                            float(r.get("adherence_mean") or 0.0),
                            float(r.get("p50") or 0.0),
                            float(r.get("win_rate_delta") or 0.0),
                            r.get("coverage_observed"), r.get("band_efficiency"), r.get("composite_score"),
                            cfg_id
                        )
            except Exception:
                pass
            if counts.get("n_results", 0) >= counts.get("n_configs", 0) and counts.get("n_configs", 0) > 0:
                if self._poll_timer is not None:
                    self._poll_timer.stop()
                try:
                    self.lbl_status.setText("Backtesting: completato")
                except Exception:
                    pass
        except Exception:
            pass

    def _browse_models(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Seleziona modelli", filter="Pickle/All (*.*)")
        if paths:
            self.ed_models.setText(";".join(paths))

    def _reset_indicator_row(self, indicator: str) -> None:
        defaults = DEFAULT_INDICATOR_TFS.get(indicator, [])
        for tf, cb in self.indicator_checks.get(indicator, {}).items():
            cb.setChecked(tf in defaults)

    def _collect_indicator_selection(self) -> Dict[str, List[str]]:
        selection: Dict[str, List[str]] = {}
        for ind, mapping in self.indicator_checks.items():
            chosen = [tf for tf, cb in mapping.items() if cb.isChecked()]
            if chosen:
                selection[ind] = chosen
        return selection

    def _collect_indicator_param_ranges(self) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
        out: Dict[str, Dict[str, Tuple[int, int, int]]] = {}
        for ind, params in self.indicator_param_ranges.items():
            inner: Dict[str, Tuple[int, int, int]] = {}
            for pkey, (sp_from, sp_to, sp_step) in params.items():
                a = int(sp_from.value()); b = int(sp_to.value());
                if a > b: a, b = b, a
                s = max(1, int(sp_step.value()))
                inner[pkey] = (a, b, s)
            if inner:
                out[ind] = inner
        return out

    def _collect_numeric_ranges(self) -> Dict[str, Tuple[int, int, int]]:
        ranges: Dict[str, Tuple[int, int, int]] = {}
        for param, (sp_from, sp_to, sp_step) in self.range_fields.items():
            start = int(sp_from.value())
            stop = int(sp_to.value())
            if start > stop:
                start, stop = stop, start
            step = max(1, int(sp_step.value()))
            ranges[param] = (start, stop, step)
        return ranges

    def _collect_boolean_choices(self) -> Dict[str, List[bool]]:
        choices: Dict[str, List[bool]] = {}
        for param, (cb_true, cb_false) in self.boolean_fields.items():
            vals: List[bool] = []
            if cb_true.isChecked():
                vals.append(True)
            if cb_false.isChecked():
                vals.append(False)
            choices[param] = vals
        return choices

    def _load_prediction_defaults(self) -> None:
        try:
            settings = PredictionSettingsDialog.get_settings_from_file()
        except Exception:
            settings = {}
        if not settings:
            return
        ind_settings = settings.get("indicator_tfs", {})
        for ind, mapping in self.indicator_checks.items():
            selected = ind_settings.get(ind.lower(), ind_settings.get(ind.upper(), []))
            selected = [str(tf) for tf in selected]
            for tf, cb in mapping.items():
                cb.setChecked(tf in selected)
        for param, (sp_from, sp_to, sp_step) in self.range_fields.items():
            value = settings.get(param)
            if value is None:
                value = settings.get(param.lower())
            if isinstance(value, (int, float)):
                v = int(value)
                sp_from.setValue(v)
                sp_to.setValue(v)
        for param, (cb_true, cb_false) in self.boolean_fields.items():
            value = settings.get(param)
            if isinstance(value, bool):
                cb_true.setChecked(value is True)
                cb_false.setChecked(value is False)
            else:
                cb_true.setChecked(True)
                cb_false.setChecked(False)

    def _persist(self):
        s = {
            "bt_types": {
                "basic": self.chk_basic.isChecked(),
                "adv": self.chk_adv.isChecked(),
                "rw": self.chk_rw.isChecked(),
            },
            "bt_tf": self.cmb_tf.currentText(),
            "bt_h": self.ed_h.text(),
            "bt_models": self.ed_models.text(),
            "bt_samples": [self.sp_s.value(), self.sp_e.value(), self.sp_p.value()],
            "bt_preset": self.cmb_preset.currentText(),
            "bt_cache": self.chk_cache.isChecked(),
            "bt_indicator_tfs": self._collect_indicator_selection(),
            "bt_indicator_param_ranges": {ind: {p: [a, b, s] for p, (a, b, s) in ((k, (sp_from.value(), sp_to.value(), max(1, sp_step.value()))) for k, (sp_from, sp_to, sp_step) in params.items())} for ind, params in self.indicator_param_ranges.items()},
            "bt_numeric_ranges": {k: [sp_from.value(), sp_to.value(), max(1, sp_step.value())] for k, (sp_from, sp_to, sp_step) in self.range_fields.items()},
            "bt_boolean_flags": {k: {"true": cb_true.isChecked(), "false": cb_false.isChecked()} for k, (cb_true, cb_false) in self.boolean_fields.items()},
            "bt_time_flags": {"use_hours": bool(self.cb_use_hours.isChecked()), "use_day": bool(self.cb_use_day.isChecked())},
        }
        set_setting("backtesting_tab", s)

    def _load_persisted(self):
        s = get_setting("backtesting_tab", {}) or {}
        try:
            t = s.get("bt_types", {})
            self.chk_basic.setChecked(bool(t.get("basic", True)))
            self.chk_adv.setChecked(bool(t.get("adv", False)))
            self.chk_rw.setChecked(bool(t.get("rw", False)))
            if s.get("bt_tf"): self.cmb_tf.setCurrentText(str(s["bt_tf"]))
            if s.get("bt_h"): self.ed_h.setText(str(s["bt_h"]))
            if s.get("bt_models"): self.ed_models.setText(str(s["bt_models"]))
            if s.get("bt_samples") and isinstance(s["bt_samples"], list) and len(s["bt_samples"])==3:
                self.sp_s.setValue(int(s["bt_samples"][0])); self.sp_e.setValue(int(s["bt_samples"][1])); self.sp_p.setValue(int(s["bt_samples"][2]))
            if s.get("bt_preset"): self.cmb_preset.setCurrentText(str(s["bt_preset"]))
            self.chk_cache.setChecked(bool(s.get("bt_cache", True)))
            ind_sel = s.get("bt_indicator_tfs", {}) or {}
            for ind, mapping in self.indicator_checks.items():
                chosen = ind_sel.get(ind) or ind_sel.get(ind.lower()) or ind_sel.get(ind.upper()) or []
                chosen = [str(tf) for tf in chosen]
                for tf, cb in mapping.items():
                    cb.setChecked(tf in chosen)
            # restore indicator param ranges
            ind_ranges = s.get("bt_indicator_param_ranges", {}) or {}
            for ind, params in self.indicator_param_ranges.items():
                saved = ind_ranges.get(ind, {}) or {}
                for p, (sp_from, sp_to, sp_step) in params.items():
                    trip = saved.get(p)
                    if isinstance(trip, (list, tuple)) and len(trip) == 3:
                        a, b, st = trip
                        sp_from.setValue(int(a)); sp_to.setValue(int(b)); sp_step.setValue(max(1, int(st)))
            ranges = s.get("bt_numeric_ranges", {}) or {}
            for param, values in ranges.items():
                sp = self.range_fields.get(param)
                if not sp:
                    continue
                sp_from, sp_to, sp_step = sp
                try:
                    start, stop, step = values
                except Exception:
                    continue
                sp_from.setValue(int(start))
                sp_to.setValue(int(stop))
                sp_step.setValue(max(1, int(step)))
            flags = s.get("bt_boolean_flags", {}) or {}
            for param, state in flags.items():
                cb_pair = self.boolean_fields.get(param)
                if not cb_pair:
                    continue
                cb_true, cb_false = cb_pair
                cb_true.setChecked(bool(state.get("true", False)))
                cb_false.setChecked(bool(state.get("false", False)))
            # restore candle flags
            try:
                tf = s.get("bt_time_flags", {}) or {}
                self.cb_use_hours.setChecked(bool(tf.get("use_hours", False)))
                self.cb_use_day.setChecked(bool(tf.get("use_day", False)))
            except Exception:
                pass
        except Exception:
            pass

    # Public: add a result row
    def add_result_row(self, model: str, ptype: str, tf: str, adh_mean: float, p50: float, win_rate: float, coverage: float | None = None, band_eff: float | None = None, score: float | None = None, config_id: int | None = None):
        r = self.tbl.rowCount(); self.tbl.insertRow(r)
        self.tbl.setItem(r, 0, QTableWidgetItem("" if config_id is None else str(int(config_id))))
        self.tbl.setItem(r, 1, QTableWidgetItem(str(model)))
        self.tbl.setItem(r, 2, QTableWidgetItem(str(ptype)))
        self.tbl.setItem(r, 3, QTableWidgetItem(str(tf)))
        self.tbl.setItem(r, 4, QTableWidgetItem(f"{adh_mean:.3f}"))
        self.tbl.setItem(r, 5, QTableWidgetItem(f"{p50:.3f}"))
        self.tbl.setItem(r, 6, QTableWidgetItem(f"{win_rate:.2f}"))
        self.tbl.setItem(r, 7, QTableWidgetItem("" if coverage is None else f"{coverage:.2f}"))
        self.tbl.setItem(r, 8, QTableWidgetItem("" if band_eff is None else f"{band_eff:.2f}"))
        self.tbl.setItem(r, 9, QTableWidgetItem("" if score is None else f"{score:.3f}"))
        try:
            if config_id is not None and int(config_id) > 0:
                self._row_by_config_id[int(config_id)] = r
                self._shown_config_ids.add(int(config_id))
        except Exception:
            pass

    def _update_row_metrics(self, row_idx: int, result_row: dict):
        try:
            # Update only numeric metrics columns 4..9
            adh_mean = float(result_row.get("adherence_mean") or 0.0)
            p50 = float(result_row.get("p50") or 0.0)
            win = float(result_row.get("win_rate_delta") or 0.0)
            cov = result_row.get("coverage_observed")
            be = result_row.get("band_efficiency")
            score = result_row.get("composite_score")
            self.tbl.item(row_idx, 4).setText(f"{adh_mean:.3f}")
            self.tbl.item(row_idx, 5).setText(f"{p50:.3f}")
            self.tbl.item(row_idx, 6).setText(f"{win:.2f}")
            self.tbl.item(row_idx, 7).setText("" if cov is None else f"{float(cov):.2f}")
            self.tbl.item(row_idx, 8).setText("" if be is None else f"{float(be):.2f}")
            self.tbl.item(row_idx, 9).setText("" if score is None else f"{float(score):.3f}")
        except Exception:
            pass

    def _on_start(self):
        self._persist()
        payload = {
            "symbol": getattr(getattr(self.parent(), "controller", None), "symbol", "EUR/USD") if hasattr(self.parent(), "controller") else "EUR/USD",
            "prediction_types": [t for t,b in [("Basic", self.chk_basic.isChecked()), ("Advanced", self.chk_adv.isChecked()), ("Baseline", self.chk_rw.isChecked())] if b],
            "models": [m for m in self.ed_models.text().split(";") if m.strip()],
            "timeframe": self.cmb_tf.currentText(),
            "horizons_raw": self.ed_h.text(),
            "samples_range": [self.sp_s.value(), self.sp_e.value(), self.sp_p.value()],
            "interval": {"type": "preset", "preset": self.cmb_preset.currentText(), "walkforward": {"train": "90d", "test": "7d", "step": "7d", "gap": "0d"}},
            "use_cache": self.chk_cache.isChecked(),
            "indicator_selection": self._collect_indicator_selection(),
            # merge indicator param ranges and generic ranges downstream
            "indicator_numeric_ranges": self._collect_indicator_param_ranges(),
            "forecast_numeric_ranges": {k: list(v) for k, v in self._collect_numeric_ranges().items()},
            "forecast_boolean_params": {k: v for k, v in self._collect_boolean_choices().items()},
            # time flag selection: if True, the backend will try both states (False/True)
            "time_flag_selection": {"use_hours": bool(self.cb_use_hours.isChecked()), "use_day": bool(self.cb_use_day.isChecked())},
        }
        self.startRequested.emit(payload)

    def _apply_selected(self, target: str):
        row = self.tbl.currentRow()
        if row < 0:
            self.lbl_status.setText("Seleziona una riga risultati.")
            return
        # Read config_id from column 0
        item = self.tbl.item(row, 0)
        if item is None or not item.text().strip():
            self.lbl_status.setText("Risultato senza config_id.")
            return
        try:
            cfg_id = int(item.text())
        except Exception:
            self.lbl_status.setText("config_id non valido.")
            return
        # DB-only apply: carica payload e scrive prediction_settings.json localmente
        try:
            from ..backtest.db import BacktestDB
            from .prediction_settings_dialog import PredictionSettingsDialog
            db = BacktestDB()
            rows = db.results_for_job(int(getattr(self, "last_job_id", 0)))
            row = next((r for r in rows if int(r.get("config_id")) == int(cfg_id)), None)
            if not row:
                self.lbl_status.setText("config non trovata nel job")
                return
            payload = row.get("payload_json") or {}
            mapped = {
                "horizons": [str(h) for h in (payload.get("horizons_sec") or [])],
                "indicator_tfs": payload.get("indicators") or {},
                "forecast_types": [target.lower()],
                "model_paths": [payload.get("model")] if payload.get("model") else [],
            }
            current = PredictionSettingsDialog.get_settings_from_file()
            current.update(mapped)
            from pathlib import Path
            import json as _json
            cfg_file = PredictionSettingsDialog.CONFIG_FILE if hasattr(PredictionSettingsDialog, "CONFIG_FILE") else None
            if cfg_file is None:
                cfg_file = Path(__file__).resolve().parents[3] / "configs" / "prediction_settings.json"
            cfg_file.parent.mkdir(parents=True, exist_ok=True)
            cfg_file.write_text(_json.dumps(current, indent=4), encoding="utf-8")
            self.lbl_status.setText(f"Applicato preset a {target}.")
        except Exception as e:
            self.lbl_status.setText(f"Apply-config exception: {e}")
        # After applying, try to load and display profiles
        try:
            self._load_and_plot_profiles(cfg_id)
        except Exception:
            pass

    def _load_and_plot_profiles(self, config_id: int):
        if self.plot is None:
            return
        try:
            import numpy as _np
            import pyqtgraph as pg
            from ..backtest.db import BacktestDB
            db = BacktestDB()
            r = db.result_for_config(int(config_id))
            if not r:
                return
            hp = r.get("horizon_profile_json") or {}
            # plot adherence mean by horizon label sorted
            if hp:
                labels = sorted(hp.keys(), key=lambda k: k)
                mean_vals = [_np.nan if hp[k] is None else float(hp[k].get("mean", 0.0)) for k in labels]
                x = list(range(len(labels)))
                self.plot.clear()
                try:
                    if hasattr(self, 'legend') and self.legend is not None:
                        self.legend.clear()
                except Exception:
                    pass
                self.plot.plot(x, mean_vals, pen=pg.mkPen('#1f77b4', width=2), name="adh_mean")
                # overlay error quantiles if available
                try:
                    q10 = [float(hp[k].get("err_q10", _np.nan)) for k in labels]
                    q50 = [float(hp[k].get("err_q50", _np.nan)) for k in labels]
                    q90 = [float(hp[k].get("err_q90", _np.nan)) for k in labels]
                    self.plot.plot(x, q10, pen=pg.mkPen('#ff7f0e', width=1, style=pg.QtCore.Qt.DashLine), name="err_q10")
                    self.plot.plot(x, q50, pen=pg.mkPen('#2ca02c', width=1, style=pg.QtCore.Qt.DashLine), name="err_q50")
                    self.plot.plot(x, q90, pen=pg.mkPen('#d62728', width=1, style=pg.QtCore.Qt.DashLine), name="err_q90")
                except Exception:
                    pass
                try:
                    ax = self.plot.getAxis('bottom')
                    ax.setTicks([[(i, labels[i]) for i in x]])
                except Exception:
                    pass
            # optionally overlay time-profile q50 as bar chart
            tp = r.get("time_profile_json") or {}
            if tp:
                buckets = sorted(tp.keys(), key=lambda k: k)
                q50 = [float(tp[b].get("q50", tp[b].get("mean", 0.0))) for b in buckets]
                x2 = list(range(len(buckets)))
                bars = pg.BarGraphItem(x=x2, height=q50, width=0.6, brush=pg.mkBrush(200, 120, 50, 120))
                self.plot.addItem(bars)
        except Exception:
            pass

    def _save_plot(self):
        if self.plot is None:
            return
        try:
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(self, "Salva Grafico", filter="PNG (*.png);;JPEG (*.jpg);;All Files (*)")
            if not path:
                return
            try:
                from pyqtgraph.exporters import ImageExporter
                exporter = ImageExporter(self.plot.plotItem)
                exporter.parameters()['width'] = 1200
                exporter.export(path)
            except Exception:
                # fallback to widget grab
                pix = self.plot.grab()
                pix.save(path)
            self.lbl_status.setText(f"Grafico salvato: {path}")
        except Exception as e:
            self.lbl_status.setText(f"Save plot error: {e}")

    def _resume_job(self):
        try:
            from ..backtest.db import BacktestDB
            db = BacktestDB()
            job_id = int(getattr(self, "last_job_id", 0))
            db.update_job_status(job_id, status="running")
            self.lbl_status.setText("Job in esecuzione")
        except Exception as e:
            self.lbl_status.setText(f"Resume exception: {e}")

    def _cancel_selected_config(self):
        row = self.tbl.currentRow()
        if row < 0:
            self.lbl_status.setText("Seleziona una riga risultati.")
            return
        item = self.tbl.item(row, 0)
        if item is None or not item.text().strip():
            self.lbl_status.setText("Risultato senza config_id.")
            return
        try:
            cfg_id = int(item.text())
        except Exception:
            self.lbl_status.setText("config_id non valido.")
            return
        try:
            from ..backtest.db import BacktestDB
            db = BacktestDB()
            db.cancel_config(int(cfg_id), reason="user")
            self.lbl_status.setText("Config cancellata")
            try:
                self._mark_row_cancelled(row)
            except Exception:
                pass
        except Exception as e:
            self.lbl_status.setText(f"Cancel config exception: {e}")

    def _mark_row_cancelled(self, row: int):
        try:
            from PySide6.QtGui import QColor
            cols = self.tbl.columnCount()
            for c in range(cols):
                it = self.tbl.item(row, c)
                if it is None:
                    continue
                # disable interaction
                it.setFlags(it.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsEnabled)
                # gray out
                it.setBackground(QColor(230, 230, 230))
                it.setForeground(QColor(120, 120, 120))
            # append '(cancelled)' to model cell
            model_it = self.tbl.item(row, 1)
            if model_it is not None and "(cancelled)" not in model_it.text().lower():
                model_it.setText(f"{model_it.text()} (cancelled)")
        except Exception:
            pass

    def _pause_job(self):
        try:
            from ..backtest.db import BacktestDB
            db = BacktestDB()
            job_id = int(getattr(self, "last_job_id", 0))
            db.update_job_status(job_id, status="paused")
            self.lbl_status.setText("Job in pausa")
        except Exception as e:
            self.lbl_status.setText(f"Pause exception: {e}")

    def _cancel_job(self):
        try:
            from ..backtest.db import BacktestDB
            db = BacktestDB()
            job_id = int(getattr(self, "last_job_id", 0))
            db.update_job_status(job_id, status="cancelled")
            self.lbl_status.setText("Job cancellato")
        except Exception as e:
            self.lbl_status.setText(f"Cancel exception: {e}")


