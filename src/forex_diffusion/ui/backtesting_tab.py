from __future__ import annotations

from typing import Any, Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QCheckBox,
    QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QGroupBox, QFileDialog
)

from ..utils.user_settings import get_setting, set_setting


class BacktestingTab(QWidget):
    startRequested = Signal(dict)
    pauseRequested = Signal()
    stopRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_persisted()

    def _build_ui(self):
        lay = QVBoxLayout(self)

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
        self.tbl.setHorizontalHeaderLabels(["ConfigId","Modello","Tipo","TF","Adh_mean","p50","Win@δ","Coverage","BandEff","Score"]) 
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

    def _browse_models(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Seleziona modelli", filter="Pickle/All (*.*)")
        if paths:
            self.ed_models.setText(";".join(paths))

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
            if s.get("bt_models")): self.ed_models.setText(str(s["bt_models"]))
            if s.get("bt_samples") and isinstance(s["bt_samples"], list) and len(s["bt_samples"])==3:
                self.sp_s.setValue(int(s["bt_samples"][0])); self.sp_e.setValue(int(s["bt_samples"][1])); self.sp_p.setValue(int(s["bt_samples"][2]))
            if s.get("bt_preset"): self.cmb_preset.setCurrentText(str(s["bt_preset"]))
            self.chk_cache.setChecked(bool(s.get("bt_cache", True)))
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
        # Emit a signal or directly call API here? The app wires startRequested only; use httpx here
        try:
            import httpx
            base = getattr(self.parent(), "engine_url", None) or "http://127.0.0.1:8000"
            job_id = getattr(self, "last_job_id", 0)
            with httpx.Client(timeout=30.0) as client:
                r = client.post(f"{base.rstrip('/')}/backtests/{int(job_id)}/apply-config", json={"config_id": cfg_id, "target": target})
                if r.status_code == 200:
                    self.lbl_status.setText(f"Applicato preset a {target}.")
                else:
                    self.lbl_status.setText(f"Apply-config error: {r.text}")
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
            import httpx
            import numpy as _np
            import pyqtgraph as pg
            base = getattr(self.parent(), "engine_url", None) or "http://127.0.0.1:8000"
            job_id = getattr(self, "last_job_id", 0)
            with httpx.Client(timeout=15.0) as client:
                r = client.get(f"{base.rstrip('/')}/backtests/{int(job_id)}/config/{int(config_id)}/profiles")
                if r.status_code != 200:
                    return
                data = r.json()
                hp = data.get("horizon_profile") or {}
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
                tp = data.get("time_profile") or {}
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
            import httpx
            base = getattr(self.parent(), "engine_url", None) or "http://127.0.0.1:8000"
            job_id = getattr(self, "last_job_id", 0)
            with httpx.Client(timeout=15.0) as client:
                r = client.post(f"{base.rstrip('/')}/backtests/{int(job_id)}/resume")
                self.lbl_status.setText("Job in esecuzione" if r.status_code == 200 else f"Resume error: {r.text}")
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
            import httpx
            base = getattr(self.parent(), "engine_url", None) or "http://127.0.0.1:8000"
            job_id = getattr(self, "last_job_id", 0)
            with httpx.Client(timeout=15.0) as client:
                r = client.post(f"{base.rstrip('/')}/backtests/{int(job_id)}/config/cancel", json={"config_id": cfg_id, "reason": "user"})
                if r.status_code == 200:
                    self.lbl_status.setText("Config cancellata")
                    try:
                        self._mark_row_cancelled(row)
                    except Exception:
                        pass
                else:
                    self.lbl_status.setText(f"Cancel config error: {r.text}")
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
            import httpx
            base = getattr(self.parent(), "engine_url", None) or "http://127.0.0.1:8000"
            job_id = getattr(self, "last_job_id", 0)
            with httpx.Client(timeout=15.0) as client:
                r = client.post(f"{base.rstrip('/')}/backtests/{int(job_id)}/pause")
                self.lbl_status.setText("Job in pausa" if r.status_code == 200 else f"Pause error: {r.text}")
        except Exception as e:
            self.lbl_status.setText(f"Pause exception: {e}")

    def _cancel_job(self):
        try:
            import httpx
            base = getattr(self.parent(), "engine_url", None) or "http://127.0.0.1:8000"
            job_id = getattr(self, "last_job_id", 0)
            with httpx.Client(timeout=15.0) as client:
                r = client.post(f"{base.rstrip('/')}/backtests/{int(job_id)}/cancel", json={"reason": "user"})
                self.lbl_status.setText("Job cancellato" if r.status_code == 200 else f"Cancel error: {r.text}")
        except Exception as e:
            self.lbl_status.setText(f"Cancel exception: {e}")


