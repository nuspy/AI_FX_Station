# src/forex_diffusion/ui/training_tab.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QSpinBox,
    QLineEdit, QGroupBox, QGridLayout, QMessageBox, QFileDialog, QTextEdit, QProgressBar
)
from loguru import logger

from ..utils.config import get_config
from ..utils.user_settings import get_setting, set_setting
from .controllers import TrainingController

INDICATORS = ["ATR", "RSI", "Bollinger", "MACD", "Donchian", "Keltner", "Hurst"]
TIMEFRAMES = ["1m","5m","15m","30m","1h","4h","1d"]
# default selection per indicator (persisted across sessions)
DEFAULTS = {
    "ATR": ["1m","5m","15m","30m","1h"],
    "RSI": ["1m","5m","15m","30m","1h"],
    "Bollinger": ["1m","5m","15m","30m"],
    "MACD": ["5m","15m","30m","1h","4h","1d"],
    "Donchian": ["15m","30m","1h","4h","1d"],
    "Keltner": ["15m","30m","1h","4h","1d"],
    "Hurst": ["30m","1h","4h","1d"],
}

class TrainingTab(QWidget):
    """
    Training Tab: configure and launch model training.
    - Selettori symbol/timeframe/giorni/horizon
    - Griglia indicatori × timeframe con persistenza e bottoni Default per riga
    - Scelta modello/encoder, opzionale ricerca evolutiva semplificata
    - Avvio training asincrono con progress bar e log
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = get_config()
        self.layout = QVBoxLayout(self)
        self.controller = TrainingController(self)
        self.controller.signals.log.connect(self._append_log)
        self.controller.signals.progress.connect(self._on_progress)
        self.controller.signals.finished.connect(self._on_finished)

        # Top controls
        top = QHBoxLayout()
        top.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox(); self.symbol_combo.addItems(["AUX/USD","GBP/NZD","AUD/JPY","GBP/EUR","GBP/AUD"])
        top.addWidget(self.symbol_combo)

        top.addWidget(QLabel("Base TF:"))
        self.tf_combo = QComboBox(); self.tf_combo.addItems(["1m","5m","15m","30m","1h","4h","1d"])
        self.tf_combo.setCurrentText("1m")
        top.addWidget(self.tf_combo)

        top.addWidget(QLabel("Days history:"))
        self.days_spin = QSpinBox(); self.days_spin.setRange(1, 3650); self.days_spin.setValue(7)
        top.addWidget(self.days_spin)

        top.addWidget(QLabel("Horizon (bars):"))
        self.horizon_spin = QSpinBox(); self.horizon_spin.setRange(1, 500); self.horizon_spin.setValue(5)
        top.addWidget(self.horizon_spin)

        top.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox(); self.model_combo.addItems(["ridge","lasso","elasticnet","rf"])
        top.addWidget(self.model_combo)

        top.addWidget(QLabel("Encoder:"))
        self.encoder_combo = QComboBox(); self.encoder_combo.addItems(["none","pca","latents"])
        top.addWidget(self.encoder_combo)

        top.addWidget(QLabel("Optimization:"))
        self.opt_combo = QComboBox(); self.opt_combo.addItems(["none","genetic-basic","nsga2"])
        top.addWidget(self.opt_combo)

        top.addWidget(QLabel("Gen:")); self.gen_spin = QSpinBox(); self.gen_spin.setRange(1, 50); self.gen_spin.setValue(5); top.addWidget(self.gen_spin)
        top.addWidget(QLabel("Pop:")); self.pop_spin = QSpinBox(); self.pop_spin.setRange(2, 64); self.pop_spin.setValue(8); top.addWidget(self.pop_spin)

        self.layout.addLayout(top)

        # Indicators × Timeframes grid
        grid_box = QGroupBox("Indicatori per Timeframe (seleziona; click 'Default' per ripristinare)")
        grid = QGridLayout(grid_box)
        grid.addWidget(QLabel(""), 0, 0)
        for j, tf in enumerate(TIMEFRAMES, start=1):
            grid.addWidget(QLabel(tf), 0, j)
        # load previous or defaults
        saved = get_setting("training_indicator_tfs", {})
        self.chk: Dict[str, Dict[str, object]] = {}
        for i, ind in enumerate(INDICATORS, start=1):
            # row title + default button
            row_box = QHBoxLayout()
            lbl = QLabel(ind)
            btn = QPushButton("Default"); btn.setFixedWidth(64)
            def _make_reset(ind_name: str):
                return lambda: self._reset_row_to_default(ind_name)
            btn.clicked.connect(_make_reset(ind))
            row_box.addWidget(lbl); row_box.addWidget(btn); row_box.addStretch()
            row_widget = QWidget(); row_widget.setLayout(row_box)
            grid.addWidget(row_widget, i, 0)
            self.chk[ind] = {}
            selected = saved.get(ind, DEFAULTS.get(ind, []))
            for j, tf in enumerate(TIMEFRAMES, start=1):
                from PySide6.QtWidgets import QCheckBox
                cb = QCheckBox()
                cb.setChecked(tf in selected)
                self.chk[ind][tf] = cb
                grid.addWidget(cb, i, j)
        self.layout.addWidget(grid_box)

        # Advanced params
        adv = QHBoxLayout()
        adv.addWidget(QLabel("warmup")); self.warmup = QSpinBox(); self.warmup.setRange(0, 5000); self.warmup.setValue(16); adv.addWidget(self.warmup)
        adv.addWidget(QLabel("atr_n")); self.atr_n = QSpinBox(); self.atr_n.setRange(1, 500); self.atr_n.setValue(14); adv.addWidget(self.atr_n)
        adv.addWidget(QLabel("rsi_n")); self.rsi_n = QSpinBox(); self.rsi_n.setRange(2, 500); self.rsi_n.setValue(14); adv.addWidget(self.rsi_n)
        adv.addWidget(QLabel("bb_n")); self.bb_n = QSpinBox(); self.bb_n.setRange(2, 500); self.bb_n.setValue(20); adv.addWidget(self.bb_n)
        adv.addWidget(QLabel("hurst_win")); self.hurst_w = QSpinBox(); self.hurst_w.setRange(8, 4096); self.hurst_w.setValue(64); adv.addWidget(self.hurst_w)
        adv.addWidget(QLabel("rv_window")); self.rv_w = QSpinBox(); self.rv_w.setRange(1, 10000); self.rv_w.setValue(60); adv.addWidget(self.rv_w)
        self.layout.addLayout(adv)

        # Output location
        out_h = QHBoxLayout()
        self.out_dir = QLineEdit(str(Path(self.cfg.model.artifacts_dir if hasattr(self.cfg, "model") else "./artifacts/models")))
        self.browse_btn = QPushButton("Scegli Cartella...")
        self.browse_btn.clicked.connect(self._browse_out)
        out_h.addWidget(QLabel("Output dir:"))
        out_h.addWidget(self.out_dir); out_h.addWidget(self.browse_btn)
        self.layout.addLayout(out_h)

        # Log & progress
        lp = QHBoxLayout()
        self.progress = QProgressBar(); self.progress.setValue(0); self.progress.setTextVisible(True)
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True); self.log_view.setMinimumHeight(140)
        lp.addWidget(self.progress, 1); lp.addWidget(self.log_view, 3)
        self.layout.addLayout(lp)

        # Actions
        actions = QHBoxLayout()
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self._start_training)
        actions.addWidget(self.train_btn)
        self.layout.addLayout(actions)

    def _reset_row_to_default(self, ind: str):
        for tf, cb in self.chk[ind].items():
            cb.setChecked(tf in DEFAULTS.get(ind, []))
        self._persist_indicator_tfs()

    def _persist_indicator_tfs(self):
        m = {ind: [tf for tf, cb in self.chk[ind].items() if cb.isChecked()] for ind in INDICATORS}
        set_setting("training_indicator_tfs", m)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Scegli cartella output", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _collect_indicator_tfs(self) -> Dict[str, List[str]]:
        m: Dict[str, List[str]] = {}
        for ind in INDICATORS:
            tfs = [tf for tf, cb in self.chk[ind].items() if cb.isChecked()]
            if tfs:
                m[ind.lower()] = tfs
        return m

    def _start_training(self):
        try:
            sym = self.symbol_combo.currentText()
            tf = self.tf_combo.currentText()
            days = int(self.days_spin.value())
            horizon = int(self.horizon_spin.value())
            model = self.model_combo.currentText()
            encoder = self.encoder_combo.currentText()
            ind_tfs = self._collect_indicator_tfs()
            self._persist_indicator_tfs()

            # Build filename with parameters
            tfs_str = "-".join(sorted(set(sum(ind_tfs.values(), [])))) if ind_tfs else "none"
            name = f"{sym.replace('/','')}_{tf}_d{days}_h{horizon}_{model}_{encoder}_ind{len(ind_tfs)}_{tfs_str}"
            out_dir = Path(self.out_dir.text())
            out_dir.mkdir(parents=True, exist_ok=True)
            # weighted_forecast.py salva il modello in artifacts/models; la UI mostra log e progresso

            # Build CLI args for weighted_forecast.py
            root = Path(__file__).resolve().parents[3]
            script = root / "tests" / "manual_tests" / "weighted_forecast.py"
            args = [
                "python", str(script),
                "--symbol", sym, "--timeframe", tf,
                "--days", str(days),
                "--horizon", str(horizon),
                "--warmup_bars", str(int(self.warmup.value())),
                "--atr_n", str(int(self.atr_n.value())),
                "--rsi_n", str(int(self.rsi_n.value())),
                "--bb_n", str(int(self.bb_n.value())),
                "--hurst_window", str(int(self.hurst_w.value())),
                "--rv_window", str(int(self.rv_w.value())),
                "--model", model,
                "--encoder", encoder,
                "--forecast_method", "supervised",
            ]
            # Avvio async: single training o GA
            strategy = self.opt_combo.currentText()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            if strategy == "none":
                self.controller.start_training(args, cwd=str(root))
                self._append_log(f"[start] { ' '.join(args) }")
            else:
                self._append_log(f"[GA start] strategy={strategy} gens={int(self.gen_spin.value())} pop={int(self.pop_spin.value())}")
                self.controller.start_training_ga(args, cwd=str(root), strategy=strategy, generations=int(self.gen_spin.value()), pop_size=int(self.pop_spin.value()))
        except Exception as e:
            logger.exception("Start training error: {}", e)
            QMessageBox.warning(self, "Training", str(e))

    def _append_log(self, line: str):
        try:
            self.log_view.append(line)
        except Exception:
            pass

    def _on_progress(self, value: int):
        if value < 0:
            self.progress.setRange(0, 0)  # indeterminate
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(value)

    def _on_finished(self, ok: bool):
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if ok else 0)
        self._append_log("[done] ok" if ok else "[done] failed")
        if ok:
            QMessageBox.information(self, "Training", "Training completato.")
        else:
            QMessageBox.warning(self, "Training", "Training fallito.")

    def _populate_empty(self):
        # Adjust columns to mirror market_data_candles schema (id,symbol,timeframe,ts_utc,open,high,low,close,volume,resampled)
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(["id","symbol","timeframe","ts_utc","open","high","low","close","volume","resampled"])
        self.table.setRowCount(0)

    def on_refresh(self):
        try:
            self.refresh(limit=200)
        except Exception as e:
            logger.exception("HistoryTab refresh failed: {}", e)
            QMessageBox.warning(self, "Refresh failed", str(e))

    def _row_to_dict(self, r):
        """
        Normalize a SQLAlchemy Row or tuple into a dict with canonical keys.
        Supports Row._mapping if present, otherwise falls back to tuple positions.
        """
        try:
            # SQLAlchemy Row supports _mapping in modern versions
            mapping = getattr(r, "_mapping", None)
            if mapping is not None:
                return dict(mapping)
        except Exception:
            pass
        # fallback: tuple -> map by known column order
        try:
            # Expected tuple layout: id,symbol,timeframe,ts_utc,open,high,low,close,volume,resampled
            rec = {
                "id": r[0],
                "symbol": r[1],
                "timeframe": r[2],
                "ts_utc": r[3],
                "open": r[4],
                "high": r[5],
                "low": r[6] if len(r) > 6 else None,
                "close": r[7] if len(r) > 7 else None,
                "volume": r[8] if len(r) > 8 else None,
                "resampled": r[9] if len(r) > 9 else False,
            }
            return rec
        except Exception:
            # last resort: enumerate tuple
            try:
                return {str(i): v for i, v in enumerate(r)}
            except Exception:
                return {}

    def on_backfill(self):
        try:
            sym = self.symbol_combo.currentText()
            tf = self.tf_combo.currentText()
            # run backfill synchronously (quick check) - may be long
            res = {}
            years = int(self.years_combo.currentText()) if self.years_combo else None
            months = int(self.months_combo.currentText()) if self.months_combo else None

            if self.market_service is not None:
                self._log(f"Using provider '{self.market_service.provider_name()}' for backfill")
                res = self.market_service.backfill_symbol_timeframe(sym, tf, force_full=True, months=months, years=years)
            else:
                res = {}
            self._log(f"Backfill result: {json.dumps(res)}")
            # Log DB engine URL (file) to ensure same DB is used across sessions
            try:
                eng = getattr(self.db_service, "engine", None)
                if eng is not None:
                    try:
                        # SQLAlchemy engines expose url attribute; for SQLite it shows file path
                        url = getattr(eng, "url", None)
                        self._log(f"Using DB engine URL: {url}")
                        logger.info("HistoryTab: DB engine URL after backfill: {}", url)
                    except Exception:
                        self._log("Using DB engine: <unknown>")
                else:
                    self._log("DB service engine not available for verification")
            except Exception:
                pass

            # VERIFICA POST-UPSET: conta righe nel DB per symbol/timeframe e logga
            try:
                from sqlalchemy import text as _text
                with self.db_service.engine.connect() as _conn:
                    cnt_q = _text("SELECT COUNT(1) FROM market_data_candles WHERE symbol = :s AND timeframe = :tf")
                    try:
                        cnt = int(_conn.execute(cnt_q, {"s": sym, "tf": tf}).scalar() or 0)
                        self._log(f"Post-backfill DB rows for {sym} {tf}: {cnt}")
                        if cnt == 0:
                            # warning both in UI log and logger
                            self._log(f"WARNING: No rows found in market_data_candles for {sym} {tf} after backfill")
                            logger.warning("HistoryTab post-backfill verification: 0 rows for %s %s", sym, tf)
                    except Exception as _e:
                        logger.exception("Post-backfill verification query failed: {}", _e)
                        self._log(f"Post-backfill verification failed: {_e}")
            except Exception as e:
                # best-effort: if verification cannot be performed, log debug
                try:
                    logger.debug("Backfill post-upsert verification skipped: {}", e)
                except Exception:
                    pass

            QMessageBox.information(self, "Backfill", "Backfill request completed (see log).")
            # refresh table
            self.refresh(limit=200)
            # update chart if connected
            if self.chart_tab is not None:
                try:
                    # load data from DB into DataFrame and plot
                    meta = MetaData()
                    meta.reflect(bind=self.db_service.engine, only=["market_data_candles"])
                    tbl = meta.tables.get("market_data_candles")
                    if tbl is not None:
                        with self.db_service.engine.connect() as conn:
                            # use same variants method to ensure GUI sees same rows regardless of stored symbol format
                            variants = self._symbol_variants(sym)
                            stmt = select(tbl).where(tbl.c.symbol.in_(variants)).where(tbl.c.timeframe == tf).order_by(tbl.c.ts_utc.asc())
                            rows = conn.execute(stmt).fetchall()
                            import pandas as pd
                            recs = [self._row_to_dict(r) for r in rows] if rows else []
                            if recs:
                                df = pd.DataFrame(recs)
                                # let chart_tab know symbol/timeframe and update via public redraw
                                try:
                                    self.chart_tab.set_symbol_timeframe(self.db_service, sym, tf)
                                except Exception:
                                    pass
                                try:
                                    # set internal buffer then request redraw (public wrapper)
                                    self.chart_tab._last_df = df
                                    self.chart_tab.redraw()
                                except Exception:
                                    # best-effort: try direct update_plot if available
                                    try:
                                        getattr(self.chart_tab, "update_plot", lambda *_: None)(df, timeframe=tf)
                                    except Exception:
                                        pass
                except Exception as e:
                    logger.exception("Failed to update chart after backfill: {}", e)
        except Exception as e:
            logger.exception("HistoryTab backfill failed: {}", e)
            QMessageBox.warning(self, "Backfill failed", str(e))

    def _log(self, msg: str):
        logger.info(msg)

    def refresh(self, limit: int = 200):
        # log provider for diagnostics
        try:
            pname = self.market_service.provider_name() if self.market_service is not None else "unknown"
            self._log(f"Refreshing history (provider={pname}) for {self.symbol_combo.currentText()} {self.tf_combo.currentText()}")
        except Exception:
            pass

        meta = MetaData()
        meta.reflect(bind=self.db_service.engine, only=["market_data_candles"])
        tbl = meta.tables.get("market_data_candles")
        if tbl is None:
            self._populate_empty()
            return
        with self.db_service.engine.connect() as conn:
            # use symbol variants so stored formats like 'EURUSD' and 'EUR/USD' both match
            sym = self.symbol_combo.currentText()
            variants = self._symbol_variants(sym)
            stmt = select(tbl).where(tbl.c.symbol.in_(variants)).where(tbl.c.timeframe == self.tf_combo.currentText()).order_by(tbl.c.ts_utc.desc()).limit(limit)
            rows = conn.execute(stmt).fetchall()
            if not rows:
                self._populate_empty()
                return
            # show table rows
            self.table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                recd = self._row_to_dict(r)
                self.table.setItem(i, 0, QTableWidgetItem(str(recd.get("id", i))))
                self.table.setItem(i, 1, QTableWidgetItem(str(recd.get("symbol", ""))))
                self.table.setItem(i, 2, QTableWidgetItem(str(recd.get("timeframe", ""))))
                # convert ts to local string for table too
                try:
                    import datetime
                    ts_local = datetime.datetime.fromtimestamp(int(recd.get("ts_utc", 0)) / 1000, tz=datetime.timezone.utc).astimezone()
                    ts_str = ts_local.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    ts_str = str(int(recd.get("ts_utc", 0)))
                self.table.setItem(i, 3, QTableWidgetItem(ts_str))
                self.table.setItem(i, 4, QTableWidgetItem(str(recd.get("open", ""))))
                self.table.setItem(i, 5, QTableWidgetItem(str(recd.get("high", ""))))
                self.table.setItem(i, 6, QTableWidgetItem(str(recd.get("low", ""))))
                self.table.setItem(i, 7, QTableWidgetItem(str(recd.get("close", ""))))
                self.table.setItem(i, 8, QTableWidgetItem(str(recd.get("volume", ""))))
                self.table.setItem(i, 9, QTableWidgetItem(str(recd.get("resampled", ""))))
            self.table.resizeColumnsToContents()

        # if chart connected, update chart with fetched rows (ascending order)
        if getattr(self, "chart_tab", None) is not None and rows:
            import pandas as pd
            # Normalize rows to dicts safely (support Row._mapping or tuple fallback)
            recs = [self._row_to_dict(r) for r in rows[::-1]]  # reverse to ascending
            if recs:
                try:
                    df = pd.DataFrame(recs)
                except Exception:
                    # fallback: try to coerce each record to dict again
                    df = pd.DataFrame([dict(x) if hasattr(x, "items") else x for x in recs])
                try:
                    self.chart_tab.set_symbol_timeframe(self.db_service, self.symbol_combo.currentText(), self.tf_combo.currentText())
                except Exception:
                    pass
                try:
                    # set internal buffer then request redraw (public wrapper)
                    self.chart_tab._last_df = df
                    self.chart_tab.redraw()
                except Exception:
                    # fallback to update_plot if available
                    try:
                        getattr(self.chart_tab, "update_plot", lambda *_: None)(df, timeframe=self.tf_combo.currentText())
                    except Exception:
                        pass
