# src/forex_diffusion/ui/training_tab.py
from __future__ import annotations

import sys
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
        # Root + scrollable page (to keep tab compact on small screens)
        from PySide6.QtWidgets import QScrollArea, QWidget
        self._root = QVBoxLayout(self)
        page = QWidget(self)
        self.layout = QVBoxLayout(page)
        self.controller = TrainingController(self)
        self.controller.signals.log.connect(self._append_log)
        self.controller.signals.progress.connect(self._on_progress)
        self.controller.signals.finished.connect(self._on_finished)

        # pending meta for current training run
        self._pending_meta: Optional[Dict] = None
        self._pending_out_dir: Optional[Path] = None

        # Top controls
        top = QHBoxLayout()
        lbl_sym = QLabel("Symbol:"); lbl_sym.setToolTip("Coppia valutaria da usare per il training.")
        top.addWidget(lbl_sym)
        self.symbol_combo = QComboBox(); self.symbol_combo.addItems(["EUR/USD","GBP/USD","AUX/USD","GBP/NZD","AUD/JPY","GBP/EUR","GBP/AUD"])
        self.symbol_combo.setToolTip("Seleziona il simbolo per cui addestrare il modello.")
        top.addWidget(self.symbol_combo)

        lbl_tf = QLabel("Base TF:"); lbl_tf.setToolTip("Timeframe base della serie su cui costruire le feature.")
        top.addWidget(lbl_tf)
        self.tf_combo = QComboBox(); self.tf_combo.addItems(["1m","5m","15m","30m","1h","4h","1d"])
        self.tf_combo.setCurrentText("1m")
        self.tf_combo.setToolTip("Timeframe delle barre usate per il training.")
        top.addWidget(self.tf_combo)

        lbl_days = QLabel("Days history:"); lbl_days.setToolTip("Numero di giorni storici da usare per l'addestramento.")
        top.addWidget(lbl_days)
        self.days_spin = QSpinBox(); self.days_spin.setRange(1, 3650); self.days_spin.setValue(7)
        self.days_spin.setToolTip("Maggiore è il valore, più dati verranno usati (training più lungo).")
        top.addWidget(self.days_spin)

        lbl_h = QLabel("Horizon (bars):"); lbl_h.setToolTip("Orizzonte in barre da prevedere durante il training.")
        top.addWidget(lbl_h)
        self.horizon_spin = QSpinBox(); self.horizon_spin.setRange(1, 500); self.horizon_spin.setValue(5)
        self.horizon_spin.setToolTip("Numero di passi futuri/target usati in fase di training.")
        top.addWidget(self.horizon_spin)

        lbl_m = QLabel("Model:"); lbl_m.setToolTip("Tipo di modello supervisato da addestrare.")
        top.addWidget(lbl_m)
        self.model_combo = QComboBox(); self.model_combo.addItems(["ridge","lasso","elasticnet","rf"])
        self.model_combo.setToolTip("Scegli l'algoritmo di base (regressione lineare regolarizzata o Random Forest).")
        top.addWidget(self.model_combo)

        lbl_e = QLabel("Encoder:"); lbl_e.setToolTip("Preprocessore/encoder opzionale per ridurre dimensionalità o apprendere rappresentazioni.")
        top.addWidget(lbl_e)
        self.encoder_combo = QComboBox(); self.encoder_combo.addItems(["none","pca","latents"])
        self.encoder_combo.setToolTip("PCA riduce la dimensione delle feature. 'latents' richiede un encoder già definito.")
        top.addWidget(self.encoder_combo)

        lbl_opt = QLabel("Optimization:"); lbl_opt.setToolTip("Ricerca automatica dell'ipermodello:\n- none: nessuna ottimizzazione\n- genetic-basic: ricerca genetica su R2\n- nsga2: ottimizzazione multi-obiettivo (R2/MAE)")
        top.addWidget(lbl_opt)
        self.opt_combo = QComboBox(); self.opt_combo.addItems(["none","genetic-basic","nsga2"])
        self.opt_combo.setToolTip("Seleziona se e come ottimizzare automaticamente gli iperparametri.")
        top.addWidget(self.opt_combo)

        lbl_gen = QLabel("Gen:"); lbl_gen.setToolTip("Numero di generazioni dell'algoritmo genetico.")
        top.addWidget(lbl_gen)
        self.gen_spin = QSpinBox(); self.gen_spin.setRange(1, 50); self.gen_spin.setValue(5); self.gen_spin.setToolTip("Quante iterazioni evolutive eseguire.")
        top.addWidget(self.gen_spin)
        lbl_pop = QLabel("Pop:"); lbl_pop.setToolTip("Dimensione della popolazione per la ricerca evolutiva.")
        top.addWidget(lbl_pop)
        self.pop_spin = QSpinBox(); self.pop_spin.setRange(2, 64); self.pop_spin.setValue(8); self.pop_spin.setToolTip("Quanti candidati valutare per generazione.")
        top.addWidget(self.pop_spin)

        self.layout.addLayout(top)

        # Indicators × Timeframes grid
        grid_box = QGroupBox("Indicatori per Timeframe (seleziona; click 'Default' per ripristinare)")
        grid_box.setToolTip("Seleziona gli indicatori da calcolare per ciascun timeframe durante il training.\nI pulsanti 'Default' ripristinano le selezioni consigliate per indicatore.")
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
        lbl_wu = QLabel("warmup"); lbl_wu.setToolTip("Barre di warmup per stabilizzare gli indicatori prima del training.")
        adv.addWidget(lbl_wu); self.warmup = QSpinBox(); self.warmup.setRange(0, 5000); self.warmup.setValue(16); self.warmup.setToolTip("Quante barre iniziali considerare come 'preriscaldamento'."); adv.addWidget(self.warmup)
        lbl_atr = QLabel("atr_n"); lbl_atr.setToolTip("Finestra dell'ATR (volatilità).")
        adv.addWidget(lbl_atr); self.atr_n = QSpinBox(); self.atr_n.setRange(1, 500); self.atr_n.setValue(14); self.atr_n.setToolTip("Numero di periodi per l'ATR."); adv.addWidget(self.atr_n)
        lbl_rsi = QLabel("rsi_n"); lbl_rsi.setToolTip("Finestra RSI (momento).")
        adv.addWidget(lbl_rsi); self.rsi_n = QSpinBox(); self.rsi_n.setRange(2, 500); self.rsi_n.setValue(14); self.rsi_n.setToolTip("Numero di periodi per l'RSI."); adv.addWidget(self.rsi_n)
        lbl_bb = QLabel("bb_n"); lbl_bb.setToolTip("Finestra per Bande di Bollinger.")
        adv.addWidget(lbl_bb); self.bb_n = QSpinBox(); self.bb_n.setRange(2, 500); self.bb_n.setValue(20); self.bb_n.setToolTip("Numero di barre per calcolare le bande di Bollinger."); adv.addWidget(self.bb_n)
        lbl_hu = QLabel("hurst_win"); lbl_hu.setToolTip("Window per l'esponente di Hurst.")
        adv.addWidget(lbl_hu); self.hurst_w = QSpinBox(); self.hurst_w.setRange(8, 4096); self.hurst_w.setValue(64); self.hurst_w.setToolTip("Lunghezza finestra per stimare 'H'."); adv.addWidget(self.hurst_w)
        lbl_rv = QLabel("rv_window"); lbl_rv.setToolTip("Finestra per stima di volatilità/standardizzazione.")
        adv.addWidget(lbl_rv); self.rv_w = QSpinBox(); self.rv_w.setRange(1, 10000); self.rv_w.setValue(60); self.rv_w.setToolTip("Barre usate per normalizzare/standardizzare le feature."); adv.addWidget(self.rv_w)
        self.layout.addLayout(adv)

        # Install scroll area into root
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        try:
            scroll.setWidget(page)
        except Exception:
            pass
        self._root.addWidget(scroll)

        # Output location (interpreted as base artifacts dir)
        out_h = QHBoxLayout()
        default_out_dir = None
        try:
            default_out_dir = getattr(getattr(self.cfg, "model", None), "artifacts_dir", None)
        except Exception:
            default_out_dir = None
        default_out_dir = default_out_dir or "./artifacts"   # il trainer salverà in <dir>/models
        self.out_dir = QLineEdit(str(Path(default_out_dir)))
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

            # Nome “umano” per log
            tfs_str = "-".join(sorted(set(sum(ind_tfs.values(), [])))) if ind_tfs else "none"
            name = f"{sym.replace('/','')}_{tf}_d{days}_h{horizon}_{model}_{encoder}_ind{len(ind_tfs)}_{tfs_str}"

            # Artifacts base dir: se l’utente ha messo .../models, passo la parent al trainer
            out_dir = Path(self.out_dir.text()).resolve()
            artifacts_dir = out_dir if out_dir.name.lower() != "models" else out_dir.parent

            # Trainer sklearn come modulo
            root = Path(__file__).resolve().parents[3]
            module = "src.forex_diffusion.training.train_sklearn"

            # Mappatura UI → CLI (RF non supportato nel trainer sklearn base: ricadiamo su ridge)
            algo = "ridge" if model == "rf" else model
            pca = "0" if encoder != "pca" else "16"  # esempio: 16 componenti se l'utente seleziona PCA

            args = [
                sys.executable, "-m", module,
                "--symbol", sym,
                "--timeframe", tf,
                "--horizon", str(horizon),
                "--algo", algo,
                "--pca", pca,
                "--artifacts_dir", str(artifacts_dir),
                "--warmup_bars", str(int(self.warmup.value())),
                "--val_frac", "0.2",
                "--alpha", "0.001",
                "--l1_ratio", "0.5",
            ]

            # Nota su ottimizzazione: non implementata nel trainer sklearn
            strategy = self.opt_combo.currentText()
            if strategy != "none":
                self._append_log(f"[warn] Optimization '{strategy}' non implementata nel trainer sklearn; eseguo una singola fit.")

            # Meta sidecar per l’ultimo file creato (<artifacts_dir>/models)
            try:
                from datetime import datetime, timezone
                meta = {
                    "symbol": sym,
                    "base_timeframe": tf,
                    "days_history": int(days),
                    "horizon_bars": int(horizon),
                    "model_type": model,
                    "encoder": encoder,
                    "indicator_tfs": ind_tfs,
                    "advanced_params": {
                        "warmup_bars": int(self.warmup.value()),
                        "atr_n": int(self.atr_n.value()),
                        "rsi_n": int(self.rsi_n.value()),
                        "bb_n": int(self.bb_n.value()),
                        "hurst_window": int(self.hurst_w.value()),
                        "rv_window": int(self.rv_w.value()),
                    },
                    "optimization": strategy,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "ui_run_name": name,
                }
                self._pending_meta = meta
                self._pending_out_dir = artifacts_dir / "models"
                self._append_log(f"[meta] prepared: {meta}")
            except Exception:
                self._pending_meta = None
                self._pending_out_dir = None

            # Avvio async
            self.progress.setRange(0, 100); self.progress.setValue(0)
            self.controller.start_training(args, cwd=str(root))
            self._append_log(f"[start] {' '.join(args)}")
        except Exception as e:
            logger.exception("Start training error: {}", e)
            QMessageBox.warning(self, "Training", str(e))

    def _append_log(self, line: str):
        try:
            self.log_view.append(line)
        except Exception:
            pass

    def _find_latest_model_file(self, out_dir: Path) -> Optional[Path]:
        """Find the most recently modified model file (.pt/.pth/.pkl/.pickle) in out_dir."""
        try:
            cand = []
            for ext in ("*.pt","*.pth","*.pkl","*.pickle"):
                cand += list(out_dir.glob(ext))
            if not cand:
                return None
            cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cand[0]
        except Exception:
            return None

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
            # Attach meta to latest model file in <artifacts_dir>/models (sidecar JSON)
            try:
                if self._pending_out_dir and self._pending_out_dir.exists() and isinstance(self._pending_meta, dict):
                    latest = self._find_latest_model_file(self._pending_out_dir)
                    if latest:
                        sidecar = latest.with_suffix(latest.suffix + ".meta.json")
                        sidecar.write_text(json.dumps(self._pending_meta, indent=2), encoding="utf-8")
                        self._append_log(f"[meta] saved sidecar: {sidecar}")
                    else:
                        self._append_log("[meta] no model file found to attach meta")
            except Exception as e:
                self._append_log(f"[meta] save failed: {e}")
            QMessageBox.information(self, "Training", "Training completato.")
        else:
            QMessageBox.warning(self, "Training", "Training fallito.")
