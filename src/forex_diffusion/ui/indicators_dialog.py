# src/forex_diffusion/ui/indicators_dialog.py
from __future__ import annotations

from typing import Dict, Any, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QCheckBox, QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QWidget,
    QComboBox, QLineEdit, QMessageBox, QColorDialog
)
from PySide6.QtGui import QColor

# Persistenza (uguale a quella usata altrove nella UI)
try:
    from ..utils.user_settings import get_setting, set_setting
except Exception:  # fallback no-op se non disponibile
    def get_setting(key: str, default=None):
        return default
    def set_setting(key: str, value):
        pass


# ------------------------ Defaults & Helpers ------------------------ #
_DEFAULTS: Dict[str, Any] = {
    # base
    "use_atr": True, "atr_n": 14,
    "use_rsi": True, "rsi_n": 14,
    "use_bollinger": True, "bb_n": 20, "bb_k": 2.0,
    "use_hurst": True, "hurst_window": 64,
    # advanced
    "use_don": False, "don_n": 20,
    "use_keltner": False, "keltner_k": 1.5,
    "use_macd": False, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "use_sma": False, "sma_n": 20,
    "use_ema": False, "ema_fast": 12, "ema_slow": 26,
    # globali
    "warmup_bars": 16, "rv_window": 60,
}

# Colori di default (NON vengono ripristinati dal bottone "Default")
_DEFAULT_COLORS: Dict[str, str] = {
    "color_atr": "#d62728",       # rosso
    "color_rsi": "#1f77b4",       # blu
    "color_bollinger": "#2ca02c", # verde
    "color_hurst": "#9467bd",     # viola
    "color_don": "#8c564b",       # marrone
    "color_keltner": "#17becf",   # ciano
    "color_macd": "#ff7f0e",      # arancio
    "color_sma": "#7f7f7f",       # grigio
    "color_ema": "#bcbd22",       # oliva
}


def _qcolor_from_hex(hex_str: str) -> QColor:
    try:
        c = QColor(hex_str)
        if not c.isValid():
            raise ValueError
        return c
    except Exception:
        return QColor("#808080")


def _hex_from_qcolor(color: QColor) -> str:
    return color.name()  # formato #RRGGBB


class IndicatorsDialog(QDialog):
    """
    Dialog completo per configurare indicatori (base + avanzati) **con colori** e
    gestione **sessioni** multiple (salvataggio/caricamento profili).

    API:
      - `IndicatorsDialog.edit(parent, initial)` → dict | None
      - `get_settings()` ritorna tutte le chiavi usate dal pipeline/UI

    Memorizzazione:
      - settings per sessione in `user_settings` sotto la chiave `indicators.sessions`
      - sessione di default in `indicators.default_session`
    """

    def __init__(self, parent: Optional[QWidget] = None, initial: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Indicatori tecnici")
        self._initial = dict(initial or {})

        # ---- Header sessioni ----
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Sessione:"))
        self.session_combo = QComboBox()
        self.session_name_edit = QLineEdit()
        self.session_name_edit.setPlaceholderText("nuovo nome sessione…")
        self.btn_session_new = QPushButton("Nuova")
        self.btn_session_save = QPushButton("Salva")
        self.btn_session_del = QPushButton("Elimina")
        self.btn_defaults = QPushButton("Default parametri")  # non tocca i colori
        top_row.addWidget(self.session_combo)
        top_row.addWidget(self.session_name_edit)
        top_row.addWidget(self.btn_session_new)
        top_row.addWidget(self.btn_session_save)
        top_row.addWidget(self.btn_session_del)
        top_row.addStretch(1)
        top_row.addWidget(self.btn_defaults)

        # ------------------------- BASE ------------------------- #
        base_box = QGroupBox("Indicatori di base")
        base_form = QFormLayout(base_box)

        # ATR
        atr_row = QHBoxLayout()
        self.cb_atr = QCheckBox("Abilita ATR")
        self.sp_atr_n = QSpinBox(); self.sp_atr_n.setRange(2, 500)
        self.btn_color_atr = QPushButton("Colore")
        self._apply_button_color(self.btn_color_atr, _DEFAULT_COLORS["color_atr"])
        atr_row.addWidget(self.cb_atr)
        atr_row.addWidget(QLabel("n")); atr_row.addWidget(self.sp_atr_n)
        atr_row.addWidget(self.btn_color_atr)
        w_atr = QWidget(); w_atr.setLayout(atr_row)
        base_form.addRow("ATR", w_atr)

        # RSI
        rsi_row = QHBoxLayout()
        self.cb_rsi = QCheckBox("Abilita RSI")
        self.sp_rsi_n = QSpinBox(); self.sp_rsi_n.setRange(2, 500)
        self.btn_color_rsi = QPushButton("Colore")
        self._apply_button_color(self.btn_color_rsi, _DEFAULT_COLORS["color_rsi"])
        rsi_row.addWidget(self.cb_rsi)
        rsi_row.addWidget(QLabel("n")); rsi_row.addWidget(self.sp_rsi_n)
        rsi_row.addWidget(self.btn_color_rsi)
        w_rsi = QWidget(); w_rsi.setLayout(rsi_row)
        base_form.addRow("RSI", w_rsi)

        # Bollinger
        bb_row = QHBoxLayout()
        self.cb_bb = QCheckBox("Abilita Bande di Bollinger")
        self.sp_bb_n = QSpinBox(); self.sp_bb_n.setRange(2, 1000)
        self.d_bb_k = QDoubleSpinBox(); self.d_bb_k.setRange(0.1, 10.0); self.d_bb_k.setSingleStep(0.1)
        self.btn_color_bb = QPushButton("Colore")
        self._apply_button_color(self.btn_color_bb, _DEFAULT_COLORS["color_bollinger"])
        bb_row.addWidget(self.cb_bb)
        bb_row.addWidget(QLabel("n")); bb_row.addWidget(self.sp_bb_n)
        bb_row.addWidget(QLabel("k")); bb_row.addWidget(self.d_bb_k)
        bb_row.addWidget(self.btn_color_bb)
        w_bb = QWidget(); w_bb.setLayout(bb_row)
        base_form.addRow("Bollinger", w_bb)

        # Hurst
        hu_row = QHBoxLayout()
        self.cb_hurst = QCheckBox("Abilita Hurst")
        self.sp_hurst = QSpinBox(); self.sp_hurst.setRange(4, 4096)
        self.btn_color_hurst = QPushButton("Colore")
        self._apply_button_color(self.btn_color_hurst, _DEFAULT_COLORS["color_hurst"])
        hu_row.addWidget(self.cb_hurst)
        hu_row.addWidget(QLabel("window")); hu_row.addWidget(self.sp_hurst)
        hu_row.addWidget(self.btn_color_hurst)
        w_hu = QWidget(); w_hu.setLayout(hu_row)
        base_form.addRow("Hurst", w_hu)

        # ----------------------- AVANZATI ----------------------- #
        adv_box = QGroupBox("Indicatori avanzati")
        adv_form = QFormLayout(adv_box)

        # Donchian
        don_row = QHBoxLayout()
        self.cb_don = QCheckBox("Abilita Donchian")
        self.sp_don = QSpinBox(); self.sp_don.setRange(2, 1000)
        self.btn_color_don = QPushButton("Colore")
        self._apply_button_color(self.btn_color_don, _DEFAULT_COLORS["color_don"])
        don_row.addWidget(self.cb_don)
        don_row.addWidget(QLabel("n")); don_row.addWidget(self.sp_don)
        don_row.addWidget(self.btn_color_don)
        w_don = QWidget(); w_don.setLayout(don_row)
        adv_form.addRow("Donchian", w_don)

        # Keltner
        kel_row = QHBoxLayout()
        self.cb_kelt = QCheckBox("Abilita Keltner")
        self.d_kelt_k = QDoubleSpinBox(); self.d_kelt_k.setRange(0.1, 10.0); self.d_kelt_k.setSingleStep(0.1)
        self.btn_color_kelt = QPushButton("Colore")
        self._apply_button_color(self.btn_color_kelt, _DEFAULT_COLORS["color_keltner"])
        kel_row.addWidget(self.cb_kelt)
        kel_row.addWidget(QLabel("k")); kel_row.addWidget(self.d_kelt_k)
        kel_row.addWidget(self.btn_color_kelt)
        w_kel = QWidget(); w_kel.setLayout(kel_row)
        adv_form.addRow("Keltner", w_kel)

        # MACD
        macd_row = QHBoxLayout()
        self.cb_macd = QCheckBox("Abilita MACD")
        self.sp_macd_fast = QSpinBox(); self.sp_macd_fast.setRange(1, 500)
        self.sp_macd_slow = QSpinBox(); self.sp_macd_slow.setRange(2, 1000)
        self.sp_macd_signal = QSpinBox(); self.sp_macd_signal.setRange(1, 500)
        self.btn_color_macd = QPushButton("Colore")
        self._apply_button_color(self.btn_color_macd, _DEFAULT_COLORS["color_macd"])
        macd_row.addWidget(self.cb_macd)
        macd_row.addWidget(QLabel("fast")); macd_row.addWidget(self.sp_macd_fast)
        macd_row.addWidget(QLabel("slow")); macd_row.addWidget(self.sp_macd_slow)
        macd_row.addWidget(QLabel("signal")); macd_row.addWidget(self.sp_macd_signal)
        macd_row.addWidget(self.btn_color_macd)
        w_macd = QWidget(); w_macd.setLayout(macd_row)
        adv_form.addRow("MACD", w_macd)

        # SMA
        sma_row = QHBoxLayout()
        self.cb_sma = QCheckBox("Abilita SMA")
        self.sp_sma = QSpinBox(); self.sp_sma.setRange(1, 500)
        self.btn_color_sma = QPushButton("Colore")
        self._apply_button_color(self.btn_color_sma, _DEFAULT_COLORS["color_sma"])
        sma_row.addWidget(self.cb_sma)
        sma_row.addWidget(QLabel("n")); sma_row.addWidget(self.sp_sma)
        sma_row.addWidget(self.btn_color_sma)
        w_sma = QWidget(); w_sma.setLayout(sma_row)
        adv_form.addRow("SMA", w_sma)

        # EMA
        ema_row = QHBoxLayout()
        self.cb_ema = QCheckBox("Abilita EMA (fast/slow)")
        self.sp_ema_fast = QSpinBox(); self.sp_ema_fast.setRange(1, 1000)
        self.sp_ema_slow = QSpinBox(); self.sp_ema_slow.setRange(1, 1000)
        self.btn_color_ema = QPushButton("Colore")
        self._apply_button_color(self.btn_color_ema, _DEFAULT_COLORS["color_ema"])
        ema_row.addWidget(self.cb_ema)
        ema_row.addWidget(QLabel("fast")); ema_row.addWidget(self.sp_ema_fast)
        ema_row.addWidget(QLabel("slow")); ema_row.addWidget(self.sp_ema_slow)
        ema_row.addWidget(self.btn_color_ema)
        w_ema = QWidget(); w_ema.setLayout(ema_row)
        adv_form.addRow("EMA", w_ema)

        # ------------------- PARAMETRI GLOBALI ------------------ #
        glob_box = QGroupBox("Parametri globali")
        glob_form = QFormLayout(glob_box)
        self.sp_warmup = QSpinBox(); self.sp_warmup.setRange(0, 5000)
        self.sp_rv_win = QSpinBox(); self.sp_rv_win.setRange(1, 10000)
        glob_form.addRow("Warmup bars", self.sp_warmup)
        glob_form.addRow("Std/rv window", self.sp_rv_win)

        # -------------- Pulsanti azione -------------- #
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Annulla")
        ok_btn.clicked.connect(self.accept); cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)

        # ---- Layout root ----
        root = QVBoxLayout(self)
        root.addLayout(top_row)
        root.addWidget(base_box)
        root.addWidget(adv_box)
        root.addWidget(glob_box)
        root.addLayout(btn_row)

        # ---- Wiring ----
        # colori
        self.btn_color_atr.clicked.connect(lambda: self._pick_color(self.btn_color_atr))
        self.btn_color_rsi.clicked.connect(lambda: self._pick_color(self.btn_color_rsi))
        self.btn_color_bb.clicked.connect(lambda: self._pick_color(self.btn_color_bb))
        self.btn_color_hurst.clicked.connect(lambda: self._pick_color(self.btn_color_hurst))
        self.btn_color_don.clicked.connect(lambda: self._pick_color(self.btn_color_don))
        self.btn_color_kelt.clicked.connect(lambda: self._pick_color(self.btn_color_kelt))
        self.btn_color_macd.clicked.connect(lambda: self._pick_color(self.btn_color_macd))
        self.btn_color_sma.clicked.connect(lambda: self._pick_color(self.btn_color_sma))
        self.btn_color_ema.clicked.connect(lambda: self._pick_color(self.btn_color_ema))

        # sessioni
        self.btn_session_new.clicked.connect(self._on_session_new)
        self.btn_session_save.clicked.connect(self._on_session_save)
        self.btn_session_del.clicked.connect(self._on_session_delete)
        self.session_combo.currentTextChanged.connect(self._on_session_selected)
        self.btn_defaults.clicked.connect(self._apply_default_params_only)

        # ---- Init ----
        self._load_sessions_into_combo()
        self._apply_initial_or_default()
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()

    # --------------------------- colori --------------------------- #
    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets"""
        from ..i18n.widget_helper import apply_tooltip
        
        if hasattr(self, 'save_preset_btn'):
            apply_tooltip(self.save_preset_btn, "save_indicator_preset", "indicators_config")
        if hasattr(self, 'load_preset_btn'):
            apply_tooltip(self.load_preset_btn, "load_indicator_preset", "indicators_config")
        if hasattr(self, 'multi_timeframe_check'):
            apply_tooltip(self.multi_timeframe_check, "multi_timeframe_indicators", "indicators_config")
        if hasattr(self, 'indicator_alerts_check'):
            apply_tooltip(self.indicator_alerts_check, "indicator_alerts", "indicators_config")
    

    def _apply_button_color(self, btn: QPushButton, hex_color: str):
        btn.setProperty("hex", hex_color)
        btn.setStyleSheet(f"QPushButton {{ background-color: {hex_color}; color: black; }}")

    def _pick_color(self, btn: QPushButton):
        current = btn.property("hex") or "#808080"
        qcol = QColorDialog.getColor(_qcolor_from_hex(current), self, "Scegli colore")
        if qcol.isValid():
            hexc = _hex_from_qcolor(qcol)
            self._apply_button_color(btn, hexc)

    # --------------------------- sessioni ------------------------- #
    def _load_sessions_into_combo(self):
        sessions: Dict[str, dict] = get_setting("indicators.sessions", {}) or {}
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        names = sorted(sessions.keys())
        if not names:
            names = ["default"]
        self.session_combo.addItems(names)
        # selezione predefinita
        default_name = get_setting("indicators.default_session", "default")
        if default_name in names:
            self.session_combo.setCurrentText(default_name)
        self.session_combo.blockSignals(False)

    def _on_session_selected(self, name: str):
        sessions: Dict[str, dict] = get_setting("indicators.sessions", {}) or {}
        data = sessions.get(name)
        if data:
            self._apply_from_dict(data)
            set_setting("indicators.default_session", name)

    def _on_session_new(self):
        name = (self.session_name_edit.text() or "").strip()
        if not name:
            QMessageBox.information(self, "Sessione", "Inserisci un nome per la nuova sessione.")
            return
        sessions: Dict[str, dict] = get_setting("indicators.sessions", {}) or {}
        if name in sessions:
            QMessageBox.warning(self, "Sessione", "Esiste già una sessione con questo nome.")
            return
        sessions[name] = self.get_settings()
        set_setting("indicators.sessions", sessions)
        set_setting("indicators.default_session", name)
        self._load_sessions_into_combo()
        self.session_combo.setCurrentText(name)

    def _on_session_save(self):
        name = self.session_combo.currentText() or (self.session_name_edit.text().strip() or "default")
        sessions: Dict[str, dict] = get_setting("indicators.sessions", {}) or {}
        sessions[name] = self.get_settings()
        set_setting("indicators.sessions", sessions)
        set_setting("indicators.default_session", name)
        QMessageBox.information(self, "Sessione", f"Sessione '{name}' salvata.")
        self._load_sessions_into_combo()

    def _on_session_delete(self):
        name = self.session_combo.currentText()
        if name == "default":
            QMessageBox.warning(self, "Sessione", "Non puoi eliminare la sessione 'default'.")
            return
        sessions: Dict[str, dict] = get_setting("indicators.sessions", {}) or {}
        if name not in sessions:
            QMessageBox.information(self, "Sessione", "Nulla da eliminare.")
            return
        del sessions[name]
        set_setting("indicators.sessions", sessions)
        set_setting("indicators.default_session", "default")
        self._load_sessions_into_combo()
        self.session_combo.setCurrentText("default")

    # --------------------------- defaults ------------------------- #
    def _apply_default_params_only(self):
        """Ripristina **solo** i parametri numerici/booleani ai default; **non** tocca i colori."""
        g = _DEFAULTS
        self.cb_atr.setChecked(bool(g["use_atr"]))
        self.sp_atr_n.setValue(int(g["atr_n"]))

        self.cb_rsi.setChecked(bool(g["use_rsi"]))
        self.sp_rsi_n.setValue(int(g["rsi_n"]))

        self.cb_bb.setChecked(bool(g["use_bollinger"]))
        self.sp_bb_n.setValue(int(g["bb_n"]))
        self.d_bb_k.setValue(float(g["bb_k"]))

        self.cb_hurst.setChecked(bool(g["use_hurst"]))
        self.sp_hurst.setValue(int(g["hurst_window"]))

        self.cb_don.setChecked(bool(g["use_don"]))
        self.sp_don.setValue(int(g["don_n"]))

        self.cb_kelt.setChecked(bool(g["use_keltner"]))
        self.d_kelt_k.setValue(float(g["keltner_k"]))

        self.cb_macd.setChecked(bool(g["use_macd"]))
        self.sp_macd_fast.setValue(int(g["macd_fast"]))
        self.sp_macd_slow.setValue(int(g["macd_slow"]))
        self.sp_macd_signal.setValue(int(g["macd_signal"]))

        self.cb_sma.setChecked(bool(g["use_sma"]))
        self.sp_sma.setValue(int(g["sma_n"]))

        self.cb_ema.setChecked(bool(g["use_ema"]))
        self.sp_ema_fast.setValue(int(g["ema_fast"]))
        self.sp_ema_slow.setValue(int(g["ema_slow"]))

        self.sp_warmup.setValue(int(g["warmup_bars"]))
        self.sp_rv_win.setValue(int(g["rv_window"]))

    # --------------------------- applicazione iniziale ------------- #
    def _apply_initial_or_default(self):
        # 1) prova initial
        if self._initial:
            self._apply_from_dict(self._initial)
            return
        # 2) prova sessione di default
        sessions: Dict[str, dict] = get_setting("indicators.sessions", {}) or {}
        def_name = get_setting("indicators.default_session", "default")
        if def_name in sessions:
            self._apply_from_dict(sessions[def_name])
            return
        # 3) fallback: default params + default colors
        self._apply_default_params_only()
        self._apply_default_colors_only()

    def _apply_default_colors_only(self):
        for key, hexc in _DEFAULT_COLORS.items():
            btn = getattr(self, f"btn_{key}", None)
            if isinstance(btn, QPushButton):
                self._apply_button_color(btn, hexc)

    def _apply_from_dict(self, cfg: Dict[str, Any]):
        # parametri
        g = {**_DEFAULTS, **(cfg or {})}
        self.cb_atr.setChecked(bool(g.get("use_atr")))
        self.sp_atr_n.setValue(int(g.get("atr_n")))
        self.cb_rsi.setChecked(bool(g.get("use_rsi")))
        self.sp_rsi_n.setValue(int(g.get("rsi_n")))
        self.cb_bb.setChecked(bool(g.get("use_bollinger")))
        self.sp_bb_n.setValue(int(g.get("bb_n")))
        self.d_bb_k.setValue(float(g.get("bb_k")))
        self.cb_hurst.setChecked(bool(g.get("use_hurst")))
        self.sp_hurst.setValue(int(g.get("hurst_window")))
        self.cb_don.setChecked(bool(g.get("use_don")))
        self.sp_don.setValue(int(g.get("don_n")))
        self.cb_kelt.setChecked(bool(g.get("use_keltner")))
        self.d_kelt_k.setValue(float(g.get("keltner_k")))
        self.cb_macd.setChecked(bool(g.get("use_macd")))
        self.sp_macd_fast.setValue(int(g.get("macd_fast")))
        self.sp_macd_slow.setValue(int(g.get("macd_slow")))
        self.sp_macd_signal.setValue(int(g.get("macd_signal")))
        self.cb_sma.setChecked(bool(g.get("use_sma")))
        self.sp_sma.setValue(int(g.get("sma_n")))
        self.cb_ema.setChecked(bool(g.get("use_ema")))
        self.sp_ema_fast.setValue(int(g.get("ema_fast")))
        self.sp_ema_slow.setValue(int(g.get("ema_slow")))
        self.sp_warmup.setValue(int(g.get("warmup_bars")))
        self.sp_rv_win.setValue(int(g.get("rv_window")))
        # colori (se presenti, altrimenti lasciali invariati)
        for key, default_hex in _DEFAULT_COLORS.items():
            hexc = cfg.get(key, None)
            btn = getattr(self, f"btn_{key}", None)
            if isinstance(btn, QPushButton) and isinstance(hexc, str):
                self._apply_button_color(btn, hexc)

    # ----------------------------- API ---------------------------- #
    def get_settings(self) -> Dict[str, Any]:
        data = {
            # base
            "use_atr": self.cb_atr.isChecked(),
            "atr_n": int(self.sp_atr_n.value()),
            "use_rsi": self.cb_rsi.isChecked(),
            "rsi_n": int(self.sp_rsi_n.value()),
            "use_bollinger": self.cb_bb.isChecked(),
            "bb_n": int(self.sp_bb_n.value()),
            "bb_k": float(self.d_bb_k.value()),
            "use_hurst": self.cb_hurst.isChecked(),
            "hurst_window": int(self.sp_hurst.value()),
            # avanzati
            "use_don": self.cb_don.isChecked(),
            "don_n": int(self.sp_don.value()),
            "use_keltner": self.cb_kelt.isChecked(),
            "keltner_k": float(self.d_kelt_k.value()),
            "use_macd": self.cb_macd.isChecked(),
            "macd_fast": int(self.sp_macd_fast.value()),
            "macd_slow": int(self.sp_macd_slow.value()),
            "macd_signal": int(self.sp_macd_signal.value()),
            "use_sma": self.cb_sma.isChecked(),
            "sma_n": int(self.sp_sma.value()),
            "use_ema": self.cb_ema.isChecked(),
            "ema_fast": int(self.sp_ema_fast.value()),
            "ema_slow": int(self.sp_ema_slow.value()),
            # globali
            "warmup_bars": int(self.sp_warmup.value()),
            "rv_window": int(self.sp_rv_win.value()),
        }
        # colori
        data.update({
            "color_atr": self.btn_color_atr.property("hex") or _DEFAULT_COLORS["color_atr"],
            "color_rsi": self.btn_color_rsi.property("hex") or _DEFAULT_COLORS["color_rsi"],
            "color_bollinger": self.btn_color_bb.property("hex") or _DEFAULT_COLORS["color_bollinger"],
            "color_hurst": self.btn_color_hurst.property("hex") or _DEFAULT_COLORS["color_hurst"],
            "color_don": self.btn_color_don.property("hex") or _DEFAULT_COLORS["color_don"],
            "color_keltner": self.btn_color_kelt.property("hex") or _DEFAULT_COLORS["color_keltner"],
            "color_macd": self.btn_color_macd.property("hex") or _DEFAULT_COLORS["color_macd"],
            "color_sma": self.btn_color_sma.property("hex") or _DEFAULT_COLORS["color_sma"],
            "color_ema": self.btn_color_ema.property("hex") or _DEFAULT_COLORS["color_ema"],
        })
        # sessione attuale (per comodità)
        data["session_name"] = self.session_combo.currentText() or "default"
        return data

    @staticmethod
    def edit(parent: Optional[QWidget] = None, initial: Optional[dict] = None) -> Optional[dict]:
        dlg = IndicatorsDialog(parent=parent, initial=initial or {})
        if dlg.exec() == QDialog.Accepted:
            # salva anche la sessione di default selezionata
            set_setting("indicators.default_session", dlg.session_combo.currentText() or "default")
            return dlg.get_settings()
        return None


__all__ = ["IndicatorsDialog"]
