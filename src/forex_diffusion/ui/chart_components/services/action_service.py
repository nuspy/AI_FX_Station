from __future__ import annotations

from loguru import logger
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from forex_diffusion.ui.trade_dialog import TradeDialog
from forex_diffusion.utils.user_settings import get_setting, set_setting

from .base import ChartServiceBase


class ActionService(ChartServiceBase):
    """Auto-generated service extracted from ChartTab."""

    def _open_trade_dialog(self) -> None:
        try:
            symbols = self._symbols_supported
            cur = getattr(self, "symbol", symbols[0])
            parent = getattr(self, "view", None) or getattr(self, "_main_window", None)
            dlg = TradeDialog(parent=parent, symbols=symbols, current_symbol=cur)
            if dlg.exec() == QDialog.Accepted:
                order = dlg.get_order()
                try:
                    ok, oid = self.broker.place_order(order)
                    if ok:
                        QMessageBox.information(self.view, "Order", f"Order placed (id={oid})")
                    else:
                        QMessageBox.warning(self.view, "Order", f"Order rejected: {oid}")
                except Exception as e:
                    QMessageBox.warning(self.view, "Order", str(e))
        except Exception as e:
            logger.exception("Trade dialog failed: {}", e)

    def _on_indicators_clicked(self) -> None:
        """Chiede al controller di aprire il dialog; in fallback mostra un info box."""
        ctrl = self.app_controller or getattr(self._main_window, "controller", None)
        try:
            logger.info("Indicators button clicked; controller is {}", type(ctrl).__name__ if ctrl else None)
        except Exception:
            pass

        if ctrl and hasattr(ctrl, "handle_indicators_requested"):
            try:
                ctrl.handle_indicators_requested()
                return
            except Exception as e:
                QMessageBox.warning(self.view, "Indicators", f"Errore apertura dialog: {e}")
                return

        try:
            dlg = QDialog(self.view)
            dlg.setWindowTitle("Indicators (fallback)")
            lay = QVBoxLayout(dlg)
            lay.addWidget(QLabel("Impostazioni rapide indicatori"))

            chk_wk = QCheckBox("Fill area tra Bande Bollinger & Keltner")
            try:
                chk_wk.setChecked(bool(get_setting("indicators.fill_wk", True)))
            except Exception:
                chk_wk.setChecked(True)
            lay.addWidget(chk_wk)

            chk_fcst = QCheckBox("Calcola/Mostra indicatori anche sul forecast (q50)")
            try:
                chk_fcst.setChecked(bool(get_setting("indicators.on_forecast", False)))
            except Exception:
                chk_fcst.setChecked(False)
            lay.addWidget(chk_fcst)

            buttons = QHBoxLayout()
            okb = QPushButton("Salva")
            canc = QPushButton("Annulla")
            buttons.addStretch(1)
            buttons.addWidget(okb)
            buttons.addWidget(canc)
            lay.addLayout(buttons)

            def _save():
                try:
                    set_setting("indicators.fill_wk", bool(chk_wk.isChecked()))
                    set_setting("indicators.on_forecast", bool(chk_fcst.isChecked()))
                except Exception:
                    pass
                dlg.accept()
                if self._last_df is not None and not self._last_df.empty:
                    self.update_plot(self._last_df)

            okb.clicked.connect(_save)
            canc.clicked.connect(dlg.reject)
            dlg.exec()
        except Exception:
            QMessageBox.information(self.view, "Indicators", "Controller non disponibile. Apri dal menu principale.")

    def _on_build_latents_clicked(self) -> None:
        """Prompt PCA dim and launch latents build via controller."""
        try:
            sym = getattr(self, "symbol", None)
            tf = getattr(self, "timeframe", None)
            if not sym or not tf:
                QMessageBox.information(self.view, "Latents", "Imposta prima symbol e timeframe.")
                return
            dim, ok = QInputDialog.getInt(self.view, "Build Latents (PCA)", "Components (dim):", 64, 2, 512, 1)
            if not ok:
                return
            bars, ok2 = QInputDialog.getInt(self.view, "Build Latents (PCA)", "History bars to use:", 100000, 1000, 1000000, 1000)
            if not ok2:
                bars = 100000
            controller = getattr(self._main_window, "controller", None)
            if controller and hasattr(controller, "handle_build_latents"):
                controller.handle_build_latents(symbol=sym, timeframe=tf, dim=dim, bars=bars)
            else:
                QMessageBox.information(self.view, "Latents", "Controller non disponibile.")
        except Exception as e:
            logger.exception("Failed to start latents build: {}", e)
            QMessageBox.warning(self.view, "Latents", str(e))
