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

# Try to import enhanced chart service
try:
    from .enhanced_chart_service import EnhancedChartService, ChartSystemSelector
    ENHANCED_CHART_AVAILABLE = True
except ImportError:
    ENHANCED_CHART_AVAILABLE = False
    EnhancedChartService = None
    ChartSystemSelector = None


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

    def _on_chart_system_clicked(self) -> None:
        """Open chart system selector dialog."""
        try:
            if ENHANCED_CHART_AVAILABLE:
                dialog = ChartSystemSelector(self.view)
                if dialog.exec() == QDialog.Accepted:
                    new_system = dialog.selected_system
                    QMessageBox.information(self.view, "Chart System",
                                          f"Chart system will be switched to: {new_system}")
                    # Note: Actual switching would be implemented in the chart controller
            else:
                QMessageBox.information(self.view, "Chart System",
                                      "Enhanced chart system not available. Using default matplotlib.")
        except Exception as e:
            logger.error(f"Error opening chart system selector: {e}")
            QMessageBox.warning(self.view, "Chart System", f"Error: {e}")

    def _on_indicators_clicked(self) -> None:
        """Open the enhanced indicators dialog with VectorBT Pro + bta-lib integration."""
        ctrl = self.app_controller or getattr(self._main_window, "controller", None)
        try:
            logger.info("Indicators button clicked; controller is {}", type(ctrl).__name__ if ctrl else None)
        except Exception:
            pass

        # Try enhanced indicators dialog first
        try:
            from ...enhanced_indicators_dialog import EnhancedIndicatorsDialog

            # Get available data columns for the dialog
            available_data = ['open', 'high', 'low', 'close', 'volume']

            # Create and show the enhanced dialog
            dialog = EnhancedIndicatorsDialog(self.view, available_data)

            if dialog.exec() == QDialog.Accepted:
                # Get enabled indicators from dialog
                enabled_indicators = dialog.get_enabled_indicators()
                logger.info(f"User enabled {len(enabled_indicators)} indicators")

                # Store the configuration for chart plotting
                try:
                    # Store enabled indicators in settings for later use
                    indicator_names = list(enabled_indicators.keys())
                    set_setting("indicators.enabled_list", indicator_names)

                    # Trigger chart update if we have data
                    if hasattr(self, '_last_df') and self._last_df is not None and not self._last_df.empty:
                        # Force chart redraw with new indicators
                        if hasattr(self, 'update_plot'):
                            self.update_plot(self._last_df)

                    QMessageBox.information(self.view, "Indicators",
                                          f"Successfully configured {len(enabled_indicators)} indicators!")

                except Exception as e:
                    logger.error(f"Error applying indicator settings: {e}")
                    QMessageBox.warning(self.view, "Indicators", f"Error applying settings: {e}")

            return

        except ImportError as e:
            logger.warning(f"Enhanced indicators dialog not available: {e}")
        except Exception as e:
            logger.error(f"Error opening enhanced indicators dialog: {e}")
            QMessageBox.warning(self.view, "Indicators", f"Error opening enhanced dialog: {e}")

        # Fallback to controller method if available
        if ctrl and hasattr(ctrl, "handle_indicators_requested"):
            try:
                ctrl.handle_indicators_requested()
                return
            except Exception as e:
                QMessageBox.warning(self.view, "Indicators", f"Errore apertura dialog: {e}")
                return

        # Final fallback to simple dialog
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
                if hasattr(self, '_last_df') and self._last_df is not None and not self._last_df.empty:
                    if hasattr(self, 'update_plot'):
                        self.update_plot(self._last_df)

            okb.clicked.connect(_save)
            canc.clicked.connect(dlg.reject)
            dlg.exec()
        except Exception:
            QMessageBox.information(self.view, "Indicators", "Controller non disponibile. Apri dal menu principale.")

    def _on_build_latents_clicked(self) -> None:
        """Prompt PCA dim and launch latents build via controller."""
        try:
            sym = getattr(self.view, "symbol", None)
            tf = getattr(self.view, "timeframe", None)
            if not sym or not tf:
                QMessageBox.information(self.view, "Latents", "Imposta prima symbol e timeframe.")
                return
            dim, ok = QInputDialog.getInt(self.view, "Build Latents (PCA)", "Components (dim):", 64, 2, 512, 1)
            if not ok:
                return
            bars, ok2 = QInputDialog.getInt(self.view, "Build Latents (PCA)", "History bars to use:", 100000, 1000, 1000000, 1000)
            if not ok2:
                bars = 100000
            controller = getattr(self.view._main_window, "controller", None)
            if controller and hasattr(controller, "handle_build_latents"):
                controller.handle_build_latents(symbol=sym, timeframe=tf, dim=dim, bars=bars)
            else:
                QMessageBox.information(self.view, "Latents", "Controller non disponibile.")
        except Exception as e:
            logger.exception("Failed to start latents build: {}", e)
            QMessageBox.warning(self.view, "Latents", str(e))
