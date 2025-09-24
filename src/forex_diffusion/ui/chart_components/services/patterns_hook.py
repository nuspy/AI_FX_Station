# src/forex_diffusion/ui/chart_components/services/patterns_hook.py
from __future__ import annotations
from typing import Optional
import pandas as pd
from loguru import logger

from .patterns_service import PatternsService

_SERVICE_ATTR = "_patterns_service"


def get_patterns_service(controller, view, create: bool = False) -> Optional[PatternsService]:
    """Ritorna il singleton del PatternsService agganciato al controller."""
    ps = getattr(controller, _SERVICE_ATTR, None)
    if ps is None and create:
        ps = PatternsService(view, controller)
        setattr(controller, _SERVICE_ATTR, ps)
    return ps


def set_patterns_toggle(controller, view, *,
                        chart: Optional[bool] = None,
                        candle: Optional[bool] = None,
                        history: Optional[bool] = None) -> None:
    """
    API di comodo per attivare/disattivare i tre toggle senza toccare attributi del controller.
    Non chiama view.repaint: il service ridisegna da solo.
    """
    ps = get_patterns_service(controller, view, create=True)
    if chart is not None:
        ps.set_chart_enabled(bool(chart))
    if candle is not None:
        ps.set_candle_enabled(bool(candle))
    if history is not None:
        ps.set_history_enabled(bool(history))

    # Se almeno un tipo è ON, trigghiamo una detection con l’ultimo df noto.
    if ps and (ps._enabled_chart or ps._enabled_candle):
        df = getattr(controller.plot_service, "_last_df", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            ps.detect_async(df)
        else:
            logger.debug("Patterns: nessun df disponibile per (ri)scan dopo toggle.")


def call_patterns_detection(controller, view,
                            df: Optional[pd.DataFrame] = None,
                            force: bool = False) -> None:
    """
    Chiamata esplicita per lanciare la detection.
    - df: se None, usa l’ultimo df del plot_service.
    - force: se True, lancia comunque anche a df identico (lascia passare il debounce del service).
    """
    ps = get_patterns_service(controller, view, create=True)
    if ps is None:
        logger.debug("Patterns: service non disponibile (get_patterns_service -> None).")
        return

    if df is None:
        df = getattr(controller.plot_service, "_last_df", None)

    if isinstance(df, pd.DataFrame) and not df.empty:
        ps.detect_async(df)   # debounce → poi draw automatico
    elif force:
        # Force: anche senza df, prova a usare quello del controller (se arriva a breve, il debounce lo cattura)
        logger.debug("Patterns: force detection senza df; in attesa di un update plot.")
    else:
        logger.debug("Patterns: call_patterns_detection senza df utile; skip.")


def bind_patterns_to_plot_updates(plot_service, controller, view) -> None:
    """
    Collega la detection all’aggiornamento del plot. Chiama questa una volta dopo aver creato la UI.
    Adatta il ‘gancio’ all’API del tuo PlotService:
      - se hai un segnale, connetti quel segnale a _on_plot_updated
      - altrimenti inserisci call_patterns_detection in coda a update_plot(...)
    """
    ps = get_patterns_service(controller, view, create=True)

    def _on_plot_updated(df: pd.DataFrame) -> None:
        # Debounced dal service; non serve repaint manuale
        ps.detect_async(df)

    # Esempi (scegline uno a seconda della tua implementazione):
    if hasattr(plot_service, "plot_updated"):  # Signal/pyqtSignal
        try:
            plot_service.plot_updated.connect(_on_plot_updated)
            logger.info("Patterns: collegato a plot_service.plot_updated")
            return
        except Exception:
            pass

    # Fallback: se il tuo PlotService espone un hook post-update, registralo:
    if hasattr(plot_service, "set_post_update_hook"):
        try:
            plot_service.set_post_update_hook(_on_plot_updated)
            logger.info("Patterns: collegato via set_post_update_hook")
            return
        except Exception:
            pass

    logger.debug("Patterns: nessun hook automatico trovato; inserisci manualmente call_patterns_detection alla fine di update_plot().")
