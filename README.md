# ForexGPT

- Requisiti: Python 3.12, `pip install -e .`
- Avvio GUI: `python scripts/run_gui.py --testserver`
- Backtesting Tab aggiornato: scroll, indicatori con range/timeframe, forecast con flag ore/giorno.
- Generazione combinazioni completa (modelli × tipi × indicatori × parametri × flag tempo).
- ChartTab su Qt Designer (.ui) + controller separati (zoom, badge, data loader).
- Simboli/timeframe dal DB (fallback se DB vuoto). Opzioni: mostra/nascondi serie, profili colore personalizzabili e persistenti.
