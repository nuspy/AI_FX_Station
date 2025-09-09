STREAMING vs POLLING (nota rapida)

- Stato attuale:
  L'app usa polling HTTP: RealTimeIngestService chiama periodicamente MarketDataService.provider.get_current_price()
  e AlphaVantageClient effettua richieste singole (httpx.get). Non è presente uno stream persistente (WebSocket / SSE).

- Come verificarlo:
  - Controlla i log: vedi righe ripetute come "RealTime: no price found for ..." o "RealTime: upsert report ..." a intervalli regolari.
  - Cerca nei sorgenti: RealTimeIngestService._poll_symbol() e AlphaVantageClient._get() (httpx) confermano il polling.

- Se vuoi STREAMING (realtime push):
  1) Scegli un provider che esponga WebSocket/SSE (es. provider commerciali come Polygon, IEX, provider FX con WS).
  2) Implementa un client streaming (es. StreamingClient) che apre connessione WS, gestisce reconnect e decodifica messaggi.
  3) In MarketDataService sostituisci/aggiungi il provider streaming e fornisci callback per i tick.
  4) Cambia RealTimeIngestService per ricevere eventi invece di fare polling: usa un loop asincrono o thread che ascolta i callback e costruisce le minute candles.
  5) Gestisci rate limit, backpressure e batching (non salvare ogni singolo tick synchronously).

- Alternative meno invasive:
  - Riduci poll_interval nella config e usa cache per non superare rate limit.
  - Usa long‑polling o server-sent events se disponibili.
  - Implementa un layer centralizzato che riceve stream esterno (es. via socket) e posta i tick in un endpoint interno che l'app consuma.

- Note pratiche:
  - AlphaVantage free API ha severe limitazioni: per minute-bars è comune dover usare backfill daily o contare sul polling per generare minute bars dal prezzo corrente.
  - Per streaming vero considera provider a pagamento o gateway che forniscono WebSocket.

Esempio rapido (roadmap):
- short-term: mantenere polling, impostare poll_interval >= 1s e rispettare rate_limit config.
- mid-term: aggiungere StreamingClient e un flag config.providers.<name>.streaming: true.
- long-term: migrare RealTimeIngestService a consumer di eventi asincroni, usare DBWriter per writes async.
