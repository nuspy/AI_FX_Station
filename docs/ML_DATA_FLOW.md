ML Data Flow — quali dati formano i tensori (esteso e operativo)

1) Sorgenti dati
- market_data_candles: tabella principale con le barre OHLCV e ts_utc (ms UTC); usata dalla GUI HistoryTab e dalla pipeline.
- latents: tabella che contiene vettori latenti (latent_json) calcolati dall'encoder VAE o pipeline.

2) Nuove time features richieste (aggiunte)
- day_of_week: 0=Monday ... 6=Sunday (intero o one-hot).
- hour (senza minuti): intero 0..23.
- hour_sin, hour_cos: codifica ciclica dell'ora (sin/cos).
- session dummies: session_tokyo, session_london, session_ny (0/1).
Nota: tutte le time-features devono essere calcolate a partire da ts_utc in UTC e poi eventualmente convertite/localizzate per visualizzazione.

3) Multi‑timeframe indicators richiesti
- Timeframe rilevanti (default): 5m,10m,20m,30m,45m,
  1h,2h,3h,4h,5h,6h,7h,8h,9h,10h,11h,12h,18h,
  1d,2d,3d,4d,5d
- Per ogni timeframe la pipeline calcola (prefisso: "<tf>_"):
  - r: Log-return
  - hl_range: High–Low Range
  - atr: Average True Range
  - rv: rolling volatility (std of returns)
  - gk_vol: Garman–Klass volatility
  - ema_fast, ema_slow
  - ema_slope (slope of fast EMA)
  - macd, macd_signal, macd_hist
  - rsi
  - bb_upper, bb_lower, bb_width, bb_pctb (Bollinger)
  - kelt_upper, kelt_lower (Keltner channels)
  - don_upper, don_lower (Donchian)
  - realized_skew, realized_kurt (rolling)
  - hurst (rolling)
  - vol_mean (rolling volume mean, if volume present)
- Nome colonne: "<tf>_<indicator>" (es. "5m_r", "1h_atr", "1d_bb_pctb").

4) Come si calcolano e si allineano (best practice)
- Per ogni timeframe: resample causale (pandas resample label='right', closed='right'), poi calcolo indicatori in modo causale (rolling, ewm adjust=False).
- Allineamento: merge_asof con il DataFrame base (timeframe di input) usando timestamp del periodo end; direzione "backward" (prendi ultimo valore disponibile ≤ ts).
- Warmup: rimuovere warmup_bars iniziali per evitare NaN (configurabile).

5) Canali VAE / tensori di input
- Configurabile in configs/default.yaml -> vae.channels; esempio consigliato:
  ["open","high","low","close","volume","hour_sin","hour_cos","session_tokyo","session_london","session_ny"]
- Per training encoder VAE: costruire patches sliding window length patch_len e normalizzare; tensore shape = [batch, channels, patch_len].
- Le features multi-timeframe possono essere usate come conditioning vector (concatenate al patch o come embedding separato).

6) Standardizzazione e persist
- Standardizer causale: fit su porzione train only (no leakage) e applicare a validation/test; pipeline offre Standardizer con fit/transform.
- Persist standardizer parameters (mu/sigma) assieme al modello se necessario.

7) Esempio pratico: prova end-to-end (PowerShell)
Passi:
A) Assicurati DB ha enough 1m candles per il periodo (use check_candles):
  .\scripts\check_candles.ps1 -Symbol "EUR/USD" -Timeframe "1m"

B) Backfill se necessario (ti mostro backfill breve 3 giorni):
  $env:TIINGO_API_KEY="YOUR_KEY"
  .\scripts\run_backfill_and_check.ps1 -Symbol "EUR/USD" -Timeframe "1m" -Days 3

C) Esegui pipeline su un segmento e salva features CSV:
  powershell:
    python - <<'PY'
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path('.').resolve()/'src'))
    from forex_diffusion.services.db_service import DBService
    from forex_diffusion.features.pipeline import pipeline_process
    import pandas as pd
    db = DBService()
    with db.engine.connect() as conn:
        df = pd.read_sql("SELECT ts_utc, open, high, low, close, volume FROM market_data_candles WHERE symbol='EUR/USD' AND timeframe='1m' ORDER BY ts_utc ASC LIMIT 10000", conn)
    feats, std = pipeline_process(df, timeframe='1m', multi_timeframes=['5m','15m','1h','1d'])
    feats.to_csv('tmp/features_sample.csv', index=False)
    print('Saved tmp/features_sample.csv, rows=', len(feats))
    PY
  (In PowerShell puoi salvare lo script in tmp/run_pipeline.py e lanciarlo: python tmp\run_pipeline.py)

D) Ispeziona features:
  .\tmp\features_sample.csv con Excel o pandas (head).

8) Note implementative per il codice
- Tutti i calcoli devono essere causali (no future leakage).
- Per indicatori che richiedono volume, gestire assenza volume con fallback (0 o NaN poi fill).
- Per Hurst/realized moments, limitare window e rendere calcoli robusti su serie corte (fillna/0.0).

9) Debug rapido (se le features risultano vuote)
- Riduci warmup in pipeline_process (features_config={"warmup_bars":16}) per test.
- Verifica che ts_utc siano coerenti e ordinati.

10) Azione successiva che posso applicare ora
- Creare lo script tests/manual_tests/extract_patches.py che estrae tensori [batch,channels,patch_len] e li salva in .npz; aggiungo wrapper PowerShell per lanciarlo.
- Oppure guidarti passo‑passo via PowerShell per eseguire i comandi A→C sopra.

CONTINUA_NECESSARIA: vuoi che crei subito lo script extract_patches.py + scripts/extract_patches.ps1 per esportare i tensori .npz (rispondere "sì") oppure preferisci che ti guidi passo‑passo nella shell per eseguire i comandi manuali? 
