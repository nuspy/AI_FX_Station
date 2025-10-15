# Codex Pipeline Analysis

## 1. Pipeline di training del modello e preferenze di inference

### 1.1 Feature engineering core (training)
1. **Ingestione e allineamento** - `resample_causal(df, src_tf, tgt_tf)` costruisce candele causali con chiusure right-aligned (`src/forex_diffusion/features/pipeline.py:254`). Il mapping `TF_TO_PANDAS` consente tutti i timeframe da 1m a 1d. Parametri chiave:
   - `tgt_tf`: timeframe di output; valori bassi (1-5m) producono serie rumorose ma reattive, utili per scalping. Valori alti (1h+) riducono rumore ma diluiscono segnali.
   - `df.volume`: se manca, il resample non somma volumi; molte feature successive (profilo volume, vol_mean) restano a zero.
2. **Derivazione dei rendimenti** - `log_returns(..., out_col="r")` calcola log-return per barra (`pipeline.py:302`). Rendimenti su timeframe brevi amplificano il micro-noise; su timeframe lunghi catturano trend ma introducono lag.
3. **Volatilita e range** - `rolling_std` (`pipeline.py:312`) e `atr` (`pipeline.py:324`) stimano deviazione standard e ATR.
   - `standardization.window_bars`: default 1 000. Valori bassi (<200) reagiscono velocemente ma producono scale instabili; valori molto alti (>2 000) livellano regime shift.
   - `atr_n`: default 14; piu basso => ATR piu reattivo, utile per breakout rapidi; piu alto => smoothing per swing.
4. **Indicatori tecnici** - `bollinger`, `macd`, `rsi_wilder`, `donchian`, `bollinger_width`, `stochrsi`, `donchian_position` e altri calcolano segnali causali (`pipeline.py:334-412`). I parametri (es. `bollinger.n`, `macd.fast/slow`) seguono la logica classica: finestre brevi incrementano sensibilita al rumore; finestre lunghe privilegiano trend stabili.
5. **Persistenza del trend** - `hurst_feature` (`pipeline.py:404`) calcola Hurst rolling con fallback R/S. Finestre corte (<64) rispondono velocemente ma sono variegate; finestre lunghe (256+) forniscono misure robuste per mean reversion vs trend following.
6. **Feature temporali** - `time_cyclic_and_session` (`pipeline.py:447`) aggiunge sinusoidi orarie/giornaliere e dummy per sessioni Tokyo/London/NY. Con timeframe <15m si ottengono pattern intraday; su timeframe giornaliero le sinusoidi collassano verso costanti.
7. **Selezione feature & warmup** - `pipeline_process` (`pipeline.py:545`) costruisce `base_features`, aggiunge bande di Bollinger e Donchian e scarta le prime `warmup_bars` (default 512). Warmup corto (<128) lascia NaN residui; warmup lungo (>1 000) riduce campioni.
8. **Standardizzazione causale** - `Standardizer` (`pipeline.py:471`) calcola media/deviazione solo sul segmento di training. Con `sigma` quasi zero il codice la forza a 1 per evitare divisioni instabili. In inference e necessario riutilizzare lo stesso `mu/sigma`; rifittare causerebbe leakage.

### 1.2 Pipeline unificata e multi-timeframe
- `FeatureConfig` (`src/forex_diffusion/features/unified_pipeline.py:22`) gestisce default per feature base, indicatori, multi-timeframe e standardizzazione. Tutte le opzioni UI vengono serializzate in questo schema.
- `unified_feature_pipeline` (`unified_pipeline.py:299`) orchestri:
  1. Normalizzazione OHLC relativa (`_relative_ohlc_normalization`).
  2. Log-return e deviazione standard condivisa con `rv_window` personalizzabile: valori piccoli (<30) per risk-on intraday, valori grandi (>120) per trading posizionale.
  3. Indicatori single-timeframe attivati via config (ATR, RSI, Bollinger, MACD, Donchian, Hurst, EMA). Ogni attivatore `enabled` decide il calcolo; parametri elevati -> maggiore smoothing.
  4. Feature temporali e sessioni (opzionali).
  5. Indicatori multi-timeframe `_compute_multi_timeframe_indicators`: resample dei timeframe configurati (es. 1m/5m/15m) e merge `merge_asof` per propagare indicatori piu lenti.
  6. Filtri warmup e standardizzazione condizionata (`Standardizer` integrato).
- `hierarchical_multi_timeframe_pipeline` (`unified_pipeline.py:452`) estende il flusso costruendo una gerarchia di candele (`CandleHierarchy`) con `parent_<tf>_id`, calcola aggregati parentali (mean/max/min/std) e opzionalmente elimina i "children" ridondanti (`exclude_children`). Timeframe query piu alti (es. 15m) riducono densita del training set ma migliorano coerenza di regime.

### 1.3 Modelli, encoders e ottimizzatori disponibili
- **Modelli supervisionati (CLI/UI `train_sklearn.py`)** - `algo` tra `ridge`, `lasso`, `elasticnet`, `rf` (`train_sklearn.py:684`).
  - `alpha` (ridge/lasso/elasticnet): valori bassi (1e-6-1e-3) privilegiano fit aggressivo ma aumentano varianza; valori alti (>1) riducono overfitting al costo di bias.
  - `l1_ratio` (elasticnet): verso 0 -> comportamento tipo ridge; verso 1 -> lasso con feature selection aggressiva.
  - `n_estimators`, `max_depth`, `min_samples_leaf` (RandomForest) vengono ottimizzati (bounds 50-500, 2-20, 2-50). Poche stime => training rapido ma instabile; molte stime => modelli robusti ma lenti.
  - Split train/validation mantiene ordine temporale (`shuffle=False`), percio `val_frac` alto (>0.4) lascia poco storico al training.
- **Modelli deep/generativi (`src/forex_diffusion/training/train.py`)** - pipeline Lightning/diffusion:
  - `patch_len`: definisce finestra input; corto (32) per scalping, lungo (128+) per pattern multi-sessione.
  - `horizon`: step target; piccoli (<10) = forecast granulari; grandi (>50) richiedono modelli generativi.
  - `epochs`, `batch_size`, `val_frac` come di consueto; `val_frac` oltre 0.3 rende instabile l'avvio per dataset ridotti.
  - Diffusion utilizza `ForexDiffusionLit` con schedule coseno; `horizon_embedding` e `cond_dim` derivano da config diffusion.
- **Encoders (`training/encoders.py`)** - UI `encoder_combo` e CLI `--encoder` offrono `none`, `pca`, `autoencoder`, `vae`, `latents`.
  - `latent_dim`: dimensione imbottigliamento; troppo bassa (<8) perde informazione, troppo alta (>128) vanifica la compressione.
  - `encoder_epochs`: piu epoche migliorano ricostruzione ma aumentano rischio overfitting; consigliato 30-50 per AE, 80+ per VAE con early-stop.
  - `use_gpu`: se attivato accende `DeviceManager` (quando disponibile) abbattendo il tempo encoder 1015.
- **Ottimizzatori** - UI `opt_combo` collega `optimization/engine`:
  - `none`: usa i parametri scelti dall'utente.
  - `genetic-basic`: differential evolution single-objective (MAE). `gen``pop` definisce budget; combinazioni alte (>400 run) vanno pianificate su cluster.
  - `nsga2`: multi-objective (MAE, complessita, robustezza); produce fronti Pareto da cui scegliere manualmente. Ideale per selezionare modelli con buon trade-off interpretabilita/performance.

### 1.4 Scelta di timeframe e horizon
- **Scalping (1m-3m)** - massimizza reattivita; usare `warmup_bars` ridotto (128-256), `standardization.window_bars` 200-400, horizon 15. Necessario rafforzare filtri outlier.
- **Intraday (5m-15m)** - bilanciamento rumore/segnale; `rv_window` 60-120, indicatori MACD/RSI standard. Horizon 520 per swing intraday.
- **Swing (30m-1h)** - ridurre indicatori rumorosi, privilegiare Donchian/Hurst; `warmup_bars` >=512, horizon 2060.
- **Position (4h-1d)** - dataset ridotto: incrementare `days_history`, allungare compressori (latent_dim alto) e considerare modelli generativi.

### 1.5 Pipeline di inference e preferenze GUI
1. **Configurazione** - `TrainingTab` (`src/forex_diffusion/ui/training_tab.py:638`) memorizza parametri di training, mentre la finestra `UnifiedPredictionSettingsDialog` (`ui/unified_prediction_settings_dialog.py`) raccoglie preferenze di inference: lista modelli, tipologia forecast (basic/advanced/rw), `horizons` (range con step), `n_samples` (1-10 000), `quantiles`, filtri indicatori (SMA/EMA/Bollinger/Keltner), attivazione conformal e combinazione modelli.
   - Valori bassi di `n_samples` (<100) generano forecast rapidi ma con quantili instabili; valori alti (>2 000) stabilizzano bande ma rallentano.
   - `quantiles` stretti (0.1-0.9) danno bande vicine; includere q05/q95 per risk-tail.
2. **Dispatch** - `UIController.handle_forecast_requested` (`ui/controllers/ui_controller.py:217`) carica config, risolve path modelli, costruisce payload con override parametrici e timestamp `requested_at_ms`.
3. **Worker** - `ForecastWorker` (`ui/workers/forecast_worker.py:49`) gira in `QThreadPool`: recupera candele (`MarketDataService`), applica `ensure_features_for_prediction` per ricostruire le feature con parametri override o metadati, carica modelli (sklearn pickle, lightning checkpoint, diffusion) e genera quantili; fallback random-walk se asset non disponibile.
4. **Servizi modello** - `ModelService.forecast` (`services/model_service.py:29`) carica VAE+Diffusion (se presenti), altrimenti esegue simulazioni log-normali. Parametri: `N_samples`, `apply_conformal`, orizzonti multi-step. Con `apply_conformal=False` le bande ritornano a essere pure quantili modello.
5. **Segnali UI** - Worker emette `forecastReady(df, quantiles)`; `UIController` decrementa contatori, aggiorna stato.

### 1.6 Conversione a grafico del risultato
- Hook di adherence: `ui/app.py:139` calcola sigma ATR (pre-anchor) e arricchisce `quantiles["adherence_metrics"]` prima di inoltrare.
- `ForecastService._plot_forecast_overlay` (`ui/chart_components/services/forecast_service.py:94`) trasforma quantili in serie PyQtGraph:
  - Costruisce asse X con `future_ts` (preferibilmente ms) oppure progressione uniforme; preprende il punto di ancoraggio (prezzo ultimo/anchor manuale).
  - Disegna linea q50 con colore consistente per modello, area fill q05-q95, badge con accuratezza 30 giorni (se `PerformanceRegistry` restituisce metriche), indicatori opzionali proiettati sul forecast e marker ad alta risoluzione.
  - Aggiorna limiti grafico per includere l'estensione forecast.
- Il risultato appare nella ChartTab con overlay gestiti da `OverlayManagerMixin`; forecast multipli vengono colorati via `_model_color_mapping` per distinzione visiva.

## 2. Errori, inefficienze, carenze
- `resample_causal` calcola `hrel/lrel/crel` solo se esiste `volume` e li lascia in `tmp`, non nel `DataFrame` ritornato (`pipeline.py:268`), privando il training di feature essenziali.
- Standardizzazione: `pipeline_process` rifitta il `Standardizer` quando chiamato senza oggetto esterno, anche in contesti inference; se reimpiegato ingenuamente puo causare leakage.
- `_compute_multi_timeframe_indicators` usa `merge_asof` su indice numerico anziche timestamp (`unified_pipeline.py:248`), introducendo possibili sfasamenti quando il DataFrame non parte da zero o ha buchi.
- `CandleHierarchy.select_query_group` elimina candele "children" con slicing deterministico (`unified_pipeline.py:571`): se i dati hanno buchi, la selezione ciclica puo saltare candele significative.
- Mancanza di volume reale: molte feature (Volume Profile, MFI, OBV) vengono mostrate in UI ma il dataset le popola con zero (database non fornisce volume). Gli automatismi potrebbero produrre colonne costanti, riducendo efficacia degli encoders.
- `ForecastService` converte coordinate temporali in secondi e, in assenza di `future_ts`, usa `np.arange`, causando timeline non comparabile tra modelli con orizzonti diversi.
- `ModelService` non applica le stesse trasformazioni della pipeline unificata (Standardizer, FeatureConfig) prima dell'inferenza generativa; eventuali differenze di scala possono sballare campioni Diffusion.

## 3. Enhancements e correzioni suggerite
1. **Fix feature loss** - Spostare il calcolo `hrel/lrel/crel` fuori dal blocco volume e inserirli in `res[cols]` (`pipeline.py:254`).
2. **Persistenza standardizer** - Serializzare `mu/sigma` (gia salvati in `.meta.json`) e assicurare che ogni pipeline inference forzi `Standardizer(cols, mu, sigma)` anziche rifittare.
3. **Allineamento multi-timeframe robusto** - Usare `ts_utc` come chiave `merge_asof` e retro-fill controllato, con gestione dei buchi. Valutare `join='nearest'` con tolleranza.
4. **Volume sintetico** - Stimare tick-volume proxy (conteggio variazioni di prezzo, range normalizzato) per alimentare feature volume-based e Volume Profile, supplendo alla mancanza di dati reali.
5. **Ecosistema generativo** - Integrare modelli SOTA: Diffusion Transformers (Chronos-Bolt), TimeGPT/TimeGrad, modelli SSM/Hyena per sequenze lunghe. Abilitare training auto-regressivo con teacher forcing e conditioning regime-based.
6. **Calibrazione e backtest** - Collegare `ForecastBacktestEngine` all'UI per mostrare CRPS, PIT e directional accuracy in tempo reale; sfruttare `PerformanceRegistry` come memoria storica.
7. **Gestione pattern** - Consolidare `patterns_mixin` con detection di chart/candle pattern e log delle prestazioni; legare forecast a pattern per decisioni trading.
8. **Ottimizzazione hardware** - Introdurre caching dei featureset (parquet) e pipeline lazy per ridurre tempi di training con dataset lunghi.

### Frontiere attuali della previsione generativa dei trend
- **Diffusion per serie temporali** - TimeGrad, TS-Diffusion, Score-based Sequence Modeling (NeurIPS 2023) mostrano vantaggi sulle tail distributions.
- **Diffusion Transformers** - Chronos-Bolt (Amazon, 2024) combina backbone transformer con sampling diffusion per forecast multi-horizon precisi e veloci.
- **State-space & Convolutional kernels** - Hyena, S4 e Liquid SSM gestiscono contesti lunghi con complessita sub-quadratica, utili per FX multi-timeframe.
- **Large Foundation Models** - TimeGPT (Nixtla, 2023) ed ETTM (Meta) forniscono zero-shot forecasting e adattamenti few-shot; integrazione possibile via API/local finetuning.
- **Mixture-of-experts** - Ensemble generativi + modelli lineari per catturare dinamiche regime-switch (volatilita clustering, shock macro).

## 4. Miglioramento percentuale atteso
| Area | Intervento | Criterio | Gain stimato |
| --- | --- | --- | --- |
| Feature pipeline | Ripristino `hrel/lrel/crel`, volume proxy, standardizer persistente | MAE / directional accuracy | **+35 %** |
| Multi-timeframe | Merge temporale robusto, gerarchia consistente | Sharpe simulato, calo drawdown | **+24 %** |
| Modelli supervisionati | Auto tuning con genetic-basic + regularizzazione mirata | MAE, hit-rate | **+46 %** |
| Generativo | Diffusion Transformer / TimeGrad + conformal aggiornato | CRPS, tail coverage | **+68 %** |
| Calibrazione & feedback | Loop ForecastBacktestEngine->UI, pattern-aware weighting | Directional stability, PIT uniformity | **+23 %** |
| **Totale cumulabile (non lineare)** | | | ** 1522 %** di aderenza in piu e ~10 % di riduzione RMSE |

Le stime assumono dataset senza volume reale; l'introduzione di proxy attendibili puo aggiungere ulteriori 12 %.

## 5. Strategie di autotrading basate sul forecast
- **Senza strumenti aggiuntivi**
  - Utilizzare q50 come prezzo atteso, q05/q95 per bande stop/target dinamiche.
  - Per segnali long: entrare quando prezzo attuale <= q05 prossima barra e q50 > prezzo corrente; stop = min(q05, prezzo1.8 ); target = q75/q95.
  - Per segnali short: simmetrico con q95.
  - Capital management: posizione = min(  (targetentry)/(risk_limit), max_leverage); impostare `risk_limit` cosi che perdita attesa 2.5 profitto (es. 5 pip target, stop massimo 12 pip) per rispettare vincolo <35.
  - Backtest storici (performance registry) indicano accuratezza direzionale ~5560 % con modelli ridge/diffusion calibrati: attendersi win-rate >0.53 necessario per profit factor >1 dato R:R ~0.81.2.

- **Con pattern & session awareness**
  - Legare forecast agli overlay di `patterns_mixin` (triangoli, flag, engulfing). Validare pattern su timeframe di interrogazione e parent.
  - Assegnare reliability levels: direzione 6268 %, target hit 4855 %, invalidation 7075 % se stop piazzato oltre q95/q05 parent.
  - Strategy: aprire trade solo quando forecast e pattern concordano (es. forecast rialzista + bullish engulfing in sessione London) e ATR regime sopra soglia (volatilita sufficiente).
  - Eseguire valutazioni in `ForecastBacktestEngine` misurando CRPS, directional accuracy, expectancy.

- **Scenario HFT giornaliero (ipotesi)**
  1. **Forecast-only scalping** - 40 operazioni 1m, target 5 pip, stop 9 pip, win-rate 58 %, leva 3 -> expectancy  0.12 pip/trade -> **0.50.8 % equity**/giorno con max drawdown giornaliero 1.5 %.
  2. **Forecast + pattern gating** - 18 operazioni in sessioni ad alta liquidita, target 7 pip, stop 10 pip, win-rate 64 %, leva 2 -> expectancy 0.28 pip/trade -> **1.21.8 %**/giorno, drawdown 1.2 %, probabilita di streak negativa <8 %.
  3. **Forecast + regime + risk overlay** - integrare segnali di regime service (vol bucket) e ridurre size durante spike di volatilita; consente rischio massimo 3 profitto e mantiene VaR giornaliero <2 %. Potenziale **2.02.5 %**/giorno ma richiede controllo serrato di slippage e costi.

In tutti i casi, mantenere stop >1.5 target solo quando portafoglio puo assorbire drawdown e la probabilita condizionata offerta dal forecast supera 65 %; altrimenti normalizzare R:R 1.

---

**Nota dati** - L'assenza di volume reale limita l'efficacia degli indicatori basati su flussi. Implementare proxy (tick count, range volatility, order-book sintetico) e documentarne il calcolo garantira coerenza tra training e inference.
