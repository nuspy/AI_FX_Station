# Forecast Pipeline Bug Fix

## Data: 2025-10-02

## Problemi Identificati

### 1. Training Pipeline ✅ CORRETTO
Il training usa il target corretto:
```python
y = (c.shift(-H) / c) - 1.0
```
Dove:
- `c` è il prezzo close corrente
- `H` è l'horizon in bars (es. 30 per 30m se TF=1m)
- `y` è il return percentuale: `(future_price / current_price) - 1`

**Esempio**: Se `c=1.1000` e dopo 30 bars `c[t+30]=1.1050`, allora:
```
y = (1.1050 / 1.1000) - 1 = 0.00454... ≈ 0.45% return
```

### 2. Multi-Horizon Scaling ❌ BUG CRITICO - FIXED

**Problema Originale** (linee 808, 817, 1289):
```python
scale_factor = bars / horizons_bars[0]
scaled_preds.append(base_pred * scale_factor)
```

**Perché era sbagliato:**
- Il modello predice un return per un horizon H (es. 30 bars)
- Il codice scalava linearmente: per 60 bars → `return * 2`, per 90 bars → `return * 3`
- Questo è matematicamente errato: un return dello 0.5% per 30 bars NON diventa 1.5% per 90 bars

**Esempio del bug:**
```
base_pred = 0.005 (0.5% return per 30 bars)
Horizon 30m (30 bars): 0.005 * 1 = 0.005 (0.5%) ✓
Horizon 1h (60 bars): 0.005 * 2 = 0.010 (1.0%) ✗ WRONG!
Horizon 2h (120 bars): 0.005 * 4 = 0.020 (2.0%) ✗ WRONG!
```

**Effetto visibile:**
- Forecasts con valori fuori scala (es. 10% per 3h)
- Segmenti retti che cambiano bruscamente agli horizon boundaries
- Tutti i forecast simili tra loro (usano lo stesso `base_pred * scale_factor`)

**Fix Applicato:**
```python
# Simple replication: the model predicted a return for horizon H
# We cannot scale it linearly - that's mathematically wrong
# Instead, we replicate it and let the compound conversion handle the trajectory
base_pred = preds[0]
preds = np.full(len(horizons_bars), base_pred)
```

### 3. Conversione Returns → Prezzi ✅ CORRETTO (ma chiarito)

**Codice Originale** (linea 832):
```python
p = last_close
for r in seq:
    p *= (1.0 + float(r))
    prices.append(p)
```

**Perché è corretto:**
Questo è **compound accumulation** ed è matematicamente corretto quando replichiamo lo stesso return:
- Se il modello predice "0.5% per ogni step"
- Primo step: `p₁ = last_close × 1.005`
- Secondo step: `p₂ = p₁ × 1.005 = last_close × 1.005²`
- Terzo step: `p₃ = p₂ × 1.005 = last_close × 1.005³`

Questo crea una traiettoria **esponenziale/geometrica**, non lineare, che è corretta per returns composti.

**Nota importante:**
Se avessimo predizioni diverse per ogni horizon (es. da un modello multi-output), allora ogni `r[i]` rappresenterebbe già il return totale dal prezzo iniziale a quell'horizon, e dovremmo usare:
```python
p = last_close * (1.0 + float(r))  # Non cumulativo
```

Ma con la replicazione, il compound è corretto.

## Risultato Atteso

Dopo il fix:

1. **Niente più scaling lineare** → I returns non vengono moltiplicati per numero di bars
2. **Traiettoria geometrica smooth** → Compound accumulation crea crescita esponenziale naturale
3. **Valori realistici** → 0.5% per 30 bars → ~1.5% per 90 bars (1.005³ ≈ 1.015)
4. **Forecasts più diversificati** → Ogni modello avrà la sua predizione base, non scalata linearmente

## Limitazioni Attuali

⚠️ **Nota**: Questo fix risolve il bug critico, ma il sistema ha ancora una limitazione:

Quando il modello predice un singolo return per un horizon H, e noi lo replichiamo per N horizons, stiamo assumendo che "il return atteso sia costante per ogni step". Questo è una **semplificazione**.

**Soluzione ideale futura:**
- Addestrare modelli **multi-output** che predicono N returns per N horizons
- Oppure fare **iterative forecasting** (autoregressive) dove ogni step usa il prezzo precedente
- Oppure usare sistemi di **scaling intelligente** che considerano volatility clustering, mean reversion, etc.

Il sistema ha già un "Enhanced Multi-Horizon Scaling" che fa questo, ma fallisce spesso e cade nel fallback. Bisognerà debuggare anche quello.

## File Modificati

1. `src/forex_diffusion/ui/workers/forecast_worker.py`:
   - Linea 803-809: Fallback replication per local inference
   - Linea 811-817: Main replication per local inference
   - Linea 1283-1291: Replication per parallel ensemble inference
   - Linea 825-847: Chiarito il comportamento del compound accumulation

## Testing

Per verificare il fix:
1. Fare un forecast su EUR/USD 1m con horizon 30 bars
2. Verificare che i prezzi siano realistici (±1% dal prezzo corrente)
3. Verificare che la traiettoria sia smooth senza salti agli horizon boundaries
4. Confrontare con forecast precedenti per vedere la differenza

## Credits

Fix identificato e implementato il 2025-10-02 dopo analisi approfondita della pipeline di training e inference.
