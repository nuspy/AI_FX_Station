# Smart Cache Implementation Spec

## Obiettivo
Cache dinamica per caricare velocemente solo i dati visualizzati + buffer di 500 candele a sx/dx

## Design

### 1. Attributi Cache (in DataService)
```python
self._cache_df = None  # DataFrame completo caricato
self._cache_symbol = None  # Simbolo cache
self._cache_timeframe = None  # Timeframe cache
self._cache_range_candles = (start_idx, end_idx)  # Range di indici candele in cache
```

### 2. Logica di Loading
Quando l'utente visualizza il grafico:
1. Determina viewport corrente (quante candele visibili)
2. Calcola range richiesto: `viewport + 500 left + 500 right`
3. Controlla cache:
   - Se cache copre range → usa cache
   - Se cache parziale → load incrementale
   - Se no cache → load completo con buffer

### 3. Unloading Automatico
Quando range esce da `cache ± 1000 candele`:
- Taglia DataFrame per mantenere solo `current_range ± 500`
- Aggiorna `_cache_range_candles`

### 4. Implementazione
Modifiche in `data_service.py`:
- `_load_candles_from_db()`: Aggiungi logica buffer
- `_on_timeframe_changed()`: Reset cache
- `_on_symbol_changed()`: Reset cache
- Aggiungi `_trim_cache()` per pulizia
