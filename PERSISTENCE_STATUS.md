# Real-time Persistence & Backfill Status

## ✅ Persistence FUNZIONA

Verifica database confermata:
- Ultimi 20 candles salvati correttamente (14:12-14:30)
- OHLC valori realistici
- Timestamp corretti e sequenziali

## ❌ Gap Identificato: 3 AM - 11:06 AM

**Root cause**: App non in esecuzione o non connessa al provider

Dati presenti:
- 11:06-11:07: 2 candles
- 11:24-11:26: 3 candles  
- 14:12-14:30: 19 candles (real-time persistence attiva)

Gap:
- 03:00-11:06: 486 minuti MISSING
- 11:07-11:24: 17 minuti MISSING
- 11:26-14:12: 166 minuti MISSING

## 🔍 Backfill Behavior

**Problema riportato**: Backfill lanciato 2 volte scarica stessi dati

**Da verificare**:
1. Gap detection logging (aggiunto ma non ancora testato)
2. Query SQL seleziona dati esistenti correttamente?
3. Fuzzy matching tolerance (2000ms) troppo stretto?
4. Weekend detection skipping gaps corretti?

**Test richiesto**:
1. Avviare app con logging attivo
2. Lanciare backfill manuale
3. Verificare nei log:
   - "Gap detection for EUR/USD 1m: expecting X candles"
   - "Found Y existing candles in DB"
   - "Found Z missing timestamps before filtering"
   - "Found N gap(s) for EUR/USD 1m: total missing M candles"
4. Lanciare backfill seconda volta
5. Verificare se trova 0 gaps

## 📝 Action Items

1. ✅ Persistence implementata e funzionante
2. ✅ Gap detection logging aggiunto
3. ⏳ Test backfill con nuovi log (serve app attiva)
4. ⏳ Verifica fuzzy matching con timestamp reali
5. ⏳ Test doppio backfill per confermare skip

## 💡 Possibili Cause Backfill Duplic duplicato

1. **Query non trova dati esistenti**: Controllare se `ts_utc BETWEEN` funziona
2. **Timestamp mismatch**: Expected != Existing (millisecondi, boundary)
3. **Tolerance troppo stretta**: 2000ms potrebbe non matchare alcuni candles
4. **Upsert non committa**: Transaction rollback silenzioso?

## 🧪 Test Diagnostico

```sql
-- Verifica candles in range specifico
SELECT COUNT(*), MIN(ts_utc), MAX(ts_utc)
FROM market_data_candles
WHERE symbol='EUR/USD' AND timeframe='1m'
AND ts_utc BETWEEN 1760670000000 AND 1760711000000;

-- Verifica distribuzione temporale
SELECT 
    datetime(ts_utc/1000, 'unixepoch') as time,
    COUNT(*) as count
FROM market_data_candles
WHERE symbol='EUR/USD' AND timeframe='1m'
AND ts_utc >= 1760670000000
GROUP BY ts_utc/3600000  -- Raggruppa per ora
ORDER BY ts_utc;
```
