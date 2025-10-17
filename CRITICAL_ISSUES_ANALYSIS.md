# Critical Issues Analysis

## Issue 1: Real-time candles NOT saved to DB
**Symptom**: Gap from 3 AM today to present  
**Root Cause**: `_on_tick_main` only updates `_last_df` (memory), never writes to DB  
**Impact**: Data lost on restart, backfill can't fill today's data  
**Fix**: Add DB persistence for real-time candles

## Issue 2: Backfill doesn't find gaps
**Symptom**: No "Found X gap(s)" log message  
**Root Cause**: `_find_missing_intervals` may not detect gaps correctly  
**Debug**: Check if expected timestamps are generated correctly  
**Fix**: Add logging to gap detection

## Issue 3: Cache not reloaded after backfill
**Symptom**: UI shows old data after backfill  
**Root Cause**: `_schedule_view_reload()` might not reload from DB  
**Fix**: Force reload from DB after backfill completion

## Issue 4: Smart cache limits scrolling
**Symptom**: Can't scroll past initial range  
**Root Cause**: `start_ms` calculated once, no dynamic loading on scroll  
**Fix**: Implement viewport detection and dynamic loading

## Priority Order
1. **Issue 1** (CRITICAL): No DB persistence = data loss
2. **Issue 2** (HIGH): Backfill not finding gaps
3. **Issue 3** (MEDIUM): UI refresh after backfill
4. **Issue 4** (MEDIUM): Unlimited scroll implementation
