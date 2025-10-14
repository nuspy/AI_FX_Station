# Performance Fix: ChartTab UI Blocking Issue

## Problem Summary
Application was experiencing severe UI lag with 144+ threads and unresponsive window dragging.

## Root Cause
**ChartTab** was the bottleneck causing the UI to freeze. Identified through systematic debugging by disabling components.

## Investigation Process
1. **Thread count**: Discovered 144 active threads (normal is ~40)
2. **Disabled services sequentially**:
   - ✅ Aggregator services → No improvement
   - ✅ DOM aggregator → No improvement  
   - ✅ Chart timers → No improvement
   - ✅ Pattern detection threads → No improvement
   - ✅ **ChartTab entirely → FIXED!**

## Changes Made

### 1. Pattern Detection Thread Reduction
**File**: `src/forex_diffusion/ui/chart_components/services/patterns/patterns_service.py`

**Before**:
```python
optimal_threads = max(1, min(32, int(cpu_count * 0.75)))
```

**After**:
```python
optimal_threads = max(1, min(8, int(cpu_count * 0.5)))  # Reduced from 32 to 8 max
```

**Impact**: Reduced max worker threads from 32 to 8, preventing thread explosion.

### 2. ChartTab Temporarily Disabled
**File**: `src/forex_diffusion/ui/app.py`

- ChartTab creation commented out for debugging
- All references to `chart_tab` guarded with `if chart_tab is not None`
- Application runs smoothly without ChartTab

### 3. Services Re-enabled
All background services work fine and don't cause blocking:
- ✅ AggregatorService
- ✅ DOMAggregatorService  
- ✅ Pattern detection threads (with reduced count)
- ✅ Real-time scan threads

## Next Steps

### Immediate (Required)
1. **Profile ChartTab initialization** to find exact bottleneck
2. **Optimize chart rendering** - likely causes:
   - PyQtGraph plotting operations
   - Pattern overlay rendering
   - Large dataframe operations on main thread
   - Synchronous database queries during init

### Recommended Fixes
1. **Lazy load chart data** - don't load on init, load on first view
2. **Move heavy operations to worker threads**:
   - Pattern detection
   - Historical data loading
   - Overlay rendering
3. **Use QThreadPool** for async operations instead of blocking main thread
4. **Reduce pattern scan frequency** or batch updates
5. **Implement progressive rendering** - show chart first, add patterns later

### Performance Targets
- **Thread count**: Stay under 60 total threads
- **UI responsiveness**: Window dragging should be < 16ms frame time (60 FPS)
- **Startup time**: Under 5 seconds with ChartTab

## Configuration Changes

### Pattern Detection Threads
Can be configured in `configs/patterns.yaml`:
```yaml
performance:
  detection_threads: 4  # Override default (was auto-detecting to 32)
  parallel_realtime: true
  parallel_historical: true
```

## Testing
1. **Without ChartTab**: Application is fast and responsive ✅
2. **With ChartTab**: Needs optimization (next session)
3. **Thread count without ChartTab**: ~40-50 (acceptable)
4. **Thread count with ChartTab**: Was 144+ (excessive)

## Files Modified
- `src/forex_diffusion/ui/app.py` - ChartTab disabled, guards added
- `src/forex_diffusion/ui/chart_tab/event_handlers.py` - Timer guards (re-enabled)
- `src/forex_diffusion/ui/chart_components/services/patterns/patterns_service.py` - Thread reduction
- `src/forex_diffusion/providers/ctrader_provider.py` - Removed sleep(0.5), DOM storage optimization

## Resolution Status
- ✅ **Identified**: ChartTab is the bottleneck
- ✅ **Workaround**: Application runs fast without ChartTab
- ⏳ **Pending**: ChartTab optimization and re-enablement

---
**Date**: 2025-01-14  
**Session**: Debug DOM & Performance Investigation
