# src/forex_diffusion/utils/time_utils.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Tuple
import pandas as pd

WEEKEND_START_HOUR = 22  # Friday 22:00 UTC
WEEKEND_END_HOUR = 22    # Sunday 22:00 UTC

# Mapping of internal timeframe labels to pandas freq strings (used for expected timestamps)
TF_TO_PANDAS = {
    "tick": None,
    # minuti -> 'min'
    "1m": "1min",
    "2m": "2min",
    "3m": "3min",
    "4m": "4min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    # ore -> 'h'
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    # giorni
    "1d": "1D",
    "1D": "1D",
    "7d": "7D",
    "30d": "30D",
    # settimane
    "1w": "1W",
    "1W": "1W",
}

def ms_to_utc_dt(ms: int) -> datetime:
    """Convert milliseconds since epoch to timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

def utc_dt_to_ms(dt: datetime) -> int:
    """Convert UTC datetime to milliseconds since epoch."""
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)

def is_in_weekend_range(dt: datetime) -> bool:
    """
    Return True if dt (timezone-aware UTC) is inside the weekend off-trading window:
    from Friday 22:00 UTC inclusive until Sunday 22:00 UTC exclusive.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # weekday(): Monday=0 ... Sunday=6
    wd = dt.weekday()
    hour = dt.hour
    # Friday
    if wd == 4 and hour >= WEEKEND_START_HOUR:
        return True
    # Saturday (all day)
    if wd == 5:
        return True
    # Sunday before 22:00
    if wd == 6 and hour < WEEKEND_END_HOUR:
        return True
    return False

def split_range_by_month(start: datetime, end: datetime) -> List[Tuple[datetime, datetime]]:
    """
    Split [start, end] into monthly subranges for API requests.
    Returns list of (month_start, month_end) pairs in UTC (both timezone-aware).
    Used only for large historical backfills to avoid API overload.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if start >= end:
        return []

    ranges: List[Tuple[datetime, datetime]] = []
    cur = start

    while cur < end:
        # Calculate first day of next month
        if cur.month == 12:
            next_month = cur.replace(year=cur.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            next_month = cur.replace(month=cur.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)

        # Month end is the minimum of next_month or overall end
        month_end = min(next_month, end)
        ranges.append((cur, month_end))
        cur = month_end

    return ranges

def tf_to_pandas_freq(tf: str) -> str:
    """Return pandas frequency string for given timeframe label if available."""
    tf = tf.strip()
    if tf in TF_TO_PANDAS:
        return TF_TO_PANDAS[tf]
    # simple heuristics with modern aliases
    if tf.endswith("m"):
        return f"{int(tf[:-1])}min"
    if tf.endswith("h"):
        return f"{int(tf[:-1])}h"
    if tf.endswith("d"):
        return f"{int(tf[:-1])}D"
    raise ValueError(f"Unsupported timeframe: {tf}")

def generate_expected_period_ends(start_ms: int, end_ms: int, timeframe: str) -> List[int]:
    """
    Generate expected period-end timestamps (ms UTC) for timeframe between start_ms and end_ms.
    This uses pandas date_range with freq from tf_to_pandas_freq and labels aligned to period end.
    EXCLUDES weekend periods (Friday 22:00 UTC - Sunday 22:00 UTC) as forex market is closed.
    """
    if timeframe == "tick":
        # ticks are not regular bars; return empty (caller should handle ticks separately)
        return []
    start_dt = ms_to_utc_dt(start_ms)
    end_dt = ms_to_utc_dt(end_ms)
    freq = tf_to_pandas_freq(timeframe)
    # Use date_range with closed='right' to align period end timestamps
    idx = pd.date_range(start=start_dt, end=end_dt, freq=freq, tz="UTC")

    # Filter out weekend timestamps (market closed: Fri 22:00 - Sun 22:00 UTC)
    filtered = []
    for ts in idx:
        if not is_in_weekend_range(ts.to_pydatetime()):
            filtered.append(int(ts.timestamp() * 1000))

    return filtered
