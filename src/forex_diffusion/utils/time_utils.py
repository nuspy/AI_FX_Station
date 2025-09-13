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
    "1m": "1T",
    "2m": "2T",
    "3m": "3T",
    "4m": "4T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "60m": "60T",
    "1h": "60T",
    "2h": "120T",
    "4h": "240T",
    "1d": "1D",
    "1D": "1D",
    "7d": "7D",
    "30d": "30D",
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

def split_range_avoid_weekend(start: datetime, end: datetime) -> List[Tuple[datetime, datetime]]:
    """
    Split [start, end] into subranges that do not overlap the weekend off-trading window.
    Returns list of (substart, subend) pairs in UTC (both timezone-aware).
    subranges are inclusive of start and end (caller should adjust for provider semantics).
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if start >= end:
        return []

    ranges: List[Tuple[datetime, datetime]] = []
    cur = start

    # helper to compute next weekend start given a datetime
    def next_weekend_start(dt: datetime) -> datetime:
        # find upcoming Friday of week of dt
        # compute days until Friday (weekday 4)
        days_ahead = (4 - dt.weekday()) % 7
        friday = (dt + timedelta(days=days_ahead)).replace(hour=WEEKEND_START_HOUR, minute=0, second=0, microsecond=0)
        # if dt already past this week's friday at 22:00, take next week's friday
        if friday <= dt:
            friday = friday + timedelta(days=7)
        return friday

    # iterate splitting out weekend periods
    while cur < end:
        # if current is inside a weekend, advance to weekend end
        if is_in_weekend_range(cur):
            # compute the Sunday 22:00 corresponding to the current weekend
            # find the last Friday before or equal cur
            # find Sunday of that weekend
            # compute days to Sunday
            # find the Sunday date for the weekend containing cur
            # approach: move to next day until it's Sunday and hour==22
            # simpler: find the most recent Friday 22:00 <= cur, then add 48 hours
            # find Friday of current week
            days_back = (cur.weekday() - 4) % 7
            friday = (cur - timedelta(days=days_back)).replace(hour=WEEKEND_START_HOUR, minute=0, second=0, microsecond=0)
            sunday22 = friday + timedelta(days=2, hours=0) + timedelta(hours=WEEKEND_END_HOUR - WEEKEND_START_HOUR)
            # sunday22 is Friday22 + 48h + (end_hour-start_hour) => effectively Sunday 22:00
            if sunday22 <= cur:
                # move to next weekend end
                sunday22 = sunday22 + timedelta(days=7)
            cur = min(sunday22, end)
            # continue loop; do not append ranges while inside weekend
            continue
        # current is not inside weekend -> find next weekend start
        nw_start = next_weekend_start(cur)
        seg_end = min(nw_start, end)
        ranges.append((cur, seg_end))
        cur = seg_end
        # if cur equals nw_start, skip weekend by moving cur to sunday22
        if cur >= nw_start and cur < end:
            # compute corresponding sunday22
            sunday22 = nw_start + timedelta(days=2, hours=(WEEKEND_END_HOUR - WEEKEND_START_HOUR))
            cur = min(sunday22, end)
    return ranges

def tf_to_pandas_freq(tf: str) -> str:
    """Return pandas frequency string for given timeframe label if available."""
    tf = tf.strip()
    if tf in TF_TO_PANDAS:
        return TF_TO_PANDAS[tf]
    # try simple heuristics
    if tf.endswith("m"):
        return f"{int(tf[:-1])}T"
    if tf.endswith("h"):
        return f"{int(tf[:-1])}H"
    if tf.endswith("d"):
        return f"{int(tf[:-1])}D"
    raise ValueError(f"Unsupported timeframe: {tf}")

def generate_expected_period_ends(start_ms: int, end_ms: int, timeframe: str) -> List[int]:
    """
    Generate expected period-end timestamps (ms UTC) for timeframe between start_ms and end_ms.
    This uses pandas date_range with freq from tf_to_pandas_freq and labels aligned to period end.
    """
    if timeframe == "tick":
        # ticks are not regular bars; return empty (caller should handle ticks separately)
        return []
    start_dt = ms_to_utc_dt(start_ms)
    end_dt = ms_to_utc_dt(end_ms)
    freq = tf_to_pandas_freq(timeframe)
    # Use date_range with closed='right' to align period end timestamps
    idx = pd.date_range(start=start_dt, end=end_dt, freq=freq, tz="UTC")
    # convert to ms
    out = [(int(x.timestamp() * 1000)) for x in idx]
    return out
