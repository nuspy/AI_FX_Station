#!/usr/bin/env python
"""Debug gap detection to see why it's finding gaps where data exists"""

from forex_diffusion.services.db_service import DBService
from forex_diffusion.utils import time_utils
from sqlalchemy import text
from datetime import datetime, timezone, timedelta

db = DBService()

symbol = "EUR/USD"
timeframe = "1m"

# Get actual date range from DB
with db.engine.connect() as conn:
    q = text("SELECT MIN(ts_utc), MAX(ts_utc), COUNT(*) FROM market_data_candles WHERE symbol = :symbol AND timeframe = :timeframe")
    result = conn.execute(q, {"symbol": symbol, "timeframe": timeframe}).fetchone()
    min_ts, max_ts, count = result

print(f"\n=== Database Stats for {symbol} {timeframe} ===")
print(f"First candle: {datetime.fromtimestamp(min_ts/1000, tz=timezone.utc)}")
print(f"Last candle:  {datetime.fromtimestamp(max_ts/1000, tz=timezone.utc)}")
print(f"Total count:  {count}")

# Pick a specific date that was being re-requested: 2025-09-19
test_date = datetime(2025, 9, 19, 0, 0, 0, tzinfo=timezone.utc)
start_ms = int(test_date.timestamp() * 1000)
end_ms = start_ms + (24 * 60 * 60 * 1000)  # +24 hours

print(f"\n=== Testing date: {test_date.date()} ===")

# Get expected timestamps for that day
expected = time_utils.generate_expected_period_ends(start_ms, end_ms, timeframe)
print(f"Expected timestamps generated: {len(expected)}")

if expected:
    print(f"First expected: {datetime.fromtimestamp(expected[0]/1000, tz=timezone.utc)}")
    print(f"Last expected:  {datetime.fromtimestamp(expected[-1]/1000, tz=timezone.utc)}")

# Get actual timestamps from DB for that day
with db.engine.connect() as conn:
    q = text("SELECT ts_utc FROM market_data_candles WHERE symbol = :symbol AND timeframe = :timeframe AND ts_utc BETWEEN :s AND :e ORDER BY ts_utc")
    rows = conn.execute(q, {"symbol": symbol, "timeframe": timeframe, "s": start_ms, "e": end_ms}).fetchall()
    existing = [int(r[0]) for r in rows]

print(f"Existing timestamps in DB: {len(existing)}")

if existing:
    print(f"First existing: {datetime.fromtimestamp(existing[0]/1000, tz=timezone.utc)}")
    print(f"Last existing:  {datetime.fromtimestamp(existing[-1]/1000, tz=timezone.utc)}")

# Find missing
expected_set = set(expected)
existing_set = set(existing)
missing = sorted(list(expected_set - existing_set))

print(f"\n=== Gap Analysis ===")
print(f"Missing timestamps: {len(missing)}")

if missing:
    print(f"\nFirst 10 missing timestamps:")
    for i, ts in enumerate(missing[:10]):
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
        print(f"  {i+1}. {dt} ({dt.strftime('%A')})")

    print(f"\nLast 10 missing timestamps:")
    for i, ts in enumerate(missing[-10:]):
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
        print(f"  {i+1}. {dt} ({dt.strftime('%A')})")

# Check if any of the "missing" are in weekend
weekend_count = 0
for ts in missing[:100]:  # Check first 100
    dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
    if time_utils.is_in_weekend_range(dt):
        weekend_count += 1

if missing:
    print(f"\nWeekend timestamps in 'missing' (first 100): {weekend_count}")

print("\n=== Sample of existing data (first 10) ===")
for ts in existing[:10]:
    dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
    print(f"  {dt}")
