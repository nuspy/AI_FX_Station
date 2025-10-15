#!/usr/bin/env python
"""Check what data exists in the database"""

from forex_diffusion.services.db_service import DBService
from sqlalchemy import text
import datetime

db = DBService()
conn = db.engine.connect()

result = conn.execute(text(
    "SELECT timeframe, COUNT(*) as count, MIN(ts_utc) as first_ts, MAX(ts_utc) as last_ts "
    "FROM market_data_candles WHERE symbol='EUR/USD' "
    "GROUP BY timeframe ORDER BY timeframe"
))

rows = result.fetchall()

print('\nDati presenti nel DB per EUR/USD:')
print('=' * 100)
print(f'{"Timeframe":<12} {"Count":<10} {"First Candle":<30} {"Last Candle":<30}')
print('=' * 100)

for row in rows:
    tf, count, first_ts, last_ts = row
    first_dt = datetime.datetime.fromtimestamp(first_ts/1000, tz=datetime.timezone.utc) if first_ts else None
    last_dt = datetime.datetime.fromtimestamp(last_ts/1000, tz=datetime.timezone.utc) if last_ts else None

    first_str = first_dt.strftime("%Y-%m-%d %H:%M UTC") if first_dt else "N/A"
    last_str = last_dt.strftime("%Y-%m-%d %H:%M UTC") if last_dt else "N/A"

    print(f'{tf:<12} {count:<10} {first_str:<30} {last_str:<30}')

print('=' * 100)
conn.close()
