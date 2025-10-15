#!/usr/bin/env python
"""Clear ALL tables from the database"""

from forex_diffusion.services.db_service import DBService
from sqlalchemy import text, inspect
import os

db = DBService()

# Get database file path and size before
db_path = str(db.engine.url).replace('sqlite:///', '')
if os.path.exists(db_path):
    size_before = os.path.getsize(db_path) / (1024 * 1024)  # MB
    print(f"Database size before: {size_before:.2f} MB")

# Get all table names
insp = inspect(db.engine)
tables = insp.get_table_names()

# Tables to skip (system/migration tables)
skip_tables = ['alembic_version']

conn = db.engine.connect()

print(f"\nClearing {len(tables) - len(skip_tables)} tables...")
for table in tables:
    if table in skip_tables:
        print(f"Skipping {table}")
        continue

    try:
        print(f"Clearing {table}...", end=' ')
        result = conn.execute(text(f"DELETE FROM {table}"))
        conn.commit()
        count = result.rowcount
        print(f"OK ({count} rows deleted)")
    except Exception as e:
        print(f"ERROR: {e}")

# VACUUM to reclaim disk space
print("\nRunning VACUUM to reclaim disk space...")
conn.execute(text("VACUUM"))
conn.commit()
print("VACUUM completed")

conn.close()

# Check size after
if os.path.exists(db_path):
    size_after = os.path.getsize(db_path) / (1024 * 1024)  # MB
    print(f"\nDatabase size after: {size_after:.2f} MB")
    print(f"Space reclaimed: {size_before - size_after:.2f} MB")

print("\nDatabase cleared successfully!")
