#!/usr/bin/env python
"""Test Tiingo API historical data access"""

import httpx
import os
from datetime import datetime, timedelta

api_key = os.environ.get("TIINGO_API_KEY") or "d867b4314010495a5fa40593610eb3deae5e2dcd"
if not api_key:
    print("ERROR: TIINGO_API_KEY not set")
    exit(1)

print(f"Using API key: {api_key[:10]}...")

# Test different date ranges
test_dates = [
    ("2024-01-01", "2024-01-31", "Recent (2024)"),
    ("2020-01-01", "2020-01-31", "5 years ago (2020)"),
    ("2015-01-01", "2015-01-31", "10 years ago (2015)"),
    ("2010-01-01", "2010-01-31", "15 years ago (2010)"),
    ("2000-01-01", "2000-01-31", "25 years ago (2000)"),
]

base_url = "https://api.tiingo.com/tiingo/fx/eurusd/prices"

for start, end, label in test_dates:
    print(f"\n{'='*60}")
    print(f"Testing {label}: {start} to {end}")
    print('='*60)

    params = {
        "startDate": start,
        "endDate": end,
        "resampleFreq": "1day",
        "format": "json"
    }

    headers = {"Authorization": f"Token {api_key}"}

    try:
        response = httpx.get(base_url, params=params, headers=headers, timeout=10.0)

        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Retrieved {len(data)} data points")
            if data:
                print(f"  First date: {data[0].get('date', 'N/A')}")
                print(f"  Last date: {data[-1].get('date', 'N/A')}")
        else:
            print(f"FAILED: HTTP {response.status_code}")
            print(f"  Response: {response.text[:200]}")

    except Exception as e:
        print(f"ERROR: {e}")

print(f"\n{'='*60}")
print("Test completed")
