#!/usr/bin/env python3
"""
Simple finplot test for ForexGPT evaluation
"""

import numpy as np
import pandas as pd

try:
    import finplot as fplt
    print("finplot imported successfully")
except ImportError as e:
    print(f"finplot not available: {e}")
    exit(1)

def main():
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    prices = np.cumsum(np.random.randn(100) * 0.01) + 100

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.randn(100) * 0.5),
        'low': prices - np.abs(np.random.randn(100) * 0.5),
        'close': np.roll(prices, -1),
        'volume': np.random.uniform(1000, 5000, 100)
    })

    df.set_index('timestamp', inplace=True)

    print(f"Created {len(df)} candles")
    print("Price range:", df['low'].min(), "-", df['high'].max())

    # Create finplot chart
    print("Creating finplot chart...")

    try:
        # Simple candlestick chart test
        fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']])
        fplt.plot(df['close'].rolling(10).mean(), legend='SMA 10', color='blue')

        print("Chart created successfully")
        print("Features demonstrated:")
        print("- OHLC candlestick rendering")
        print("- Moving average overlay")
        print("- Professional appearance")

        # Show chart (this will open a window)
        print("Opening chart window...")
        fplt.show()
        print("Evaluation complete")

        return True

    except Exception as e:
        print(f"Error creating chart: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("finplot test successful - library is working correctly")
    else:
        print("finplot test failed - issues detected")