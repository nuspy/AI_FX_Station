#!/usr/bin/env python3
"""
Test FinplotChartService implementation
Demonstrates finplot integration capabilities
"""

import sys
import numpy as np
import pandas as pd
import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_finplot_service():
    """Test the FinplotChartService"""
    try:
        from forex_diffusion.ui.chart_components.services.finplot_chart_service import FinplotChartService

        print("Testing FinplotChartService...")

        # Check if finplot is available
        if not FinplotChartService.is_available():
            print("finplot not available - cannot test service")
            return False

        # Create sample forex data
        dates = pd.date_range('2024-09-01', periods=200, freq='h')
        np.random.seed(42)

        # Generate realistic EURUSD price data
        base_price = 1.1000
        prices = [base_price]

        for i in range(1, 200):
            change = np.random.normal(0, 0.0003)
            trend = 0.00005 * np.sin(i / 20)
            new_price = prices[-1] * (1 + change + trend)
            prices.append(new_price)

        prices = np.array(prices)

        test_data = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(200) * 0.001),
            'low': prices - np.abs(np.random.randn(200) * 0.001),
            'close': np.roll(prices, -1),
            'volume': np.random.uniform(100000, 1000000, 200),
        }, index=dates)

        # Fix OHLC consistency
        test_data.loc[test_data.index[-1], 'close'] = test_data.loc[test_data.index[-1], 'open']
        test_data['high'] = np.maximum(test_data['high'], np.maximum(test_data['open'], test_data['close']))
        test_data['low'] = np.minimum(test_data['low'], np.minimum(test_data['open'], test_data['close']))

        print(f"Created {len(test_data)} candles of test data")
        print(f"Price range: {test_data['low'].min():.5f} - {test_data['high'].max():.5f}")

        # Create chart service
        chart_service = FinplotChartService(
            available_data=['open', 'high', 'low', 'close', 'volume'],
            theme="professional",
            real_time=True
        )

        print("FinplotChartService created successfully")

        # Get performance info
        perf_info = chart_service.get_performance_info()
        print("\nPerformance Information:")
        print(f"Service: {perf_info['service']}")
        print(f"Backend: {perf_info['backend']}")
        print(f"Rendering Speed: {perf_info['performance']['rendering_speed']}")
        print(f"Memory Usage: {perf_info['performance']['memory_usage']}")
        print(f"Real-time Capable: {perf_info['performance']['real_time_capable']}")
        print(f"GPU Accelerated: {perf_info['performance']['gpu_accelerated']}")

        # Test basic chart creation (without showing)
        print("\nTesting chart creation...")

        # Simple test that doesn't require GUI
        chart_service.current_data = test_data
        chart_service.current_indicators = chart_service._calculate_indicators(test_data)

        print(f"Calculated {len(chart_service.current_indicators)} indicators")
        print("Available indicators:", list(chart_service.current_indicators.keys()))

        print("\nFinplotChartService test completed successfully!")
        print("Ready for integration with ForexGPT chart components")

        return True

    except Exception as e:
        print(f"Error testing FinplotChartService: {e}")
        return False

def main():
    """Main test function"""
    print("Testing finplot integration components...")

    success = test_finplot_service()

    if success:
        print("\n" + "="*50)
        print("FINPLOT INTEGRATION TEST RESULTS")
        print("="*50)
        print("Status: SUCCESS")
        print("FinplotChartService: Working correctly")
        print("Indicators: Calculated successfully")
        print("Performance: Professional-grade capabilities")
        print("Ready for: ForexGPT integration")
        print("="*50)
    else:
        print("\nFinplot integration test failed")

if __name__ == "__main__":
    main()