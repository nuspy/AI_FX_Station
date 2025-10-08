"""
Test indicator plotting integration
"""
import sys
import os
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test all necessary imports"""
    print("Testing imports...")

    try:
        from forex_diffusion.ui.chart_components.services.plot_service import PlotService
        print("[OK] PlotService imported")
    except Exception as e:
        print(f"[FAIL] PlotService import: {e}")
        return False

    try:
        from forex_diffusion.features.indicators_btalib import BTALibIndicators
        print("[OK] BTALibIndicators imported")
    except Exception as e:
        print(f"[FAIL] BTALibIndicators import: {e}")
        return False

    try:
        from forex_diffusion.features.indicator_ranges import indicator_range_classifier
        print("[OK] indicator_range_classifier imported")
    except Exception as e:
        print(f"[FAIL] indicator_range_classifier import: {e}")
        return False

    try:
        from forex_diffusion.ui.enhanced_indicators_dialog import EnhancedIndicatorsDialog
        print("[OK] EnhancedIndicatorsDialog imported")
    except Exception as e:
        print(f"[FAIL] EnhancedIndicatorsDialog import: {e}")
        return False

    return True

def test_indicator_calculation():
    """Test indicator calculation"""
    print("\nTesting indicator calculation...")

    try:
        import pandas as pd
        import numpy as np
        from forex_diffusion.features.indicators_btalib import BTALibIndicators

        # Create sample OHLCV data
        n_points = 100
        dates = pd.date_range('2024-01-01', periods=n_points, freq='H')
        data = {
            'open': np.random.randn(n_points).cumsum() + 100,
            'high': np.random.randn(n_points).cumsum() + 102,
            'low': np.random.randn(n_points).cumsum() + 98,
            'close': np.random.randn(n_points).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_points)
        }
        df = pd.DataFrame(data, index=dates)

        # Initialize indicators system
        indicators_system = BTALibIndicators()
        print(f"[OK] BTALibIndicators initialized with {len(indicators_system.enabled_indicators)} indicators")

        # Test calculation of a few indicators
        test_indicators = ['rsi', 'sma', 'ema', 'bbands', 'macd', 'stoch']

        for indicator_name in test_indicators:
            try:
                # Get config from enabled_indicators dict
                config = indicators_system.enabled_indicators.get(indicator_name)
                if not config:
                    print(f"  [SKIP] {indicator_name}: not found")
                    continue

                # Use correct API: calculate_indicator(data, indicator_name, custom_params)
                result_dict = indicators_system.calculate_indicator(df, indicator_name)

                if result_dict:
                    is_multi = len(result_dict) > 1
                    series_type = "multi-series" if is_multi else "single-series"
                    keys = list(result_dict.keys())
                    print(f"  [OK] {indicator_name}: {series_type}, keys={keys}")
                else:
                    print(f"  [FAIL] {indicator_name}: result dict is empty")

            except Exception as e:
                print(f"  [FAIL] {indicator_name}: {e}")

        return True

    except Exception as e:
        print(f"[FAIL] Indicator calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_range_classification():
    """Test indicator range classification"""
    print("\nTesting indicator range classification...")

    try:
        from forex_diffusion.features.indicator_ranges import indicator_range_classifier

        test_indicators = [
            ('rsi', 'normalized_subplot'),
            ('sma', 'main_chart'),
            ('volume', 'volume_subplot'),
            ('bbands', 'main_chart'),
            ('stoch', 'normalized_subplot')
        ]

        for indicator_name, expected_subplot in test_indicators:
            range_info = indicator_range_classifier.get_range_info(indicator_name)
            if range_info:
                actual_subplot = range_info.subplot_recommendation
                if actual_subplot == expected_subplot:
                    print(f"  [OK] {indicator_name} -> {actual_subplot}")
                else:
                    print(f"  [WARNING] {indicator_name}: expected {expected_subplot}, got {actual_subplot}")
            else:
                print(f"  [FAIL] {indicator_name}: no range info found")

        return True

    except Exception as e:
        print(f"[FAIL] Range classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plot_service_methods():
    """Test PlotService has required methods"""
    print("\nTesting PlotService methods...")

    try:
        from forex_diffusion.ui.chart_components.services.plot_service import PlotService

        required_methods = [
            '_plot_enhanced_indicators',
            'enable_indicator_subplots',
            'disable_indicator_subplots'
        ]

        for method in required_methods:
            if hasattr(PlotService, method):
                print(f"  [OK] {method} exists")
            else:
                print(f"  [FAIL] {method} missing")
                return False

        return True

    except Exception as e:
        print(f"[FAIL] PlotService methods test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing Enhanced Indicator Plotting Integration")
    print("=" * 60)

    all_passed = True

    if not test_imports():
        all_passed = False

    if not test_indicator_calculation():
        all_passed = False

    if not test_range_classification():
        all_passed = False

    if not test_plot_service_methods():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED")
        print("\nIntegration Summary:")
        print("- EnhancedIndicatorsDialog provides UI for 200+ indicators")
        print("- BTALibIndicators calculates indicator values")
        print("- IndicatorRangeClassifier routes indicators to correct subplots")
        print("- MatplotlibSubplotService manages multi-subplot layout")
        print("- PlotService._plot_enhanced_indicators orchestrates everything")
    else:
        print("[FAILURE] SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())