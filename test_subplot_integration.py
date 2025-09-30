"""
Quick test to verify matplotlib subplot service integration
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
    """Test all critical imports"""
    print("Testing imports...")

    try:
        from forex_diffusion.ui.chart_components.services.matplotlib_subplot_service import MatplotlibSubplotService
        print("[OK] MatplotlibSubplotService imported successfully")
    except Exception as e:
        print(f"[FAIL] MatplotlibSubplotService import failed: {e}")
        return False

    try:
        from forex_diffusion.ui.chart_components.services.plot_service import PlotService
        print("[OK] PlotService imported successfully")
    except Exception as e:
        print(f"[FAIL] PlotService import failed: {e}")
        return False

    try:
        from forex_diffusion.ui.enhanced_indicators_dialog import EnhancedIndicatorsDialog
        print("[OK] EnhancedIndicatorsDialog imported successfully")
    except Exception as e:
        print(f"[FAIL] EnhancedIndicatorsDialog import failed: {e}")
        return False

    return True

def test_subplot_service():
    """Test subplot service functionality"""
    print("\nTesting MatplotlibSubplotService functionality...")

    try:
        import matplotlib.pyplot as plt
        from forex_diffusion.ui.chart_components.services.matplotlib_subplot_service import MatplotlibSubplotService

        # Create figure
        fig = plt.figure(figsize=(12, 8))
        service = MatplotlibSubplotService(fig)
        print("[OK] MatplotlibSubplotService instance created")

        # Test subplot creation
        axes = service.create_subplots(has_normalized=True, has_volume=True)
        print(f"[OK] Subplots created: {list(axes.keys())}")

        # Test indicator classification
        test_indicators = ['rsi', 'sma', 'volume', 'bbands', 'stoch', 'ema']
        for ind in test_indicators:
            subplot = service.classify_indicator(ind)
            print(f"  {ind} -> {subplot}")

        print("[OK] Indicator classification working")

        plt.close(fig)
        return True

    except Exception as e:
        print(f"[FAIL] Subplot service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plot_service_integration():
    """Test PlotService subplot integration"""
    print("\nTesting PlotService integration...")

    try:
        # This is a basic check - full testing requires Qt environment
        from forex_diffusion.ui.chart_components.services.plot_service import PlotService

        # Check methods exist
        assert hasattr(PlotService, '__init__'), "Missing __init__"
        assert hasattr(PlotService, 'enable_indicator_subplots'), "Missing enable_indicator_subplots"
        assert hasattr(PlotService, 'disable_indicator_subplots'), "Missing disable_indicator_subplots"

        print("[OK] PlotService has all required subplot methods")
        return True

    except Exception as e:
        print(f"[FAIL] PlotService integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Testing Matplotlib Subplot Integration")
    print("=" * 60)

    all_passed = True

    if not test_imports():
        all_passed = False

    if not test_subplot_service():
        all_passed = False

    if not test_plot_service_integration():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED")
    else:
        print("[FAILURE] SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())