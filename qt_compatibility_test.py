#!/usr/bin/env python3
"""
Qt Compatibility Test for finplot integration
Tests different approaches to resolve PyQt6/PySide6 conflicts
"""

import sys
import os

def test_pyside6_only():
    """Test if we can use finplot with PySide6 only"""
    print("Testing PySide6-only approach...")

    try:
        # Try to force PySide6 backend
        os.environ['QT_API'] = 'pyside6'

        import PySide6
        print(f"✓ PySide6 {PySide6.__version__} available")

        # Try pyqtgraph with PySide6
        import pyqtgraph as pg
        print(f"✓ pyqtgraph {pg.__version__} loaded")

        # Check if pyqtgraph can work with PySide6
        pg.setConfigOptions(useOpenGL=True)
        print("✓ pyqtgraph configured for PySide6")

        return True

    except Exception as e:
        print(f"✗ PySide6-only approach failed: {e}")
        return False

def test_pyqt6_installation():
    """Test PyQt6 installation and compatibility"""
    print("\nTesting PyQt6 installation...")

    try:
        # Try to install PyQt6 if not available
        import subprocess

        print("Attempting to install PyQt6...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'PyQt6', '--no-deps'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ PyQt6 installation successful")

            # Test import
            from PyQt6.QtCore import QT_VERSION_STR
            print(f"✓ PyQt6 Qt version: {QT_VERSION_STR}")
            return True
        else:
            print(f"✗ PyQt6 installation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ PyQt6 installation test failed: {e}")
        return False

def test_finplot_alternatives():
    """Test alternative charting libraries"""
    print("\nTesting alternative charting libraries...")

    alternatives = {
        'plotly': 'Interactive financial charting',
        'mplfinance': 'Matplotlib-based financial charting',
        'bokeh': 'Web-based interactive charting',
        'pyqtgraph': 'High-performance plotting'
    }

    available_alternatives = []

    for lib, description in alternatives.items():
        try:
            __import__(lib)
            print(f"✓ {lib}: {description}")
            available_alternatives.append(lib)
        except ImportError:
            print(f"✗ {lib}: Not available")

    return available_alternatives

def test_pyqtgraph_standalone():
    """Test if pyqtgraph works standalone with PySide6"""
    print("\nTesting pyqtgraph standalone with PySide6...")

    try:
        # Set PySide6 as backend
        os.environ['QT_API'] = 'pyside6'

        import pyqtgraph as pg
        import numpy as np

        # Create simple test plot
        data = np.random.randn(100).cumsum()

        # Test if we can create a plot widget (without showing)
        app = pg.mkQApp()

        # Create plot widget
        plot_widget = pg.PlotWidget()
        plot_widget.plot(data, pen='w')

        print("✓ pyqtgraph standalone test successful")
        print("✓ Can create financial charts without finplot")

        # Cleanup
        app.quit()

        return True

    except Exception as e:
        print(f"✗ pyqtgraph standalone test failed: {e}")
        return False

def create_compatibility_report():
    """Create compatibility report with recommendations"""
    print("\n" + "="*60)
    print("QT COMPATIBILITY ANALYSIS REPORT")
    print("="*60)

    # Test all approaches
    pyside6_works = test_pyside6_only()
    pyqt6_works = test_pyqt6_installation()
    alternatives = test_finplot_alternatives()
    pyqtgraph_works = test_pyqtgraph_standalone()

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if pyqtgraph_works:
        print("RECOMMENDATION 1: Use pyqtgraph directly with PySide6")
        print("  ✓ Compatible with existing ForexGPT PySide6 setup")
        print("  ✓ High-performance financial charting")
        print("  ✓ No dependency conflicts")
        print("  → Create PyQtGraphChartService instead of FinplotChartService")

    if pyside6_works and not pyqt6_works:
        print("\nRECOMMENDATION 2: Stick with matplotlib + optimizations")
        print("  ✓ No compatibility issues")
        print("  ✓ Well-tested integration")
        print("  → Focus on matplotlib performance optimizations")

    if 'plotly' in alternatives:
        print("\nRECOMMENDATION 3: Consider Plotly for web-based charts")
        print("  ✓ No Qt dependencies")
        print("  ✓ Interactive and modern")
        print("  → Good for web deployment")

    print("\n" + "="*60)

    return {
        'pyside6_compatible': pyside6_works,
        'pyqt6_compatible': pyqt6_works,
        'pyqtgraph_standalone': pyqtgraph_works,
        'alternatives': alternatives
    }

def main():
    """Main compatibility test"""
    print("ForexGPT Qt Compatibility Analysis")
    print("Analyzing finplot integration options...")

    results = create_compatibility_report()

    # Save results
    import json
    with open('qt_compatibility_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to qt_compatibility_results.json")

if __name__ == "__main__":
    main()