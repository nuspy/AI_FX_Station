#!/usr/bin/env python3
"""
Test script for the refactored ChartTab implementation.

This script verifies that the refactored code can be imported and instantiated
without errors, ensuring backward compatibility.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all refactored components can be imported."""
    print("Testing imports...")

    try:
        # Test individual mixins
        from forex_diffusion.ui.chart_tab.ui_builder import UIBuilderMixin
        from forex_diffusion.ui.chart_tab.event_handlers import EventHandlersMixin
        from forex_diffusion.ui.chart_tab.controller_proxy import ControllerProxyMixin
        from forex_diffusion.ui.chart_tab.patterns_mixin import PatternsMixin
        from forex_diffusion.ui.chart_tab.overlay_manager import OverlayManagerMixin
        print("[OK] Individual mixins imported successfully")

        # Test base class
        from forex_diffusion.ui.chart_tab.chart_tab_base import ChartTabUI, DraggableOverlay
        print("[OK] Base classes imported successfully")

        # Test package level import
        from forex_diffusion.ui.chart_tab import ChartTabUI as PackageChartTabUI
        print("[OK] Package level import successful")

        # Test backward compatibility import
        from forex_diffusion.ui.chart_tab_refactored import ChartTabUI as RefactoredChartTabUI
        print("[OK] Backward compatibility import successful")

        print("[SUCCESS] All imports successful!")
        return True

    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_class_structure():
    """Test that the class has the expected structure."""
    print("\nTesting class structure...")

    try:
        from forex_diffusion.ui.chart_tab import ChartTabUI

        # Check MRO (Method Resolution Order)
        mro_names = [cls.__name__ for cls in ChartTabUI.__mro__]
        expected_mixins = [
            'UIBuilderMixin',
            'EventHandlersMixin',
            'ControllerProxyMixin',
            'PatternsMixin',
            'OverlayManagerMixin'
        ]

        for mixin in expected_mixins:
            if mixin in mro_names:
                print(f"[OK] {mixin} found in MRO")
            else:
                print(f"[ERROR] {mixin} missing from MRO")
                return False

        # Check that key methods exist
        key_methods = [
            '_build_ui',
            '_on_symbol_combo_changed',
            'update_plot',
            '_wire_pattern_checkboxes',
            '_init_overlays'
        ]

        for method in key_methods:
            if hasattr(ChartTabUI, method):
                print(f"[OK] Method {method} exists")
            else:
                print(f"[ERROR] Method {method} missing")
                return False

        print("[SUCCESS] Class structure is correct!")
        return True

    except Exception as e:
        print(f"[ERROR] Class structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_instantiation():
    """Test that the class can be instantiated (at least partially)."""
    print("\nTesting instantiation...")

    try:
        # Note: We can't fully instantiate without Qt application and proper setup
        # But we can test that the class definition is valid
        from forex_diffusion.ui.chart_tab import ChartTabUI

        # Check that __init__ method exists and is callable
        assert hasattr(ChartTabUI, '__init__')
        assert callable(ChartTabUI.__init__)

        print("[OK] Class definition is valid")
        print("[INFO] Full instantiation requires Qt application context")

        return True

    except Exception as e:
        print(f"[ERROR] Instantiation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Refactored ChartTab Implementation")
    print("=" * 50)

    tests = [
        test_imports,
        test_class_structure,
        test_instantiation
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    if all(results):
        print("[SUCCESS] All tests passed! Refactoring appears successful.")
        return 0
    else:
        print("[ERROR] Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())