#!/usr/bin/env python3
"""
Run GUI helper: launches the UI module within package context and starts a Qt event loop.

Behavior:
 - Ensures the project's 'src' directory is on sys.path so `import forex_diffusion` works.
 - Temporarily avoids local shadowing when importing stdlib 'logging'.
 - Attempts to import forex_diffusion.ui.app and use its setup_ui to start a QApplication.
 - Falls back to runpy.run_module if no UI entrypoint is available.
"""
from __future__ import annotations

import runpy
import sys
import importlib
from pathlib import Path

# Ensure project 'src' directory is first on sys.path
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Avoid local shadowing when importing stdlib logging: temporarily filter sys.path
_orig_sys_path = sys.path[:]
try:
    def _is_project_path(p: str) -> bool:
        if not isinstance(p, str):
            return False
        pp = p.replace("\\", "/")
        return pp == str(SRC_DIR) or "/src/" in pp or pp.endswith("/src")
    sys.path = [p for p in _orig_sys_path if not _is_project_path(p)]
    std_logging = importlib.import_module("logging")
    sys.modules["logging"] = std_logging
finally:
    sys.path = _orig_sys_path

# Try to import the UI app module and launch the GUI via setup_ui if available.
try:
    mod = importlib.import_module("forex_diffusion.ui.app")

    # If module exposes a callable 'main', call it and exit.
    maybe_main = getattr(mod, "main", None)
    if callable(maybe_main):
        maybe_main()
        raise SystemExit(0)

    # If module exposes setup_ui, attempt to start a minimal QApplication and call it.
    setup_ui = getattr(mod, "setup_ui", None)
    if callable(setup_ui):
        try:
            # Import UI helper types
            ViewerWidget = getattr(importlib.import_module("forex_diffusion.ui.viewer"), "ViewerWidget", None)
            MainMenuBar = getattr(importlib.import_module("forex_diffusion.ui.menus"), "MainMenuBar", None)
        except Exception:
            ViewerWidget = None
            MainMenuBar = None

        # Import PySide6 and build a minimal main window
        try:
            from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QStatusBar
        except Exception as e:
            print("PySide6 not available or failed to import:", e)
            # Fallback to executing the module as a script to surface errors
            runpy.run_module("forex_diffusion.ui.app", run_name="__main__")
            raise SystemExit(1)

        app = QApplication(sys.argv)
        main_win = QMainWindow()
        central = QWidget()
        layout = QVBoxLayout(central)

        # Viewer placeholder
        if ViewerWidget is not None:
            viewer = ViewerWidget()
        else:
            viewer = QLabel("Viewer (placeholder)")

        # Menu bar placeholder
        menu_bar = MainMenuBar(main_win) if MainMenuBar is not None else None
        if menu_bar is not None:
            main_win.setMenuBar(menu_bar)

        status = QStatusBar(main_win)
        main_win.setStatusBar(status)
        status_label = QLabel("Status: ready")
        status.addWidget(status_label)

        # Do not add the viewer yet — call setup_ui first so it can add SignalsTab or modify the layout.
        main_win.setCentralWidget(central)
        central.setLayout(layout)  # explicit layout assignment

        # Call setup_ui to wire the rest (SignalsTab, DBWriter, controller, etc.)
        try:
            setup_result = setup_ui(main_window=main_win, layout=layout, menu_bar=menu_bar, viewer=viewer, status_label=status_label)
        except Exception as e:
            print("setup_ui raised an exception:", e)
            # Fallback: run as script to show error traceback
            runpy.run_module("forex_diffusion.ui.app", run_name="__main__")
            raise SystemExit(1)

        # If setup_ui didn't add any visible widgets, add the viewer or a helpful label as fallback.
        try:
            # Count direct widgets in the layout
            count_widgets = layout.count()
            if count_widgets == 0:
                # Nothing added -> add viewer or placeholder
                if hasattr(viewer, "update_plot") or not isinstance(viewer, type(QLabel())):
                    layout.addWidget(viewer)
                else:
                    placeholder = QLabel("UI initialized but no widgets added. Viewer placeholder.")
                    placeholder.setMinimumHeight(200)
                    layout.addWidget(placeholder)
            else:
                # Ensure viewer is present alongside SignalsTab if appropriate
                found_viewer = False
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    w = item.widget() if item is not None else None
                    if w is viewer:
                        found_viewer = True
                        break
                if not found_viewer:
                    # add viewer after existing widgets
                    layout.addWidget(viewer)
        except Exception:
            # best-effort: still show main window
            try:
                layout.addWidget(viewer)
            except Exception:
                pass

        # Force resize and show
        main_win.resize(1000, 700)
        main_win.show()
        # Ensure UI repaints/update
        try:
            central.update()
            main_win.repaint()
        except Exception:
            pass

        sys.exit(app.exec())

    # No setup_ui/main found: fallback to running module as script
    runpy.run_module("forex_diffusion.ui.app", run_name="__main__")

except Exception as exc:
    # Last-resort: provide clearer guidance for YAML parsing errors then re-raise
    try:
        import yaml as _yaml  # lazy import to avoid extra dependency if not present
        cur = exc
        yaml_found = False
        # Walk the exception chain to detect any nested YAML errors
        while cur is not None:
            if isinstance(cur, _yaml.YAMLError):
                yaml_found = True
                break
            cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    except Exception:
        yaml_found = False

    if yaml_found:
        print("run_gui: failed due to a YAML parsing error while loading configuration.")
        print("Error:", exc)
        print("Suggested actions:")
        print("  - Open configs/default.yaml and inspect the line/column shown in the traceback.")
        print("  - Check for missing ':' after keys, incorrect indentation or malformed lists.")
        print("  - Validate the file with an online YAML validator or 'yamllint'.")
        # Re-raise to preserve original traceback for debugging
        raise

    # Default fallback: print and re-raise for other error types
    print("run_gui: failed to launch UI:", exc)
    raise
