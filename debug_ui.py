# debug_ui.py - helper to start UI and attach a console debug handler to the Refresh button
import sys
import importlib

sys.path.insert(0, ".")
try:
    m = importlib.import_module("forex_diffusion.ui.app")
    from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout

    app = QApplication([])
    main = QWidget()
    layout = QVBoxLayout(main)
    status = QLabel("Status: ready")
    layout.addWidget(status)

    res = m.setup_ui(
        main_window=main,
        layout=layout,
        menu_bar=None,
        viewer=QLabel("viewer"),
        status_label=status,
    )
    st = res.get("signals_tab")
    print("signals_tab present?", bool(st))

    if st:
        try:
            def _dbg():
                print("DEBUG: refresh_btn clicked (console)")

            # connect debug handler (will print to the PowerShell console)
            st.refresh_btn.clicked.connect(_dbg)
            print("Connected debug handler to refresh_btn.")
        except Exception as e:
            print("Failed to connect debug handler:", repr(e))
    else:
        print("No signals_tab returned by setup_ui")

    main.resize(1000, 700)
    main.show()
    print("GUI started - click 'Refresh Signals' now.")
    sys.exit(app.exec())
except Exception as e:
    print("setup_ui import/exec error:", repr(e))
