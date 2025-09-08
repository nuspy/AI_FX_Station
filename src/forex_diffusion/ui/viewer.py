from __future__ import annotations

from typing import Any
from PySide6.QtWidgets import QWidget


class ViewerWidget(QWidget):
    """
    Minimal ViewerWidget used by the UI. Provides update_plot(df, q) method expected by controller.
    Implementation is intentionally minimal (no plotting) and can be extended by the real app.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

    def update_plot(self, df: Any, q: Any) -> None:
        """
        Update the displayed plot with dataframe df and quantiles q.
        Placeholder: real implementation should render the chart.
        """
        # TODO: implement plotting (this is a no-op placeholder)
        return
