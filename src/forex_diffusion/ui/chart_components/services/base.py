from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forex_diffusion.ui.chart_tab import ChartTab
    from forex_diffusion.ui.chart_components.controllers.chart_controller import ChartTabController


class ChartServiceBase:
    """Base class for ChartTab services providing access to the view and controller."""

    _OWN_ATTRS = {"view", "controller"}

    def __init__(self, view: "ChartTab", controller: "ChartTabController") -> None:
        object.__setattr__(self, "view", view)
        object.__setattr__(self, "controller", controller)

    def __getattr__(self, name: str):
        return getattr(self.view, name)

    def __setattr__(self, name: str, value) -> None:
        if name in self._OWN_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self.view, name, value)

    @property
    def app_controller(self):
        """Return the main UI controller exposed by the view."""
        return getattr(self.controller, "app_controller", None)
