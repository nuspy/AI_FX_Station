"""
Minimal ChartTab for debugging - No pyqtgraph, no pattern detection, just basic UI.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Signal


class MinimalChartTab(QWidget):
    """Minimal chart tab to test if pyqtgraph is the bottleneck."""
    
    forecastRequested = Signal(dict)
    
    def __init__(self, main_window, dom_service=None, order_flow_analyzer=None):
        super().__init__(main_window)
        
        layout = QVBoxLayout(self)
        
        # Just a simple label - no chart rendering
        label = QLabel("Chart Tab (Minimal - No Rendering)")
        label.setStyleSheet("font-size: 24px; padding: 50px;")
        layout.addWidget(label)
        
        self.symbol = "EUR/USD"
        self.timeframe = "1m"
        
        print("✓ MinimalChartTab created - No pyqtgraph rendering")
    
    def set_symbol_timeframe(self, db_service, symbol, timeframe):
        """Dummy method for compatibility."""
        self.symbol = symbol
        self.timeframe = timeframe
        print(f"✓ MinimalChartTab.set_symbol_timeframe({symbol}, {timeframe})")
    
    def on_forecast_ready(self, df, quantiles):
        """Dummy method for compatibility."""
        print("✓ MinimalChartTab.on_forecast_ready() called")
    
    def _handle_tick(self, tick):
        """Dummy method for compatibility."""
        pass
