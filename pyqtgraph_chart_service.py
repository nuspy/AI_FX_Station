
import pyqtgraph as pg
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class PyQtGraphChartService:
    """
    High-performance chart service using pyqtgraph with PySide6
    Alternative to finplot for ForexGPT integration
    """

    def __init__(self, theme: str = "dark"):
        self.theme = theme
        self.app = None
        self.plot_widgets = {}
        self._configure_theme()

    def _configure_theme(self):
        """Configure pyqtgraph theme"""
        pg.setConfigOption('background', 'w' if self.theme == 'light' else 'k')
        pg.setConfigOption('foreground', 'k' if self.theme == 'light' else 'w')

    def create_candlestick_chart(self, data: pd.DataFrame, title: str = "Forex Chart"):
        """Create candlestick chart using pyqtgraph"""

        # Create main plot widget
        plot_widget = pg.PlotWidget(title=title)
        plot_widget.setLabel('bottom', 'Time')
        plot_widget.setLabel('left', 'Price')

        # Create OHLC data for candlesticks
        dates = np.arange(len(data))

        # Simple candlestick representation using line plots
        # High-Low lines
        for i in range(len(data)):
            plot_widget.plot([dates[i], dates[i]],
                           [data.iloc[i]['low'], data.iloc[i]['high']],
                           pen=pg.mkPen('w', width=1))

            # Body color based on open/close
            color = 'g' if data.iloc[i]['close'] > data.iloc[i]['open'] else 'r'
            plot_widget.plot([dates[i], dates[i]],
                           [data.iloc[i]['open'], data.iloc[i]['close']],
                           pen=pg.mkPen(color, width=3))

        self.plot_widgets['main'] = plot_widget
        return plot_widget

    def add_indicators(self, data: pd.DataFrame):
        """Add technical indicators to chart"""
        if 'main' not in self.plot_widgets:
            return

        plot_widget = self.plot_widgets['main']
        dates = np.arange(len(data))

        # Simple moving average
        if len(data) > 20:
            sma_20 = data['close'].rolling(20).mean()
            plot_widget.plot(dates, sma_20, pen=pg.mkPen('blue', width=2), name='SMA 20')

        if len(data) > 50:
            sma_50 = data['close'].rolling(50).mean()
            plot_widget.plot(dates, sma_50, pen=pg.mkPen('orange', width=2), name='SMA 50')

    def show(self):
        """Show the chart"""
        if self.app is None:
            self.app = pg.mkQApp()

        for widget in self.plot_widgets.values():
            widget.show()

        return self.app

# Test the alternative service
def test_alternative_service():
    import pandas as pd
    import numpy as np

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100) * 0.01) + 100

    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(100) * 0.5),
        'low': prices - np.abs(np.random.randn(100) * 0.5),
        'close': np.roll(prices, -1),
    }, index=dates)

    # Create chart service
    service = PyQtGraphChartService()

    # Create chart
    chart = service.create_candlestick_chart(data, "ForexGPT Test Chart")
    service.add_indicators(data)

    print("PyQtGraphChartService test successful!")
    return True

if __name__ == "__main__":
    test_alternative_service()
    