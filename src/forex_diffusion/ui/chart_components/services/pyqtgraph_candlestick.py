"""
PyQtGraph Candlestick Chart Implementation
High-performance candlestick rendering using PyQtGraph
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore


class CandlestickItem(pg.GraphicsObject):
    """
    Candlestick chart item for PyQtGraph
    Efficient rendering of OHLC data
    """

    def __init__(self, data):
        """
        data: pandas DataFrame with columns: time, open, high, low, close
        """
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generate_picture()

    def generate_picture(self):
        """Pre-render the candlesticks for performance"""
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        # Convert datetime index to numeric
        if isinstance(self.data.index, pd.DatetimeIndex):
            x_data = np.arange(len(self.data))
        else:
            x_data = self.data.index.values

        w = 0.4  # candlestick width

        for i, (idx, row) in enumerate(self.data.iterrows()):
            try:
                open_price = float(row['open'])
                close_price = float(row['close'])
                high_price = float(row['high'])
                low_price = float(row['low'])

                x = x_data[i]

                # Color based on price movement
                if close_price > open_price:
                    # Bullish candle (green)
                    p.setPen(pg.mkPen('#26a69a'))
                    p.setBrush(pg.mkBrush('#26a69a'))
                else:
                    # Bearish candle (red)
                    p.setPen(pg.mkPen('#ef5350'))
                    p.setBrush(pg.mkBrush('#ef5350'))

                # Draw high-low line (wick)
                p.drawLine(QtCore.QPointF(x, low_price), QtCore.QPointF(x, high_price))

                # Draw open-close rectangle (body)
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                p.drawRect(QtCore.QRectF(x - w/2, body_bottom, w, body_height))

            except Exception:
                continue

        p.end()

    def paint(self, p, *args):
        """Draw the pre-rendered picture"""
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        """Return the bounding rectangle of the data"""
        return QtCore.QRectF(self.picture.boundingRect())


def add_candlestick(plot_widget, data):
    """
    Add candlestick chart to a PlotWidget

    Args:
        plot_widget: PyQtGraph PlotWidget or PlotItem
        data: pandas DataFrame with columns: open, high, low, close
    """
    candle_item = CandlestickItem(data)
    plot_widget.addItem(candle_item)

    # Set axis labels
    if isinstance(data.index, pd.DatetimeIndex):
        # Custom time axis
        axis = pg.DateAxisItem(orientation='bottom')
        plot_widget.setAxisItems({'bottom': axis})

    return candle_item