"""
Sentiment Panel

Displays real-time market sentiment metrics including long/short positioning,
sentiment strength, contrarian signals, and sentiment-based trading alerts.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QGroupBox, QProgressBar, QComboBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger


class SentimentPanel(QWidget):
    """
    Sentiment monitoring panel showing:
    - Long/short positioning percentages
    - Sentiment strength and direction
    - Contrarian signals
    - Sentiment-based trading alerts
    """

    def __init__(self, parent=None, sentiment_service=None):
        super().__init__(parent)
        self.current_metrics: Dict[str, Any] = {}
        self.sentiment_service = sentiment_service
        self.current_symbol = 'EURUSD'
        
        # Cache to prevent redundant UI updates
        self._metrics_cache: Dict[str, Any] = {
            "sentiment": None,
            "confidence": None,
            "ratio": None,
            "total_traders": None,
            "long_pct": None,
            "short_pct": None,
            "contrarian_signal": None,
            "contrarian_state": None,
        }

        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        
        # Set minimum height to ensure visibility in splitter
        self.setMinimumHeight(250)
        self.setMinimumWidth(200)
        
        # Set styling with visible border for debugging
        self.setStyleSheet("""
            QWidget#sentimentPanel {
                background-color: #2b2b2b;
                border: 2px solid #0078d7;
                border-radius: 4px;
            }
        """)
        self.setObjectName("sentimentPanel")

        # Title
        title_layout = QHBoxLayout()
        title = QLabel("Market Sentiment Analysis")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        title_layout.addWidget(title)

        # Symbol selector
        title_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'])
        self.symbol_combo.currentTextChanged.connect(self.on_symbol_changed)
        title_layout.addWidget(self.symbol_combo)

        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Current Sentiment Section
        sentiment_group = self._create_sentiment_section()
        layout.addWidget(sentiment_group)

        # Positioning Section
        positioning_group = self._create_positioning_section()
        layout.addWidget(positioning_group)

        # Contrarian Signals Section
        contrarian_group = self._create_contrarian_section()
        layout.addWidget(contrarian_group)

        # Alerts Section
        alerts_group = self._create_alerts_section()
        layout.addWidget(alerts_group)

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds

    def _create_sentiment_section(self) -> QGroupBox:
        """Create current sentiment display"""
        group = QGroupBox("Current Market Sentiment")
        layout = QVBoxLayout()

        # Sentiment Label and Confidence
        sentiment_layout = QHBoxLayout()

        sentiment_layout.addWidget(QLabel("Sentiment:"))
        self.sentiment_label = QLabel("NEUTRAL")
        self.sentiment_label.setStyleSheet(
            "font-weight: bold; font-size: 16px; "
            "padding: 5px 10px; border-radius: 3px; background-color: #E0E0E0;"
        )
        sentiment_layout.addWidget(self.sentiment_label)

        sentiment_layout.addWidget(QLabel("Confidence:"))
        self.confidence_label = QLabel("--")
        self.confidence_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        sentiment_layout.addWidget(self.confidence_label)

        sentiment_layout.addStretch()
        layout.addLayout(sentiment_layout)

        # Sentiment Ratio
        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("Sentiment Ratio:"))
        self.ratio_label = QLabel("--")
        self.ratio_label.setStyleSheet("font-weight: bold;")
        ratio_layout.addWidget(self.ratio_label)

        ratio_layout.addWidget(QLabel("Traders:"))
        self.traders_label = QLabel("--")
        ratio_layout.addWidget(self.traders_label)

        ratio_layout.addStretch()
        layout.addLayout(ratio_layout)

        group.setLayout(layout)
        return group

    def _create_positioning_section(self) -> QGroupBox:
        """Create long/short positioning indicators"""
        group = QGroupBox("Long/Short Positioning")
        layout = QVBoxLayout()

        # Long Percentage
        long_layout = QHBoxLayout()
        long_layout.addWidget(QLabel("Long:"))

        self.long_bar = QProgressBar()
        self.long_bar.setRange(0, 100)
        self.long_bar.setValue(50)
        self.long_bar.setTextVisible(True)
        self.long_bar.setFormat("%v%")
        self.long_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        long_layout.addWidget(self.long_bar)

        self.long_label = QLabel("50%")
        self.long_label.setStyleSheet("color: green; font-weight: bold;")
        long_layout.addWidget(self.long_label)

        layout.addLayout(long_layout)

        # Short Percentage
        short_layout = QHBoxLayout()
        short_layout.addWidget(QLabel("Short:"))

        self.short_bar = QProgressBar()
        self.short_bar.setRange(0, 100)
        self.short_bar.setValue(50)
        self.short_bar.setTextVisible(True)
        self.short_bar.setFormat("%v%")
        self.short_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        short_layout.addWidget(self.short_bar)

        self.short_label = QLabel("50%")
        self.short_label.setStyleSheet("color: red; font-weight: bold;")
        short_layout.addWidget(self.short_label)

        layout.addLayout(short_layout)

        group.setLayout(layout)
        return group

    def _create_contrarian_section(self) -> QGroupBox:
        """Create contrarian signal indicators"""
        group = QGroupBox("Contrarian Signals")
        layout = QVBoxLayout()

        # Contrarian Signal Strength
        contrarian_layout = QHBoxLayout()
        contrarian_layout.addWidget(QLabel("Contrarian Signal:"))

        self.contrarian_bar = QProgressBar()
        self.contrarian_bar.setRange(-100, 100)
        self.contrarian_bar.setValue(0)
        self.contrarian_bar.setTextVisible(True)
        self.contrarian_bar.setFormat("%v%")
        contrarian_layout.addWidget(self.contrarian_bar)

        self.contrarian_label = QLabel("Neutral")
        contrarian_layout.addWidget(self.contrarian_label)

        layout.addLayout(contrarian_layout)

        # Explanation
        self.contrarian_explanation = QLabel(
            "Contrarian signals indicate extreme crowd positioning. "
            "Positive = crowd is short, negative = crowd is long."
        )
        self.contrarian_explanation.setWordWrap(True)
        self.contrarian_explanation.setStyleSheet(
            "color: gray; font-size: 10px; padding: 5px;"
        )
        layout.addWidget(self.contrarian_explanation)

        group.setLayout(layout)
        return group

    def _create_alerts_section(self) -> QGroupBox:
        """Create alerts section"""
        group = QGroupBox("Trading Alerts")
        layout = QVBoxLayout()

        # Extreme Positioning Alert
        self.extreme_alert = QLabel("")
        self.extreme_alert.setStyleSheet(
            "background-color: #FFF3CD; color: #856404; "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )
        self.extreme_alert.hide()
        layout.addWidget(self.extreme_alert)

        # Sentiment Shift Alert
        self.shift_alert = QLabel("")
        self.shift_alert.setStyleSheet(
            "background-color: #D1ECF1; color: #0C5460; "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )
        self.shift_alert.hide()
        layout.addWidget(self.shift_alert)

        # Contrarian Opportunity Alert
        self.opportunity_alert = QLabel("")
        self.opportunity_alert.setStyleSheet(
            "background-color: #D4EDDA; color: #155724; "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )
        self.opportunity_alert.hide()
        layout.addWidget(self.opportunity_alert)

        group.setLayout(layout)
        return group

    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update sentiment metrics.

        Args:
            metrics: Dictionary with sentiment metrics
        """
        self.current_metrics = metrics

        # Sentiment Label
        sentiment = metrics.get('sentiment', 'NEUTRAL')
        self.sentiment_label.setText(sentiment.upper())

        # Color code sentiment
        if sentiment.lower() == 'bullish':
            self.sentiment_label.setStyleSheet(
                "font-weight: bold; font-size: 16px; "
                "padding: 5px 10px; border-radius: 3px; "
                "background-color: #D4EDDA; color: #155724;"
            )
        elif sentiment.lower() == 'bearish':
            self.sentiment_label.setStyleSheet(
                "font-weight: bold; font-size: 16px; "
                "padding: 5px 10px; border-radius: 3px; "
                "background-color: #F8D7DA; color: #721C24;"
            )
        else:
            self.sentiment_label.setStyleSheet(
                "font-weight: bold; font-size: 16px; "
                "padding: 5px 10px; border-radius: 3px; "
                "background-color: #E0E0E0; color: #333;"
            )

        # Confidence
        confidence = metrics.get('confidence', 0.0)
        self.confidence_label.setText(f"{confidence:.1%}")

        # Ratio
        ratio = metrics.get('ratio', 0.0)
        self.ratio_label.setText(f"{ratio:.2f}")

        # Traders
        total_traders = metrics.get('total_traders', 0)
        self.traders_label.setText(f"{total_traders:,}")

        # Long/Short Positioning
        long_pct = metrics.get('long_pct', 50.0)
        short_pct = metrics.get('short_pct', 50.0)

        self.long_bar.setValue(int(long_pct))
        self.long_label.setText(f"{long_pct:.1f}%")

        self.short_bar.setValue(int(short_pct))
        self.short_label.setText(f"{short_pct:.1f}%")

        # Contrarian Signal (-1 to +1)
        contrarian_signal = metrics.get('contrarian_signal', 0.0)
        
        # Update bar value if changed
        if self._metrics_cache["contrarian_signal"] != contrarian_signal:
            contrarian_pct = int(contrarian_signal * 100)
            self.contrarian_bar.setValue(contrarian_pct)
            self._metrics_cache["contrarian_signal"] = contrarian_signal
            logger.debug(f"Contrarian signal updated: {contrarian_signal:.2f} ({contrarian_pct}%)")

        # Update state if changed
        if contrarian_signal > 0.3:
            contrarian_state = "fade_short"
        elif contrarian_signal < -0.3:
            contrarian_state = "fade_long"
        else:
            contrarian_state = "neutral"
            
        if self._metrics_cache["contrarian_state"] != contrarian_state:
            if contrarian_state == "fade_short":
                self.contrarian_label.setText("FADE SHORT (Crowd Bearish)")
                self.contrarian_label.setStyleSheet("color: green; font-weight: bold;")
                self.contrarian_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            elif contrarian_state == "fade_long":
                self.contrarian_label.setText("FADE LONG (Crowd Bullish)")
                self.contrarian_label.setStyleSheet("color: red; font-weight: bold;")
                self.contrarian_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
            else:
                self.contrarian_label.setText("Neutral")
                self.contrarian_label.setStyleSheet("color: gray;")
                self.contrarian_bar.setStyleSheet("QProgressBar::chunk { background-color: gray; }")
            self._metrics_cache["contrarian_state"] = contrarian_state
            logger.debug(f"Contrarian state updated: {contrarian_state}")

        # Alerts
        self._update_alerts(metrics)

    def _update_alerts(self, metrics: Dict[str, Any]):
        """Update alert labels"""
        long_pct = metrics.get('long_pct', 50.0)
        short_pct = metrics.get('short_pct', 50.0)
        confidence = metrics.get('confidence', 0.0)
        contrarian_signal = metrics.get('contrarian_signal', 0.0)

        # Extreme Positioning Alert
        if long_pct > 75 or short_pct > 75:
            extreme_side = "LONG" if long_pct > 75 else "SHORT"
            self.extreme_alert.setText(
                f"⚠️ EXTREME {extreme_side} POSITIONING: {max(long_pct, short_pct):.1f}% - "
                f"Potential reversal zone!"
            )
            self.extreme_alert.show()
        else:
            self.extreme_alert.hide()

        # Sentiment Shift Alert (would need historical data)
        # For now, hide by default
        self.shift_alert.hide()

        # Contrarian Opportunity Alert
        if abs(contrarian_signal) > 0.5 and confidence > 0.6:
            if contrarian_signal > 0:
                self.opportunity_alert.setText(
                    f"✅ CONTRARIAN BUY OPPORTUNITY: Crowd is heavily short "
                    f"({short_pct:.1f}%), consider fading"
                )
            else:
                self.opportunity_alert.setText(
                    f"✅ CONTRARIAN SELL OPPORTUNITY: Crowd is heavily long "
                    f"({long_pct:.1f}%), consider fading"
                )
            self.opportunity_alert.show()
        else:
            self.opportunity_alert.hide()

    def refresh_display(self):
        """Refresh display (called by timer) - fetches real-time sentiment data"""
        if not self.sentiment_service:
            logger.debug("Sentiment service not available")
            return

        try:
            # Get latest sentiment metrics from service
            metrics = self.sentiment_service.get_latest_sentiment_metrics(self.current_symbol)

            if not metrics:
                logger.debug(f"No sentiment data available for {self.current_symbol}")
                return

            logger.debug(f"Sentiment refresh for {self.current_symbol}: long={metrics.get('long_pct')}%, contrarian={metrics.get('contrarian_signal')}")
            self.update_metrics(metrics)

        except Exception as e:
            logger.error(f"Error refreshing sentiment display: {e}", exc_info=True)

    def on_symbol_changed(self, symbol: str):
        """Handle symbol change"""
        self.current_symbol = symbol
        self.current_metrics = {}

        # Immediately fetch new data
        self.refresh_display()

        logger.info(f"Sentiment Panel switched to {symbol}")

    def clear_data(self):
        """Clear all data"""
        self.current_metrics = {}
        self.sentiment_label.setText("NEUTRAL")
        self.confidence_label.setText("--")
        self.ratio_label.setText("--")
        self.traders_label.setText("--")
        self.long_bar.setValue(50)
        self.long_label.setText("50%")
        self.short_bar.setValue(50)
        self.short_label.setText("50%")
        self.contrarian_bar.setValue(0)
        self.contrarian_label.setText("Neutral")
        self.extreme_alert.hide()
        self.shift_alert.hide()
        self.opportunity_alert.hide()
