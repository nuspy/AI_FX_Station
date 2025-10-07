"""
News & Event Signal Processor

Generates trading signals from scheduled events and news sentiment.
Integrates with existing event calendar and sentiment services.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class EventType(Enum):
    """Event types"""
    ECONOMIC_DATA = "economic_data"
    CENTRAL_BANK = "central_bank"
    GEOPOLITICAL = "geopolitical"
    EARNINGS = "earnings"
    NEWS_RELEASE = "news_release"


class EventImpact(Enum):
    """Event impact levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SignalTiming(Enum):
    """Signal timing relative to event"""
    PRE_EVENT = "pre_event"
    POST_EVENT = "post_event"
    REACTION = "reaction"


@dataclass
class EconomicEvent:
    """Economic calendar event"""
    event_id: int
    event_type: EventType
    event_name: str
    timestamp: int
    affected_currencies: List[str]
    impact_level: EventImpact
    consensus: Optional[float] = None
    previous: Optional[float] = None
    actual: Optional[float] = None
    surprise_factor: Optional[float] = None


@dataclass
class EventSignal:
    """Trading signal generated from event"""
    event: EconomicEvent
    signal_direction: str  # bull, bear, neutral
    signal_strength: float  # 0-1
    signal_timing: SignalTiming
    affected_symbols: List[str]
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None
    sentiment_score: Optional[float] = None
    sentiment_velocity: Optional[float] = None
    confidence: float = 0.5
    metadata: Optional[Dict[str, Any]] = None


class EventSignalProcessor:
    """
    Processes economic events and news for trading signal generation.

    Features:
    - Pre-event positioning signals
    - Post-event reaction signals
    - Sentiment integration
    - Surprise factor calculation
    - Cross-asset impact analysis
    """

    # Major currency pairs mapping
    CURRENCY_PAIRS = {
        'USD': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF'],
        'EUR': ['EURUSD', 'EURGBP', 'EURJPY', 'EURAUD'],
        'GBP': ['GBPUSD', 'EURGBP', 'GBPJPY'],
        'JPY': ['USDJPY', 'EURJPY', 'GBPJPY'],
        'AUD': ['AUDUSD', 'EURAUD'],
        'CAD': ['USDCAD'],
        'CHF': ['USDCHF'],
    }

    # Event-specific trading windows (minutes before/after event)
    EVENT_WINDOWS = {
        EventType.CENTRAL_BANK: (60, 120),  # 1hr before, 2hr after
        EventType.ECONOMIC_DATA: (30, 60),
        EventType.GEOPOLITICAL: (0, 180),
        EventType.NEWS_RELEASE: (0, 30),
        EventType.EARNINGS: (0, 60)
    }

    def __init__(
        self,
        surprise_threshold: float = 0.5,  # Threshold for significant surprise
        sentiment_threshold: float = 0.3,  # Threshold for strong sentiment
        min_impact_level: EventImpact = EventImpact.MEDIUM
    ):
        """
        Initialize event signal processor.

        Args:
            surprise_threshold: Threshold for surprise factor (consensus vs actual)
            sentiment_threshold: Threshold for sentiment strength
            min_impact_level: Minimum impact level to generate signals
        """
        self.surprise_threshold = surprise_threshold
        self.sentiment_threshold = sentiment_threshold
        self.min_impact_level = min_impact_level

        # Sentiment tracking
        self.sentiment_history: Dict[str, List[Tuple[int, float]]] = {}

    def calculate_surprise_factor(
        self,
        consensus: Optional[float],
        actual: Optional[float],
        previous: Optional[float]
    ) -> Optional[float]:
        """
        Calculate surprise factor from economic data release.

        Args:
            consensus: Consensus forecast
            actual: Actual released value
            previous: Previous value

        Returns:
            Surprise factor (-1 to +1, or None if insufficient data)
        """
        if consensus is None or actual is None:
            return None

        # Use previous as baseline if available
        if previous is not None and previous != 0:
            expected_change = consensus - previous
            actual_change = actual - previous
            surprise = (actual_change - expected_change) / abs(previous)
        elif consensus != 0:
            surprise = (actual - consensus) / abs(consensus)
        else:
            return None

        # Normalize to -1 to +1
        return float(np.clip(surprise, -1.0, 1.0))

    def update_sentiment(
        self,
        symbol: str,
        timestamp: int,
        sentiment_score: float
    ):
        """
        Update sentiment history for tracking velocity.

        Args:
            symbol: Trading symbol
            timestamp: Unix timestamp in milliseconds
            sentiment_score: Sentiment score (-1 to +1)
        """
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = []

        self.sentiment_history[symbol].append((timestamp, sentiment_score))

        # Keep last 100 entries
        if len(self.sentiment_history[symbol]) > 100:
            self.sentiment_history[symbol].pop(0)

    def calculate_sentiment_velocity(
        self,
        symbol: str,
        window_minutes: int = 60
    ) -> Optional[float]:
        """
        Calculate sentiment velocity (rate of change).

        Args:
            symbol: Trading symbol
            window_minutes: Time window for velocity calculation

        Returns:
            Sentiment velocity (change per minute, or None)
        """
        if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 2:
            return None

        history = self.sentiment_history[symbol]
        current_time = history[-1][0]
        cutoff_time = current_time - (window_minutes * 60 * 1000)

        # Filter to window
        recent = [(ts, sent) for ts, sent in history if ts >= cutoff_time]

        if len(recent) < 2:
            return None

        # Calculate linear slope
        timestamps = np.array([ts for ts, _ in recent])
        sentiments = np.array([sent for _, sent in recent])

        # Convert timestamps to minutes
        timestamps_minutes = (timestamps - timestamps[0]) / (60 * 1000)

        # Linear regression
        if len(timestamps_minutes) > 1 and np.std(timestamps_minutes) > 0:
            slope, _ = np.polyfit(timestamps_minutes, sentiments, 1)
            return float(slope)

        return None

    def generate_pre_event_signal(
        self,
        event: EconomicEvent,
        current_sentiment: Optional[float] = None
    ) -> Optional[EventSignal]:
        """
        Generate pre-event positioning signal.

        Args:
            event: Economic event
            current_sentiment: Current sentiment score

        Returns:
            Event signal or None
        """
        # Only for high impact events
        if event.impact_level != EventImpact.HIGH:
            return None

        # Determine affected symbols
        affected_symbols = []
        for currency in event.affected_currencies:
            affected_symbols.extend(self.CURRENCY_PAIRS.get(currency, []))
        affected_symbols = list(set(affected_symbols))

        if not affected_symbols:
            return None

        # Signal based on consensus vs previous
        if event.consensus is not None and event.previous is not None:
            expected_change = event.consensus - event.previous
            if abs(expected_change) < 0.01:  # Negligible change
                return None

            direction = 'bull' if expected_change > 0 else 'bear'
            strength = min(abs(expected_change), 1.0)
        else:
            # Use sentiment if available
            if current_sentiment is not None and abs(current_sentiment) > self.sentiment_threshold:
                direction = 'bull' if current_sentiment > 0 else 'bear'
                strength = abs(current_sentiment)
            else:
                return None

        # Calculate signal window
        window_before, _ = self.EVENT_WINDOWS.get(event.event_type, (30, 60))
        valid_from = event.timestamp - (window_before * 60 * 1000)
        valid_until = event.timestamp

        return EventSignal(
            event=event,
            signal_direction=direction,
            signal_strength=strength,
            signal_timing=SignalTiming.PRE_EVENT,
            affected_symbols=affected_symbols,
            valid_from=valid_from,
            valid_until=valid_until,
            sentiment_score=current_sentiment,
            confidence=0.6,
            metadata={'strategy': 'pre_position'}
        )

    def generate_post_event_signal(
        self,
        event: EconomicEvent,
        current_sentiment: Optional[float] = None,
        sentiment_velocity: Optional[float] = None
    ) -> Optional[EventSignal]:
        """
        Generate post-event reaction signal.

        Args:
            event: Economic event (with actual data)
            current_sentiment: Current sentiment
            sentiment_velocity: Sentiment rate of change

        Returns:
            Event signal or None
        """
        # Need actual data for post-event signal
        if event.actual is None:
            return None

        # Calculate surprise
        surprise = self.calculate_surprise_factor(
            event.consensus, event.actual, event.previous
        )

        if surprise is None or abs(surprise) < self.surprise_threshold:
            return None

        # Determine direction
        direction = 'bull' if surprise > 0 else 'bear'
        strength = abs(surprise)

        # Adjust with sentiment
        if current_sentiment is not None:
            sentiment_alignment = (surprise * current_sentiment) > 0  # Same direction
            if sentiment_alignment:
                strength = min(strength + abs(current_sentiment) * 0.3, 1.0)
                confidence = 0.75
            else:
                confidence = 0.55  # Conflict between data and sentiment

        else:
            confidence = 0.65

        # Affected symbols
        affected_symbols = []
        for currency in event.affected_currencies:
            affected_symbols.extend(self.CURRENCY_PAIRS.get(currency, []))
        affected_symbols = list(set(affected_symbols))

        # Signal window
        _, window_after = self.EVENT_WINDOWS.get(event.event_type, (30, 60))
        valid_from = event.timestamp
        valid_until = event.timestamp + (window_after * 60 * 1000)

        return EventSignal(
            event=event,
            signal_direction=direction,
            signal_strength=strength,
            signal_timing=SignalTiming.POST_EVENT,
            affected_symbols=affected_symbols,
            valid_from=valid_from,
            valid_until=valid_until,
            sentiment_score=current_sentiment,
            sentiment_velocity=sentiment_velocity,
            confidence=confidence,
            metadata={
                'surprise_factor': surprise,
                'strategy': 'event_reaction'
            }
        )

    def process_event(
        self,
        event: EconomicEvent,
        current_time: int,
        current_sentiment: Optional[float] = None
    ) -> List[EventSignal]:
        """
        Process an event and generate appropriate signals.

        Args:
            event: Economic event
            current_time: Current timestamp
            current_sentiment: Current sentiment score

        Returns:
            List of generated signals
        """
        signals: List[EventSignal] = []

        # Determine if pre-event or post-event
        if current_time < event.timestamp:
            # Pre-event
            signal = self.generate_pre_event_signal(event, current_sentiment)
            if signal:
                signals.append(signal)

        elif current_time >= event.timestamp and event.actual is not None:
            # Post-event
            # Calculate sentiment velocity
            affected_symbols = []
            for currency in event.affected_currencies:
                affected_symbols.extend(self.CURRENCY_PAIRS.get(currency, []))

            sentiment_velocity = None
            if affected_symbols:
                sentiment_velocity = self.calculate_sentiment_velocity(affected_symbols[0])

            signal = self.generate_post_event_signal(
                event, current_sentiment, sentiment_velocity
            )
            if signal:
                signals.append(signal)

        return signals

    def filter_by_impact(
        self,
        signals: List[EventSignal]
    ) -> List[EventSignal]:
        """
        Filter signals by minimum impact level.

        Args:
            signals: List of signals

        Returns:
            Filtered signals
        """
        impact_order = {
            EventImpact.HIGH: 3,
            EventImpact.MEDIUM: 2,
            EventImpact.LOW: 1
        }

        min_level = impact_order[self.min_impact_level]

        return [
            signal for signal in signals
            if impact_order.get(signal.event.impact_level, 0) >= min_level
        ]

    def get_upcoming_events(
        self,
        events: List[EconomicEvent],
        current_time: int,
        hours_ahead: int = 24
    ) -> List[EconomicEvent]:
        """
        Get upcoming events within time window.

        Args:
            events: List of all events
            current_time: Current timestamp
            hours_ahead: Hours to look ahead

        Returns:
            Upcoming events
        """
        cutoff_time = current_time + (hours_ahead * 60 * 60 * 1000)

        upcoming = [
            event for event in events
            if current_time <= event.timestamp <= cutoff_time
        ]

        # Sort by timestamp
        upcoming.sort(key=lambda e: e.timestamp)

        return upcoming
