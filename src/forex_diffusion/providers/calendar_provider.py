"""
Economic Calendar Providers

Fetches economic events from external sources:
- ForexFactory
- Investing.com
- FXStreet
- Trading Economics
"""
from __future__ import annotations

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from loguru import logger

from .base import DataProvider, DataType


class ForexFactoryCalendarProvider(DataProvider):
    """
    ForexFactory economic calendar provider.

    Scrapes calendar data from ForexFactory website.
    """

    def __init__(self):
        super().__init__(
            name="forexfactory",
            data_types=[DataType.CALENDAR],
            requires_auth=False
        )
        self.base_url = "https://www.forexfactory.com/calendar"

    async def _get_economic_calendar_impl(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        currency: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch economic calendar from ForexFactory"""

        try:
            # ForexFactory uses week-based calendar
            # We'll fetch current week
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, headers=headers) as response:
                    response.raise_for_status()
                    html = await response.text()

            soup = BeautifulSoup(html, 'html.parser')

            events = []

            # Find calendar table
            calendar_table = soup.find('table', class_='calendar__table')

            if not calendar_table:
                logger.warning("Could not find calendar table on ForexFactory")
                return []

            rows = calendar_table.find_all('tr', class_='calendar__row')

            current_date = None

            for row in rows:
                # Check if this is a date row
                date_cell = row.find('td', class_='calendar__date')
                if date_cell and date_cell.get_text(strip=True):
                    # Parse date
                    date_str = date_cell.get_text(strip=True)
                    try:
                        current_date = datetime.strptime(date_str, "%a%b%d")
                        # Add current year
                        current_date = current_date.replace(year=datetime.now().year)
                    except:
                        pass

                # Parse event data
                time_cell = row.find('td', class_='calendar__time')
                currency_cell = row.find('td', class_='calendar__currency')
                impact_cell = row.find('td', class_='calendar__impact')
                event_cell = row.find('td', class_='calendar__event')
                actual_cell = row.find('td', class_='calendar__actual')
                forecast_cell = row.find('td', class_='calendar__forecast')
                previous_cell = row.find('td', class_='calendar__previous')

                if not event_cell:
                    continue

                # Extract data
                time_str = time_cell.get_text(strip=True) if time_cell else None
                event_currency = currency_cell.get_text(strip=True) if currency_cell else None
                event_title = event_cell.get_text(strip=True)

                # Skip if currency filter doesn't match
                if currency and event_currency != currency:
                    continue

                # Determine impact
                impact = "low"
                if impact_cell:
                    impact_span = impact_cell.find('span')
                    if impact_span:
                        classes = impact_span.get('class', [])
                        if 'high' in ' '.join(classes):
                            impact = "high"
                        elif 'medium' in ' '.join(classes):
                            impact = "medium"

                # Parse actual, forecast, previous
                actual = actual_cell.get_text(strip=True) if actual_cell else None
                forecast = forecast_cell.get_text(strip=True) if forecast_cell else None
                previous = previous_cell.get_text(strip=True) if previous_cell else None

                # Create timestamp
                event_timestamp = None
                if current_date and time_str:
                    try:
                        # Parse time (e.g., "8:30am")
                        time_parts = time_str.replace('am', ' AM').replace('pm', ' PM').split()
                        if len(time_parts) >= 1:
                            time_obj = datetime.strptime(time_parts[0] + ' ' + time_parts[-1], "%I:%M %p")
                            event_dt = current_date.replace(
                                hour=time_obj.hour,
                                minute=time_obj.minute,
                                second=0,
                                microsecond=0
                            )
                            event_timestamp = int(event_dt.timestamp() * 1000)
                    except:
                        pass

                if not event_timestamp:
                    # Use current_date at midnight if time not available
                    if current_date:
                        event_timestamp = int(current_date.timestamp() * 1000)
                    else:
                        continue

                events.append({
                    'timestamp': event_timestamp,
                    'event': event_title,
                    'currency': event_currency,
                    'impact': impact,
                    'forecast': forecast,
                    'previous': previous,
                    'actual': actual,
                    'source': 'forexfactory'
                })

            logger.info(f"Fetched {len(events)} events from ForexFactory")
            return events

        except Exception as e:
            logger.error(f"Failed to fetch ForexFactory calendar: {e}")
            return []


class InvestingCalendarProvider(DataProvider):
    """
    Investing.com economic calendar provider.

    Uses Investing.com API (unofficial).
    """

    def __init__(self):
        super().__init__(
            name="investing",
            data_types=[DataType.CALENDAR],
            requires_auth=False
        )
        self.base_url = "https://www.investing.com/economic-calendar"

    async def _get_economic_calendar_impl(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        currency: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch economic calendar from Investing.com"""

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'X-Requested-With': 'XMLHttpRequest'
            }

            # Investing.com uses AJAX for calendar data
            # Format dates
            if not start_date:
                start_date = datetime.now()
            if not end_date:
                end_date = start_date + timedelta(days=7)

            params = {
                'dateFrom': start_date.strftime('%Y-%m-%d'),
                'dateTo': end_date.strftime('%Y-%m-%d'),
            }

            if currency:
                # Map currency to Investing.com country codes
                currency_map = {
                    'USD': '5',   # United States
                    'EUR': '72',  # Euro Zone
                    'GBP': '4',   # United Kingdom
                    'JPY': '35',  # Japan
                    'CHF': '12',  # Switzerland
                    'AUD': '25',  # Australia
                    'CAD': '6',   # Canada
                    'NZD': '43',  # New Zealand
                }
                if currency in currency_map:
                    params['country[]'] = currency_map[currency]

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    html = await response.text()

            soup = BeautifulSoup(html, 'html.parser')

            events = []

            # Find event rows
            event_rows = soup.find_all('tr', {'event_timestamp': True})

            for row in event_rows:
                timestamp = int(row.get('event_timestamp', 0))

                if timestamp == 0:
                    continue

                # Extract event details
                event_cell = row.find('td', class_='event')
                currency_cell = row.find('td', class_='flagCur')
                impact_cell = row.find('td', class_='sentiment')
                actual_cell = row.find('td', id=lambda x: x and x.endswith('_actual'))
                forecast_cell = row.find('td', id=lambda x: x and x.endswith('_forecast'))
                previous_cell = row.find('td', id=lambda x: x and x.endswith('_previous'))

                event_name = event_cell.get_text(strip=True) if event_cell else "Unknown"
                event_currency = currency_cell.get_text(strip=True) if currency_cell else None

                # Determine impact (1-3 bulls)
                impact = "low"
                if impact_cell:
                    bulls = len(impact_cell.find_all('i', class_='grayFullBullishIcon'))
                    if bulls >= 3:
                        impact = "high"
                    elif bulls == 2:
                        impact = "medium"

                actual = actual_cell.get_text(strip=True) if actual_cell else None
                forecast = forecast_cell.get_text(strip=True) if forecast_cell else None
                previous = previous_cell.get_text(strip=True) if previous_cell else None

                events.append({
                    'timestamp': timestamp * 1000,  # Convert to ms
                    'event': event_name,
                    'currency': event_currency,
                    'impact': impact,
                    'forecast': forecast,
                    'previous': previous,
                    'actual': actual,
                    'source': 'investing.com'
                })

            logger.info(f"Fetched {len(events)} events from Investing.com")
            return events

        except Exception as e:
            logger.error(f"Failed to fetch Investing.com calendar: {e}")
            return []


class EconomicCalendarAggregator:
    """
    Aggregates economic calendar from multiple providers.
    """

    def __init__(self):
        self.providers = [
            ForexFactoryCalendarProvider(),
            # InvestingCalendarProvider(),  # Disabled by default (requires anti-bot handling)
        ]

    async def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        currency: Optional[str] = None,
        impact_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get economic events from all providers.

        Args:
            start_date: Start date for events
            end_date: End date for events
            currency: Filter by currency (e.g., "USD")
            impact_filter: Filter by impact ("high", "medium", "low")

        Returns:
            List of event dictionaries, sorted by timestamp
        """

        tasks = [
            provider.get_economic_calendar(start_date, end_date, currency)
            for provider in self.providers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_events = []

        for provider, result in zip(self.providers, results):
            if isinstance(result, Exception):
                logger.error(f"Provider {provider.name} failed: {result}")
                continue

            if result:
                all_events.extend(result)

        # Remove duplicates (same event from multiple providers)
        seen = set()
        unique_events = []

        for event in all_events:
            # Create unique key
            key = (event['timestamp'], event['event'], event['currency'])
            if key not in seen:
                seen.add(key)
                unique_events.append(event)

        # Filter by impact
        if impact_filter:
            unique_events = [e for e in unique_events if e['impact'] == impact_filter]

        # Sort by timestamp
        unique_events.sort(key=lambda x: x['timestamp'])

        logger.info(f"Aggregated {len(unique_events)} unique events from {len(self.providers)} providers")

        return unique_events

    async def get_high_impact_events(
        self,
        hours_ahead: int = 24,
        currency: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming high-impact events.

        Args:
            hours_ahead: Look ahead this many hours
            currency: Filter by currency

        Returns:
            List of high-impact events
        """

        start_date = datetime.now()
        end_date = start_date + timedelta(hours=hours_ahead)

        events = await self.get_events(
            start_date=start_date,
            end_date=end_date,
            currency=currency,
            impact_filter="high"
        )

        return events

    def get_risk_score(
        self,
        events: List[Dict[str, Any]],
        symbol: str
    ) -> float:
        """
        Calculate risk score based on upcoming events.

        Args:
            events: List of events
            symbol: Trading symbol (e.g., "EUR/USD")

        Returns:
            Risk score 0-100 (higher = more risky)
        """

        if not events:
            return 0.0

        # Extract currencies from symbol
        symbol_currencies = []
        if '/' in symbol:
            base, quote = symbol.split('/')
            symbol_currencies = [base, quote]

        risk_score = 0.0

        for event in events:
            if event['currency'] not in symbol_currencies:
                continue

            # Weight by impact
            if event['impact'] == 'high':
                risk_score += 30
            elif event['impact'] == 'medium':
                risk_score += 15
            else:
                risk_score += 5

            # Weight by time proximity (events in next 1 hour = higher risk)
            time_until = (event['timestamp'] / 1000) - datetime.now().timestamp()
            hours_until = time_until / 3600

            if hours_until < 1:
                risk_score *= 2.0
            elif hours_until < 4:
                risk_score *= 1.5

        return min(risk_score, 100.0)


# Convenience functions
async def fetch_upcoming_events(
    hours_ahead: int = 24,
    currency: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch upcoming economic events.

    Usage:
        events = await fetch_upcoming_events(hours_ahead=48, currency="USD")
        for event in events:
            print(f"{event['event']} - {event['impact']}")
    """
    aggregator = EconomicCalendarAggregator()
    return await aggregator.get_events(
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(hours=hours_ahead),
        currency=currency
    )


async def get_trading_risk(symbol: str, hours_ahead: int = 24) -> Dict[str, Any]:
    """
    Get trading risk assessment based on upcoming events.

    Returns:
        Dictionary with risk score and event details
    """
    aggregator = EconomicCalendarAggregator()

    events = await aggregator.get_high_impact_events(
        hours_ahead=hours_ahead
    )

    risk_score = aggregator.get_risk_score(events, symbol)

    return {
        'symbol': symbol,
        'risk_score': risk_score,
        'risk_level': 'high' if risk_score > 60 else 'medium' if risk_score > 30 else 'low',
        'high_impact_events_count': len(events),
        'events': events
    }
