"""
External Sentiment Data Providers

Fetches sentiment indicators from external sources:
- Fear & Greed Index (CNN, Alternative.me)
- VIX (CBOE Volatility Index)
- Put/Call Ratio
- Crypto Fear & Greed Index
"""
from __future__ import annotations

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from .base import DataProvider, DataType


class FearGreedProvider(DataProvider):
    """
    CNN Fear & Greed Index provider.

    Scale: 0-100
    - 0-25: Extreme Fear
    - 25-45: Fear
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed
    """

    def __init__(self):
        super().__init__(
            name="fear_greed",
            data_types=[DataType.SENTIMENT],
            requires_auth=False
        )
        # Use Alternative.me API (free, no key required)
        self.base_url = "https://api.alternative.me/fng/"

    async def _fetch_sentiment_impl(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Fetch Fear & Greed Index data"""

        # Fear & Greed is market-wide, not symbol-specific
        # But we'll tag it with symbol for consistency

        try:
            params = {}

            # Determine how many days to fetch
            if start_time and end_time:
                days = (end_time - start_time).days
                params['limit'] = min(max(days, 1), 365)  # Max 1 year
            else:
                params['limit'] = 30  # Default last 30 days

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if 'data' not in data:
                logger.warning("No Fear & Greed data returned")
                return []

            results = []
            for item in data['data']:
                timestamp = int(item['timestamp'])
                value = int(item['value'])
                classification = item['value_classification']  # e.g., "Extreme Fear"

                results.append({
                    'timestamp': timestamp * 1000,  # Convert to ms
                    'symbol': symbol,  # Tag with symbol even though it's market-wide
                    'indicator': 'fear_greed_index',
                    'value': value,
                    'classification': classification,
                    'source': 'alternative.me'
                })

            logger.info(f"Fetched {len(results)} Fear & Greed data points")
            return results

        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed data: {e}")
            return []


class VIXProvider(DataProvider):
    """
    VIX (CBOE Volatility Index) provider.

    The VIX is known as the "fear index" - measures expected volatility.
    - VIX < 12: Low volatility (complacency)
    - VIX 12-20: Normal volatility
    - VIX 20-30: Elevated volatility (concern)
    - VIX > 30: High volatility (fear/panic)
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="vix",
            data_types=[DataType.SENTIMENT],
            requires_auth=False  # Can use free sources
        )
        self.api_key = api_key
        # Use Alpha Vantage or Yahoo Finance for VIX data
        self.base_url = "https://query1.finance.yahoo.com/v7/finance/quote"

    async def _fetch_sentiment_impl(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Fetch VIX data"""

        try:
            # Fetch current VIX value from Yahoo Finance
            params = {
                'symbols': '^VIX',
                'fields': 'regularMarketPrice,regularMarketTime'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if 'quoteResponse' not in data or 'result' not in data['quoteResponse']:
                logger.warning("No VIX data returned")
                return []

            result = data['quoteResponse']['result'][0]

            vix_value = result.get('regularMarketPrice')
            timestamp = result.get('regularMarketTime', int(datetime.now().timestamp()))

            if vix_value is None:
                return []

            # Classify VIX level
            if vix_value < 12:
                classification = "Complacency"
            elif vix_value < 20:
                classification = "Normal"
            elif vix_value < 30:
                classification = "Concern"
            else:
                classification = "Fear"

            return [{
                'timestamp': timestamp * 1000,
                'symbol': symbol,
                'indicator': 'vix',
                'value': vix_value,
                'classification': classification,
                'source': 'yahoo_finance'
            }]

        except Exception as e:
            logger.error(f"Failed to fetch VIX data: {e}")
            return []


class CryptoFearGreedProvider(DataProvider):
    """
    Crypto Fear & Greed Index provider (Alternative.me).

    Useful for crypto pairs like BTC/USD.
    """

    def __init__(self):
        super().__init__(
            name="crypto_fear_greed",
            data_types=[DataType.SENTIMENT],
            requires_auth=False
        )
        self.base_url = "https://api.alternative.me/fng/"

    async def _fetch_sentiment_impl(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Fetch Crypto Fear & Greed Index"""

        # Only fetch for crypto symbols
        if not any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'CRYPTO']):
            return []

        try:
            params = {}

            if start_time and end_time:
                days = (end_time - start_time).days
                params['limit'] = min(max(days, 1), 365)
            else:
                params['limit'] = 30

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if 'data' not in data:
                return []

            results = []
            for item in data['data']:
                timestamp = int(item['timestamp'])
                value = int(item['value'])
                classification = item['value_classification']

                results.append({
                    'timestamp': timestamp * 1000,
                    'symbol': symbol,
                    'indicator': 'crypto_fear_greed',
                    'value': value,
                    'classification': classification,
                    'source': 'alternative.me'
                })

            logger.info(f"Fetched {len(results)} Crypto Fear & Greed data points")
            return results

        except Exception as e:
            logger.error(f"Failed to fetch Crypto Fear & Greed data: {e}")
            return []


class SentimentAggregator:
    """
    Aggregates sentiment from multiple providers.
    """

    def __init__(self):
        self.providers = [
            FearGreedProvider(),
            VIXProvider(),
            CryptoFearGreedProvider(),
        ]

    async def get_composite_sentiment(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get composite sentiment score from all providers.

        Returns:
            Dictionary with individual indicators and composite score
        """

        tasks = [
            provider.fetch_sentiment(symbol, start_time, end_time)
            for provider in self.providers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sentiment_data = {}

        for provider, result in zip(self.providers, results):
            if isinstance(result, Exception):
                logger.error(f"Provider {provider.name} failed: {result}")
                continue

            if result:
                # Get most recent data point
                latest = max(result, key=lambda x: x['timestamp'])
                sentiment_data[latest['indicator']] = latest

        # Calculate composite score (normalize to 0-100)
        composite_score = None
        scores = []

        if 'fear_greed_index' in sentiment_data:
            scores.append(sentiment_data['fear_greed_index']['value'])

        if 'vix' in sentiment_data:
            # VIX: inverse relationship (high VIX = fear = low score)
            # Normalize VIX (typical range 10-80) to 0-100, inverted
            vix = sentiment_data['vix']['value']
            vix_normalized = max(0, min(100, 100 - ((vix - 10) / 70 * 100)))
            scores.append(vix_normalized)

        if 'crypto_fear_greed' in sentiment_data and any(c in symbol.upper() for c in ['BTC', 'ETH', 'CRYPTO']):
            scores.append(sentiment_data['crypto_fear_greed']['value'])

        if scores:
            composite_score = sum(scores) / len(scores)

        # Classify composite sentiment
        classification = "neutral"
        if composite_score is not None:
            if composite_score < 25:
                classification = "extreme_fear"
            elif composite_score < 45:
                classification = "fear"
            elif composite_score < 55:
                classification = "neutral"
            elif composite_score < 75:
                classification = "greed"
            else:
                classification = "extreme_greed"

        return {
            'symbol': symbol,
            'timestamp': datetime.now().timestamp() * 1000,
            'composite_score': composite_score,
            'classification': classification,
            'indicators': sentiment_data,
            'providers_count': len(sentiment_data)
        }

    async def get_sentiment_signal(
        self,
        symbol: str,
        strategy: str = "contrarian"
    ) -> Optional[str]:
        """
        Get trading signal based on sentiment.

        Args:
            symbol: Trading symbol
            strategy: "contrarian" or "momentum"

        Returns:
            "bullish", "bearish", or "neutral"
        """

        sentiment = await self.get_composite_sentiment(symbol)

        if sentiment['composite_score'] is None:
            return None

        score = sentiment['composite_score']

        if strategy == "contrarian":
            # Contrarian: extreme fear = buy, extreme greed = sell
            if score < 25:
                return "bullish"
            elif score > 75:
                return "bearish"
            else:
                return "neutral"

        elif strategy == "momentum":
            # Momentum: follow the crowd
            if score < 45:
                return "bearish"
            elif score > 55:
                return "bullish"
            else:
                return "neutral"

        return "neutral"


# Convenience function
async def fetch_current_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Fetch current sentiment for a symbol.

    Usage:
        sentiment = await fetch_current_sentiment("EUR/USD")
        print(f"Composite score: {sentiment['composite_score']}")
        print(f"Classification: {sentiment['classification']}")
    """
    aggregator = SentimentAggregator()
    return await aggregator.get_composite_sentiment(symbol)
