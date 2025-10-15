"""
Example: Sentiment & Economic Calendar Integration

Demonstrates how to use sentiment indicators and economic calendar
in trading decisions.

Usage:
    python examples/sentiment_calendar_integration_example.py
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from forex_diffusion.providers.sentiment_provider import (
    SentimentAggregator,
    fetch_current_sentiment
)
from forex_diffusion.providers.calendar_provider import (
    EconomicCalendarAggregator,
    fetch_upcoming_events,
    get_trading_risk
)


async def demonstrate_sentiment():
    """Demonstrate sentiment analysis"""

    logger.info("=" * 80)
    logger.info("SENTIMENT ANALYSIS DEMONSTRATION")
    logger.info("=" * 80)

    # Create sentiment aggregator
    sentiment_agg = SentimentAggregator()

    # Fetch sentiment for EUR/USD
    symbol = "EUR/USD"

    logger.info(f"\nFetching sentiment for {symbol}...")

    sentiment = await sentiment_agg.get_composite_sentiment(symbol)

    logger.info(f"\nSentiment Results:")
    logger.info(f"  Composite Score: {sentiment['composite_score']:.1f}/100")
    logger.info(f"  Classification: {sentiment['classification']}")
    logger.info(f"  Providers: {sentiment['providers_count']}")

    logger.info(f"\nIndividual Indicators:")
    for indicator_name, indicator_data in sentiment['indicators'].items():
        logger.info(f"  {indicator_name}:")
        logger.info(f"    Value: {indicator_data['value']}")
        logger.info(f"    Classification: {indicator_data['classification']}")
        logger.info(f"    Source: {indicator_data['source']}")

    # Get trading signals
    logger.info(f"\nTrading Signals:")

    contrarian_signal = await sentiment_agg.get_sentiment_signal(symbol, strategy="contrarian")
    logger.info(f"  Contrarian Strategy: {contrarian_signal}")

    momentum_signal = await sentiment_agg.get_sentiment_signal(symbol, strategy="momentum")
    logger.info(f"  Momentum Strategy: {momentum_signal}")


async def demonstrate_economic_calendar():
    """Demonstrate economic calendar usage"""

    logger.info("\n" + "=" * 80)
    logger.info("ECONOMIC CALENDAR DEMONSTRATION")
    logger.info("=" * 80)

    # Create calendar aggregator
    calendar_agg = EconomicCalendarAggregator()

    # Fetch upcoming events
    logger.info(f"\nFetching upcoming events for next 48 hours...")

    events = await calendar_agg.get_events(
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(hours=48)
    )

    logger.info(f"\nFound {len(events)} events")

    # Filter high-impact events
    high_impact = [e for e in events if e['impact'] == 'high']

    logger.info(f"\nHigh-Impact Events ({len(high_impact)}):")
    for event in high_impact[:10]:  # Show first 10
        event_time = datetime.fromtimestamp(event['timestamp'] / 1000)
        logger.info(f"  {event_time.strftime('%Y-%m-%d %H:%M')} - {event['currency']} - {event['event']}")
        if event.get('forecast'):
            logger.info(f"    Forecast: {event['forecast']}, Previous: {event.get('previous', 'N/A')}")


async def demonstrate_risk_assessment():
    """Demonstrate risk assessment based on events"""

    logger.info("\n" + "=" * 80)
    logger.info("RISK ASSESSMENT DEMONSTRATION")
    logger.info("=" * 80)

    symbols = ["EUR/USD", "GBP/USD", "USD/JPY"]

    for symbol in symbols:
        logger.info(f"\nAssessing risk for {symbol}...")

        risk = await get_trading_risk(symbol, hours_ahead=24)

        logger.info(f"  Risk Score: {risk['risk_score']:.1f}/100")
        logger.info(f"  Risk Level: {risk['risk_level']}")
        logger.info(f"  High-Impact Events: {risk['high_impact_events_count']}")

        if risk['events']:
            logger.info(f"  Upcoming Events:")
            for event in risk['events'][:3]:  # Show first 3
                event_time = datetime.fromtimestamp(event['timestamp'] / 1000)
                hours_until = (event['timestamp'] / 1000 - datetime.now().timestamp()) / 3600
                logger.info(f"    {event['event']} ({event['currency']}) in {hours_until:.1f}h")


async def demonstrate_integrated_decision():
    """
    Demonstrate integrated trading decision using both sentiment and calendar.
    """

    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATED TRADING DECISION")
    logger.info("=" * 80)

    symbol = "EUR/USD"

    logger.info(f"\nMaking trading decision for {symbol}...")

    # 1. Get sentiment
    sentiment_agg = SentimentAggregator()
    sentiment = await sentiment_agg.get_composite_sentiment(symbol)

    # 2. Get risk from calendar
    risk = await get_trading_risk(symbol, hours_ahead=12)

    # 3. Make decision
    logger.info(f"\nDecision Factors:")
    logger.info(f"  Sentiment Score: {sentiment['composite_score']:.1f}/100 ({sentiment['classification']})")
    logger.info(f"  Event Risk Score: {risk['risk_score']:.1f}/100 ({risk['risk_level']})")

    # Decision logic
    sentiment_signal = "bullish" if sentiment['composite_score'] < 40 else "bearish" if sentiment['composite_score'] > 60 else "neutral"

    logger.info(f"\n  Sentiment Signal: {sentiment_signal}")

    # Risk-adjusted decision
    if risk['risk_score'] > 60:
        decision = "AVOID - High event risk"
        position_size = 0.0
    elif risk['risk_score'] > 30:
        decision = f"{sentiment_signal.upper()} - Reduced size due to medium risk"
        position_size = 0.5
    else:
        decision = f"{sentiment_signal.upper()} - Full size (low risk)"
        position_size = 1.0

    logger.info(f"\n  FINAL DECISION: {decision}")
    logger.info(f"  Position Size Multiplier: {position_size}")

    # Additional context
    if risk['events']:
        logger.info(f"\n  WARNING: {risk['high_impact_events_count']} high-impact events upcoming:")
        for event in risk['events'][:3]:
            event_time = datetime.fromtimestamp(event['timestamp'] / 1000)
            logger.info(f"    - {event['event']} ({event['currency']}) at {event_time.strftime('%H:%M')}")


async def main():
    """Run all demonstrations"""

    logger.info("Sentiment & Economic Calendar Integration Examples")
    logger.info("=" * 80)

    try:
        # 1. Sentiment analysis
        await demonstrate_sentiment()

        # 2. Economic calendar
        await demonstrate_economic_calendar()

        # 3. Risk assessment
        await demonstrate_risk_assessment()

        # 4. Integrated decision
        await demonstrate_integrated_decision()

        logger.info("\n" + "=" * 80)
        logger.info("✅ All demonstrations completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"❌ Demonstration failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
