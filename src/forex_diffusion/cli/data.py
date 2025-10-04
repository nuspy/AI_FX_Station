"""
CLI commands for data operations.

Usage:
    python -m forex_diffusion.cli.data backfill --provider tiingo --symbol EURUSD --days 30
    python -m forex_diffusion.cli.data vacuum
    python -m forex_diffusion.cli.data stats
"""
import sys
import argparse
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import create_engine

from ..providers import get_provider_manager
from ..credentials import get_credentials_manager
from ..utils.config import get_config


def data_cli():
    """Main CLI entry point for data commands."""
    parser = argparse.ArgumentParser(description="ForexGPT Data Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backfill command
    backfill_parser = subparsers.add_parser("backfill", help="Backfill historical data")
    backfill_parser.add_argument("--provider", required=True, choices=["tiingo", "ctrader", "alphavantage"],
                                  help="Data provider to use")
    backfill_parser.add_argument("--symbol", required=True, help="Symbol to backfill (e.g., EURUSD, EUR/USD)")
    backfill_parser.add_argument("--timeframe", default="1h", help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    backfill_parser.add_argument("--days", type=int, default=30, help="Number of days to backfill")
    backfill_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    backfill_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")

    # Vacuum command
    vacuum_parser = subparsers.add_parser("vacuum", help="Vacuum database to reclaim space")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument("--symbol", help="Filter by symbol")

    args = parser.parse_args()

    if args.command == "backfill":
        backfill_data(args)
    elif args.command == "vacuum":
        vacuum_database()
    elif args.command == "stats":
        show_stats(args.symbol)
    else:
        parser.print_help()
        sys.exit(1)


def backfill_data(args):
    """Backfill historical data."""
    try:
        import asyncio
        import pandas as pd

        manager = get_provider_manager()
        creds_manager = get_credentials_manager()
        config = get_config()

        print(f"\n=== Backfilling {args.symbol} {args.timeframe} from {args.provider} ===\n")

        # Load credentials
        creds = creds_manager.load(args.provider)
        if not creds:
            print(f"Error: {args.provider} not configured. Run 'python -m forex_diffusion.cli.providers add {args.provider}' first.")
            sys.exit(1)

        # Calculate date range
        if args.start_date and args.end_date:
            start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=args.days)

        start_ts_ms = int(start_dt.timestamp() * 1000)
        end_ts_ms = int(end_dt.timestamp() * 1000)

        print(f"Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        print(f"Provider: {args.provider}")

        # Fetch data
        async def fetch_data():
            config_dict = creds.to_dict()
            provider = manager.create_provider(args.provider, config=config_dict)

            print("\nConnecting to provider...")
            connected = await provider.connect()
            if not connected:
                print("✗ Connection failed")
                return None

            print("✓ Connected")

            print(f"\nFetching {args.symbol} {args.timeframe} data...")
            df = await provider.get_historical_bars(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_ts_ms=start_ts_ms,
                end_ts_ms=end_ts_ms
            )

            await provider.disconnect()
            return df

        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(fetch_data())

        if df is None or df.empty:
            print("✗ No data retrieved")
            sys.exit(1)

        print(f"✓ Retrieved {len(df)} bars")

        # Save to database
        print("\nSaving to database...")
        from ..data import io as data_io

        db_path = config.database.path if hasattr(config, 'database') else "./data/forex_diffusion.db"
        engine = create_engine(f"sqlite:///{db_path}")

        report = data_io.upsert_candles(engine, df, args.symbol, args.timeframe)
        print(f"✓ Saved: {report.get('inserted', 0)} new, {report.get('updated', 0)} updated, {report.get('skipped', 0)} skipped")

        print("\n✓ Backfill completed successfully")

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        sys.exit(1)


def vacuum_database():
    """Vacuum database to reclaim space."""
    try:
        config = get_config()
        db_path = config.database.path if hasattr(config, 'database') else "./data/forex_diffusion.db"

        print(f"\n=== Vacuuming Database ===\n")
        print(f"Database: {db_path}")

        engine = create_engine(f"sqlite:///{db_path}")

        with engine.begin() as conn:
            conn.execute("VACUUM")

        print("\n✓ Database vacuumed successfully")

    except Exception as e:
        logger.error(f"Vacuum failed: {e}")
        sys.exit(1)


def show_stats(symbol: str = None):
    """Show database statistics."""
    try:
        from sqlalchemy import text

        config = get_config()
        db_path = config.database.path if hasattr(config, 'database') else "./data/forex_diffusion.db"

        print(f"\n=== Database Statistics ===\n")
        print(f"Database: {db_path}")

        engine = create_engine(f"sqlite:///{db_path}")

        with engine.connect() as conn:
            # Candles count
            if symbol:
                query = text("SELECT timeframe, provider_source, COUNT(*) as count FROM market_data_candles WHERE symbol = :symbol GROUP BY timeframe, provider_source")
                rows = conn.execute(query, {"symbol": symbol}).fetchall()
                print(f"\nCandles for {symbol}:")
            else:
                query = text("SELECT symbol, timeframe, provider_source, COUNT(*) as count FROM market_data_candles GROUP BY symbol, timeframe, provider_source")
                rows = conn.execute(query).fetchall()
                print("\nCandles by symbol/timeframe/provider:")

            for row in rows:
                if symbol:
                    timeframe, provider, count = row
                    print(f"  {timeframe} ({provider}): {count:,} bars")
                else:
                    symbol_name, timeframe, provider, count = row
                    print(f"  {symbol_name} {timeframe} ({provider}): {count:,} bars")

            # Market depth
            query = text("SELECT symbol, provider, COUNT(*) as count FROM market_depth GROUP BY symbol, provider")
            rows = conn.execute(query).fetchall()
            if rows:
                print("\nMarket Depth snapshots:")
                for row in rows:
                    symbol_name, provider, count = row
                    print(f"  {symbol_name} ({provider}): {count:,} snapshots")

            # Sentiment
            query = text("SELECT symbol, provider, COUNT(*) as count FROM sentiment_data GROUP BY symbol, provider")
            rows = conn.execute(query).fetchall()
            if rows:
                print("\nSentiment data:")
                for row in rows:
                    symbol_name, provider, count = row
                    print(f"  {symbol_name} ({provider}): {count:,} points")

            # News
            query = text("SELECT provider, COUNT(*) as count FROM news_events GROUP BY provider")
            rows = conn.execute(query).fetchall()
            if rows:
                print("\nNews events:")
                for row in rows:
                    provider, count = row
                    print(f"  {provider}: {count:,} events")

            # Calendar
            query = text("SELECT provider, COUNT(*) as count FROM economic_calendar GROUP BY provider")
            rows = conn.execute(query).fetchall()
            if rows:
                print("\nEconomic calendar:")
                for row in rows:
                    provider, count = row
                    print(f"  {provider}: {count:,} events")

    except Exception as e:
        logger.error(f"Stats failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    data_cli()
