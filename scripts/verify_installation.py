"""
Verification script for multi-provider installation.

Run this script to verify that the multi-provider system is correctly installed.

Usage:
    python scripts/verify_installation.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

print("\n" + "="*60)
print("ForexGPT Multi-Provider Installation Verification")
print("="*60 + "\n")

# Track verification status
all_passed = True

# 1. Check imports
print("1. Checking imports...")
try:
    from forex_diffusion.providers import (
        BaseProvider,
        ProviderCapability,
        ProviderManager,
        TiingoProvider,
        CTraderProvider,
        get_provider_manager
    )
    print("   ✓ Providers module imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import providers: {e}")
    all_passed = False

try:
    from forex_diffusion.credentials import (
        CredentialsManager,
        OAuth2Flow,
        ProviderCredentials,
        get_credentials_manager
    )
    print("   ✓ Credentials module imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import credentials: {e}")
    all_passed = False

try:
    from forex_diffusion.services.aggregator import AggregatorService
    from forex_diffusion.services.dom_aggregator import DOMAggreg atorService
    from forex_diffusion.services.sentiment_aggregator import SentimentAggregatorService
    print("   ✓ Aggregators imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import aggregators: {e}")
    all_passed = False

# 2. Check provider availability
print("\n2. Checking provider availability...")
try:
    manager = get_provider_manager()
    available = manager.get_available_providers()
    print(f"   ✓ Available providers: {', '.join(available)}")

    if 'tiingo' in available:
        print("   ✓ Tiingo provider registered")
    else:
        print("   ✗ Tiingo provider missing")
        all_passed = False

    if 'ctrader' in available:
        print("   ✓ cTrader provider registered")
    else:
        print("   ✗ cTrader provider missing")
        all_passed = False

except Exception as e:
    print(f"   ✗ Failed to check providers: {e}")
    all_passed = False

# 3. Check provider capabilities
print("\n3. Checking provider capabilities...")
try:
    manager = get_provider_manager()

    # Test Tiingo
    tiingo = manager.create_provider("tiingo", config={"api_key": "test"})
    tiingo_caps = [cap.name for cap in tiingo.capabilities]
    print(f"   ✓ Tiingo capabilities: {', '.join(tiingo_caps[:3])}...")

    # Test cTrader
    ctrader = manager.create_provider("ctrader", config={
        "client_id": "test",
        "client_secret": "test"
    })
    ctrader_caps = [cap.name for cap in ctrader.capabilities]
    print(f"   ✓ cTrader capabilities: {', '.join(ctrader_caps[:3])}...")

except Exception as e:
    print(f"   ✗ Failed to check capabilities: {e}")
    all_passed = False

# 4. Check database migration
print("\n4. Checking database schema...")
try:
    from sqlalchemy import create_engine, inspect

    db_path = project_root / "data" / "forex_diffusion.db"
    if db_path.exists():
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        required_tables = [
            'market_data_candles',
            'market_depth',
            'sentiment_data',
            'news_events',
            'economic_calendar'
        ]

        all_tables_exist = True
        for table in required_tables:
            if table in tables:
                print(f"   ✓ Table '{table}' exists")
            else:
                print(f"   ✗ Table '{table}' missing - run 'alembic upgrade head'")
                all_tables_exist = False
                all_passed = False

        if all_tables_exist:
            # Check for new columns in market_data_candles
            columns = [col['name'] for col in inspector.get_columns('market_data_candles')]
            if 'tick_volume' in columns and 'real_volume' in columns and 'provider_source' in columns:
                print("   ✓ market_data_candles has new columns (tick_volume, real_volume, provider_source)")
            else:
                print("   ✗ market_data_candles missing new columns - run 'alembic upgrade head'")
                all_passed = False
    else:
        print(f"   ! Database not found at {db_path}")
        print("     Run 'alembic upgrade head' to create it")

except Exception as e:
    print(f"   ✗ Failed to check database: {e}")
    all_passed = False

# 5. Check credentials manager
print("\n5. Checking credentials manager...")
try:
    creds_manager = get_credentials_manager()
    print("   ✓ CredentialsManager initialized")

    # Check if keyring is available
    import keyring
    print(f"   ✓ Keyring backend: {keyring.get_keyring().__class__.__name__}")

except Exception as e:
    print(f"   ✗ Failed to initialize credentials manager: {e}")
    all_passed = False

# 6. Check CLI commands
print("\n6. Checking CLI commands...")
try:
    from forex_diffusion.cli import provider_cli, data_cli
    print("   ✓ CLI commands available")
    print("     - python -m forex_diffusion.cli.providers <command>")
    print("     - python -m forex_diffusion.cli.data <command>")
except Exception as e:
    print(f"   ✗ Failed to import CLI: {e}")
    all_passed = False

# 7. Check GUI components
print("\n7. Checking GUI components...")
try:
    from forex_diffusion.ui.settings_dialog import SettingsDialog
    from forex_diffusion.ui.news_calendar_tab import NewsCalendarTab
    print("   ✓ GUI components available")
    print("     - SettingsDialog (with provider configuration)")
    print("     - NewsCalendarTab (news and economic calendar)")
except Exception as e:
    print(f"   ✗ Failed to import GUI components: {e}")
    all_passed = False

# 8. Check documentation
print("\n8. Checking documentation...")
docs_dir = project_root / "docs"
if docs_dir.exists():
    required_docs = ["ARCHITECTURE.md", "PROVIDERS.md", "DATABASE.md", "DECISIONS.md"]
    for doc in required_docs:
        doc_path = docs_dir / doc
        if doc_path.exists():
            print(f"   ✓ {doc} exists")
        else:
            print(f"   ✗ {doc} missing")
            all_passed = False
else:
    print("   ✗ docs/ directory not found")
    all_passed = False

# Final summary
print("\n" + "="*60)
if all_passed:
    print("✅ ALL CHECKS PASSED - Installation verified successfully!")
    print("\nNext steps:")
    print("1. Run 'alembic upgrade head' to apply database migration")
    print("2. Configure providers:")
    print("   python -m forex_diffusion.cli.providers add tiingo")
    print("   python -m forex_diffusion.cli.providers add ctrader")
    print("3. Test connections:")
    print("   python -m forex_diffusion.cli.providers test tiingo")
    print("4. Launch GUI:")
    print("   python -m forex_diffusion.ui.main")
else:
    print("❌ SOME CHECKS FAILED - Please review errors above")
    print("\nCommon fixes:")
    print("- Run 'pip install -e .' to install dependencies")
    print("- Run 'alembic upgrade head' to create database")
    print("- Check that all files were committed and pulled")

print("="*60 + "\n")

sys.exit(0 if all_passed else 1)
