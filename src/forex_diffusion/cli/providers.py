"""
CLI commands for provider management.

Usage:
    python -m forex_diffusion.cli.providers list
    python -m forex_diffusion.cli.providers add <provider_name>
    python -m forex_diffusion.cli.providers test <provider_name>
    python -m forex_diffusion.cli.providers delete <provider_name>
"""
import sys
import asyncio
from typing import Optional

from loguru import logger

from ..providers import get_provider_manager, ProviderCapability
from ..credentials import get_credentials_manager, OAuth2Flow, ProviderCredentials


def provider_cli():
    """Main CLI entry point for provider commands."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        list_providers()
    elif command == "add":
        if len(sys.argv) < 3:
            print("Error: provider name required")
            print_usage()
            sys.exit(1)
        add_provider(sys.argv[2])
    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: provider name required")
            print_usage()
            sys.exit(1)
        test_provider(sys.argv[2])
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Error: provider name required")
            print_usage()
            sys.exit(1)
        delete_provider(sys.argv[2])
    elif command == "capabilities":
        if len(sys.argv) < 3:
            print("Error: provider name required")
            print_usage()
            sys.exit(1)
        show_capabilities(sys.argv[2])
    else:
        print(f"Error: Unknown command '{command}'")
        print_usage()
        sys.exit(1)


def print_usage():
    """Print CLI usage."""
    print("""
ForexGPT Provider Management CLI

Usage:
    python -m forex_diffusion.cli.providers <command> [args]

Commands:
    list                    List available providers and their status
    add <provider>          Add/configure a provider (runs OAuth for cTrader)
    test <provider>         Test provider connection
    delete <provider>       Remove provider credentials
    capabilities <provider> Show provider capabilities

Examples:
    python -m forex_diffusion.cli.providers list
    python -m forex_diffusion.cli.providers add ctrader
    python -m forex_diffusion.cli.providers test tiingo
    python -m forex_diffusion.cli.providers delete ctrader
    python -m forex_diffusion.cli.providers capabilities ctrader
""")


def list_providers():
    """List all available providers."""
    try:
        manager = get_provider_manager()
        creds_manager = get_credentials_manager()

        print("\n=== Available Providers ===\n")

        providers = ["tiingo", "ctrader", "alphavantage"]

        for provider_name in providers:
            # Check if credentials exist
            creds = creds_manager.load(provider_name)
            status = "✓ Configured" if creds else "✗ Not configured"

            print(f"{provider_name.upper()}")
            print(f"  Status: {status}")

            if creds:
                print(f"  Environment: {getattr(creds, 'environment', 'N/A')}")

            # Try to create provider and show capabilities
            try:
                if creds:
                    config = creds.to_dict()
                    provider = manager.create_provider(provider_name, config=config)
                    caps = [cap.name for cap in provider.capabilities]
                    print(f"  Capabilities: {', '.join(caps)}")
                else:
                    print(f"  Capabilities: (configure to view)")
            except Exception as e:
                print(f"  Error: {e}")

            print()

    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        sys.exit(1)


def add_provider(provider_name: str):
    """Add/configure a provider."""
    try:
        creds_manager = get_credentials_manager()
        provider_name = provider_name.lower()

        if provider_name == "ctrader":
            print("\n=== cTrader OAuth Setup ===\n")

            # Get credentials from user
            client_id = input("Enter cTrader Client ID: ").strip()
            client_secret = input("Enter cTrader Client Secret: ").strip()
            environment = input("Enter environment (demo/live) [demo]: ").strip() or "demo"

            if not client_id or not client_secret:
                print("Error: Client ID and Secret are required")
                sys.exit(1)

            print("\nStarting OAuth flow...")
            print("A browser window will open for authorization.")

            # Run OAuth flow
            async def run_oauth():
                oauth = OAuth2Flow(client_id=client_id, client_secret=client_secret)
                token_data = await oauth.authorize()

                creds = ProviderCredentials(
                    provider_name='ctrader',
                    client_id=client_id,
                    client_secret=client_secret,
                    access_token=token_data['access_token'],
                    refresh_token=token_data.get('refresh_token'),
                    environment=environment
                )

                creds_manager.save(creds)
                return True

            loop = asyncio.get_event_loop()
            success = loop.run_until_complete(run_oauth())

            if success:
                print("\n✓ cTrader configured successfully!")
                print(f"  Environment: {environment}")
                print("  Credentials saved securely in OS keyring")

        elif provider_name == "tiingo":
            print("\n=== Tiingo Setup ===\n")

            api_key = input("Enter Tiingo API Key: ").strip()

            if not api_key:
                print("Error: API Key is required")
                sys.exit(1)

            creds = ProviderCredentials(
                provider_name='tiingo',
                api_key=api_key
            )

            creds_manager.save(creds)
            print("\n✓ Tiingo configured successfully!")

        elif provider_name == "alphavantage":
            print("\n=== AlphaVantage Setup ===\n")

            api_key = input("Enter AlphaVantage API Key: ").strip()

            if not api_key:
                print("Error: API Key is required")
                sys.exit(1)

            creds = ProviderCredentials(
                provider_name='alphavantage',
                api_key=api_key
            )

            creds_manager.save(creds)
            print("\n✓ AlphaVantage configured successfully!")

        else:
            print(f"Error: Unknown provider '{provider_name}'")
            print("Supported providers: tiingo, ctrader, alphavantage")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to add provider: {e}")
        sys.exit(1)


def test_provider(provider_name: str):
    """Test provider connection."""
    try:
        manager = get_provider_manager()
        creds_manager = get_credentials_manager()
        provider_name = provider_name.lower()

        print(f"\n=== Testing {provider_name.upper()} ===\n")

        # Load credentials
        creds = creds_manager.load(provider_name)
        if not creds:
            print(f"Error: {provider_name} not configured. Run 'add {provider_name}' first.")
            sys.exit(1)

        # Test connection
        async def test_connection():
            config = creds.to_dict()
            provider = manager.create_provider(provider_name, config=config)

            print("Connecting...")
            connected = await provider.connect()

            if not connected:
                print("✗ Connection failed")
                return False

            print("✓ Connected successfully")

            # Try to get a test quote
            print("\nTesting data retrieval...")
            try:
                price = await provider.get_current_price("EUR/USD")
                if price:
                    print(f"✓ Retrieved EUR/USD quote: {price.get('price', 'N/A')}")
                    print(f"  Bid: {price.get('bid', 'N/A')}")
                    print(f"  Ask: {price.get('ask', 'N/A')}")
                else:
                    print("✗ Failed to retrieve quote")
            except Exception as e:
                print(f"✗ Data retrieval error: {e}")

            await provider.disconnect()
            return True

        loop = asyncio.get_event_loop()
        success = loop.run_until_complete(test_connection())

        if success:
            print("\n✓ Provider test completed successfully")
        else:
            print("\n✗ Provider test failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to test provider: {e}")
        sys.exit(1)


def delete_provider(provider_name: str):
    """Delete provider credentials."""
    try:
        creds_manager = get_credentials_manager()
        provider_name = provider_name.lower()

        creds = creds_manager.load(provider_name)
        if not creds:
            print(f"Error: {provider_name} not configured")
            sys.exit(1)

        confirm = input(f"Delete {provider_name} credentials? (yes/no): ").strip().lower()

        if confirm == "yes":
            creds_manager.delete(provider_name)
            print(f"\n✓ {provider_name} credentials deleted")
        else:
            print("Cancelled")

    except Exception as e:
        logger.error(f"Failed to delete provider: {e}")
        sys.exit(1)


def show_capabilities(provider_name: str):
    """Show provider capabilities."""
    try:
        manager = get_provider_manager()
        creds_manager = get_credentials_manager()
        provider_name = provider_name.lower()

        print(f"\n=== {provider_name.upper()} Capabilities ===\n")

        # Load credentials (may not be required for capability check)
        creds = creds_manager.load(provider_name)
        config = creds.to_dict() if creds else {}

        try:
            provider = manager.create_provider(provider_name, config=config)
            capabilities = provider.capabilities

            for cap in ProviderCapability:
                supported = cap in capabilities
                status = "✓" if supported else "✗"
                print(f"{status} {cap.name}")

        except Exception as e:
            print(f"Error: {e}")

    except Exception as e:
        logger.error(f"Failed to show capabilities: {e}")
        sys.exit(1)


if __name__ == "__main__":
    provider_cli()
