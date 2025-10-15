"""
Helper script to obtain cTrader OAuth access token.

This script helps you get an access token from cTrader using the OAuth 2.0 flow.

IMPORTANT SECURITY NOTE:
- This script is for DEMO/TESTING purposes only
- Never commit real credentials to version control
- Use environment variables or .env file for production
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from forex_diffusion.credentials.oauth import OAuth2Flow


async def get_token_interactive():
    """
    Interactive OAuth flow to get access token.

    This will:
    1. Open your browser
    2. Ask you to login to cTrader
    3. Return access token
    """

    print("=" * 70)
    print("cTrader OAuth 2.0 - Access Token Generator")
    print("=" * 70)
    print()

    # Get client credentials
    print("You need your cTrader Open API application credentials.")
    print("Get them from: https://openapi.ctrader.com/\n")

    client_id = input("Enter your Client ID: ").strip()
    if not client_id:
        print("❌ Client ID is required")
        return

    client_secret = input("Enter your Client Secret: ").strip()
    if not client_secret:
        print("❌ Client Secret is required")
        return

    print(f"\n✓ Client ID: {client_id}")
    print(f"✓ Client Secret: {client_secret[:10]}..." + "*" * 10)

    # Create OAuth flow
    try:
        oauth = OAuth2Flow(client_id, client_secret)
        print("\n🌐 Opening browser for authorization...")
        print("Please login with your cTrader credentials.")
        print()

        # Run authorization flow
        token_data = await oauth.authorize(auto_open_browser=True)

        print("\n" + "=" * 70)
        print("✅ Authorization Successful!")
        print("=" * 70)
        print()
        print(f"Access Token: {token_data['access_token']}")
        print(f"Refresh Token: {token_data['refresh_token']}")
        print(f"Expires In: {token_data['expires_in']} seconds")
        print(f"Token Type: {token_data.get('token_type', 'Bearer')}")
        print()
        print("=" * 70)
        print()

        # Save to .env template
        env_template = f"""
# Add these to your .env file or environment variables:

CTRADER_CLIENT_ID={client_id}
CTRADER_CLIENT_SECRET={client_secret}
CTRADER_ACCESS_TOKEN={token_data['access_token']}
CTRADER_REFRESH_TOKEN={token_data['refresh_token']}
CTRADER_ACCOUNT_ID=<your_trading_account_id>

# To set in PowerShell:
$env:CTRADER_CLIENT_ID = "{client_id}"
$env:CTRADER_CLIENT_SECRET = "{client_secret}"
$env:CTRADER_ACCESS_TOKEN = "{token_data['access_token']}"
$env:CTRADER_REFRESH_TOKEN = "{token_data['refresh_token']}"
$env:CTRADER_ACCOUNT_ID = "<your_account_id>"
"""

        print("Environment Variable Template:")
        print(env_template)

        # Optionally save to file
        save = input("\nSave to .env.ctrader file? (y/n): ").strip().lower()
        if save == 'y':
            env_file = project_root / ".env.ctrader"
            with open(env_file, 'w') as f:
                f.write(env_template)
            print(f"✓ Saved to {env_file}")
            print("⚠️  Remember: Do NOT commit this file to git!")

        print("\n✓ Done! You can now use these credentials to test the WebSocket.")
        return token_data

    except Exception as e:
        logger.exception(f"❌ OAuth flow failed: {e}")
        return None


async def get_token_with_credentials(username: str, password: str, client_id: str, client_secret: str):
    """
    ALTERNATIVE: Try to get token using username/password.

    Note: cTrader typically requires OAuth browser flow,
    but this function attempts direct authentication if supported.
    """
    print("=" * 70)
    print("Attempting direct authentication...")
    print("=" * 70)
    print()

    # NOTE: cTrader may not support this method
    # Standard OAuth 2.0 requires browser-based authorization

    print("⚠️  WARNING: cTrader typically requires browser-based OAuth flow.")
    print("This method may not work. Use get_token_interactive() instead.\n")

    # Try Resource Owner Password Credentials flow (if supported)
    try:
        import httpx

        token_url = "https://openapi.ctrader.com/apps/token"

        data = {
            'grant_type': 'password',
            'username': username,
            'password': password,
            'client_id': client_id,
            'client_secret': client_secret,
            'scope': 'trading'
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )

            if response.status_code == 200:
                token_data = response.json()
                print("✅ Success!")
                print(f"Access Token: {token_data['access_token']}")
                return token_data
            else:
                print(f"❌ Failed: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                print("\n💡 Try using the interactive browser flow instead.")
                return None

    except Exception as e:
        logger.exception(f"❌ Direct auth failed: {e}")
        print("\n💡 Use the interactive OAuth flow with get_token_interactive()")
        return None


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    print("""
Choose authentication method:

1. Interactive OAuth (recommended)
   - Opens browser for login
   - Most secure
   - Works with all cTrader accounts

2. Direct with username/password (may not work)
   - Requires password grant support
   - May be disabled by cTrader

Enter your choice (1 or 2): """, end="")

    choice = input().strip()

    if choice == "1":
        # Interactive OAuth
        asyncio.run(get_token_interactive())

    elif choice == "2":
        # Direct authentication (may not work)
        print("\nEnter your cTrader credentials:")
        username = input("Username/Email: ").strip()
        password = input("Password: ").strip()
        client_id = input("Client ID: ").strip()
        client_secret = input("Client Secret: ").strip()

        asyncio.run(get_token_with_credentials(username, password, client_id, client_secret))

    else:
        print("Invalid choice")
        sys.exit(1)
