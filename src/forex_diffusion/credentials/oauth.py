"""
OAuth 2.0 flow implementation for cTrader authorization.
"""

from __future__ import annotations

import asyncio
import secrets
import webbrowser
from typing import Optional, Dict, Any
from urllib.parse import urlencode, parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from loguru import logger

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False
    logger.warning("httpx not installed. OAuth flow will not work.")


class OAuth2CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    authorization_code: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self):
        """Handle GET request for OAuth callback."""
        # Parse query parameters
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        # Extract authorization code
        if "code" in params:
            OAuth2CallbackHandler.authorization_code = params["code"][0]

        if "state" in params:
            OAuth2CallbackHandler.state = params["state"][0]

        if "error" in params:
            OAuth2CallbackHandler.error = params["error"][0]

        # Send response
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        if OAuth2CallbackHandler.authorization_code:
            html = """
            <html>
                <head><title>Authorization Successful</title></head>
                <body>
                    <h1>✓ Authorization Successful</h1>
                    <p>You can close this window and return to ForexGPT.</p>
                </body>
            </html>
            """
        else:
            html = f"""
            <html>
                <head><title>Authorization Failed</title></head>
                <body>
                    <h1>✗ Authorization Failed</h1>
                    <p>Error: {OAuth2CallbackHandler.error or 'Unknown error'}</p>
                    <p>Please try again.</p>
                </body>
            </html>
            """

        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """Suppress HTTP server logs."""
        pass


class OAuth2Flow:
    """
    OAuth 2.0 Authorization Code Flow for cTrader.

    Implements:
    - Authorization URL generation
    - Localhost callback server
    - Code exchange for access token
    - Token refresh
    """

    # cTrader OAuth endpoints
    AUTH_URL = "https://openapi.ctrader.com/apps/auth"
    TOKEN_URL = "https://openapi.ctrader.com/apps/token"

    # Localhost callback
    REDIRECT_HOST = "localhost"
    REDIRECT_PORT = 5000

    def __init__(self, client_id: str, client_secret: str):
        """
        Initialize OAuth flow.

        Args:
            client_id: cTrader application client ID
            client_secret: cTrader application client secret
        """
        if not _HAS_HTTPX:
            raise ImportError("httpx package required. Install with: pip install httpx")

        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = f"http://{self.REDIRECT_HOST}:{self.REDIRECT_PORT}/callback"

        self._state: Optional[str] = None
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    def get_authorization_url(self, scopes: Optional[list] = None) -> str:
        """
        Generate authorization URL for user to visit.

        Args:
            scopes: List of requested scopes (default: ["trading"])

        Returns:
            Authorization URL
        """
        if scopes is None:
            scopes = ["trading"]

        # Generate random state for CSRF protection
        self._state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(scopes),
            "response_type": "code",
            "state": self._state,
        }

        url = f"{self.AUTH_URL}?{urlencode(params)}"
        return url

    def start_callback_server(self) -> None:
        """Start local HTTP server to receive OAuth callback."""
        # Reset handler state
        OAuth2CallbackHandler.authorization_code = None
        OAuth2CallbackHandler.state = None
        OAuth2CallbackHandler.error = None

        # Create server
        self._server = HTTPServer(
            (self.REDIRECT_HOST, self.REDIRECT_PORT),
            OAuth2CallbackHandler
        )

        # Run in separate thread
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True
        )
        self._server_thread.start()

        logger.info(f"OAuth callback server started on {self.redirect_uri}")

    def stop_callback_server(self) -> None:
        """Stop local callback server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._server_thread = None
            logger.info("OAuth callback server stopped")

    async def wait_for_authorization_code(self, timeout: int = 300) -> str:
        """
        Wait for authorization code from callback.

        Args:
            timeout: Maximum wait time in seconds (default: 5 minutes)

        Returns:
            Authorization code

        Raises:
            TimeoutError: If no code received within timeout
            ValueError: If state mismatch or error in callback
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check for code
            if OAuth2CallbackHandler.authorization_code:
                # Verify state
                if OAuth2CallbackHandler.state != self._state:
                    raise ValueError("State mismatch - possible CSRF attack")

                code = OAuth2CallbackHandler.authorization_code
                logger.info("Authorization code received")
                return code

            # Check for error
            if OAuth2CallbackHandler.error:
                raise ValueError(f"Authorization error: {OAuth2CallbackHandler.error}")

            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError("Authorization timeout - no code received")

            # Wait a bit
            await asyncio.sleep(0.5)

    async def exchange_code_for_token(self, authorization_code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.

        Args:
            authorization_code: Code received from authorization

        Returns:
            Dict with keys: access_token, refresh_token, expires_in, token_type
        """
        data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.TOKEN_URL,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            response.raise_for_status()
            token_data = response.json()

        logger.info("Successfully exchanged code for access token")
        return token_data

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            Dict with keys: access_token, refresh_token, expires_in, token_type
        """
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.TOKEN_URL,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            response.raise_for_status()
            token_data = response.json()

        logger.info("Successfully refreshed access token")
        return token_data

    async def authorize(self, auto_open_browser: bool = True) -> Dict[str, Any]:
        """
        Complete OAuth flow: open browser, wait for callback, exchange code.

        Args:
            auto_open_browser: Automatically open browser (default: True)

        Returns:
            Token data dict
        """
        try:
            # Start callback server
            self.start_callback_server()

            # Generate auth URL
            auth_url = self.get_authorization_url()
            logger.info(f"Authorization URL: {auth_url}")

            # Open browser
            if auto_open_browser:
                webbrowser.open(auth_url)
                logger.info("Opened browser for authorization")
            else:
                print(f"\nPlease visit this URL to authorize:\n{auth_url}\n")

            # Wait for callback
            code = await self.wait_for_authorization_code()

            # Exchange code for token
            token_data = await self.exchange_code_for_token(code)

            return token_data

        finally:
            # Always stop server
            self.stop_callback_server()
