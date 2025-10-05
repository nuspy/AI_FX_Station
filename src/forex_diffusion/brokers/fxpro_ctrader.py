"""
FxPro cTrader Broker Integration

Official FxPro cTrader REST API integration with OAuth2 authentication.

Features:
- OAuth2 authentication flow
- Symbol mapping (FxPro specific naming)
- Order management (market, limit, stop orders)
- Position tracking
- Real-time account info
- Multi-account support
"""
from __future__ import annotations

import time
import json
import hashlib
import secrets
import base64
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode, parse_qs, urlparse

import requests
from loguru import logger

from .base import (
    BrokerBase, Order, Position, AccountInfo,
    OrderType, OrderSide, OrderStatus, PositionSide
)
from ..utils.user_settings import SETTINGS_DIR


# FxPro cTrader API endpoints
FXPRO_OAUTH_BASE = "https://openapi.ctrader.com"
FXPRO_API_BASE = "https://api.ctrader.com"

# FxPro-specific symbol mapping
# Maps standard symbols (e.g., "EUR/USD") to FxPro cTrader symbols
FXPRO_SYMBOL_MAP = {
    # Major Forex pairs
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/JPY": "USDJPY",
    "USD/CHF": "USDCHF",
    "AUD/USD": "AUDUSD",
    "USD/CAD": "USDCAD",
    "NZD/USD": "NZDUSD",

    # Cross pairs
    "EUR/GBP": "EURGBP",
    "EUR/JPY": "EURJPY",
    "EUR/CHF": "EURCHF",
    "EUR/AUD": "EURAUD",
    "EUR/CAD": "EURCAD",
    "EUR/NZD": "EURNZD",
    "GBP/JPY": "GBPJPY",
    "GBP/CHF": "GBPCHF",
    "GBP/AUD": "GBPAUD",
    "GBP/CAD": "GBPCAD",
    "GBP/NZD": "GBPNZD",
    "AUD/JPY": "AUDJPY",
    "AUD/CHF": "AUDCHF",
    "AUD/CAD": "AUDCAD",
    "AUD/NZD": "AUDNZD",
    "CAD/JPY": "CADJPY",
    "CAD/CHF": "CADCHF",
    "CHF/JPY": "CHFJPY",
    "NZD/JPY": "NZDJPY",
    "NZD/CHF": "NZDCHF",
    "NZD/CAD": "NZDCAD",

    # Exotic pairs
    "USD/SEK": "USDSEK",
    "USD/NOK": "USDNOK",
    "USD/DKK": "USDDKK",
    "USD/ZAR": "USDZAR",
    "USD/TRY": "USDTRY",
    "USD/MXN": "USDMXN",
    "USD/SGD": "USDSGD",
    "USD/HKD": "USDHKD",
    "USD/CNH": "USDCNH",

    # Commodities (CFDs)
    "XAU/USD": "XAUUSD",  # Gold
    "XAG/USD": "XAGUSD",  # Silver
    "XPT/USD": "XPTUSD",  # Platinum
    "XPD/USD": "XPDUSD",  # Palladium

    # Indices (CFDs)
    "US30": "US30",       # Dow Jones
    "US500": "US500",     # S&P 500
    "US100": "US100",     # Nasdaq
    "DE30": "DE30",       # DAX
    "UK100": "UK100",     # FTSE 100
    "JP225": "JP225",     # Nikkei

    # Crypto (CFDs)
    "BTC/USD": "BTCUSD",
    "ETH/USD": "ETHUSD",
    "LTC/USD": "LTCUSD",
    "XRP/USD": "XRPUSD",
}

# Reverse mapping
FXPRO_SYMBOL_REVERSE_MAP = {v: k for k, v in FXPRO_SYMBOL_MAP.items()}


class FxProCTraderBroker(BrokerBase):
    """
    FxPro cTrader broker implementation with OAuth2.

    Authentication Flow:
    1. User authorizes via browser (OAuth2 Authorization Code flow)
    2. Exchange authorization code for access token
    3. Use access token for API requests
    4. Refresh token when expired
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "http://localhost:8080/callback",
        account_id: Optional[str] = None,
        credentials_file: Optional[Path] = None
    ):
        """
        Initialize FxPro cTrader broker.

        Args:
            client_id: OAuth2 client ID (from FxPro developer portal)
            client_secret: OAuth2 client secret
            redirect_uri: OAuth2 redirect URI
            account_id: Trading account ID (if multiple accounts)
            credentials_file: Path to store OAuth2 tokens
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.account_id = account_id

        self.credentials_file = credentials_file or (SETTINGS_DIR / "fxpro_credentials.json")

        # OAuth2 tokens
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None

        # Session for API requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ForexGPT/1.0',
            'Accept': 'application/json',
        })

        # Connection state
        self._connected = False

        # Load saved credentials
        self._load_credentials()

    def _load_credentials(self):
        """Load OAuth2 credentials from file"""
        try:
            if self.credentials_file.exists():
                with open(self.credentials_file, 'r') as f:
                    data = json.load(f)
                    self.access_token = data.get('access_token')
                    self.refresh_token = data.get('refresh_token')

                    expiry_str = data.get('token_expiry')
                    if expiry_str:
                        self.token_expiry = datetime.fromisoformat(expiry_str)

                    self.account_id = data.get('account_id', self.account_id)

                logger.info("Loaded FxPro credentials from file")
        except Exception as e:
            logger.warning(f"Failed to load FxPro credentials: {e}")

    def _save_credentials(self):
        """Save OAuth2 credentials to file"""
        try:
            self.credentials_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'access_token': self.access_token,
                'refresh_token': self.refresh_token,
                'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None,
                'account_id': self.account_id,
            }

            with open(self.credentials_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info("Saved FxPro credentials to file")
        except Exception as e:
            logger.error(f"Failed to save FxPro credentials: {e}")

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """
        Get OAuth2 authorization URL for user to visit.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL
        """
        state = state or secrets.token_urlsafe(32)

        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'trading',
            'state': state,
        }

        url = f"{FXPRO_OAUTH_BASE}/oauth/authorize?{urlencode(params)}"
        return url

    def exchange_code_for_token(self, code: str) -> bool:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth2 callback

        Returns:
            bool: True if successful
        """
        try:
            data = {
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': self.redirect_uri,
                'client_id': self.client_id,
                'client_secret': self.client_secret,
            }

            response = self.session.post(
                f"{FXPRO_OAUTH_BASE}/oauth/token",
                data=data
            )
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data['access_token']
            self.refresh_token = token_data.get('refresh_token')

            expires_in = token_data.get('expires_in', 3600)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

            self._save_credentials()
            logger.info("Successfully obtained access token")
            return True

        except Exception as e:
            logger.error(f"Failed to exchange code for token: {e}")
            return False

    def refresh_access_token(self) -> bool:
        """
        Refresh the access token using refresh token.

        Returns:
            bool: True if successful
        """
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False

        try:
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
                'client_id': self.client_id,
                'client_secret': self.client_secret,
            }

            response = self.session.post(
                f"{FXPRO_OAUTH_BASE}/oauth/token",
                data=data
            )
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data['access_token']

            if 'refresh_token' in token_data:
                self.refresh_token = token_data['refresh_token']

            expires_in = token_data.get('expires_in', 3600)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

            self._save_credentials()
            logger.info("Successfully refreshed access token")
            return True

        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            return False

    def _ensure_token_valid(self) -> bool:
        """Ensure access token is valid, refresh if needed"""
        if not self.access_token:
            logger.error("No access token - please authenticate first")
            return False

        # Check if token is expired or about to expire (within 5 minutes)
        if self.token_expiry and datetime.now() >= self.token_expiry - timedelta(minutes=5):
            logger.info("Access token expired or expiring soon, refreshing...")
            return self.refresh_access_token()

        return True

    def _api_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Make authenticated API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Form data
            json_data: JSON data

        Returns:
            Response JSON or None if error
        """
        if not self._ensure_token_valid():
            return None

        headers = {
            'Authorization': f'Bearer {self.access_token}',
        }

        url = f"{FXPRO_API_BASE}{endpoint}"

        try:
            response = self.session.request(
                method,
                url,
                params=params,
                data=data,
                json=json_data,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"API request failed: {e} - {e.response.text if e.response else ''}")
            return None
        except Exception as e:
            logger.error(f"API request error: {e}")
            return None

    def connect(self) -> bool:
        """Connect to FxPro cTrader API"""
        if not self.access_token:
            logger.error("No access token - please authenticate first using get_authorization_url()")
            return False

        # Test connection by getting account info
        account_info = self.get_account_info()
        if account_info:
            self._connected = True
            logger.info(f"Connected to FxPro cTrader - Account: {account_info.account_id}")
            return True

        return False

    def disconnect(self) -> bool:
        """Disconnect from FxPro cTrader API"""
        self._connected = False
        logger.info("Disconnected from FxPro cTrader")
        return True

    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connected and self.access_token is not None

    def normalize_symbol(self, symbol: str) -> str:
        """Convert standard symbol to FxPro format"""
        # Try exact match first
        if symbol in FXPRO_SYMBOL_MAP:
            return FXPRO_SYMBOL_MAP[symbol]

        # Try without slash
        symbol_no_slash = symbol.replace('/', '')
        if symbol_no_slash in FXPRO_SYMBOL_MAP.values():
            return symbol_no_slash

        # Default: remove slash and uppercase
        return symbol.replace('/', '').upper()

    def denormalize_symbol(self, symbol: str) -> str:
        """Convert FxPro symbol to standard format"""
        if symbol in FXPRO_SYMBOL_REVERSE_MAP:
            return FXPRO_SYMBOL_REVERSE_MAP[symbol]

        # Default: add slash for 6-character forex pairs
        if len(symbol) == 6 and symbol.isalpha():
            return f"{symbol[:3]}/{symbol[3:]}"

        return symbol

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        endpoint = f"/v3/accounts/{self.account_id}" if self.account_id else "/v3/accounts"

        response = self._api_request('GET', endpoint)
        if not response:
            return None

        # If multiple accounts, use first one or specified account_id
        if isinstance(response, list):
            account_data = response[0] if response else None
        else:
            account_data = response

        if not account_data:
            return None

        try:
            return AccountInfo(
                account_id=str(account_data.get('accountId', '')),
                balance=float(account_data.get('balance', 0)),
                equity=float(account_data.get('equity', 0)),
                margin_used=float(account_data.get('marginUsed', 0)),
                margin_available=float(account_data.get('freeMargin', 0)),
                currency=account_data.get('currency', 'USD'),
                leverage=int(account_data.get('leverage', 1)),
                unrealized_pnl=float(account_data.get('unrealizedPnL', 0)),
            )
        except Exception as e:
            logger.error(f"Failed to parse account info: {e}")
            return None

    def place_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[str]]:
        """Place an order"""
        endpoint = f"/v3/accounts/{self.account_id}/orders"

        # Convert to FxPro symbol
        symbol = self.normalize_symbol(order.symbol)

        # Build order payload
        payload = {
            'symbolName': symbol,
            'orderType': self._convert_order_type(order.order_type),
            'tradeSide': 'BUY' if order.side == OrderSide.BUY else 'SELL',
            'volume': int(order.quantity * 100000),  # Convert to units (1 lot = 100k)
        }

        if order.price:
            payload['limitPrice'] = order.price

        if order.stop_price:
            payload['stopPrice'] = order.stop_price

        if order.take_profit:
            payload['takeProfitPrice'] = order.take_profit

        if order.stop_loss:
            payload['stopLossPrice'] = order.stop_loss

        if order.comment:
            payload['comment'] = order.comment

        response = self._api_request('POST', endpoint, json_data=payload)

        if response and 'orderId' in response:
            order_id = str(response['orderId'])
            logger.info(f"Order placed successfully: {order_id}")
            return True, order_id, None
        else:
            error_msg = response.get('description', 'Unknown error') if response else 'API request failed'
            logger.error(f"Failed to place order: {error_msg}")
            return False, None, error_msg

    def cancel_order(self, order_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel an order"""
        endpoint = f"/v3/accounts/{self.account_id}/orders/{order_id}"

        response = self._api_request('DELETE', endpoint)

        if response:
            logger.info(f"Order cancelled: {order_id}")
            return True, None
        else:
            return False, "Failed to cancel order"

    def modify_order(self, order_id: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """Modify an order"""
        endpoint = f"/v3/accounts/{self.account_id}/orders/{order_id}"

        payload = {}

        if 'price' in kwargs:
            payload['limitPrice'] = kwargs['price']
        if 'stop_price' in kwargs:
            payload['stopPrice'] = kwargs['stop_price']
        if 'take_profit' in kwargs:
            payload['takeProfitPrice'] = kwargs['take_profit']
        if 'stop_loss' in kwargs:
            payload['stopLossPrice'] = kwargs['stop_loss']

        response = self._api_request('PATCH', endpoint, json_data=payload)

        if response:
            logger.info(f"Order modified: {order_id}")
            return True, None
        else:
            return False, "Failed to modify order"

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details"""
        endpoint = f"/v3/accounts/{self.account_id}/orders/{order_id}"

        response = self._api_request('GET', endpoint)

        if response:
            return self._parse_order(response)
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders"""
        endpoint = f"/v3/accounts/{self.account_id}/orders"

        params = {'status': 'ACTIVE'}
        if symbol:
            params['symbolName'] = self.normalize_symbol(symbol)

        response = self._api_request('GET', endpoint, params=params)

        if response and isinstance(response, list):
            return [self._parse_order(o) for o in response]

        return []

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions"""
        endpoint = f"/v3/accounts/{self.account_id}/positions"

        params = {}
        if symbol:
            params['symbolName'] = self.normalize_symbol(symbol)

        response = self._api_request('GET', endpoint, params=params)

        if response and isinstance(response, list):
            return [self._parse_position(p) for p in response]

        return []

    def close_position(self, symbol: str, quantity: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """Close a position"""
        # Get current position
        positions = self.get_positions(symbol)

        if not positions:
            return False, "No position found"

        position = positions[0]

        # Create closing order (opposite side)
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        close_qty = quantity if quantity else position.quantity

        close_order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=close_qty,
            comment="Close position"
        )

        success, order_id, error = self.place_order(close_order)
        return success, error

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information"""
        endpoint = f"/v3/symbols/{self.normalize_symbol(symbol)}"

        response = self._api_request('GET', endpoint)

        if response:
            return {
                'symbol': self.denormalize_symbol(response.get('symbolName', symbol)),
                'pip_size': response.get('pipSize', 0.0001),
                'min_volume': response.get('minVolume', 0.01),
                'max_volume': response.get('maxVolume', 100.0),
                'volume_step': response.get('volumeStep', 0.01),
                'digits': response.get('digits', 5),
                'description': response.get('description', ''),
            }

        return None

    def get_available_symbols(self) -> List[str]:
        """Get available trading symbols"""
        endpoint = "/v3/symbols"

        response = self._api_request('GET', endpoint)

        if response and isinstance(response, list):
            return [self.denormalize_symbol(s['symbolName']) for s in response]

        return []

    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType enum to FxPro API format"""
        mapping = {
            OrderType.MARKET: 'MARKET',
            OrderType.LIMIT: 'LIMIT',
            OrderType.STOP: 'STOP',
            OrderType.STOP_LIMIT: 'STOP_LIMIT',
        }
        return mapping.get(order_type, 'MARKET')

    def _parse_order(self, data: Dict) -> Order:
        """Parse order data from API response"""
        return Order(
            symbol=self.denormalize_symbol(data.get('symbolName', '')),
            side=OrderSide.BUY if data.get('tradeSide') == 'BUY' else OrderSide.SELL,
            order_type=OrderType.MARKET,  # Simplified
            quantity=float(data.get('volume', 0)) / 100000,
            price=data.get('limitPrice'),
            stop_price=data.get('stopPrice'),
            take_profit=data.get('takeProfitPrice'),
            stop_loss=data.get('stopLossPrice'),
            order_id=str(data.get('orderId', '')),
            status=self._parse_order_status(data.get('orderStatus', '')),
            filled_quantity=float(data.get('filledVolume', 0)) / 100000,
            comment=data.get('comment'),
        )

    def _parse_position(self, data: Dict) -> Position:
        """Parse position data from API response"""
        volume = float(data.get('volume', 0))
        side = PositionSide.LONG if volume > 0 else PositionSide.SHORT

        return Position(
            symbol=self.denormalize_symbol(data.get('symbolName', '')),
            side=side,
            quantity=abs(volume) / 100000,
            entry_price=float(data.get('entryPrice', 0)),
            current_price=float(data.get('currentPrice', 0)),
            unrealized_pnl=float(data.get('unrealizedPnL', 0)),
            realized_pnl=float(data.get('realizedPnL', 0)),
        )

    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse order status from API"""
        mapping = {
            'ACTIVE': OrderStatus.OPEN,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED,
        }
        return mapping.get(status, OrderStatus.PENDING)


# Convenience function to create authenticated broker instance
def create_fxpro_broker(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    account_id: Optional[str] = None
) -> FxProCTraderBroker:
    """
    Create FxPro cTrader broker instance with credentials from environment or settings.

    Args:
        client_id: OAuth2 client ID (or read from env/settings)
        client_secret: OAuth2 client secret (or read from env/settings)
        account_id: Trading account ID (optional)

    Returns:
        FxProCTraderBroker instance
    """
    import os
    from ..utils.user_settings import get_setting

    client_id = client_id or os.getenv('FXPRO_CLIENT_ID') or get_setting('fxpro_client_id', '')
    client_secret = client_secret or os.getenv('FXPRO_CLIENT_SECRET') or get_setting('fxpro_client_secret', '')
    account_id = account_id or os.getenv('FXPRO_ACCOUNT_ID') or get_setting('fxpro_account_id', '')

    if not client_id or not client_secret:
        logger.warning("FxPro credentials not found - please set FXPRO_CLIENT_ID and FXPRO_CLIENT_SECRET")

    return FxProCTraderBroker(
        client_id=client_id,
        client_secret=client_secret,
        account_id=account_id if account_id else None
    )
