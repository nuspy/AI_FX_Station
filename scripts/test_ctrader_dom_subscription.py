"""
Test cTrader DOM Subscription - Raw Events Monitor

This script subscribes to cTrader DOM (Depth of Market) and prints all raw events received.
Useful for debugging DOM subscription issues and understanding the data structure.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from twisted.internet import reactor

# Configure logger for this test
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG"
)
logger.add(
    "test_dom_subscription.log",
    rotation="10 MB",
    level="DEBUG"
)

try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage, ProtoHeartbeatEvent
    from ctrader_open_api.messages import OpenApiMessages_pb2 as Messages
    from ctrader_open_api.messages import OpenApiCommonMessages_pb2 as CommonMessages
except ImportError as e:
    logger.error(f"ctrader-open-api not installed: {e}")
    logger.error("Install with: pip install ctrader-open-api twisted")
    sys.exit(1)


class DOMTestMonitor:
    """Monitor for testing DOM subscription and displaying raw events."""
    
    def __init__(self, client_id, client_secret, access_token, account_id, environment="demo"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.account_id_raw = account_id  # Can be username like "a.taini" or numeric ID
        self.account_id = None  # Will be set after fetching account list if needed
        self.environment = environment
        
        # Check if account_id is numeric or needs to be fetched
        try:
            self.account_id = int(account_id)
            self.auto_fetch_account = False
            logger.info(f"Using numeric account ID: {self.account_id}")
        except (ValueError, TypeError):
            self.auto_fetch_account = True
            logger.info(f"Account ID '{account_id}' is not numeric - will fetch from cTrader API")
        
        # Connection state
        self.client = None
        self.connected = False
        self.authenticated = False
        self.subscribed_symbols = {}
        
        # Event counters
        self.event_counts = {
            'spot': 0,
            'depth': 0,
            'depth_subscription_response': 0,
            'symbols_list': 0,
            'errors': 0
        }
        
        # Determine host
        self.host = EndPoints.PROTOBUF_DEMO_HOST if environment == "demo" else EndPoints.PROTOBUF_LIVE_HOST
        self.port = EndPoints.PROTOBUF_PORT
        
        logger.info(f"=" * 80)
        logger.info(f"cTrader DOM Subscription Test Monitor")
        logger.info(f"=" * 80)
        logger.info(f"Environment: {environment}")
        logger.info(f"Host: {self.host}:{self.port}")
        logger.info(f"Account ID: {account_id}")
        logger.info(f"=" * 80)
    
    def start(self):
        """Start the test monitor."""
        logger.info("Starting test monitor...")
        
        # Create client
        self.client = Client(self.host, self.port, TcpProtocol)
        self.client.setMessageReceivedCallback(self._on_message)
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        
        # Start service
        self.client.startService()
        
        logger.info("Client service started. Connecting...")
    
    def stop(self):
        """Stop the monitor."""
        logger.info("Stopping monitor...")
        if reactor.running:
            reactor.stop()
    
    def _on_connected(self, client):
        """Called when connected."""
        self.connected = True
        logger.success("✓ Connected to cTrader")
        
        # Authenticate
        reactor.callLater(0, self._authenticate)
    
    def _on_disconnected(self, client, reason):
        """Called when disconnected."""
        self.connected = False
        self.authenticated = False
        logger.warning(f"✗ Disconnected: {reason}")
    
    def _authenticate(self):
        """Authenticate with cTrader."""
        try:
            logger.info("Authenticating application...")
            
            # Application auth
            app_auth_req = Messages.ProtoOAApplicationAuthReq()
            app_auth_req.clientId = self.client_id
            app_auth_req.clientSecret = self.client_secret
            
            self.client.send(app_auth_req)
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
    
    def _fetch_account_list(self):
        """Fetch account list to get numeric account ID."""
        try:
            logger.info("Fetching account list from cTrader API...")
            
            account_list_req = Messages.ProtoOAGetAccountListByAccessTokenReq()
            account_list_req.accessToken = self.access_token
            
            self.client.send(account_list_req)
            
        except Exception as e:
            logger.error(f"Error fetching account list: {e}")
    
    def _authorize_account(self):
        """Authorize trading account."""
        try:
            if self.account_id is None:
                logger.error("Cannot authorize account: account_id is None")
                return
            
            logger.info(f"Authorizing account {self.account_id}...")
            
            account_auth_req = Messages.ProtoOAAccountAuthReq()
            account_auth_req.ctidTraderAccountId = int(self.account_id)
            account_auth_req.accessToken = self.access_token
            
            self.client.send(account_auth_req)
            
        except Exception as e:
            logger.error(f"Account authorization error: {e}")
    
    def _request_symbols(self):
        """Request symbols list."""
        try:
            logger.info("Requesting symbols list...")
            
            symbols_req = Messages.ProtoOASymbolsListReq()
            symbols_req.ctidTraderAccountId = int(self.account_id)
            
            self.client.send(symbols_req)
            
        except Exception as e:
            logger.error(f"Symbols request error: {e}")
    
    def _subscribe_depth_quotes(self, symbol_ids):
        """Subscribe to DOM for symbols."""
        try:
            if not symbol_ids:
                logger.warning("No symbol IDs to subscribe")
                return
            
            logger.info(f"🎯 SUBSCRIBING TO DEPTH QUOTES for symbol IDs: {symbol_ids}")
            logger.info("=" * 80)
            logger.info("WAITING FOR DOM EVENTS...")
            logger.info("If timeout after 10s → Account doesn't support DOM")
            logger.info("=" * 80)
            
            depth_req = Messages.ProtoOASubscribeDepthQuotesReq()
            depth_req.ctidTraderAccountId = int(self.account_id)
            
            for symbol_id in symbol_ids:
                depth_req.symbolId.append(symbol_id)
            
            self.client.send(depth_req)
            
            # Schedule timeout check
            reactor.callLater(10.0, self._check_dom_timeout)
            
        except Exception as e:
            logger.error(f"Depth subscription error: {e}")
    
    def _check_dom_timeout(self):
        """Check if DOM subscription succeeded."""
        if self.event_counts['depth_subscription_response'] == 0:
            logger.error("=" * 80)
            logger.error("❌ DOM SUBSCRIPTION TIMEOUT (10s)")
            logger.error("=" * 80)
            logger.error("Conclusion: This account does NOT support DOM (Depth of Market)")
            logger.error("Reason: ProtoOASubscribeDepthQuotesRes (type 2138) was never received")
            logger.error(f"Note: Received {self.event_counts['spot']} spot events (price updates) but NO DOM response")
            logger.error("=" * 80)
            
            # Show summary and exit
            reactor.callLater(2.0, self._show_summary_and_exit)
    
    def _show_summary_and_exit(self):
        """Show event summary and exit."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("EVENT SUMMARY")
        logger.info("=" * 80)
        for event_type, count in self.event_counts.items():
            logger.info(f"{event_type:30s}: {count}")
        logger.info("=" * 80)
        
        # Show message type distribution
        if hasattr(self, '_message_type_counts'):
            logger.info("")
            logger.info("MESSAGE TYPE DISTRIBUTION:")
            logger.info("=" * 80)
            for msg_type, count in sorted(self._message_type_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"Type {msg_type:6d}: {count:6d} messages")
            logger.info("=" * 80)
        
        self.stop()
    
    def _on_message(self, client, message):
        """Handle incoming message."""
        try:
            payload_type = message.payloadType
            
            # Get message type name from Messages module
            message_type_name = "Unknown"
            
            # Check all message types in the Messages module (OpenApiMessages)
            for attr_name in dir(Messages):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(Messages, attr_name)
                        if isinstance(attr_value, int) and attr_value == payload_type:
                            message_type_name = attr_name
                            break
                    except:
                        pass
            
            # If not found, check CommonMessages module
            if message_type_name == "Unknown":
                for attr_name in dir(CommonMessages):
                    if not attr_name.startswith('_'):
                        try:
                            attr_value = getattr(CommonMessages, attr_name)
                            if isinstance(attr_value, int) and attr_value == payload_type:
                                message_type_name = attr_name
                                break
                        except:
                            pass
            
            logger.debug(f"📨 Received: {message_type_name} (type={payload_type})")
            
            # Count message types
            if not hasattr(self, '_message_type_counts'):
                self._message_type_counts = {}
            self._message_type_counts[payload_type] = self._message_type_counts.get(payload_type, 0) + 1
            
            # Application auth response (type 2101)
            if payload_type == 2101:
                logger.success("=" * 80)
                logger.success("✓ Application authenticated")
                logger.success("=" * 80)
                logger.info("RAW MESSAGE:")
                logger.info(str(message))
                logger.success("=" * 80)
                
                # If we need to fetch account list, do it now; otherwise authorize directly
                if self.auto_fetch_account:
                    reactor.callLater(0, self._fetch_account_list)
                else:
                    reactor.callLater(0, self._authorize_account)
            
            # Account list response (type 2150 - ProtoOAGetAccountListByAccessTokenRes)
            elif payload_type == 2150:
                # Parse the payload - it's a ProtoOAGetAccountListByAccessTokenRes
                account_list_msg = Messages.ProtoOAGetAccountListByAccessTokenRes()
                account_list_msg.ParseFromString(message.payload)
                
                logger.info("=" * 80)
                logger.info(f"✓ Received account list ({len(account_list_msg.ctidTraderAccount)} accounts)")
                logger.info("=" * 80)
                
                # Find matching account
                found = False
                for account in account_list_msg.ctidTraderAccount:
                    logger.info(f"  Account: ID={account.ctidTraderAccountId}, Login={account.traderLogin}, isLive={account.isLive}")
                    
                    # Match by login name or use first account
                    if self.account_id_raw and str(account.traderLogin) == str(self.account_id_raw):
                        self.account_id = account.ctidTraderAccountId
                        found = True
                        logger.success(f"  → Matched account: {account.traderLogin} → ID={self.account_id}")
                    elif not self.account_id_raw and not found:
                        # Use first account if no specific one requested
                        self.account_id = account.ctidTraderAccountId
                        found = True
                        logger.info(f"  → Using first account: ID={self.account_id}")
                
                if not found and self.account_id_raw:
                    # If no match found, use first account
                    if len(account_list_msg.ctidTraderAccount) > 0:
                        self.account_id = account_list_msg.ctidTraderAccount[0].ctidTraderAccountId
                        logger.warning(f"  Account '{self.account_id_raw}' not found, using first account: ID={self.account_id}")
                    else:
                        logger.error("  No accounts found!")
                        self.stop()
                        return
                
                logger.info("")
                logger.info("RAW MESSAGE (Parsed):")
                logger.info(str(account_list_msg))
                logger.info("=" * 80)
                
                # Now authorize the account
                reactor.callLater(0, self._authorize_account)
            
            # Account auth response (type 2103)
            elif payload_type == 2103:
                self.authenticated = True
                logger.success("=" * 80)
                logger.success("✓ Account authorized")
                logger.success("=" * 80)
                logger.info("RAW MESSAGE:")
                logger.info(str(message))
                logger.success("=" * 80)
                reactor.callLater(0, self._request_symbols)
            
            # Symbols list response (type 2115 - actual response type, not 2127)
            elif payload_type == 2115:
                # Parse the payload - it's a ProtoOASymbolsListRes
                symbols_msg = Messages.ProtoOASymbolsListRes()
                symbols_msg.ParseFromString(message.payload)
                
                self.event_counts['symbols_list'] += 1
                logger.info("=" * 80)
                logger.info(f"✓ Received symbols list ({len(symbols_msg.symbol)} symbols)")
                logger.info("=" * 80)
                
                # Find EURUSD only
                symbol_ids = []
                for symbol in symbols_msg.symbol:
                    if symbol.symbolName == "EURUSD":
                        symbol_ids.append(symbol.symbolId)
                        self.subscribed_symbols[symbol.symbolName] = symbol.symbolId
                        logger.info(f"  → {symbol.symbolName}: ID={symbol.symbolId}")
                        break  # Only EURUSD
                
                logger.info("")
                logger.info("RAW MESSAGE (first 1000 chars):")
                msg_str = str(symbols_msg)
                logger.info(msg_str[:1000] + ("..." if len(msg_str) > 1000 else ""))
                logger.info("=" * 80)
                
                # Subscribe to depth quotes
                if symbol_ids:
                    reactor.callLater(0, lambda: self._subscribe_depth_quotes(symbol_ids))
                else:
                    logger.warning("No target symbols found")
            
            # Symbols list response (type 2127)
            elif payload_type == 2127:
                # Parse the payload - it's a ProtoOASymbolsListRes
                symbols_msg = Messages.ProtoOASymbolsListRes()
                symbols_msg.ParseFromString(message.payload)
                
                self.event_counts['symbols_list'] += 1
                logger.info("=" * 80)
                logger.info(f"✓ Received symbols list ({len(symbols_msg.symbol)} symbols)")
                logger.info("=" * 80)
                
                # Find EURUSD only
                symbol_ids = []
                for symbol in symbols_msg.symbol:
                    if symbol.symbolName == "EURUSD":
                        symbol_ids.append(symbol.symbolId)
                        self.subscribed_symbols[symbol.symbolName] = symbol.symbolId
                        logger.info(f"  → {symbol.symbolName}: ID={symbol.symbolId}")
                        break  # Only EURUSD
                
                logger.info("")
                logger.info("RAW MESSAGE (first 1000 chars):")
                msg_str = str(symbols_msg)
                logger.info(msg_str[:1000] + ("..." if len(msg_str) > 1000 else ""))
                logger.info("=" * 80)
                
                # Subscribe to depth quotes
                if symbol_ids:
                    reactor.callLater(0, lambda: self._subscribe_depth_quotes(symbol_ids))
                else:
                    logger.warning("No target symbols found")
            
            # Depth quotes subscription response (type 2138 or 2157)
            elif payload_type == 2138 or payload_type == 2157:
                # Parse the payload - it's a ProtoOASubscribeDepthQuotesRes
                depth_sub_msg = Messages.ProtoOASubscribeDepthQuotesRes()
                depth_sub_msg.ParseFromString(message.payload)
                
                self.event_counts['depth_subscription_response'] += 1
                logger.success("=" * 80)
                logger.success("✓✓✓ DOM SUBSCRIPTION SUCCESSFUL ✓✓✓")
                logger.success("=" * 80)
                logger.info("RAW MESSAGE:")
                logger.info(str(depth_sub_msg))
                logger.success("=" * 80)
                logger.success("This account SUPPORTS DOM (Depth of Market)")
                logger.success("Waiting for ProtoOADepthEvent messages...")
                logger.success("=" * 80)
                
                # Wait for some depth events
                reactor.callLater(30.0, self._show_summary_and_exit)
            
            # Depth event (ORDER BOOK UPDATE) (type 2139 or 2155 - ACTUAL DOM DATA!)
            elif payload_type == 2139 or payload_type == 2155:
                # Parse the payload - it's a ProtoOADepthEvent
                depth_event = Messages.ProtoOADepthEvent()
                depth_event.ParseFromString(message.payload)
                
                self.event_counts['depth'] += 1
                
                # Find symbol name
                symbol_name = "Unknown"
                for sym, sid in self.subscribed_symbols.items():
                    if sid == depth_event.symbolId:
                        symbol_name = sym
                        break
                
                logger.info("=" * 80)
                logger.info(f"📊 ProtoOADepthEvent #{self.event_counts['depth']} for {symbol_name}")
                logger.info("=" * 80)
                
                # Print raw message
                logger.info("RAW MESSAGE:")
                logger.info(str(depth_event))
                logger.info("")
                
                # Parse new quotes
                if hasattr(depth_event, 'newQuotes') and depth_event.newQuotes:
                    logger.info(f"New Quotes: {len(depth_event.newQuotes)}")
                    for i, quote in enumerate(depth_event.newQuotes[:5]):  # Show first 5
                        bid = quote.bid / 100000.0 if hasattr(quote, 'bid') and quote.bid else None
                        ask = quote.ask / 100000.0 if hasattr(quote, 'ask') and quote.ask else None
                        size = quote.size / 100.0 if hasattr(quote, 'size') else 0
                        logger.info(f"  Quote {i+1}: bid={bid}, ask={ask}, size={size}")
                
                # Parse deleted quotes
                if hasattr(depth_event, 'deletedQuotes') and depth_event.deletedQuotes:
                    logger.info(f"Deleted Quotes: {len(depth_event.deletedQuotes)}")
                
                logger.info("=" * 80)
            
            # Spot event (PRICE UPDATE) (type 2124)
            elif payload_type == 2124:
                self.event_counts['spot'] += 1
                
                # Show first 3 events - try ALL message types to find the right one
                if self.event_counts['spot'] <= 3:
                    logger.info("=" * 80)
                    logger.info(f"💹 Message type 2155 #{self.event_counts['spot']} - TRYING ALL DECODERS")
                    logger.info("=" * 80)
                    
                    # Get all ProtoOA message types
                    all_message_types = []
                    for attr_name in dir(Messages):
                        if attr_name.startswith('ProtoOA') and not attr_name.endswith('Req') and not attr_name.endswith('Res'):
                            try:
                                msg_class = getattr(Messages, attr_name)
                                if hasattr(msg_class, 'ParseFromString'):
                                    all_message_types.append((attr_name, msg_class))
                            except:
                                pass
                    
                    logger.info(f"Trying {len(all_message_types)} message types...")
                    logger.info("")
                    
                    # Try each message type
                    for msg_name, msg_class in all_message_types:
                        try:
                            decoded = msg_class()
                            decoded.ParseFromString(message.payload)
                            
                            # Check if decoding produced useful data (more than just IDs)
                            fields = list(decoded.ListFields())
                            if len(fields) > 2:  # More than just account ID and symbol ID
                                logger.success(f"✓✓✓ FOUND MATCH: {msg_name} ✓✓✓")
                                logger.success(f"Fields found: {len(fields)}")
                                logger.success("")
                                for field_desc, value in fields:
                                    if isinstance(value, (list, tuple)) and len(value) > 0:
                                        logger.success(f"  {field_desc.name}: {len(value)} items")
                                        logger.success(f"    First: {value[0]}")
                                    else:
                                        logger.success(f"  {field_desc.name}: {value}")
                                logger.success("=" * 80)
                                break
                        except:
                            pass
                    
                    logger.info("=" * 80)
                
                # Skip normal parsing
                try:
                    spot_event = Messages.ProtoOASpotEvent()
                    spot_event.ParseFromString(message.payload)
                    
                    if False:  # Disabled old code
                        
                        logger.info("=" * 80)
                        logger.info(f"💹 ProtoOASpotEvent #{self.event_counts['spot']}")
                        logger.info("=" * 80)
                        
                        # Show FULL RAW PAYLOAD in HEX
                        logger.info("RAW PAYLOAD (HEX):")
                        hex_str = message.payload.hex()
                        # Split into lines of 64 chars
                        for i in range(0, len(hex_str), 64):
                            logger.info(f"  {hex_str[i:i+64]}")
                        logger.info("")
                        
                        # Show RAW PAYLOAD as ASCII (where printable)
                        logger.info("RAW PAYLOAD (ASCII where printable):")
                        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in message.payload)
                        for i in range(0, len(ascii_str), 64):
                            logger.info(f"  {ascii_str[i:i+64]}")
                        logger.info("")
                        
                        # Try decoding with ListFields() which shows ONLY set fields
                        logger.info("DECODED MESSAGE (ListFields - shows only SET fields):")
                        for field_desc, value in spot_event.ListFields():
                            logger.info(f"  {field_desc.name} = {value}")
                        
                        logger.info("=" * 80)
                    
                    # Log every 10th spot event summary
                    elif self.event_counts['spot'] % 10 == 0:

                except Exception as e:
                    logger.error(f"Failed to parse spot event #{self.event_counts['spot']}: {e}")
                    logger.error(f"Raw payload: {message.payload[:100]}...")
            
            # Error response (type 2142)
            elif payload_type == 2142:
                self.event_counts['errors'] += 1
                error_code = message.errorCode if hasattr(message, 'errorCode') else 'unknown'
                error_desc = message.description if hasattr(message, 'description') else ''
                logger.error("=" * 80)
                logger.error(f"❌ Error: {error_code} - {error_desc}")
                logger.error("=" * 80)
                logger.error("RAW MESSAGE:")
                logger.error(str(message))
                logger.error("=" * 80)
            
            # Heartbeat (type 51 from CommonMessages)
            elif payload_type == 51:
                logger.debug("💓 Heartbeat")
            
            # CATCH-ALL: Try to decode ANY unknown message type
            else:
                logger.warning("=" * 80)
                logger.warning(f"🔍 UNKNOWN MESSAGE TYPE: {payload_type}")
                logger.warning("=" * 80)
                logger.warning(f"Payload length: {len(message.payload)} bytes")
                logger.warning(f"Has clientMsgId: {message.clientMsgId if hasattr(message, 'clientMsgId') else 'N/A'}")
                logger.warning(f"Raw payload (hex, first 100 bytes): {message.payload[:100].hex()}")
                logger.warning("")
                
                # Try to find a matching message type and decode it
                logger.warning("Attempting to decode with common message types...")
                
                # List of common message types to try
                message_types_to_try = [
                    ('ProtoOADepthEvent', Messages.ProtoOADepthEvent),
                    ('ProtoOAQuoteEvent', getattr(Messages, 'ProtoOAQuoteEvent', None)),
                    ('ProtoOATickDataEvent', getattr(Messages, 'ProtoOATickDataEvent', None)),
                    ('ProtoOAMarketDataEvent', getattr(Messages, 'ProtoOAMarketDataEvent', None)),
                ]
                
                for msg_name, msg_class in message_types_to_try:
                    if msg_class is None:
                        continue
                    try:
                        decoded = msg_class()
                        decoded.ParseFromString(message.payload)
                        logger.success(f"✓ Successfully decoded as {msg_name}!")
                        logger.success(str(decoded))
                        break
                    except Exception:
                        pass
                
                logger.warning("=" * 80)
        
        except Exception as e:
            logger.exception(f"Error handling message: {e}")


def main():
    """Main entry point."""
    logger.info("Loading credentials...")
    
    try:
        # Load from .env or user_settings
        from src.forex_diffusion.utils.user_settings import get_setting
        
        client_id = get_setting("provider.ctrader.client_id") or get_setting("ctrader_client_id")
        client_secret = get_setting("provider.ctrader.client_secret") or get_setting("ctrader_client_secret")
        access_token = get_setting("provider.ctrader.access_token") or get_setting("ctrader_access_token")
        account_id = get_setting("provider.ctrader.account_id") or get_setting("ctrader_account_id")
        environment = get_setting("provider.ctrader.environment") or get_setting("ctrader_environment") or "demo"
        
        if not all([client_id, client_secret, access_token, account_id]):
            logger.error("Missing cTrader credentials. Please configure in Settings.")
            return
        
        logger.info(f"Credentials loaded: account={account_id}, env={environment}")
        
        # Create monitor
        monitor = DOMTestMonitor(client_id, client_secret, access_token, account_id, environment)
        monitor.start()
        
        # Run reactor
        logger.info("Starting Twisted reactor...")
        reactor.run()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
