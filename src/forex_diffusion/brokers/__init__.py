"""
Broker integrations package.

Provides unified interface for multiple broker platforms:
- FxPro cTrader (OAuth2 + REST API)
- Interactive Brokers (TWS API)
- MetaTrader 4/5 (MQL/Python bridge)
- Paper trading (simulation)
"""
from __future__ import annotations

from .base import BrokerBase, OrderType, OrderSide, OrderStatus
from .fxpro_ctrader import FxProCTraderBroker
from .paper_broker import PaperBroker

__all__ = [
    'BrokerBase',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'FxProCTraderBroker',
    'PaperBroker',
]
