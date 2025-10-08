from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import time, json, os
from pathlib import Path
from loguru import logger
from ..utils.user_settings import get_setting, set_setting, SETTINGS_DIR

ACCOUNTS_FILE = SETTINGS_DIR / "trading_accounts.json"
ORDERS_FILE = SETTINGS_DIR / "trading_orders.json"

def _load_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _save_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

class BrokerServiceBase:
    def place_order(self, order: Dict[str, Any]) -> Tuple[bool, str]:
        raise NotImplementedError
    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError
    def get_open_orders(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

class PaperBroker(BrokerServiceBase):
    """Simulated broker with persistence and multi-account support."""
    def __init__(self):
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._accounts: Dict[str, Dict[str, Any]] = {}
        self._active: Optional[str] = None
        self._load_state()

    def _load_state(self):
        acc = _load_json(ACCOUNTS_FILE) or {}
        ords = _load_json(ORDERS_FILE) or {}
        self._accounts = acc.get("accounts", {})
        self._active = acc.get("active")
        self._orders = {oid: o for oid, o in ords.items()} if isinstance(ords, dict) else {}

        # Ensure at least one account
        if not self._accounts:
            self._accounts = {"default": {"name":"default","baseCurrency":"USD","balance":100000.0,"leverage":30,"tiingo_api_key":get_setting("tiingo_api_key","")}}
            self._active = "default"
            self._persist()

    def _persist(self):
        _save_json(ACCOUNTS_FILE, {"accounts": self._accounts, "active": self._active})
        _save_json(ORDERS_FILE, self._orders)

    def _active_account(self) -> Dict[str, Any]:
        if not self._active or self._active not in self._accounts:
            self._active = list(self._accounts.keys())[0]
        return self._accounts[self._active]

    # Account management
    def create_account(self, name: str, base_currency: str = "USD", balance: float = 100000.0, leverage: int = 30, tiingo_api_key: str = "") -> bool:
        if name in self._accounts:
            return False
        self._accounts[name] = {"name":name,"baseCurrency":base_currency,"balance":float(balance),"leverage":int(leverage),"tiingo_api_key":tiingo_api_key}
        if not self._active: self._active = name
        self._persist(); return True

    def remove_account(self, name: str) -> bool:
        if name in self._accounts:
            if self._active == name:
                self._active = None
            del self._accounts[name]
            self._persist(); return True
        return False

    def set_active(self, name: str) -> bool:
        if name in self._accounts:
            self._active = name; self._persist(); return True
        return False

    # Orders
    def place_order(self, order: Dict[str, Any]) -> Tuple[bool, str]:
        oid = str(int(time.time() * 1000))
        o = dict(order)
        o["id"] = oid
        o["status"] = "OPEN"
        # local time
        o["time"] = o.get("time") or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # placeholder PnL
        o["pnl"] = 0.0
        self._orders[oid] = o
        self._persist()
        logger.info("Paper order placed: {}", o)
        return True, oid

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders and self._orders[order_id].get("status") in ("OPEN","PARTIAL"):
            self._orders[order_id]["status"] = "CANCELLED"
            self._persist()
            return True
        return False

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return [o for o in self._orders.values() if o.get("status") in ("OPEN","PARTIAL")]

class IBroker(BrokerServiceBase):
    """Interactive Brokers stub delegating to PaperBroker for now."""
    def __init__(self):
        self._fallback = PaperBroker()

    def place_order(self, order: Dict[str, Any]) -> Tuple[bool, str]:
        return self._fallback.place_order(order)

    def cancel_order(self, order_id: str) -> bool:
        return self._fallback.cancel_order(order_id)

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return self._fallback.get_open_orders()

class MetaTraderBroker(BrokerServiceBase):
    """MetaTrader (MT4/MT5) stub delegating to PaperBroker for now."""
    def __init__(self):
        self._fallback = PaperBroker()

    def place_order(self, order: Dict[str, Any]) -> Tuple[bool, str]:
        return self._fallback.place_order(order)

    def cancel_order(self, order_id: str) -> bool:
        return self._fallback.cancel_order(order_id)

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return self._fallback.get_open_orders()

class CTraderBroker(BrokerServiceBase):
    """cTrader broker stub delegating to PaperBroker for now."""
    def __init__(self):
        self._fallback = PaperBroker()
        # Read cTrader settings
        self.client_id = get_setting("provider.ctrader.client_id", "")
        self.client_secret = get_setting("provider.ctrader.client_secret", "")
        self.environment = get_setting("provider.ctrader.environment", "demo")
        # TODO: Initialize real cTrader client when credentials are available

    def place_order(self, order: Dict[str, Any]) -> Tuple[bool, str]:
        # TODO: Implement real cTrader order placement
        logger.info(f"CTrader order (using paper fallback): {order}")
        return self._fallback.place_order(order)

    def cancel_order(self, order_id: str) -> bool:
        # TODO: Implement real cTrader order cancellation
        return self._fallback.cancel_order(order_id)

    def get_open_orders(self) -> List[Dict[str, Any]]:
        # TODO: Implement real cTrader open orders retrieval
        return self._fallback.get_open_orders()

def get_broker_service() -> BrokerServiceBase:
    mode = str(get_setting("broker_mode","paper")).lower()
    if mode == "ib":
        return IBroker()
    if mode in ("mt4","mt5","metatrader"):
        return MetaTraderBroker()
    if mode == "ctrader":
        return CTraderBroker()
    return PaperBroker()
