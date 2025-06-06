# File: core/broker/base_broker.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseBroker(ABC):
    """
    Abstract base class defining the broker interface for all implementations (e.g., MT5, OANDA).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker with credentials and other parameters.
        """
        self.config = config

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker API/platform."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection to broker is active."""
        pass

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Fetch account details like balance, equity, margin, etc."""
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Return contract specifications for a given symbol."""
        pass

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, bars: int) -> List[Dict[str, Any]]:
        """Return OHLCV data in dictionary format."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        action: str,
        volume: float,
        sl: Optional[float],
        tp: Optional[float],
        price: Optional[float] = None,
        type: Optional[str] = None,
        stoplimit_price: Optional[float] = None,
        comment: str = "",
        expiration: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place a market order.

        Args:
            symbol: Trading symbol.
            action: "buy" or "sell".
            volume: Lot size.
            sl: Stop loss in price.
            tp: Take profit in price.
            comment: Trade annotation.
        """
        pass

    @abstractmethod
    def close_order(self, order_id: int) -> bool:
        """Close an open position."""
        pass

    @abstractmethod
    def modify_order(self, order_id: int, sl: float, tp: float) -> bool:
        """Modify SL/TP of an existing order."""
        pass

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all open positions (optionally filtered by symbol)."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: int, comment: Optional[str] = None) -> bool:
        """Cancel all order requests."""
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Fetch the latest mid-price (bid+ask)/2 for the given symbol."""
        pass

    @abstractmethod
    def modify_sl(self, order_id: str, new_sl: float) -> bool:
        """
        Modify stop loss for an open trade.

        Args:
            order_id (str): Broker-assigned trade/ticket ID.
            new_sl (float): New stop loss price.

        Returns:
            bool: True if successful, else False.
        """
        pass
