# File: core/execution/order_manager.py

import asyncio
from typing import Optional, Dict, Any
from core.broker.base_broker import BaseBroker
from core.utils.logger import get_logger

logger = get_logger(__name__)


class OrderManager:
    """
    Executes market and pending orders and manages open/pending positions through the broker.
    """

    def __init__(self, broker: BaseBroker) -> None:
        """
        Initialize with a broker object.

        Args:
            broker (BaseBroker): Instance of the connected broker API wrapper.
        """
        self.broker = broker

    def open_market_order(
        self,
        symbol: str,
        direction: str,
        lot: float,
        sl: Optional[float],
        tp: Optional[float],
        comment: str = ""
    ) -> Dict[str, Any]:
        """
        Opens a market order (buy/sell).

        Returns:
            Dictionary with broker response including order ID.
        """
        logger.info(f"Placing market order {direction.upper()} on {symbol} with lot={lot}")
        return self.broker.place_order(
            symbol=symbol,
            action=direction,
            volume=lot,
            sl=sl,
            tp=tp,
            comment=comment
        )

    def open_pending_order(
        self,
        symbol: str,
        order_type: str,
        lot: float,
        price: float,
        sl: Optional[float],
        tp: Optional[float],
        comment: str = "",
        expiration: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Places a pending order (e.g., buy_limit, sell_stop) at specific price.

        Returns:
            Dictionary with broker response.
        """
        logger.info(f"Placing pending {order_type.upper()} order on {symbol} at {price}")
        result = self.broker.place_order(
            symbol=symbol,
            action=order_type,
            volume=lot,
            price=price,
            sl=sl,
            tp=tp,
            comment=comment,
            expiration=expiration
        )
        
        # Fallback to standard format if broker returned unexpected structure
        if isinstance(result, dict) and "order" in result:
            return result

        return {
            "order": None,
            "status": "failed",
            "error": "Unexpected broker return format",
            "raw": result
        }

    async def close_position(self, order_id: int) -> bool:
        """
        Closes an open trade by ticket.

        Returns:
            True if successful, else False.
        """
        logger.info(f"Closing order with ID {order_id}")
        return self.broker.close_order(order_id)

    async def cancel_pending_order(self, order_id: int) -> bool:
        """
        Cancel a specific pending order.

        Returns:
            True if successful, else False.
        """
        logger.info(f"Cancelling pending order ID {order_id}")
        return self.broker.cancel_order(order_id)

    async def cancel_all_pending_orders(self, symbol: Optional[str] = None) -> None:
        """
        Cancel all pending limit/stop orders for a symbol (or all if not specified).
        """
        logger.info(f"Cancelling all pending orders for symbol: {symbol or '[ALL]'}")
        open_orders = self.broker.get_open_orders(symbol=symbol)
        for order in open_orders:
            if order.get("type") in {"buy_limit", "sell_limit", "buy_stop", "sell_stop"}:
                try:
                    self.broker.cancel_order(order["ticket"])
                    logger.info(f"Cancelled order: {order['ticket']}")
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order['ticket']}: {e}")
