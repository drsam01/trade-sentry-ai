# File: core/broker/mt5.py

import MetaTrader5 as mt5
from typing import Dict, Any, List, Optional
from core.broker.base_broker import BaseBroker
import pandas as pd


class MT5Broker(BaseBroker):
    """
    MetaTrader 5 broker implementation using MetaTrader5 module.
    """

    def connect(self) -> bool:
        return mt5.initialize(  # type: ignore
            login=self.config["login"],
            server=self.config["server"],
            password=self.config["password"],
            path=self.config.get("path")
        )

    def is_connected(self) -> bool:
        return mt5.terminal_info() is not None  # type: ignore

    def get_account_info(self) -> Dict[str, Any]:
        info = mt5.account_info()._asdict()  # type: ignore
        return dict(info)

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        return mt5.symbol_info(symbol)._asdict()  # type: ignore

    def fetch_ohlcv(self, symbol: str, timeframe: str, bars: int) -> List[Dict[str, Any]]:
        tf = getattr(mt5, f"TIMEFRAME_{timeframe.upper()}")
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)  # type: ignore
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df.to_dict("records") # type: ignore

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
        Place a market or pending order.

        Args:
            symbol (str): Trading symbol.
            action (str): One of 'buy', 'sell', 'buy_limit', 'sell_limit', 'buy_stop', 'sell_stop', 'buy_stop_limit', 'sell_stop_limit'.
            volume (float): Order volume.
            sl (Optional[float]): Stop loss price.
            tp (Optional[float]): Take profit price.
            price (Optional[float]): Entry price for pending orders.
            stoplimit_price (Optional[float]): Limit price for stop-limit orders.
            comment (str): Optional order comment.
            expiration (Optional[str]): Optional expiration datetime string (e.g., "2025-06-01 12:00:00").

        Returns:
            Dict[str, Any]: Result of order_send.
        """
        type_map = {
            "buy": mt5.ORDER_TYPE_BUY,
            "sell": mt5.ORDER_TYPE_SELL,
            "buy_limit": mt5.ORDER_TYPE_BUY_LIMIT,
            "sell_limit": mt5.ORDER_TYPE_SELL_LIMIT,
            "buy_stop": mt5.ORDER_TYPE_BUY_STOP,
            "sell_stop": mt5.ORDER_TYPE_SELL_STOP,
            "buy_stop_limit": mt5.ORDER_TYPE_BUY_STOP_LIMIT,
            "sell_stop_limit": mt5.ORDER_TYPE_SELL_STOP_LIMIT,
        }

        if action not in type_map:
            raise ValueError(f"Invalid order type: {action}")

        order_type = type_map[action]
        is_market = action in {"buy", "sell"}

        if is_market:
            price = mt5.symbol_info_tick(symbol).ask if action == "buy" else mt5.symbol_info_tick(symbol).bid # type: ignore
        elif price is None:
            raise ValueError("Pending orders require a price.")

        request = {
            "action": mt5.TRADE_ACTION_DEAL if is_market else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": type or order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.config.get("deviation", 10),
            "magic": self.config.get("magic_number", 123456),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC if is_market else mt5.ORDER_FILLING_RETURN,
        }

        if expiration:
            from datetime import datetime
            request["expiration"] = int(datetime.strptime(expiration, "%Y-%m-%d %H:%M:%S").timestamp())

        if order_type in [mt5.ORDER_TYPE_BUY_STOP_LIMIT, mt5.ORDER_TYPE_SELL_STOP_LIMIT]:
            if stoplimit_price is None:
                raise ValueError("Stop-limit orders require stoplimit_price.")
            request["price"] = price  # stop trigger
            request["price_limit"] = stoplimit_price  # limit execution price

        result = mt5.order_send(request)  # type: ignore
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"order": result.order, "status": "success"}
        else:
            return {
                "order": None,
                "status": "failed",
                "error": result.comment,
                "retcode": result.retcode
            }

    def close_order(self, order_id: int) -> bool:
        """
        Closes an open market order (position).
        """
        positions = mt5.positions_get(ticket=order_id)  # type: ignore
        if not positions:
            print(f"[ERROR] Position with ticket {order_id} not found.")
            return False

        position = positions[0]
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(position.symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask  # type: ignore

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": self.config.get("deviation", 10),
            "magic": self.config.get("magic_number", 123456),
            "comment": "Close order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)  # type: ignore
        return result.retcode == mt5.TRADE_RETCODE_DONE  # type: ignore

    def cancel_order(self, order_id: int, comment: Optional[str] = "Cancelled by system") -> bool:
        """
        Cancel a specific pending order by ticket ID.
        
        Args:
            order_id (int): The ticket number of the order to cancel.
            comment (str, optional): Optional reason for cancellation.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        order = mt5.orders_get(ticket=order_id) # type: ignore
        if not order:
            print(f"[ERROR] No pending order found with ticket: {order_id}")
            return False

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": order_id,
            "comment": comment
        }

        result = mt5.order_send(request)  # type: ignore
        if result.retcode == mt5.TRADE_RETCODE_DONE:  # type: ignore
            print(f"[SUCCESS] Order {order_id} cancelled.")
            return True
        else:
            print(f"[FAILURE] Could not cancel order {order_id}: retcode {result.retcode}")
            return False

    def close_all_pending_orders(self, symbol: Optional[str] = None) -> None:
        """
        Cancels all pending (unfilled) orders, optionally filtered by symbol.
        """
        orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()  # type: ignore
        if orders is None:
            print("[INFO] No pending orders found.")
            return

        pending_types = {
            mt5.ORDER_TYPE_BUY_LIMIT,
            mt5.ORDER_TYPE_SELL_LIMIT,
            mt5.ORDER_TYPE_BUY_STOP,
            mt5.ORDER_TYPE_SELL_STOP,
            mt5.ORDER_TYPE_BUY_STOP_LIMIT,
            mt5.ORDER_TYPE_SELL_STOP_LIMIT,
        }

        for order in orders:
            if order.type in pending_types:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "comment": "Cancelled by system"
                }
                result = mt5.order_send(request)  # type: ignore
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"[FAIL] Could not cancel order {order.ticket}: {result.retcode}")
                else:
                    print(f"[SUCCESS] Cancelled order {order.ticket}")

    def modify_order(self, order_id: int, sl: float, tp: float) -> bool:
        return mt5.order_modify(  # type: ignore
            ticket=order_id,
            sl=sl,
            tp=tp
        )

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        orders = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()  # type: ignore
        return [order._asdict() for order in orders]

    def get_pip_value(self, symbol: str) -> float:
        info = mt5.symbol_info(symbol) # type: ignore
        return info.point if info else 0.0001
    
    def get_latest_price(self, symbol: str) -> float:
        tick = mt5.symbol_info_tick(symbol) # type: ignore
        if tick is None:
            raise ValueError(f"Unable to fetch tick data for symbol: {symbol}")
        return (tick.bid + tick.ask) / 2

    def modify_sl(self, order_id: str, new_sl: float) -> bool:
        """
        Modify the SL of an open position in MetaTrader 5.
        """
        position = mt5.positions_get(ticket=int(order_id)) # type: ignore
        if not position or len(position) == 0:
            print(f"[MT5] No position found for ticket {order_id}")
            return False

        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "sl": round(new_sl, 5),
            "tp": pos.tp,
            "symbol": pos.symbol,
            "magic": pos.magic,
            "comment": "Trailing SL Update"
        }

        result = mt5.order_send(request) # type: ignore
        return result.retcode == mt5.TRADE_RETCODE_DONE


