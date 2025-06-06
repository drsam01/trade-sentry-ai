# File: core/monitoring/trade_watcher.py

from typing import List, Dict, Optional, Callable
from core.execution.trailing_stop_manager import TrailingStopManager
from core.utils.logger import get_logger

logger = get_logger(__name__)

class TradeWatcher:
    """
    Monitors open trades and checks if SL/TP or trailing SL should be triggered.
    """

    def __init__(self, trailing_stop_func: Optional[Callable] = None, broker=None): # type: ignore
        """
        Args:
            trailing_stop_func: Function that calculates new SL if trailing conditions met.
        """
        self.trailing_stop_func = trailing_stop_func # type: ignore
        self.broker = broker

    def evaluate_trades( # type: ignore
        self,
        trades: List[Dict], # type: ignore
        current_price: float,
        direction: str
    ) -> List[Dict]:    # type: ignore
        """
        Evaluate each trade for stop loss, take profit, or trailing stop.

        Args:
            trades: List of open trades.
            current_price: Latest market price.
            direction: "buy" or "sell".

        Returns:
            List of trades to be closed.
        """
        to_close = []

        for trade in trades: # type: ignore
            sl = trade.get("sl") # type: ignore
            tp = trade.get("tp") # type: ignore

            # --- Check for exit due to SL or TP ---
            if (
                direction == "buy"
                and (sl is not None and current_price <= sl or tp is not None and current_price >= tp)
            ) or (
                direction == "sell"
                and (sl is not None and current_price >= sl or tp is not None and current_price <= tp)
            ):
                to_close.append(trade)
                continue  # no trailing stop if we're closing
            
            # Optional: apply trailing stop
            # --- Apply centralized trailing stop logic ---
            new_sl = TrailingStopManager.get_new_sl(trade, current_price, direction)
            if new_sl is not None:
                trade["sl"] = new_sl
                trade_id = trade.get("ticket")  # or trade["order_id"]
                if trade_id and hasattr(self, "broker"):
                    if self.broker.modify_sl(trade_id, new_sl): # type: ignore
                        logger.info(f"[TradeWatcher] Updated SL for trade {trade_id} to {new_sl}")
                    else:
                        logger.info(f"[TradeWatcher] Failed to update SL for trade {trade_id}")
            
            # --- Delegate to external trailing_stop_func if available ---
            elif self.trailing_stop_func:
                new_sl = self.trailing_stop_func(current_price, direction)
                if new_sl and (
                    (direction == "buy" and (sl is None or new_sl > sl)) or
                    (direction == "sell" and (sl is None or new_sl < sl))
                ):
                    trade["sl"] = new_sl  # Update SL inline
                    trade_id = trade.get("ticket")  # or trade["order_id"]
                    if trade_id and hasattr(self, "broker"):
                        if self.broker.modify_sl(trade_id, new_sl): # type: ignore
                            logger.info(f"[TradeWatcher] Updated SL for trade {trade_id} to {new_sl}")
                        else:
                            logger.info(f"[TradeWatcher] Failed to update SL for trade {trade_id}")

        return to_close # type: ignore
