# File: core/monitoring/trade_watcher.py

from typing import List, Dict, Optional, Callable


class TradeWatcher:
    """
    Monitors open trades and checks if SL/TP or trailing SL should be triggered.
    """

    def __init__(self, trailing_stop_func: Optional[Callable] = None): # type: ignore
        """
        Args:
            trailing_stop_func: Function that calculates new SL if trailing conditions met.
        """
        self.trailing_stop_func = trailing_stop_func # type: ignore

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

            if (
                direction == "buy"
                and (current_price <= sl or current_price >= tp) # type: ignore
                or direction != "buy"
                and (current_price >= sl or current_price <= tp) # type: ignore
            ):
                to_close.append(trade) # type: ignore
            # Optional: apply trailing stop
            if self.trailing_stop_func: # type: ignore
                new_sl = self.trailing_stop_func(current_price, direction) # type: ignore
                if new_sl and (
                    (direction == "buy" and new_sl > sl) or
                    (direction == "sell" and new_sl < sl)
                ):
                    trade["sl"] = new_sl  # Update SL inline

        return to_close # type: ignore
