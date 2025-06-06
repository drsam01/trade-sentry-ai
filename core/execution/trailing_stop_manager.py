# File: core/execution/trailing_stop_manager.py

from typing import Dict, Optional


class TrailingStopManager:
    """
    Handles trailing stop logic per trade.
    """

    @staticmethod
    def get_new_sl(
        trade: Dict,
        current_price: float,
        direction: str
    ) -> Optional[float]:
        """
        Calculate a new stop-loss based on trailing distance if conditions allow.

        Args:
            trade (Dict): Trade dictionary with at least 'sl' and 'trailing_stop'
            current_price (float): Latest market price
            direction (str): "buy" or "sell"

        Returns:
            Optional[float]: Updated SL if adjustment is needed, else None
        """
        sl = trade.get("sl")
        trailing_stop = trade.get("trailing_stop")

        if trailing_stop is None:
            return None

        if direction == "buy":
            new_sl = current_price - trailing_stop
            if sl is None or new_sl > sl:
                return new_sl

        elif direction == "sell":
            new_sl = current_price + trailing_stop
            if sl is None or new_sl < sl:
                return new_sl

        return None
