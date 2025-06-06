# File: core/execution/risk_manager.py

from typing import Dict, Tuple, Optional
from core.utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Manages risk: lot size, SL/TP, break-even logic, and trailing stops.
    """

    def __init__(self, balance: float, config: Dict[str, float]):
        """
        Args:
            balance: Current account balance.
            config: Dictionary with keys:
                - risk_percent: % of account balance to risk per trade
                - sl_buffer_pips: Extra pips beyond ATR or structure
                - tp_ratio: Target reward-to-risk ratio
                - pip_value: Pip size per symbol
                - trailing_stop_pips: Distance in pips for trailing stop
        """
        self.balance = balance
        self.risk_percent = config.get("risk_percent", 1.0)
        self.sl_buffer_pips = config.get("sl_buffer_pips", 10)
        self.tp_ratio = config.get("tp_ratio", 2.0)
        self.pip_value = config.get("pip_value", 0.0001)
        self.trailing_stop_pips = config.get("trailing_stop_pips", 15)

        '''    
        def compute_lot_size(self, sl_pips: float, pip_value_per_lot: float) -> float:
        """
        Calculate the trade lot size to keep risk within bounds.
        """
        risk_amount = self.balance * (self.risk_percent / 100)
        sl_value = sl_pips * pip_value_per_lot
        if sl_value == 0:
            logger.warning("SL value is zero — defaulting to minimum lot size.")
            return 0.01
        lot_size = risk_amount / sl_value
        logger.info(f"Computed lot size: {lot_size:.2f}")
        return round(max(0.01, lot_size), 2)'''
        
    def compute_lot_size(self, stop_loss_pips: float, pip_value: float, max_lot: float = 1.0) -> float:
        if pip_value * stop_loss_pips == 0:
            return 0.01
        risk_amount = self.balance * (self.risk_percent / 100)
        return round(min(risk_amount / (pip_value * stop_loss_pips), max_lot), 2)

    def calculate_sl_tp(self, entry_price: float, direction: str, atr: float) -> Tuple[float, float]:
        """
        Determine stop loss and take profit based on ATR and direction.
        """
        sl_distance = atr + self.sl_buffer_pips * self.pip_value
        tp_distance = sl_distance * self.tp_ratio

        if direction == "buy":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        logger.info(f"Calculated SL: {sl}, TP: {tp}")
        return sl, tp

    def should_move_to_breakeven(self, current_price: float, entry_price: float, direction: str, breakeven_buffer: float = 0.0) -> bool:
        """
        Returns True if price has moved enough to move SL to entry (break-even).
        """
        if direction == "buy":
            return current_price >= entry_price + breakeven_buffer
        else:
            return current_price <= entry_price - breakeven_buffer

    def apply_trailing_stop(
        self,
        current_price: float,
        direction: str
    ) -> Optional[float]:
        """
        Calculates updated SL price using trailing stop logic.

        Args:
            current_price: Latest market price
            direction: "buy" or "sell"

        Returns:
            New SL price or None if trailing stop distance is zero.
        """
        if self.trailing_stop_pips <= 0:
            logger.debug("Trailing stop disabled.")
            return None

        trailing_distance = self.trailing_stop_pips * self.pip_value

        if direction == "buy":
            new_sl = current_price - trailing_distance
        else:
            new_sl = current_price + trailing_distance

        logger.info(f"Updated trailing SL: {new_sl}")
        return new_sl
