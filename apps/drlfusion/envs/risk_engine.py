# File: apps/drlfusion/envs/risk_engine.py

import numpy as np
from typing import Optional, Tuple, Dict, Any
from core.utils.logger import get_logger

logger = get_logger(__name__)


class TrailingStopEngine:
    """
    Manages ATR-based trailing stop loss with optional break-even logic.
    """

    def __init__(self, config: Dict[str, Any]):
        params = config.get("trading_params", {})
        self.activation_atr = params.get("tsl_activation_atr", 1.5)
        self.trailing_step = params.get("tsl_step_atr", 0.5)
        self.enable_break_even = params.get("tsl_break_even", True)

    def update(
        self,
        position: int,
        entry_price: float,
        current_price: float,
        previous_trailing: Optional[float],
        previous_peak: float,
        current_atr: float
    ) -> Tuple[float, float]:
        if current_atr <= 0 or position == 0 or previous_trailing is None:
            return previous_trailing, previous_peak # type: ignore

        peak = previous_peak
        new_trailing = previous_trailing
        moved = False

        if position == 1:
            peak = max(previous_peak, current_price)
            move = peak - entry_price
            if move >= self.activation_atr * current_atr:
                raw = peak - current_atr
                stepped = entry_price + (
                    (int((move - self.activation_atr * current_atr) / (self.trailing_step * current_atr)) + 1)
                    * self.trailing_step * current_atr
                )
                updated = min(raw, stepped)
                if updated > previous_trailing:
                    new_trailing = updated
                    moved = True
            elif self.enable_break_even and move > 0 and previous_trailing < entry_price:
                new_trailing = entry_price
                moved = True

        elif position == -1:
            peak = min(previous_peak, current_price)
            move = entry_price - peak
            if move >= self.activation_atr * current_atr:
                raw = peak + current_atr
                stepped = entry_price - (
                    (int((move - self.activation_atr * current_atr) / (self.trailing_step * current_atr)) + 1)
                    * self.trailing_step * current_atr
                )
                updated = max(raw, stepped)
                if updated < previous_trailing:
                    new_trailing = updated
                    moved = True
            elif self.enable_break_even and move > 0 and previous_trailing > entry_price:
                new_trailing = entry_price
                moved = True

        if moved:
            logger.info(f"[TSL] {'LONG' if position == 1 else 'SHORT'}: peak={peak:.5f}, tsl={new_trailing:.5f}")

        return new_trailing, peak


class TradeLogicEvaluator:
    """
    Encapsulates trade signal logic: entries, exits, and PnL.
    """

    def should_enter(self, action: int, current_position: int) -> bool:
        return action in {-1, 1} and current_position == 0

    def should_force_exit(self, position: int, hold_duration: int, max_hold: int) -> bool:
        return position != 0 and hold_duration >= max_hold

    def evaluate_exit(
        self,
        position: int,
        price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        trailing_stop: Optional[float]
    ) -> bool:
        if position == 1:
            return (stop_loss is not None and price <= stop_loss) or \
                    (take_profit is not None and price >= take_profit) or \
                    (trailing_stop is not None and price <= trailing_stop)
        elif position == -1:
            return (stop_loss is not None and price >= stop_loss) or \
                    (take_profit is not None and price <= take_profit) or \
                    (trailing_stop is not None and price >= trailing_stop)
        return False

    def exit_pnl(
        self,
        position: int,
        price: float,
        entry_price: float,
        position_size: float,
        transaction_cost: float
    ) -> float:
        profit = (price - entry_price) * position * position_size
        cost = transaction_cost * position_size
        return profit - cost


class EnvRiskManager:
    """
    Computes SL/TP levels and manages trailing stops for environment trades.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_percentage = config.get("trading_params", {}).get("risk_percentage", 0.01)
        self.max_leverage = config.get("risk_limits", {}).get("max_leverage", 5)
        self.min_equity = config.get("risk_limits", {}).get("min_equity", 100)
        self.tsl_engine = TrailingStopEngine(config)

    def compute_levels(
        self,
        position: int,
        entry_price: float,
        current_atr: float,
        risk_reward_ratio: float
    ) -> Tuple[Optional[float], Optional[float]]:
        if current_atr <= 0 or position == 0:
            return None, None
        if position == 1:
            return entry_price - current_atr, entry_price + risk_reward_ratio * current_atr
        else:
            return entry_price + current_atr, entry_price - risk_reward_ratio * current_atr

    def update_trailing_stop(
        self,
        position: int,
        entry_price: float,
        current_price: float,
        previous_trailing: Optional[float],
        previous_peak: float,
        current_atr: float
    ) -> Tuple[float, float]:
        return self.tsl_engine.update(position, entry_price, current_price, previous_trailing, previous_peak, current_atr)

    def can_enter_trade(self, price: float, balance: float, position_size: float) -> bool:
        if balance < self.min_equity:
            return False

        notional_value = price * abs(position_size)
        leverage_used = notional_value / max(balance, 1e-6)  # Avoid div-by-zero
        return leverage_used <= self.max_leverage
    
    def should_exit_trade(
        self,
        position: int,
        price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        trailing_stop: Optional[float]
    ) -> bool:
        evaluator = TradeLogicEvaluator()
        return evaluator.evaluate_exit(position, price, stop_loss, take_profit, trailing_stop)
