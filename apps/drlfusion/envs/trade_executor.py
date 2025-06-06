# File: apps/drlfusion/envs/trade_executor.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulated_trading_env import SimulatedTradingEnv


class TradeExecutor:
    """
    Manages trade execution logic — entry, exit, SL/TP, trailing stop.
    """

    def __init__(self, env: "SimulatedTradingEnv"):
        self.env = env

    def enter_trade(self, action: int, price: float) -> None:
        """
        Opens a new trade in the given direction.
        """
        self.env.position = action
        self.env.entry_price = price

        # Risk-managed position sizing
        self.env.position_size = int(round(
            self.env.risk_mgr.risk_percentage * self.env.balance /
            max(self.env.atr[self.env.current_step], 1e-6) # type: ignore
        ))

        # Compute SL and TP levels
        self.env.stop_loss, self.env.take_profit = self.env.risk_mgr.compute_levels( # type: ignore
            position=action,
            entry_price=price,
            current_atr=self.env.atr[self.env.current_step], # type: ignore
            risk_reward_ratio=self.env.risk_reward
        )

        self.env.trailing_stop = self.env.stop_loss
        self.env.peak_price = price
        self.env.hold_duration = 0

    def exit_trade(self, price: float, tag: str) -> float:
        """
        Exits the current position and logs result.
        """
        pnl = self.env.trade_checker.exit_pnl(
            position=self.env.position,
            price=price,
            entry_price=self.env.entry_price,
            position_size=self.env.position_size,
            transaction_cost=self.env.transaction_cost
        )
        self.env.balance += pnl

        self.env.position = 0
        self.hold_duration = 0
        return pnl
