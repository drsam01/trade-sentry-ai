# File: apps/drlfusion/envs/env_metrics.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulated_trading_env import SimulatedTradingEnv


class EnvironmentMetrics:
    """
    Tracks performance metrics like drawdown, return buffer.
    """

    def __init__(self, env: "SimulatedTradingEnv"):
        self.env = env

    def update(self, pnl: float) -> None:
        """
        Updates balance-related metrics.
        """
        self.env.peak_balance = max(self.env.peak_balance, self.env.balance)
        drawdown = (self.env.balance - self.env.peak_balance) / self.env.peak_balance
        self.env.max_drawdown = min(self.env.max_drawdown, drawdown)

        self.env.return_buffer.append(pnl)
        if len(self.env.return_buffer) > 10:
            self.env.return_buffer.pop(0)
