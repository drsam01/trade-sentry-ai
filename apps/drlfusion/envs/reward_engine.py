# File: apps/drlfusion/envs/reward_engine.py

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulated_trading_env import SimulatedTradingEnv


class ZhangRewardEngine:
    """
    Computes Zhang-style volatility-scaled reward with regime penalties.
    Optionally supports log(balance / initial_balance) based reward.
    """

    def __init__(self, env: "SimulatedTradingEnv", use_log_balance_reward: bool = True):
        self.env = env
        self.use_log_balance_reward = use_log_balance_reward
        self.initial_balance = env.config.get("trading_params", {}).get("initial_account_balance", 10000)

    def compute(self, prev_price: float, last_price: float) -> float:
        r_t = last_price - prev_price
        step = self.env.current_step
        start = max(0, step - self.env.vol_window) # type: ignore

        sigma_tm1 = max(np.std(np.diff(self.env.prices[start:step + 1])), 1e-8) # type: ignore
        sigma_tm2 = max(np.std(np.diff(self.env.prices[max(0, start - 1):step])), 1e-8) # type: ignore

        penalty = self.env.transaction_cost * (sigma_tm1 / self.env.sigma_target) # type: ignore
        pos_prev = self.env.prev_position
        pos_prev2 = self.env.prev_prev_position

        reward = (
            self.env.mu * (self.env.sigma_target / sigma_tm1) * pos_prev * r_t # type: ignore
            - penalty * prev_price * self.env.sigma_target # type: ignore
              * abs((pos_prev / sigma_tm1) - (pos_prev2 / sigma_tm2))
        )
        
        # Optional log-equity scaling
        if self.use_log_balance_reward:
            balance = max(self.env.balance, 1e-6)
            reward += np.log(balance / self.initial_balance + 1e-6)

        # Shift position memory
        self.env.prev_prev_position = pos_prev
        self.env.prev_position = self.env.position
        return reward
