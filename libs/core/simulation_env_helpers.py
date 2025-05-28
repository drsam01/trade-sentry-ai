# File: drlfusion/modelling/envs/simulation_env_helpers.py

"""
Simulation environment helpers for DRLFusion.
Encapsulates reusable logic for trading step execution, reward calculation, and observations.
"""

import numpy as np


class SimEnvLogic:
    """
    Encapsulates trade logic, Zhang reward calculation, and normalized observation construction.
    Used by SimulatedTradingEnv.
    """

    def __init__(self, env):
        """
        Bind this logic handler to the environment.

        Args:
            env: Instance of SimulatedTradingEnv.
        """
        self.env = env

    def enter_trade(self, action: int, price: float) -> None:
        """
        Handle trade entry logic: position size, SL/TP, trailing stop.

        Args:
            action (int): Direction to trade (-1 = short, 1 = long).
            price (float): Entry price.
        """
        self.env.position = action
        self.env.entry_price = price

        # Calculate position size
        self.env.position_size = (
            self.env.risk_mgr.risk_percentage * self.env.balance /
            max(self.env.atr[self.env.current_step], 1e-6)
        )

        # Define SL/TP
        self.env.stop_loss, self.env.take_profit = self.env.risk_mgr._compute_levels(
            position=action,
            entry_price=price,
            current_atr=self.env.atr[self.env.current_step],
            risk_reward_ratio=self.env.risk_reward
        )

        # Initialize trailing logic
        self.env.trailing_stop = self.env.stop_loss
        self.env.peak_price = price
        self.env.hold_duration = 0

        self.env.logger.info(f"Enter {'LONG' if action == 1 else 'SHORT'} @ {price:.5f}")

    def exit_trade(self, price: float, tag: str) -> float:
        """
        Close position and update balance.

        Args:
            price (float): Exit price.
            tag (str): Reason for exit (e.g., Exit, ForcedExit).

        Returns:
            float: Net PnL from the trade.
        """
        pnl = self.env.trade_checker.exit_pnl(
            position=self.env.position,
            price=price,
            entry_price=self.env.entry_price,
            position_size=self.env.position_size,
            transaction_cost=self.env.transaction_cost
        )

        self.env.balance += pnl
        self.env.logger.info(f"{tag}: {'LONG' if self.env.position == 1 else 'SHORT'} exit @ {price:.5f}, pnl={pnl:.2f}")

        self.env.position = 0
        self.env.hold_duration = 0

        return pnl

    def zhang_reward(self, prev_price: float, last_price: float) -> float:
        """
        Zhang-style reward with volatility scaling and regime penalties.

        Args:
            prev_price (float): Price at t-1.
            last_price (float): Price at t.

        Returns:
            float: Computed reward.
        """
        r_t = last_price - prev_price

        start = max(0, self.env.current_step - self.env.vol_window)
        sigma_tm1 = max(np.std(np.diff(self.env.prices[start:self.env.current_step + 1])), 1e-8)
        sigma_tm2 = max(np.std(np.diff(self.env.prices[max(0, start - 1):self.env.current_step])), 1e-8)

        pos_prev = self.env.prev_position
        pos_prev2 = self.env.prev_prev_position

        penalty = self.env.transaction_cost * (sigma_tm1 / self.env.sigma_target)

        reward = (
            self.env.mu * (self.env.sigma_target / sigma_tm1) * pos_prev * r_t
            - penalty * prev_price * self.env.sigma_target
              * abs((pos_prev / sigma_tm1) - (pos_prev2 / sigma_tm2))
        )

        self.env.prev_prev_position = self.env.prev_position
        self.env.prev_position = self.env.position

        return reward

    def update_metrics(self, pnl: float) -> None:
        """
        Track drawdown and recent returns.

        Args:
            pnl (float): Realized profit/loss.
        """
        self.env.peak_balance = max(self.env.peak_balance, self.env.balance)
        drawdown = (self.env.balance - self.env.peak_balance) / self.env.peak_balance
        self.env.max_drawdown = min(self.env.max_drawdown, drawdown)

        self.env.return_buffer.append(pnl)
        if len(self.env.return_buffer) > 10:
            self.env.return_buffer.pop(0)

    def get_obs(self) -> np.ndarray:
        """
        Build normalized, flattened state window.

        Returns:
            np.ndarray: Normalized feature vector of shape (window_size * feature_dim,).
        """
        idx = min(self.env.current_step, self.env.n_timesteps - 1)
        window = self.env.features[idx - self.env.candle_window:idx]
        flat = window.flatten()

        self.env.normalizer.update(flat)
        norm = self.env.normalizer.normalize(flat)

        return norm.astype(np.float32)
