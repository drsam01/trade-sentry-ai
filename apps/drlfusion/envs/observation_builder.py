# File: apps/drlfusion/envs/observation_builder.py

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulated_trading_env import SimulatedTradingEnv


class ObservationBuilder:
    """
    Builds and normalizes the observation space.
    """

    def __init__(self, env: "SimulatedTradingEnv"):
        self.env = env

    def get_obs(self) -> np.ndarray:
        idx = min(self.env.current_step, self.env.n_timesteps - 1)
        window = self.env.features[idx - self.env.candle_window:idx]
        flat = window.flatten()

        self.env.normalizer.update(flat)
        norm = self.env.normalizer.normalize(flat)
        return norm.astype(np.float32)
