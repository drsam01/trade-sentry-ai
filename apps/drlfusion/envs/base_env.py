# File: apps/drlfusion/envs/base_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

class TradingEnvBase(gym.Env, ABC):
    """
    Abstract base class for RL trading environments.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, obs_shape: Tuple[int], action_space: int = 3):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(action_space)

    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]: # type: ignore
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        pass

    def render(self, mode: str = "human") -> None:
        pass
