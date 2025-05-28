# File: drlfusion/modelling/agents/agent_architectures/base_agent.py

"""
Abstract base class for DRLFusion trading agents.
Defines the required interface for all agent types:
- choose_action()
- update()
- save()
- load()
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class BaseAgent(ABC):
    """
    Abstract base class for all DRLFusion Agents (PPO, DQN, etc).
    """

    def __init__(self, observation_space, action_space) -> None:
        """
        Initialize the agent interface with observation and action space info.

        Args:
            observation_space: The environment's observation space object.
            action_space: The environment's action space object.
        """
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def choose_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float, np.ndarray]:
        """
        Choose an action given a state.

        Args:
            state (np.ndarray): The current environment observation.
            deterministic (bool): Whether to act greedily or stochastically.

        Returns:
            Tuple containing action index, log-probability, value estimate, and action probabilities.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update the agent using collected experiences.
        Typically overridden by external trainer/managers.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model parameters to file.

        Args:
            path (str): Destination file path for saving model weights.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model parameters from file.

        Args:
            path (str): Path to load model from.
        """
        pass
