# File: drlfusion/modelling/agents/agent_managers/base_manager.py

"""
Abstract base class for all RL agent managers (e.g., PPOManager, DQNManager).
Handles the interface between the agent, training logic, and environment interaction.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Tuple, Dict


class BaseManager(ABC):
    """
    Abstract manager that wraps an agent and exposes consistent training and acting APIs.
    """

    def __init__(self) -> None:
        """
        Initialize base manager.
        """
        super().__init__()

    @abstractmethod
    def choose_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float, np.ndarray]:
        """
        Choose action from the wrapped agent.

        Args:
            state (np.ndarray): Current state from the environment.
            deterministic (bool): Whether to act greedily.

        Returns:
            Tuple containing:
                - action (int): Chosen action index.
                - log_prob (float): Log probability of chosen action.
                - value (float): Estimated value of the current state.
                - probs (np.ndarray): Action probabilities.
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the agent using a batch of experiences.

        Args:
            batch (dict): Training batch with keys like:
                - states (List[np.ndarray])
                - actions (List[int])
                - log_probs (List[float])
                - returns (List[float])
                - advantages (List[float])

        Returns:
            dict: Training metrics (e.g., losses, entropy).
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the agent’s model and optimizer state to a file.

        Args:
            path (str): Path to save the checkpoint.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model and optimizer state from file.

        Args:
            path (str): Path to checkpoint file.
        """
        pass
