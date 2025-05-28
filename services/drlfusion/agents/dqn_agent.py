# File: drlfusion/modelling/agents/agent_architectures/dqn_agent.py

"""
DQN Agent for DRLFusion System.
Implements an epsilon-greedy Deep Q-Network using the shared BaseAgent interface.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from drlfusion.modelling.agents.agent_architectures.base_agent import BaseAgent
from drlfusion.modelling.utils.device_utils import get_best_device
from drlfusion.core.utils.logger import get_logger

# Set device (GPU or CPU)
device = get_best_device()
logger = get_logger("dqn_agent")


class QNetwork(nn.Module):
    """
    Feedforward Q-Network that estimates action-values for each discrete action.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Initialize the Q-network.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of possible actions.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Q-values for each action (batch_size, output_dim).
        """
        return self.net(x)


class DQNAgent(BaseAgent):
    """
    Deep Q-Learning agent using epsilon-greedy exploration.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        epsilon: float = 0.1
    ) -> None:
        """
        Initialize the DQN agent.

        Args:
            observation_space: Gym observation space.
            action_space: Gym action space.
            epsilon (float): Probability of selecting a random action (exploration).
        """
        super().__init__(observation_space, action_space)
        self.input_dim = int(observation_space.shape[0])
        self.output_dim = action_space.n
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.q_net = QNetwork(self.input_dim, self.output_dim, self.hidden_dim).to(device)
        
        logger.info(f"DQNAgent initialized with input_dim={self.input_dim}, output_dim={self.output_dim}")

    def choose_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float, np.ndarray]:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state (np.ndarray): Current observation.
            deterministic (bool): If True, no exploration.

        Returns:
            action (int): Chosen action index.
            log_prob (float): Not applicable for DQN (returns 0.0).
            value (float): Q-value of chosen action.
            probs (np.ndarray): One-hot of the chosen action.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.q_net(state_tensor).cpu().numpy().squeeze()

        if deterministic or np.random.rand() > self.epsilon:
            action = int(np.argmax(q_values))
        else:
            action = np.random.randint(self.output_dim)

        value = float(q_values[action])
        probs = np.eye(self.output_dim)[action]  # one-hot
        return action, 0.0, value, probs

    def update(self, *args, **kwargs):
        """
        This method is handled by DQNManager.
        """
        raise NotImplementedError("DQNAgent training is managed externally.")

    def save(self, path: str) -> None:
        """
        Save Q-network weights.

        Args:
            path (str): Target file path.
        """
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load Q-network weights.

        Args:
            path (str): Source file path.
        """
        self.q_net.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"Loaded DQNAgent weights from {path}")
