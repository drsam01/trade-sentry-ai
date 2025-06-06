# File: apps/drlfusion/agents/frameworks/a2c_agent.py

"""
A2C (Advantage Actor-Critic) agent implementation for DRLFusion System. 
With integrated A2CNetwork that implements synchronous actor-critic model 
using shared base agent interface.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from .base_agent import BaseAgent
from core.utils.logger import get_logger
from apps.drlfusion.training.utils.device_utils import get_best_device

logger = get_logger(__name__)


class A2CNetwork(nn.Module):
    """
    Actor-Critic neural network used by A2CAgent.
    Contains shared base layers and separate policy and value heads.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        """
        Initialize the A2C network.

        Args:
            input_dim (int): Number of input features (flattened state).
            output_dim (int): Number of possible discrete actions.
            hidden_dim (int): Size of hidden layers.
        """
        super(A2CNetwork, self).__init__()

        # Shared feature extractor (common to both actor and critic)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head: outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

        # Value head: outputs scalar state value
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # type: ignore
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Batch of input state tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: action probabilities, value estimates
        """
        base = self.shared(x)
        return self.policy_head(base), self.value_head(base)


class A2CAgent(BaseAgent):
    """
    A2C Agent with a shared actor-critic architecture for discrete action spaces.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        **kwargs
    ) -> None:
        """
        Initialize the A2CAgent with model and optimizer.

        Args:
            observation_space: Gym-style observation space object.
            action_space: Gym-style discrete action space object.
            hidden_dim (int): Size of hidden layers in A2C network.
            learning_rate (float): Learning rate for Adam optimizer.
            **kwargs: Optional future config overrides.
        """
        super().__init__(observation_space, action_space)
        self.device = get_best_device()

        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Build the actor-critic model and optimizer
        self.model = A2CNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        # Fixed optimizer (Adam); learning rate is tunable
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        logger.info(
            f"A2CAgent initialized | input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dim=self.{hidden_dim}, lr={self.learning_rate}"
        )

    def choose_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float, np.ndarray]:
        """
        Choose an action from the current policy given an input state.

        Args:
            state (np.ndarray): Environment state.
            deterministic (bool): If True, choose argmax action (greedy).

        Returns:
            Tuple[int, float, float, np.ndarray]:
                - Action index
                - Log probability of chosen action
                - Value estimate of current state
                - Raw action probability vector
        """
        self.model.eval()

        # Convert state to torch tensor and infer through model
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) # type: ignore
        action_probs, value = self.model(state_tensor)
        dist = torch.distributions.Categorical(action_probs) # type: ignore

        # Sample or take max action based on mode
        action = torch.argmax(action_probs, dim=-1) if deterministic else dist.sample() # type: ignore
        log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.item(),
            value.item(),
            action_probs.squeeze(0).detach().cpu().numpy()
        )

    def update(self, *args, **kwargs):
        """
        Model update is delegated to A2CManager and trainer routines.
        """
        raise NotImplementedError("Use A2CManager to update the A2C model.")

    def save(self, path: str) -> None:
        """
        Save model checkpoint to file.

        Args:
            path (str): File path to save model weights.
        """
        torch.save(self.model.state_dict(), path) # type: ignore
        logger.info(f"📦 A2C model saved to: {path}")

    def load(self, path: str) -> None:
        """
        Load model checkpoint from file.

        Args:
            path (str): Path to model weights.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device)) # type: ignore
        logger.info(f"📥 A2C model loaded from: {path}")
