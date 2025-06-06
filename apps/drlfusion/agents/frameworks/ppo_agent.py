# File: apps/drlfusion/agents/frameworks/ppo_agent.py

"""
PPO Agent and Actor-Critic Network for DRLFusion System.
Inherits from BaseAgent and provides full PPO agent functionalities.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from .base_agent import BaseAgent
from apps.drlfusion.training.utils.device_utils import get_best_device
from core.utils.logger import get_logger


logger = get_logger(__name__)


class PPONetwork(nn.Module):
    """
    Shared actor-critic neural network with an auxiliary decoder head.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        """
        Initialize the actor-critic architecture.

        Args:
            input_dim (int): Flattened state vector dimension.
            output_dim (int): Number of possible actions.
        """
        super(PPONetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # type: ignore
        """
        Forward pass through the actor-critic model.

        Args:
            state (torch.Tensor): Input state tensor (batch_size, input_dim)

        Returns:
            Tuple containing:
                - action probabilities (batch_size, output_dim)
                - state value estimates (batch_size, 1)
                - reconstructed state (batch_size, input_dim)
        """
        latent = self.encoder(state)
        action_probs = self.actor(latent)
        state_value = self.critic(latent)
        reconstructed = self.decoder(latent)
        return action_probs, state_value, reconstructed


class PPOAgent(BaseAgent):
    """
    PPO Agent wrapping the Actor-Critic model.
    Used for both action selection and model storage.
    """

    def __init__(
        self, 
        observation_space, 
        action_space,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        **kwargs) -> None:
        """
        Initialize the PPO agent with model and logging.

        Args:
            observation_space: Gym observation space.
            action_space: Gym action space.
        """
        super().__init__(observation_space, action_space)
        self.device = get_best_device()
        
        self.input_dim = int(observation_space.shape[0])
        self.output_dim = action_space.n
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        self.model = PPONetwork(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
        
        logger.info(f"PPOAgent initialized with input_dim={self.input_dim}, output_dim={self.output_dim}, hidden_dim={self.hidden_dim}, lr={self.learning_rate}")

    def choose_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float, np.ndarray]:
        """
        Choose an action from the policy given the current state.

        Args:
            state (np.ndarray): Normalized state vector.
            deterministic (bool): Whether to use argmax (greedy) or sample from policy.

        Returns:
            Tuple[int, float, float, np.ndarray]:
                - chosen action index
                - log probability of the action
                - critic value estimate
                - full action probabilities
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device) # type: ignore

        with torch.no_grad(): # type: ignore
            action_probs, state_value, _ = self.model(state_tensor)
            probs = action_probs.cpu().numpy().squeeze(0)

        if deterministic:
            action = int(np.argmax(probs))
            log_prob = float(np.log(probs[action] + 1e-8))  # avoid log(0)
        else:
            dist = torch.distributions.Categorical(torch.from_numpy(probs)) # type: ignore
            action_tensor = dist.sample()
            action = int(action_tensor.item())
            log_prob = float(dist.log_prob(action_tensor).item())

        return action, log_prob, float(state_value.item()), probs

    def update(self, *args, **kwargs):
        """
        PPOAgent delegates training to PPOManager. Raises NotImplementedError.
        """
        raise NotImplementedError("PPOAgent uses PPOManager for training.")

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path (str): File path to store model weights.
        """
        torch.save(self.model.state_dict(), path) # type: ignore

    def load(self, path: str) -> None:
        """
        Load model weights from disk.

        Args:
            path (str): File path to load model weights from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device)) # type: ignore
        logger.info(f"Loaded PPOAgent model from: {path}")
