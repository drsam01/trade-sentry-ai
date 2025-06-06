# File: apps/drlfusion/agents/frameworks/sac_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from .base_agent import BaseAgent
from apps.drlfusion.training.utils.device_utils import get_best_device


class Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # type: ignore
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2) # type: ignore
        return mean, log_std


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: # type: ignore
        x = torch.cat([state, action], dim=-1) # type: ignore
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) agent with Gaussian actor and discrete action mapping.

    Outputs 4 elements: discrete action (int), log probability (float),
    placeholder value (float), and one-hot action probabilities (np.ndarray).
    """
    def __init__(
        self, 
        observation_space,
        action_space,
        hidden_dim: int = 256,
        **kwargs
    ):
        """
        Initialize the SAC agent.

        Args:
            observation_space: Gym-style observation space.
            action_space: Gym-style action space.
            hidden_dim (int): Number of hidden units in actor network.
        """
        super().__init__(observation_space, action_space)
        self.device = get_best_device()
        self.input_dim = observation_space.shape[0]
        self.output_dim = 1  # Scalar action in continuous [-1, 1] space
        self.hidden_dim = hidden_dim

        # Gaussian actor network
        self.actor = Actor(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)

    def choose_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[int, float, float, np.ndarray]:
        """
        Choose an action given the environment state.

        Args:
            state (np.ndarray): Observation from the environment.
            deterministic (bool): If True, use mean action. If False, sample from distribution.

        Returns:
            Tuple containing:
                - discrete_action (int): {0=short, 1=hold, 2=long}
                - log_prob (float): Log probability of action (0.0 if deterministic)
                - value (float): Placeholder value (SAC doesn't use value head directly)
                - action_probs (np.ndarray): One-hot encoded discrete action vector
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) # type: ignore
        mean, log_std = self.actor(state_tensor)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)  # type: ignore # Always define dist

        if deterministic:
            action_tensor = torch.tanh(mean) # type: ignore
            log_prob = 0.0  # No log prob for deterministic action
        else:
            sampled_action = dist.rsample()  # Reparameterized sample
            action_tensor = torch.tanh(sampled_action) # type: ignore
            log_prob = dist.log_prob(sampled_action).sum(dim=-1).item()

        action_scalar = action_tensor.item()

        # Convert to discrete action space:
        # -1.0 to -0.33 → short (0), -0.33 to 0.33 → hold (1), 0.33 to 1.0 → long (2)
        discrete_action = int(np.digitize(action_scalar, bins=[-0.33, 0.33]))

        # One-hot encode the discrete action
        action_probs = np.zeros(3)
        action_probs[discrete_action] = 1.0

        value_placeholder = 0.0

        return discrete_action, log_prob, value_placeholder, action_probs

    def update(self, *args, **kwargs):
        pass  # Managed externally by SACManager

    def save(self, path: str) -> None:
        torch.save(self.actor.state_dict(), path) # type: ignore

    def load(self, path: str) -> None:
        self.actor.load_state_dict(torch.load(path, map_location=self.device)) # type: ignore
