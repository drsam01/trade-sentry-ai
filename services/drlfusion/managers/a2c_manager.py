# File: drlfusion/modelling/agents/agent_managers/a2c_manager.py

"""
A2C Manager for DRLFusion System.
Handles training updates, value loss, and entropy regularization.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List

from drlfusion.modelling.agents.agent_managers.base_manager import BaseManager
from drlfusion.modelling.agents.agent_architectures.a2c_agent import A2CAgent
from drlfusion.modelling.utils.device_utils import get_best_device
from drlfusion.core.utils.logger import get_logger

logger = get_logger("a2c_manager")


class A2CManager(BaseManager):
    """
    Manager for Advantage Actor-Critic algorithm.
    Handles training loop and model parameter updates.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        agent_config: dict = None,
        training_config: dict = None
    ) -> None:
        """
        Initialize A2C manager and optimizer.

        Args:
            observation_space: Env observation space.
            action_space: Env action space.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            entropy_coef (float): Entropy bonus scaling.
            value_coef (float): Value loss scaling.
        """
        agent_config = agent_config or {}
        training_config = training_config or {}
        
        self.device = get_best_device()        
        self.agent = A2CAgent(observation_space, action_space, **agent_config)
        self.optimizer = self.agent.optimizer
        
        self.gamma = training_config.get("gamma", 0.99)
        self.entropy_coef = training_config.get("entropy_coef", 0.01)
        self.value_coef = training_config.get("value_loss_coef", 0.5)

    def choose_action(self, state: np.ndarray, deterministic: bool = False):
        return self.agent.choose_action(state, deterministic)

    def update(self, batch: Dict[str, List]) -> Dict[str, float]:
        """
        Updates actor and critic using the computed returns.

        Args:
            batch: Dict with keys: states, actions, returns, log_probs.

        Returns:
            Dict of training statistics.
        """
        states = torch.tensor(batch["states"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long).to(self.device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32).to(self.device)

        self.agent.optimizer.zero_grad()
        probs, values = self.agent.model(states)
        dist = torch.distributions.Categorical(probs)

        # Compute actor loss (negative log prob * advantage)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        advantages = returns - values.squeeze(-1)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Compute critic loss (MSE)
        critic_loss = advantages.pow(2).mean()

        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        loss.backward()
        self.agent.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item()
        }
    
    '''
    def update(self, batch: Dict[str, list]) -> Dict[str, float]:
        """
        Update A2C model using collected experience.

        Args:
            batch (dict): Contains 'states', 'actions', 'log_probs', 'returns', 'advantages'

        Returns:
            Dict[str, float]: Training loss components.
        """

        states = torch.tensor(np.array(batch['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
        returns = torch.tensor(batch['returns'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(batch['advantages'], dtype=torch.float32).to(self.device)

        # Forward pass
        action_probs, values = self.agent.model(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Loss components
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values.squeeze(), returns)
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item()
        }'''

    def compute_returns(self, memory: dict, gamma: float = 0.99) -> List[float]:
        """
        Compute discounted returns for A2C.

        Args:
            memory (dict): Rollout memory with rewards and dones.
            gamma (float): Discount factor.

        Returns:
            List[float]: Discounted returns.
        """
        rewards = memory["rewards"]
        dones = memory["dones"]
        returns = []
        G = 0.0

        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + gamma * G * (1 - d)
            returns.insert(0, G)

        return returns

    def save(self, path: str) -> None:
        torch.save({
            "model": self.agent.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)
        logger.info(f"A2C checkpoint saved to {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.agent.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"A2C checkpoint loaded from {path}")
