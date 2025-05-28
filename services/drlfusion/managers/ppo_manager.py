# === ppo_manager.py ===

"""
PPO Model Manager for DRLFusion Trading System.
Handles training loop, checkpointing, and PPO update steps.
"""

import os
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple, List, Dict

from torch.optim.lr_scheduler import CosineAnnealingLR
from drlfusion.modelling.agents.agent_managers.base_manager import BaseManager
from drlfusion.modelling.agents.agent_architectures.ppo_agent import PPOAgent

from drlfusion.core.utils.logger import get_logger

logger = get_logger("ppo_manager")

class PPOManager(BaseManager):
    """
    PPO Manager handles optimization, saving, and model coordination.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        agent_config: dict = None,
        training_config: dict = None
    ) -> None:
        """
        Initialize PPO manager with model and optimizer.

        Args:
            observation_space: Env observation space.
            action_space: Env action space.
            lr: Learning rate.
            gamma: Discount factor.
            clip_epsilon: PPO clip threshold.
            aux_loss_weight: Weight for auxiliary decoder loss.
            entropy_coef: Entropy bonus weight.
        """
        super().__init__()
        agent_config = agent_config or {}
        training_config = training_config or {}
        
        self.agent = PPOAgent(observation_space, action_space, **agent_config)
        
        self.optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=self.agent.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000)
        
        self.gamma = training_config.get("GAMMA", 0.99)
        self.clip_epsilon = training_config.get("CLIP_EPSILON", 0.2)
        self.aux_loss_weight = training_config.get("AUX_LOSS_WEIGHT", 0.01)
        self.entropy_coef = training_config.get("ENTROPY_COEF", 0.01)

    def choose_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Select action using PPO agent.

        Args:
            state (np.ndarray): Current state.
            deterministic (bool): Whether to use greedy policy.

        Returns:
            Tuple[int, float, float, np.ndarray]: action, log_prob, value, action_probs
        """
        return self.agent.choose_action(state, deterministic)

    def update(self, batch: dict) -> dict:
        """
        Train PPO agent using a batch of experiences.

        Args:
            batch (dict): Batch with keys ['states', 'actions', 'log_probs', 'returns', 'advantages']

        Returns:
            dict: Training metrics including losses and entropy.
        """
        device = next(self.agent.model.parameters()).device

        states = torch.tensor(np.array(batch["states"]), dtype=torch.float32).to(device)
        actions = torch.tensor(batch["actions"], dtype=torch.long).to(device)
        old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32).to(device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32).to(device)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        action_probs, values, decoded = self.agent.model(states)
        dist = torch.distributions.Categorical(action_probs)

        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # PPO objective/surrogate loss
        ratios = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        value_loss = F.mse_loss(values.squeeze(), returns)

        # Auxiliary loss (reconstruction)
        aux_loss = F.mse_loss(decoded, states)

        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy + self.aux_loss_weight * aux_loss

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "aux_loss": aux_loss.item(),
            "total_loss": total_loss.item()
        }

    def compute_returns_and_advantages(self, memory: dict, gamma: float = 0.99, lam: float = 0.95) -> Tuple[List[float], List[float]]:
        """
        Computes returns and GAE-based advantages for PPO update.

        Args:
            memory (dict): Rollout memory with rewards, values, and dones.
            gamma (float): Discount factor.
            lam (float): Lambda for GAE smoothing.

        Returns:
            Tuple[List[float], List[float]]: returns and advantages
        """
        rewards = memory["rewards"]
        values = memory["values"]
        dones = memory["dones"]

        returns = []
        advantages = []
        gae = 0.0

        # Ensure last value is bootstrapped
        values = values + [values[-1]]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return returns, advantages

    def save(self, path: str) -> None:
        """
        Save model and optimizer state to file.

        Args:
            path (str): File path for checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.agent.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {
                "gamma": self.gamma,
                "clip_epsilon": self.clip_epsilon,
                "aux_loss_weight": self.aux_loss_weight,
                "entropy_coef": self.entropy_coef
            }
        }
        tmp_path = f"{path}.tmp"
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
            logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load(self, path: str) -> None:
        """
        Load model and optimizer from file.

        Args:
            path (str): File path for checkpoint.
        """
        if not os.path.exists(path):
            logger.error(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        try:
            self.agent.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.warning(f"Partial model load: {e}")
            self.agent.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Failed to load optimizer/scheduler: {e}")

        config = checkpoint.get("config", {})
        self.gamma = config.get("gamma", self.gamma)
        self.clip_epsilon = config.get("clip_epsilon", self.clip_epsilon)
        self.aux_loss_weight = config.get("aux_loss_weight", self.aux_loss_weight)
        self.entropy_coef = config.get("entropy_coef", self.entropy_coef)
