# File: apps/drlfusion/modelling/agents/managers/dqn_manager.py

"""
DQN Manager for DRLFusion System.
Handles replay buffer training, target network syncing, and model optimization.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import List, Dict, Any, Optional
from .base_manager import BaseManager
from apps.drlfusion.agents.frameworks.dqn_agent import DQNAgent, QNetwork
from core.utils.device_utils import get_best_device
from core.utils.logger import get_logger

logger = get_logger(__name__)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Simple cyclic replay buffer for off-policy learning.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


class DQNManager(BaseManager):
    """
    Manages DQNAgent training using experience replay and target networks.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        agent_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize manager and training components.

        Args:
            observation_space: Env observation space.
            action_space: Env action space.
            buffer_capacity: Replay buffer max size.
            batch_size: Mini-batch size.
            gamma: Discount factor.
            lr: Learning rate.
            target_update_freq: Frequency (steps) to update target net.
        """
        '''buffer_capacity: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        target_update_freq: int = 500'''

        agent_config = agent_config or {}
        training_config = training_config or {}
        self.device = get_best_device()
        
        self.agent = DQNAgent(observation_space, action_space, **agent_config)
        self.target_net = QNetwork(self.agent.input_dim, self.agent.output_dim, self.agent.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.agent.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.agent.q_net.parameters(), lr=self.agent.learning_rate) # type: ignore
        self.buffer = ReplayBuffer(capacity=training_config.get("REPLAY_BUFFER_SIZE", 100_000))

        self.batch_size = training_config.get("BATCH_SIZE", 64)
        self.gamma = training_config.get("GAMMA", 0.99)
        self.target_update_freq = training_config.get("TARGET_UPDATE_FREQ", 500)
        self.step_counter = 0

    def choose_action(self, state: np.ndarray, deterministic: bool = False):
        return self.agent.choose_action(state, deterministic)

    def update(self, batch: Dict[str, Any] | None = None) -> Dict[str, float]:
        """
        Sample from replay and update Q-network.

        Returns:
            dict: Training statistics.
        """
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0}

        transitions = self.buffer.sample(self.batch_size)
        batch = batch or Transition(*zip(*transitions)) # type: ignore

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device) # type: ignore
        action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device) # type: ignore
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device) # type: ignore
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device) # type: ignore
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device) # type: ignore

        # Q(s,a)
        q_values = self.agent.q_net(state_batch).gather(1, action_batch)

        # max_a' Q_target(s',a')
        with torch.no_grad(): # type: ignore
            max_next_q = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            q_target = reward_batch + self.gamma * max_next_q * (1 - done_batch)

        loss = F.mse_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodic target update
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.agent.q_net.state_dict())
            logger.info("Target network updated.")

        return {"loss": loss.item()}

    def remember(self, state, action, reward, next_state, done):
        """Store experience in buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def save(self, path: str) -> None:
        torch.save({ # type: ignore
            "q_net": self.agent.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)
        logger.info(f"DQN checkpoint saved to {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        checkpoint = torch.load(path) # type: ignore
        self.agent.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"DQN checkpoint loaded from {path}")
