# File: drlfusion/modelling/agents/agent_managers/sac_manager.py

"""
SAC Manager for DRLFusion System.
Handles updates for twin critics, actor, and entropy tuning in Soft Actor-Critic.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from drlfusion.modelling.agents.agent_managers.base_manager import BaseManager
from drlfusion.modelling.agents.agent_architectures.sac_agent import SACAgent, Critic
from drlfusion.modelling.utils.device_utils import get_best_device
from drlfusion.core.utils.logger import get_logger

logger = get_logger("sac_manager")


class SACManager(BaseManager):
    """
    Manages Soft Actor-Critic updates including twin Q-functions, policy, and entropy coefficient.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        agent_config: dict = None,
        training_config: dict = None
#        actor_lr: float = 3e-4,
#        critic_lr: float = 3e-4,
#        alpha: float = 0.2,
#        gamma: float = 0.99,
#        tau: float = 0.005
    ):
        agent_config = agent_config or {}
        training_config = training_config or {}
        
        self.device = get_best_device()
        
        self.agent = SACAgent(observation_space, action_space, **agent_config)
        self.input_dim = self.agent.input_dim
        self.output_dim = self.agent.output_dim
        self.hidden_dim = self.agent.hidden_dim

        self.critic_1 = Critic(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
        self.critic_2 = Critic(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
        self.target_critic_1 = Critic(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
        self.target_critic_2 = Critic(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_opt = torch.optim.Adam(self.agent.actor.parameters(), lr=training_config.get("ACTOR_LR", 1e-4))
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=training_config.get("CRITIC_LR", 1e-4))
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=training_config.get("CRITIC_LR", 1e-4))

        self.gamma = training_config.get("GAMMA", 0.99)
        self.tau = training_config.get("TAU", 0.005)
        self.alpha = training_config.get("ALPHA", 0.2)  # Entropy coefficient
        
    def choose_action(self, state: np.ndarray, deterministic: bool = False):
        return self.agent.choose_action(state, deterministic)

    def update(self, batch: Dict[str, list]) -> Dict[str, float]:
        states = torch.tensor(np.array(batch["states"]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(batch["actions"]), dtype=torch.float32).to(self.device)
        # Ensure shape is [batch_size, action_dim]
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(batch["next_states"]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32).unsqueeze(1).to(self.device)

        # Critic loss
        with torch.no_grad():
            mean, log_std = self.agent.actor(next_states)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            next_actions = torch.tanh(dist.rsample())
            log_probs = dist.log_prob(next_actions).sum(-1, keepdim=True)

            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_min = torch.min(target_q1, target_q2) - self.alpha * log_probs
            target_value = rewards + (1 - dones) * self.gamma * target_min

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_1_loss = F.mse_loss(q1, target_value)
        critic_2_loss = F.mse_loss(q2, target_value)

        self.critic_1_opt.zero_grad()
        critic_1_loss.backward()
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        critic_2_loss.backward()
        self.critic_2_opt.step()

        # Actor loss
        mean, log_std = self.agent.actor(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        actions_sampled = torch.tanh(dist.rsample())
        log_probs = dist.log_prob(actions_sampled).sum(dim=-1, keepdim=True)

        q1_pi = self.critic_1(states, actions_sampled)
        q2_pi = self.critic_2(states, actions_sampled)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_probs - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update target networks
        self._soft_update(self.critic_1, self.target_critic_1)
        self._soft_update(self.critic_2, self.target_critic_2)

        return {
            "actor_loss": actor_loss.item(),
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "entropy": log_probs.mean().item()
        }

    def _soft_update(self, source_net: Critic, target_net: Critic):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        torch.save({
            "actor": self.agent.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict()
        }, path)
        logger.info(f"SAC checkpoint saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        checkpoint = torch.load(path)
        self.agent.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        logger.info(f"SAC checkpoint loaded from {path}")
