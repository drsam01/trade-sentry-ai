# File: drlfusion/modelling/training/utils/rollout_manager.py

""" 
A unified, reusable Rollout Manager class that wraps all functionality from the DRLFusion rollout toolkit:

- Modular rollout for on-policy and off-policy
- Built-in logging and step counters
- Auto-detection of agent type (on_policy=True|False)
- Compatible with any Gym/Gymnasium environment

with built-in support for:

- Reward normalization (optional: running mean/std)
- Curriculum scheduling hooks (callable per rollout)
- Still compatible with both on-policy and off-policy agents
"""
from typing import Callable, List, Tuple, Dict, Union, Optional
from drlfusion.modelling.utils.rollout_tools import (
    collect_on_policy,
    collect_off_policy,
    build_gae_batch,
    build_replay_batch
)
from drlfusion.core.utils.logger import get_logger
import numpy as np

logger = get_logger("rollout_manager")


class RolloutManager:
    """
    DRLFusion's modular rollout manager for agents with support for on-policy/off-policy collection,
    reward normalization, and curriculum adjustment hooks.
    """

    def __init__(
        self,
        env,
        choose_action_fn: Callable,
        on_policy: bool = True,
        gamma: float = 0.99,
        lam: float = 0.95,
        is_ppo: bool = False,
        normalize_rewards: bool = False,
        curriculum_fn: Optional[Callable[[int, float], None]] = None,
    ):
        """
        Args:
            env: Gym-compatible environment
            choose_action_fn: agent.choose_action(state)
            on_policy (bool): True for PPO/A2C, False for DQN/SAC
            gamma (float): Discount factor
            lam (float): GAE lambda
            is_ppo (bool): PPO flag to include log_probs
            normalize_rewards (bool): Enable reward normalization
            curriculum_fn (Callable): Optional hook: (episode_index, episode_reward) → None
        """
        self.env = env
        self.choose_action_fn = choose_action_fn
        self.on_policy = on_policy
        self.gamma = gamma
        self.lam = lam
        self.is_ppo = is_ppo

        self.replay_buffer: List[Tuple] = []
        self.step_counter = 0
        self.episode_counter = 0

        # Reward normalization state
        self.normalize_rewards = normalize_rewards
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 1e-6  # avoid division by zero

        # Optional curriculum scheduler hook
        self.curriculum_fn = curriculum_fn

    def _normalize(self, rewards: List[float]) -> List[float]:
        """
        Applies online normalization to a reward sequence using running stats.

        Args:
            rewards (List[float]): Unnormalized rewards

        Returns:
            List[float]: Normalized rewards
        """
        rewards_np = np.array(rewards)
        self.running_mean = 0.99 * self.running_mean + 0.01 * rewards_np.mean()
        self.running_var = 0.99 * self.running_var + 0.01 * rewards_np.var()
        normalized = (rewards_np - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)
        return normalized.tolist()

    def collect(self, batch_size: int = 2048) -> Union[Dict, float]:
        """
        Collects experiences and applies reward normalization / curriculum scheduling.

        Returns:
            On-policy: memory (dict)
            Off-policy: total reward (float)
        """
        if not self.on_policy:
            return self._collect_off_policy_wrapper(batch_size)
        
        # On-policy mode: use rollout_tools utility
        memory = collect_on_policy(self.env, self.choose_action_fn, batch_size)
        if self.normalize_rewards and memory["rewards"]:
            memory["rewards"] = self._normalize(memory["rewards"])
        
        # Optionally normalize collected rewards
        self.step_counter += len(memory["states"])
        logger.info(f"🎯 On-policy: {len(memory['states'])} steps collected")
        return memory

    def _collect_off_policy_wrapper(self, batch_size):
        """
        Off-policy rollout wrapper that also triggers curriculum updates.

        Args:
            batch_size (int): Number of steps to collect

        Returns:
            float: Reward collected during the rollout
        """
        reward = collect_off_policy(self.env, self.choose_action_fn, self.replay_buffer, batch_size)
        self.step_counter += batch_size
        self.episode_counter += 1
        logger.info(f"🎯 Off-policy: {batch_size} steps | Reward: {reward:.2f}")

        # Trigger curriculum hook if provided
        if self.curriculum_fn:
            self.curriculum_fn(self.episode_counter, reward)

        return reward

    def build_batch(self, memory: Union[Dict, None] = None) -> Dict:
        """
        Constructs a batch from collected transitions.

        Args:
            memory (Optional[Dict]): For on-policy only (passed from collect())

        Returns:
            Dict: Training-ready batch
        """
        if self.on_policy:
            return build_gae_batch(memory, gamma=self.gamma, lam=self.lam, is_ppo=self.is_ppo)
        else:
            return build_replay_batch(self.replay_buffer)

    def clear_buffer(self) -> None:
        """
        Clears stored off-policy replay transitions.
        """
        if not self.on_policy:
            logger.debug("🧹 Cleared off-policy replay buffer.")
            self.replay_buffer.clear()
