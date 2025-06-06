# File: drlfusion/modelling/agents/agent_trainers/dqn_trainer.py

"""
Trainer for Deep Q-Network (DQN) agent using replay buffer.
"""

import os
from typing import Optional, Any
from .base_trainer import BaseTrainer
from apps.drlfusion.agents.managers.dqn_manager import DQNManager
from core.utils.rollout_manager import RolloutManager


class DQNTrainer(BaseTrainer):
    """
    Deep Q-Network trainer using experience replay and target network updates.
    """

    def __init__(
        self,
        manager: DQNManager,
        env,
        val_env: Optional[Any] = None,
        save_dir: Optional[str] = None,
        patience: int = 20,
        batch_size: int = 2048,
        plot_file: str = "training_progress.html"
    ) -> None:
        super().__init__(manager, env, val_env, save_dir, patience, batch_size, plot_file)

    def run_train(self, total_epochs: int) -> None:
        """
        Training loop for DQN using off-policy buffer.
        """
        for epoch in range(total_epochs):
            rollout = RolloutManager(env=self.env, choose_action_fn=self.manager.choose_action, on_policy=False)
            train_reward = rollout.collect(batch_size=self.batch_size)
            batch = rollout.build_batch()
            rollout.clear_buffer()

            if len(batch["states"]) == 0:
                continue

            stats = self.manager.update(batch)
            val_reward = self.evaluate(self.val_env) if self.val_env else None
            
            should_stop = self.log_and_maybe_stop(
                epoch, 
                train_reward,  # type: ignore
                val_reward, 
                stats, 
                model_tag=self.manager.agent.__class__.__name__
            )

            self.plot_progress(epoch)
            
            if should_stop:
                break


