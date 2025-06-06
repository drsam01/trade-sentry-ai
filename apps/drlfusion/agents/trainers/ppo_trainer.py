# File: drlfusion/modelling/agents/agent_trainers/ppo_trainer.py

"""
Trainer for PPO agent using on-policy rollouts and early stopping.
"""

import os
from typing import Optional, Any
from .base_trainer import BaseTrainer
from apps.drlfusion.agents.managers.ppo_manager import PPOManager


class PPOTrainer(BaseTrainer):
    """
    Proximal Policy Optimization trainer. Collects on-policy rollouts,
    computes advantages, and performs policy/value updates.
    """

    def __init__(
        self,
        manager: PPOManager,
        env,
        val_env: Optional[Any] = None,
        save_dir: Optional[str] = None,
        patience: int = 20,
        batch_size: int = 2048,
        plot_file: str = "training_progress.html"
    ) -> None:
        """
        Args:
            manager (PPOManager): The PPO manager object.
            env: Training environment.
            val_env: Optional validation environment.
            save_dir: Path to save model checkpoints.
            patience: Early stopping patience.
            batch_size: Rollout batch size.
            plot_file: HTML file name for progress plots.
        """
        super().__init__(manager, env, val_env, save_dir, patience, batch_size, plot_file)

    def run_train(self, total_epochs: int) -> None:
        """
        Training loop for PPO.

        Args:
            total_epochs (int): Number of training epochs.
        """
        for epoch in range(total_epochs):
            memory = self.collect_experiences()
            train_reward = sum(memory["rewards"])

            # Compute returns and advantages using GAE
            returns, advantages = self.manager.compute_returns_and_advantages(memory)
            batch = {**memory, "returns": returns, "advantages": advantages}
            stats = self.manager.update(batch)

            val_reward = self.evaluate(self.val_env) if self.val_env else None

            # Log results, stop if needed
            should_stop = self.log_and_maybe_stop(
                epoch, 
                train_reward, 
                val_reward, 
                stats, 
                model_tag=self.manager.agent.__class__.__name__
            )

            self.plot_progress(epoch)
            
            if should_stop:
                break

