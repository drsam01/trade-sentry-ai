# File: drlfusion/modelling/agents/agent_trainers/sac_trainer.py

"""
Trainer for Soft Actor-Critic (SAC) agent using off-policy replay buffer.
"""

import os
from typing import Optional
from drlfusion.modelling.agents.agent_trainers.base_trainer import BaseTrainer
from drlfusion.modelling.agents.agent_managers.sac_manager import SACManager
from drlfusion.modelling.utils.rollout_manager import RolloutManager


class SACTrainer(BaseTrainer):
    """
    Trainer for Soft Actor-Critic using experience replay and stochastic policies.
    """

    def __init__(
        self,
        manager: SACManager,
        env,
        val_env: Optional[any] = None,
        save_dir: Optional[str] = None,
        patience: int = 20,
        batch_size: int = 2048,
        plot_file: str = "training_progress.html"
    ) -> None:
        super().__init__(manager, env, val_env, save_dir, patience, batch_size, plot_file)

    def run_train(self, total_epochs: int) -> None:
        """
        Training loop for SAC using off-policy buffer.
        """
        for epoch in range(total_epochs):
            rollout = RolloutManager(env=self.env, choose_action_fn=self.manager.choose_action, on_policy=False)
            train_reward = rollout.collect(batch_size=self.batch_size)
            batch = rollout.build_batch()
            rollout.clear_buffer()

            # Skip if batch was empty
            if len(batch["states"]) == 0:
                continue

            stats = self.manager.update(batch)
            val_reward = self.evaluate(self.val_env) if self.val_env else None

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

