# File: drlfusion/modelling/agents/agent_trainers/a2c_trainer.py

"""
Trainer for A2C agent using synchronous policy/value updates.
"""

import os
from typing import Optional
from drlfusion.modelling.agents.agent_trainers.base_trainer import BaseTrainer
from drlfusion.modelling.agents.agent_managers.a2c_manager import A2CManager


class A2CTrainer(BaseTrainer):
    """
    Advantage Actor-Critic trainer using synchronous updates and return-based value learning.
    """

    def __init__(
        self,
        manager: A2CManager,
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
        Training loop for A2C.

        Args:
            total_epochs (int): Number of training epochs.
        """
        for epoch in range(total_epochs):
            memory = self.collect_experiences()
            train_reward = sum(memory["rewards"])

            # A2C uses discounted returns directly
            returns = self.manager.compute_returns(memory)
            batch = {**memory, "returns": returns}
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
