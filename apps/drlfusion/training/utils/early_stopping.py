# File: drlfusion/modelling/utils/early_stopping.py

import logging
from typing import Optional

logger = logging.getLogger("early_stopping")

class EarlyStoppingMonitor:
    """
    Early stopping monitor for RL training based on reward plateaus.

    Attributes:
        min_delta (float): Minimum improvement in reward to reset patience.
        patience (int): Max epochs to wait without improvement.
        min_epochs (int): Number of epochs to train before early stopping is allowed.
        best_reward (float): Best observed reward.
        epochs_no_improve (int): Counter for epochs without significant improvement.
    """

    def __init__(self, min_delta: float = 1.0, patience: int = 20, min_epochs: int = 50):
        self.min_delta = min_delta
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_reward = float("-inf")
        self.epochs_no_improve = 0

    def check(self, epoch: int, reward: float) -> bool:
        """
        Update monitor with current reward. Return True if early stopping is triggered.

        Args:
            epoch (int): Current epoch number (zero-indexed).
            reward (float): Latest training reward.

        Returns:
            bool: True if early stopping condition is met.
        """
        if reward > self.best_reward + self.min_delta:
            logger.info(f"✅ Reward improved: {self.best_reward:.2f} → {reward:.2f}")
            self.best_reward = reward
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            logger.info(
                f"⚠️ No improvement (Δ={reward - self.best_reward:.2f}), patience={self.epochs_no_improve}/{self.patience}"
            )

        if epoch >= self.min_epochs and self.epochs_no_improve >= self.patience:
            logger.info(f"🛑 Early stopping at epoch {epoch + 1}. Best reward: {self.best_reward:.2f}")
            return True

        return False
