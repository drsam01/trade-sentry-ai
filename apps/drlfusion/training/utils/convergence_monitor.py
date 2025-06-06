# File: drlfusion/modelling/utils/convergence_monitor.py

from typing import Optional, List
import logging

logger = logging.getLogger("convergence_monitor")

class ConvergenceMonitor:
    """
    Monitors convergence of RL training using validation reward or entropy stability.

    Args:
        min_delta (float): Minimum improvement to consider significant.
        window_size (int): Number of recent epochs to average.
        max_plateau (int): Maximum tolerated plateaus (flat progress).
    """

    def __init__(self, min_delta: float = 0.5, window_size: int = 5, max_plateau: int = 10):
        self.min_delta = min_delta
        self.window_size = window_size
        self.max_plateau = max_plateau

        self.history: List[float] = []
        self.plateau_counter = 0
        self.prev_avg: Optional[float] = None

    def update(self, value: float) -> bool:
        """
        Update monitor with a new metric value (e.g., val reward or entropy).

        Args:
            value (float): New value to track.

        Returns:
            bool: True if convergence criteria are met (training can stop).
        """
        self.history.append(value)
        if len(self.history) < self.window_size:
            return False

        current_avg = sum(self.history[-self.window_size:]) / self.window_size

        if self.prev_avg is None:
            self.prev_avg = current_avg
            return False

        delta = current_avg - self.prev_avg

        if abs(delta) < self.min_delta:
            self.plateau_counter += 1
            logger.debug(f"Convergence plateau {self.plateau_counter}/{self.max_plateau} (Δ={delta:.4f})")
        else:
            self.plateau_counter = 0

        self.prev_avg = current_avg

        if self.plateau_counter >= self.max_plateau:
            logger.info("🧘 Convergence detected via plateau in metric.")
            return True

        return False
