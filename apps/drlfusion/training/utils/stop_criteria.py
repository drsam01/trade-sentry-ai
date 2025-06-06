# File: drlfusion/modelling/utils/stop_criteria.py

from drlfusion.modelling.utils.early_stopping import EarlyStoppingMonitor
from drlfusion.modelling.utils.convergence_monitor import ConvergenceMonitor
import logging

logger = logging.getLogger("stop_criteria")


class TrainingStopCriteria:
    """
    Unified controller for early stopping and convergence monitoring.

    Args:
        early_stop (bool): Enable early stopping by train reward.
        convergence (bool): Enable convergence tracking on val reward or entropy.
    """

    def __init__(
        self,
        early_stop: bool = True,
        convergence: bool = False,
        early_stop_params: dict = None,
        convergence_params: dict = None,
    ):
        self.use_early_stop = early_stop
        self.use_convergence = convergence

        self.early_monitor = EarlyStoppingMonitor(**(early_stop_params or {}))
        self.converge_monitor = ConvergenceMonitor(**(convergence_params or {}))

    def check(
        self,
        epoch: int,
        train_reward: float,
        val_metric: float = None,
    ) -> bool:
        """
        Run all active stopping checks.

        Args:
            epoch (int): Current epoch number.
            train_reward (float): Reward from training rollout.
            val_metric (float): Optional validation reward or entropy.

        Returns:
            bool: True if training should stop.
        """
        stop = False

        if self.early_monitor.check(epoch, train_reward) and self.use_early_stop:
            logger.info("🛑 Stopping: early reward plateau.")
            stop = True

        if self.use_convergence and val_metric is not None and self.converge_monitor.update(val_metric):
            logger.info("🛑 Stopping: convergence on validation metric.")
            stop = True

        return stop
