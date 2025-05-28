# File: drlfusion/modelling/agents/agent_trainers/base_trainer.py

"""
Unified RL trainer base class for DRLFusion agents.
Supports reward tracking, early stopping, evaluation, plotting,
and both on-policy and off-policy training compatibility.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from drlfusion.modelling.utils.stop_criteria import TrainingStopCriteria
from drlfusion.modelling.utils.rollout_tools import safe_step, safe_reset

from drlfusion.core.utils.logger import get_logger

logger = get_logger("base_trainer")


class BaseTrainer(ABC):
    """
    Shared base class for all RL agent trainers.
    """

    def __init__(
        self,
        manager: Any,
        env: Any,
        val_env: Optional[Any] = None,
        save_dir: Optional[str] = None,
        patience: int = 20,
        batch_size: int = 2048,
        plot_file: str = "training_progress.html"
    ) -> None:
        """
        Initialize trainer with training and validation environments.

        Args:
            manager (Any): The agent manager (e.g., PPOManager).
            env (Any): Training environment.
            val_env (Optional[Any]): Optional validation environment.
            save_dir (Optional[str]): Path to save models and plots.
            patience (int): Early stopping patience.
            batch_size (int): Number of environment steps per batch.
            plot_file (str): Output filename for reward plots.
        """
        self.manager = manager
        self.env = env
        self.val_env = val_env
        self.save_dir = save_dir
        self.patience = patience
        self.batch_size = batch_size
        self.plot_file = plot_file

        self.train_rewards: List[float] = []
        self.val_rewards: List[float] = []
        self.best_reward = -np.inf

        # Configure stopping logic
        self.stopper = TrainingStopCriteria(
            early_stop=True,
            convergence=False,
            early_stop_params={"min_delta": 1.0, "patience": patience, "min_epochs": 50}
        )

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    @abstractmethod
    def run_train(self, total_epochs: int) -> None:
        """
        Main training loop. To be implemented by each trainer subclass.

        Args:
            total_epochs (int): Number of epochs to train.
        """
        pass

    def collect_experiences(self) -> Dict[str, list]:
        """
        Collect on-policy experiences from the environment using the current agent.

        Returns:
            dict: Rollout memory containing state/action/log-prob/reward/advantage info.
        """
        memory = {k: [] for k in ['states', 'actions', 'log_probs', 'rewards', 'dones', 'values']}

        # Safe reset across Gym versions
        state = safe_reset(self.env)
        done = False
        truncated = False

        while len(memory["states"]) < self.batch_size and not done and not truncated:
            # Sample action and track log_prob + value for advantage calc
            action, log_prob, value, _ = self.manager.choose_action(state, deterministic=False)

            # Safe step across Gym versions
            step_result = self.env.step(action)
            next_state, reward, done, truncated = safe_step(step_result)

            memory["states"].append(state)
            memory["actions"].append(action)
            memory["log_probs"].append(log_prob)
            memory["rewards"].append(reward)
            memory["dones"].append(done or truncated)
            memory["values"].append(value)

            state = next_state

        # Bootstrap final value for GAE if needed (supports 2 or 3 return values from model)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.manager.agent.device)

            # Run model forward to obtain value estimate (and optionally reconstruction)
            model_output = self.manager.agent.model(state_tensor)

            if not isinstance(model_output, tuple):
                raise TypeError(f"Expected model to return tuple, got {type(model_output)}")

            if len(model_output) == 2:
                _, last_value = model_output
            elif len(model_output) == 3:
                _, last_value, _ = model_output
            else:
                raise ValueError(f"Unsupported number of outputs from model: {len(model_output)}")
            memory["values"].append(last_value.item())

        return memory

    def evaluate(self, env: Any) -> float:
        """
        Evaluate agent's performance on a single episode using deterministic policy.

        Args:
            env (Any): Gym-compatible environment.

        Returns:
            float: Total reward collected during evaluation episode.
        """
        # Support Gym reset() returning state or (state, info)
        state = safe_reset(env)
        total_reward = 0.0
        done, truncated = False, False

        while not (done or truncated):
            # Greedy action selection
            action, _, _, _ = self.manager.choose_action(state, deterministic=True)
            
            # Universal unpacking for step()
            step_result = env.step(action)
            next_state, reward, done, truncated = safe_step(step_result)

            total_reward += reward
            state = next_state

        return total_reward

    def build_batch(self, memory: Dict[str, list]) -> Dict[str, list]:
        """
        Compute GAE-based advantages and returns.

        Args:
            memory (dict): On-policy buffer from collect_experiences.

        Returns:
            dict: Final training batch with returns and advantages.
        """
        gamma = 0.99
        lam = 0.95
        values = memory["values"]
        rewards = memory["rewards"]
        dones = memory["dones"]

        returns = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])

        advantages = [ret - val for ret, val in zip(returns, values[:-1])]

        return {
            "states": memory["states"],
            "actions": memory["actions"],
            "log_probs": memory["log_probs"],
            "returns": returns,
            "advantages": advantages
        }

    def log_and_maybe_stop(
        self,
        epoch: int,
        train_reward: float,
        val_reward: Optional[float],
        stats: Dict[str, float],
        model_tag: str
    ) -> bool:
        """
        Log performance, apply early stopping, and save best model if improved.

        Args:
            epoch (int): Current training epoch.
            train_reward (float): Total reward from training rollout.
            val_reward (Optional[float]): Optional validation score.
            stats (dict): Loss/metric stats from update().
            model_tag (str): Agent identifier for filename.

        Returns:
            bool: True if training should stop.
        """
        self.train_rewards.append(train_reward)
        if val_reward is not None:
            self.val_rewards.append(val_reward)

        logger.info(f"Epoch {epoch+1} - Reward: {train_reward:.2f}" +
                    (f", Val: {val_reward:.2f}" if val_reward else "") +
                    f", Loss: {stats.get('loss', stats.get('total_loss', 0.0)):.4f}")

        if self.save_dir and train_reward > self.best_reward:
            self.best_reward = train_reward
            filename = f"best_{model_tag}_model.pt"
            self.manager.save(os.path.join(self.save_dir, filename))
            logger.info(f"📦 Model checkpoint saved: {filename} (reward={self.best_reward:.2f})")

        return self.stopper.check(epoch, train_reward, val_reward)
    
    '''
    def plot_progress(self, epoch: int) -> None:
        """
        Generate and optionally save reward plots.

        Args:
            epoch (int): Current epoch for plot labeling.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.train_rewards, mode="lines", name="Train Reward"))
        if self.val_rewards:
            fig.add_trace(go.Scatter(y=self.val_rewards, mode="lines", name="Val Reward"))

        fig.update_layout(
            title=f"Training Progress - Epoch {epoch+1}",
            xaxis_title="Epoch",
            yaxis_title="Reward",
            template="plotly_white"
        )'''
    
    def plot_progress(self, epoch: int) -> None:
        """
        Generate multi-subplot training progress visualization:
        - Subplot 1: Train vs Val reward
        - Subplot 2: Convergence Metric (Validation delta (Δ))
        - Subplot 3: Patience (Epochs Since Last Improvement)
        """
        # === Clean val rewards for plotting (removing None/nan) ===
        val_rewards_cleaned = [v for v in self.val_rewards if v is not None and not np.isnan(v)]
        
        # === Compute delta changes in validation reward to visualize convergence ===
        val_deltas = [
            val_rewards_cleaned[i] - val_rewards_cleaned[i - 1]
            for i in range(1, len(val_rewards_cleaned))
        ] if len(val_rewards_cleaned) > 1 else []
        
        # === Initialize patience tracker list if not already present ===
        if not hasattr(self, "patience_tracker"):
            self.patience_tracker: list[int] = []

            # === Append current epoch's early stopping patience count ===
        current_patience = self.stopper.early_monitor.epochs_no_improve
        self.patience_tracker.append(current_patience)

        '''
        # === Patience tracker from EarlyStoppingMonitor ===
        patience_values = [
            min(i, self.stopper.early_monitor.epochs_no_improve)
            for i in range(len(self.train_rewards))
        ]
        '''        
        
        # === Build subplots ===
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=[
            "Train vs Validation Reward",
            "Validation Reward Δ (Convergence Monitor)",
            "Epochs Without Improvement (Patience Tracker)"
        ]
            #("Episode vs Validation Reward", "Validation Reward Delta (Convergence)", "Raw Validation Reward")
        )

        # === Subplot 1: Train and Validation Rewards ===
        fig.add_trace(go.Scatter(
            y=self.train_rewards,
            mode="lines+markers",
            name="Train Reward"
        ), row=1, col=1)

        if val_rewards_cleaned:
            fig.add_trace(go.Scatter(
                y=val_rewards_cleaned,
                mode="lines+markers",
                name="Val Reward"
            ), row=1, col=1)

        # === Subplot 2: Δ in Validation Reward ===
        if val_deltas:
            fig.add_trace(go.Scatter(
                y=val_deltas,
                mode="lines+markers",
                name="Val Reward Δ"
            ), row=2, col=1)

        # === Subplot 3: Patience Tracker ===
        fig.add_trace(go.Bar(
            y=self.patience_tracker,
            name="Epochs w/o Improvement"
        ), row=3, col=1)

        # === Overlay patience threshold line ===
        patience_threshold = self.stopper.early_monitor.patience
        fig.add_trace(go.Scatter(
            y=[patience_threshold] * len(self.patience_tracker),
            mode="lines",
            name="Patience Threshold",
            line=dict(dash="dash", color="red")
        ), row=3, col=1)

        # Final layout update
        fig.update_layout(
            height=850,
            title=f"Training Progress - Epoch {epoch+1}",
            template="plotly_white",
            showlegend=True
        )
        
        # === Save to disk ===
        if self.save_dir:
            fig.write_html(os.path.join(self.save_dir, self.plot_file))
