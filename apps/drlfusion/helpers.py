# File: apps/drlfusion/helpers.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from apps.drlfusion.agents.frameworks.ppo_agent import PPOAgent
from apps.drlfusion.agents.frameworks.a2c_agent import A2CAgent
from apps.drlfusion.agents.frameworks.dqn_agent import DQNAgent
from apps.drlfusion.agents.frameworks.sac_agent import SACAgent
from apps.drlfusion.envs.feature_builder import FeatureBuilder, RunningNormalizer
from core.utils.logger import get_logger

logger = get_logger(__name__)


class DRLFusionHelper:
    """
    Handles feature extraction, normalization, agent ensemble inference, and soft voting.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = {}
        self.feature_builder: Optional[FeatureBuilder] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.feature_names: list[str] = []
        self.normalizer: Optional[RunningNormalizer] = None
        self.state_dim: Optional[int] = None
        self.candle_window = config.get("global_hyperparams", {}).get("candle_window", 30)

        self._load_agents()

    def _load_agents(self) -> None:
        """
        Load DRL agents based on model paths in config.
        """
        paths = self.config.get("model_paths", {})
        self.agents = {
            "ppo": PPOAgent.load(paths.get("ppo")), # type: ignore
            "a2c": A2CAgent.load(paths.get("a2c")), # type: ignore
            "dqn": DQNAgent.load(paths.get("dqn")), # type: ignore
            "sac": SACAgent.load(paths.get("sac")), # type: ignore
        }
        logger.info(f"Loaded DRL agents: {list(self.agents.keys())}")

    def prepare(self, df: pd.DataFrame) -> None:
        """
        Constructs features from OHLCV data and initializes normalizer.

        Args:
            df (pd.DataFrame): Raw OHLCV DataFrame
        """
        self.feature_builder = FeatureBuilder(df)
        data_raw = self.feature_builder.prepare()
        self.feature_names = self.feature_builder.get_feature_columns()
        self.feature_df = data_raw[self.feature_names]

        feat_dim = self.feature_df.shape[1]
        self.state_dim = feat_dim
        obs_shape = (self.candle_window * feat_dim,)
        self.normalizer = RunningNormalizer(np.prod(obs_shape)) # type: ignore
        self.normalizer._update(self.feature_df) # type: ignore

    def get_window_features(self, current_index: int) -> np.ndarray:
        """
        Extract normalized flat observation vector of candle window size ending at current_index.

        Args:
            current_index (int): Index of last included row

        Returns:
            np.ndarray: Flattened normalized window with shape [1, obs_dim]
        """
        if self.feature_df is None or current_index < self.candle_window:
            raise ValueError("Insufficient data for window features.")

        window = self.feature_df.iloc[current_index - self.candle_window:current_index].values
        obs = window.reshape(1, -1)  # shape: [1, candle_window * feat_dim]
        return self.normalizer.normalize(obs) # type: ignore

    def infer_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, Dict[str, int], Dict[str, float], Dict[int, float]]:
        """
        Run ensemble agent inference and aggregate via soft-voting.

        Args:
            state (np.ndarray): Flattened normalized state vector [1, obs_dim]
            deterministic (bool): Use greedy policy (vs. stochastic)

        Returns:
            - final_action (int): Ensemble-voted action
            - vote_map (Dict[str, int]): agent → chosen action
            - conf_map (Dict[str, float]): agent → confidence score
            - probs (Dict[int, float]): action → average confidence
        """
        vote_map: Dict[str, int] = {}
        conf_map: Dict[str, float] = {}
        action_confidence: Dict[int, list] = {}

        for name, agent in self.agents.items():
            action, _, _, probs = agent.choose_action(state, deterministic=deterministic)
            vote_map[name] = action
            conf_map[name] = float(probs[action])

            if action not in action_confidence:
                action_confidence[action] = []
            action_confidence[action].append(conf_map[name])

        probs = {action: float(np.mean(scores)) for action, scores in action_confidence.items()}
        final_action = max(set(vote_map.values()), key=list(vote_map.values()).count)

        return final_action, vote_map, conf_map, probs
