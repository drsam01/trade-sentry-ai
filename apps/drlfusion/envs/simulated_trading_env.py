# File: apps/drlfusion/envs/simulated_trading_env.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

from .base_env import TradingEnvBase
from .feature_builder import FeatureBuilder, RunningNormalizer
from .reward_engine import ZhangRewardEngine
from .regime_filter import RegimeFilter
from .observation_builder import ObservationBuilder
from .trade_executor import TradeExecutor
from .env_metrics import EnvironmentMetrics
from .risk_engine import EnvRiskManager, TradeLogicEvaluator
from .env_config_loader import load_env_config
from core.utils.logger import get_logger



class SimulatedTradingEnv(TradingEnvBase):
    """
    DRL-compatible trading environment built on TradingEnvBase.
    Integrates regime detection, transaction-aware reward, and live feature normalization.
    """

    def __init__(self, data: pd.DataFrame, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the environment with OHLCV data and optional config dictionary.

        Args:
            data (pd.DataFrame): Raw historical OHLCV price data
            config (Dict): Environment-specific YAML config
        """
        self.logger = get_logger(__name__)
        self.config = config or load_env_config()

        # === Extract config sections ===
        self.tparams = self.config.get("trading_params", {})
        self.hparams = self.config.get("global_hyperparams", {})
        self.mtrain = self.config.get("model_training_config", {})
        
        self.max_hold_duration = self.tparams.get('max_hold_duration', 50)
        self.transaction_cost = self.tparams.get("transaction_cost", 0.0002)
        self.risk_reward = self.tparams.get("rr_ratio", 2.0)
        self.candle_window = self.mtrain.get("candle_window", 30)

        # === Preprocess raw OHLCV data into enriched + feature matrix ===
        self.feature_builder = FeatureBuilder(data)
        self.data_raw = self.feature_builder.prepare()
        self.feature_names = self.feature_builder.get_feature_columns()
        self.features = self.data_raw[self.feature_names].values.astype(np.float32)
        
        if not np.isfinite(self.features).all():
            raise ValueError("Feature matrix contains NaN or Inf. Please check preprocessing.")
        
        # === Core market series for tracking ===
        self.prices = self.data_raw["Close"].values
        self.atr = self.data_raw["ATR14"].values  # Use longer-term ATR for trailing stops       
        self.n = len(self.features) 

        feat_dim = self.features.shape[1]
        obs_shape = (self.candle_window * feat_dim,)
        super().__init__(obs_shape=obs_shape, action_space=3)

        # === Modular components ===
        self.normalizer = RunningNormalizer(np.prod(obs_shape))  # type: ignore
        self.obs_builder = ObservationBuilder(self)
        self.reward_engine = ZhangRewardEngine(self)
        self.risk_mgr = EnvRiskManager(self.config)
        self.trade_checker = TradeLogicEvaluator()
        self.regime_filter = RegimeFilter()
        self.trade_executor = TradeExecutor(self)
        self.metrics = EnvironmentMetrics(self)

        self._reset_all()

    def _reset_all(self):
        """
        Reset environment state at the beginning of an episode.
        """
        self.current_step = self.candle_window
        self.balance = self.tparams.get('initial_account_balance', 10000)
        self.position = 0
        self.prev_position = 0
        self.prev_prev_position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.peak_price = 0.0
        self.peak_balance = self.balance
        self.max_drawdown = 0.0
        self.return_buffer = []
        self.hold_duration = 0
        self.logger.info(f"[Reset] Balance = {self.balance}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and return the initial observation.

        Args:
            seed (Optional[int]): Random seed
            options (Optional[Dict[str, Any]]): Optional overrides (e.g. initial_balance)

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Normalized observation and info dict
        """
        if seed is not None:
            np.random.seed(seed)
        if options and 'initial_balance' in options:
            self.balance = options['initial_balance']

        self._reset_all()
        obs = self.obs_builder.get_obs()
        return obs, {'balance': self.balance}

    def step(self, raw_action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: # type: ignore
        """
        Take a step in the environment using an agent-supplied action.

        Args:
            raw_action (int): Action index from agent (0 = short, 1 = hold, 2 = long)

        Returns:
            Tuple: obs, reward, done, truncated, info
        """
        # Index boundary check for current_step
        if self.current_step >= self.n - 1:
            self.logger.info("Terminating episode: end of data at step %d", self.current_step)
            return self.obs_builder.get_obs(), 0.0, True, False, {"reason": "end_of_data"}
            
        action = raw_action - 1  # map [0, 1, 2] -> [-1, 0, 1]
        if self.hparams.get('regime_filtering', True):
            action = self.regime_filter.apply_filter(action, self.data_raw.iloc[self.current_step])

        price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1]
        base_reward = 0.0

        # === Exit logic ===
        if self.position != 0:
            self.trailing_stop, self.peak_price = self.risk_mgr.update_trailing_stop(
                self.position,
                self.entry_price,
                price, # type: ignore
                self.trailing_stop,
                self.peak_price,
                self.atr[self.current_step] # type: ignore
            )
            if self.risk_mgr.should_exit_trade(self.position, price, self.stop_loss, self.take_profit, self.trailing_stop): # type: ignore
                base_reward = self.trade_executor.exit_trade(price, tag="Exit") # type: ignore
                self.logger.info(f"[Exit] Trade exited at step {self.current_step} | price = {price:.5f} | PnL = {base_reward:.2f}")
            elif self.trade_checker.should_force_exit(self.position, self.hold_duration, self.max_hold_duration):
                base_reward = self.trade_executor.exit_trade(price, tag="ForcedExit") # type: ignore
                self.logger.warning(f"Forced exit at step {self.current_step} | hold_duration = {self.hold_duration}")
            else:
                self.hold_duration += 1

        # === Entry logic ===
        if self.trade_checker.should_enter(action, self.position):
            
            # Dynamically compute position size using risk model
            current_atr = self.atr[self.current_step]
            risk_pct = self.risk_mgr.risk_percentage
            position_size = risk_pct * self.balance / max(current_atr, 1e-6) # type: ignore
            
            if self.risk_mgr.can_enter_trade(price, self.balance, position_size): # type: ignore
                self.position_size = int(round(position_size))
                self.trade_executor.enter_trade(action, price) # type: ignore
                side = "LONG" if action > 0 else "SHORT"
                self.logger.info(f"Entered {side} trade | size = {self.position_size} | price = {price:.5f}")

        # === Reward calculation ===
        try:
            z_reward = self.reward_engine.compute(prev_price, price) # type: ignore
        except Exception as e:
            z_reward = 0.0
            self.logger.warning(f"Reward computation failed at step {self.current_step}: {str(e)}")
        
        total_reward = base_reward + z_reward
        self.metrics.update(base_reward)

        # === Step forward ===
        self.current_step += self.mtrain.get("time_skip", 1)
        done = self.current_step >= self.n
        truncated = self.current_step >= self.mtrain.get("max_steps", 1000)
        
        obs = None if done else self.obs_builder.get_obs()
        self.logger.info(f"[Step] Current step = {self.current_step}, Balance = {self.balance}")

        return obs, total_reward, done, truncated, { # type: ignore
            'balance': self.balance,
            'pnl': base_reward,
            'zhang_r': z_reward,
            'max_drawdown': self.max_drawdown
        }

    def update_features(self, selected_features: list[str]) -> None:
        """
        Dynamically update the environment's feature matrix with a new set of features.

        Args:
            selected_features (List[str]): List of column names to retain in self.data_raw
        """
        if any(col not in self.data_raw.columns for col in selected_features):
            raise ValueError("One or more selected features are missing from self.data_raw.")

        self.features = self.data_raw[selected_features].values.astype(np.float32)
        self.feature_names = selected_features

        # Update observation shape and components
        feat_dim = len(selected_features)
        self.obs_shape = (self.candle_window * feat_dim,)
        self.normalizer = RunningNormalizer(np.prod(self.obs_shape))  # type: ignore # reset with new shape
        self.obs_builder = ObservationBuilder(self)
        self.logger.info(f"[SimulatedTradingEnv] Updated to {feat_dim} features: {selected_features}")
