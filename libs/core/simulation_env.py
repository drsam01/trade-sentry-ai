# === simulation_env.py ===
# File: drlfusion/modelling/ens/simulation_env.py

"""
SimulatedTradingEnv: Gym-compatible trading environment using historical OHLCV data.
Supports regime filtering, Zhang-style rewards, risk-managed trades, and normalized features.
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces

from drlfusion.modelling.envs.simulation_env_helpers import SimEnvLogic
from drlfusion.modelling.features.data_preprocessing import RunningNormalizer, DataPreprocessor
from drlfusion.core.utils.regime_filter import RegimeFilter
from drlfusion.core.trading.risk_manager import RiskManager, TradeConditionChecker
from drlfusion.core.brokers.broker_base import BrokerBase
from drlfusion.configs.config import Config
from drlfusion.core.utils.logger import get_logger


class InvalidMarketDataException(Exception):
    """Raised when required market data columns are missing."""
    pass


class DummyBroker(BrokerBase):
    """Dummy broker for simulation purposes."""
    def connect(self): pass
    def shutdown(self): pass
    def fetch_account_info(self): return {}
    def fetch_symbol_info(self, symbol): return {}
    def fetch_open_positions(self): return []
    def place_order(self, *args, **kwargs): return {"order_id": "test"}
    def close_position(self, *args, **kwargs): return {"status": "closed"}
    def fetch_latest_candle(self, *args, **kwargs): return {"close": 1.0}
    def fetch_historical_data(self, *args, **kwargs): return []


class SimulatedTradingEnv(gym.Env):
    """
    Main trading environment supporting backtesting-like simulation for RL agents.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, data: pd.DataFrame, config: Config, render_mode: Optional[str] = None):
        """
        Initialize environment.

        Args:
            data (pd.DataFrame): Historical market data.
            config (Config): System configuration object.
            render_mode (str, optional): Rendering mode (unused).
        """
        super().__init__()

        self.data_raw = data.copy().reset_index(drop=True)
        self.config = config

        # Validate required columns
        required_cols = ['Close', 'SwingHigh', 'SwingLow', 'ATR', 'ATRRatio']
        if any(col not in self.data_raw.columns for col in required_cols):
            raise InvalidMarketDataException(f"Missing required columns: {required_cols}")

        # Extract useful series
        self.prices = self.data_raw['Close'].values
        self.atr = self.data_raw['ATR'].values
        self.n = len(self.data_raw)

        # Config params
        self.hparams = config.GLOBAL_HYPERPARAMS
        self.tparams = config.TRADING_PARAMS
        self.mt = config.MODEL_TRAINING_CONFIG

        self.transaction_cost = self.tparams.get('TRANSACTION_COST', 0.0002)
        self.risk_reward = self.hparams.get('RISK_REWARD_RATIO', 1.5)
        self.time_skip = self.mt.get('TIME_SKIP', 1)
        self.max_steps = self.mt.get('MAX_STEPS', 1000)
        self.sigma_target = self.hparams.get('VOLATILITY_TARGET', 0.02)
        self.mu = self.hparams.get('REWARD_SCALE', 1.0)
        self.vol_window = self.mt.get('VOLATILITY_WINDOW', 60)

        # Preprocess features
        dp = DataPreprocessor()
        self.features = dp.create_feature_matrix(self.data_raw, dp.selected_feature_columns)
        self.candle_window = config.CANDLE_WINDOW
        self.n_timesteps = len(self.features)

        feat_dim = self.features.shape[1]
        self.obs_shape = (self.candle_window * feat_dim,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # short, hold, long

        # Components
        self.normalizer = RunningNormalizer(size=np.prod(self.obs_shape))
        self.regime_filter = RegimeFilter()
        self.risk_mgr = RiskManager(DummyBroker(), config)
        self.trade_checker = TradeConditionChecker()
        self.logger = get_logger('drlfusion_env')
        self.env_logic = SimEnvLogic(self)

        self._reset_all()
        self.logger.info("SimulatedTradingEnv initialized.")

    def _reset_all(self) -> None:
        """Reset environment state at the beginning of an episode."""
        self.current_step = self.candle_window
        self.balance = self.tparams.get('INITIAL_ACCOUNT_BALANCE', 10_000)
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

    def reset(self, seed: int = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Gym reset method.

        Returns:
            observation (np.ndarray), info (dict)
        """
        if seed is not None:
            np.random.seed(seed)
        if options and 'initial_balance' in options:
            self.balance = options['initial_balance']

        self._reset_all()
        return self.env_logic.get_obs(), {'balance': self.balance}

    def step(self, raw_action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step forward in environment.

        Args:
            raw_action (int): Action from agent (0=short, 1=hold, 2=long)

        Returns:
            Tuple of (obs, reward, done, truncated, info)
        """
        action = raw_action - 1  # map to {-1, 0, 1}

        if self.hparams.get('REGIME_FILTERING', True):
            action = self.regime_filter.apply_filter(action, self.data_raw.iloc[self.current_step])

        price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1]
        base_reward = 0.0

        # Check for trade exit
        if self.position != 0:
            self.trailing_stop, self.peak_price = self.risk_mgr._update_trailing_stop(
                self.position, self.entry_price, price, self.trailing_stop, self.peak_price, self.atr[self.current_step]
            )
            if self.risk_mgr._should_exit_trade(self.position, price, self.stop_loss, self.take_profit, self.trailing_stop):
                base_reward = self.env_logic.exit_trade(price, 'Exit')
            elif self.trade_checker.should_force_exit(
                self.position, self.hold_duration, self.hparams.get('MAX_HOLD_DURATION', 50)):
                base_reward = self.env_logic.exit_trade(price, 'ForcedExit')
            else:
                self.hold_duration += 1

        # Check for trade entry
        if self.trade_checker.should_enter(action, self.position):
            self.env_logic.enter_trade(action, price)

        # Compute reward
        z_reward = self.env_logic.zhang_reward(prev_price, price)
        total_reward = base_reward + z_reward
        self.env_logic.update_metrics(base_reward)

        self.current_step += self.time_skip
        done = self.current_step >= self.n
        truncated = self.current_step >= self.max_steps
        obs = None if done else self.env_logic.get_obs()

        return obs, total_reward, done, truncated, {
            'balance': self.balance,
            'pnl': base_reward,
            'zhang_r': z_reward,
            'max_drawdown': self.max_drawdown
        }

    def render(self, mode: str = 'human') -> None:
        """Log environment state (used during debug)."""
        self.logger.info(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Pos: {self.position}")

if __name__=='__main__':
    from drlfusion.configs.config import Config
    symbol = 'EURUSD'
    timeframe = 'M15'
    filename = f"{symbol.lower()}_{timeframe.lower()}_train.csv"
    df = pd.read_csv(os.path.join(Config().DATA_DIR, "trainsets", symbol, filename), parse_dates=['Date'])
    env = SimulatedTradingEnv(df, Config())
    obs,_ = env.reset(seed=42)
    done=False
    tot=0.0
    while not done:
        a=env.action_space.sample()
        obs,r,done,tr,info=env.step(a)
        tot+=r
    print(f"Total reward: {tot:.2f}, Final balance: {env.balance:.2f}")