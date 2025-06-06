# File: apps/drlfusion/strategy.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from core.strategy.base_strategy import BaseStrategy
from apps.drlfusion.envs.regime_filter import RegimeFilter
from apps.drlfusion.helpers import DRLFusionHelper
from core.utils.logger import get_logger

logger = get_logger(__name__)


class DRLFusionStrategy(BaseStrategy):
    """
    DRLFusion Strategy:
    - Ensemble voting using PPO, A2C, DQN, SAC
    - Confidence threshold and regime filtering
    - DRLFusionHelper handles feature prep and inference
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.df: Optional[pd.DataFrame] = None
        self.helper = DRLFusionHelper(config)
        self.regime_filter = RegimeFilter()

        self.conf_threshold: float = config.get("ensemble_conf_threshold", 0.5)
        self.last_signal: Optional[Dict[str, Any]] = None

        self.trend_active = False
        self.trend_direction: Optional[str] = None
        self.trend_entry_price: Optional[float] = None

    def load_data(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.helper.prepare(df)

    def generate_signal(self, current_index: int) -> Optional[Dict[str, Any]]:
        if self.df is None:
            return None

        state = self.helper.get_window_features(current_index)
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            return None

        action, _, _, probs = self.helper.infer_action(state, deterministic=True)
        confidence = float(probs[action])

        if confidence < self.conf_threshold:
            logger.info(f"[DRLFusion] Low confidence: {confidence:.2f}")
            return None

        intended_action = action - 1  # [0,1,2] → [-1,0,+1]
        filtered_action = self.regime_filter.apply_filter(intended_action, self.df.iloc[current_index])
        if filtered_action == 0:
            logger.info("[DRLFusion] Regime filter blocked trade.")
            return None

        direction = "long" if filtered_action > 0 else "short"
        price = self.df["Close"].iloc[current_index]
        atr = self.df["ATR14"].iloc[current_index]
        sl_mult = self.config.get("trading_params", {}).get("sl_atr_multiplier", 2.0)
        rr_mult = self.config.get("trading_params", {}).get("rr_ratio", 2.0)

        sl = price - sl_mult * atr if direction == "long" else price + sl_mult * atr
        tp = price + rr_mult * sl_mult * atr if direction == "long" else price - rr_mult * sl_mult * atr

        signal = {
            "type": "market",
            "direction": direction,
            "price": price,
            "sl": sl,
            "tp": tp,
            "confidence": confidence,
            "tag": "drlfusion_ensemble"
        }

        self.last_signal = signal
        return signal

    def generate_signals(self) -> list[Dict[str, Any]]:
        if self.df is None:
            logger.warning("[DRLFusion] Data not loaded")
            return []

        signals = []
        for idx in range(self.helper.candle_window, len(self.df)):
            if signal := self.generate_signal(idx):
                signal["index"] = idx
                signals.append(signal)

        logger.info(f"[DRLFusion] Generated {len(signals)} signals.")
        return signals

    def execute_trade(self, signal: Dict[str, Any]) -> None:
        self.last_signal = signal

        tag = signal.get("tag", "")
        direction = signal.get("direction", "")
        price = signal.get("price")
        sl = signal.get("sl")
        tp = signal.get("tp")
        confidence = signal.get("confidence")

        logger.info(
            f"[DRLFusion] Executed {tag.upper()} trade | "
            f"Direction: {direction}, Entry: {price}, SL: {sl}, TP: {tp}, Confidence: {confidence:.2f}"
        )

        self.trend_active = True
        self.trend_direction = direction
        self.trend_entry_price = price

        logger.debug(f"[DRLFusion] Trend state updated: {self.trend_direction} @ {self.trend_entry_price}")

    async def log_performance(self, trade_tracker: Any, symbol: str) -> None:
        summary = trade_tracker.get_closed_summary(symbol)
        logger.info(f"[{symbol}] DRLFusion Performance: {summary}")

    def reset_state(self) -> None:
        self.last_signal = None
        self.trend_active = False
        self.trend_direction = None
        self.trend_entry_price = None
        logger.info("[DRLFusion] Strategy state reset.")
