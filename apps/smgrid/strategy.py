# File: apps/smgrid/strategy.py

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from core.strategy.base_strategy import BaseStrategy
from apps.smgrid.helpers import SMGridHelper
from core.utils.logger import get_logger

logger = get_logger(__name__)


class SMGridStrategy(BaseStrategy):
    """
    SMGrid Strategy:
    - Dynamically switches between grid trading during consolidation
    and directional trend breakout during trending regimes.
    - Resets grid on new consolidation zones or scheduled intervals
    - Avoids re-placing redundant grid orders.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SMGrid strategy.

        Args:
            config (Dict): Configuration dictionary with strategy, risk, and broker details.
        """
        super().__init__(config)
        strategy_config = config.get("strategy", {})
        
        self.helper = SMGridHelper(
            atr_period=strategy_config.get("atr_period", 14),
            spacing_multiplier=strategy_config.get("atr_multiplier", 1.0),
        )
        
        self.grid_count: int = strategy_config.get("grid_count", 3)
        self.reset_cycle: str = strategy_config.get("reset_cycle", "daily")
        self.grid_levels: List[float] = []
        self.current_regime: str = "consolidation"
        self.last_midpoint: Optional[float] = None
        self.atr: Optional[float] = None
        self.df: Optional[pd.DataFrame] = None
        self.last_trade: Optional[Dict[str, Any]] = None
        self.trend_active: bool = False
        self.trend_entry_price: Optional[float] = None
        self.trend_direction: Optional[str] = None

    def load_data(self, df: pd.DataFrame) -> None:
        """Load historical data and initialize state."""
        self.df = df.copy()
        self.atr = self.helper.calculate_atr(self.df)
        self.current_regime = self.helper.detect_regime(self.df)
        
        new_midpoint = self.helper.get_midpoint_of_zone()
        if self.current_regime == "consolidation" and (
            not self.grid_levels or new_midpoint != self.last_midpoint
        ):
            self.grid_levels = self.helper.get_grid_levels(new_midpoint, self.atr, count=self.grid_count) # type: ignore
            self.last_midpoint = new_midpoint

    def generate_signal(self, current_index: int, injected_lot: Optional[float]) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on current market regime.

        Args:
            current_index (int): Latest row index in the dataframe.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing signal mode and order list.
        """
        if self.df.empty or current_index < 1: # type: ignore
            return None

        row = self.df.iloc[current_index] # type: ignore
        price = row["close"]

        if self.current_regime == "consolidation":
            
            grid_orders = []
            for i, level in enumerate(self.grid_levels):
                if self.atr is not None and abs(price - level) <= self.atr * 0.05:
                    direction = "buy" if self.last_midpoint is not None and level < self.last_midpoint else "sell"
                    tp = self._get_next_grid_tp(level, direction)
                    
                    # Compute SL: below for buy, above for sell
                    sl = None
                    if direction == "buy":
                        sl = level - self.atr * self.config.get("risk", {}).get("atr_sl_multiplier", 1.5)
                    elif direction == "sell":
                        sl = level + self.atr * self.config.get("risk", {}).get("atr_sl_multiplier", 1.5)
                    
                    lot = injected_lot or self.config.get("risk", {}).get("default_lot", 0.01)
                    
                    grid_orders.append({
                        "type": "limit",
                        "direction": direction,
                        "lot": lot,
                        "price": level,
                        "tp": tp,
                        "sl": sl,
                        "tag": f"grid_{i}",
                    })
            return {
                "mode": "grid",
                "orders": grid_orders
            }

        elif self.current_regime == "trending":
            if direction := self.helper.detect_breakout_from_zone(
                self.df[:current_index] # type: ignore
            ):
                sl, tp = self._get_sl_tp(price, direction)
                lot = injected_lot or self.config.get("risk", {}).get("default_lot", 0.01)
                return {
                    "mode": "trend",
                    "orders": {
                        "type": "market",
                        "direction": direction,
                        "lot": lot,
                        "price": price,
                        "tp": tp,
                        "sl": sl,
                        "tag": "trend_breakout"                        
                    }
                }

        return None

    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate signals for all rows in the dataset for backtesting purposes.

        Returns:
            List of signal packages per row in historical dataset.
        """
        signals = []
        for idx in range(len(self.df)): # type: ignore
            signal = self.generate_signal(current_index=idx, injected_lot=0.01)
            signals.append(signal or {})
        return signals

    def execute_trade(self, signal: Dict[str, Any]) -> None:
        """
        Record or process executed trade.

        - For grid trades: remove the executed grid level from active grid levels.
        - For trend trades: mark the trend as active or perform strategy-specific updates.

        Args:
            signal (Dict[str, Any]): Trade signal dictionary containing at least 'tag' and 'price'.
        """
        self.last_trade = signal
        tag = signal.get("tag", "")
        price = signal.get("price")

        if tag.startswith("grid_"):
            # Grid trade: remove filled level from grid
            self.grid_levels = [lvl for lvl in self.grid_levels if lvl != price]
            logger.info(f"[SMGrid] Executed grid trade at level {price}. Remaining grid levels: {self.grid_levels}")

        elif tag == "trend_breakout" or tag.startswith("trend_"):
            self._trend_trade_execution(price, signal)

    async def log_performance(self, trade_tracker, symbol: str) -> None:
        """
        Log current strategy performance metrics.
        
        Args:
            trade_tracker: The TradeTracker instance
            symbol: Symbol for which performance is to be logged
        """
        summary = trade_tracker.get_closed_summary(symbol)
        logger.info(f"[{symbol}] Strategy Performance: {summary}")

    def reset_grid_parameters(self) -> None:
        """
        Reset internal grid levels and any state required to reinitialize the strategy.
        Useful during scheduled resets (e.g., daily or weekly).
        """
        self.grid_levels = []
        self.last_trade = None
        self.trend_active = False
        self.trend_entry_price = None
        self.trend_direction = None
        logger.info("[SMGridStrategy] Grid parameters reset.")

    def _trend_trade_execution(self, price, signal):
        # Trend trade: flag trend state or reset grid as needed
        self.trend_active = True
        self.trend_entry_price = price
        self.trend_direction = signal.get("direction")
        logger.info(f"[SMGrid] Executed trend breakout trade at {price} in direction {self.trend_direction}")

        # Optional: clear grid levels upon breakout
        self.grid_levels = []
        logger.debug("[SMGrid] Cleared grid levels after trend entry.")

    def _get_sl_tp(self, price: float, direction: str) -> Tuple[float, float]:
        """
        Calculate SL and TP using ATR-based logic with optional risk-reward ratio.
        
        Args:
            price (float): Entry price.
            direction (str): 'buy' or 'sell'.
        
        Returns:
            Tuple[float, float]: (SL, TP) price levels.
        """
        atr = self.atr or 0.001
        risk_multiplier = self.config.get("risk", {}).get("atr_sl_multiplier", 1.5)
        reward_multiplier = self.config.get("risk", {}).get("rr_ratio", 2.0)

        sl_offset = atr * risk_multiplier
        tp_offset = atr * risk_multiplier * reward_multiplier

        if direction == "buy":
            sl = price - sl_offset
            tp = price + tp_offset
        else:  # sell
            sl = price + sl_offset
            tp = price - tp_offset

        return round(sl, 5), round(tp, 5)


    def _get_next_grid_tp(self, price: float, direction: str) -> Optional[float]:
        """
        Determines the TP level based on grid logic:
        - For BUY: next level above current price.
        - For SELL: next level below current price.

        Args:
            price (float): The price of the current grid level.
            direction (str): "buy" or "sell"

        Returns:
            float: Next grid TP level, or None if at edge of grid.
        """
        if not self.grid_levels:
            return None

        sorted_levels = sorted(self.grid_levels)
        
        if direction == "buy":
            for level in sorted_levels:
                if level > price:
                    return level
        elif direction == "sell":
            for level in reversed(sorted_levels):
                if level < price:
                    return level

        # If no next level in direction (edge case)
        return None

