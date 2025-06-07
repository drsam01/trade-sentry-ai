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
    
    def generate_signal(self, current_index: int) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on current market regime.

        Args:
            current_index (int): Latest row index in the dataframe.

        Returns:
            Optional[Dict[str, Any]]: Dictionary with 'mode' and order(s) or None.
        """
        if self.df.empty or current_index < 1:  # type: ignore
            return None

        row = self.df.iloc[current_index]  # type: ignore
        required_fields = ["close"]
        if any(field not in row for field in required_fields):
            logger.warning("Missing required fields in data row.")
            return None

        price = row["close"]

        if self.current_regime == "consolidation":
            return self._generate_consolidation_signals()

        elif self.current_regime == "trending":
            return self._generate_trend_breakout_signal(current_index, price)

        logger.warning(f"Unrecognized regime: {self.current_regime}")
        return None

    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate signals for all rows in the dataset for backtesting purposes.

        Returns:
            List of signal packages per row in historical dataset.
        """
        signals = []
        for idx in range(len(self.df)): # type: ignore
            signal = self.generate_signal(current_index=idx)
            signals.append(signal or {})
        return signals
    
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

    def _generate_consolidation_signals(self) -> Dict[str, Any]:
        """
        Generate limit orders for consolidation regime based on grid levels.
        """
        if self.last_midpoint is None or not self.grid_levels:
            logger.warning("Cannot generate consolidation signals: Missing midpoint or grid levels.")
            return {"mode": "grid", "orders": []}

        spacing = getattr(self.helper, "spacing", None)
        if spacing is None:
            logger.warning("Spacing attribute not found in helper.")
            spacing = 0.0010  # fallback or default

        orders = []
        for i, level in enumerate(self.grid_levels):
            direction = "buy" if level < self.last_midpoint else "sell"
            tp = self._get_next_grid_tp(level, direction)

            orders.append({
                "type": "limit",
                "direction": direction,
                "price": level,
                "tp": tp,
                "sl": None,  # SL may be applied later by risk or trailing logic
                "spacing": spacing,
                "tag": f"grid_{i}",
            })

        return {"mode": "grid", "orders": orders}

    def _generate_trend_breakout_signal(self, current_index: int, price: float) -> Optional[Dict[str, Any]]:
        """
        Generate a market order signal if a trend breakout is detected.
        """
        direction = self.helper.detect_breakout_from_zone(self.df[:current_index])  # type: ignore
        if not direction:
            logger.info("No breakout direction detected.")
            return None

        sl, tp = self._get_sl_tp(price, direction)

        return {
            "mode": "trend",
            "orders": {
                "type": "market",
                "direction": direction,
                "price": price,
                "tp": tp,
                "sl": sl,
                "spacing": abs(price - sl),
                "tag": "trend_breakout"
            }
        }

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
        risk_multiplier = self.config.get("risk", {}).get("sl_multiplier", 1.5)
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
        Determines the next take profit (TP) level based on grid logic:
        - For BUY: returns the next level above the current price.
        - For SELL: returns the next level below the current price.

        Args:
            price (float): The current price level.
            direction (str): Trade direction ("buy" or "sell").

        Returns:
            Optional[float]: The next TP level, or None if at the edge.
        """
        if not self.grid_levels:
            return None

        levels = sorted(self.grid_levels)
        
        if direction.lower() == "buy":
            return next((lvl for lvl in levels if lvl > price), None)
        
        elif direction.lower() == "sell":
            return next((lvl for lvl in reversed(levels) if lvl < price), None)

        # If no next level in direction (edge case)
        return None

