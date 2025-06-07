# File: apps/smgrid/helpers.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal, Tuple
from datetime import datetime
from core.utils.indicators import compute_atr


class SMGridHelper:
    """
    Encapsulates all helper methods for SMGrid strategy:
    - ATR & regime detection
    - Grid level calculation
    - Breakout detection
    - Midpoint estimation
    - Reset cycle tracking
    """

    def __init__(self, atr_period: int = 14, spacing_multiplier: float = 1.5) -> None:
        self.atr_period = atr_period
        self.spacing_multiplier = spacing_multiplier
        self.last_reset_date: Optional[datetime] = None
        self.cached_zone: Optional[Dict[str, float]] = None  # Stores last consolidation zone
        
        self.last_zone_fingerprint: Optional[Tuple[float, float]] = None
        self.active_grid_levels: List[float] = []
        self.trade_closed_since_last_grid: bool = False

    def calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR from recent candles."""
        return compute_atr(df, period=self.atr_period)
    
    def detect_regime(self, df: pd.DataFrame) -> Literal["consolidation", "trending"]:
        """
        Determine if the market is in consolidation or trending.
        Uses high-low range compression as proxy.
        
        Args:
            df (pd.DataFrame): OHLCV data

        Returns:
            Literal["consolidation", "trending"]
        """
        if df is None or len(df) < self.atr_period:
            return "consolidation"

        atr = self.calculate_atr(df)
        recent_high = df["high"].iloc[-self.atr_period:].max()
        recent_low = df["low"].iloc[-self.atr_period:].min()
        range_span = recent_high - recent_low

        # If range is < 2x ATR, likely consolidating
        return "consolidation" if range_span <= atr * 2 else "trending" # CONSIDER using a tunable multiplier in place of 2

    def detect_breakout_from_zone(self, df: pd.DataFrame) -> Optional[str]:
        """
        Check if price broke above or below the last known consolidation zone.
        
        Args:
            df (pd.DataFrame): OHLCV data

        Returns:
            Optional[str]: "buy", "sell", or None
        """
        if not self.cached_zone:
            return None

        recent_close = df["close"].iloc[-1]
        upper = self.cached_zone["upper"]
        lower = self.cached_zone["lower"]

        if recent_close > upper:
            return "buy"
        elif recent_close < lower:
            return "sell"
        return None

    def get_midpoint_of_zone(self) -> Optional[float]:
        """
        Compute midpoint of the last known consolidation zone.
        
        Returns:
            Optional[float]: Midpoint value
        """
        if not self.cached_zone:
            return None
        return round((self.cached_zone["upper"] + self.cached_zone["lower"]) / 2, 5)

    def get_grid_levels(self, midpoint: float, atr: float, count: int = 3) -> List[float]:
        """
        Compute grid levels spaced by ATR around a midpoint.

        Args:
            midpoint (float): Base price around which to place grid
            atr (float): ATR value for spacing
            count (int): Number of levels in each direction

        Returns:
            List[float]: Sorted list of levels
        """
        self.spacing = atr * self.spacing_multiplier
        levels = []

        for i in range(1, count + 1):
            levels.extend(
                (
                    round(midpoint - self.spacing * i, 5), # Buy below
                    round(midpoint + self.spacing * i, 5), # Sell above
                )
            )
        return sorted(levels)

    def update_consolidation_zone(self, df: pd.DataFrame) -> None:
        """
        Capture a new consolidation zone - Refresh cached consolidation zone using latest ATR range.

        Args:
            df (pd.DataFrame): OHLCV data
        """
        high = df["high"].iloc[-self.atr_period:].max()
        low = df["low"].iloc[-self.atr_period:].min()
        self.cached_zone = {"upper": high, "lower": low}

    def should_reset(self, now: datetime, cycle: str = "daily") -> bool:
        """
        Determine whether a scheduled grid reset is due.

        Args:
            now (datetime): Current timestamp
            cycle (str): "daily" or "weekly"

        Returns:
            bool: True if reset is due
        """
        if not self.last_reset_date:
            self.last_reset_date = now
            return True

        if cycle == "daily" and now.date() != self.last_reset_date.date():
            self.last_reset_date = now
            return True

        if cycle == "weekly" and now.isocalendar().week != self.last_reset_date.isocalendar().week:
            self.last_reset_date = now
            return True

        return False

    def compute_zone_fingerprint(self) -> Optional[Tuple[float, float]]:
            """
            Compute a simple fingerprint tuple of the cached zone.
            """
            if not self.cached_zone:
                return None
            return (self.cached_zone["upper"], self.cached_zone["lower"])

    def should_regrid(self) -> bool:
        """
        Determine whether to place new grid orders based on:
        - Zone change
        - Trade closed since last placement
        """
        current_fp = self.compute_zone_fingerprint()

        if current_fp != self.last_zone_fingerprint:
            self.last_zone_fingerprint = current_fp
            self.trade_closed_since_last_grid = False
            return True

        if self.trade_closed_since_last_grid:
            self.trade_closed_since_last_grid = False
            return True

        return False

    def mark_trade_closed(self) -> None:
        """
        Signal that a trade from grid was closed and may require regrid.
        """
        self.trade_closed_since_last_grid = True

    def update_active_grid_levels(self, levels: List[float]) -> None:
        """
        Store active grid level prices.
        """
        self.active_grid_levels = levels
