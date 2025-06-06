# File: core/strategy/base_strategy.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class that all trading strategies must implement.

    Methods:
        load_data: Load and preprocess data required by the strategy.
        generate_signal: Produce a trading signal based on the current state.
        execute_trade: Define logic for executing trades.
        update_state: Update internal state with new market data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy with user-defined configuration.
        """
        self.config = config

    @abstractmethod
    def load_data(self, df: pd.DataFrame) -> None:
        """Load and preprocess historical or live data."""
        pass

    @abstractmethod
    def generate_signal(self, current_index: int) -> Optional[Dict[str, Any]]:
        """
        Generate a trading signal at a given index (e.g., long, short, hold).

        Returns:
            Dict with keys like 'direction', order type, 'confidence', 'timestamp', etc.
        """
        pass

    @abstractmethod
    async def log_performance(self, trade_tracker: Any, symbol: str) -> None:
        """
        Log or persist the strategy's performance.
        
        Args:
            trade_tracker: A TradeTracker instance with trade stats.
            symbol: Trading symbol for which to log performance.
        """
        pass
