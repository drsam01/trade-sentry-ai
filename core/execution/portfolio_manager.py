# File: core/execution/portfolio_manager.py

from typing import Dict, List
from core.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """
    Monitors overall portfolio exposure across all strategies and symbols.
    Prevents overtrading or violating risk boundaries.
    """

    def __init__(self, config: Dict[str, float]):
        """
        Args:
            config: Configuration dict with keys:
                - max_total_exposure (float): Total lot size across all trades.
                - max_trades_per_symbol (int): Limit of concurrent trades per symbol.
                - max_trades_global (int): Max number of open trades.
        """
        self.max_trades_per_symbol = config.get("max_trades_per_symbol", 3)
        self.max_trades_global = config.get("max_trades_global", 10)
        self.max_exposure_per_symbol = config.get("max_exposure_per_symbol", 1.0)  # fallback per symbol
        self.max_total_exposure = config.get("max_total_exposure", 10.0)  # in lots
        self.total_budget = config.get("total_budget", 1000.0)

        self.symbol_budget_limits: Dict[str, float] = config.get("symbol_budgets", {}) # type: ignore
        self.open_positions: List[Dict] = []  # Active trades
        self.symbol_exposure: Dict[str, Dict[str, float]] = {}  # { "EURUSD": { "buy": 0.5, "sell": 0.3 } }

    def update_positions(self, positions: List[Dict]) -> None:
        """Update the internal list of active positions."""
        self.open_positions = positions
        logger.debug(f"Updated open positions: {len(positions)} positions tracked.")

    def total_exposure(self) -> float:
        """Returns the total lot exposure across all trades."""
        exposure = sum(p["volume"] for p in self.open_positions)
        logger.debug(f"Current total exposure: {exposure:.2f}")
        return exposure

    def count_trades_per_symbol(self, symbol: str) -> int:
        """Count active trades for a given symbol."""
        count = sum(p["symbol"] == symbol for p in self.open_positions)
        logger.debug(f"Open trades for {symbol}: {count}")
        return count

    def can_open_trade(self, symbol: str, volume: float) -> bool:
        """
        Checks whether a new trade can be opened under all exposure rules.
        Applies dynamic budget limits per symbol if configured.
        """
        total_exp = self.total_exposure()
        symbol_exp = sum(self.symbol_exposure.get(symbol, {}).values())

        symbol_budget = self.symbol_budget_limits.get(symbol, self.max_exposure_per_symbol)

        if total_exp + volume > self.max_total_exposure:
            logger.warning("Blocked trade: total exposure limit exceeded.")
            return False

        if self.count_trades_per_symbol(symbol) >= self.max_trades_per_symbol:
            logger.warning(f"Blocked trade: symbol limit exceeded for {symbol}.")
            return False

        if len(self.open_positions) >= self.max_trades_global:
            logger.warning("Blocked trade: global trade count exceeded.")
            return False

        if symbol_exp + volume > symbol_budget:
            logger.warning(f"Blocked trade: exposure limit exceeded for {symbol}.")
            return False

        return True

    def update_exposure(self, symbol: str, lot: float, direction: str) -> None:
        """
        Update the current exposure for a symbol when a new trade is executed.
        """
        if symbol not in self.symbol_exposure:
            self.symbol_exposure[symbol] = {"buy": 0.0, "sell": 0.0}
        self.symbol_exposure[symbol][direction] += lot

        logger.info(
            f"[Portfolio] Updated {direction.upper()} exposure for {symbol}: {self.symbol_exposure[symbol][direction]:.2f} lots"
        )

    def reduce_exposure(self, symbol: str, lot: float, direction: str) -> None:
        """
        Decrease exposure for a symbol and direction after trade close.
        """
        if symbol not in self.symbol_exposure:
            logger.warning(f"[Portfolio] Attempted to reduce exposure for unknown symbol: {symbol}")
            return

        current = self.symbol_exposure[symbol].get(direction, 0.0)
        new_exposure = max(0.0, current - lot)
        self.symbol_exposure[symbol][direction] = new_exposure

        logger.info(
            f"[Portfolio] Reduced {direction.upper()} exposure for {symbol}: {current:.2f} -> {new_exposure:.2f} lots"
        )

        if self.symbol_exposure[symbol]["buy"] == 0 and self.symbol_exposure[symbol]["sell"] == 0:
            del self.symbol_exposure[symbol]

    def set_budget_limit(self, symbol: str, new_limit: float) -> None:
        """
        Dynamically update budget cap for a symbol (used during portfolio rebalancing).
        """
        self.symbol_budget_limits[symbol] = new_limit
        logger.info(f"[Portfolio] Budget limit for {symbol} set to {new_limit:.2f} lots")

    def get_position_ids(self) -> List[int]:
        """Returns a list of all open order IDs (tickets)."""
        return [p["ticket"] for p in self.open_positions]
