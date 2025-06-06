# File: core/monitoring/trade_tracker.py

import os
import asyncio
import aiosqlite
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from core.monitoring.metrics_collector import MetricsCollector
from core.utils.logger import get_logger

logger = get_logger(__name__)

class TradeTracker:
    """
    Tracks current and historical trades during live or simulated sessions.
    """

    def __init__(self, persist: bool = False, db_path: str = "logs/trades.db"):
        self.open_trades: Dict[int, Dict] = {}
        self.metrics = MetricsCollector(db_path=db_path, persist=persist)

    def register_open_trade(self, trade_id: int, trade_data: Dict) -> None: # type: ignore
        """Add or update an open trade."""
        self.open_trades[trade_id] = trade_data # type: ignore

    async def close_trade(self, trade_id: int, close_price: float, timestamp: str) -> None:
        """
        Close and archive a trade, compute PnL and R-multiple, and optionally persist it.
        """
        trade = self.open_trades.pop(trade_id, None)
        if not trade:
            logger.warning(f"[TradeTracker] Tried to close unknown trade ID: {trade_id}")
            return

        entry_price = trade.get("entry_price")
        lot = trade.get("lot", 1.0)
        direction = trade.get("direction")

        if None in [entry_price, direction]:
            logger.warning(f"[TradeTracker] Incomplete trade data for trade ID: {trade_id}")
            return

        pnl = (close_price - entry_price) if direction == "buy" else (entry_price - close_price) # type: ignore
        pnl *= lot

        risk = trade.get("risk", abs(entry_price - trade.get("sl", entry_price)) * lot) # type: ignore
        risk = risk if risk != 0 else 1e-6

        trade.update({
            "close_price": close_price,
            "close_time": timestamp,
            "pnl": round(pnl, 2),
            "r_multiple": round(pnl / risk, 2),
        })

        self.metrics.record_trade(trade)
        await self.metrics.persist_trade(trade)
        
        logger.info(f"[TradeTracker] Closed trade {trade_id} | PnL: {pnl:.2f}, R: {pnl / risk:.2f}")

    def get_open_trades(self) -> List[Dict]: # type: ignore
        """Return all current open trades."""
        return list(self.open_trades.values()) # type: ignore

    def get_closed_summary(self, symbol: str) -> Dict[str, Any]:
        """Get performance summary for a symbol."""
        symbol_trades = [t for t in self.metrics.trades if t.get("symbol") == symbol]
        self.metrics.trades = symbol_trades  # temporarily filter
        summary = self.metrics.compute_summary()
        self.metrics.trades = symbol_trades  # restore
        return summary