# File: core/monitoring/metrics_collector.py

import os
import aiosqlite
import statistics
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

class MetricsCollector:
    """
    Collects, computes, and optionally persists key trading performance metrics.
    """

    def __init__(self, db_path: str = "logs/trades.db", persist: bool = False):
        self.trades: List[Dict[str, Any]] = []
        self.db_path = db_path
        self.persistence_enabled = persist
        if self.persistence_enabled:
            asyncio.create_task(self._init_db())

    async def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    order_type TEXT,
                    direction TEXT,
                    lot REAL,
                    entry_price REAL,
                    close_price REAL,
                    pnl REAL,
                    r_multiple REAL,
                    strategy TEXT,
                    tag TEXT,
                    timestamp TEXT
                )
            """)
            await db.commit()

    async def persist_trade(self, trade: Dict[str, Any]) -> None:
        """Persist a completed trade to the database."""
        if not self.persistence_enabled:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO closed_trades (
                    symbol, order_type, direction, lot, entry_price,
                    close_price, pnl, r_multiple, strategy, tag, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get("symbol"),
                trade.get("order_type"),
                trade.get("direction"),
                trade.get("lot"),
                trade.get("entry_price"),
                trade.get("close_price"),
                trade.get("pnl"),
                trade.get("r_multiple"),
                trade.get("strategy"),
                trade.get("tag"),
                trade.get("close_time", datetime.now(timezone.utc).isoformat())
            ))
            await db.commit()
    
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Add a completed trade to the metrics record."""
        self.trades.append(trade)

    def get_equity_curve(self) -> List[Dict]:
        curve = []
        equity = 0.0
        for i, t in enumerate(self.trades):
            equity += t["pnl"]
            curve.append({"step": i, "equity": equity})
        return curve
    
    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary metrics from recorded trades."""
        if not self.trades:
            return {}

        total_trades = len(self.trades)
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        r_multiples = [t.get("r_multiple", 0) for t in self.trades]
        pnl_series = [t["pnl"] for t in self.trades]

        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        win_rate = len(wins) / total_trades * 100
        avg_win = statistics.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = statistics.mean([t["pnl"] for t in losses]) if losses else 0
        expectancy = ((avg_win * len(wins)) + (avg_loss * len(losses))) / total_trades if total_trades else 0
        profit_factor = gross_win / gross_loss if gross_loss else float('inf')

        returns = [t["pnl"] for t in self.trades]
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0
        sharpe_ratio = (mean_return / std_return) if std_return else 0

        # Max drawdown
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in pnl_series:
            equity += t
            peak = max(peak, equity)
            dd = equity - peak
            max_dd = min(max_dd, dd)

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "avg_r_multiple": round(statistics.mean(r_multiples), 2) if r_multiples else 0,
            "expectancy": round(expectancy, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "win_loss_ratio": round(avg_win / abs(avg_loss), 2) if avg_loss else float('inf'),
            "cumulative_pnl": round(sum(pnl_series), 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round((max_dd / peak) * 100, 2) if peak else 0
        }

    def reset(self) -> None:
        """Clear all recorded trades."""
        self.trades.clear()
