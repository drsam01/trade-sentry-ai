# File: apps/smedge/orchestrator.py

import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, List

from core.broker.mt5 import MT5Broker
from core.execution.order_manager import OrderManager
from core.execution.risk_manager import RiskManager
from core.execution.portfolio_manager import PortfolioManager
from core.monitoring.trade_tracker import TradeTracker
from core.monitoring.trade_watcher import TradeWatcher
from core.utils.logger import get_logger
from core.utils.notifier import send_telegram_alert

from apps.smedge.strategy import SMEdgeStrategy

from datetime import timezone
logger = get_logger(__name__)


class SMEdgeLiveTrader:
    """
    Asynchronous live trading orchestrator for SMEdge (Smart Money Concepts) strategy.
    """

    def __init__(self, symbol: str, config: Dict): # type: ignore
        self.symbol = symbol
        self.config = config # type: ignore
        self.tf = config["timeframe"] # type: ignore
        self.lookback = config["lookback_bars"] # type: ignore

        self.broker = MT5Broker(config["broker"]) # type: ignore
        self.order_manager = OrderManager(self.broker)
        self.risk_manager = RiskManager(config["initial_balance"], config["risk"]) # type: ignore
        self.portfolio_manager = PortfolioManager(config["portfolio"]) # type: ignore
        self.trade_tracker = TradeTracker()
        self.trade_watcher = TradeWatcher(self.risk_manager.apply_trailing_stop)

        self.strategy = SMEdgeStrategy(config) # type: ignore

        self.current_position = 0
        self.entry_price = 0.0

    async def run(self):
        if not self.broker.connect():
            logger.error(f"[{self.symbol}] Could not connect to broker.")
            return

        logger.info(f"[{self.symbol}] SMEdge trader initialized.")

        while True:
            try:
                await self.process_tick()
                await asyncio.sleep(self.config.get("poll_interval", 5)) # type: ignore
            except Exception as e:
                logger.exception(f"[{self.symbol}] Runtime error: {e}")
                await asyncio.sleep(5)

    async def process_tick(self):
        """Polls new data, runs zone detection + signal logic, and manages trades."""
        bars = self.broker.get_ohlcv(self.symbol, self.tf, self.lookback) # type: ignore
        df = pd.DataFrame(bars)

        if df.empty or len(df) < self.lookback: # type: ignore
            logger.warning(f"[{self.symbol}] Skipping tick: insufficient data.")
            return

        signal = self.strategy.generate_signal(df)

        if not signal or signal["action"] == "hold":
            logger.debug(f"[{self.symbol}] No actionable SMC signal.")
            return

        action = signal["action"]
        price = df.iloc[-1]["close"] # type: ignore
        atr = signal.get("atr", 0.002)

        sl, tp = self.risk_manager.calculate_sl_tp(price, action, atr) # type: ignore
        lot = self.risk_manager.compute_lot_size(abs(price - sl) / self.risk_manager.pip_value, self.config["pip_value"]) # type: ignore

        if not self.portfolio_manager.can_open_trade(self.symbol, lot):
            logger.info(f"[{self.symbol}] Trade blocked by portfolio rules.")
            return

        order_result = self.order_manager.open_position(
            symbol=self.symbol,
            action=action,
            volume=lot,
            sl=sl,
            tp=tp,
            comment="SMEdge"
        )

        if trade_id := order_result.get("order", 0):
            trade_data = { # type: ignore
                "symbol": self.symbol,
                "direction": action,
                "entry_price": price,
                "volume": lot,
                "sl": sl,
                "tp": tp,
                "risk": abs(price - sl) * lot, # type: ignore
                "entry_time": datetime.now(timezone.utc).isoformat(),
            }
            self.trade_tracker.register_open_trade(trade_id, trade_data) # type: ignore
            send_telegram_alert(f"📌 SMEdge {action.upper()} on {self.symbol} @ {price:.5f}")

        await self.monitor_trades(price, action) # type: ignore

    async def monitor_trades(self, price: float, direction: str):
        """Evaluates SL/TP or trailing exit for open trades."""
        open_trades = self.trade_tracker.get_open_trades() # type: ignore
        to_close = self.trade_watcher.evaluate_trades(open_trades, price, direction) # type: ignore

        for trade in to_close: # type: ignore
            if success := self.order_manager.close_position(trade["ticket"]): # type: ignore
                self.trade_tracker.close_trade(
                    trade["ticket"], price, datetime.now(timezone.utc).isoformat() # type: ignore
                )
                send_telegram_alert(f"📉 SMEdge exit @ {price:.5f} on {self.symbol}")

        self.portfolio_manager.update_positions(self.broker.get_open_orders()) # type: ignore


class SMEdgeAsyncOrchestrator:
    """
    Manages multiple SMEdge trader instances concurrently.
    """

    def __init__(self, trading_plan: List[Dict], config: Dict): # type: ignore
        self.trading_plan = trading_plan # type: ignore
        self.config = config # type: ignore

    async def launch_all(self):
        tasks = []
        for plan in self.trading_plan: # type: ignore
            bot = SMEdgeLiveTrader(symbol=plan["symbol"], config=self.config) # type: ignore
            tasks.append(asyncio.create_task(bot.run())) # type: ignore

        logger.info(f"🚀 Launching {len(tasks)} SMEdge bots...") # type: ignore
        await asyncio.gather(*tasks) # type: ignore
