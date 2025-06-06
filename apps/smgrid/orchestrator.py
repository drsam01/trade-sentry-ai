# File: apps/smgrid/orchestrator.py

import asyncio
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.broker.base_broker import BaseBroker
from core.execution.order_manager import OrderManager
from core.execution.risk_manager import RiskManager
from core.execution.portfolio_manager import PortfolioManager
from core.monitoring.trade_tracker import TradeTracker
from core.monitoring.trade_watcher import TradeWatcher
from core.utils.logger import get_logger
from core.utils.notifier import send_telegram_alert

from apps.smgrid.strategy import SMGridStrategy

logger = get_logger(__name__)

class SMGridTrader:
    """
    Trader class for SMGrid Strategy supporting regime-based grid execution.
    """

    def __init__(self, symbol: str, broker: BaseBroker, config: Dict[str, Any]) -> None:
        self.symbol = symbol
        self.broker = broker
        self.config = config
        self.timeframe = config.get("timeframe", "M5")
        self.lookback = config.get("lookback_bars", 100)

        # Fetch initial balance dynamically from broker account info
        account_info = self.broker.get_account_info()
        initial_balance = account_info.get("balance", 10000.0)  # fallback if unavailable
        logger.info(f"[{self.symbol}] Initial broker balance: {initial_balance}")
        
        # Components
        self.order_manager = OrderManager(self.broker)
        self.risk_manager = RiskManager(balance=initial_balance, config=config["risk"])
        self.portfolio_manager = PortfolioManager(config["portfolio"])
        self.trade_tracker = TradeTracker()
        self.trade_watcher = TradeWatcher(self.risk_manager.apply_trailing_stop)
        
        # Strategy
        self.strategy = SMGridStrategy(config)

    async def run(self) -> None:
        """Main execution loop - Periodically poll price data, evaluate grid regime, and execute accordingly."""
        if not self.broker.connect():
            logger.error(f"[{self.symbol}] Broker connection failed.")
            return

        logger.info(f"[{self.symbol}] SMGrid trader started.")
        
        while True:
            try:
                await self.process_tick()
            except Exception as e:
                logger.exception(f"[{self.symbol}] Error in grid bot loop: {e}")
            await asyncio.sleep(self.config.get("poll_interval", 10))

    async def process_tick(self) -> None:
        """Handle each tick - signal generation and order management."""
        if not self.broker.is_connected():
            logger.warning(f"[{self.symbol}] Broker disconnected. Reconnecting...")
            if not self.broker.connect():
                logger.error(f"[{self.symbol}] Reconnection failed.")
                return
        
        bars = self.broker.fetch_ohlcv(self.symbol, self.timeframe, self.lookback)
        df = pd.DataFrame(bars)

        if df.empty or len(df) < self.lookback:
            logger.warning(f"[{self.symbol}] Insufficient data.")
            return
        
        # Load data into strategy and generate signal package
        self.strategy.load_data(df)
        
        # Use RiskManager to compute dynamic lot size
        atr = self.strategy.atr or 0.001  # fallback ATR
        stop_loss_pips = atr / self.config["risk"].get("pip_value", 0.0001)
        
        lot = self.risk_manager.compute_lot_size(
            stop_loss_pips=stop_loss_pips,
            pip_value=self.config["risk"].get("pip_value", 0.0001),
            max_lot=self.config["risk"].get("max_lot", 1.0)
        )
        
        signal_package = self.strategy.generate_signal(current_index=len(df) - 1, injected_lot=lot)

        if not signal_package:
            logger.debug(f"[{self.symbol}] No valid signal.")
            return
        
        mode = signal_package.get("mode", "unknown")
        logger.info(f"[{self.symbol}] Strategy mode: {mode.upper()}")

        if mode == "trend":
            # Cancel existing pending orders before placing new ones
            await self.order_manager.cancel_all_pending_orders(self.symbol)
            self._handle_trend_mode(signal_package.get("orders", {}))
        elif mode == "grid":
            self._handle_grid_mode(signal_package)
        else:
            logger.warning(f"[{self.symbol}] Unrecognized mode in signal package: {mode}")

        orders = signal_package.get("orders", [])
        direction = orders[0].get("action") if orders else signal_package.get("orders",{})["direction"]
        current_price = df["close"].iloc[-1]
        await self.monitor_trades(current_price, direction)

    def _handle_trend_mode(self, signal: Dict[str, Any]) -> None:
        """
        Execute market trade for trend breakout mode.

        Args:
            signal (Dict[str, Any]): Signal dict with direction, lot, sl, tp.
        """        
        direction = signal.get("direction","")
        lot = signal.get("lot", 0.01)
        sl = signal.get("sl")
        tp = signal.get("tp")
        price = signal.get("price")
        
        if None in [direction, lot, sl, tp]:
            logger.warning(f"[{self.symbol}] Incomplete signal: {signal}")
            return

        if self.portfolio_manager.can_open_trade(self.symbol, lot):
            result = self.order_manager.open_market_order(
                symbol=self.symbol,
                direction=direction,
                lot=lot,
                sl=sl,
                tp=tp,
                comment="SMGrid Trend Breakout"
            )
            self.strategy.execute_trade(signal)            
            send_telegram_alert(f"🚀 SMGrid {direction.upper()} breakout trade on {self.symbol}")
            
            if trade_id := result.get('order'):
                trade_record = {
                    "symbol": self.symbol,
                    "order_type": signal.get("type", "market"),
                    "direction": direction,
                    "lot": lot,
                    "entry_price": price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tp": tp,
                    "sl": sl,
                    "tag": signal.get("tag", "smgrid_trade"),
                    "strategy": "SMGrid",
                    "status": "open",
                    "ticket": trade_id
                }
                
                self.trade_tracker.register_open_trade(
                    trade_id=trade_id, # type: ignore
                    trade_data=trade_record # type: ignore
                )
                self.portfolio_manager.update_exposure(self.symbol, lot, direction)
        else:
            logger.info(f"[{self.symbol}] Trade blocked by portfolio constraints.")

    def _handle_grid_mode(self, signal: Dict[str, Any]) -> None:
        """
        Place grid-based pending orders around the base price.

        Args:
            signal (Dict[str, Any]): Signal dict with pending grid orders.
        """
        for order in signal.get("orders", []):
            lot = order["lot"]
            direction = order["direction"]
            price = order["price"]
            tp = order["tp"]
            tag = order.get("tag", "grid_order")
            
            if self.portfolio_manager.can_open_trade(self.symbol, lot):
                order_type = "buy_limit" if order["direction"] == "buy" else "sell_limit"
                
                result = self.order_manager.open_pending_order(
                    symbol=self.symbol,
                    order_type=order_type,
                    lot=lot,
                    price=price,
                    sl=None,
                    tp=tp,
                    comment="SMGrid Grid Order"
                )
                
                if trade_id := result.get('order'):
                    trade_record = {
                        "symbol": self.symbol,
                        "order_type": order_type,
                        "direction": direction,
                        "lot": lot,
                        "entry_price": price,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "tp": tp,
                        "sl": None,
                        "tag": tag,
                        "strategy": "SMGrid",
                        "status": "pending",
                        "ticket": trade_id
                    }
                    self.trade_tracker.register_open_trade(
                        trade_id=trade_id, # type: ignore
                        trade_data=trade_record # type: ignore
                    )
                    self.portfolio_manager.update_exposure(self.symbol, lot, direction)
            else:
                logger.info(f"[{self.symbol}] Skipping grid order due to exposure limit.")

    async def monitor_trades(self, current_price: float, direction: str) -> None:
        """
        Monitor existing trades for SL/TP and apply trailing logic.
        """
        open_trades = self.trade_tracker.get_open_trades()
        to_close = self.trade_watcher.evaluate_trades(open_trades, current_price, direction)

        for trade in to_close:
            if await self.order_manager.close_position(trade["ticket"]):
                await self.trade_tracker.close_trade(
                    trade_id=trade["ticket"],
                    close_price=current_price,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                send_telegram_alert(f"💥 Closed SMGrid trade on {self.symbol} @ {current_price:.5f}")

    async def scheduled_grid_reset(self) -> None:
        logger.info(f"[{self.symbol}] ⏳ Performing scheduled grid reset...")
        await self.order_manager.cancel_all_pending_orders(self.symbol)
        self.strategy.reset_grid_parameters()
        logger.info(f"[{self.symbol}] 🔄 Grid reset completed.")

class SMGridAsyncOrchestrator:
    """
    Launches and manages multiple SMGridTrader bots concurrently for different symbols.
    """

    def __init__(self, trading_plan: List[Dict[str, Any]], config: Dict[str, Any], broker: BaseBroker):
        """
        Args:
            trading_plan (List[Dict]): List of dicts with at least "symbol" key.
            config (Dict): Full configuration dictionary.
            broker (BaseBroker): Shared broker instance for all traders.
        """
        self.trading_plan = trading_plan
        self.config = config
        self.broker = broker
        self.scheduler = AsyncIOScheduler()
        self.traders: Dict[str, SMGridTrader] = {}

    def _create_cron_trigger(self, config: Dict[str, Any]) -> CronTrigger:
        time_str = config.get("time", "00:00")
        hour, minute = map(int, time_str.split(":"))
        frequency = config.get("frequency", "daily")

        if frequency == "daily":
            return CronTrigger(hour=hour, minute=minute)
        elif frequency == "weekly":
            day = config.get("day_of_week", "sun")
            return CronTrigger(day_of_week=day, hour=hour, minute=minute)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

    def _schedule_tasks(self):
        sched_cfg = self.config.get("schedule_config", {})

        for symbol, trader in self.traders.items():
            grid_cfg = sched_cfg.get("grid_reset", {})
            if grid_cfg.get("enabled", False):
                trigger = self._create_cron_trigger(grid_cfg)
                self.scheduler.add_job(
                    trader.scheduled_grid_reset,
                    trigger,
                    name=f"{symbol}_grid_reset"
                )

        if sched_cfg.get("rebalance", {}).get("enabled", False):
            trigger = self._create_cron_trigger(sched_cfg["rebalance"])
            self.scheduler.add_job(
                self._rebalance_portfolio,
                trigger,
                name="portfolio_rebalance"
            )

        if sched_cfg.get("performance_log", {}).get("enabled", False):
            trigger = self._create_cron_trigger(sched_cfg["performance_log"])
            self.scheduler.add_job(
                self._log_all_performance,
                trigger,
                name="performance_log"
            )

        self.scheduler.start()

    async def _rebalance_portfolio(self):
        """
        Redistribute portfolio capital based on recent strategy performance (e.g., PnL).
        Strategies with higher net PnL are allocated more exposure budget.
        """
        logger.info("🔁 Rebalancing portfolio based on recent performance...")

        performance_data = []
        for symbol, trader in self.traders.items():
            summary = trader.trade_tracker.get_closed_summary(symbol)
            if summary["total_trades"] > 0:
                performance_data.append({
                    "symbol": symbol,
                    "net_pnl": summary["net_pnl"]
                })

        if not performance_data:
            logger.warning("⚠️ No performance data available for rebalancing.")
            return

        total_pnl = sum(abs(p["net_pnl"]) for p in performance_data)
        if total_pnl == 0:
            logger.warning("⚠️ Total PnL is zero. Skipping rebalancing.")
            return

        for perf in performance_data:
            weight = abs(perf["net_pnl"]) / total_pnl
            new_allocation = self.config["portfolio"]["total_budget"] * weight

            # Apply new allocation
            trader = self.traders[perf["symbol"]]
            trader.portfolio_manager.set_budget_limit(perf["symbol"], new_allocation)

            logger.info(f"🔧 {perf['symbol']}: Rebalanced exposure budget to {new_allocation:.2f} (weight: {weight:.2%})")

# Inside orchestrator.py -> in SMGridAsyncOrchestrator class

    async def _log_all_performance(self):
        logger.info("📊 Logging performance for all strategies...")
        for symbol, trader in self.traders.items():
            await trader.strategy.log_performance(trader.trade_tracker, symbol)

    async def launch_all(self):
        """
        Launch all configured SMGridTrader bots asynchronously.
        """
        tasks = []

        for plan in self.trading_plan:
            symbol = plan["symbol"]
            trader = SMGridTrader(symbol=symbol, config=self.config, broker=self.broker)
            self.traders[symbol] = trader
            tasks.append(asyncio.create_task(trader.run()))

        self._schedule_tasks()
        
        logger.info(f"🚀 Launching {len(tasks)} SMGrid bots with scheduling...")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("SMGrid orchestrator shutdown initiated.")
