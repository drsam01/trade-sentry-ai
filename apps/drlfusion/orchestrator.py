# File: apps/drlfusion/orchestrator.py


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
from core.utils.timeframe_aggregator import TimeframeAggregator
from core.utils.logger import get_logger
from core.utils.notifier import send_telegram_alert
from core.broker.broker_utils import get_price_safely

from apps.drlfusion.strategy import DRLFusionStrategy

logger = get_logger(__name__)


class DRLFusionTrader:
    def __init__(self, symbol: str, broker: BaseBroker, config: Dict[str, Any]) -> None:
        self.symbol = symbol
        self.broker = broker
        self.config = config
        self.timeframe = config["timeframe"]
        self.lookback = config.get("lookback_bars", 100)

        account_info = self.broker.get_account_info()
        initial_balance = account_info.get("balance", 10000.0)

        self.order_manager = OrderManager(self.broker)
        self.risk_manager = RiskManager(balance=initial_balance, config=config["risk"])
        self.portfolio_manager = PortfolioManager(config["portfolio"])
        self.trade_tracker = TradeTracker()
        self.trade_watcher = TradeWatcher(self.risk_manager.apply_trailing_stop)

        self.strategy = DRLFusionStrategy(config)
        self.aggregator = TimeframeAggregator([self.timeframe])
        self.trailing_cfg = config.get("trailing_stop", {"enabled": False})

    async def run(self) -> None:
        if not self.broker.connect():
            logger.error(f"[{self.symbol}] Broker connection failed.")
            return

        logger.info(f"[{self.symbol}] DRLFusion trader started.")
        while True:
            try:
                await self.process_tick()
                await self.watch_open_trades()
            except Exception as e:
                logger.exception(f"[{self.symbol}] Error: {e}")
            await asyncio.sleep(self.config.get("poll_interval", 10))

    async def process_tick(self) -> None:
        if not self.broker.is_connected():
            logger.warning(f"[{self.symbol}] Broker disconnected. Reconnecting...")
            if not self.broker.connect():
                return

        m1_bars = self.broker.fetch_ohlcv(self.symbol, "M1", self.lookback)
        aggregated_bars = []

        for bar in m1_bars:
            result = self.aggregator.add_candle(bar)
            aggregated_bars.extend(result.get(self.timeframe, []))

        if len(aggregated_bars) < self.lookback:
            logger.warning(f"[{self.symbol}] Not enough aggregated data.")
            return

        df = pd.DataFrame(aggregated_bars)
        self.strategy.load_data(df)

        signal = self.strategy.generate_signal(current_index=len(df) - 1)
        if not signal:
            return

        direction = signal["direction"]
        price, sl, tp = signal["price"], signal["sl"], signal["tp"]
        stop_loss_pips = abs(price - sl) / self.config["risk"].get("pip_value", 0.0001)

        lot = self.risk_manager.compute_lot_size(
            stop_loss_pips=stop_loss_pips,
            pip_value=self.config["risk"].get("pip_value", 0.0001),
            max_lot=self.config["risk"].get("max_lot", 1.0)
        )

        if self.portfolio_manager.can_open_trade(self.symbol, lot):
            result = self.order_manager.open_market_order(
                symbol=self.symbol,
                direction=direction,
                lot=lot,
                sl=sl,
                tp=tp,
                comment="DRLFusion Trade"
            )

            self.strategy.execute_trade(signal)
            send_telegram_alert(f"📈 DRLFusion {direction.upper()} trade on {self.symbol}")

            if trade_id := result.get("order"):
                trade_data = {
                    "symbol": self.symbol,
                    "order_type": signal.get("type", "market"),
                    "direction": direction,
                    "lot": lot,
                    "entry_price": price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tp": tp,
                    "sl": sl,
                    "tag": signal.get("tag", "drlfusion_trade"),
                    "strategy": "DRLFusion",
                    "status": "open",
                    "ticket": trade_id
                }

                self.trade_tracker.register_open_trade(trade_id, trade_data)
                self.portfolio_manager.update_exposure(self.symbol, lot, direction)

    async def watch_open_trades(self) -> None:
        if not self.trailing_cfg.get("enabled", False):
            return

        price = await get_price_safely(self.broker, self.symbol)
        open_trades = self.trade_tracker.get_open_trades(self.symbol) # type: ignore
        direction = next(iter(open_trades.values()), {}).get("direction", "long")  # fallback

        to_close = self.trade_watcher.evaluate_trades(open_trades, price, direction)
        for trade in to_close:
            if await self.order_manager.close_position(trade["ticket"]):
                await self.trade_tracker.close_trade(
                    trade_id=trade["ticket"],
                    close_price=price,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                send_telegram_alert(f"💥 Closed DRLFusion trade on {self.symbol} @ {price:.5f}")


class DRLFusionAsyncOrchestrator:
    def __init__(self, trading_plan: List[Dict[str, Any]], config: Dict[str, Any], broker: BaseBroker):
        self.trading_plan = trading_plan
        self.config = config
        self.broker = broker
        self.scheduler = AsyncIOScheduler()
        self.traders: Dict[str, DRLFusionTrader] = {}

    def _create_cron_trigger(self, config: Dict[str, Any]) -> CronTrigger:
        time_str = config.get("time", "00:00")
        hour, minute = map(int, time_str.split(":"))
        frequency = config.get("frequency", "daily")

        if frequency == "daily":
            return CronTrigger(hour=hour, minute=minute)
        elif frequency == "weekly":
            return CronTrigger(day_of_week=config.get("day_of_week", "sun"), hour=hour, minute=minute)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

    def _schedule_tasks(self):
        sched_cfg = self.config.get("schedule_config", {})

        for symbol, trader in self.traders.items():
            if sched_cfg.get("rebalance", {}).get("enabled", False):
                trigger = self._create_cron_trigger(sched_cfg["rebalance"])
                self.scheduler.add_job(
                    self._rebalance_portfolio,
                    trigger,
                    name=f"{symbol}_rebalance"
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
        logger.info("🔁 Rebalancing portfolio for DRLFusion...")
        perf_data = []

        for symbol, trader in self.traders.items():
            summary = trader.trade_tracker.get_closed_summary(symbol)
            if summary["total_trades"] > 0:
                perf_data.append({
                    "symbol": symbol,
                    "net_pnl": summary["net_pnl"]
                })

        total_pnl = sum(abs(x["net_pnl"]) for x in perf_data)
        if not perf_data or total_pnl == 0:
            return

        for p in perf_data:
            weight = abs(p["net_pnl"]) / total_pnl
            allocation = self.config["portfolio"]["total_budget"] * weight
            self.traders[p["symbol"]].portfolio_manager.set_budget_limit(p["symbol"], allocation)
            logger.info(f"{p['symbol']}: Rebalanced to {allocation:.2f} (weight: {weight:.2%})")

    async def _log_all_performance(self):
        logger.info("📊 Logging DRLFusion performance...")
        for symbol, trader in self.traders.items():
            await trader.strategy.log_performance(trader.trade_tracker, symbol)

    async def launch_all(self):
        tasks = []

        for plan in self.trading_plan:
            symbol = plan["symbol"]
            trader = DRLFusionTrader(symbol=symbol, broker=self.broker, config=self.config)
            self.traders[symbol] = trader
            tasks.append(asyncio.create_task(trader.run()))

        self._schedule_tasks()
        logger.info(f"🚀 Launched {len(tasks)} DRLFusion traders.")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.warning("DRLFusion orchestrator shutdown.")
