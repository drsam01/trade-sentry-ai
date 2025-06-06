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
from core.utils.timeframe_aggregator import TimeframeAggregator

from apps.smclassic.strategy import SMClassicStrategy

logger = get_logger(__name__)


class SMClassicTrader:
    def __init__(self, symbol: str, broker: BaseBroker, config: Dict[str, Any]) -> None:
        self.symbol = symbol
        self.broker = broker
        self.config = config
        self.htf_minutes = config["timeframes"]["htf"]
        self.ctf_minutes = config["timeframes"]["ctf"]
        self.lookback = config.get("lookback_bars", 100)

        account_info = self.broker.get_account_info()
        initial_balance = account_info.get("balance", 10000.0)

        self.order_manager = OrderManager(self.broker)
        self.risk_manager = RiskManager(balance=initial_balance, config=config["risk"])
        self.portfolio_manager = PortfolioManager(config.get("portfolio", {}))
        self.trade_tracker = TradeTracker()
        self.trade_watcher = TradeWatcher(self.risk_manager.apply_trailing_stop)
        self.strategy = SMClassicStrategy(config)

        self.aggregator = TimeframeAggregator([self.ctf_minutes, self.htf_minutes])

    async def run(self) -> None:
        if not self.broker.connect():
            logger.error(f"[{self.symbol}] Broker connection failed.")
            return

        logger.info(f"[{self.symbol}] SMClassic trader started.")
        while True:
            try:
                await self.process_tick()
            except Exception as e:
                logger.exception(f"[{self.symbol}] Runtime error: {e}")
            await asyncio.sleep(self.config.get("poll_interval", 10))

    async def process_tick(self) -> None:
        if not self.broker.is_connected():
            logger.warning(f"[{self.symbol}] Reconnecting to broker...")
            if not self.broker.connect():
                return

        m1_data = self.broker.fetch_ohlcv(self.symbol, "1min", self.lookback)
        if not m1_data:
            logger.warning(f"[{self.symbol}] No M1 data retrieved.")
            return

        # Feed M1 candles into the aggregator
        for bar in m1_data:
            aggregated = self.aggregator.add_candle(bar)

        # Convert aggregated lists to DataFrames
        htf_bars = aggregated.get(self.htf_minutes, []) # type: ignore
        ctf_bars = aggregated.get(self.ctf_minutes, []) # type: ignore

        if len(htf_bars) < 20 or len(ctf_bars) < 20:
            logger.debug(f"[{self.symbol}] Waiting for enough HTF/CTF data.")
            return

        df_htf = pd.DataFrame(htf_bars)
        df_ctf = pd.DataFrame(ctf_bars)

        self.strategy.load_data(df_htf)

        signal = self.strategy.generate_signal(len(df_ctf) - 1, injected_lot=None)
        
        if not signal or signal.get("price") is None or signal.get("sl") is None:
            logger.warning(f"[{self.symbol}] Incomplete signal. Skipping trade setup.")
            return

        sl_pips = abs(signal["price"] - signal["sl"]) / self.config.get("pip_size",0.0001) # type: ignore

        if sl_pips == 0:
            logger.warning("SL pips computed as 0. Skipping trade.")
            return

        lot = self.risk_manager.compute_lot_size(
            stop_loss_pips=sl_pips,
            pip_value=self.config["risk"].get("pip_value", 0.01),
            max_lot=self.config["risk"].get("max_lot", 1.0)
        )
        
        signal = self.strategy.finalize_signal(signal, lot)
        
        if signal and self.portfolio_manager.can_open_trade(self.symbol, lot):
            await self.place_limit_order(signal)

    async def place_limit_order(self, signal: Dict[str, Any]) -> None:
        order_type = "buy_limit" if signal["direction"] == "bullish" else "sell_limit"

        result = self.order_manager.open_pending_order(
            symbol=self.symbol,
            order_type=order_type,
            lot=signal["lot"],
            price=signal["price"],
            sl=signal["sl"],
            tp=signal["tp"],
            comment="SMClassic Entry"
        )

        if trade_id := result.get("order"):
            self.strategy.execute_trade(signal)
            send_telegram_alert(f"📘 SMClassic {signal['direction'].upper()} entry on {self.symbol}")

            trade_data = {
                "symbol": self.symbol,
                "order_type": order_type,
                "direction": signal["direction"],
                "lot": signal["lot"],
                "entry_price": signal["price"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tp": signal["tp"],
                "sl": signal["sl"],
                "tag": signal.get("tag", "smclassic_entry"),
                "strategy": "SMClassic",
                "status": "pending",
                "ticket": trade_id
            }

            self.trade_tracker.register_open_trade(trade_id, trade_data)
            self.portfolio_manager.update_exposure(self.symbol, signal["lot"], signal["direction"])

    async def monitor_trades(self, current_price: float, direction: str) -> None:
        open_trades = self.trade_tracker.get_open_trades()
        to_close = self.trade_watcher.evaluate_trades(open_trades, current_price, direction)

        for trade in to_close:
            if await self.order_manager.close_position(trade["ticket"]):
                await self.trade_tracker.close_trade(
                    trade_id=trade["ticket"],
                    close_price=current_price,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                send_telegram_alert(f"💥 Closed SMClassic trade on {self.symbol} @ {current_price:.5f}")


class SMClassicAsyncOrchestrator:
    def __init__(self, trading_plan: List[Dict[str, Any]], config: Dict[str, Any], broker: BaseBroker):
        self.trading_plan = trading_plan
        self.config = config
        self.broker = broker
        self.scheduler = AsyncIOScheduler()
        self.traders: Dict[str, SMClassicTrader] = {}

    def _create_cron_trigger(self, time_str: str, frequency: str = "daily", day_of_week: str = "sun") -> CronTrigger:
        hour, minute = map(int, time_str.split(":"))
        if frequency == "daily":
            return CronTrigger(hour=hour, minute=minute)
        elif frequency == "weekly":
            return CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute)
        else:
            raise ValueError(f"Unsupported schedule frequency: {frequency}")

    def _schedule_tasks(self):
        perf_cfg = self.config.get("schedule_config", {}).get("performance_log", {})
        if perf_cfg.get("enabled", False):
            trigger = self._create_cron_trigger(perf_cfg.get("time", "23:00"), perf_cfg.get("frequency", "daily"))
            self.scheduler.add_job(self._log_all_performance, trigger, name="performance_log")

        self.scheduler.start()

    async def _log_all_performance(self):
        logger.info("📊 Logging performance for SMClassic strategies...")
        for symbol, trader in self.traders.items():
            await trader.strategy.log_performance(trader.trade_tracker, symbol)

    async def launch_all(self):
        tasks = []
        for plan in self.trading_plan:
            symbol = plan["symbol"]
            trader = SMClassicTrader(symbol=symbol, config=self.config, broker=self.broker)
            self.traders[symbol] = trader
            tasks.append(asyncio.create_task(trader.run()))

        self._schedule_tasks()
        logger.info(f"🚀 Launching {len(tasks)} SMClassic bots with HTF/CTF logic.")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Shutdown initiated.")
