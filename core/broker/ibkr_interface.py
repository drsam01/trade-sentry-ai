# File: drlfusion/core/brokers/ibkr_interface.py

import asyncio
import pandas as pd
from typing import Optional, Dict, List
from ib_insync import IB, Contract, util

from .base_broker import BaseBroker
from drlfusion.core.config.config import Config
from drlfusion.core.utils.logger import get_logger

logger = get_logger('ibkr_interface')


class IBKRInterface(BaseBroker):
    """
    Interactive Brokers (IBKR) API Interface.
    """

    def __init__(self, config: Config):
        self.config = config
        self.ib = IB()
        self.host = config.IBKR_HOST
        self.port = config.IBKR_PORT
        self.client_id = config.IBKR_CLIENT_ID

    async def connect(self) -> bool:
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        logger.info("Connected to Interactive Brokers Gateway/TWS.")
        return True

    async def shutdown(self) -> None:
        self.ib.disconnect()
        logger.info("Disconnected from Interactive Brokers.")

    async def fetch_historical_data(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        from core.utils.timeframe_mapper import map_timeframe
        timeframe = map_timeframe("IBKR", timeframe)
        
        contract = self._create_contract(symbol)

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=f'{lookback} D',
            barSizeSetting=timeframe,
            whatToShow='MIDPOINT',
            useRTH=True,
            formatDate=1
        )
        await asyncio.sleep(0)

        df = util.df(bars)
        df.rename(columns={'date': 'time', 'volume': 'volume'}, inplace=True)
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        df['time'] = pd.to_datetime(df['time'])
        return df.reset_index(drop=True)

    async def fetch_latest_candle(self, symbol: str, timeframe: str) -> Dict[str, any]:
        from strategies.drlfusion.core.utils.timeframe_mapper import map_timeframe
        timeframe = map_timeframe("IBKR", timeframe)
        
        df = await self.fetch_historical_data(symbol, timeframe, lookback=2)
        latest = df.iloc[-2]
        return {
            'time': latest['time'],
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'close': latest['close'],
            'volume': latest['volume']
        }

    async def place_order(self, symbol: str, side: str, volume: float, price: Optional[float] = None) -> Dict[str, any]:
        contract = self._create_contract(symbol)

        action = 'BUY' if side.lower() == 'buy' else 'SELL'

        order = util.MarketOrder(action, volume)
        trade = self.ib.placeOrder(contract, order)
        await asyncio.sleep(0)

        if trade.orderStatus.status not in ['Filled', 'Submitted']:
            logger.error(f"IBKR order failed: {trade.orderStatus.status}")
            return {"success": False, "error": trade.orderStatus.status}

        logger.info(f"IBKR order successful: {side.upper()} {volume} units {symbol}")
        return {"success": True, "order": trade}

    async def close_position(self, symbol: str) -> Dict[str, any]:
        open_positions = await self.fetch_open_positions(symbol)
        if not open_positions:
            return {"success": True}

        for pos in open_positions:
            action = 'SELL' if pos['position'] > 0 else 'BUY'
            await self.place_order(symbol, action, abs(pos['position']))
        return {"success": True}

    async def fetch_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, any]]:
        positions = self.ib.positions()
        await asyncio.sleep(0)

        if symbol:
            positions = [p for p in positions if p.contract.symbol == symbol]

        return [
            {
                'symbol': p.contract.symbol,
                'position': p.position,
                'avgCost': p.avgCost,
            }
            for p in positions
        ]

    async def fetch_account_info(self) -> Dict[str, any]:
        account_values = self.ib.accountSummary()
        await asyncio.sleep(0)

        return {
            "balance": float(account_values.loc['NetLiquidation', 'value']),
            "equity": float(account_values.loc['NetLiquidation', 'value']),
            "margin": float(account_values.loc['MaintenanceMarginReq', 'value']),
            "margin_free": float(account_values.loc['AvailableFunds', 'value']),
            "margin_level": None
        }

    async def fetch_symbol_info(self, symbol: str) -> Dict[str, any]:
        # IBKR symbol info retrieval is not direct; we'll assume reasonable defaults
        return {
            "tick_size": 0.01,
            "volume_min": 1,
            "volume_max": 10000,
            "volume_step": 1,
            "margin_initial": None,
            "margin_maintenance": None
        }

    def _create_contract(self, symbol: str) -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'CASH'
        contract.exchange = 'IDEALPRO'
        contract.currency = 'USD'
        return contract

    async def get_latest_price(self, symbol: str) -> float:
        contract = self._create_contract(symbol)
        tick = self.ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(1)
        bid = tick.bid or 0
        ask = tick.ask or 0
        self.ib.cancelMktData(contract)
        if not bid or not ask:
            raise ValueError(f"No market data received for symbol: {symbol}")
        return (bid + ask) / 2
