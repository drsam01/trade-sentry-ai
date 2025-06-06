# File: drlfusion/core/brokers/binance_interface.py

import aiohttp
import asyncio
import pandas as pd
from typing import Optional, Dict, Any, List
import time
from drlfusion.core.brokers.broker_base import BrokerBase
from drlfusion.core.config.config import Config
from drlfusion.core.utils.logger import get_logger

logger = get_logger('binance_interface')


class BinanceInterface(BrokerBase):
    """
    Binance Futures API Interface (inherits BrokerBase).
    """

    def __init__(self, config: Config):
        self.config = config
        self.session = None
        self.api_key = config.BINANCE_API_KEY
        self.api_secret = config.BINANCE_API_SECRET
        self.base_url = "https://fapi.binance.com"

    async def connect(self) -> bool:
        self.session = aiohttp.ClientSession()
        logger.info("Binance Futures session started.")
        return True

    async def shutdown(self) -> None:
        if self.session:
            await self.session.close()
            logger.info("Binance Futures session closed.")

    async def fetch_historical_data(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        from strategies.drlfusion.core.utils.timeframe_mapper import map_timeframe
        timeframe = map_timeframe("BINANCE", timeframe)
        
        url = f"{self.base_url}/fapi/v1/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": lookback
        }
        async with self.session.get(url, params=params) as resp:
            data = await resp.json()

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.astype(float).set_index('timestamp')

        return df.reset_index()

    async def fetch_latest_candle(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        from strategies.drlfusion.core.utils.timeframe_mapper import map_timeframe
        timeframe = map_timeframe("BINANCE", timeframe)

        df = await self.fetch_historical_data(symbol, timeframe, lookback=2)
        latest = df.iloc[-2]
        return {
            'time': latest['timestamp'],
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'close': latest['close'],
            'volume': latest['volume']
        }

    async def place_order(self, symbol: str, side: str, volume: float, price: Optional[float] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/fapi/v1/order"
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": round(volume, 3),  # Binance needs rounded quantities
            "timestamp": int(time.time() * 1000)
        }
        headers = {
            "X-MBX-APIKEY": self.api_key
        }

        async with self.session.post(url, params=params, headers=headers) as resp:
            response = await resp.json()
            if resp.status != 200:
                logger.error(f"Order failed: {response}")
                return {"success": False, "error": response}
            logger.info(f"Order successful: {side} {volume} {symbol}")
            return {"success": True, "order": response}

    async def close_position(self, symbol: str) -> Dict[str, Any]:
        positions = await self.fetch_open_positions(symbol)
        if not positions:
            return {"success": True}

        for pos in positions:
            side = "SELL" if pos['positionAmt'] > 0 else "BUY"
            quantity = abs(float(pos['positionAmt']))
            await self.place_order(symbol, side, quantity)

        return {"success": True}

    async def fetch_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/fapi/v2/positionRisk"
        headers = {"X-MBX-APIKEY": self.api_key}
        params = {"timestamp": int(time.time() * 1000)}

        async with self.session.get(url, params=params, headers=headers) as resp:
            data = await resp.json()

        if symbol:
            data = [pos for pos in data if pos['symbol'] == symbol.upper()]

        return data

    async def fetch_account_info(self) -> Dict[str, Any]:
        url = f"{self.base_url}/fapi/v2/account"
        headers = {"X-MBX-APIKEY": self.api_key}
        params = {"timestamp": int(time.time() * 1000)}

        async with self.session.get(url, params=params, headers=headers) as resp:
            data = await resp.json()

        return {
            "balance": float(data['totalWalletBalance']),
            "equity": float(data['totalCrossWalletBalance']),
            "margin": float(data['totalMaintMargin']),
            "margin_free": float(data['availableBalance']),
            "margin_level": None
        }

    async def fetch_symbol_info(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        async with self.session.get(url) as resp:
            info = await resp.json()

        for s in info['symbols']:
            if s['symbol'] == symbol.upper():
                return {
                    "tick_size": float([f['tickSize'] for f in s['filters'] if f['filterType'] == 'PRICE_FILTER'][0]),
                    "volume_min": float([f['minQty'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE'][0]),
                    "volume_max": float([f['maxQty'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE'][0]),
                    "volume_step": float([f['stepSize'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE'][0]),
                    "margin_initial": None,
                    "margin_maintenance": None
                }
        raise ValueError(f"Symbol info not found for {symbol}")

    async def get_latest_price(self, symbol: str) -> float:
        url = f"{self.base_url}/fapi/v1/ticker/bookTicker?symbol={symbol.upper()}"
        async with self.session.get(url) as resp:
            data = await resp.json()
        return (float(data["bidPrice"]) + float(data["askPrice"])) / 2
