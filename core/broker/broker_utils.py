# File: core/broker/broker_utils.py

import inspect
import asyncio
from typing import Any


async def get_price_safely(broker: Any, symbol: str) -> float:
    """
    Get the latest price from a broker (handles both sync and async methods).
    
    Args:
        broker: An instance of a broker (may have sync or async get_latest_price).
        symbol: Symbol string like 'EURUSD' or 'BTCUSDT'.

    Returns:
        float: Latest price (mid-point)
    """
    get_price = getattr(broker, "get_latest_price", None)
    if not callable(get_price):
        raise NotImplementedError("Broker does not implement get_latest_price()")

    if inspect.iscoroutinefunction(get_price):
        return await get_price(symbol)
    else:
        return get_price(symbol) # type: ignore
