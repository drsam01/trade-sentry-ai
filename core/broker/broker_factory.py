# File: drlfusion/core/brokers/broker_factory.py

from .base_broker import BaseBroker
from typing import Dict, Any

# Dictionary to dynamically map broker names to classes
BROKER_REGISTRY = {
    "MT5": "strategies.drlfusion.core.brokers.mt5_interface.MT5Interface",
    "BINANCE": "strategies.drlfusion.core.brokers.binance_interface.BinanceInterface",
    "IBKR": "strategies.drlfusion.core.brokers.ibkr_interface.IBKRInterface"
}

def create_broker(broker_type: str, config: Dict[str, Any]) -> BaseBroker:
    """
    Factory method to create broker instances dynamically.

    Args:
        broker_type (str): Broker type (e.g., 'MT5', 'BINANCE').
        config (Config): Configuration object.

    Returns:
        BrokerBase: Instantiated broker object.
    """
    try:
        module_path, class_name = BROKER_REGISTRY[broker_type.upper()].rsplit(".", 1)
    except KeyError as e:
        raise ValueError(f"Unsupported broker type: {broker_type}") from e

    module = __import__(module_path, fromlist=[class_name])
    broker_class = getattr(module, class_name)
    return broker_class(config=config)
