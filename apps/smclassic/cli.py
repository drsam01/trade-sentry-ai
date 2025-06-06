# File: apps/smclassic/cli.py

import asyncio
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

from core.utils.logger import get_logger
from core.utils.config_loader import load_config
from core.broker.mt5 import MT5Broker
from apps.smclassic.orchestrator import SMClassicAsyncOrchestrator

logger = get_logger(__name__)

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Parsed configuration.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"⚠️ Config file not found: {config_path}")
    return load_config(config_path)

def build_trading_plan(config: Dict[str, Any]) -> list[Dict[str, str]]:
    """
    Construct trading plan from the config.

    Args:
        config (Dict[str, Any]): Loaded config.

    Returns:
        list[Dict[str, str]]: Trading plan per symbol.
    """
    return [{"symbol": symbol} for symbol in config.get("symbols", [])]

async def main():
    parser = argparse.ArgumentParser(description="Run SMClassic multi-symbol trading bot.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the smclassic YAML config file."
    )
    args = parser.parse_args()

    config = load_config_file(args.config)
    broker = MT5Broker(config["broker"])
    trading_plan = build_trading_plan(config)

    if not broker.connect():
        logger.error("❌ Failed to connect to broker. Exiting...")
        return

    orchestrator = SMClassicAsyncOrchestrator(trading_plan, config, broker)
    await orchestrator.launch_all()

if __name__ == "__main__":
    asyncio.run(main())
