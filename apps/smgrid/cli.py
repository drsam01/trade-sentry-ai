# File: apps/smgrid/cli.py

import asyncio
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

from core.utils.logger import get_logger
from core.utils.config_loader import load_config
from core.broker.mt5 import MT5Broker
from apps.smgrid.orchestrator import SMGridAsyncOrchestrator

logger = get_logger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_config(config_path)

def build_trading_plan(config: Dict[str, Any]) -> list:
    """
    Construct trading plan from config symbols.

    Args:
        config (Dict): Full config.

    Returns:
        list: List of trading instructions per symbol.
    """
    return [{"symbol": symbol} for symbol in config["symbols"]]

async def main():
    parser = argparse.ArgumentParser(description="Run SMGrid multi-symbol trading bot.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the SMGrid YAML config file."
    )
    args = parser.parse_args()

    # Load config and prepare components
    config = load_config(args.config)
    broker = MT5Broker(config["broker"])
    trading_plan = build_trading_plan(config)

    if not broker.connect():
        logger.error("❌ Failed to connect to broker. Exiting...")
        return

    orchestrator = SMGridAsyncOrchestrator(trading_plan, config, broker)
    await orchestrator.launch_all()

if __name__ == "__main__":
    asyncio.run(main())
