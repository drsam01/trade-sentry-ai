# File: apps/smedge/cli.py

import asyncio
import argparse
import yaml
from pathlib import Path

from apps.smedge.orchestrator import SMEdgeAsyncOrchestrator


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed config dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SMEdge Live Trader")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml file",
    )
    return parser.parse_args()


def main():
    """
    Entry point for SMEdge live trading session.
    """
    args = parse_args()
    config = load_config(args.config)
    symbols = config["symbols"]

    orchestrator = SMEdgeAsyncOrchestrator(symbols, config)
    asyncio.run(orchestrator.launch_all())


if __name__ == "__main__":
    main()
