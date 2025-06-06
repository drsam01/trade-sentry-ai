# File: apps/drlfusion/cli.py

import asyncio
import argparse
from core.broker.mt5 import MT5Broker
from core.utils.config_loader import load_config
from apps.drlfusion.orchestrator import DRLFusionAsyncOrchestrator

def parse_args():
    parser = argparse.ArgumentParser(description="Launch DRLFusion strategy.")
    parser.add_argument("--config", type=str, required=True, help="Path to drlfusion_config.yaml")
    return parser.parse_args()

def build_trading_plan(symbols):
    return [{"symbol": s} for s in symbols]

def main():
    args = parse_args()
    config = load_config(args.config)
    broker = MT5Broker(config["broker"])
    trading_plan = build_trading_plan(config["symbols"])

    orchestrator = DRLFusionAsyncOrchestrator(
        trading_plan=trading_plan,
        config=config,
        broker=broker
    )

    asyncio.run(orchestrator.launch_all())

if __name__ == "__main__":
    main()
