# File: apps/smclassic/backtest/cli_backtest.py

import argparse
import yaml
from apps.smclassic.backtest.backtest_runner import SMClassicBacktester
from apps.smclassic.backtest.walk_forward import WalkForwardEngine


def load_yaml(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="SMClassic Backtest & Walk-Forward CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to smclassic_backtest_config.yaml")
    parser.add_argument("--mode", choices=["backtest", "walkforward"], default="backtest", help="Execution mode")

    args = parser.parse_args()
    config = load_yaml(args.config)

    if args.mode == "backtest":
        engine = SMClassicBacktester(config)
        results = engine.run_all()
        engine.visualize_results(results)

    elif args.mode == "walkforward":
        engine = WalkForwardEngine(config)
        results = engine.run()
        SMClassicBacktester({}).visualize_results(results)


if __name__ == "__main__":
    main()
