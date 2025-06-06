# apps/smgrid/backtest/cli.py

import argparse
import yaml
from apps.smgrid.backtest.backtest_runner import SMGridBacktester
from apps.smgrid.backtest.walk_forward import WalkForwardEngine


def main():
    parser = argparse.ArgumentParser(description="Run SMGrid Backtest/Walk-Forward")
    parser.add_argument("--config", type=str, default="apps/smgrid/backtest/smgrid_backtest_config.yaml")
    parser.add_argument("--walkforward", action="store_true", help="Enable walk-forward testing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.walkforward:
        engine = WalkForwardEngine(config)
        results = engine.run()
    else:
        results = SMGridBacktester(config).run_all()

    SMGridBacktester(config).visualize_results(results)


if __name__ == "__main__":
    main()
