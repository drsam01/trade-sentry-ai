# File: drlfusion/modelling/training/multi_agent_trainer.py

"""
Parallel multi-agent training launcher for DRLFusion.
- Reads YAML file with multiple timeframes per symbol.
- Trains specified agents in parallel using Ray.
- Supports override injection and modular hyperparameter source selection.
- Saves output logs for each run to per-agent log files.
"""

import sys
import os
import ray
import yaml
import argparse
import subprocess
from typing import List, Dict, Any
from drlfusion.configs.config import Config

config = Config()

# Default set of supported agent types
DEFAULT_AGENT_TYPES = ["ppo", "dqn", "a2c", "sac"]

def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """
    Convert CLI-style overrides (key=value) into a dictionary.
    """
    overrides = {}
    for pair in pairs:
        k, v = pair.split("=")
        overrides[k] = eval(v)
    return overrides

def parse_yaml_file(yaml_path: str) -> List[Dict[str, str]]:
    """
    Load training tasks from a YAML file that maps each symbol to multiple timeframes.

    YAML format:
    ---
    - symbol: EURUSD
        timeframes: [M15, H1]
    - symbol: GBPUSD
        timeframes: [M30]

    Returns:
        List of {symbol, timeframe} pairs.
    """
    with open(yaml_path, "r") as f:
        entries = yaml.safe_load(f)

    # Flatten structure: one entry per symbol-timeframe
    plan = []
    for entry in entries:
        symbol = entry["symbol"]
        plan.extend(
            {"symbol": symbol, "timeframe": timeframe}
            for timeframe in entry["timeframes"]
        )
    return plan

@ray.remote
def train_agent_ray(
    symbol: str,
    timeframe: str,
    agent_type: str,
    overrides: Dict[str, Any],
    log_dir: str,
    hparam_source: str = "yaml"
) -> str:
    """
    Launch agent training subprocess for a specific configuration and redirect logs.

    Args:
        symbol (str): Trading symbol (e.g. EURUSD)
        timeframe (str): Timeframe (e.g. M15)
        agent_type (str): One of [ppo, dqn, a2c, sac]
        overrides (dict): Hyperparameter overrides
        log_dir (str): Directory to store logs
        hparam_source (str): One of [yaml, config, file]

    Returns:
        str: Completion message with log file path
    """
    # Construct log file name
    log_path = os.path.join(
        log_dir,
        f"{symbol.lower()}_{timeframe.lower()}_{agent_type}.log"
    )
    #os.makedirs(log_dir, exist_ok=True)

    # Convert overrides into command-line arguments
    override_args = []
    for k, v in overrides.items():
        override_args.extend(["--override", f"{k}={v}"])

    # Command to run training module
    cmd = [
        sys.executable,
        "-m", "drlfusion.modelling.training.agent_train_worker",
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--agent-type", agent_type,
        "--hparam-source", hparam_source,
        *override_args
    ]

    # Run subprocess and redirect output to log
    with open(log_path, "w") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=log_file)

    return f"✅ {agent_type.upper()} done for {symbol}-{timeframe} → {log_path}"

def main():
    """
    CLI entry point to train all agents for multiple symbol-timeframe pairs from a YAML file.
    
    USAGE:
        cd strategies
        python -m drlfusion.modelling.training.multi_agent_trainer --yaml drlfusion/configs/training_config.yaml
    """
    log_dir = os.path.join(config.LOGS_DIR, "agent_train")
    os.makedirs(log_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description="Multi-agent parallel training runner.")
    parser.add_argument("--yaml", type=str, required=True, help="YAML file defining {symbol, timeframes} list.")
    parser.add_argument("--override", nargs="*", help="Override values, e.g., LEARNING_RATE=0.001")
    parser.add_argument("--agents", nargs="+", choices=DEFAULT_AGENT_TYPES, default=DEFAULT_AGENT_TYPES, help="Agent types to run.")
    parser.add_argument("--hparam-source", type=str, default="yaml", choices=["yaml", "config", "file"], help="Where to pull hyperparameters from.")
    parser.add_argument("--log-dir", type=str, default=log_dir, help="Directory for log output.") # "./drlfusion/logs/agent_train"

    args = parser.parse_args()

    overrides: Dict[str, Any] = parse_overrides(args.override) if args.override else {}
    training_plan: List[Dict[str, str]] = parse_yaml_file(args.yaml)

    ray.init(ignore_reinit_error=True)

    # Submit one Ray task per (symbol, timeframe, agent)
    futures = []
    for task in training_plan:
        symbol, timeframe = task["symbol"], task["timeframe"]        
        futures.extend(
            train_agent_ray.remote(
                symbol=symbol,
                timeframe=timeframe,
                agent_type=agent_type,
                overrides=overrides,
                log_dir=args.log_dir,
                hparam_source=args.hparam_source
            )
            for agent_type in args.agents
        )
    # Wait for all tasks to complete
    results = ray.get(futures)
    for res in results:
        print(res)


if __name__ == "__main__":
    main()
