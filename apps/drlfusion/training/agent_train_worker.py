# File: drlfusion/modelling/training/agent_train_worker.py

import os
import json
import argparse
import pandas as pd
import yaml
from typing import Dict, Optional

from apps.drlfusion.modelling.envs.simulated_trading_env import SimulatedTradingEnv

from drlfusion.modelling.agents.managers.ppo_manager import PPOManager
from drlfusion.modelling.agents.managers.dqn_manager import DQNManager
from drlfusion.modelling.agents.managers.a2c_manager import A2CManager
from drlfusion.modelling.agents.managers.sac_manager import SACManager

from drlfusion.modelling.agents.trainers.ppo_trainer import PPOTrainer
from drlfusion.modelling.agents.trainers.dqn_trainer import DQNTrainer
from drlfusion.modelling.agents.trainers.a2c_trainer import A2CTrainer
from drlfusion.modelling.agents.trainers.sac_trainer import SACTrainer

from drlfusion.modelling.utils.data_loader import load_raw_data
from drlfusion.modelling.utils.data_prep import preprocess_data



# === Registry to associate agent types with their manager and trainer classes ===
AGENT_REGISTRY = {
    "ppo": {
        "manager": PPOManager,
        "trainer": PPOTrainer
    },
    "dqn": {
        "manager": DQNManager,
        "trainer": DQNTrainer
    },
    "a2c": {
        "manager": A2CManager,
        "trainer": A2CTrainer
    },
    "sac": {
        "manager": SACManager,
        "trainer": SACTrainer
    }
}

def load_optimized_hparams(
    agent_type: str,
    symbol: str,
    timeframe: str,
    models_dir: str,
    config: Config,
    source: str = "file"
) -> Dict[str, Dict]:
    """
    Load agent-specific hyperparameters from one of:
    - JSON file (if source='file') the expected location:
        models_dir/<agent_type>/<symbol>/<agent_type>_optimized_hparams_<symbol>_<timeframe>.json
    - YAML config (if source='yaml') the expected location:
        drlfusion/configs/agents_training_config.yaml
    - config.py constants (if source='config') the expected location:
        drlfusion/configs/config.yaml

    Args:
        agent_type (str): Type of agent (e.g., "ppo", "dqn").
        symbol (str): Trading symbol (e.g., "EURUSD").
        timeframe (str): Timeframe (e.g., "M15").
        models_dir (str): Base path for model artifacts.
        config (Config): Configuration object for accessing default hyperparams.
        source (file): One of [yaml, config, file]

    Returns:
        Dict: Dictionary of hyperparameters.
    """
    agent_type = agent_type.lower()
    if source == "config":
        global_hparams = config.GLOBAL_HYPERPARAMS
        agent_upper = agent_type.upper()

        return {
            "agent_config": {
                "hidden_dim": global_hparams[agent_upper].get("HIDDEN_DIM"),
                "learning_rate": global_hparams[agent_upper].get("LEARNING_RATE"),
                "epsilon": global_hparams[agent_upper].get("EPSILON"),  # DQN only
            },
            "training_config": {
                "gamma": global_hparams[agent_upper].get("GAMMA"),
                "clip_epsilon": global_hparams[agent_upper].get("CLIP_EPSILON"),
                "aux_loss_weight": global_hparams[agent_upper].get("AUX_LOSS_WEIGHT"),
                "entropy_coef": global_hparams[agent_upper].get("ENTROPY_COEF"),
                "value_loss_coef": global_hparams[agent_upper].get("VALUE_LOSS_COEF"),
                "replay_buffer_size": global_hparams[agent_upper].get("REPLAY_BUFFER_SIZE"),
                "target_update_freq": global_hparams[agent_upper].get("TARGET_UPDATE_FREQ"),
                "actor_lr": global_hparams[agent_upper].get("ACTOR_LR"),
                "critic_lr": global_hparams[agent_upper].get("CRITIC_LR"),
                "alpha": global_hparams[agent_upper].get("ALPHA"),
                "tau": global_hparams[agent_upper].get("TAU"),
                "batch_size": global_hparams[agent_upper].get("BATCH_SIZE"),
            },
            "trainer_config": {
                "patience": config.MODEL_TRAINING_CONFIG.get("VALIDATION_PATIENCE", 20),
                #"batch_size": global_hparams[agent_upper].get("BATCH_SIZE", 2048),
                "plot_file": f"{agent_type}_training_plot.html"
            }
        }

    elif source == "file":
        filename = f"{agent_type}_optimized_hparams_{symbol.lower()}_{timeframe.lower()}.json"
        dirpath = os.path.join(models_dir, agent_type.upper(), symbol)
        filepath = os.path.join(dirpath, filename)
        os.makedirs(dirpath, exist_ok=True)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                full_config = json.load(f)
                return {
                    "agent_config": full_config.get("agent_config", {}),
                    "training_config": full_config.get("training_config", {}),
                    "trainer_config": {
                        "patience": config.MODEL_TRAINING_CONFIG.get("VALIDATION_PATIENCE", 20),
                        #"batch_size": global_hparams[agent_upper].get("BATCH_SIZE", 2048),
                        "plot_file": f"{agent_type}_training_plot.html"
                    } # full_config.get("trainer_config", {})
                }

    elif source == "yaml":
        full_yaml = load_yaml_config()
        agent_block = full_yaml.get(agent_type, {})
        return {
            "agent_config": agent_block.get("agent", {}),
            "training_config": agent_block.get("training", {}),
            "trainer_config": agent_block.get("trainer", {})
        }

    raise ValueError(f"Unsupported hyperparameter source: {source}")

def load_training_dataframe(
    symbol: str,
    timeframe: str,
    config: Config
) -> pd.DataFrame:
    """
    Attempts to load a preprocessed training DataFrame from:
        config.DATA_DIR/trainsets/<symbol>/<symbol>_<timeframe>_train.csv

    If not found, falls back to raw + preprocessed data from the pipeline.

    Returns:
        pd.DataFrame: The training DataFrame.
    """
    filename = f"{symbol.lower()}_{timeframe.lower()}_train.csv"
    filepath = os.path.join(config.DATA_DIR, "trainsets", symbol, filename)

    if os.path.exists(filepath):
        return pd.read_csv(filepath)

    # Fallback: build from raw source
    df = load_raw_data(symbol, timeframe, config)
    return preprocess_data(df, config)


def load_yaml_config(path: str = "drlfusion/configs/agents_training_config.yaml") -> Dict[str, dict]:
    """
    Load nested YAML RL config (for PPO, DQN, A2C, SAC).

    Args:
        path (str): Path to YAML config.

    Returns:
        Dict[str, dict]: Mapping of agent -> {agent, training, trainer}.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_agent_worker(
    symbol: str,
    timeframe: str,
    agent_type: str,
    overrides: Optional[Dict] = None,
    hparam_source: str = "yaml"
) -> str:
    """
    Main training pipeline for any DRL agent with modular config source support.
    - Trains a single DRL agent for a given symbol, timeframe, and agent type.
    - Loads optimized hyperparameters if available; otherwise falls back to config defaults.

    Args:
        symbol (str): Trading symbol to train on (e.g., "EURUSD").
        timeframe (str): Timeframe to train on (e.g., "M15").
        agent_type (str): Agent type (e.g., "ppo", "dqn").
        overrides (Optional[Dict]): Additional manual override for any config values.
        hparam_source (str): One of [yaml, config, file]

    Returns:
        str: Status message with model save location.
    """
    overrides = overrides or {}
    agent_type = agent_type.lower()

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unsupported agent type: {agent_type}. Must be one of: {list(AGENT_REGISTRY.keys())}")

    # Initialize a temporary config to locate model directory
    base_config = Config()

    # Load hyperparameters from chosen source
    hparams = load_optimized_hparams(
        agent_type=agent_type, 
        symbol=symbol, 
        timeframe=timeframe, 
        models_dir=base_config.MODELS_DIR, 
        config=base_config, 
        source=hparam_source
    )

    # Merge override config values
    combined_overrides = {**hparams.get("agent_config", {}), **hparams.get("training_config", {}), **overrides}
    config = Config(symbol=symbol, timeframe=timeframe, overrides=combined_overrides)

    # Load training data and wrap in DRLFusion environment
    df = load_training_dataframe(symbol, timeframe, config)
    env = SimulatedTradingEnv(df, config)

    # Define model artifacts save directory path
    save_dir = os.path.join(config.MODELS_DIR, agent_type.upper(), symbol, timeframe)
    os.makedirs(save_dir, exist_ok=True)

    # Load agent-specific manager and trainer classes
    AgentManagerClass = AGENT_REGISTRY[agent_type]["manager"]
    AgentTrainerClass = AGENT_REGISTRY[agent_type]["trainer"]

    # Initialize agent manager and training engine
    manager = AgentManagerClass(
        env.observation_space, 
        env.action_space,
        agent_config=hparams.get("agent_config", {}),
        training_config=hparams.get("training_config", {})
    )
    
    trainer = AgentTrainerClass(
        manager,
        env,
        env,
        save_dir=save_dir,
        patience=hparams.get("trainer_config", {}).get("patience", 20),
        batch_size=hparams.get("trainer_config", {}).get("batch_size", 2048),
        plot_file=hparams.get("trainer_config", {}).get("plot_file", f"{agent_type}_training_plot.html")
    )

    # Run the agent training loop
    trainer.run_train(
        config.MODEL_TRAINING_CONFIG["NUM_TRAINING_STEPS"] //
        config.MODEL_TRAINING_CONFIG["UPDATE_TIMESTEPS"]
    )

    return f"✅ Trained {agent_type.upper()} agent for {symbol}-{timeframe}, saved to {save_dir}"

def main():
    """
    Command-line (CLI) entry point for training a single agent.
    Usage example:
        python -m drlfusion.modelling.training.agent_train_worker --symbol EURUSD --timeframe M15 --agent_type ppo --hparam_source yaml
    """
    parser = argparse.ArgumentParser(description="Train a DRL agent with tunable config source.")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., EURUSD)")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe (e.g., M15)")
    parser.add_argument("--agent_type", type=str, required=True, choices=AGENT_REGISTRY.keys())
    parser.add_argument("--hparam_source", type=str, default="yaml", choices=["yaml", "config", "file"],
                        help="Source of hyperparameters: yaml, config, or file")

    args = parser.parse_args()

    try:
        result = train_agent_worker(
            symbol=args.symbol,
            timeframe=args.timeframe,
            agent_type=args.agent_type,
            hparam_source=args.hparam_source
        )
        print(result)
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")



if __name__ == "__main__":
    main()