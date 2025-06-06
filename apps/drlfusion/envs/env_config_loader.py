# File: apps/drlfusion/envs/env_config_loader.py

import yaml
import os
from typing import Dict, Any, Optional
from core.utils.config_loader import load_config

def load_env_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads and parses the YAML config for the simulation environment.

    Args:
        config_path (str): Optional path override for the config file.

    Returns:
        dict: Parsed config dictionary.
    """
    if config_path is None:
        base = os.path.dirname(__file__)
        config_path = os.path.join(base, "env_config.yaml")

    return load_config(config_path)
