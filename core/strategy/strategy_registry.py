# File: core/strategy/strategy_registry.py

import importlib
from core.strategy.base_strategy import BaseStrategy
from typing import Dict, Type


class StrategyRegistry:
    """
    Registry for loading trading strategies dynamically by name.
    Supports pluggable architecture via dynamic imports and class mapping.
    """

    _registry: Dict[str, str] = {
        "drlfusion": "apps.drlfusion.strategy",
        "smclassic": "apps.smclassic.strategy",
        "smedge": "apps.smedge.strategy",
        "smgrid": "apps.smgrid.grid_trader",
        "breakout": "apps.breakout.strategy"
    }

    _class_map: Dict[str, str] = {
        "drlfusion": "DRLFusionStrategy",
        "smclassic": "SMClassicStrategy",
        "smedge": "SMEdgeStrategy",
        "smgrid": "SMGridStrategy",
        "breakout": "BreakoutStrategy"
    }

    @classmethod
    def get_strategy(cls, name: str, config: Dict) -> BaseStrategy:
        """
        Load and instantiate the strategy class by strategy name.

        Args:
            name (str): Strategy alias (e.g., 'drlfusion')
            config (Dict): Config dictionary to initialize the strategy

        Returns:
            BaseStrategy: Instantiated strategy class
        """
        if name not in cls._registry:
            raise ValueError(f"Strategy '{name}' not found in registry.")

        module_path = cls._registry[name]
        class_name = cls._class_map.get(name, "Strategy")  # fallback

        module = importlib.import_module(module_path)
        strategy_class: Type[BaseStrategy] = getattr(module, class_name)
        return strategy_class(config)
