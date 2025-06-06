# File: apps/smedge/strategy.py

import pandas as pd
from typing import Dict, Any, Optional

from apps.smedge.zones import detect_zones # type: ignore
from apps.smedge.signals import detect_choch_breakout # type: ignore
from core.utils.regime_filter import RegimeFilter
from core.utils.logger import get_logger

logger = get_logger(__name__)


class SMEdgeStrategy:
    """
    Core Smart Money Concept (SMC) strategy logic:
    - Detect OB/FVG zones
    - Confirm valid entry signals
    - Filter with trend/volatility regime logic
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_zone_age = config["strategy"].get("max_zone_age", 10)
        self.confidence_threshold = config["strategy"].get("confidence_threshold", 0.6)
        self.regime_filter = RegimeFilter(config.get("regime_thresholds"))

    def get_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Core method to produce a live trading signal from market data.

        Args:
            df: OHLCV DataFrame with indicators (ATRRatio, AroonUp, AroonDown, etc.)

        Returns:
            A signal dictionary or None if no valid trade detected.
        """
        try:
            if df.empty or len(df) < 50:
                return None

            zones = detect_zones(df, max_age=self.max_zone_age) # type: ignore
            if not zones:
                logger.debug("No valid zones detected.")
                return None

            signal = detect_choch_breakout(df, zones) # type: ignore
            if not signal or signal.get("action") not in {"buy", "sell"}: # type: ignore
                return None

            action = signal["action"] # type: ignore
            atr = df["ATR"].iloc[-1] # type: ignore
            row = df.iloc[-1] # type: ignore

            filtered_action = self.regime_filter.apply_filter( # type: ignore
                intended_action=-1 if action == "sell" else 1,
                row=row
            )

            if filtered_action == 0:
                return None  # Action filtered by regime

            return {
                "action": action,
                "atr": atr,
                "confidence": signal.get("confidence", 1.0), # type: ignore
                "zone": signal.get("zone", {}), # type: ignore
                "timestamp": row["timestamp"]
            }

        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None
