# File: apps/smedge/strategy.py

from typing import List, Dict, Optional
import pandas as pd
from core.strategy.base_strategy import BaseStrategy
from apps.smedge.helpers import SMEdgeHelper


class SMEdgeStrategy(BaseStrategy):
    """
    SMEdge Strategy: Momentum-based Smart Money Concept (SMC) strategy.
    Detects trend-aligned Order Blocks (OB) and Fair Value Gaps (FVG), confirms entries with CHoCH-style price action.
    """

    def __init__(self, config: Dict):
        """
        Initialize the strategy with configuration parameters.

        Args:
            config (Dict): Configuration dictionary containing timeframes, thresholds, etc.
        """
        super().__init__(config)
        self.ctf_minutes: int = config["strategy"]["ctf_minutes"]
        self.htf_minutes: int = config["strategy"]["htf_minutes"]
        self.max_zone_age: int = config["strategy"].get("max_zone_age", 20)
        self.helper = SMEdgeHelper(ctf_minutes=self.ctf_minutes, htf_minutes=self.htf_minutes)

    def generate_signals(
        self, df_m1: pd.DataFrame, symbol: str
    ) -> Optional[Dict]:
        """
        Generate a trade signal using multi-timeframe SMC logic.

        Args:
            df_m1 (pd.DataFrame): 1-minute OHLCV candles.
            symbol (str): Instrument being analyzed.

        Returns:
            Optional[Dict]: Trade signal dictionary or None if no valid setup found.
        """
        # Step 1: Aggregate to HTF and CTF from M1
        aggregated = self.helper.aggregate_candles(df_m1.to_dict("records"))
        df_ctf, df_htf = aggregated["ctf"], aggregated["htf"]

        # Step 2: Basic data validation
        if df_htf.shape[0] < 25 or df_ctf.shape[0] < 25:
            return None

        # Step 3: Determine trend from HTF
        trend = self.helper.detect_trend(df_htf)

        # Step 4: Detect OB and FVG zones in HTF
        zones = self.helper.detect_zones(df_htf, trend)

        # Step 5: Iterate through unmitigated, trend-aligned zones
        for zone in reversed(zones[-self.max_zone_age:]):
            if not zone.get("trend_aligned", False):
                continue

            entry_index = self.helper.confirm_choch_breakout(df_ctf, zone, trend)
            if entry_index is None:
                continue

            entry_bar = df_ctf.iloc[entry_index]
            direction = zone["direction"]

            return {
                "symbol": symbol,
                "direction": direction,
                "entry_time": entry_bar["timestamp"],
                "entry_price": entry_bar["close"],
                "zone": zone,
                "strategy": "smedge",
            }

        return None
