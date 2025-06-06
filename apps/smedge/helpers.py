# File: apps/smedge/helpers.py

import pandas as pd
import numpy as np
from typing import List, Dict, Literal, Optional, Any
from core.utils.indicators import compute_indicators, detect_trend
from core.utils.timeframe_aggregator import TimeframeAggregator

class SMEdgeHelper:
    """
    Helper class for SMEdge strategy operations, including trend detection,
    zone identification (OB/FVG), and valid entry confirmation.
    """

    def __init__(self, ctf_minutes: int, htf_minutes: int) -> None:
        """
        Initialize with desired confirmation and higher timeframes.

        Args:
            ctf_minutes (int): Confirmation timeframe in minutes (e.g., 5 or 15).
            htf_minutes (int): Higher timeframe in minutes (e.g., 60).
        """
        self.ctf_minutes = ctf_minutes
        self.htf_minutes = htf_minutes
        self.aggregator = TimeframeAggregator([ctf_minutes, htf_minutes])

    def aggregate_candles(self, m1_bars: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Aggregate M1 candles into CTF and HTF DataFrames.

        Args:
            m1_bars (List[Dict]): List of M1 OHLCV bars.

        Returns:
            Dict[str, pd.DataFrame]: Keys are "ctf" and "htf" with their respective DataFrames.
        """
        ctf_bars, htf_bars = [], []
        for bar in m1_bars:
            agg_result = self.aggregator.add_candle(bar)
            ctf_bars.extend(agg_result.get(self.ctf_minutes, []))
            htf_bars.extend(agg_result.get(self.htf_minutes, []))

        return {
            "ctf": pd.DataFrame(ctf_bars),
            "htf": pd.DataFrame(htf_bars),
        }

    def detect_trend(self, df_htf: pd.DataFrame) -> Literal['UP', 'DOWN', 'NEUTRAL']:
        """
        Detect trend using HTF pivot structure + indicators.

        Args:
            df_htf (pd.DataFrame): Higher timeframe OHLCV data.

        Returns:
            Literal['UP', 'DOWN', 'NEUTRAL']: Detected trend direction.
        """
        df_indicators = compute_indicators(df_htf)
        return detect_trend(df_indicators)

    def detect_zones(self, df: pd.DataFrame, trend: str) -> List[Dict[str, Any]]:
        """
        Identify Order Blocks and FVGs and annotate them with trend alignment.

        Args:
            df (pd.DataFrame): HTF data.
            trend (str): Current trend direction.

        Returns:
            List[Dict[str, Any]]: List of zones with `trend_aligned` key.
        """
        zones = []
        for i in range(2, len(df)):
            bar = df.iloc[i]
            prev_bar = df.iloc[i - 1]

            # OB Detection
            if prev_bar["close"] < prev_bar["open"] and bar["close"] > bar["open"]:
                direction = "buy"
                trend_aligned = trend == "UP"
                zones.append({
                    "type": "ob",
                    "direction": direction,
                    "low": prev_bar["low"],
                    "high": prev_bar["high"],
                    "index": i,
                    "mitigated": False,
                    "trend_aligned": trend_aligned,
                })
            elif prev_bar["close"] > prev_bar["open"] and bar["close"] < bar["open"]:
                direction = "sell"
                trend_aligned = trend == "DOWN"
                zones.append({
                    "type": "ob",
                    "direction": direction,
                    "low": prev_bar["low"],
                    "high": prev_bar["high"],
                    "index": i,
                    "mitigated": False,
                    "trend_aligned": trend_aligned,
                })

            # FVG Detection
            if i >= 2:
                # Bullish FVG
                gap_low = df.iloc[i - 1]["high"]
                gap_high = df.iloc[i]["low"]
                if gap_high > gap_low:
                    trend_aligned = trend == "UP"
                    zones.append({
                        "type": "fvg",
                        "direction": "buy",
                        "low": gap_low,
                        "high": gap_high,
                        "index": i,
                        "mitigated": False,
                        "trend_aligned": trend_aligned,
                    })
                # Bearish FVG
                gap_low = df.iloc[i]["high"]
                gap_high = df.iloc[i - 1]["low"]
                if gap_high < gap_low:
                    trend_aligned = trend == "DOWN"
                    zones.append({
                        "type": "fvg",
                        "direction": "sell",
                        "low": gap_high,
                        "high": gap_low,
                        "index": i,
                        "mitigated": False,
                        "trend_aligned": trend_aligned,
                    })

        return zones

    def confirm_choch_breakout(
        self,
        df_ctf: pd.DataFrame,
        zone: Dict[str, Any],
        trend: str
    ) -> Optional[int]:
        """
        Confirm a CHoCH (Change of Character) breakout and retest into the zone.

        Args:
            df_ctf (pd.DataFrame): CTF OHLCV candles.
            zone (Dict[str, Any]): The candidate OB or FVG zone.
            trend (str): Current trend direction.

        Returns:
            Optional[int]: Index of valid entry in df_ctf, or None if no confirmation.
        """
        direction = zone["direction"]
        zone_low, zone_high = zone["low"], zone["high"]

        for i in range(len(df_ctf)):
            bar = df_ctf.iloc[i]
            # Confirm price enters zone and breaks in direction
            if direction == "buy":
                if bar["low"] <= zone_high and bar["close"] > bar["open"]:
                    return i
            elif direction == "sell":
                if bar["high"] >= zone_low and bar["close"] < bar["open"]:
                    return i

        return None
