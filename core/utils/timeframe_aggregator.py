# File: core/utils/timeframe_aggregator.py

import pandas as pd
from datetime import datetime
from collections import deque
from typing import Dict, List, Any


class TimeframeAggregator:
    """
    Aggregates 1-minute OHLCV bars into one or multiple target timeframes.

    Supports:
    - Single timeframe aggregation (e.g. ETF: 5min)
    - Multi-timeframe aggregation (e.g. CTF=15min, HTF=60min)
    
    Attributes:
        timeframes (List[int]): List of timeframes to aggregate (e.g., [15, 60]).
        buffers (Dict[int, deque]): Buffer of raw M1 bars for each timeframe.
    
    USAGE:
        aggregator = TimeframeAggregator([15, 60])  # CTF = 15, HTF = 60
        for bar in m1_bars:
            result = aggregator.add_candle(bar)
            ctf_bars = result.get(15, [])
            htf_bars = result.get(60, [])
    """

    def __init__(self, timeframes: List[int]) -> None:
        """
        Initialize the aggregator for the specified timeframes.

        Args:
            timeframes (List[int]): Timeframes to support (e.g., [5], [15, 60])
        """
        self.timeframes = sorted(set(timeframes))
        self.buffers: Dict[int, deque] = {tf: deque() for tf in self.timeframes}
        self.last_aggregated_time: Dict[int, datetime] = {}

    def _is_boundary(self, candle_time: datetime, tf_minutes: int) -> bool:
        """
        Check if a timestamp is aligned to the exact boundary for a given timeframe.

        Args:
            candle_time (datetime): The time of the incoming M1 candle.
            tf_minutes (int): Timeframe to check (e.g., 5, 60).

        Returns:
            bool: True if this time is a timeframe boundary.
        """
        return (
            candle_time.minute % tf_minutes == 0 and 
            candle_time.second == 0
        )
    
    def add_candle(self, bar: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Ingest a new M1 candle and emit completed aggregates per timeframe.

        Args:
            bar (Dict[str, Any]): M1 OHLCV bar with keys:
                'time', 'open', 'high', 'low', 'close', 'tick_volume'

        Returns:
            Dict[int, List[Dict[str, Any]]]: Aggregated bars per timeframe.
        """
        results: Dict[int, List[Dict[str, Any]]] = {}
        timestamp = pd.to_datetime(bar["time"])

        for tf in self.timeframes:
            self.buffers[tf].append(bar)

            if self._is_boundary(timestamp, tf) and len(self.buffers[tf]) >= tf:
                segment = list(self.buffers[tf])[:tf]
                agg_bar = self._aggregate(segment)
                agg_bar["time"] = segment[0]["time"]
                results.setdefault(tf, []).append(agg_bar)

                for _ in range(tf):
                    self.buffers[tf].popleft()

        return results

    def _aggregate(self, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine a list of M1 bars into one higher timeframe bar.

        Args:
            bars (List[Dict[str, Any]]): M1 bars.

        Returns:
            Dict[str, Any]: Aggregated OHLCV bar.
        """
        return {
            "open": bars[0]["open"],
            "high": max(bar["high"] for bar in bars),
            "low": min(bar["low"] for bar in bars),
            "close": bars[-1]["close"],
            "tick_volume": sum(bar.get("tick_volume", bar.get("volume", 0)) for bar in bars)
        }
