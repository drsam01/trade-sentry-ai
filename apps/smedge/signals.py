# File: apps/smedge/signals.py

import pandas as pd
from typing import Dict, List, Optional


def detect_choch_breakout(df: pd.DataFrame, zones: List[Dict]) -> Optional[Dict]: # type: ignore
    """
    Confirms if a recent price movement triggered a CHoCH breakout within a zone.

    Args:
        df: OHLCV DataFrame.
        zones: List of OB/FVG zones.

    Returns:
        A signal dictionary with 'action', 'zone', and 'confidence' or None.
    """
    if len(df) < 5:
        return None

    last_candle = df.iloc[-1] # type: ignore
    prev_high = df["high"].iloc[-5:-1].max() # type: ignore
    prev_low = df["low"].iloc[-5:-1].min() # type: ignore

    for zone in reversed(zones): # type: ignore
        if zone["direction"] == "bullish" and last_candle["close"] > zone["high"] and last_candle["low"] > prev_low:
            return {
                "action": "buy",
                "zone": zone,
                "confidence": 0.8
            } # type: ignore
        if zone["direction"] == "bearish" and last_candle["close"] < zone["low"] and last_candle["high"] < prev_high:
            return {
                "action": "sell",
                "zone": zone,
                "confidence": 0.8
            } # type: ignore

    return None
