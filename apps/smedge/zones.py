# File: apps/smedge/zones.py

import pandas as pd
from typing import List, Dict


def detect_zones(df: pd.DataFrame, max_age: int = 10) -> List[Dict]: # type: ignore
    """
    Detects Order Blocks and Fair Value Gaps from recent OHLCV data.

    Args:
        df: DataFrame with OHLCV data.
        max_age: How far back to look for unmitigated zones.

    Returns:
        List of zone dictionaries.
    """
    zones: List[Dict] = [] # type: ignore

    if len(df) < 10:
        return zones # type: ignore

    for i in range(len(df) - max_age, len(df) - 1):
        row = df.iloc[i] # type: ignore
        next_row = df.iloc[i + 1] # type: ignore

        # Simplified Bullish OB: last down candle followed by strong up
        if row["close"] < row["open"] and next_row["close"] > next_row["open"]:
            zone = { # type: ignore
                "type": "OB",
                "direction": "bullish",
                "low": row["low"],
                "high": row["high"],
                "index": i
            }
            zones.append(zone) # type: ignore

        # Simplified Bearish OB: last up candle followed by strong down
        if row["close"] > row["open"] and next_row["close"] < next_row["open"]:
            zone = { # type: ignore
                "type": "OB",
                "direction": "bearish",
                "low": row["low"],
                "high": row["high"],
                "index": i
            }
            zones.append(zone) # type: ignore

        # Detect simple FVG: Gap between candles
        if i + 2 < len(df):
            r1 = df.iloc[i] # type: ignore
            r2 = df.iloc[i + 1] # type: ignore
            r3 = df.iloc[i + 2] # type: ignore

            # Bullish FVG: r3.low > r1.high
            if r3["low"] > r1["high"]:
                zone = { # type: ignore
                    "type": "FVG",
                    "direction": "bullish",
                    "low": r1["high"],
                    "high": r3["low"],
                    "index": i
                }
                zones.append(zone) # type: ignore

            # Bearish FVG: r3.high < r1.low
            if r3["high"] < r1["low"]:
                zone = { # type: ignore
                    "type": "FVG",
                    "direction": "bearish",
                    "low": r3["high"],
                    "high": r1["low"],
                    "index": i
                }
                zones.append(zone) # type: ignore

    return zones # type: ignore
