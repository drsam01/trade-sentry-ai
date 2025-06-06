import pandas as pd
from typing import Optional, Tuple, Dict
from datetime import datetime


def detect_market_structure_htf(df: pd.DataFrame, lookback: int = 100) -> str:
    """
    Determine recent market structure using swing highs/lows.

    Args:
        df (pd.DataFrame): OHLC dataframe.
        lookback (int): Number of candles to analyze.

    Returns:
        str: 'bullish', 'bearish', or 'sideways'
    """
    df = df[-lookback:]

    highs = df['high'].rolling(3).apply(lambda x: x[1] > x[0] and x[1] > x[2], raw=True)
    lows = df['low'].rolling(3).apply(lambda x: x[1] < x[0] and x[1] < x[2], raw=True)

    swing_highs = df['high'][highs == 1].values
    swing_lows = df['low'][lows == 1].values

    if len(swing_highs) >= 2 and all(x < y for x, y in zip(swing_highs, swing_highs[1:])):
        return "bullish"
    elif len(swing_lows) >= 2 and all(x > y for x, y in zip(swing_lows, swing_lows[1:])):
        return "bearish"
    else:
        return "sideways"


def detect_order_blocks(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 50,
    volume_threshold: Optional[float] = None
) -> Optional[Tuple[int, float, float]]:
    """
    Detect OB as last bullish/bearish candle before impulse breaking structure.

    Args:
        df (pd.DataFrame): OHLC dataframe.
        direction (str): 'bullish' or 'bearish'
        lookback (int): Number of bars to scan.
        volume_threshold (Optional[float]): Minimum volume.

    Returns:
        Optional[Tuple[int, float, float]]: (index, high, low)
    """
    df = df[-lookback:].reset_index(drop=True)

    for i in range(len(df) - 4, 3, -1):
        ob = df.loc[i]
        n1 = df.loc[i + 1]
        n2 = df.loc[i + 2]

        if volume_threshold and ob["volume"] < volume_threshold: # type: ignore
            continue

        if direction == "bearish":
            if ob["close"] > ob["open"]:  # type: ignore # bullish candle
                swing_low = df.loc[i - 3:i, "low"].min()
                if n1["low"] < swing_low or n2["low"] < swing_low:
                    return (i, ob["high"], ob["low"]) # type: ignore

        elif direction == "bullish":
            if ob["close"] < ob["open"]:  # type: ignore # bearish candle
                swing_high = df.loc[i - 3:i, "high"].max()
                if n1["high"] > swing_high or n2["high"] > swing_high:
                    return (i, ob["high"], ob["low"]) # type: ignore

    return None


def detect_break_of_structure(df: pd.DataFrame, direction: str) -> Optional[float]:
    """
    Identify BOS level using recent swing highs/lows.

    Args:
        df (pd.DataFrame): OHLC dataframe.
        direction (str): 'bullish' or 'bearish'

    Returns:
        float: BOS price level.
    """
    highs = df['high'].rolling(3).apply(lambda x: x[1] > x[0] and x[1] > x[2], raw=True)
    lows = df['low'].rolling(3).apply(lambda x: x[1] < x[0] and x[1] < x[2], raw=True)

    if direction == "bullish":
        swing_highs = df[highs == 1]
        if len(swing_highs) >= 2:
            prev = swing_highs['high'].iloc[-2]
            curr = swing_highs['high'].iloc[-1]
            if curr > prev:
                return curr
    elif direction == "bearish":
        swing_lows = df[lows == 1]
        if len(swing_lows) >= 2:
            prev = swing_lows['low'].iloc[-2]
            curr = swing_lows['low'].iloc[-1]
            if curr < prev:
                return curr

    return None


def detect_fvg(df: pd.DataFrame, direction: str, window: int = 20) -> Optional[Tuple[int, float, float]]:
    """
    Detect Fair Value Gap from price inefficiencies.

    Args:
        df (pd.DataFrame): OHLC dataframe.
        direction (str): 'bullish' or 'bearish'
        window (int): Number of bars to scan

    Returns:
        Optional[Tuple[int, float, float]]: (index, gap_high, gap_low)
    """
    for i in range(len(df) - window, len(df) - 2):
        if direction == "bullish":
            if df.loc[i + 2, "low"] > df.loc[i, "high"]: # type: ignore
                return (i + 1, df.loc[i + 2, "low"], df.loc[i, "high"]) # type: ignore
        elif direction == "bearish":
            if df.loc[i + 2, "high"] < df.loc[i, "low"]: # type: ignore
                return (i + 1, df.loc[i, "low"], df.loc[i + 2, "high"]) # type: ignore
    return None


def detect_internal_liquidity_grab(
    ob: Tuple[float, float],
    bos: float,
    df: pd.DataFrame,
    direction: str
) -> Optional[Tuple[float, float]]:
    """
    Detect liquidity grab between OB and BOS wick zone.

    Args:
        ob (Tuple[float, float]): OB high/low
        bos (float): Break of structure price
        df (pd.DataFrame): OHLC dataframe
        direction (str): 'bullish' or 'bearish'

    Returns:
        Optional[Tuple[float, float]]: (zone_low, zone_high)
    """
    if direction == "bullish":
        mask = (df['low'] >= ob[1]) & (df['low'] <= bos)
        wick = None if df[mask].empty else df[mask]['low'].min()
        return (wick, bos) if wick else None
    elif direction == "bearish":
        mask = (df['high'] <= ob[1]) & (df['high'] >= bos)
        wick = None if df[mask].empty else df[mask]['high'].max()
        return (bos, wick) if wick else None
    return None


def _within_session_filter(timestamp: pd.Timestamp, start_hour: int, end_hour: int) -> bool:
    """
    Check if timestamp is within session time window (UTC).

    Args:
        timestamp (pd.Timestamp): Timestamp
        start_hour (int): Start of session (0-23)
        end_hour (int): End of session (0-23)

    Returns:
        bool: True if within window
    """
    hour = timestamp.hour
    return start_hour <= hour <= end_hour


def generate_trade_setups(
    df: pd.DataFrame,
    market_structure: str,
    session_filter: Optional[Tuple[int, int]] = None,
    volume_threshold: Optional[float] = None
) -> Optional[Dict]:
    """
    Combine conditions to detect trade setup.

    Args:
        df (pd.DataFrame): OHLC dataframe.
        market_structure (str): 'bullish' or 'bearish'
        session_filter (Tuple[int, int]): Hour range (UTC)
        volume_threshold (float): Optional filter

    Returns:
        Optional[Dict]: Trade setup
    """
    # Session filter (last bar timestamp)
    if session_filter:
        timestamp = df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else pd.to_datetime(df["timestamp"].iloc[-1])
        if not _within_session_filter(timestamp, *session_filter):
            return None

    ob_data = detect_order_blocks(df, market_structure, volume_threshold=volume_threshold)
    fvg_data = detect_fvg(df, market_structure)
    bos_price = detect_break_of_structure(df, market_structure)

    if not ob_data or not bos_price or not fvg_data:
        return None

    ob_idx, ob_high, ob_low = ob_data
    grab_zone = detect_internal_liquidity_grab((ob_high, ob_low), bos_price, df, market_structure)

    if grab_zone:
        return {
            "direction": market_structure,
            "ob": (ob_high, ob_low),
            "ob_index": ob_idx,
            "bos": bos_price,
            "fvg": fvg_data,
            "liquidity_grab": grab_zone
        }
    return None
