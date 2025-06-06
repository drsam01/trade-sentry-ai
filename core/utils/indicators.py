# File: core/utils/indicators.py

import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Literal


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR).

    Args:
        df (pd.DataFrame): OHLCV data with columns: open, high, low, close.

    Returns:
        float: ATR value.
    """
    high_low = df["high"].to_numpy() - df["low"].to_numpy()
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1) # type: ignore
    atr = tr.rolling(window=period).mean().iloc[-1]
    return round(float(atr), 5)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators including MACD, RSI, Aroon, and pivots.

    Args:
        df (pd.DataFrame): OHLCV data with 'high', 'low', 'close'.

    Returns:
        pd.DataFrame: DataFrame with added indicator columns.
    """
    df = df.copy()

    # === RSI ===
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # === MACD ===
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # === Aroon ===
    aroon_len = 14
    aroon_up = []
    aroon_down = []

    for i in range(len(df)):
        if i < aroon_len:
            aroon_up.append(np.nan)
            aroon_down.append(np.nan)
            continue

        high_idx = df["high"].iloc[i - aroon_len + 1:i + 1].idxmax()
        low_idx = df["low"].iloc[i - aroon_len + 1:i + 1].idxmin()
        aroon_up.append(100 * (aroon_len - (i - high_idx)) / aroon_len) # type: ignore
        aroon_down.append(100 * (aroon_len - (i - low_idx)) / aroon_len) # type: ignore

    df["aroon_up"] = aroon_up
    df["aroon_down"] = aroon_down
    
    # === ADX (Average Directional Index) ===
    adx_len = 14
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0 # type: ignore
    minus_dm[minus_dm > 0] = 0 # type: ignore

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(adx_len).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/adx_len).mean() / atr)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1/adx_len).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["adx"] = dx.ewm(alpha=1/adx_len).mean()

    # === Pivot Points ===
    df["pivot_high"] = df["high"][(df["high"].shift(1) < df["high"]) & (df["high"].shift(-1) < df["high"])]
    df["pivot_low"] = df["low"][(df["low"].shift(1) > df["low"]) & (df["low"].shift(-1) > df["low"])]

    return df

def determine_market_regime(adx_val: float) -> Literal['trending', 'ranging']:
    """
    Classify market regime based on ADX value.

    Args:
        adx_val (float): Latest ADX value.

    Returns:
        str: One of "trending" or "ranging"
    """
    return "trending" if adx_val >= 20 else "ranging"

def detect_trend(df: pd.DataFrame, verbose: bool = False) -> Literal['UP', 'DOWN', 'NEUTRAL']:
    """
    Determine trend using pivot slope and weighted confirmation from MACD, RSI, Aroon.
    Automatically adjusts weights based on ADX-based market regime.
    """
    try:
        # Pivots
        pivots_high = df[df["pivot_high"].notna()].tail(5)
        pivots_low = df[df["pivot_low"].notna()].tail(5)
        pivot_trend = "NEUTRAL"
        high_slope = low_slope = 0.0

        if len(pivots_high) >= 3 and len(pivots_low) >= 3:
            high_slope = linregress(range(len(pivots_high)), pivots_high["pivot_high"]).slope # type: ignore
            low_slope = linregress(range(len(pivots_low)), pivots_low["pivot_low"]).slope # type: ignore
            if high_slope > 0 and low_slope > 0:
                pivot_trend = "UP"
            elif high_slope < 0 and low_slope < 0:
                pivot_trend = "DOWN"

        # Indicators
        latest = df.iloc[-1]
        adx_value = latest["adx"]
        regime = determine_market_regime(adx_value)

        macd_up = latest["macd"] > latest["macd_signal"]
        macd_down = latest["macd"] < latest["macd_signal"]
        rsi_up = latest["rsi"] > 55
        rsi_down = latest["rsi"] < 45
        aroon_diff = abs(latest["aroon_up"] - latest["aroon_down"])
        aroon_up = latest["aroon_up"] > 70 and aroon_diff > 40
        aroon_down = latest["aroon_down"] > 70 and aroon_diff > 40

        # Regime-based weights
        if regime == "trending":
            weights = {"macd": 1.2, "rsi": 0.4, "aroon": 1.2}
        else:
            weights = {"macd": 0.8, "rsi": 1.2, "aroon": 0.6}

        up_score = (
            weights["macd"] * int(macd_up) +
            weights["rsi"] * int(rsi_up) +
            weights["aroon"] * int(aroon_up)
        )
        down_score = (
            weights["macd"] * int(macd_down) +
            weights["rsi"] * int(rsi_down) +
            weights["aroon"] * int(aroon_down)
        )

        if verbose:
            print(f"[TrendCheck] Regime: {regime}, ADX: {adx_value:.2f}, Pivot: {pivot_trend}")
            print(f"Slopes(H,L): ({high_slope:.4f}, {low_slope:.4f}) | Scores → UP: {up_score:.2f}, DOWN: {down_score:.2f}")

        # Decision logic
        if pivot_trend == "UP" and up_score >= 1.5:
            return "UP"
        elif pivot_trend == "DOWN" and down_score >= 1.5:
            return "DOWN"
        else:
            return "NEUTRAL"

    except Exception as e:
        if verbose:
            print(f"[Trend Detection Error] {e}")
        return "NEUTRAL"