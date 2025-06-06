import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List


class FeatureBuilder:
    """
    Extracts and prepares enriched features from raw OHLCV data
    for use in regime detection and DRL trading environments.
    """

    def __init__(self, df: pd.DataFrame, atr_short: int = 5, atr_long: int = 14, aroon_period: int = 25):
        """
        Args:
            df (pd.DataFrame): Raw OHLCV data with columns:
                ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.df = df.copy().reset_index(drop=True)
        self.atr_short = atr_short
        self.atr_long = atr_long
        self.aroon_period = aroon_period
        self.feature_columns: List[str] = []

    def prepare(self) -> pd.DataFrame:
        """
        Prepares and returns the enriched feature set.

        Returns:
            pd.DataFrame: DataFrame with appended features.
        """
        df = self.df

        # === 1. ATR and ATR Ratio ===
        df["ATR5"] = ta.atr(df["High"], df["Low"], df["Close"], length=self.atr_short)
        df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=self.atr_long)
        df["ATRRatio"] = df["ATR5"] / (df["ATR14"] + 1e-6)
        self.feature_columns += ["ATR5", "ATR14", "ATRRatio"]

        # === 2. Aroon Up / Down ===
        aroon = ta.aroon(df["High"], df["Low"], length=self.aroon_period)
        df["AroonUp"] = aroon["AROONU_{self.aroon_period}"] # type: ignore
        df["AroonDown"] = aroon["AROOND_{self.aroon_period}"] # type: ignore
        df["AroonDiff"] = abs(df["AroonUp"] - df["AroonDown"])
        self.feature_columns += ["AroonUp", "AroonDown", "AroonDiff"]

        # === 3. Volume Regime ===
        df["VolumeMA"] = df["Volume"].rolling(20).mean()
        df["VolumeRatio"] = df["Volume"] / (df["VolumeMA"] + 1e-6)
        self.feature_columns += ["Volume", "VolumeMA", "VolumeRatio"]

        # === 4. Swing High / Low Detection ===
        df["SwingHigh"] = df["High"].rolling(3).apply(
            lambda x: float(x[1] > x[0] and x[1] > x[2]), raw=True
        ).shift(-1)
        df["SwingLow"] = df["Low"].rolling(3).apply(
            lambda x: float(x[1] < x[0] and x[1] < x[2]), raw=True
        ).shift(-1)
        df["SwingHighs_lookback"] = df["SwingHigh"].rolling(10).sum()
        df["SwingLows_lookback"] = df["SwingLow"].rolling(10).sum()
        self.feature_columns += ["SwingHigh", "SwingLow", "SwingHighs_lookback", "SwingLows_lookback"]

        # === 5. Donchian Channel Width ===
        donchian_high = df["High"].rolling(20).max()
        donchian_low = df["Low"].rolling(20).min()
        df["DonchianWidth"] = (donchian_high - donchian_low) / (df["Close"] + 1e-6)
        self.feature_columns += ["DonchianWidth"]

        # === 6. Fractal High / Low ===
        df["FractalHigh"] = df["High"].rolling(5).apply(
            lambda x: float(x[2] > x[0] and x[2] > x[1] and x[2] > x[3] and x[2] > x[4]), raw=True
        ).shift(-2)
        df["FractalLow"] = df["Low"].rolling(5).apply(
            lambda x: float(x[2] < x[0] and x[2] < x[1] and x[2] < x[3] and x[2] < x[4]), raw=True
        ).shift(-2)
        self.feature_columns += ["FractalHigh", "FractalLow"]

        # === 7. OBV (On-Balance Volume) ===
        df["OBV"] = ta.obv(df["Close"], df["Volume"])
        df["OBVNormalized"] = (df["OBV"] - df["OBV"].rolling(20).mean()) / (df["OBV"].rolling(20).std() + 1e-6)
        self.feature_columns += ["OBVNormalized"]

        # Final cleanup
        df = df.dropna().reset_index(drop=True)

        return df

    def get_feature_columns(self) -> List[str]:
        """
        Returns list of selected predictive features.

        Returns:
            List[str]: Feature column names
        """
        return self.feature_columns

class RunningNormalizer:
    """
    Online normalizer that updates running mean and std.
    """

    def __init__(self, shape: tuple, epsilon: float = 1e-6):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def normalize(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(-1, len(self.mean))
        self._update(data)
        std = np.sqrt(self.var)
        return (data - self.mean) / (std + 1e-8)

    def _update(self, data: np.ndarray):
        batch_mean = np.mean(data, axis=0)
        batch_var = np.var(data, axis=0)
        batch_count = data.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count
