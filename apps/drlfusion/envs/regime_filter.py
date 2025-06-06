# File: apps/drlfusion/envs/regime_filter.py

import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional, Dict


class RegimeFilter:
    """
    A hybrid regime classification engine combining market structure and indicator-based logic.
    Supports regime-aware trading, strategy switching, and DRL agent routing.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        default_thresholds = {
            'ATR_LOW_THRESHOLD': 0.85,
            'ATR_HIGH_THRESHOLD': 1.15,
            'AROON_UP_THRESHOLD': 70,
            'AROON_DOWN_THRESHOLD': 70,
            'TREND_CLARITY_THRESHOLD': 40,
            'MIN_SWING_COUNT': 2,
            'VOLUME_EXPAND_THRESHOLD': 1.15,
            'VOLUME_CONTRACT_THRESHOLD': 0.85,
            'VOLUME_ROLLING_WINDOW': 20
        }
        self.thresholds = thresholds or default_thresholds
        self.__dict__.update(self.thresholds)

    def apply_filter(self, intended_action: int, row: pd.Series) -> int:
        """
        Filter the agent's intended action based on the detected regime.

        Args:
            intended_action (int): [-1, 0, 1] = short, hold, long
            row (pd.Series): Enriched feature row at current step

        Returns:
            int: Modified action if regime advises against current one
        """
        try:
            vol_regime, trend_regime, volume_regime = self.check_conditions(row)

            if trend_regime == 'Sideways' and intended_action != 0:
                return 0
            if trend_regime == 'Uptrend' and intended_action == -1:
                return 0
            if trend_regime == 'Downtrend' and intended_action == 1:
                return 0
            if vol_regime == 'Low' and intended_action == -1:
                return 0
            if vol_regime == 'High' and intended_action == 1:
                return 0
            if volume_regime == 'Contraction' and intended_action != 0:
                return 0

            return intended_action

        except Exception:
            return 0

    def check_conditions(self, row: pd.Series) -> Tuple[str, str, str]:
        """
        Derive all three regimes from the current row of data.

        Args:
            row (pd.Series): Must include ATRRatio, AroonUp, AroonDown, VolumeRatio

        Returns:
            Tuple[str, str, str]: (volatility_regime, trend_regime, volume_regime)
        """
        vol_regime = self._infer_volatility_regime(row)
        trend_regime = self._infer_trend_regime(row)
        volume_regime = self._infer_volume_regime(row)
        return vol_regime, trend_regime, volume_regime

    def _infer_volatility_regime(self, row: pd.Series) -> str:
        atr_ratio = self._safe_extract(row.get("ATRRatio")) # type: ignore
        if atr_ratio < self.ATR_LOW_THRESHOLD: # type: ignore
            return "Low"
        elif atr_ratio > self.ATR_HIGH_THRESHOLD: # type: ignore
            return "High"
        return "Med"

    def _infer_trend_regime(self, row: pd.Series) -> str:
        swing_lows_in_range = row.get("SwingLows_lookback", 0)      # Local bottoms -> potential support
        swing_highs_in_range = row.get("SwingHighs_lookback", 0)    # Local tops -> potential resistance

        aroon_up = self._safe_extract(row.get("AroonUp")) # type: ignore
        aroon_down = self._safe_extract(row.get("AroonDown")) # type: ignore
        clarity = self._safe_extract(row.get("AroonDiff")) # type: ignore
        
        # Price has formed many reaction lows -> healthy pullbacks -> potential uptrend
        if (
            swing_lows_in_range >= self.MIN_SWING_COUNT # type: ignore
            and aroon_up > self.AROON_UP_THRESHOLD # type: ignore
            and clarity >= self.TREND_CLARITY_THRESHOLD # type: ignore
        ):
            return "Uptrend"
        
        # Price has formed many reaction highs -> healthy pullbacks -> potential downtrend
        elif (
            swing_highs_in_range >= self.MIN_SWING_COUNT # type: ignore
            and aroon_down > self.AROON_DOWN_THRESHOLD # type: ignore
            and clarity >= self.TREND_CLARITY_THRESHOLD # type: ignore
        ):
            return "Downtrend"

        return "Sideways"

    def _infer_volume_regime(self, row: pd.Series) -> str:
        volume_ratio = self._safe_extract(row.get("VolumeRatio", 1.0))
        if volume_ratio > self.VOLUME_EXPAND_THRESHOLD: # type: ignore
            return "Expansion"
        elif volume_ratio < self.VOLUME_CONTRACT_THRESHOLD: # type: ignore
            return "Contraction"
        return "Normal"

    def _safe_extract(self, value: Union[float, np.ndarray, pd.Series, list]) -> float:
        """
        Converts any input type to a scalar float safely.
        """
        if isinstance(value, (np.ndarray, pd.Series, list)):
            return float(np.array(value).flatten()[0])
        return float(value)
