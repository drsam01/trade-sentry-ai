import pandas as pd
from typing import Dict, Any, Optional

from core.strategy.base_strategy import BaseStrategy
from apps.smclassic.helpers import (
    detect_market_structure_htf,
    generate_trade_setups,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)


class SMClassicStrategy(BaseStrategy):
    """
    SMClassic Strategy:
    - Identifies HTF market structure (bullish, bearish, sideways)
    - Detects OB, BOS, FVG, and internal liquidity grab
    - Places limit orders with SL and TP based on structure
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.df: Optional[pd.DataFrame] = None
        self.last_signal: Optional[Dict[str, Any]] = None
        self.pending_order: Optional[Dict[str, Any]] = None
        self.market_structure: str = "sideways"

    def load_data(self, df: pd.DataFrame) -> None:
        """Load HTF data and detect market structure."""
        self.df = df.copy()
        self.market_structure = detect_market_structure_htf(df, lookback=100)

    def generate_signal(self, current_index: int, injected_lot: Optional[float]) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on market structure and trade confirmation.

        Args:
            current_index (int): Latest row index in dataframe.
            injected_lot (Optional[float]): Custom lot override.

        Returns:
            Optional[Dict[str, Any]]: Trade signal or None.
        """
        if self.df is None or current_index < 20:
            return None

        symbol_cfg = self.config.get("strategy", {})
        pip_value = self.config.get("risk", {}).get("pip_value", 0.01)
        sl_buffer = symbol_cfg.get("sl_buffer_pips", 5) * pip_value if pip_value else 0.0005

        sub_df = self.df.iloc[: current_index + 1].copy()
        setup = generate_trade_setups(sub_df, self.market_structure)

        if not setup:
            return None

        ob_high, ob_low = setup["ob"]
        direction = setup["direction"]

        # Find recent swing high or low
        tp = self._find_recent_swing(sub_df, direction)
        if tp is None:
            logger.warning(f"No valid TP swing found for {direction} setup.")
            return None

        if direction == "bullish":
            entry = ob_high
            sl = ob_low - sl_buffer
        else:
            entry = ob_low
            sl = ob_high + sl_buffer

        if injected_lot is None:
            return {
                "type": "limit",
                "direction": direction,
                "price": entry,
                "sl": sl,
                "tp": tp,
                "tag": "smclassic_entry"
            }
        
        signal = {
            "type": "limit",
            "direction": direction,
            "lot": injected_lot,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "tag": "smclassic_entry"
        }

        self.last_signal = signal
        return signal

    def finalize_signal(self, signal: Dict[str, Any], lot: float) -> Dict[str, Any]:
        signal["lot"] = lot
        self.last_signal = signal
        return signal

    def generate_signals(self) -> list[Dict[str, Any]]:
        """Backtest mode: generate signal list across history."""
        if self.df is None:
            return []

        signals = []
        for idx in range(len(self.df)):
            signal = self.generate_signal(idx, injected_lot=0.01)
            signals.append(signal or {})
        return signals

    def execute_trade(self, signal: Dict[str, Any]) -> None:
        """Store the signal as pending trade."""
        self.pending_order = signal
        logger.info(f"[SMClassic] Executed trade at {signal['price']} with SL {signal['sl']} and TP {signal['tp']}")

    async def log_performance(self, trade_tracker, symbol: str) -> None:
        """Log performance summary using tracker."""
        summary = trade_tracker.get_closed_summary(symbol)
        logger.info(f"[{symbol}] Strategy Performance: {summary}")

    def reset_state(self) -> None:
        """Reset strategy internal state."""
        self.last_signal = None
        self.pending_order = None
        self.market_structure = "sideways"
        logger.info("[SMClassic] Strategy state reset.")
    
    def _find_recent_swing(self, df: pd.DataFrame, direction: str, window: int = 20) -> Optional[float]:
        if len(df) < 3:
            return None  # Not enough data to detect swing

        recent_df = df.tail(window + 2).copy()  # extra rows to support rolling window
        highs = recent_df['high'].rolling(3).apply(
            lambda x: x[1] > x[0] and x[1] > x[2] if len(x) == 3 else False,
            raw=True
        )
        lows = recent_df['low'].rolling(3).apply(
            lambda x: x[1] < x[0] and x[1] < x[2] if len(x) == 3 else False,
            raw=True
        )

        if direction == "bullish":
            swing_highs = recent_df[highs == 1]
            return None if swing_highs.empty else swing_highs['high'].iloc[-1]
        elif direction == "bearish":
            swing_lows = recent_df[lows == 1]
            return None if swing_lows.empty else swing_lows['low'].iloc[-1]
        return None
