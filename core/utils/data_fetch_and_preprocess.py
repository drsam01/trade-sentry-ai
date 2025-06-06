# File: drlfusion/data/data_fetch_and_preprocess.py

import ray
import logging
import pandas as pd
from typing import Optional
from drlfusion.modelling.features.data_preprocessing import DataPreprocessor
from drlfusion.data.data_download import MT5DataDownload
from drlfusion.core.utils.logger import get_logger

# Initialize module-level logger
logger = get_logger("data_pipeline")


def _attempt_latest_fetch(downloader: MT5DataDownload, symbol: str, timeframe: str, initial_bars: int) -> pd.DataFrame:
    """
    Attempt to fetch the latest data from MT5, with retries and fallback to smaller bar counts on failure.

    Args:
        downloader (MT5DataDownload): The data download utility instance.
        symbol (str): Trading symbol (e.g., 'EURUSD').
        timeframe (str): Timeframe string (e.g., 'M15', 'H1').
        initial_bars (int): Number of bars to initially attempt.

    Returns:
        pd.DataFrame: Retrieved data.

    Raises:
        RuntimeError: If all retries fail to fetch valid data.
    """
    bars = initial_bars
    for retry in range(3):
        try:
            df = downloader.get_latest_data(symbol=symbol, timeframe=timeframe, lookback_bars=bars)
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.info(f"✅ Fetched {len(df)} latest bars for {symbol}-{timeframe} using bars={bars}")
                return df
            else:
                logger.warning(f"⚠️ No data returned for {symbol}-{timeframe} with bars={bars}. Retrying...")
                bars = max(100, bars // 2)
        except Exception as e:
            logger.warning(f"⚠️ Attempt {retry + 1} failed for {symbol}-{timeframe}: {e}")
            bars = max(100, bars // 2)

    raise RuntimeError(f"❌ All latest data fetch attempts failed for {symbol}-{timeframe}.")


def _attempt_historical_fetch(downloader: MT5DataDownload, symbol: str, timeframe: str, initial_bars: int) -> pd.DataFrame:
    """
    Attempt to fetch historical data from MT5 with retry logic.

    Args:
        downloader (MT5DataDownload): The data download utility instance.
        symbol (str): Trading symbol (e.g., 'EURUSD').
        timeframe (str): Timeframe string (e.g., 'M15', 'H1').
        initial_bars (int): Number of bars to initially attempt.

    Returns:
        pd.DataFrame: Retrieved data.

    Raises:
        RuntimeError: If all retries fail to fetch valid data.
    """
    bars = initial_bars
    for retry in range(3):
        try:
            df = downloader.get_historical_data(symbol=symbol, timeframe=timeframe, lookback_bars=bars)
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.info(f"✅ Fetched {len(df)} historical bars for {symbol}-{timeframe} using bars={bars}")
                return df
            else:
                logger.warning(f"⚠️ No data returned for {symbol}-{timeframe} with bars={bars}. Retrying...")
                bars = max(100, bars // 2)
        except Exception as e:
            logger.warning(f"⚠️ Attempt {retry + 1} failed for {symbol}-{timeframe}: {e}")
            bars = max(100, bars // 2)

    raise RuntimeError(f"❌ All historical data fetch attempts failed for {symbol}-{timeframe}.")


@ray.remote
def fetch_and_preprocess(symbol: str, timeframe: str, lookback_bars: int = 5000, use_latest: Optional[bool] = True) -> None:
    """
    Ray-executed task to fetch and preprocess trading data for a symbol-timeframe pair.

    This function:
    - Fetches latest or historical bars from MT5
    - Saves raw data to disk
    - Processes and assembles features
    - Splits and saves the dataset into train/test partitions

    Args:
        symbol (str): Trading symbol (e.g., 'EURUSD').
        timeframe (str): Timeframe string (e.g., 'M15', 'H1').
        lookback_bars (int): Number of bars to fetch.
        use_latest (bool, optional): Whether to fetch the most recent live bars (True) or a fixed historical window (False).

    Raises:
        Exception: Re-raised for Ray error reporting and external handling.
    """
    downloader = MT5DataDownload()

    try:
        logger.info(f"[START] Fetching {symbol}-{timeframe} | use_latest={use_latest}")

        # Fetch data depending on mode
        if use_latest:
            df = _attempt_latest_fetch(downloader, symbol, timeframe, initial_bars=lookback_bars)
        else:
            df = _attempt_historical_fetch(downloader, symbol, timeframe, initial_bars=lookback_bars)

        # Save raw historical data to disk
        downloader.save_historical_data(df, symbol, timeframe)

        # Preprocess and engineer features
        preprocessor = DataPreprocessor(symbol=symbol, timeframe=timeframe)
        preprocessor.load_data()
        preprocessor.assemble_features()
        preprocessor.save_data_splits()

        logger.info(f"[COMPLETE] ✅ {symbol}-{timeframe} preprocessing succeeded.")

    except Exception as e:
        logger.error(f"[FAIL] ❌ {symbol}-{timeframe} encountered an error: {e}")
        raise
