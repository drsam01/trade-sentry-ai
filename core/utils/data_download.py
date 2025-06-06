# File: drlfusion/data/data_download.py

import MetaTrader5 as mt5
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from functools import wraps
from typing import Callable, Type, Optional

import pytz  # For handling UTC time conversion

from drlfusion.configs.config import Config
from drlfusion.core.utils.logger import get_logger

config = Config()

REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
DEFAULT_LOOKBACK_BARS = 10_000
DEFAULT_SERVER_OFFSET_HOURS = config.MT5_CONFIG_PARAMS.get("SERVER_OFFSET_HOURS", 0)


def retry(
    exceptions: tuple[Type[BaseException]], 
    tries: int = 3, 
    delay: int = 2, 
    backoff: int = 2
) -> Callable:
    """
    Decorator that retries a function on specified exceptions using exponential backoff.

    Args:
        exceptions (tuple): Exception types to catch and retry.
        tries (int): Number of retry attempts.
        delay (int): Initial delay between retries in seconds.
        backoff (int): Multiplier for delay between attempts.

    Returns:
        Callable: Wrapped function with retry logic.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    args[0].logger.warning(f"{func.__name__} failed with error: {e}. Retrying in {mdelay}s...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator


def infer_lookback_timedelta(timeframe: str, lookback_bars: int) -> timedelta:
    """
    Infers a timedelta representing the lookback period based on the timeframe and number of bars.

    Args:
        timeframe (str): Timeframe string such as "M15", "H1", "D1".
        lookback_bars (int): Number of bars to look back.

    Returns:
        timedelta: Corresponding time duration.
    
    Raises:
        ValueError: If the timeframe format is not supported.
    """
    if timeframe.startswith("M"):
        return timedelta(minutes=lookback_bars * int(timeframe[1:]))
    elif timeframe.startswith("H"):
        return timedelta(hours=lookback_bars * int(timeframe[1:]))
    elif timeframe.startswith("D"):
        return timedelta(days=lookback_bars * int(timeframe[1:]))
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")


class MT5DataDownload:
    """
    Class for downloading historical and live data from MetaTrader5 (MT5).
    Handles connection initialization, data retrieval, schema validation, and CSV export.
    """

    def __init__(self) -> None:
        self.logger = get_logger("data_download")
        self.initialized = False
        self.local_tz = ZoneInfo(config.LOCAL_TIMEZONE)

    def initialize(self) -> None:
        """
        Initializes the MT5 connection with credentials provided in the configuration.

        Raises:
            ConnectionError: If initialization fails.
        """
        if not mt5.initialize(
            config.MT5_CONFIG_PARAMS['PATH'],
            login=config.MT5_CONFIG_PARAMS['LOGIN'],
            password=config.MT5_CONFIG_PARAMS['PASSWORD'],
            server=config.MT5_CONFIG_PARAMS['SERVER']
        ):
            self.logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
            raise ConnectionError("Could not initialize MT5.")
        self.initialized = True
        self.logger.info("MT5 initialized successfully.")

    def shutdown(self) -> None:
        """
        Shuts down the MT5 connection.
        """
        mt5.shutdown()
        self.initialized = False
        self.logger.info("MT5 connection shutdown.")

    @retry((ValueError, RuntimeError, ConnectionError), tries=3)
    def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        utc_from: Optional[datetime] = None, 
        utc_to: Optional[datetime] = None,
        lookback_bars: Optional[int] = None,
        server_offset_hours: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV data from MT5. Data can be fetched either by specifying a date range
        (utc_from and utc_to) or by specifying a lookback window in bars.

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD').
            timeframe (str): Configured timeframe key (e.g., 'H1', 'M15').
            utc_from (Optional[datetime]): Start datetime in UTC for data retrieval.
            utc_to (Optional[datetime]): End datetime in UTC for data retrieval.
            lookback_bars (Optional[int]): Number of bars to look back if no date range is provided.
            server_offset_hours (Optional[int]): Server offset in hours for time adjustment.

        Returns:
            pd.DataFrame: Historical OHLCV data.

        Raises:
            RuntimeError: If data retrieval fails.
        """
        if not self.initialized:
            self.initialize()

        # Determine the UTC time range:
        # If both utc_from and utc_to are provided, use them directly.
        # If only one is provided, fill in the missing value.
        # Otherwise, use lookback_bars to compute the date range.
        if utc_from is None and utc_to is None:
            lookback_bars = lookback_bars or DEFAULT_LOOKBACK_BARS
            delta = infer_lookback_timedelta(timeframe, lookback_bars)
            utc_now = datetime.now(pytz.utc)
            utc_from = utc_now - delta
            utc_to = utc_now
        elif utc_from is None:
            lookback_bars = lookback_bars or DEFAULT_LOOKBACK_BARS
            delta = infer_lookback_timedelta(timeframe, lookback_bars)
            utc_from = utc_to - delta
        elif utc_to is None:
            utc_to = datetime.now(pytz.utc)

        self.logger.info(f"Requesting MT5 data for {symbol} from {utc_from} to {utc_to}...")

        mt5_timeframe = config.MT5_TIMEFRAMES[timeframe]
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, utc_from, utc_to)

        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            self.logger.warning(f"Empty rates returned: {err}. Retrying...")
            raise RuntimeError("Failed to fetch raw candles or no data returned.")

        # Shutdown MT5 connection after data retrieval.
        self.shutdown()
        offset = server_offset_hours or DEFAULT_SERVER_OFFSET_HOURS
        df = self._rates_to_df(rates, server_offset_hours=offset)
        self._validate_schema(df)

        self.logger.info(f"Historical data retrieved for {symbol}.")
        return df

    @retry((ValueError, RuntimeError, ConnectionError), tries=3)
    def get_latest_data(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback_bars: int = 100,
        server_offset_hours: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetches the most recent live data (candles) from MT5.

        Args:
            symbol (str): Trading symbol.
            timeframe (str): Configured timeframe key.
            lookback_bars (int): Number of candles to retrieve.
            server_offset_hours (Optional[int]): Server time offset.

        Returns:
            pd.DataFrame: DataFrame containing the latest live candles.
        """
        if not self.initialized:
            self.initialize()

        mt5_timeframe = config.MT5_TIMEFRAMES[timeframe]
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, lookback_bars)

        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            self.logger.warning(f"Empty rates returned: {err}. Retrying...")
            raise RuntimeError("Failed to fetch raw candles or no data returned.")

        self.shutdown()
        offset = server_offset_hours or DEFAULT_SERVER_OFFSET_HOURS
        df = self._rates_to_df(rates, server_offset_hours=offset)
        self._validate_schema(df)

        self.logger.info(f"Latest live data retrieved for {symbol}.")
        return df[REQUIRED_COLUMNS]

    def _rates_to_df(
        self, 
        rates: np.ndarray, 
        server_offset_hours: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Converts raw MT5 rates data into a standardized and timezone-adjusted DataFrame.

        Args:
            rates (np.ndarray): Raw rates data from MT5.
            server_offset_hours (Optional[int]): Hours to offset the server time.

        Returns:
            pd.DataFrame: Converted DataFrame sorted by date.
        
        Raises:
            ValueError: If no data is received.
        """
        if rates is None or len(rates) == 0:
            self.logger.error("No rates data received.")
            raise ValueError("No data returned from MT5.")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(self.local_tz)

        if server_offset_hours is not None:
            df['time'] -= timedelta(hours=server_offset_hours)

        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        df.rename(columns={
            'time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)

        return df.sort_values('Date').reset_index(drop=True)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validates that the DataFrame contains the required OHLCV columns.

        Args:
            df (pd.DataFrame): DataFrame to validate.

        Raises:
            ValueError: If any required column is missing.
        """
        if missing := [col for col in REQUIRED_COLUMNS if col not in df.columns]:
            self.logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Invalid data schema. Missing columns: {missing}")

    def save_historical_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Saves the retrieved historical data as a CSV file under a structured directory path.

        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data.
            symbol (str): Trading symbol.
            timeframe (str): Timeframe string.
        """
        dir_prefix = os.path.join(config.DATA_DIR, "ohlcv")
        dir_path = os.path.join(dir_prefix, f"{symbol}")
        os.makedirs(dir_path, exist_ok=True)

        filename = f"{symbol.lower()}_{timeframe.lower()}.csv"
        file_path = os.path.join(dir_path, filename)

        df.to_csv(file_path, index=False)
        self.logger.info(f"MT5 data saved to {file_path} as CSV.")
