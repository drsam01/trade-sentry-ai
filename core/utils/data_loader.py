# File: drlfusion/modelling/models/training_pipeline/utils/data_loader.py

import os
import pandas as pd
#from drlfusion.core.brokers.mt5_interface import MT5Interface
from drlfusion.data.data_download import MT5DataDownload
from drlfusion.configs.config import Config
from drlfusion.data.data_fetch_and_preprocess import _attempt_latest_fetch


def load_raw_data(symbol: str, timeframe: str, config: Config) -> pd.DataFrame:
    filename = f"{symbol.lower()}_{timeframe.lower()}.csv"
    csv_path = os.path.join(config.DATA_DIR, "ohlcv", symbol, filename)

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["Date"])

    downloader = MT5DataDownload()

    return _attempt_latest_fetch(
        downloader, symbol, timeframe, initial_bars=config.HISTORICAL_LOOKBACK_BARS
    )
