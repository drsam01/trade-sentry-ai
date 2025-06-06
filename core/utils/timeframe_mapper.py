# File: core/utils/timeframe_mapper.py

def map_timeframe(broker_type: str, timeframe: str) -> str:
    """
    Map a generic timeframe to broker-specific format.

    Args:
        broker_type (str): Broker type (e.g., 'MT5', 'BINANCE', 'IBKR').
        timeframe (str): Generic timeframe (e.g., 'M5', 'H1', 'D1').

    Returns:
        str: Broker-specific timeframe format.
    """

    # Normalize input
    broker_type = broker_type.upper()
    timeframe = timeframe.upper()

    if broker_type == "MT5":
        return timeframe  # MT5 uses "M5", "H1", etc.

    elif broker_type == "BINANCE":
        mapping = {
            "M1": "1m",
            "M5": "5m",
            "M15": "15m",
            "M30": "30m",
            "H1": "1h",
            "H4": "4h",
            "D1": "1d"
        }
        return mapping.get(timeframe, timeframe)

    elif broker_type == "IBKR":
        mapping = {
            "M1": "1 min",
            "M5": "5 mins",
            "M15": "15 mins",
            "M30": "30 mins",
            "H1": "1 hour",
            "H4": "4 hours",
            "D1": "1 day"
        }
        return mapping.get(timeframe, timeframe)

    else:
        raise ValueError(f"Unsupported broker type for timeframe mapping: {broker_type}")
