# File: core/utils/logger.py

import logging
from typing import Optional
from logging.handlers import RotatingFileHandler

_is_logging_configured = False

def get_logger( # type: ignore
    module: str,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    log_to_file: bool = True,
    log_file: str = "logs/tradesentry.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3
) -> logging.Logger:
    """
    Get a logger named after the current module file only (e.g. 'my_module').

    Args:
        module (str): Typically passed as __name__.
        level (int): Logging level.
        fmt (Optional[str]): Log message format.
        log_to_file (bool): Whether to write logs to file.
        log_file (str): Path to log file.
        max_bytes (int): Max size before rotation.
        backup_count (int): Number of rotated files to keep.

    Returns:
        logging.Logger: Configured logger.
    """
    global _is_logging_configured

    short_name = module.split(".")[-1]  # strips to 'my_module'

    log_format = fmt or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if not _is_logging_configured:
        console_handler = logging.StreamHandler()
        date_format = '%Y-%m-%d %H:%M:%S'

        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers = [console_handler]
        if log_to_file:
            import os
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
            file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
            handlers.append(file_handler) # type: ignore

        logging.basicConfig(level=level, handlers=handlers)
        _is_logging_configured = True

    logger = logging.getLogger(short_name)
    logger.setLevel(level)
    return logger

'''
import logging
import sys


def get_logger(name: str = "trade_sentry") -> logging.Logger:
    """
    Creates and returns a configured logger instance.

    Args:
        name: Name of the logger (default: 'trade_sentry').

    Returns:
        A logging.Logger object.
    """
    short_name = name.split(".")[-1]  # strips to 'my_module'
    logger = logging.getLogger(short_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
'''