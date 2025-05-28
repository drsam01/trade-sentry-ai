import logging

def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up the logging configuration for the application.

    Args:
        level (int): Logging level (default: logging.INFO).
    """
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_logger(name: str) -> logging.Logger:
    """
    Create or retrieve a logger with the specified name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

    """        
        observation_space: Space representing the agent's observations.
        action_space: Space representing the agent's actions.
        config: Configuration dictionary for the agent.
    """