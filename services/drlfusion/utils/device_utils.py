# File: drlfusion/utils/device_utils.py

import torch
from libs.utils.logger import get_logger

logger = get_logger("device_utils")

def get_best_device(warn_if_cpu: bool = True) -> torch.device:
    """
    Returns the best available torch device and logs hardware info.

    Args:
        warn_if_cpu (bool): Whether to warn if only CPU is used.

    Returns:
        torch.device: torch.device("cuda") if available, else "cpu"
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"✅ Using GPU: {gpu_name} ({total_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        if warn_if_cpu:
            logger.warning("⚠️ CUDA is not available. Training will run on CPU. This may be significantly slower.")
        else:
            logger.info("Using CPU as fallback device.")

    return device
