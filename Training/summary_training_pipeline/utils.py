"""Utility functions for ELM training."""

import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_file: Optional path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for a module.

    Args:
        name: Module name

    Returns:
        Logger instance
    """
    return logging.getLogger(f"training_pipeline.{name}")


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Trainable parameter count
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get available device (CUDA if available, else CPU).

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger = get_logger("utils")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        logger = get_logger("utils")
        logger.warning("CUDA not available, using CPU")

    return device


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def validate_file_exists(path: Path, description: str = "File"):
    """Validate that a file exists.

    Args:
        path: Path to file
        description: Description of file for error message

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def validate_directory_exists(path: Path, description: str = "Directory"):
    """Validate that a directory exists.

    Args:
        path: Path to directory
        description: Description of directory for error message

    Raises:
        FileNotFoundError: If directory does not exist
    """
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")


def get_gpu_memory_info() -> dict:
    """Get GPU memory information.

    Returns:
        Dictionary with GPU memory stats (in GB)
    """
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    max_reserved = torch.cuda.max_memory_reserved() / 1e9

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "max_reserved_gb": max_reserved,
    }


def log_gpu_memory(logger: logging.Logger, prefix: str = ""):
    """Log current GPU memory usage.

    Args:
        logger: Logger instance
        prefix: Optional prefix for log message
    """
    if not torch.cuda.is_available():
        return

    info = get_gpu_memory_info()
    msg = f"{prefix}GPU Memory - Allocated: {info['allocated_gb']:.2f} GB, Reserved: {info['reserved_gb']:.2f} GB"
    logger.info(msg)
