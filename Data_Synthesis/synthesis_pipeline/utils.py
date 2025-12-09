"""
Utility functions for the ELM synthesis pipeline.
Follows patterns from Data_Preparation module.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    logger = logging.getLogger("synthesis_pipeline")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (will be prefixed with 'synthesis_pipeline.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"synthesis_pipeline.{name}")


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1m 30s", "2h 15m")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def count_tokens_approx(text: str) -> int:
    """Approximate token count using simple heuristic.

    Args:
        text: Input text

    Returns:
        Approximate token count (words * 1.3)
    """
    return int(len(text.split()) * 1.3)
