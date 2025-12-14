"""
Simple logging setup for statqa.

Provides minimal logging configuration with debug support via environment variable.
No complex logging infrastructure - just simple, useful debugging.
"""

import logging
import os
from typing import Literal


def setup_logging(
    logger_name: str,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None,
) -> logging.Logger:
    """
    Set up simple logging for statqa modules.

    Respects STATQA_DEBUG environment variable:
    - STATQA_DEBUG=1: DEBUG level
    - Default: INFO level

    Args:
        logger_name: Usually __name__ from calling module
        level: Override log level (optional)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(logger_name)

    # Don't add multiple handlers
    if logger.handlers:
        return logger

    # Determine log level
    if level:
        log_level = getattr(logging, level.upper())
    elif os.environ.get("STATQA_DEBUG", "").lower() in ("1", "true", "yes"):
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Simple formatter - no timestamps for CLI tool
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")

    # Console handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(log_level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module with statqa's simple configuration."""
    return setup_logging(name)
