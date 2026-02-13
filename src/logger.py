"""Logging configuration."""

import logging
import sys
from src.config import settings


def setup_logger(name: str) -> logging.Logger:
    """Configure logger for a module."""

    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.log_level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add handler
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


# Create module logger
logger = setup_logger(__name__)
