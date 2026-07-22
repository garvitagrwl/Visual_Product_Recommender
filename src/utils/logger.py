"""Logging setup shared across the project.

Using the standard `logging` module instead of print statements so
output is leveled, timestamped, and can be redirected to a file
without touching call sites.
"""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """Create (or fetch) a configured logger.

    Args:
        name: Logger name, typically `__name__` of the calling module.
        level: Logging level as a string (e.g. "INFO", "DEBUG").
        log_file: Optional path to also write logs to a file. Parent
            directories are created if they don't exist.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid attaching duplicate handlers if get_logger is called
    # multiple times for the same logger name (e.g. re-imports).
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
