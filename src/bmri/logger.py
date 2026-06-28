"""Logging configuration for bMRI.

This module provides a centralized logging setup using Rich for beautiful
terminal output with colors, formatting, and structured logging.
"""

import logging
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# Install rich traceback handler globally
install_rich_traceback(show_locals=False)

# Global console instance
console = Console()

# Logger levels
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel = "INFO",
    log_file: Path | None = None,
    show_time: bool = True,
    show_path: bool = False,
) -> logging.Logger:
    """Configure logging for bMRI application.

    Sets up Rich-based logging with optional file output. This should be called
    once at application startup.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to write logs to file
        show_time: Whether to show timestamps in console output
        show_path: Whether to show file paths in console output

    Returns:
        Configured root logger

    Example:
        >>> from bmri.logger import setup_logging
        >>> logger = setup_logging(level="DEBUG")
        >>> logger.info("Application started")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Console handler with Rich formatting
    console_handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=True,
        tracebacks_show_locals=level == "DEBUG",
        markup=True,
    )
    console_handler.setLevel(numeric_level)

    # Simple format - Rich adds colors and formatting
    formatter = logging.Formatter(
        "%(message)s",
        datefmt="[%X]",
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(numeric_level)

        # Detailed format for file logs
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance

    Example:
        >>> from bmri.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data...")
    """
    return logging.getLogger(name)


# Module-level logger
logger = get_logger(__name__)
