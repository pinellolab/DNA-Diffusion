import logging
import os

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


def configure_logging(logger_name: str = "dnadiffusion") -> logging.Logger:
    """
    Configures logging with rich handler and checks for valid log level from
    environment.

    Defaults to `INFO` if no valid log level is found.
    """
    console_theme = Theme(
        {
            "logging.level.info": "dim cyan",
            "logging.level.warning": "magenta",
            "logging.level.error": "bold red",
            "logging.level.debug": "green",
        }
    )
    console = Console(theme=console_theme)
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        log_time_format="[%X]",
    )
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    if log_level not in valid_log_levels:
        log_level = "INFO"

    logging.basicConfig(
        level=log_level,
        format="%(name)s %(message)s",
        datefmt="[%X]",
        handlers=[rich_handler],
    )
    return logging.getLogger(logger_name)
