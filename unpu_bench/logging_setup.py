from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure process-wide logging.

    Args:
        level: "DEBUG", "INFO", "WARNING", "ERROR".
        log_file: Optional path to a log file; if given, logs go to both console and file.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    logging.getLogger(__name__).debug(
        "Logging configured (level=%s, log_file=%s)", level, log_file
    )
