"""Privacy-safe local logging configuration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.infrastructure.config import LoggingConfig


def configure_logging(log_dir: Path, config: LoggingConfig) -> logging.Logger:
    """Configure a rotating local log without document-content handlers."""

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("document_intelligence")
    logger.setLevel(getattr(logging, config.level, logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    return logger

