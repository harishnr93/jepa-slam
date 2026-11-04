"""Structured logging: human-readable console output plus optional JSON-lines file."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT_NAME = "latent_loop"
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line; ``extra=`` fields are included."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """``HH:MM:SS LEVEL logger: message key=value ...``"""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        extras = {
            k: v
            for k, v in vars(record).items()
            if k not in _STANDARD_ATTRS and not k.startswith("_")
        }
        if extras:
            text += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        return text


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure the ``latent_loop`` logger hierarchy (idempotent)."""
    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level if isinstance(level, int) else level.upper())
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter())
    logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under ``latent_loop`` (``name`` may be a module ``__name__``)."""
    if name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
