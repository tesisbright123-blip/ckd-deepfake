"""Logging setup: console + optional file handler.

Central entry point is :func:`get_logger`. Scripts use this instead of
plain ``print`` so that progress reports can be captured both on stdout
and in a log file under ``runs/<experiment>/<name>.log``.

Called by: scripts/01-08, src/data/face_extractor.py, src/training/trainer.py,
           src/training/continual_trainer.py, src/evaluation/evaluator.py
Writes: plain-text .log files.
        Line format: ``%Y-%m-%dT%H:%M:%S | LEVEL   | name | message``
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Track which loggers we've configured so repeated calls don't stack handlers.
_CONFIGURED: set[str] = set()


def get_logger(
    name: str = "ckd",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name. Conventionally the script or module name.
        log_file: Optional path to a text log file. Parent dir is created
            automatically.
        level: Logging level (e.g. ``logging.INFO``).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if name in _CONFIGURED:
        if log_file is not None and not _has_file_handler(logger, Path(log_file)):
            logger.addHandler(_make_file_handler(log_file, level))
        return logger

    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(console)

    if log_file is not None:
        logger.addHandler(_make_file_handler(log_file, level))

    _CONFIGURED.add(name)
    return logger


def _make_file_handler(log_file: str | Path, level: int) -> logging.FileHandler:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    return handler


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    resolved = path.resolve()
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == resolved:
            return True
    return False
