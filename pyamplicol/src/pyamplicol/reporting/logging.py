# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, TextIO

LOGGER_NAME = "pyamplicol"
DEFAULT_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
_HANDLER_MARKER = "_pyamplicol_cli_handler"
_NULL_MARKER = "_pyamplicol_null_handler"
_PREVIOUS_STATE_MARKER = "_pyamplicol_cli_previous_state"


class _CliFormatter(logging.Formatter):
    """Color terminal diagnostics while leaving ordinary informational logs alone."""

    def __init__(self, fmt: str, *, color: bool) -> None:
        super().__init__(fmt)
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if not self._color:
            return rendered
        try:
            colorama: Any = importlib.import_module("colorama")
        except ImportError:
            return rendered
        if record.levelno >= logging.ERROR:
            foreground = colorama.Fore.RED
        elif record.levelno >= logging.WARNING:
            foreground = colorama.Fore.YELLOW
        else:
            return rendered
        return f"{foreground}{rendered}{colorama.Style.RESET_ALL}"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger or a named child without configuring logging."""

    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def _ensure_null_handler() -> None:
    logger = get_logger()
    if any(getattr(handler, _NULL_MARKER, False) for handler in logger.handlers):
        return
    handler = logging.NullHandler()
    setattr(handler, _NULL_MARKER, True)
    logger.addHandler(handler)


def _normalize_level(level: int | str) -> int:
    if isinstance(level, bool):
        raise ValueError(f"invalid logging level {level!r}")
    if isinstance(level, int):
        return level
    if not isinstance(level, str):
        raise ValueError(f"invalid logging level {level!r}")
    normalized = logging.getLevelNamesMapping().get(level.upper())
    if normalized is None:
        raise ValueError(f"unknown logging level {level!r}")
    return normalized


def configure_cli_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    fmt: str = DEFAULT_LOG_FORMAT,
    color: bool = False,
) -> logging.Logger:
    """Install one package-local CLI handler, leaving the root logger untouched."""

    selected_level = _normalize_level(level)
    logger = get_logger()
    handler = next(
        (
            candidate
            for candidate in logger.handlers
            if getattr(candidate, _HANDLER_MARKER, False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stderr if stream is None else stream)
        setattr(handler, _HANDLER_MARKER, True)
        setattr(
            handler,
            _PREVIOUS_STATE_MARKER,
            (logger.level, logger.propagate),
        )
        logger.addHandler(handler)
    elif stream is not None and isinstance(handler, logging.StreamHandler):
        handler.setStream(stream)
    handler.setLevel(selected_level)
    handler.setFormatter(_CliFormatter(fmt, color=color))
    logger.setLevel(selected_level)
    logger.propagate = False
    return logger


def reset_cli_logging() -> None:
    """Remove CLI handlers and restore the embedding application's logger state."""

    logger = get_logger()
    previous_state: tuple[int, bool] | None = None
    for handler in tuple(logger.handlers):
        if not getattr(handler, _HANDLER_MARKER, False):
            continue
        candidate_state = getattr(handler, _PREVIOUS_STATE_MARKER, None)
        if (
            previous_state is None
            and isinstance(candidate_state, tuple)
            and len(candidate_state) == 2
            and isinstance(candidate_state[0], int)
            and isinstance(candidate_state[1], bool)
        ):
            previous_state = candidate_state
        logger.removeHandler(handler)
        handler.close()
    if previous_state is not None:
        logger.setLevel(previous_state[0])
        logger.propagate = previous_state[1]
    _ensure_null_handler()


_ensure_null_handler()


__all__ = [
    "DEFAULT_LOG_FORMAT",
    "LOGGER_NAME",
    "configure_cli_logging",
    "get_logger",
    "reset_cli_logging",
]
