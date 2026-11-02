# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Iterable


class PyAmpliColError(Exception):
    """Base class for stable pyAmpliCol errors."""


class ConfigurationError(PyAmpliColError, ValueError):
    """Raised when configuration input violates schema version 1."""

    def __init__(self, message: str | Iterable[str]) -> None:
        messages = (message,) if isinstance(message, str) else tuple(message)
        if not messages:
            messages = ("invalid configuration",)
        if not all(isinstance(entry, str) and entry for entry in messages):
            raise TypeError("configuration error messages must be non-empty strings")
        self.messages = messages
        rendered = messages[0]
        if len(messages) > 1:
            rendered = f"{len(messages)} configuration errors:\n" + "\n".join(
                f"- {entry}" for entry in messages
            )
        super().__init__(rendered)


__all__ = ["ConfigurationError", "PyAmpliColError"]
