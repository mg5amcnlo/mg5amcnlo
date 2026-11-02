# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    if name != "SymbolicaEvaluatorSettings":
        raise AttributeError(name)
    value = getattr(importlib.import_module(".symbolica", __name__), name)
    globals()[name] = value
    return value


__all__ = ["SymbolicaEvaluatorSettings"]
