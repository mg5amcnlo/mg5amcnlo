# SPDX-License-Identifier: 0BSD
"""Small expression helpers shared by built-in-SM lowering phases."""

from __future__ import annotations

from typing import Any


def _number(value: float) -> Any:
    from symbolica import Expression

    return Expression.num(value)


def _current_key_tuple(current: Any) -> tuple[int, tuple[int, ...], int]:
    return (
        int(current.pdg),
        tuple(int(label) for label in current.external_labels),
        int(current.chirality),
    )
