# SPDX-License-Identifier: 0BSD
"""Complete scalar tolerance-window candidate indexes.

The index is only a screening device.  Callers must still replay every
returned candidate against every captured component.  For one selected
complex observation, let ``X`` and ``Y`` be the infinity norms of the current
and signed representative, and ``D`` their infinity-norm residual.  Passing
gives ``D <= a + r*max(X,Y)``; reverse triangle gives ``Y <= X+D``.  Thus, for
``r < 1``, ``D <= (a+rX)/(1-r)``, and the chosen scalar residual is no larger
than ``D``.  The returned one-dimensional window therefore has no false
negatives.  Residual tolerance is independent per complex observation, so an
unrelated large observation does not enlarge the selected pair's bound.
Policies with ``r >= 1`` deliberately fall back to the complete earlier set.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

_RawScalar = TypeVar("_RawScalar")
_IndexScalar = TypeVar("_IndexScalar")


@dataclass(frozen=True, slots=True)
class NumericalObservationCandidateIndex(Generic[_IndexScalar]):
    """Most-discriminating deterministic scalar projection for one contract."""

    observation_index: int
    scalar_component: int
    entries: tuple[tuple[_IndexScalar, int], ...]


def build_numerical_observation_candidate_index(
    member_ids: Sequence[int],
    observations: Mapping[
        int,
        Sequence[tuple[_RawScalar, _RawScalar]],
    ],
    *,
    normalize: Callable[[_RawScalar], _IndexScalar],
) -> NumericalObservationCandidateIndex[_IndexScalar]:
    """Build one deterministic complete index without copying full vectors."""

    members = tuple(member_ids)
    if not members:
        raise ValueError("numerical candidate index has no members")
    observation_count = len(observations[members[0]])
    if observation_count == 0 or any(
        len(observations[current_id]) != observation_count for current_id in members
    ):
        raise ValueError("numerical candidate index members have inconsistent widths")

    def discrimination(choice: tuple[int, int]) -> tuple[int, int, int, int]:
        observation_index, scalar_component = choice
        distinct_values: set[_RawScalar] = set()
        nonzero_count = 0
        for current_id in members:
            value = observations[current_id][observation_index][scalar_component]
            distinct_values.add(value)
            nonzero_count += value != 0
        return (
            len(distinct_values),
            nonzero_count,
            -observation_index,
            -scalar_component,
        )

    observation_index, scalar_component = max(
        (
            (observation_index, scalar_component)
            for observation_index in range(observation_count)
            for scalar_component in (0, 1)
        ),
        key=discrimination,
    )
    entries = tuple(
        sorted(
            (
                normalize(
                    observations[current_id][observation_index][scalar_component]
                ),
                current_id,
            )
            for current_id in members
        )
    )
    return NumericalObservationCandidateIndex(
        observation_index=observation_index,
        scalar_component=scalar_component,
        entries=entries,
    )


def numerical_observation_tolerance_window_ids(
    index: NumericalObservationCandidateIndex[_IndexScalar],
    current_values: Sequence[tuple[_RawScalar, _RawScalar]],
    *,
    relation_kind: Literal["equal", "opposite"],
    current_id: int,
    relative_tolerance: _IndexScalar,
    absolute_tolerance: _IndexScalar,
    normalize: Callable[[_RawScalar], _IndexScalar],
) -> tuple[int, ...]:
    """Return every earlier representative that can pass the full test."""

    if relation_kind not in {"equal", "opposite"}:
        raise ValueError("numerical candidate index relation is unsupported")
    if not 0 <= index.observation_index < len(current_values):
        raise ValueError("numerical candidate index observation is invalid")
    relative: Any = relative_tolerance
    absolute: Any = absolute_tolerance
    if relative >= 1:
        return tuple(
            representative_id
            for _value, representative_id in index.entries
            if representative_id < current_id
        )
    selected = current_values[index.observation_index]
    selected_scalar: Any = normalize(selected[index.scalar_component])
    target = selected_scalar if relation_kind == "equal" else -selected_scalar
    current_scale = max(
        abs(normalize(selected[0])),
        abs(normalize(selected[1])),
    )
    radius = (absolute + relative * current_scale) / (1 - relative)
    entries: Any = index.entries
    lower = bisect_left(entries, (target - radius, -1))
    upper = bisect_right(entries, (target + radius, current_id - 1))
    return tuple(
        representative_id
        for _value, representative_id in entries[lower:upper]
        if representative_id < current_id
    )


__all__ = [
    "NumericalObservationCandidateIndex",
    "build_numerical_observation_candidate_index",
    "numerical_observation_tolerance_window_ids",
]
