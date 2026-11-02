# SPDX-License-Identifier: 0BSD
"""Structural catalogues of ordered soft colour connections at arbitrary order.

Every active coloured leg can emit a gluon; only an auxiliary gluon can split
into a quark pair. Born labels are positive. At order r, the r final auxiliary
partons receive every assignment of labels -1,...,-r. Retired intermediates
have separate dummy labels. Only independent chronological interleavings are
identified: shared-line ordering and creation dependencies are retained.

The catalogue contains all directed overlaps with matching final typed legs,
including self-overlaps. It is not a minimal operator basis or a subtraction
formula. No multiplicities, flavour sums, kinematic factors or numerical
weights are attached to the canonical histories. The complete catalogue can
grow rapidly; explicit requests remain preferable when only a subset is used.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from itertools import permutations

from .connections import (
    ColorConnection,
    ColorEmission,
    ColorLeg,
    EmitGluon,
    SplitGluon,
)
from .correlator_matrices import ColorCorrelator

_HistoryKey = tuple[tuple[int, ...], ...]


def _parent(operation: ColorEmission) -> int:
    return (
        operation.emitter_label
        if isinstance(operation, EmitGluon)
        else operation.parent_label
    )


def _children(operation: ColorEmission) -> tuple[int, ...]:
    return (
        (operation.emitted_label,)
        if isinstance(operation, EmitGluon)
        else (operation.quark_label, operation.antiquark_label)
    )


def _operation_key(operation: ColorEmission) -> tuple[int, ...]:
    return (
        (0, operation.emitter_label, operation.emitted_label)
        if isinstance(operation, EmitGluon)
        else (
            1,
            operation.parent_label,
            operation.quark_label,
            operation.antiquark_label,
        )
    )


def _canonical_order(
    operations: tuple[ColorEmission, ...],
) -> tuple[_HistoryKey, tuple[ColorEmission, ...]]:
    """Return a comparison key and a valid chronological representative.

    Sorting operation tokens alone would move consumers before producers.
    Greedily taking the smallest available token gives the lexicographically
    first topological order without enumerating independent interleavings.
    """
    dependencies = tuple(
        (before, after)
        for before in range(len(operations))
        for after in range(before + 1, len(operations))
        if _parent(operations[before]) == _parent(operations[after])
        or _parent(operations[after]) in _children(operations[before])
    )
    remaining = set(range(len(operations)))
    order = []
    while remaining:
        available = (
            index
            for index in remaining
            if not any(
                after == index and before in remaining for before, after in dependencies
            )
        )
        next_index = min(available, key=lambda index: _operation_key(operations[index]))
        order.append(next_index)
        remaining.remove(next_index)
    chronological = tuple(operations[index] for index in order)
    return tuple(
        _operation_key(operation) for operation in chronological
    ), chronological


def _rename(operation: ColorEmission, labels: dict[int, int]) -> ColorEmission:
    def label(original: int) -> int:
        return labels.get(original, original)

    if isinstance(operation, EmitGluon):
        return EmitGluon(label(operation.emitter_label), label(operation.emitted_label))
    return SplitGluon(
        label(operation.parent_label),
        label(operation.quark_label),
        label(operation.antiquark_label),
    )


def _labelled_histories(
    connection: ColorConnection,
) -> Iterator[tuple[_HistoryKey, tuple[ColorEmission, ...]]]:
    final = tuple(leg.label for leg in connection.output_legs if leg.label < 0)
    created = {child for step in connection.emissions for child in _children(step)}
    retired = tuple(sorted(created.difference(final)))
    final_labels = tuple(-index for index in range(1, connection.order + 1))
    retired_labels = tuple(
        -connection.order - index for index in range(1, len(retired) + 1)
    )
    assert len(final) == connection.order
    for final_assignment in permutations(final_labels):
        representatives = []
        for retired_assignment in permutations(retired_labels):
            labels = dict(zip(final, final_assignment, strict=True))
            labels.update(zip(retired, retired_assignment, strict=True))
            representatives.append(
                _canonical_order(
                    tuple(_rename(step, labels) for step in connection.emissions)
                )
            )
        # Retired names are dummy indices, unlike the externally matched final
        # labels. They must not create additional catalogue connections.
        yield min(representatives, key=lambda candidate: candidate[0])


def _extensions(connection: ColorConnection) -> Iterator[ColorConnection]:
    created = tuple(child for step in connection.emissions for child in _children(step))
    fresh = min(created, default=0) - 1
    for leg in connection.output_legs:
        if leg.representation == 1:
            continue
        yield ColorConnection(
            connection.initial_legs,
            (*connection.emissions, EmitGluon(leg.label, fresh)),
        )
        if leg.label < 0 and leg.representation == 8:
            yield ColorConnection(
                connection.initial_legs,
                (*connection.emissions, SplitGluon(leg.label, fresh, fresh - 1)),
            )


def build_color_correlator_catalogue(
    legs: Sequence[ColorLeg], *, through_order: int
) -> tuple[ColorCorrelator, ...]:
    """Prepare all ordered soft connections at orders 1 through ``through_order``.

    ``legs`` are all-outgoing Born colour roles with positive public labels.
    Singlets are allowed but do not emit. ``through_order`` is a positive integer.
    The ordinary ``born`` identity is not included; ``CorrelatorConfig`` adds it.

    IDs ``N{r}LO/c{i}/c{j}`` use zero-based indices in the deterministically
    sorted connection list at each order. They are catalogue-local identifiers,
    not process-independent operator names; the bra/ket operations encode the
    actual connection. Every compatible directed pair is retained, without
    assuming that the result is real, nonzero, positive or independent.
    """
    if type(through_order) is not int or through_order < 1:
        raise ValueError("catalogue through_order must be a positive integer")
    initial = ColorConnection(tuple(legs))
    if any(leg.label <= 0 for leg in initial.initial_legs):
        raise ValueError("catalogue Born legs must have positive labels")
    histories: tuple[ColorConnection, ...] = (initial,)
    result = []
    for order in range(1, through_order + 1):
        histories = tuple(
            child for parent in histories for child in _extensions(parent)
        )
        unique = dict(
            labelled
            for connection in histories
            for labelled in _labelled_histories(connection)
        )
        ordered = tuple(unique[key] for key in sorted(unique))
        by_output: dict[tuple[ColorLeg, ...], list[int]] = defaultdict(list)
        outputs = tuple(
            ColorConnection(initial.initial_legs, operations).output_legs
            for operations in ordered
        )
        for index, output in enumerate(outputs):
            by_output[output].append(index)
        for left, bra in enumerate(ordered):
            for right in by_output[outputs[left]]:
                result.append(
                    ColorCorrelator(f"N{order}LO/c{left}/c{right}", bra, ordered[right])
                )
    return tuple(result)


__all__ = ["build_color_correlator_catalogue"]
