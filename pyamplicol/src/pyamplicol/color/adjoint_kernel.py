# SPDX-License-Identifier: 0BSD
"""Exact DDM Gram kernels without constructing the quadratic trace matrix.

With Tr(Ta Tb) = delta_ab, a half-ladder is the nested-commutator trace
Tr([...[T1,T2],...,T(n-1)] Tn).  Both U(1) contributions vanish under the
commutators, so the inexpensive U(Nc) trace cycle kernel is sufficient.
Applying the left/right commutator finite differences takes 2*(n-2) sweeps
over S_(n-1); the result retained on the two-anchor subgroup has (n-2)!
entries.  All arithmetic is integer, including NLC's Nc-power truncation.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import chain, permutations

import numpy as np

MAX_ADJOINT_GLUONS = 12  # Two anchors plus the supported FFT degree of ten.


def _permutation_ranks(rows: np.ndarray) -> np.ndarray:
    """Rank a batch of image permutations in lexicographic order."""

    degree = rows.shape[1]
    ranks = np.zeros(len(rows), dtype=np.int64)
    for position in range(degree - 1):
        digit = rows[:, position].astype(np.int64)
        for preceding in range(position):
            digit -= rows[:, preceding] < rows[:, position]
        ranks += digit * math.factorial(degree - position - 1)
    return ranks


def _trace_cycle_counts(rows: np.ndarray) -> np.ndarray:
    """Cycles of c^-1 h c h^-1, with h fixing the final external label."""

    count, degree = rows.shape
    total = degree + 1
    indices = np.arange(count)
    commutator = np.empty((count, total), dtype=np.uint8)
    for position in range(degree - 1):
        commutator[indices, rows[:, position]] = (
            rows[:, position + 1].astype(np.int16) - 1
        ) % total
    commutator[indices, rows[:, -1]] = total - 2
    commutator[:, -1] = (rows[:, 0].astype(np.int16) - 1) % total
    visited = np.zeros(count, dtype=np.uint32)
    cycles = np.zeros(count, dtype=np.uint8)
    bits = np.left_shift(np.uint32(1), np.arange(total, dtype=np.uint32))
    for start in range(total):
        active = (visited & bits[start]) == 0
        cycles += active
        position = np.full(count, start, dtype=np.uint8)
        while np.any(active):
            visited |= np.where(active, bits[position], np.uint32(0))
            position = commutator[indices, position]
            active &= (visited & bits[position]) == 0
    return cycles


@lru_cache(maxsize=4)
def adjoint_color_kernel(
    total_gluons: int,
    *,
    accuracy: str = "full",
    nc: int = 3,
    full_col_acc: int = 20,
) -> np.ndarray:
    """Return an immutable exact relative DDM kernel in lexicographic order.

    ``lc`` and ``nlc`` retain powers >= n and >= n-2 respectively.  Since the
    finite differences have Nc-independent integer coefficients, truncating
    the seed commutes with the basis transformation.  The LC helper exists
    for factor diagnostics; adjoint generation itself requires NLC/full.
    """

    if type(total_gluons) is not int or not 2 <= total_gluons <= MAX_ADJOINT_GLUONS:
        raise ValueError("adjoint colour kernel requires 2 to 12 external adjoints")
    if accuracy not in {"lc", "nlc", "full"}:
        raise ValueError(f"unknown adjoint colour accuracy {accuracy!r}")
    if type(nc) is not int or nc < 2:
        raise ValueError("adjoint colour kernel requires integer Nc >= 2")
    if type(full_col_acc) is not int or full_col_acc < 0:
        raise ValueError("adjoint full-colour accuracy must be nonnegative")
    if total_gluons == 2:
        # The degenerate ladder is Tr(Ta Tb)=delta_ab. There is no
        # commutator to remove U(1), so retain SU(Nc)'s exact dimension.
        leading_only = accuracy == "lc" or (accuracy == "full" and full_col_acc == 0)
        kernel = np.asarray([nc * nc - (0 if leading_only else 1)], dtype=np.int64)
        kernel.flags.writeable = False
        return kernel
    degree = total_gluons - 1
    # Each difference at most doubles the largest absolute input.  Check once
    # before allocating; numpy's int64 arithmetic must never silently wrap.
    if nc**total_gluons * 4 ** (total_gluons - 2) > np.iinfo(np.int64).max:
        raise ValueError("adjoint finite differences exceed exact int64 range")
    order = math.factorial(degree)
    rows = np.fromiter(
        chain.from_iterable(permutations(range(degree))),
        dtype=np.uint8,
        count=order * degree,
    ).reshape(order, degree)
    cycles = _trace_cycle_counts(rows)
    powers = np.asarray(
        [nc**power for power in range(total_gluons + 1)], dtype=np.int64
    )
    current = powers[cycles]
    minimum_power = max(
        total_gluons
        - (2 * full_col_acc if accuracy == "full" else 2 if accuracy == "nlc" else 0),
        0,
    )
    if minimum_power:
        current[cycles < minimum_power] = 0
    for generator in range(degree, 1, -1):
        # s_i^-1 acts on images: 0->1->...->i-1->0.
        relabel = np.arange(degree, dtype=np.uint8)
        relabel[:generator] = (relabel[:generator] + 1) % generator
        relative = relabel[rows]
        ranks = _permutation_ranks(relative)
        current = current - current[ranks]
    for generator in range(degree, 1, -1):
        # Right multiplication by s_i moves position i-1 to the beginning.
        relative = rows.copy()
        relative[:, 0] = rows[:, generator - 1]
        relative[:, 1:generator] = rows[:, : generator - 1]
        ranks = _permutation_ranks(relative)
        current = current - current[ranks]
    kernel = current[: math.factorial(degree - 1)].copy()
    kernel.flags.writeable = False
    return kernel


def adjoint_color_factor(
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    accuracy: str,
    nc: int = 3,
    full_col_acc: int = 20,
) -> int:
    """Evaluate one DDM overlap from its middle-label relative permutation."""

    if (
        len(left) < 2
        or len(left) != len(right)
        or left[0] != right[0]
        or left[-1] != right[-1]
        or len(set(left)) != len(left)
        or set(left) != set(right)
    ):
        raise ValueError("DDM colour words must have matching fixed endpoints")
    ranks_by_label = {label: index for index, label in enumerate(left[1:-1])}
    relative = [ranks_by_label[label] for label in right[1:-1]]
    rank = 0
    remaining = list(range(len(relative)))
    for position, value in enumerate(relative):
        offset = remaining.index(value)
        rank += offset * math.factorial(len(relative) - position - 1)
        remaining.pop(offset)
    return int(
        adjoint_color_kernel(
            len(left), accuracy=accuracy, nc=nc, full_col_acc=full_col_acc
        )[rank]
    )
