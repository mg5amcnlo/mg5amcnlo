# SPDX-License-Identifier: 0BSD
"""Additive quantum numbers conserved by every vertex of a compiled model.

A charge assigns a rational value to every particle, the opposite value to its
antiparticle and zero to self-conjugate particles. It is conserved when the
values of the particles of every vertex term sum to zero, so the conserved
charges span the rational nullspace of the vertex/flavour incidence matrix.
They are derived from the vertex list alone, which makes them a model-generic
prefilter: a process whose external states do not balance one of them has no
tree-level amplitude, whatever the coupling orders.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from ..models.contracts import CompiledModelIR, CompiledParticleRecord


@dataclass(frozen=True, slots=True)
class ConservedCharges:
    """Integer values of a basis of conserved charges, by particle name."""

    charges: Mapping[str, tuple[int, ...]]
    count: int

    def total(self, names: Iterable[str]) -> tuple[int, ...]:
        totals = [0] * self.count
        zero = (0,) * self.count
        for name in names:
            for index, value in enumerate(self.charges.get(name, zero)):
                totals[index] += value
        return tuple(totals)

    def balanced(self, initial: Sequence[str], final: Sequence[str]) -> bool:
        return self.total(initial) == self.total(final)


def _self_conjugate(particle: CompiledParticleRecord) -> bool:
    declared = particle.self_conjugate
    return (
        bool(declared) if declared is not None else particle.name == particle.antiname
    )


def _nullspace(rows: Sequence[Sequence[int]], width: int) -> list[list[Fraction]]:
    """Rational basis of ``{x : row . x = 0 for every row}``."""

    reduced: list[list[Fraction]] = []
    pivots: list[int] = []
    for row in rows:
        vector = [Fraction(value) for value in row]
        for pivot_row, pivot in zip(reduced, pivots, strict=True):
            if vector[pivot]:
                factor = vector[pivot]
                vector = [
                    a - factor * b for a, b in zip(vector, pivot_row, strict=True)
                ]
        lead = next((index for index, value in enumerate(vector) if value), None)
        if lead is None:
            continue
        vector = [value / vector[lead] for value in vector]
        for index, other in enumerate(reduced):
            if other[lead]:
                factor = other[lead]
                reduced[index] = [
                    a - factor * b for a, b in zip(other, vector, strict=True)
                ]
        reduced.append(vector)
        pivots.append(lead)
    basis = []
    for free in (column for column in range(width) if column not in pivots):
        vector = [Fraction(0)] * width
        vector[free] = Fraction(1)
        for pivot_row, pivot in zip(reduced, pivots, strict=True):
            vector[pivot] = -pivot_row[free]
        basis.append(vector)
    return basis


def _integer(vector: Sequence[Fraction]) -> tuple[int, ...]:
    scale = math.lcm(*(value.denominator for value in vector))
    values = [int(value * scale) for value in vector]
    divisor = math.gcd(*values) or 1
    return tuple(value // divisor for value in values)


def conserved_charges(model: CompiledModelIR) -> ConservedCharges:
    """Return the additive charges conserved by every vertex term of ``model``."""

    by_name = {particle.name: particle for particle in model.particles}
    # One column per particle/antiparticle pair: (column, +1 or -1).
    columns: dict[str, tuple[int, int]] = {}
    width = 0
    for particle in model.particles:
        if _self_conjugate(particle) or particle.name in columns:
            continue
        antiparticle = by_name.get(particle.antiname)
        pair = [particle] if antiparticle is None else [particle, antiparticle]
        canonical = max(pair, key=lambda record: (record.pdg_code, record.name))
        for record in pair:
            columns[record.name] = (width, 1 if record is canonical else -1)
        width += 1
    rows: set[tuple[int, ...]] = set()
    for term in model.vertex_terms:
        row = [0] * width
        for name in term.particles:
            if name not in by_name:
                raise ValueError(
                    f"vertex term {term.id} refers to unknown particle {name!r}"
                )
            if name in columns:
                column, sign = columns[name]
                row[column] += sign
        if any(row):
            rows.add(tuple(row))
    basis = [_integer(vector) for vector in _nullspace(sorted(rows), width)]
    charges = {
        name: tuple(sign * vector[column] for vector in basis)
        for name, (column, sign) in columns.items()
    }
    return ConservedCharges(charges=charges, count=len(basis))


__all__ = ["ConservedCharges", "conserved_charges"]
