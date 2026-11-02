# SPDX-License-Identifier: 0BSD
"""Opt-in declarations for tree-level colour and spin correlations.

Colour connections are ordered and use all-outgoing colour charges. External
legs keep the process's one-based labels; negative labels are convenient for
auxiliary emitted partons. These are colour operations, not additional external
momenta. The ordinary Born colour metric is always available as ``"born"``.

Spin classes declare the exact sets of vector legs that may be replaced
simultaneously. Numerical vectors are supplied to the runtime, not generation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import cast

from pyamplicol.color.connections import ColorLeg, EmitGluon, SplitGluon
from pyamplicol.color.correlator_matrices import ColorCorrelator

__all__ = [
    "ColorCorrelator",
    "CorrelatedRequest",
    "CorrelatedValue",
    "CorrelatorConfig",
    "EmitGluon",
    "SplitGluon",
]


@dataclass(frozen=True)
class CorrelatedRequest:
    """One labelled request passed to ``Runtime.evaluate_correlated_many``.

    ``spin_vectors=None`` inherits the runtime setter's state at the start of
    the call; ``{}`` selects ordinary physical helicities for this request.
    Explicit maps use the setter's broadcast/per-point vector convention.
    Decimal real components, or pairs of Decimal real/imaginary components,
    retain their precision until the requested arithmetic is applied.
    The runtime validates and copies all requests before numerical work.
    """

    color_correlation: str = "born"
    spin_vectors: Mapping[int, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.color_correlation, str) or not self.color_correlation:
            raise ValueError("a correlated request requires a colour correlation ID")
        if self.spin_vectors is not None and not isinstance(self.spin_vectors, Mapping):
            raise TypeError("spin vectors must map public leg labels to vectors")


@dataclass(frozen=True)
class CorrelatedValue:
    """One possibly complex correlation, retaining the requested precision.

    ``real`` and ``imag`` are decimals. ``complex(value)`` converts to ordinary
    double precision. Ordered colour connections need not give a real result;
    their imaginary part must not be discarded before physical combinations.
    """

    real: Decimal
    imag: Decimal = Decimal(0)

    def __complex__(self) -> complex:
        return complex(float(self.real), float(self.imag))


def _fields(value: object, allowed: set[str], context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {context} fields: {sorted(unknown, key=str)}")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a list")
    return value


def _emission_from_json(value: object) -> EmitGluon | SplitGluon:
    if not isinstance(value, Mapping):
        raise TypeError("a colour connection step must be an object")
    kind = value.get("kind")
    if kind == "emit-gluon":
        data = _fields(value, {"kind", "emitter", "emitted"}, "gluon emission")
        try:
            return EmitGluon(cast(int, data["emitter"]), cast(int, data["emitted"]))
        except KeyError as exc:
            raise ValueError(f"gluon emission is missing {exc.args[0]!r}") from exc
    if kind == "split-gluon":
        data = _fields(
            value, {"kind", "parent", "quark", "antiquark"}, "gluon splitting"
        )
        try:
            return SplitGluon(
                cast(int, data["parent"]),
                cast(int, data["quark"]),
                cast(int, data["antiquark"]),
            )
        except KeyError as exc:
            raise ValueError(f"gluon splitting is missing {exc.args[0]!r}") from exc
    raise ValueError(f"unknown colour connection step {kind!r}")


@dataclass(frozen=True)
class CorrelatorConfig:
    """The correlations to prepare alongside a complete tree amplitude.

    ``color_correlations`` contains the requested ``bra† ket`` connections,
    keyed by unique IDs. Each side has the same number of ordered operations.
    ``spin_correlations=((3,), (3, 4))`` permits replacing leg 3 alone or legs
    3 and 4 together. An empty runtime vector map always restores the ordinary
    helicity sum. Each replaced leg contributes one vector state; the usual
    initial-state averaging and identical-particle factors are retained.

    The first correlated executor uses the generic compiled amplitude and
    direct exact colour algebra. It does not use FFT or helicity-specific
    current recycling. Uncorrelated generation does not construct this object.
    """

    color_correlations: tuple[ColorCorrelator, ...] = ()
    spin_correlations: tuple[tuple[int, ...], ...] = ()
    all_color_through_order: int | None = None

    def __post_init__(self) -> None:
        order = self.all_color_through_order
        if order is not None and (
            type(order) is not int or order < 1
        ):
            raise ValueError(
                "all_color_through_order must be a positive integer"
            )
        colors = tuple(self.color_correlations)
        if any(not isinstance(item, ColorCorrelator) for item in colors):
            raise TypeError("color_correlations must contain ColorCorrelator records")
        ids = tuple(item.id for item in colors)
        if "born" in ids:
            raise ValueError("the colour correlation ID 'born' is reserved")
        if len(set(ids)) != len(ids):
            raise ValueError("colour correlation IDs must be unique")
        if order is not None and any(
            re.fullmatch(r"N[1-9][0-9]*LO/c[0-9]+/c[0-9]+", identifier)
            for identifier in ids
        ):
            raise ValueError("automatic colour catalogue IDs are reserved")
        classes: list[tuple[int, ...]] = []
        for group in self.spin_correlations:
            legs = tuple(group)
            if not legs:
                raise ValueError("a spin-correlation class must contain a leg")
            if any(type(leg) is not int or leg <= 0 for leg in legs):
                raise ValueError("spin-correlation legs must be positive integers")
            if len(set(legs)) != len(legs):
                raise ValueError("a spin-correlation class repeats a leg")
            canonical = tuple(sorted(legs))
            if canonical in classes:
                raise ValueError("spin-correlation classes must be distinct")
            classes.append(canonical)
        object.__setattr__(self, "color_correlations", colors)
        object.__setattr__(self, "spin_correlations", tuple(sorted(classes)))

    @classmethod
    def all_color(
        cls,
        *,
        through_order: int,
        spin_correlations: tuple[tuple[int, ...], ...] = (),
    ) -> CorrelatorConfig:
        """Prepare every tree-soft colour connection through N^kLO.

        Generation resolves the coloured Born legs separately for each process.
        Histories include emission from emitted partons and splitting of emitted
        gluons into quark pairs; this is not a catalogue of dipole products.
        All directed overlaps with matching final colour spaces are retained.
        No splitting kernels, flavour sums or combinatorial weights are included.
        ``through_order=k`` includes all orders 1 through k without a fixed cap.
        The complete catalogue is nonminimal and grows rapidly with k.
        """
        if type(through_order) is not int or through_order < 1:
            raise ValueError("all_color_through_order must be a positive integer")
        return cls(
            spin_correlations=spin_correlations,
            all_color_through_order=through_order,
        )

    def _resolve_color(self, legs: Sequence[ColorLeg]) -> CorrelatorConfig:
        """Expand the opt-in declaration once the process representations exist."""
        if self.all_color_through_order is None:
            return self
        from pyamplicol.color.connection_catalogue import (
            build_color_correlator_catalogue,
        )

        return replace(
            self,
            color_correlations=(
                *self.color_correlations,
                *build_color_correlator_catalogue(
                    legs, through_order=self.all_color_through_order
                ),
            ),
            all_color_through_order=None,
        )

    @property
    def spin_legs(self) -> tuple[int, ...]:
        """All external legs whose source vectors must remain unrestricted."""
        return tuple(sorted({leg for group in self.spin_correlations for leg in group}))

    @property
    def color_requests(self) -> tuple[ColorCorrelator, ...]:
        """Requested operators, including the ordinary Born overlap."""
        if self.all_color_through_order is not None:
            raise ValueError(
                "automatic colour requests must first be resolved for a process"
            )
        return (ColorCorrelator("born"), *self.color_correlations)

    def to_json_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "color_correlations": [
                item.to_json_dict() for item in self.color_correlations
            ],
            "spin_correlations": [list(group) for group in self.spin_correlations],
        }
        if self.all_color_through_order is not None:
            result["all_color_through_order"] = self.all_color_through_order
        return result

    @classmethod
    def from_json_dict(cls, value: object) -> CorrelatorConfig:
        """Read the same declaration used by Python and ``--correlators``."""
        data = _fields(
            value,
            {"color_correlations", "spin_correlations", "all_color_through_order"},
            "correlator declaration",
        )
        colors: list[ColorCorrelator] = []
        for record in _sequence(
            data.get("color_correlations", ()), "color_correlations"
        ):
            item = _fields(record, {"id", "bra", "ket"}, "colour correlation")
            if "id" not in item:
                raise ValueError("a colour correlation requires an ID")
            colors.append(
                ColorCorrelator(
                    item["id"],  # type: ignore[arg-type]
                    tuple(
                        _emission_from_json(step)
                        for step in _sequence(item.get("bra", ()), "bra")
                    ),
                    tuple(
                        _emission_from_json(step)
                        for step in _sequence(item.get("ket", ()), "ket")
                    ),
                )
            )
        spins = tuple(
            tuple(_sequence(group, "spin-correlation class"))
            for group in _sequence(
                data.get("spin_correlations", ()), "spin_correlations"
            )
        )
        return cls(
            tuple(colors),
            cast(tuple[tuple[int, ...], ...], spins),
            cast(int | None, data.get("all_color_through_order")),
        )
